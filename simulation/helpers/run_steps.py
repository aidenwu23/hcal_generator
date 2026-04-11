#!/usr/bin/env python3
"""Execution steps used by conductor."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .geometry_index import eval_geometry_length_mm
from .spectrum import build_gps_macro_text, load_g4gps_spec
from .run_plan import RunPlan, RunRecord

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
GEOMETRY_DIRECTORY = PROJECT_DIRECTORY / "geometries"
SIMULATION_DIRECTORY = PROJECT_DIRECTORY / "simulation"
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
REFERENCE_MIP_PATH = SIMULATION_DIRECTORY / "calibration" / "reference_mip_4mm.json"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_remove_file(args: argparse.Namespace, path: Path) -> None:
    # Remove one intermediate file after every downstream consumer has finished.
    if not args.delete_intermediates:
        return
    if not path.exists():
        return
    path.unlink()
    print(f"[delete_intermediates] removed {path}")


# Print and execute one external command so the orchestration logs show exactly what ran.
def run_cmd(command: Sequence[str], *, label: str) -> subprocess.CompletedProcess:
    """Execute a subprocess, emitting the command for easier debugging."""
    print(f"[{label}] {' '.join(shlex.quote(token) for token in command)}")
    return subprocess.run(command, check=True)


# Split the optional processor extras from shell-like chunks into the flag list expected by subprocess.
def flatten_process_extras(chunks: Sequence[Optional[str]]) -> List[str]:
    """Split quoted processor option chunks into individual flags."""
    flags: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        flags.extend(shlex.split(chunk))
    return flags


# Materialize the generated geometry set before conductor builds run plans from it.
def maybe_run_sweeps(args: argparse.Namespace, spec_paths: List[Path]) -> None:
    sweep_script = GEOMETRY_DIRECTORY / "sweep_geometries.py"
    if not sweep_script.exists():
        raise FileNotFoundError(f"{sweep_script} not found.")
    missing_specs = [spec_path for spec_path in spec_paths if not spec_path.exists()]
    if missing_specs:
        pretty = ", ".join(str(spec_path) for spec_path in missing_specs)
        raise FileNotFoundError(f"Sweep spec(s) not found: {pretty}")

    # Run the sweep helper once per requested spec so each sweep file can generate its own geometry set.
    for spec_path in spec_paths:
        command = [
            args.python,
            str(sweep_script),
            "--spec",
            str(spec_path),
        ]
        if args.overwrite_geos:
            command.append("--overwrite")
        run_cmd(command, label="sweep")


def write_scaled_mip_calibration(
    args: argparse.Namespace,
    run_plan: RunPlan,
) -> Path:
    """Write one run-local calibration.json by scaling a measured 4 mm reference MIP."""
    ensure_dir(run_plan.calibration_path.parent)
    calibration_path = run_plan.calibration_path

    if calibration_path.exists() and not args.overwrite:
        print(f"[mip_calibration] Skipping calibration for {run_plan.run_id}, output exists at {calibration_path}")
        return calibration_path

    if not REFERENCE_MIP_PATH.exists():
        raise FileNotFoundError(f"{REFERENCE_MIP_PATH} not found.")

    with REFERENCE_MIP_PATH.open("r", encoding="utf-8") as reference_file:
        reference_payload = json.load(reference_file)

    reference_thickness_mm = float(reference_payload["reference_thickness_mm"])
    reference_mpv_gev = float(reference_payload["reference_mpv_GeV"])
    if reference_thickness_mm <= 0.0:
        raise ValueError("reference_thickness_mm must be positive.")
    if reference_mpv_gev <= 0.0:
        raise ValueError("reference_mpv_GeV must be positive.")

    segment_thicknesses_mm: List[float] = []
    mpvs: List[float] = []
    thresholds: List[float] = []

    for segment_index in range(1, 4):
        key = f"t_scin_seg{segment_index}"
        if key not in run_plan.geometry_variant.params:
            raise ValueError(f"Missing {key} in geometry parameters for {run_plan.geometry_variant.geometry_id}.")
        thickness_mm = eval_geometry_length_mm(run_plan.geometry_variant.params[key])
        if thickness_mm <= 0.0:
            raise ValueError(f"{key} must be positive for {run_plan.geometry_variant.geometry_id}.")
        
        # Scale the reference MIP linearly with scintillator thickness.
        mpv_gev = reference_mpv_gev * (thickness_mm / reference_thickness_mm)
        segment_thicknesses_mm.append(thickness_mm)
        mpvs.append(mpv_gev)
        thresholds.append(args.mip_alpha * mpv_gev)

    payload: Dict[str, Any] = {
        "alpha": args.mip_alpha,
        "reference_geometry_id": reference_payload.get("reference_geometry_id"),
        "reference_thickness_mm": reference_thickness_mm,
        "reference_mpv_GeV": reference_mpv_gev,
        "segment_scintillator_thicknesses_mm": segment_thicknesses_mm,
        "mpvs": mpvs,
        "thresholds": thresholds,
    }

    with calibration_path.open("w", encoding="utf-8") as calibration_file:
        json.dump(payload, calibration_file, indent=2)
        calibration_file.write("\n")
    print(f"[mip_calibration] Wrote {calibration_path}")
    return calibration_path


def _load_g4gps_metadata(spec_path: Path) -> Dict[str, Any]:
    spec = load_g4gps_spec(spec_path)
    return {
        "g4gps_spec_path": str(spec_path),
        "spectrum_id": spec.spec_id,
        "spectrum_x_axis": spec.x_axis,
        "spectrum_x_min_GeV": spec.x_min_GeV,
        "spectrum_x_max_GeV": spec.x_max_GeV,
    }


# Run the detector simulation step for one planned sample and return its wall-clock time.
def run_ddsim(args: argparse.Namespace, run_plan: RunPlan) -> float:
    """Fire ddsim for this plan and return the wall-clock seconds spent."""
    ensure_dir(run_plan.raw_path.parent)

    # Keep the gun settings and particle-tracking options explicit so the raw file contains the
    # full truth record needed by later calibration and performance steps.
    command = [
        args.ddsim,
        "--compactFile",
        str(run_plan.geometry_variant.xml_path),
        "--physicsList",
        args.physics_list,
        "--outputFile",
        str(run_plan.raw_path),
        "--part.keepAllParticles",
        "true",
        "--part.minimalKineticEnergy",
        "0*MeV",
        "--part.minDistToParentVertex",
        "0*mm",
    ]

    # Use the generated G4GPS macro path.
    if run_plan.beam_mode == "g4gps_spec":
        if run_plan.g4gps_spec_path is None or run_plan.macro_path is None:
            raise ValueError("G4GPS spec runs require g4gps_spec_path and macro_path.")
        ensure_dir(run_plan.macro_path.parent)
        spec = load_g4gps_spec(run_plan.g4gps_spec_path)
        macro_text = build_gps_macro_text(
            spec,
            event_count=run_plan.n_events,
        )
        with run_plan.macro_path.open("w", encoding="utf-8") as macro_file:
            macro_file.write(macro_text)
        command.extend(
            [
                "--runType",
                "run",
                "--macroFile",
                str(run_plan.macro_path),
                "--enableG4GPS",
            ]
        )
        
    # Otherwise use the default ddsim particle gun.
    else:
        if run_plan.momentum_GeV is None or run_plan.gun_direction is None or run_plan.gun_position is None:
            raise ValueError("Fixed-gun runs require momentum_GeV, gun_direction, and gun_position.")
        momentum_expression = f"{run_plan.momentum_GeV}*GeV"
        command.extend(
            [
                "--numberOfEvents",
                str(run_plan.n_events),
                "--enableGun",
                "--gun.particle",
                run_plan.gun_particle,
                "--gun.energy",
                momentum_expression,
                "--gun.direction",
                run_plan.gun_direction,
                "--gun.position",
                run_plan.gun_position,
            ]
        )
    if run_plan.seed is not None:
        command.extend(["--random.seed", str(run_plan.seed)])
    start = time.time()
    run_cmd(command, label="ddsim")
    return time.time() - start


# Run the processor on one raw EDM4hep file.
def run_process(
    args: argparse.Namespace,
    run_plan: RunPlan,
    extra_process_flags: List[str],
) -> float:
    """Invoke the processor on the raw EDM4hep file and report elapsed time."""
    ensure_dir(run_plan.events_path.parent)

    # Start from the raw EDM4hep file for this run, then append any campaign-wide processor extras.
    command = [
        args.process_bin,
        str(run_plan.raw_path),
        "--out",
        str(run_plan.events_path),
    ] + extra_process_flags

    # Attach the expected primary PDG when the run plan provides it.
    if run_plan.expected_pdg is not None:
        command.extend(["--expected-pdg", str(run_plan.expected_pdg)])
    start = time.time()
    run_cmd(command, label="process")
    return time.time() - start


# Write the minimal metadata that downstream analysis steps need to identify the run conditions.
def write_metadata(
    args: argparse.Namespace,
    run_plan: RunPlan,
) -> float:
    """Write meta.json directly from conductor-owned run descriptors."""
    payload: Dict[str, Any] = {
        "geometry_id": run_plan.geometry_variant.geometry_id,
        "gun_particle": run_plan.gun_particle,
        "beam_mode": run_plan.beam_mode,
        "beam_label": run_plan.beam_label,
        "seed": run_plan.seed,
    }
    if run_plan.momentum_GeV is not None:
        payload["momentum_GeV"] = run_plan.momentum_GeV
    if run_plan.g4gps_spec_path is not None:
        payload.update(_load_g4gps_metadata(run_plan.g4gps_spec_path))
    if run_plan.macro_path is not None:
        payload["macro_path"] = str(run_plan.macro_path)
    start = time.time()

    ensure_dir(run_plan.meta_path.parent)
    with run_plan.meta_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2)
    print(f"[meta] wrote {run_plan.meta_path}")
    return time.time() - start


# Run the fixed ROOT performance macro on one processed sample.
def run_performance_analysis(args: argparse.Namespace, run_plan: RunPlan) -> float:
    """Run the fixed performance analysis step for one processed run."""
    ensure_dir(run_plan.performance_path.parent)
    performance_macro_path = PROJECT_DIRECTORY / "simulation" / "processing" / "performance.C"
    command = [
        args.root_bin,
        "-l",
        "-b",
        "-q",
        f'{performance_macro_path}("{run_plan.events_path}","{run_plan.meta_path}","{run_plan.calibration_path}","{run_plan.performance_path}")',
    ]
    start = time.time()
    run_cmd(command, label="performance")
    return time.time() - start


# Write the compact campaign index that points to every run and its key output files.
def write_run_manifests(run_records: List[RunRecord], json_path: Path, csv_path: Path) -> None:
    """Write the compact run index used for output discovery."""
    ensure_dir(json_path.parent)
    rows = []
    fieldnames = [
        "geometry_id",
        "run_id",
        "particle",
        "beam_mode",
        "beam_label",
        "momentum_GeV",
        "spectrum_id",
        "spectrum_x_axis",
        "spectrum_x_min_GeV",
        "spectrum_x_max_GeV",
        "seed",
        "status",
        "raw_path",
        "events_path",
        "meta_path",
        "calibration_path",
        "performance_path",
        "error",
    ]

    # Build one manifest row per run so later tools can discover outputs without walking the data tree.
    for run_record in run_records:
        row = {
            "geometry_id": run_record.plan.geometry_variant.geometry_id,
            "run_id": run_record.plan.run_id,
            "particle": run_record.plan.gun_particle,
            "beam_mode": run_record.plan.beam_mode,
            "beam_label": run_record.plan.beam_label,
            "momentum_GeV": run_record.plan.momentum_GeV,
            "spectrum_id": None,
            "spectrum_x_axis": None,
            "spectrum_x_min_GeV": None,
            "spectrum_x_max_GeV": None,
            "seed": run_record.plan.seed,
            "status": run_record.status,
            "raw_path": str(run_record.plan.raw_path),
            "events_path": str(run_record.plan.events_path),
            "meta_path": str(run_record.plan.meta_path),
            "calibration_path": str(run_record.plan.calibration_path),
            "performance_path": str(run_record.plan.performance_path),
            "error": run_record.error,
        }
        if run_record.plan.g4gps_spec_path is not None:
            row.update(_load_g4gps_metadata(run_record.plan.g4gps_spec_path))
        rows.append(row)

    # Keep both JSON and CSV forms so the manifest is easy to read by both scripts and spreadsheets.
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump({"runs": rows}, json_file, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})
