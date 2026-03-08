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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .geometry_index import GeometryVariant
from .run_plan import RunPlan, RunRecord

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
GEOMETRY_DIRECTORY = PROJECT_DIRECTORY / "geometries"
SIMULATION_DIRECTORY = PROJECT_DIRECTORY / "simulation"
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"


# Create the directory tree needed by a later output file.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Print and execute one external command so the orchestration logs show exactly what ran.
def run_cmd(command: Sequence[str], *, dry_run: bool, label: str) -> subprocess.CompletedProcess:
    """Execute a subprocess, emitting the command for easier debugging."""
    print(f"[{label}] {' '.join(shlex.quote(token) for token in command)}")
    if dry_run:
        return subprocess.CompletedProcess(command, returncode=0)
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


# Materialize the generated geometry set unless conductor was told to reuse what is already on disk.
def maybe_run_sweeps(args: argparse.Namespace, spec_paths: List[Path]) -> None:
    if args.skip_sweep:
        return
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
        run_cmd(command, dry_run=args.dry_run, label="sweep")


# Produce the muon control calibration that defines the visible-energy detect threshold for one geometry.
def run_muon_calibration(args: argparse.Namespace, geometry_variant: GeometryVariant) -> Path:
    calibration_directory = DATA_DIRECTORY / "processed" / geometry_variant.geometry_id / "run_mu_ctrl"
    ensure_dir(calibration_directory)
    output_path = calibration_directory / "fpr_calibration.json"

    # Reuse the calibration file if it is already present and the caller did not ask to overwrite it.
    if output_path.exists() and not args.overwrite:
        print(f"[muon_calib] Skipping calibration for {geometry_variant.geometry_id}, output exists at {output_path}")
        return output_path

    # Muons are minimum-ionizing particles, so they produce a well-defined, small visible-energy
    # deposit with no hadronic response. The muon control sample runs through the same ddsim and
    # processor chain, then calibrate_fpr.py fits the visible-energy distribution and returns the
    # threshold that keeps the false-positive rate at the requested level.
    raw_output_path = DATA_DIRECTORY / "raw" / geometry_variant.geometry_id / "run_mu_ctrl" / "run_mu_ctrl.edm4hep.root"
    ensure_dir(raw_output_path.parent)
    command = [
        args.python,
        str(SIMULATION_DIRECTORY / "calibration" / "calibrate_fpr.py"),
        "--compact-xml",
        str(geometry_variant.xml_path),
        "--raw-out",
        str(raw_output_path),
        "--events-out",
        str(calibration_directory / "events.root"),
        "--json-out",
        str(output_path),
        "--ddsim",
        args.ddsim,
        "--process-bin",
        args.process_bin,
        "--root-bin",
        args.root_bin,
        "--gun-particle",
        "mu-",
        "--gun-energy",
        "10*GeV",
        "--n-events",
        "2000",
        "--metric",
        "visible_E",
        "--target-fpr",
        "0.01",
    ]
    run_cmd(command, dry_run=args.dry_run, label="muon_calibration")
    return output_path


# Run the detector simulation step for one planned sample and return its wall-clock time.
def run_ddsim(args: argparse.Namespace, run_plan: RunPlan) -> float:
    """Fire ddsim for this plan and return the wall-clock seconds spent."""
    ensure_dir(run_plan.raw_path.parent)
    energy_expression = f"{run_plan.gun_energy_GeV}*GeV"

    # Keep the gun settings and particle-tracking options explicit so the raw file contains the
    # full truth record needed by later calibration and performance steps.
    command = [
        args.ddsim,
        "--compactFile",
        str(run_plan.geometry_variant.xml_path),
        "--outputFile",
        str(run_plan.raw_path),
        "--numberOfEvents",
        str(run_plan.n_events),
        "--enableGun",
        "--gun.particle",
        run_plan.gun_particle,
        "--gun.energy",
        energy_expression,
        "--gun.direction",
        run_plan.gun_direction,
        "--gun.position",
        run_plan.gun_position,
        "--part.keepAllParticles",
        "true",
        "--part.minimalKineticEnergy",
        "0*MeV",
        "--part.minDistToParentVertex",
        "0*mm",
    ]
    if run_plan.seed is not None:
        command.extend(["--random.seed", str(run_plan.seed)])
    start = time.time()
    run_cmd(command, dry_run=args.dry_run, label="ddsim")
    return time.time() - start


# Run the processor on one raw EDM4hep file.
def run_process(
    args: argparse.Namespace,
    run_plan: RunPlan,
    extra_process_flags: List[str],
) -> Tuple[float, List[str]]:
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
    run_cmd(command, dry_run=args.dry_run, label="process")
    return time.time() - start, command


# Write the minimal metadata that downstream analysis steps need to identify the run conditions.
def write_metadata(
    args: argparse.Namespace,
    run_plan: RunPlan,
) -> float:
    """Write meta.json directly from conductor-owned run descriptors."""
    payload: Dict[str, Any] = {
        "geometry_id": run_plan.geometry_variant.geometry_id,
        "gun_energy_GeV": run_plan.gun_energy_GeV,
        "seed": run_plan.seed,
    }
    start = time.time()

    # In dry-run mode report the intended output path without creating files.
    if args.dry_run:
        print(f"[meta] write {run_plan.meta_path}")
        return time.time() - start
    ensure_dir(run_plan.meta_path.parent)
    with run_plan.meta_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2)
    print(f"[meta] wrote {run_plan.meta_path}")
    return time.time() - start


# Write the calibration constants that belong to one processed run.
def write_calibration(
    args: argparse.Namespace,
    run_plan: RunPlan,
    neutron_scale: Optional[float] = None,
) -> None:
    """Write calibration.json with the run-level calibration constants."""
    payload: Dict[str, Any] = {
        "muon_threshold_GeV": args.muon_threshold,
    }

    # Add the neutron response scale only for runs where that calibration was actually derived.
    if neutron_scale is not None:
        payload["neutron_scale"] = neutron_scale
    if args.dry_run:
        print(f"[calibration] write {run_plan.calibration_path}")
        return
    ensure_dir(run_plan.calibration_path.parent)
    with run_plan.calibration_path.open("w", encoding="utf-8") as calibration_file:
        json.dump(payload, calibration_file, indent=2)
    print(f"[calibration] wrote {run_plan.calibration_path}")


# Derive the neutron visible-to-truth response scale from the processed events tree for one run.
def run_neutron_calibration(
    args: argparse.Namespace,
    run_plan: RunPlan,
) -> Tuple[float, Optional[float]]:
    """Derive one neutron visible-to-truth scale from the processed events tree."""
    calibration_path = run_plan.calibration_path
    ensure_dir(calibration_path.parent)
    macro_path = SIMULATION_DIRECTORY / "calibration" / "calibrate_neutron_response.C"

    # Hand the processed events tree, beam energy, and detect threshold to the ROOT calibration macro.
    command = [
        args.root_bin,
        "-l",
        "-b",
        "-q",
        f'{macro_path}("{run_plan.events_path}",{run_plan.gun_energy_GeV:.12g},{args.muon_threshold:.12g},"{calibration_path}")',
    ]
    start = time.time()
    run_cmd(command, dry_run=args.dry_run, label="neutron_calibration")
    elapsed = time.time() - start
    if args.dry_run:
        return elapsed, None

    # Read back the neutron_scale so conductor can write it into the run-level calibration file.
    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing neutron calibration output: {calibration_path}")
    with calibration_path.open("r", encoding="utf-8") as calibration_file:
        payload = json.load(calibration_file)
    neutron_scale = payload.get("neutron_scale")
    if neutron_scale is None:
        raise ValueError(f"neutron_scale missing in {calibration_path}")
    return elapsed, float(neutron_scale)


# Run the fixed ROOT performance macro on one processed sample.
def run_performance_analysis(args: argparse.Namespace, run_plan: RunPlan) -> float:
    """Run the fixed performance analysis step for one processed run."""
    ensure_dir(run_plan.performance_path.parent)
    performance_macro_path = PROJECT_DIRECTORY / "analysis" / "simulation" / "performance.C"
    command = [
        args.root_bin,
        "-l",
        "-b",
        "-q",
        f'{performance_macro_path}("{run_plan.events_path}","{run_plan.meta_path}","{run_plan.calibration_path}","{run_plan.performance_path}")',
    ]
    start = time.time()
    run_cmd(command, dry_run=args.dry_run, label="performance")
    return time.time() - start


# Write the compact campaign index that points to every run and its key output files.
def write_run_manifests(run_records: List[RunRecord], json_path: Path, csv_path: Path) -> None:
    """Write the compact run index used for output discovery."""
    ensure_dir(json_path.parent)
    rows = []

    # Build one manifest row per run so later tools can discover outputs without walking the data tree.
    for run_record in run_records:
        rows.append(
            {
                "geometry_id": run_record.plan.geometry_variant.geometry_id,
                "run_id": run_record.plan.run_id,
                "particle": run_record.plan.gun_particle,
                "energy_GeV": run_record.plan.gun_energy_GeV,
                "seed": run_record.plan.seed,
                "status": run_record.status,
                "raw_path": str(run_record.plan.raw_path),
                "events_path": str(run_record.plan.events_path),
                "meta_path": str(run_record.plan.meta_path),
                "calibration_path": str(run_record.plan.calibration_path),
                "performance_path": str(run_record.plan.performance_path),
                "error": run_record.error,
            }
        )

    # Keep both JSON and CSV forms so the manifest is easy to read by both scripts and spreadsheets.
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump({"runs": rows}, json_file, indent=2)
    fieldnames = [
        "geometry_id",
        "run_id",
        "particle",
        "energy_GeV",
        "seed",
        "status",
        "raw_path",
        "events_path",
        "meta_path",
        "calibration_path",
        "performance_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})
