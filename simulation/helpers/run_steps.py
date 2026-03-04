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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def run_cmd(command: Sequence[str], *, dry_run: bool, label: str) -> subprocess.CompletedProcess:
    """Execute a subprocess, emitting the command for easier debugging."""
    print(f"[{label}] {' '.join(shlex.quote(token) for token in command)}")
    if dry_run:
        return subprocess.CompletedProcess(command, returncode=0)
    return subprocess.run(command, check=True)



def flatten_process_extras(chunks: Sequence[Optional[str]]) -> List[str]:
    """Split quoted processor option chunks into individual flags."""
    flags: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        flags.extend(shlex.split(chunk))
    return flags



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



def run_muon_calibration(args: argparse.Namespace, geometry_variant: GeometryVariant) -> Path:
    calibration_directory = DATA_DIRECTORY / "processed" / geometry_variant.geometry_id / "run_mu_ctrl"
    ensure_dir(calibration_directory)
    output_path = calibration_directory / "fpr_calibration.json"
    if output_path.exists() and not args.overwrite:
        print(f"[muon_calib] Skipping calibration for {geometry_variant.geometry_id}, output exists at {output_path}")
        return output_path

    raw_output_path = DATA_DIRECTORY / "raw" / geometry_variant.geometry_id / "run_mu_ctrl" / "run_mu_ctrl.edm4hep.root"
    ensure_dir(raw_output_path.parent)
    events_output_path = calibration_directory / "events.root"
    json_output_path = calibration_directory / "fpr_calibration.json"
    command = [
        args.python,
        str(SIMULATION_DIRECTORY / "calibrate_fpr.py"),
        "--compact-xml",
        str(geometry_variant.xml_path),
        "--raw-out",
        str(raw_output_path),
        "--events-out",
        str(events_output_path),
        "--json-out",
        str(json_output_path),
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
        "sim_E",
        "--target-fpr",
        "0.01",
    ]
    run_cmd(command, dry_run=args.dry_run, label="muon_calibration")
    return json_output_path



def run_ddsim(args: argparse.Namespace, run_plan: RunPlan) -> float:
    """Fire ddsim for this plan and return the wall-clock seconds spent."""
    ensure_dir(run_plan.raw_path.parent)
    energy_expression = f"{run_plan.gun_energy_GeV}*GeV"
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



def run_process(
    args: argparse.Namespace,
    run_plan: RunPlan,
    extra_process_flags: List[str],
) -> Tuple[float, List[str]]:
    """Invoke the processor on the raw EDM4hep file and report elapsed time."""
    ensure_dir(run_plan.events_path.parent)
    command = [
        args.process_bin,
        str(run_plan.raw_path),
        "--out",
        str(run_plan.events_path),
    ] + extra_process_flags
    if run_plan.expected_pdg is not None:
        command.extend(["--expected-pdg", str(run_plan.expected_pdg)])
    start = time.time()
    run_cmd(command, dry_run=args.dry_run, label="process")
    return time.time() - start, command



def write_metadata(
    args: argparse.Namespace,
    run_plan: RunPlan,
) -> float:
    """Write meta.json directly from conductor-owned values."""
    payload: Dict[str, Any] = {
        "geometry_id": run_plan.geometry_variant.geometry_id,
        "gun_energy_GeV": run_plan.gun_energy_GeV,
        "detect_threshold_GeV": args.detect_threshold,
    }
    start = time.time()
    if args.dry_run:
        print(f"[meta] write {run_plan.meta_path}")
        return time.time() - start
    run_plan.meta_path.parent.mkdir(parents=True, exist_ok=True)
    with run_plan.meta_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2)
    print(f"[meta] wrote {run_plan.meta_path}")
    return time.time() - start



def run_performance_analysis(args: argparse.Namespace, run_plan: RunPlan) -> float:
    """Run the fixed performance analysis step for one processed run."""
    ensure_dir(run_plan.performance_path.parent)
    performance_macro_path = SIMULATION_DIRECTORY / "analysis" / "performance.C"
    command = [
        args.root_bin,
        "-l",
        "-b",
        "-q",
        f'{performance_macro_path}("{run_plan.events_path}","{run_plan.meta_path}","{run_plan.performance_path}")',
    ]
    start = time.time()
    run_cmd(command, dry_run=args.dry_run, label="performance")
    return time.time() - start



def write_run_manifests(run_records: List[RunRecord], json_path: Path, csv_path: Path) -> None:
    """Write the compact run index used for output discovery."""
    ensure_dir(json_path.parent)
    rows = []
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
                "performance_path": str(run_record.plan.performance_path),
                "error": run_record.error,
            }
        )
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
        "performance_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})
