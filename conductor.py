#!/usr/bin/env python3
"""conductor.py

Run the HCAL production campaign from geometry sweeps through manifests.
Example: 
python3 conductor.py --spec geometries/sweeps/bhcal.yaml \
  --muon-threshold 0.02 \
  --events 3000 \
  --gun-particle neutron kaon0L \
  --gun-kinetic-energy 1 \
  --seeds 67 --delete-intermediates

python3 conductor.py --spec geometries/sweeps/bhcal.yaml \
  --muon-events 10000 \
  --events 3000 \
  --seeds 67
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from simulation.helpers.geometry_index import inspect_geometry_rows, load_geometry_variants
from simulation.helpers.run_plan import RunRecord, build_run_plans
from simulation.helpers.run_steps import (
    DATA_DIRECTORY,
    flatten_process_extras,
    maybe_remove_file,
    maybe_run_sweeps,
    run_ddsim,
    run_particle_response_calibration,
    run_muon_calibration,
    run_performance_analysis,
    run_process,
    write_calibration,
    write_metadata,
    write_run_manifests,
)

PROJECT_DIRECTORY = Path(__file__).resolve().parent


# Parse the CLI options for one campaign run.
def parse_args() -> argparse.Namespace:
    raw_args = sys.argv[1:]
    gun_energy_requested = any(token == "--gun-energy" or token.startswith("--gun-energy=") for token in raw_args)
    gun_kinetic_energy_requested = any(
        token == "--gun-kinetic-energy" or token.startswith("--gun-kinetic-energy=") for token in raw_args
    )
    parser = argparse.ArgumentParser(description="Orchestrate HCAL production runs.")
    parser.add_argument("--spec", "-s", nargs="+", required=True, help="Sweep spec(s) to materialise (YAML).")
    parser.add_argument("--overwrite-geos", action="store_true", help="Pass --overwrite to sweep_geometries.py.")
    parser.add_argument("--ddsim", default="ddsim", help="Path to ddsim executable (default: ddsim on PATH).")
    parser.add_argument("--root-bin", default="root", help="Path to ROOT executable.")
    parser.add_argument("--process-bin", default=str(PROJECT_DIRECTORY / "build" / "bin" / "process"), help="Path to the processor binary.")
    parser.add_argument(
        "--process-extra",
        action="append",
        nargs="?",
        const="",
        default=[],
        help="Optional chunk appended to the processor call.",
    )
    parser.add_argument(
        "--gun-particle",
        nargs="+",
        default=["neutron"],
        help="Signal gun particle name(s).",
    )
    parser.add_argument("--gun-energy", type=float, nargs="+", default=None, help="Total gun energies in GeV passed to ddsim.")
    parser.add_argument(
        "--gun-kinetic-energy",
        type=float,
        nargs="+",
        default=None,
        help="Gun kinetic energies in GeV. Converted to total energy before calling ddsim.",
    )
    parser.add_argument("--gun-position", default="0 0 0", help="Gun position string passed to ddsim.")
    parser.add_argument("--gun-direction", default="0 0 -1", help="Gun direction string passed to ddsim.")
    parser.add_argument("--events", type=int, default=1, help="Number of signal events per ddsim run.")
    parser.add_argument(
        "--muon-events",
        type=int,
        default=2000,
        help="Number of muon-control events used for threshold calibration.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Random seeds for ddsim. Omit to let ddsim pick its own.")
    parser.add_argument(
        "--muon-threshold",
        type=float,
        default=None,
        help="Fixed visible-energy threshold in GeV. Omit to calibrate one threshold per geometry.",
    )
    parser.add_argument("--manifest-json", default=str(DATA_DIRECTORY / "manifests" / "run_manifest.json"), help="Path for JSON run manifest.")
    parser.add_argument("--manifest-csv", default=str(DATA_DIRECTORY / "manifests" / "run_manifest.csv"), help="Path for CSV run manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing external commands.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if outputs already exist.")
    parser.add_argument(
        "--delete-intermediates",
        action="store_true",
        help="Delete intermediate ROOT files after downstream steps finish successfully.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for helper scripts.")
    args = parser.parse_args()
    if gun_energy_requested and gun_kinetic_energy_requested:
        print("WARNING: --gun-energy and --gun-kinetic-energy cannot be used together.", file=sys.stderr)
        parser.error("choose only one of --gun-energy or --gun-kinetic-energy.")
    if args.gun_energy is None and args.gun_kinetic_energy is None:
        args.gun_energy = [5.0]
    return args


# Resolve a CLI path against the project root.
def resolve_runtime_path(path_text: str) -> Path:
    raw_path = Path(path_text).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (PROJECT_DIRECTORY / raw_path).resolve()


# Resolve each requested sweep spec path.
def resolve_spec_paths(spec_texts: List[str]) -> List[Path]:
    return [resolve_runtime_path(spec_text) for spec_text in spec_texts]


# Run one full campaign and write the final manifests.
def main() -> None:
    args = parse_args()

    # Resolve output paths.
    args.process_bin = str(resolve_runtime_path(args.process_bin))
    args.manifest_json = str(resolve_runtime_path(args.manifest_json))
    args.manifest_csv = str(resolve_runtime_path(args.manifest_csv))

    # Materialize the requested sweep specs before building run plans.
    spec_paths = resolve_spec_paths(args.spec)
    maybe_run_sweeps(args, spec_paths)

    # Load the geometry variants.
    geometry_rows = inspect_geometry_rows(args.python, spec_paths)
    require_geometry_files = not args.dry_run
    geometry_variants = load_geometry_variants(
        geometry_rows,
        require_geometry_files=require_geometry_files,
    )
    if not geometry_variants:
        print("No geometry variants found; nothing to do.")
        return

    # Flatten any extra processor flags once before planning runs.
    extra_process_flags = flatten_process_extras(args.process_extra)

    # Parse all input gun particles.
    requested_particles = [particle.strip() for particle in args.gun_particle if particle.strip()]
    if not requested_particles:
        raise ValueError("At least one --gun-particle value is required.")
    args.gun_particle = requested_particles

    requested_particle_names = {particle.lower() for particle in requested_particles}
    all_requested_particles_are_muons = requested_particle_names.issubset({"mu-", "mu+"})
    has_non_muon_signal_particle = not all_requested_particles_are_muons

    # Run one geometry at a time so each geometry finishes fully before the next one starts.
    run_records: List[RunRecord] = []
    any_run_plans = False
    for geometry_variant in geometry_variants:
        geometry_muon_threshold = args.muon_threshold

        # For non-muon campaigns, derive one threshold per geometry only when the caller did
        # not provide a fixed threshold on the CLI.
        if has_non_muon_signal_particle and geometry_muon_threshold is None:
            calibration_json_path = run_muon_calibration(args, geometry_variant)
            if not args.dry_run:
                with calibration_json_path.open("r", encoding="utf-8") as calibration_file:
                    payload = json.load(calibration_file)
                threshold_value = payload.get("muon_threshold_GeV")
                if threshold_value is None:
                    raise ValueError(f"muon_threshold_GeV missing in {calibration_json_path}")
                geometry_muon_threshold = float(threshold_value)

        # Expand only this geometry into run plans so all of its energies and seeds stay together.
        geometry_run_plans = build_run_plans(args, [geometry_variant], extra_process_flags)
        if not geometry_run_plans:
            continue
        any_run_plans = True

        # Finish every planned run for this geometry before moving to the next geometry.
        for run_plan in geometry_run_plans:
            run_record = RunRecord(plan=run_plan, status="pending")

            # Reuse an existing processed run unless overwrite was requested.
            if not args.overwrite and run_plan.events_path.exists():
                run_record.status = "skipped_existing"
                run_records.append(run_record)
                print(f"[skip] {run_plan.run_id} (events already present)")
                continue

            saved_muon_threshold = args.muon_threshold

            try:
                response_scale = None
                is_muon_signal = run_plan.gun_particle.strip().lower() in ("mu-", "mu+")

                # Use either the fixed CLI threshold or the derived geometry threshold on each
                # non-muon run.
                if not is_muon_signal and geometry_muon_threshold is not None:
                    args.muon_threshold = geometry_muon_threshold

                # Run the full chain for this plan from simulation through analysis outputs.
                run_record.ddsim_seconds = run_ddsim(args, run_plan)
                run_record.process_seconds, _ = run_process(
                    args,
                    run_plan,
                    extra_process_flags,
                )
                maybe_remove_file(args, run_plan.raw_path)
                if not is_muon_signal:
                    _, response_scale = run_particle_response_calibration(args, run_plan)
                run_record.meta_seconds = write_metadata(args, run_plan)
                write_calibration(args, run_plan, response_scale)
                if not is_muon_signal:
                    run_record.performance_seconds = run_performance_analysis(args, run_plan)
                maybe_remove_file(args, run_plan.events_path)
                run_record.status = "completed"
            except subprocess.CalledProcessError as exc:
                run_record.status = "failed"
                run_record.error = f"{exc.cmd} -> exit {exc.returncode}"
            except Exception as exc:
                run_record.status = "failed"
                run_record.error = str(exc)
            finally:
                # Restore the caller-level threshold before the next run starts.
                args.muon_threshold = saved_muon_threshold

            run_records.append(run_record)

    if not any_run_plans:
        print("No run plans generated; adjust seeds/energies.")
        return

    # Write the final manifest after all planned runs finish.
    write_run_manifests(run_records, Path(args.manifest_json), Path(args.manifest_csv))
    completed = sum(1 for run_record in run_records if run_record.status == "completed")
    skipped = sum(1 for run_record in run_records if run_record.status.startswith("skipped"))
    failed = sum(1 for run_record in run_records if run_record.status == "failed")
    print(f"[summary] completed={completed} skipped={skipped} failed={failed} total={len(run_records)}")


if __name__ == "__main__":
    main()
