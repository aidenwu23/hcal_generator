#!/usr/bin/env python3
"""conductor.py

Backbone orchestration script for the backward HCAL neutron campaign.
It stitches together geometry sweeps, ddsim production, processing,
metadata writing, performance analysis, and run manifests.

Example CLI
-----------
python3 conductor.py \
    --spec geometries/sweeps/test_run.yaml \
    --process-bin ./build/bin/process \
    --ddsim ddsim \
    --root-bin root \
    --events-per-run 2000 \
    --gun-particle neutron \
    --gun-energy 5 \
    --seeds 67 \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from simulation.helpers.geometry_index import inspect_geometry_rows, load_geometry_variants
from simulation.helpers.run_plan import RunRecord, build_run_plans
from simulation.helpers.run_steps import (
    DATA_DIRECTORY,
    flatten_process_extras,
    maybe_run_sweeps,
    run_ddsim,
    run_neutron_calibration,
    run_muon_calibration,
    run_performance_analysis,
    run_process,
    write_calibration,
    write_metadata,
    write_run_manifests,
)

PROJECT_DIRECTORY = Path(__file__).resolve().parent


# Collect the runtime switches that define one orchestration pass over the geometry set.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate neutron HCAL production runs.")
    parser.add_argument("--spec", "-s", nargs="+", required=True, help="Sweep spec(s) to materialise (YAML).")
    parser.add_argument("--skip-sweep", action="store_true", help="Assume geometries already generated; skip sweep_geometries.py.")
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
    parser.add_argument("--gun-particle", default="neutron", help="Primary gun particle (default: neutron).")
    parser.add_argument("--gun-energy", type=float, nargs="+", default=[10.0], help="Gun energies in GeV (default: 10).")
    parser.add_argument("--gun-position", default="0 0 0", help="Gun position string passed to ddsim.")
    parser.add_argument("--gun-direction", default="0 0 -1", help="Gun direction string passed to ddsim.")
    parser.add_argument("--events-per-run", type=int, default=1, help="Number of events per ddsim run.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Random seeds for ddsim. Omit to let ddsim pick its own.")
    parser.add_argument(
        "--expected-pdg",
        type=int,
        help="Override the expected MC PDG code; defaults to gun particle lookup when available.",
    )
    parser.add_argument("--muon-threshold", type=float, default=0.05, help="Visible-energy threshold (GeV) used when no muon control calibration overrides it.")
    parser.add_argument("--manifest-json", default=str(DATA_DIRECTORY / "manifests" / "run_manifest.json"), help="Path for JSON run manifest.")
    parser.add_argument("--manifest-csv", default=str(DATA_DIRECTORY / "manifests" / "run_manifest.csv"), help="Path for CSV run manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing external commands.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if outputs already exist.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for helper scripts.")
    return parser.parse_args()


# Resolve CLI paths relative to the project root so conductor.py behaves the same whether it
# is launched from the repo root or from somewhere else.
def resolve_runtime_path(path_text: str) -> Path:
    raw_path = Path(path_text).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (PROJECT_DIRECTORY / raw_path).resolve()


# Normalize every requested sweep spec into an absolute path before the geometry pipeline runs.
def resolve_spec_paths(spec_texts: List[str]) -> List[Path]:
    return [resolve_runtime_path(spec_text) for spec_text in spec_texts]


# Run one full campaign pass: materialize geometries, build run plans, execute each run,
# then write the manifest that records what completed, skipped, or failed.
def main() -> None:
    args = parse_args()

    # Resolve the runtime file locations once so the helper calls below all see stable paths.
    args.process_bin = str(resolve_runtime_path(args.process_bin))
    args.manifest_json = str(resolve_runtime_path(args.manifest_json))
    args.manifest_csv = str(resolve_runtime_path(args.manifest_csv))

    # Turn the requested sweep specs into generated geometries before any run planning starts.
    spec_paths = resolve_spec_paths(args.spec)
    maybe_run_sweeps(args, spec_paths)

    # Build the in-memory geometry list that every downstream run will reference.
    geometry_rows = inspect_geometry_rows(args.python, spec_paths)
    require_geometry_files = args.skip_sweep or not args.dry_run
    geometry_variants = load_geometry_variants(
        geometry_rows,
        require_geometry_files=require_geometry_files,
    )
    if not geometry_variants:
        print("No geometry variants found; nothing to do.")
        return

    # Prepare any optional processor flags that should be threaded through every run.
    extra_process_flags = flatten_process_extras(args.process_extra)

    # For non-muon runs, collect a per-geometry muon calibration threshold before any signal
    # events are processed. The threshold is used downstream to separate real hadronic deposits
    # from noise, so it must be ready before the first neutron run starts.
    muon_threshold_by_geometry_id: Dict[str, float] = {}
    gun_particle = args.gun_particle.strip().lower()
    running_muon_sample = gun_particle in ("mu-", "mu+")
    if not running_muon_sample:
        for geometry_variant in geometry_variants:
            calibration_json_path = run_muon_calibration(args, geometry_variant)
            if args.dry_run:
                muon_threshold_by_geometry_id[geometry_variant.geometry_id] = float(args.muon_threshold)
                continue
            with calibration_json_path.open("r", encoding="utf-8") as calibration_file:
                payload = json.load(calibration_file)
            threshold_value = payload.get("muon_threshold_GeV")
            if threshold_value is None:
                raise ValueError(f"muon_threshold_GeV missing in {calibration_json_path}")
            muon_threshold_by_geometry_id[geometry_variant.geometry_id] = float(threshold_value)

    # Expand the geometry set, energies, and seeds into the concrete list of ddsim runs.
    run_plans = build_run_plans(args, geometry_variants, extra_process_flags)
    if not run_plans:
        print("No run plans generated; adjust seeds/energies.")
        return

    # Execute each planned run in order and keep a record of timing, status, and failures
    # so the manifest at the end reflects the full campaign history.
    run_records: List[RunRecord] = []
    for run_plan in run_plans:
        run_record = RunRecord(plan=run_plan, status="pending")

        # Reuse existing event files unless the user explicitly asked to overwrite them.
        if not args.overwrite and run_plan.events_path.exists():
            run_record.status = "skipped_existing"
            run_records.append(run_record)
            print(f"[skip] {run_plan.run_id} (events already present)")
            continue

        # Save the default muon threshold so it can be restored after this run, regardless of
        # whether the per-geometry calibration overrides it or the run fails partway through.
        saved_muon_threshold = args.muon_threshold

        try:
            neutron_scale = None
            is_neutron = run_plan.gun_particle.strip().lower() == "neutron"

            # Replace the default threshold with the value derived from the muon control sample
            # for this specific geometry, so the detection cut reflects its actual noise floor.
            if not running_muon_sample:
                geometry_id = run_plan.geometry_variant.geometry_id
                if geometry_id not in muon_threshold_by_geometry_id:
                    raise RuntimeError(f"Missing muon calibration threshold for geometry {geometry_id}")
                args.muon_threshold = muon_threshold_by_geometry_id[geometry_id]

            # Run the detector simulation first, then process the output, then attach any
            # calibration and performance products that belong to this sample type.
            run_record.ddsim_seconds = run_ddsim(args, run_plan)
            run_record.process_seconds, _ = run_process(
                args,
                run_plan,
                extra_process_flags,
            )
            if is_neutron:
                _, neutron_scale = run_neutron_calibration(args, run_plan)
            run_record.meta_seconds = write_metadata(args, run_plan)
            write_calibration(args, run_plan, neutron_scale)
            if is_neutron:
                run_record.performance_seconds = run_performance_analysis(args, run_plan)
            run_record.status = "completed"
        except subprocess.CalledProcessError as exc:
            # Keep the shell command and exit code so failed external steps are visible in the manifest.
            run_record.status = "failed"
            run_record.error = f"{exc.cmd} -> exit {exc.returncode}"
        except Exception as exc:
            run_record.status = "failed"
            run_record.error = str(exc)
        finally:
            # Restore the caller-level muon threshold before moving to the next planned run.
            args.muon_threshold = saved_muon_threshold

        run_records.append(run_record)

    # Write the run manifest after every planned run has either completed, failed, or been skipped.
    write_run_manifests(run_records, Path(args.manifest_json), Path(args.manifest_csv))
    completed = sum(1 for run_record in run_records if run_record.status == "completed")
    skipped = sum(1 for run_record in run_records if run_record.status.startswith("skipped"))
    failed = sum(1 for run_record in run_records if run_record.status == "failed")
    print(f"[summary] completed={completed} skipped={skipped} failed={failed} total={len(run_records)}")


if __name__ == "__main__":
    main()
