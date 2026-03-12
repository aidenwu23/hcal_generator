#!/usr/bin/env python3
"""
Sweep muon thresholds for one neutron run and write a CSV.

Example:
python3 analysis/result_validation/scan_muon_threshold.py \
  --events-root data/processed/81c3da7d/run19734f434f/events.root \
  --threshold-min 0.02 \
  --threshold-max 0.04 \
  --threshold-step 0.002 \
  --out-csv data/result_validation/81c3da7d_threshold_scan.csv

python3 analysis/result_validation/scan_muon_threshold.py \
  --events-root data/processed/06529cd5/runb78e64f42d/events.root \
  --threshold-min 0.02 \
  --threshold-max 0.04 \
  --threshold-step 0.002 \
  --out-csv data/result_validation/06529cd5_threshold_scan.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from simulation.helpers.geometry_index import GeometryVariant
from simulation.helpers.run_steps import (
    run_neutron_calibration,
    run_performance_analysis,
    write_calibration,
)

GEOMETRY_ROOT_CANDIDATES = [
    PROJECT_DIRECTORY / "geometries" / "generated",
    PROJECT_DIRECTORY / "geometries" / "baseline",
]
VALIDATION_ROOT_DIRECTORY = PROJECT_DIRECTORY / "data" / "result_validation"
TAG_PATTERN = re.compile(r"^\s*Tag:\s*(.+?)\s*$", re.MULTILINE)
USE_CONDUCTOR_ERROR = "Use conductor.py to generate processed run files first."


@dataclass
class ValidationRun:
    geometry_variant: GeometryVariant
    gun_energy_GeV: float
    run_id: str
    events_path: Path
    meta_path: Path
    calibration_path: Path
    performance_path: Path


@dataclass
class EfficiencyScanRow:
    muon_threshold_GeV: float
    detection_efficiency: float
    eff_lo: float
    eff_hi: float


def parse_arguments() -> argparse.Namespace:
    # Read one processed run input, the threshold scan, and the CSV output path.
    parser = argparse.ArgumentParser(
        description="Sweep muon thresholds for one processed neutron run and write a CSV."
    )
    parser.add_argument(
        "--events-root",
        required=True,
        help="Path to an existing processed events.root from conductor.py.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="Explicit muon threshold in GeV. Repeat to add more values.",
    )
    parser.add_argument("--threshold-min", type=float, help="Minimum muon threshold in GeV.")
    parser.add_argument("--threshold-max", type=float, help="Maximum muon threshold in GeV.")
    parser.add_argument("--threshold-step", type=float, help="Muon threshold step in GeV.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--root-bin", default="root", help="Path to ROOT.")
    parser.add_argument(
        "--overwrite-validation",
        action="store_true",
        help="Recompute validation outputs even when they already exist.",
    )
    return parser.parse_args()


def build_threshold_list(arguments: argparse.Namespace) -> List[float]:
    # Accept either an explicit threshold list or a uniform threshold range.
    explicit_thresholds = [float(value) for value in arguments.threshold]
    if explicit_thresholds:
        return sorted(dict.fromkeys(explicit_thresholds))

    range_values = [
        arguments.threshold_min,
        arguments.threshold_max,
        arguments.threshold_step,
    ]
    if any(value is None for value in range_values):
        raise ValueError(
            "Provide either repeated --threshold values or the full "
            "--threshold-min/--threshold-max/--threshold-step range."
        )

    threshold_min = float(arguments.threshold_min)
    threshold_max = float(arguments.threshold_max)
    threshold_step = float(arguments.threshold_step)
    if threshold_step <= 0.0:
        raise ValueError("--threshold-step must be positive.")
    if threshold_max < threshold_min:
        raise ValueError("--threshold-max must be greater than or equal to --threshold-min.")

    # Build the scan points explicitly so the CSV keeps one stable threshold row per step.
    threshold_values: List[float] = []
    point_count = int(round((threshold_max - threshold_min) / threshold_step))
    # Walk the threshold range one fixed step at a time.
    for point_index in range(point_count + 1):
        threshold_value = threshold_min + point_index * threshold_step
        if threshold_value > threshold_max + 1e-12:
            break
        threshold_values.append(round(threshold_value, 12))

    if not threshold_values:
        raise ValueError("No threshold points were generated.")
    return threshold_values


def find_geometry_directory(geometry_id: str) -> Path:
    for root_directory in GEOMETRY_ROOT_CANDIDATES:
        geometry_directory = root_directory / geometry_id
        if geometry_directory.exists():
            return geometry_directory
    raise FileNotFoundError(f"Geometry directory not found for geometry_id={geometry_id}.")


def read_geometry_tag(xml_path: Path) -> str:
    xml_text = xml_path.read_text(encoding="utf-8")
    match = TAG_PATTERN.search(xml_text)
    if match:
        return match.group(1).strip()
    return xml_path.parent.name


def load_geometry_variant(geometry_id: str) -> GeometryVariant:
    # Load the generated geometry metadata that the wrapped run helpers expect.
    geometry_directory = find_geometry_directory(geometry_id)
    params_path = geometry_directory / "geometry.json"
    xml_path = geometry_directory / "geometry.xml"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing geometry.json: {params_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing geometry.xml: {xml_path}")

    with params_path.open("r", encoding="utf-8") as params_file:
        params = json.load(params_file)

    return GeometryVariant(
        geometry_id=geometry_id,
        tag=read_geometry_tag(xml_path),
        geometry_directory=geometry_directory,
        params_path=params_path,
        xml_path=xml_path,
        spec_path=params_path,
        params={str(key): value for key, value in params.items()},
    )


def require_processed_run_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(USE_CONDUCTOR_ERROR)


def load_processed_run_plan(events_root_path: Path) -> ValidationRun:
    # Resolve one processed run from the standard conductor file layout.
    require_processed_run_file(events_root_path)
    if events_root_path.name != "events.root":
        raise ValueError("--events-root must point to an events.root file.")

    processed_directory = events_root_path.parent
    meta_path = processed_directory / "meta.json"
    require_processed_run_file(meta_path)

    with meta_path.open("r", encoding="utf-8") as meta_file:
        meta_payload = json.load(meta_file)

    geometry_id = str(meta_payload.get("geometry_id", "")).strip()
    if not geometry_id:
        raise ValueError(f"geometry_id missing in {meta_path}")

    gun_energy_GeV = meta_payload.get("gun_energy_GeV")
    if gun_energy_GeV is None:
        raise ValueError(f"gun_energy_GeV missing in {meta_path}")

    geometry_variant = load_geometry_variant(geometry_id)
    run_id = processed_directory.name
    return ValidationRun(
        geometry_variant=geometry_variant,
        gun_energy_GeV=float(gun_energy_GeV),
        run_id=run_id,
        events_path=events_root_path,
        meta_path=meta_path,
        calibration_path=processed_directory / "calibration.json",
        performance_path=processed_directory / "performance.json",
    )


def build_threshold_run_plan(
    processed_run_plan: ValidationRun,
    threshold_GeV: float,
) -> ValidationRun:
    # Keep threshold-specific calibration and performance outputs separate by threshold value.
    threshold_label = f"{threshold_GeV:.12g}".replace("-", "m").replace(".", "p")
    validation_directory = (
        VALIDATION_ROOT_DIRECTORY
        / processed_run_plan.geometry_variant.geometry_id
        / processed_run_plan.run_id
        / f"threshold_{threshold_label}"
    )
    return ValidationRun(
        geometry_variant=processed_run_plan.geometry_variant,
        gun_energy_GeV=processed_run_plan.gun_energy_GeV,
        run_id=processed_run_plan.run_id,
        events_path=processed_run_plan.events_path,
        meta_path=processed_run_plan.meta_path,
        calibration_path=validation_directory / "calibration.json",
        performance_path=validation_directory / "performance.json",
    )


def build_runtime_args(arguments: argparse.Namespace, muon_threshold_GeV: float) -> SimpleNamespace:
    return SimpleNamespace(
        delete_intermediates=False,
        dry_run=False,
        root_bin=arguments.root_bin,
        muon_threshold=muon_threshold_GeV,
    )


def load_efficiency_row(
    performance_path: Path,
    muon_threshold_GeV: float,
) -> EfficiencyScanRow:
    # Read the central efficiency value and the Wilson interval from one performance summary.
    with performance_path.open("r", encoding="utf-8") as performance_file:
        payload = json.load(performance_file)

    detection_efficiency = payload.get("detection_efficiency")
    eff_lo = payload.get("eff_lo")
    eff_hi = payload.get("eff_hi")
    if detection_efficiency is None or eff_lo is None or eff_hi is None:
        raise ValueError(
            f"detection_efficiency, eff_lo, or eff_hi missing in {performance_path}"
        )
    return EfficiencyScanRow(
        muon_threshold_GeV=muon_threshold_GeV,
        detection_efficiency=float(detection_efficiency),
        eff_lo=float(eff_lo),
        eff_hi=float(eff_hi),
    )


def sweep_thresholds(
    arguments: argparse.Namespace,
    processed_run_plan: ValidationRun,
    threshold_values: Iterable[float],
) -> List[EfficiencyScanRow]:
    # Reuse the same processed events file for every threshold point in the scan.
    rows: List[EfficiencyScanRow] = []
    for threshold_GeV in threshold_values:
        threshold_run_plan = build_threshold_run_plan(processed_run_plan, threshold_GeV)
        runtime_args = build_runtime_args(arguments, muon_threshold_GeV=threshold_GeV)

        # Recompute the threshold-specific calibration and performance on the same events.root.
        if arguments.overwrite_validation or not threshold_run_plan.performance_path.exists():
            _, neutron_scale = run_neutron_calibration(runtime_args, threshold_run_plan)
            write_calibration(runtime_args, threshold_run_plan, neutron_scale)
            run_performance_analysis(runtime_args, threshold_run_plan)

        rows.append(load_efficiency_row(threshold_run_plan.performance_path, threshold_GeV))
    return rows


def write_csv(output_csv_path: Path, rows: Iterable[EfficiencyScanRow]) -> None:
    # Write one CSV row per threshold point in the scan.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["muon_threshold_GeV", "detection_efficiency", "eff_lo", "eff_hi"])
        for row in rows:
            writer.writerow(
                [
                    f"{row.muon_threshold_GeV:.12g}",
                    f"{row.detection_efficiency:.12g}",
                    f"{row.eff_lo:.12g}",
                    f"{row.eff_hi:.12g}",
                ]
            )


def main() -> int:
    arguments = parse_arguments()
    threshold_values = build_threshold_list(arguments)
    events_root_path = Path(arguments.events_root).expanduser().resolve()
    processed_run_plan = load_processed_run_plan(events_root_path)

    rows = sweep_thresholds(arguments, processed_run_plan, threshold_values)
    output_csv_path = Path(arguments.out_csv).expanduser().resolve()
    write_csv(output_csv_path, rows)
    print(f"[scan_muon_threshold] wrote {output_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
