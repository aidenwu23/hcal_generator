#!/usr/bin/env python3
"""
Compact a run-level surrogate training CSV into one row per geometry.
Example:
python3 surrogate/compact_training_csv.py \
  --in surrogate/campaigns/0.5-2.5_GeV_raw.csv \
  --out surrogate/campaigns/0.5-2.5_GeV_compact.csv

python3 surrogate/compact_training_csv.py \
  --in csv_data/test.csv \
  --out csv_data/test_compact.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

GEOMETRY_FEATURES = [
    "nLayers",
    "seg1_layers",
    "seg2_layers",
    "seg3_layers",
    "t_absorber_seg1",
    "t_absorber_seg2",
    "t_absorber_seg3",
    "t_scin_seg1",
    "t_scin_seg2",
    "t_scin_seg3",
    "t_spacer",
]

RUN_METRICS = [
    "detection_efficiency",
    "tiles_mean",
    "layers_mean",
]

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact a run-level surrogate training CSV into one geometry row."
    )
    parser.add_argument(
        "--in",
        dest="input_csv",
        required=True,
        help="Input run-level training CSV path.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output geometry compact CSV path.",
    )
    return parser.parse_args()


def parse_float(value_text: str) -> float | None:
    value_text = value_text.strip()
    if not value_text:
        return None
    try:
        return float(value_text)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    average_value = sum(values) / len(values)
    variance = sum((value - average_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def normalize_particle_name(particle_name: str) -> str:
    return particle_name.strip()


def particle_column_prefix(particle_name: str) -> str:
    # Convert a particle name into a CSV-safe suffix.
    normalized_name = normalize_particle_name(particle_name)
    column_prefix = normalized_name.replace("+", "_plus_").replace("-", "_minus_")
    column_prefix = re.sub(r"[^A-Za-z0-9]+", "_", column_prefix).strip("_")
    if not column_prefix:
        raise ValueError(f"Cannot derive a column prefix from particle name {particle_name!r}.")
    if column_prefix[0].isdigit():
        column_prefix = f"particle_{column_prefix}"
    return column_prefix


def build_particle_column_map(particle_names: list[str]) -> dict[str, str]:
    # Keep one unique output prefix per particle name.
    column_map: dict[str, str] = {}
    names_by_prefix: dict[str, str] = {}
    for particle_name in particle_names:
        column_prefix = particle_column_prefix(particle_name)
        if column_prefix in names_by_prefix and names_by_prefix[column_prefix] != particle_name:
            raise ValueError(
                f"Particle names {names_by_prefix[column_prefix]!r} and {particle_name!r} "
                f"both map to the same output prefix {column_prefix!r}."
            )
        names_by_prefix[column_prefix] = particle_name
        column_map[particle_name] = column_prefix
    return column_map


def main() -> int:
    # Collapse the run-level CSV into one geometry row with per-particle metrics.
    arguments = parse_arguments()
    input_path = Path(arguments.input_csv).expanduser().resolve()
    output_path = Path(arguments.out).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {input_path}")

    # Read the input rows once and collect the particle names that appear in the file.
    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} does not contain a CSV header.")
        required_columns = [
            "geometry_id",
            "gun_particle",
            *RUN_METRICS,
            *GEOMETRY_FEATURES,
        ]
        missing_required_columns = [
            column_name for column_name in required_columns if column_name not in reader.fieldnames
        ]
        if missing_required_columns:
            raise ValueError(f"{input_path} is missing required columns: {missing_required_columns}")

        rows_by_geometry: dict[str, list[dict[str, str]]] = defaultdict(list)
        particle_names: set[str] = set()
        for row in reader:
            geometry_id = row.get("geometry_id", "").strip()
            if not geometry_id:
                raise ValueError("Encountered a row with an empty geometry_id.")
            particle_name = normalize_particle_name(row.get("gun_particle", ""))
            if not particle_name:
                raise ValueError(f"Encountered a row with an empty gun_particle for geometry_id={geometry_id}.")
            particle_names.add(particle_name)
            rows_by_geometry[geometry_id].append(row)

    ordered_particle_names = sorted(particle_names)
    particle_column_map = build_particle_column_map(ordered_particle_names)
    output_rows: list[dict[str, object]] = []

    # Build one output row per geometry.
    for geometry_id in sorted(rows_by_geometry):
        geometry_rows = rows_by_geometry[geometry_id]
        reference_row = geometry_rows[0]
        output_row: dict[str, object] = {
            "geometry_id": geometry_id,
        }
        for column_name in GEOMETRY_FEATURES:
            output_row[column_name] = reference_row.get(column_name, "")

        # Split this geometry slice by particle before averaging the metrics.
        rows_by_particle: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in geometry_rows:
            particle_name = normalize_particle_name(row.get("gun_particle", ""))
            rows_by_particle[particle_name].append(row)

        for particle_name in ordered_particle_names:
            particle_rows = rows_by_particle.get(particle_name, [])
            column_prefix = particle_column_map[particle_name]

            metric_values: dict[str, list[float]] = {
                metric_name: [] for metric_name in RUN_METRICS
            }
            # Average the available metrics for this geometry-particle slice.
            for row in particle_rows:
                for metric_name in RUN_METRICS:
                    parsed_metric = parse_float(row.get(metric_name, ""))
                    if parsed_metric is not None:
                        metric_values[metric_name].append(parsed_metric)

            efficiency_column_name = f"{column_prefix}_efficiency"
            efficiency_values = metric_values["detection_efficiency"]
            output_row[efficiency_column_name] = "" if not efficiency_values else mean(efficiency_values)
            output_row[f"{efficiency_column_name}_std"] = "" if not efficiency_values else sample_std(efficiency_values)

            for metric_name in ("tiles_mean", "layers_mean"):
                metric_column_name = f"{column_prefix}_{metric_name}"
                values = metric_values[metric_name]
                output_row[metric_column_name] = "" if not values else mean(values)

        output_rows.append(output_row)

    fieldnames = [
        "geometry_id",
        *GEOMETRY_FEATURES,
    ]
    for particle_name in ordered_particle_names:
        column_prefix = particle_column_map[particle_name]
        fieldnames.extend(
            [
                f"{column_prefix}_efficiency",
                f"{column_prefix}_efficiency_std",
                f"{column_prefix}_tiles_mean",
                f"{column_prefix}_layers_mean",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Wrote {len(output_rows)} geometry rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
