#!/usr/bin/env python3
"""
Compact a run-level surrogate training CSV into one row per geometry and kinetic energy.
Example:
python3 surrogate/compact_training_csv.py \
  --in surrogate/iterations/1-3_GeV/iteration_2/training_raw_0-2.csv \
  --out surrogate/iterations/1-3_GeV/iteration_2/training_compact_0-2.csv

python3 surrogate/compact_training_csv.py \
  --in csv_data/data.csv \
  --out csv_data/data_compact.csv
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

# Kinetic energy is used as the row identity. 
ENERGY_COLUMN = "kinetic_energy_GeV"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact a run-level surrogate training CSV into one geometry-and-energy row."
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
        help="Output geometry-and-energy compact CSV path.",
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


def normalize_energy_value(energy_text: str) -> float:
    parsed_energy = parse_float(energy_text)
    if parsed_energy is None:
        raise ValueError(f"Encountered a row with an invalid {ENERGY_COLUMN}: {energy_text!r}.")
    return parsed_energy


def main() -> int:
    # Collapse the run-level CSV into one geometry-and-energy row with per-particle metrics.
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
            ENERGY_COLUMN,
            "detection_efficiency",
            "energy_resolution",
            *GEOMETRY_FEATURES,
        ]
        missing_required_columns = [
            column_name for column_name in required_columns if column_name not in reader.fieldnames
        ]
        if missing_required_columns:
            raise ValueError(f"{input_path} is missing required columns: {missing_required_columns}")

        rows_by_geometry_energy: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
        particle_names: set[str] = set()
        for row in reader:
            geometry_id = row.get("geometry_id", "").strip()
            if not geometry_id:
                raise ValueError("Encountered a row with an empty geometry_id.")
            kinetic_energy_gev = normalize_energy_value(row.get(ENERGY_COLUMN, ""))
            particle_name = normalize_particle_name(row.get("gun_particle", ""))
            if not particle_name:
                raise ValueError(f"Encountered a row with an empty gun_particle for geometry_id={geometry_id}.")
            particle_names.add(particle_name)
            rows_by_geometry_energy[(geometry_id, kinetic_energy_gev)].append(row)

    ordered_particle_names = sorted(particle_names)
    particle_column_map = build_particle_column_map(ordered_particle_names)
    output_rows: list[dict[str, object]] = []

    # Build one output row per geometry-and-energy pair.
    for geometry_id, kinetic_energy_gev in sorted(rows_by_geometry_energy):
        geometry_energy_rows = rows_by_geometry_energy[(geometry_id, kinetic_energy_gev)]
        reference_row = geometry_energy_rows[0]
        output_row: dict[str, object] = {
            "geometry_id": geometry_id,
            ENERGY_COLUMN: kinetic_energy_gev,
        }
        for column_name in GEOMETRY_FEATURES:
            output_row[column_name] = reference_row.get(column_name, "")

        # Split this geometry-and-energy slice by particle before averaging the metrics.
        rows_by_particle: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in geometry_energy_rows:
            particle_name = normalize_particle_name(row.get("gun_particle", ""))
            rows_by_particle[particle_name].append(row)

        for particle_name in ordered_particle_names:
            particle_rows = rows_by_particle.get(particle_name, [])
            column_prefix = particle_column_map[particle_name]

            efficiency_values = []
            resolution_values = []
            # Average the available metrics for this geometry-particle slice.
            for row in particle_rows:
                parsed_efficiency = parse_float(row.get("detection_efficiency", ""))
                if parsed_efficiency is not None:
                    efficiency_values.append(parsed_efficiency)
                parsed_resolution = parse_float(row.get("energy_resolution", ""))
                if parsed_resolution is not None:
                    resolution_values.append(parsed_resolution)

            efficiency_column_name = f"{column_prefix}_efficiency"
            resolution_column_name = f"{column_prefix}_energy_resolution"
            output_row[efficiency_column_name] = "" if not efficiency_values else mean(efficiency_values)
            output_row[f"{efficiency_column_name}_std"] = "" if not efficiency_values else sample_std(efficiency_values)
            output_row[resolution_column_name] = "" if not resolution_values else mean(resolution_values)
            output_row[f"{resolution_column_name}_std"] = "" if not resolution_values else sample_std(resolution_values)

        output_rows.append(output_row)

    fieldnames = [
        "geometry_id",
        ENERGY_COLUMN,
        *GEOMETRY_FEATURES,
    ]
    for particle_name in ordered_particle_names:
        column_prefix = particle_column_map[particle_name]
        fieldnames.extend(
            [
                f"{column_prefix}_efficiency",
                f"{column_prefix}_efficiency_std",
                f"{column_prefix}_energy_resolution",
                f"{column_prefix}_energy_resolution_std",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Wrote {len(output_rows)} geometry-energy rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
