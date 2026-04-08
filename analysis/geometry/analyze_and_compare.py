#!/usr/bin/env python3
"""
Run theory interaction-depth analysis and comparison for two geometries.
Example:
python analysis/geometry/analyze_and_compare.py \
  --reference geometries/generated/<ref_geometry>/geometry.json \
  --candidate geometries/generated/<cand_geometry>/geometry.json

python analysis/geometry/analyze_and_compare.py \
  --reference geometries/generated/04e3fdfb/geometry.json \
  --candidate geometries/generated/57fc2ba4/geometry.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
THEORY_WRAPPER_PATH = (
    PROJECT_DIRECTORY / "analysis" / "geometry" / "theory" / "run_interaction_depth.py"
)
COMPARE_PROBABILITY_PATH = (
    PROJECT_DIRECTORY / "analysis" / "geometry" / "compare" / "compare_probability.py"
)


@dataclass
class GeometryInput:
    geometry_json_path: Path
    geometry_id: str


def parse_arguments() -> argparse.Namespace:
    # Read the two geometry inputs used by the theory analysis and comparison steps.
    parser = argparse.ArgumentParser(
        description="Run theory geometry comparison for two geometry.json inputs."
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference geometry.json path.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate geometry.json path.",
    )
    return parser.parse_args()


def build_geometry_input(path_value: str) -> GeometryInput:
    # Derive the geometry ID from one generated geometry.json path.
    geometry_json_path = Path(path_value).expanduser().resolve()
    if not geometry_json_path.exists():
        raise FileNotFoundError(f"Geometry JSON not found: {geometry_json_path}")
    if geometry_json_path.name != "geometry.json":
        raise ValueError(f"Expected a geometry.json path: {geometry_json_path}")

    geometry_id = geometry_json_path.parent.name
    if not geometry_id:
        raise ValueError(f"Geometry path is missing a geometry ID directory: {geometry_json_path}")

    return GeometryInput(
        geometry_json_path=geometry_json_path,
        geometry_id=geometry_id,
    )


def theory_layers_path(geometry_id: str) -> Path:
    return PROJECT_DIRECTORY / "data" / "geometry_analysis" / geometry_id / "layers.csv"


def run_command(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=PROJECT_DIRECTORY,
        check=True,
    )


def run_theory_analysis(geometry_json_path: Path) -> None:
    run_command(
        [
            sys.executable,
            str(THEORY_WRAPPER_PATH),
            "--geometry-json",
            str(geometry_json_path),
        ]
    )


def run_theory_comparison(reference_geometry: GeometryInput, candidate_geometry: GeometryInput) -> None:
    run_command(
        [
            sys.executable,
            str(COMPARE_PROBABILITY_PATH),
            "--reference",
            str(theory_layers_path(reference_geometry.geometry_id)),
            "--candidate",
            str(theory_layers_path(candidate_geometry.geometry_id)),
        ]
    )


def validate_inputs(reference_geometry: GeometryInput, candidate_geometry: GeometryInput) -> None:
    # Make sure both inputs point to generated geometry files.
    for geometry_input in (reference_geometry, candidate_geometry):
        if not geometry_input.geometry_json_path.exists():
            raise FileNotFoundError(f"Geometry JSON not found: {geometry_input.geometry_json_path}")


def main() -> int:
    # Resolve both inputs, run the theory producer, then compare the two geometries.
    arguments = parse_arguments()
    reference_geometry = build_geometry_input(arguments.reference)
    candidate_geometry = build_geometry_input(arguments.candidate)
    validate_inputs(reference_geometry, candidate_geometry)

    # Run the theory producer once per unique geometry before comparing the geometry-level curves.
    geometry_inputs_by_id = {
        geometry_input.geometry_id: geometry_input
        for geometry_input in (reference_geometry, candidate_geometry)
    }
    for geometry_input in geometry_inputs_by_id.values():
        run_theory_analysis(geometry_input.geometry_json_path)

    run_theory_comparison(reference_geometry, candidate_geometry)

    print("[analyze_and_compare] finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
