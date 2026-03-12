#!/usr/bin/env python3
"""
Run observed analysis, theory analysis, and both comparison steps for two raw runs.
Example:
python analysis/geometry/analyze_and_compare.py \
  --reference data/raw/<ref_geometry>/<ref_run>.edm4hep.root \
  --candidate data/raw/<cand_geometry>/<cand_run>.edm4hep.root

python analysis/geometry/analyze_and_compare.py \
  --reference data/raw/81c3da7d/rune3896ec0d8.edm4hep.root \
  --candidate data/raw/3a2e74e3/runfc6edabae2.edm4hep.root

python3 analysis/geometry/analyze_and_compare.py \
  --reference data/raw/81c3da7d/rune71b4906cd.edm4hep.root \
  --candidate data/raw/ff336efe/rundeb51474c4.edm4hep.root
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
OBSERVED_MACRO_PATH = (
    PROJECT_DIRECTORY / "analysis" / "geometry" / "observed" / "plot_observed_interaction_depth.C"
)
THEORY_WRAPPER_PATH = (
    PROJECT_DIRECTORY / "analysis" / "geometry" / "theory" / "run_interaction_depth.py"
)
COMPARE_OBSERVED_PATH = (
    PROJECT_DIRECTORY / "analysis" / "geometry" / "compare" / "compare_observed.py"
)
COMPARE_PROBABILITY_PATH = (
    PROJECT_DIRECTORY / "analysis" / "geometry" / "compare" / "compare_probability.py"
)


@dataclass
class RawRunInput:
    raw_path: Path
    geometry_id: str
    run_id: str


def parse_arguments() -> argparse.Namespace:
    # Read the two raw run inputs and the optional ROOT executable override.
    parser = argparse.ArgumentParser(
        description="Run observed and theory geometry comparisons for two raw runs."
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference raw EDM4hep path.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate raw EDM4hep path.",
    )
    parser.add_argument(
        "--root-bin",
        default="root",
        help="ROOT executable to use for the observed plotting step.",
    )
    return parser.parse_args()


def build_raw_run_input(path_value: str) -> RawRunInput:
    # Derive the geometry ID and run ID from one raw EDM4hep path.
    raw_path = Path(path_value).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw EDM4hep file not found: {raw_path}")
    if raw_path.parent == raw_path:
        raise ValueError(f"Cannot derive geometry ID from raw path: {raw_path}")

    file_name = raw_path.name
    suffix = ".edm4hep.root"
    if not file_name.endswith(suffix):
        raise ValueError(
            f"Raw EDM4hep path must end with {suffix}: {raw_path}"
        )

    geometry_id = raw_path.parent.name
    run_id = file_name[: -len(suffix)]
    if not geometry_id:
        raise ValueError(f"Raw path is missing a geometry ID directory: {raw_path}")
    if not run_id:
        raise ValueError(f"Raw path is missing a run ID file name: {raw_path}")

    return RawRunInput(
        raw_path=raw_path,
        geometry_id=geometry_id,
        run_id=run_id,
    )


def geometry_json_path(geometry_id: str) -> Path:
    return PROJECT_DIRECTORY / "geometries" / "generated" / geometry_id / "geometry.json"


def observed_csv_path(run_input: RawRunInput) -> Path:
    return (
        PROJECT_DIRECTORY
        / "data"
        / "geometry_analysis"
        / run_input.geometry_id
        / run_input.run_id
        / "start_layer_observed_layers.csv"
    )


def theory_layers_path(geometry_id: str) -> Path:
    return PROJECT_DIRECTORY / "data" / "geometry_analysis" / geometry_id / "layers.csv"


def run_command(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=PROJECT_DIRECTORY,
        check=True,
    )


def run_observed_analysis(run_input: RawRunInput, root_bin: str) -> None:
    macro_call = f'{OBSERVED_MACRO_PATH}("{run_input.raw_path}")'
    run_command([root_bin, "-l", "-b", "-q", macro_call])


def run_theory_analysis(geometry_id: str) -> None:
    run_command(
        [
            sys.executable,
            str(THEORY_WRAPPER_PATH),
            "--geometry-json",
            str(geometry_json_path(geometry_id)),
        ]
    )


def run_observed_comparison(reference_run: RawRunInput, candidate_run: RawRunInput) -> None:
    run_command(
        [
            sys.executable,
            str(COMPARE_OBSERVED_PATH),
            "--reference",
            str(observed_csv_path(reference_run)),
            "--candidate",
            str(observed_csv_path(candidate_run)),
        ]
    )


def run_theory_comparison(reference_run: RawRunInput, candidate_run: RawRunInput) -> None:
    run_command(
        [
            sys.executable,
            str(COMPARE_PROBABILITY_PATH),
            "--reference",
            str(theory_layers_path(reference_run.geometry_id)),
            "--candidate",
            str(theory_layers_path(candidate_run.geometry_id)),
        ]
    )


def validate_inputs(reference_run: RawRunInput, candidate_run: RawRunInput) -> None:
    # Make sure both raw inputs point to geometries that have generated theory inputs.
    for run_input in (reference_run, candidate_run):
        geometry_path = geometry_json_path(run_input.geometry_id)
        if not geometry_path.exists():
            raise FileNotFoundError(f"Geometry JSON not found: {geometry_path}")


def main() -> int:
    # Resolve both inputs, run the producers, then run the two comparison steps.
    arguments = parse_arguments()
    reference_run = build_raw_run_input(arguments.reference)
    candidate_run = build_raw_run_input(arguments.candidate)
    validate_inputs(reference_run, candidate_run)

    # Run the observed producer first so both run-level CSVs exist for the comparison step.
    run_observed_analysis(reference_run, arguments.root_bin)
    run_observed_analysis(candidate_run, arguments.root_bin)

    # Run the theory producer once per unique geometry before comparing the geometry-level curves.
    for geometry_id in dict.fromkeys([reference_run.geometry_id, candidate_run.geometry_id]):
        run_theory_analysis(geometry_id)

    run_observed_comparison(reference_run, candidate_run)
    run_theory_comparison(reference_run, candidate_run)

    print("[analyze_and_compare] finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
