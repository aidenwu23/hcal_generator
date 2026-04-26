#!/usr/bin/env python3

"""
Build training CSVs from processed results, train the surrogate model, and
propose the next geometry batch for conductor.py.

Example:
python3 orchestrator.py \
  --training-csv surrogate/campaigns/0.5/training_compact_0-1.csv \
  --run-level-csv surrogate/iterations/1-3_GeV/iteration_1/training_raw_0-1.csv \
  --model surrogate/model/1-3_GeV/lgbm_surrogate_0-1.joblib \
  --bo-spec geometries/sweeps/bo_spec.yaml \
  --sweep-yaml geometries/sweeps/proposed/1-3_GeV/proposed_1.yaml \
  --pool 20000 \
  --bo-variants 5 \
  --seed 10 \
  --processed-root data/processed
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from surrogate.scoring import score_row, safe_eval_expr


def run_cmd(cmd: list[str]) -> None:
    cmd = [str(value) for value in cmd]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_processed_root(processed_root: Path | None) -> Path:
    # Require processed_root only when the script needs to rebuild observed CSV artifacts.
    if processed_root is None:
        raise ValueError("--processed-root is required when rebuilding training CSV artifacts.")
    return processed_root


def refresh_geometry_training_csv(
    processed_root: Path,
    run_level_csv: Path,
    geometry_training_csv: Path,
) -> None:
    run_cmd([
        "python3", "surrogate/build_raw_csv.py",
        "--processed-root", str(processed_root),
        "--out", str(run_level_csv),
    ])
    run_cmd([
        "python3", "surrogate/compact_training_csv.py",
        "--in", str(run_level_csv),
        "--out", str(geometry_training_csv),
    ])


def train_surrogate_model(training_csv: Path, model_path: Path) -> None:
    run_cmd([
        "python3", "surrogate/train_surrogate.py",
        "--training-csv", str(training_csv),
        "--output-model", str(model_path),
    ])


def propose_next_geometries(
    model_path: Path,
    spec_path: Path,
    sweep_yaml: Path,
    pool_size: int,
    variant_count: int,
    seed_value: int,
) -> None:
    # Score sampled candidates with the surrogate and emit the next BO sweep YAML.
    run_cmd([
        "python3", "surrogate/propose_bo.py",
        "--model", str(model_path),
        "--spec", str(spec_path),
        "--out", str(sweep_yaml),
        "--pool", str(pool_size),
        "--k", str(variant_count),
        "--seed", str(seed_value),
    ])


def select_best_observed_geometry(
    training_csv: Path,
    objective_expr: str | None,
    scoring: dict[str, object] | None,
    scoring_relative_to: Path,
    output_csv: Path,
) -> None:
    df = pd.read_csv(training_csv)
    if "geometry_id" not in df.columns:
        raise ValueError("Compact training CSV must contain geometry_id.")

    best_geometry_id = None
    best_geometry_score = None
    best_geometry_rows = None
    for geometry_id, geometry_df in df.groupby("geometry_id", sort=False):
        geometry_row = geometry_df.iloc[[0]].copy()
        row_values = geometry_row.iloc[0].to_dict()
        if objective_expr:
            try:
                geometry_score = safe_eval_expr(objective_expr, row_values)
            except Exception as error:
                raise ValueError(
                    f"Failed to evaluate best-objective expression {objective_expr!r} "
                    f"against the geometry training CSV."
                ) from error
        elif scoring is not None:
            geometry_score, extra_values = score_row(row_values, scoring, scoring_relative_to)
            for column_name, column_value in extra_values.items():
                geometry_row[column_name] = column_value
        else:
            raise ValueError("Either objective_expr or scoring must be provided.")
        if best_geometry_score is None or geometry_score > best_geometry_score:
            best_geometry_id = geometry_id
            best_geometry_score = geometry_score
            best_geometry_rows = geometry_row

    if best_geometry_rows is None or best_geometry_id is None or best_geometry_score is None:
        raise ValueError("No observed geometry rows were available for scoring.")

    best_geometry_rows["aggregated_objective"] = best_geometry_score
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    best_geometry_rows.to_csv(output_csv, index=False)
    print("Best geometry configuration saved to:", output_csv)
    print("Best objective expression:", objective_expr)
    print("Best geometry_id:", best_geometry_id)
    print("Aggregated objective:", best_geometry_score)
    print(best_geometry_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the geometry surrogate model and write the next BO sweep."
    )
    parser.add_argument("--processed-root", default=None, help="Path to hcal_optimizer/data/processed.")
    parser.add_argument("--training-csv", required=True, help="Geometry training CSV path.")
    parser.add_argument("--run-level-csv", default=None, help="Run-level observed-results CSV path.")
    parser.add_argument("--model", required=True, help="Path to write the surrogate model bundle.")
    parser.add_argument("--bo-spec", default="geometries/sweeps/bo_spec.yaml", help="Path to bo_spec.yaml.")
    parser.add_argument("--sweep-yaml", default="geometries/sweeps/sweep_bo001.yaml", help="Path to BO sweep YAML.")
    parser.add_argument("--pool", type=int, default=20000, help="Number of random or Sobol candidates to sample.")
    parser.add_argument("--bo-variants", type=int, default=5, help="Number of BO variants to output.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for candidate sampling.")
    parser.add_argument(
        "--best-objective",
        default=None,
        help="Optional objective expression used to select the best observed geometry.",
    )
    parser.add_argument(
        "--best-observed-csv",
        default="csv_data/best_geos/best_observed_geo.csv",
        help="Path to the best-observed geometry summary CSV.",
    )
    return parser.parse_args()


# In repeated runs, this script does the following:
# Rebuild the observed geometry training file from processed results when needed.
# Fit the surrogate using that training file when needed.
# Propose a new batch of geometries.
# ... run conductor.py separately on the proposed geometries ...
# Rebuild the training file again, now including the newly evaluated geometries.
# Repeat.
def main() -> None:
    args = parse_args()

    processed_root = Path(args.processed_root) if args.processed_root else None
    geometry_training_csv = Path(args.training_csv)
    run_level_csv = (
        Path(args.run_level_csv)
        if args.run_level_csv
        else geometry_training_csv.with_name(f"{geometry_training_csv.stem}_runs{geometry_training_csv.suffix}")
    )
    model_path = Path(args.model)
    spec_path = Path(args.bo_spec)
    sweep_yaml = Path(args.sweep_yaml)
    best_observed_csv = Path(args.best_observed_csv)
    with spec_path.open("r", encoding="utf-8") as input_file:
        bo_spec = yaml.safe_load(input_file) or {}
    scoring = bo_spec.get("scoring", {}) or {}

    # Rebuild the training CSV only when it does not already exist.
    if not geometry_training_csv.exists():
        if run_level_csv.exists():
            run_cmd([
                "python3", "surrogate/compact_training_csv.py",
                "--in", str(run_level_csv),
                "--out", str(geometry_training_csv),
            ])
        else:
            refresh_geometry_training_csv(
                ensure_processed_root(processed_root),
                run_level_csv,
                geometry_training_csv,
            )

    if not geometry_training_csv.exists():
        raise FileNotFoundError(f"Geometry training CSV not found: {geometry_training_csv}")

    if not model_path.exists():
        train_surrogate_model(geometry_training_csv, model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Surrogate model bundle not found: {model_path}")

    # Propose a new batch of geometries by emitting one sweep YAML.
    propose_next_geometries(
        model_path=model_path,
        spec_path=spec_path,
        sweep_yaml=sweep_yaml,
        pool_size=args.pool,
        variant_count=args.bo_variants,
        seed_value=args.seed,
    )

    # Find the best observed geometry.
    select_best_observed_geometry(
        training_csv=geometry_training_csv,
        objective_expr=args.best_objective,
        scoring=None if args.best_objective else scoring,
        scoring_relative_to=spec_path.parent,
        output_csv=best_observed_csv,
    )


if __name__ == "__main__":
    main()
