#!/usr/bin/env python3

"""
Build training CSVs from processed results, train the surrogate model, and
propose the next geometry batch for conductor.py.

Example:
python3 orchestrator.py \
  --training-csv surrogate/csv_data/merged/training_NK_compact_0-1.csv \
  --run-level-csv surrogate/csv_data/merged/training_NK_raw_0-1.csv \
  --model surrogate/model/lgbm_surrogate_NK_0-1.joblib \
  --bo-spec geometries/sweeps/bo_spec.yaml \
  --sweep-yaml geometries/sweeps/proposed/validation_test_0-1.yaml \
  --pool 20000 \
  --bo-variants 5 \
  --seed 10 \
  --processed-root data/processed \
  --overwrite

Don't use overwrite if you already have the training CSV you want to reuse.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def run_cmd(cmd: list[str]) -> None:
    cmd = [str(value) for value in cmd]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_eval_expr(expr: str, local_vars: dict[str, object]) -> float:
    return float(eval(expr, {"__builtins__": {}}, local_vars))


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


def select_best_observed_geometry(training_csv: Path, objective_expr: str, output_csv: Path) -> None:
    df = pd.read_csv(training_csv)
    scores = []
    # Loop through rows and find the one with the highest objective_expr value.
    for _, row in df.iterrows():
        row_values = row.to_dict()
        try:
            scores.append(safe_eval_expr(objective_expr, row_values))
        except Exception as error:
            raise ValueError(
                f"Failed to evaluate best-objective expression {objective_expr!r} "
                f"against the geometry-level training CSV."
            ) from error

    max_row = df.loc[pd.Series(scores).idxmax()]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    max_row.to_frame().T.to_csv(output_csv, index=False)
    print("Best geometry configuration saved to:", output_csv)
    print("Best objective expression:", objective_expr)
    print(max_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the geometry-level surrogate model and write the next BO sweep."
    )
    parser.add_argument("--processed-root", default=None, help="Path to hcal_optimizer/data/processed.")
    parser.add_argument("--training-csv", required=True, help="Geometry-level training CSV path.")
    parser.add_argument("--run-level-csv", default=None, help="Run-level observed-results CSV path.")
    parser.add_argument("--model", required=True, help="Path to write the surrogate model bundle.")
    parser.add_argument("--bo-spec", default="geometries/sweeps/bo_spec.yaml", help="Path to bo_spec.yaml.")
    parser.add_argument("--sweep-yaml", default="geometries/sweeps/sweep_bo001.yaml", help="Path to BO sweep YAML.")
    parser.add_argument("--pool", type=int, default=20000, help="Number of random or Sobol candidates to sample.")
    parser.add_argument("--bo-variants", type=int, default=5, help="Number of BO variants to output.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for candidate sampling.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild training CSV artifacts from processed results and retrain the model even if outputs already exist.",
    )
    parser.add_argument(
        "--best-objective",
        default="neutron_efficiency + kaon0L_efficiency",
        help="Geometry-level objective expression used to select the best observed geometry.",
    )
    parser.add_argument(
        "--best-observed-csv",
        default="surrogate/csv_data/merged/best_geometry_configuration.csv",
        help="Path to the best-observed geometry summary CSV.",
    )
    return parser.parse_args()


# In repeated runs, this script does the following:
# Rebuild the observed geometry-level training file from processed results when needed.
# Fit the surrogate using that training file when needed.
# Propose a new batch of geometries.
# ... run conductor.py separately on the proposed geometries ...
# Rebuild the training file again, now including the newly evaluated geometries.
# Repeat.
def main() -> None:
    args = parse_args()

    processed_root = Path(args.processed_root) if args.processed_root else None
    geometry_training_csv = Path(args.training_csv)
    if args.run_level_csv:
        run_level_csv = Path(args.run_level_csv)
    else:
        run_level_csv = geometry_training_csv.with_name(
            f"{geometry_training_csv.stem}_runs{geometry_training_csv.suffix}"
        )
    model_path = Path(args.model)
    spec_path = Path(args.bo_spec)
    sweep_yaml = Path(args.sweep_yaml)
    best_observed_csv = Path(args.best_observed_csv)

    # Rebuild the training.csv and retrain the model if overwrite.
    if args.overwrite or not geometry_training_csv.exists():
        if not args.overwrite and run_level_csv.exists():
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
        raise FileNotFoundError(f"Geometry-level training CSV not found: {geometry_training_csv}")

    if args.overwrite or not model_path.exists():
        train_surrogate_model(geometry_training_csv, model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Surrogate model bundle not found: {model_path}")

    # Propose a new batch of geometries by emitting yamls.
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
        output_csv=best_observed_csv,
    )


if __name__ == "__main__":
    main()
