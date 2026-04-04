#!/usr/bin/env python3
"""
Run k-fold validation on a geometry-and-energy compact training CSV.
Example:
python3 surrogate/k_fold_validation.py \
  -i surrogate/iterations/1_GeV/iteration_0/training_compact_0.csv \
  -o surrogate/iterations/1_GeV/predictions/k_fold_summary_0.csv \
  --predictions-out surrogate/iterations/1_GeV/predictions/k_fold_predictions_0.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from train_surrogate import FEATURE_COLUMNS, infer_target_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run k-fold validation on a compact geometry-and-energy training CSV."
    )
    parser.add_argument("-i", "--in", dest="input_csv", required=True, help="Input compact training CSV.")
    parser.add_argument("-o", "--out", dest="output_csv", required=True, help="Output summary CSV.")
    parser.add_argument(
        "--predictions-out",
        default=None,
        help="Optional CSV path for held-out predictions from every fold.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of folds. Default: 5.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold shuffling. Default: 42.")
    return parser.parse_args()


def build_model(random_seed: int) -> MultiOutputRegressor:
    base_regressor = LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
    )
    return MultiOutputRegressor(base_regressor)


def build_objective_series(df: pd.DataFrame) -> pd.Series:
    # Use the same combined objective used in the BO loop for one geometry-and-energy row.
    required_columns = ["neutron_efficiency", "kaon0L_efficiency"]
    if any(column_name not in df.columns for column_name in required_columns):
        raise ValueError("Objective MAE requires neutron_efficiency and kaon0L_efficiency columns.")
    return df["neutron_efficiency"] + df["kaon0L_efficiency"]


def summarize_fold_errors(
    fold_index: int,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> dict[str, object]:
    # Compute one summary row for this held-out fold.
    summary_row: dict[str, object] = {
        "fold": fold_index,
        "n_test": len(y_true),
    }

    for column_name in y_true.columns:
        summary_row[f"{column_name}_mae"] = mean_absolute_error(y_true[column_name], y_pred[column_name])

    if "neutron_efficiency" in y_true.columns and "kaon0L_efficiency" in y_true.columns:
        true_objective = build_objective_series(y_true)
        pred_objective = build_objective_series(y_pred)
        summary_row["objective_mae"] = mean_absolute_error(true_objective, pred_objective)

    return summary_row


def append_prediction_rows(
    prediction_rows: list[dict[str, object]],
    fold_index: int,
    test_df: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> None:
    # Save the held-out predictions so each geometry-and-energy row can be inspected later.
    for row_index, (_, actual_row) in enumerate(test_df.iterrows()):
        prediction_row: dict[str, object] = {
            "fold": fold_index,
        }
        if "geometry_id" in actual_row:
            prediction_row["geometry_id"] = actual_row["geometry_id"]
        for column_name in FEATURE_COLUMNS:
            prediction_row[column_name] = actual_row[column_name]
        for column_name in y_pred.columns:
            prediction_row[f"actual_{column_name}"] = actual_row[column_name]
            prediction_row[f"predicted_{column_name}"] = y_pred.iloc[row_index][column_name]
        if "neutron_efficiency" in y_pred.columns and "kaon0L_efficiency" in y_pred.columns:
            prediction_row["actual_objective"] = (
                actual_row["neutron_efficiency"] + actual_row["kaon0L_efficiency"]
            )
            prediction_row["predicted_objective"] = (
                y_pred.iloc[row_index]["neutron_efficiency"] + y_pred.iloc[row_index]["kaon0L_efficiency"]
            )
        prediction_rows.append(prediction_row)


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    # Split the compact training CSV into folds, train on each subset, and score the held-out rows.
    args = parse_args()

    input_csv_path = Path(args.input_csv)
    output_csv_path = Path(args.output_csv)
    predictions_output_path = Path(args.predictions_out) if args.predictions_out else None

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Compact training CSV not found: {input_csv_path}")

    full_df = pd.read_csv(input_csv_path)
    if "valid_flag" in full_df.columns:
        full_df = full_df[full_df["valid_flag"] == 1].copy()

    if len(full_df) < args.k:
        raise ValueError(f"Need at least {args.k} rows for {args.k}-fold validation.")

    missing_features = [column_name for column_name in FEATURE_COLUMNS if column_name not in full_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    target_columns = infer_target_columns(full_df)

    feature_df = full_df[FEATURE_COLUMNS].copy()
    target_df = full_df[target_columns].copy()

    splitter = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    # Train a fresh surrogate for each fold and evaluate it on the held-out rows.
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(feature_df), start=1):
        X_train = feature_df.iloc[train_indices]
        X_test = feature_df.iloc[test_indices]
        y_train = target_df.iloc[train_indices]
        y_test = target_df.iloc[test_indices]

        model = build_model(args.seed)
        model.fit(X_train, y_train)

        y_pred = pd.DataFrame(model.predict(X_test), columns=target_columns, index=y_test.index)

        summary_rows.append(summarize_fold_errors(fold_index, y_test, y_pred))
        if predictions_output_path is not None:
            append_prediction_rows(prediction_rows, fold_index, full_df.iloc[test_indices], y_pred)

    # Add mean and std rows so the summary file is self-contained.
    metric_columns = [column_name for column_name in summary_rows[0].keys() if column_name not in {"fold", "n_test"}]
    mean_row: dict[str, object] = {"fold": "mean", "n_test": len(full_df) / args.k}
    std_row: dict[str, object] = {"fold": "std", "n_test": 0}
    for column_name in metric_columns:
        values = [float(row[column_name]) for row in summary_rows]
        mean_row[column_name] = pd.Series(values).mean()
        std_row[column_name] = pd.Series(values).std(ddof=1)
    summary_rows.extend([mean_row, std_row])

    write_csv(output_csv_path, summary_rows)
    print(f"Wrote {len(summary_rows)} summary rows -> {output_csv_path}")

    if predictions_output_path is not None:
        write_csv(predictions_output_path, prediction_rows)
        print(f"Wrote {len(prediction_rows)} held-out prediction rows -> {predictions_output_path}")


if __name__ == "__main__":
    main()
