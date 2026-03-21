#!/usr/bin/env python3

"""
train_surrogate.py

Train a LightGBM-based surrogate model mapping calorimeter geometry
(absorber/scintillator thickness, segmentation, etc.)
to geometry-level multi-particle performance metrics.

Targets:
  - inferred from the geometry-level training CSV
  - metric standard-deviation columns are excluded automatically

Usage: 

python ./surrogate/train_surrogate.py \
  --training-csv ./surrogate/csv_data/training_compact/training_NK_compact_0.csv \
  --output-model ./surrogate/model/lgbm_surrogate_NK_0.joblib

python ./surrogate/train_surrogate.py \
  --training-csv ./surrogate/csv_data/merged/training_NK_compact_0-2.csv \
  --load-model ./surrogate/model/lgbm_surrogate_NK_0.joblib \
  --output-model ./surrogate/model/lgbm_surrogate_NK_0-2.joblib


"""

import argparse
import os
import joblib
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


# -----------------------------------------------------------------------------------------
# Feature / target schema
# -----------------------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "seg1_layers",
    "seg2_layers",
    "seg3_layers",
    "t_absorber_seg1",
    "t_absorber_seg2",
    "t_absorber_seg3",
    "t_scin_seg1",
    "t_scin_seg2",
    "t_scin_seg3",
]

# Since there are multiple targets to predict, we simply tell the model what not to predict.
NON_TARGET_COLUMNS = {
    "geometry_id",
    "nLayers",
    "t_spacer",
    *FEATURE_COLUMNS,
}

# -----------------------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LightGBM surrogate model on geometry-level performance metrics"
    )

    parser.add_argument(
        "--training-csv",
        required=True,
        help="Path to the geometry-level training CSV"
    )

    parser.add_argument(
        "--load-model",
        default=None,
        help="Optional path to an existing surrogate model bundle to continue training"
    )

    parser.add_argument(
        "--output-model",
        required=True,
        help="Path to save the trained surrogate model bundle (joblib format)"
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data held out for validation (default: 0.2)"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    return parser.parse_args()


def infer_target_columns(df: pd.DataFrame) -> list[str]:
    # Use all geometry-level metric columns except the per-metric uncertainty columns.
    target_columns: list[str] = []
    for column_name in df.columns:
        if column_name in NON_TARGET_COLUMNS or column_name.endswith("_std"):
            continue
        target_columns.append(column_name)

    if not target_columns:
        raise ValueError("No geometry-level target columns found in training CSV.")

    return target_columns


# -----------------------------------------------------------------------------------------
# Main training logic
# -----------------------------------------------------------------------------------------

def main():
    args = parse_args()

    # -------------------------
    # Load training data
    # -------------------------
    if not os.path.exists(args.training_csv):
        raise FileNotFoundError(f"Training file not found: {args.training_csv}")

    df = pd.read_csv(args.training_csv)

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    # Drop invalid rows if requested by schema
    if "valid_flag" in df.columns:
        df = df[df["valid_flag"] == 1]

    target_columns = infer_target_columns(df)

    # Extract features / targets
    X = df[FEATURE_COLUMNS]
    y = df[target_columns]

    # -------------------------
    # Train / validation split
    # -------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_fraction,
        random_state=args.random_seed,
    )

    # -------------------------
    # Create or load model
    # -------------------------
    if args.load_model:
        print(f"Loading existing surrogate model: {args.load_model}")
        loaded_payload = joblib.load(args.load_model)
        if isinstance(loaded_payload, dict) and "model" in loaded_payload:
            model = loaded_payload["model"]
            loaded_features = list(loaded_payload.get("feature_columns", []))
            loaded_targets = list(loaded_payload.get("target_columns", []))
            if loaded_features and loaded_features != FEATURE_COLUMNS:
                raise ValueError(
                    "Loaded model feature columns do not match the geometry-level training schema."
                )
            if loaded_targets and loaded_targets != target_columns:
                raise ValueError(
                    "Loaded model target columns do not match the current training CSV."
                )
        else:
            model = loaded_payload
    else:
        print("Creating new LightGBM surrogate model")

        base_regressor = LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.random_seed,
        )

        model = MultiOutputRegressor(base_regressor)

    # -------------------------
    # Train model
    # -------------------------
    print("Training surrogate model")
    model.fit(X_train, y_train)

    # -------------------------
    # Validation diagnostics
    # -------------------------
    val_pred = model.predict(X_val)
    val_pred = pd.DataFrame(val_pred, columns=target_columns)

    print("\nValidation summary (mean absolute percentage error):")
    for col in target_columns:
        mape_normalized = mean_absolute_percentage_error(y_val[col].values, val_pred[col])
        mape_percentage = mape_normalized * 100
        print(f"  {col:20s}: MAPE = {mape_percentage:.2f}")

    # -------------------------
    # Save trained model
    # -------------------------
    output_path = os.path.abspath(args.output_model)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": target_columns,
    }
    joblib.dump(payload, output_path)

    print(f"\nTrained surrogate model saved to: {output_path}")



if __name__ == "__main__":
    main()
