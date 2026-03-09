#!/usr/bin/env python3

"""
train_surrogate.py

Train a LightGBM-based surrogate model mapping calorimeter geometry
(absorber/scintillator thickness, segmentation, etc.)
to neutron performance metrics at fixed muon false-positive rate.

Targets:
  - detection_efficiency
  - eff_lo
  - eff_hi
  - energy_resolution

Usage: 

python ./surrogate/train_surrogate.py \
  --training-csv ./csv_data/training.csv \
  --output-model ./model/lgbm_surrogate.joblib

python ./surrogate/train_surrogate.py \
  --training-csv ./csv_data/training.csv \
  --load-model ./model/lgbm_surrogate.joblib \
  --output-model ./model/lgbm_surrogate_updated.joblib


"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split


# -------------------------
# Feature / target schema
# -------------------------

FEATURE_COLUMNS = [
    "gun_energy_GeV",
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

TARGET_COLUMNS = [
    "detection_efficiency",
    "eff_lo",
    "eff_hi",
    "energy_resolution",
]

#ENERGY_WEIGHTS = {
#    0.9396: 0.258087,
#    0.9496: 0.410015,
#    0.9696: 0.244815,
#    1.0396: 0.0824557,
#    1.9396: 0.00462638,
#}

# -------------------------
# Argument parsing
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LightGBM surrogate model for neutron performance"
    )

    parser.add_argument(
        "--training-csv",
        required=True,
        help="Path to training.csv input file"
    )

    parser.add_argument(
        "--load-model",
        default=None,
        help="Optional path to an existing surrogate model to continue training"
    )

    parser.add_argument(
        "--output-model",
        required=True,
        help="Path to save trained surrogate model (joblib format)"
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


# -------------------------
# Main training logic
# -------------------------

def main():
    args = parse_args()

    # -------------------------
    # Load training data
    # -------------------------
    if not os.path.exists(args.training_csv):
        raise FileNotFoundError(f"Training file not found: {args.training_csv}")

    df = pd.read_csv(args.training_csv)

    # Basic sanity checks
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    missing_targets = [c for c in TARGET_COLUMNS if c not in df.columns]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    # Drop invalid rows if requested by schema
    if "valid_flag" in df.columns:
        df = df[df["valid_flag"] == 1]

    # Extract features / targets
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    # Compute sample weights based on gun energy
    #energies = df["gun_energy_GeV"].astype(float)

    #make sure it exists
    #GUN_POINTS = np.array(sorted(ENERGY_WEIGHTS.keys()), dtype=float)
    #if len(GUN_POINTS) == 0:
    #    raise ValueError("ENERGY_WEIGHTS is empty. Please define energy weights.")

    
    #def snap_energy(e):
    #    return float(GUN_POINTS[np.argmin(np.abs(GUN_POINTS - e))])

    #e_snap = energies.apply(snap_energy)

    #q(E): how many samples you have at each gun energy
    #counts = e_snap.value_counts().to_dict()

    #p(E): realistic spectrum weight at that energy (from histogram-derived dict)
    #p = e_snap.map(lambda e: ENERGY_WEIGHTS[e]).astype(float)

    # importance weights: p(E)/q(E)
    #w = np.array([p_i / counts[e] for p_i, e in zip(p.values, e_snap.values)], dtype=float)
    
    
    # FIXME: Normalize to make mean weight ~1
    #w = w / np.mean(w)


    
    


    # -------------------------
    # Train / validation split (X, y)
    # -------------------------
    X_train, X_val, y_train, y_val = train_test_split( #w_train, w_val
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
        model = joblib.load(args.load_model)
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
    #sample_weight=w_train

    # -------------------------
    # Validation diagnostics
    # -------------------------
    val_pred = model.predict(X_val)
    val_pred = pd.DataFrame(val_pred, columns=TARGET_COLUMNS)

    print("\nValidation summary (mean absolute error):")
    for col in TARGET_COLUMNS:
        mae = np.mean(np.abs(val_pred[col] - y_val[col].values))
        print(f"  {col:20s}: MAE = {mae:.5f}")

    #print("\nValidation summary (weighted mean absolute error):")
    #for col in TARGET_COLUMNS:
    #    err = np.abs(val_pred[col].values - y_val[col].values)
    #    wmae = np.sum(err * w_val) / np.sum(w_val)
    #    print(f" {col:20s}: wMAE = {wmae:.5f}")

    # -------------------------
    # Save trained model
    # -------------------------
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(model, args.output_model)

    print(f"\nTrained surrogate model saved to: {args.output_model}")



if __name__ == "__main__":
    main()

