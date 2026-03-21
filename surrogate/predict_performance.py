#!/usr/bin/env python3
"""
Predict performance metrics for the geometries listed in a sweep YAML.
Example:
python3 surrogate/predict_performance.py \
  --model surrogate/model/lgbm_surrogate_NK_0-1.joblib \
  --in-yaml geometries/sweeps/proposed/proposed_1.yaml \
  --out surrogate/csv_data/predictions/proposed_1_predictions.csv \
  --objective-expr "neutron_efficiency + kaon0L_efficiency"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from propose_bo import GEOM_VARS, SURROGATE_FEATURES, SURROGATE_TARGETS, geom_to_surrogate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict performance metrics for geometries defined in a sweep YAML."
    )
    parser.add_argument("--model", required=True, help="Path to the trained surrogate model bundle.")
    parser.add_argument("--in-yaml", required=True, help="Path to the input sweep YAML.")
    parser.add_argument("--out", required=True, help="Path to the output prediction CSV.")
    parser.add_argument(
        "--objective-expr",
        default=None,
        help="Optional objective expression evaluated from the predicted targets.",
    )
    return parser.parse_args()


def load_model_bundle(model_path: Path) -> tuple[Any, list[str], list[str]]:
    loaded_payload = joblib.load(model_path)
    if isinstance(loaded_payload, dict) and "model" in loaded_payload:
        model = loaded_payload["model"]
        feature_columns = list(loaded_payload.get("feature_columns", []))
        target_columns = list(loaded_payload.get("target_columns", []))
    else:
        model = loaded_payload
        feature_columns = SURROGATE_FEATURES
        target_columns = SURROGATE_TARGETS

    if not feature_columns:
        raise ValueError("Loaded model bundle is missing feature_columns.")
    if not target_columns:
        raise ValueError("Loaded model bundle is missing target_columns.")

    return model, feature_columns, target_columns


def load_yaml_object(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as input_file:
        loaded_object = yaml.safe_load(input_file)

    if loaded_object is None:
        return {}
    if not isinstance(loaded_object, dict):
        raise ValueError(f"Sweep YAML must contain a top-level mapping: {yaml_path}")
    return loaded_object


def build_geometry_rows(specification: dict[str, Any]) -> list[dict[str, Any]]:
    constants = specification.get("constants", {}) or {}
    if not isinstance(constants, dict):
        raise ValueError("Sweep YAML constants must be a mapping.")

    variants = specification.get("variants", []) or []
    if not isinstance(variants, list):
        raise ValueError("Sweep YAML variants must be a list.")

    geometry_rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError("Each sweep YAML variant must be a mapping.")

        merged_row = dict(constants)
        merged_row.update(variant)
        merged_row.setdefault("tag", f"variant{variant_index:03d}")

        missing_columns = [name for name in GEOM_VARS if name not in merged_row]
        if missing_columns:
            raise ValueError(f"Variant is missing geometry columns required by the surrogate: {missing_columns}")

        geometry_rows.append(merged_row)

    if not geometry_rows:
        raise ValueError("Sweep YAML does not contain any variants.")

    return geometry_rows


def safe_eval_expr(expr: str, local_vars: dict[str, object]) -> float:
    return float(eval(expr, {"__builtins__": {}}, local_vars))


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    input_yaml_path = Path(args.in_yaml)
    output_csv_path = Path(args.out)

    model, feature_columns, target_columns = load_model_bundle(model_path)
    specification = load_yaml_object(input_yaml_path)
    geometry_rows = build_geometry_rows(specification)

    X_geom = np.array(
        [[float(row[column_name]) for column_name in GEOM_VARS] for row in geometry_rows],
        dtype=float,
    )
    Xs = geom_to_surrogate_features(
        X_geom=X_geom,
        geom_var_names=GEOM_VARS,
        feature_columns=feature_columns,
        fixed_features={},
    )

    Xs_df = pd.DataFrame(Xs, columns=feature_columns)
    Y = np.asarray(model.predict(Xs_df))
    if Y.ndim != 2 or Y.shape[1] != len(target_columns):
        raise ValueError(
            f"Unexpected surrogate output shape {Y.shape}; expected (N, {len(target_columns)})."
        )

    output_rows: list[dict[str, object]] = []
    for row_index, geometry_row in enumerate(geometry_rows):
        output_row: dict[str, object] = {
            "tag": geometry_row["tag"],
        }
        for column_name in GEOM_VARS:
            if "layers" in column_name:
                output_row[column_name] = int(float(geometry_row[column_name]))
            else:
                output_row[column_name] = float(geometry_row[column_name])
        for target_index, target_name in enumerate(target_columns):
            output_row[target_name] = float(Y[row_index, target_index])
        if args.objective_expr:
            output_row["predicted_objective"] = safe_eval_expr(args.objective_expr, output_row)
        output_rows.append(output_row)

    fieldnames = ["tag", *GEOM_VARS, *target_columns]
    if args.objective_expr:
        fieldnames.append("predicted_objective")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Wrote {len(output_rows)} prediction rows -> {output_csv_path}")


if __name__ == "__main__":
    main()
