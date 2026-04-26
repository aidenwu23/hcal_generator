#!/usr/bin/env python3
"""
propose_bo.py 

Reads a BO spec (bounds/constraints + sweep_base), samples candidate geometries,
scores them using a trained LightGBM surrogate (point predictions), and writes
a sweep YAML compatible with hcal_optimizer/geometries/sweep_geometries.py.

- Geometry knobs may be split between:
    * bounds        -> optimized/proposed variables
    * fixed_features -> fixed surrogate inputs
    * sweep_base.constants -> fixed geometry written into output YAML
- Surrogate expects features:
    seg1_layers, seg2_layers, seg3_layers,
    t_absorber_seg1/2/3, t_scin_seg1/2/3
  
- Variants written to YAML contain ONLY valid geometry keys (no extra bookkeeping keys),
  so hcal_optimizer/geometries/generate_hcal.py won’t see unknown --set parameters.

Example:
python3 surrogate/propose_bo.py \
  --model surrogate/model/lgbm_surrogate_NK_0.joblib \
  --spec  geometries/sweeps/bo_spec.yaml \
  --out   geometries/sweeps/proposed/proposed_0_test.yaml \
  --pool  20000 \
  --k     5 \
  --seed  0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from surrogate.scoring import safe_eval_expr, score_prediction_dict

# Use Sobol sampling when PyTorch is available, otherwise fall back to random sampling.
_TORCH_OK = False
try:
    import torch  # type: ignore
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

SURROGATE_FEATURES = [
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

SURROGATE_TARGETS = [
    "neutron_efficiency",
]

GEOM_VARS = [
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


# -------------------------------------------------------------------------------------------------
# Sampling helpers
# -------------------------------------------------------------------------------------------------
def sobol_u01(n: int, d: int, seed: int) -> np.ndarray:
    # Draw candidates in the unit hypercube before scaling to geometry bounds.
    if _TORCH_OK:
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return eng.draw(n).cpu().numpy()
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def load_model_bundle(model_path: str) -> Tuple[Any, List[str], List[str]]:
    # Load either the newer model bundle format or a bare legacy model.
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
        raise SystemExit("Loaded model bundle is missing feature_columns.")
    if not target_columns:
        raise SystemExit("Loaded model bundle is missing target_columns.")

    return model, feature_columns, target_columns


def parse_bounds(bounds_spec: Dict[str, Any], var_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    # Convert YAML bounds into arrays aligned with the geometry variable order.
    lows, highs, meta = [], [], []
    for name in var_names:
        if name not in bounds_spec:
            raise SystemExit(f"Missing bounds for '{name}' in bo_spec.yaml")

        spec = bounds_spec[name]
        if isinstance(spec, dict):
            low = float(spec["low"])
            high = float(spec["high"])
            meta.append(spec)
        else:
            low = float(spec[0])
            high = float(spec[1])
            meta.append({"type": "float"})
        if high <= low:
            raise SystemExit(f"Invalid bounds for '{name}': low={low} high={high}")

        lows.append(low)
        highs.append(high)

    return np.array(lows, dtype=float), np.array(highs, dtype=float), meta


def apply_discreteness(X: np.ndarray, var_names: List[str], meta: List[Dict[str, Any]]) -> np.ndarray:
    # Snap integer or stepped variables back onto their allowed grid.
    X2 = X.copy()
    for j, meta_entry in enumerate(meta):
        discrete_type = str(meta_entry.get("type", "float")).lower()
        step = float(meta_entry.get("step", 1.0))
        if discrete_type in ("int", "integer"):
            X2[:, j] = np.round(X2[:, j] / step) * step
            X2[:, j] = np.round(X2[:, j]).astype(int)
        elif discrete_type == "discrete":
            X2[:, j] = np.round(X2[:, j] / step) * step
    return X2


def normalize01(X: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    # Normalize geometry vectors so distance cuts treat each varied knob evenly.
    denom = np.maximum(highs - lows, 1e-12)
    return (X - lows) / denom


def diverse_topk(X: np.ndarray, scores: np.ndarray, k: int, min_dist: float, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    # Select high-scoring candidates while keeping a minimum normalized separation.
    idx_sorted = np.argsort(-scores)
    Xn = normalize01(X, lows, highs)
    chosen: List[int] = []
    for idx in idx_sorted:
        if len(chosen) >= k:
            break
        if min_dist <= 0 or not chosen:
            chosen.append(int(idx))
            continue
        d = np.linalg.norm(Xn[idx] - Xn[chosen], axis=1)
        if np.all(d >= min_dist):
            chosen.append(int(idx))

    # Fill remaining slots by score if the distance cut is too restrictive.
    if len(chosen) < k:
        for idx in idx_sorted:
            ii = int(idx)
            if ii not in chosen:
                chosen.append(ii)
            if len(chosen) >= k:
                break
    return np.array(chosen[:k], dtype=int)


# -------------------------------------------------------------------------------------------------
# Constraint + scoring helpers
# -------------------------------------------------------------------------------------------------
def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    # Load a YAML file that must contain a mapping at the top level.
    with path.open("r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a top-level mapping.")
    return payload


def filter_design_constraints(X: np.ndarray, var_names: List[str], exprs: List[str]) -> np.ndarray:
    # Evaluate geometry-space constraints before running surrogate prediction.
    if not exprs:
        return np.ones((X.shape[0],), dtype=bool)
    ok = np.ones((X.shape[0],), dtype=bool)
    for i in range(X.shape[0]):
        loc = {var_names[j]: float(X[i, j]) for j in range(len(var_names))}
        # Expose layer variables as integers for constraint expressions.
        for k, v in list(loc.items()):
            if "layers" in k:
                loc[k] = int(round(v))
        for e in exprs:
            try:
                if not safe_eval_expr(e, loc):
                    ok[i] = False
                    break
            except Exception:
                ok[i] = False
                break
    return ok


def filter_predicted_constraints(pred: Dict[str, np.ndarray], exprs: List[str]) -> np.ndarray:
    # Evaluate constraints that depend on surrogate-predicted quantities.
    if not exprs:
        return np.ones((len(next(iter(pred.values()))),), dtype=bool)
    ok = np.ones((len(next(iter(pred.values()))),), dtype=bool)
    for i in range(ok.shape[0]):
        loc = {k: float(v[i]) for k, v in pred.items()}
        for e in exprs:
            try:
                if not safe_eval_expr(e, loc):
                    ok[i] = False
                    break
            except Exception:
                ok[i] = False
                break
    return ok


# -------------------------------------------------------------------------------------------------
# YAML emission
# -------------------------------------------------------------------------------------------------
def build_sweep_yaml(sweep_base: Dict[str, Any], X_geom: np.ndarray, geom_var_names: List[str]) -> Dict[str, Any]:
    """
    Output sweep YAML using sweep_base plus generated variants.

    Each variant gets:
      - tag: <tag_prefix><zero-padded index>
      - only proposed geometry keys from geom_var_names

    Fixed geometry values should live in sweep_base.constants.
    """
    out = dict(sweep_base)
    out["variants"] = []

    tag_prefix = str(out.get("tag_prefix", "var"))
    index_base = int(out.get("index_base", 0))

    for i in range(X_geom.shape[0]):
        # Keep only geometry keys so downstream geometry generation sees valid inputs.
        v: Dict[str, Any] = {
            "tag": f"{tag_prefix}{index_base + i:03d}"
        }
        for j, key in enumerate(geom_var_names):
            val = X_geom[i, j]
            if "layers" in key:
                v[key] = int(round(float(val)))
            else:
                v[key] = round(float(val), 4)
        out["variants"].append(v)

    return out


def build_feature_rows(
    X_geom: np.ndarray,
    geom_var_names: List[str],
    feature_columns: List[str],
    fixed_features: Dict[str, Any],
) -> List[Dict[str, float]]:
    # Build one surrogate feature row per geometry candidate.
    feature_rows: List[Dict[str, float]] = []
    for geometry_values in X_geom:
        geometry_feature_values = {
            geom_var_names[j]: float(geometry_values[j]) for j in range(len(geom_var_names))
        }
        feature_row: Dict[str, float] = {}
        for feature_name in feature_columns:
            if feature_name in geometry_feature_values:
                feature_row[feature_name] = geometry_feature_values[feature_name]
            elif feature_name in fixed_features:
                feature_row[feature_name] = float(fixed_features[feature_name])
            else:
                raise SystemExit(
                    f"Cannot build surrogate feature rows because '{feature_name}' is missing."
                )
        feature_rows.append(feature_row)
    return feature_rows


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Propose the next geometry batch and write a sweep YAML.")
    ap.add_argument("--model", required=True, help="Path to trained surrogate .joblib bundle.")
    ap.add_argument("--spec", required=True, help="Path to bo_spec.yaml.")
    ap.add_argument("--out", required=True, help="Output sweep YAML path.")
    ap.add_argument("--pool", type=int, default=20000, help="Number of random/Sobol candidates to sample.")
    ap.add_argument("--k", type=int, default=32, help="Number of variants to output.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load spec.
    spec_path = Path(args.spec)
    output_path = Path(args.out)
    spec = load_yaml_mapping(spec_path)

    sweep_base = spec.get("sweep_base")
    if not isinstance(sweep_base, dict):
        raise SystemExit("bo_spec.yaml must contain a 'sweep_base' mapping.")

    bounds_spec = spec.get("bounds")
    if not isinstance(bounds_spec, dict):
        raise SystemExit("bo_spec.yaml must contain a 'bounds' mapping.")

    # Fixed features supply surrogate inputs that are not varied in this proposal batch.
    fixed_features = spec.get("fixed_features", {}) or {}
    if not isinstance(fixed_features, dict):
        raise SystemExit("'fixed_features' must be a mapping if provided.")
    constraints = spec.get("constraints", {}) or {}
    design_exprs = constraints.get("design", []) or []
    pred_exprs = constraints.get("predicted", []) or []
    if not isinstance(design_exprs, list) or not isinstance(pred_exprs, list):
        raise SystemExit("'constraints.design' and 'constraints.predicted' must be lists of expressions.")

    scoring = spec.get("scoring", {}) or {}
    diversity = spec.get("diversity", {}) or {}
    min_dist = float(diversity.get("min_l2_norm", 0.05))
    model, feature_columns, target_columns = load_model_bundle(args.model)

    # Determine which geometry variables we propose.
    # Allow only a subset of GEOM_VARS to be optimized; the rest may come from fixed_features.
    geom_var_names = [k for k in GEOM_VARS if k in bounds_spec]

    # Reject typos in the bounds block before sampling candidates.
    unknown_bounds = [k for k in bounds_spec.keys() if k not in GEOM_VARS]
    if unknown_bounds:
        raise SystemExit(
            f"Unknown bounds keys not recognized as geometry variables: {unknown_bounds}"
        )

    missing_for_surrogate = [
        k for k in feature_columns
        if k not in geom_var_names and k not in fixed_features
    ]

    if missing_for_surrogate:
        raise SystemExit(
            "Some surrogate features are neither proposed in bounds nor provided in fixed_features: "
            f"{missing_for_surrogate}"
        )
    
    # Parse bounds.
    lows, highs, meta = parse_bounds(bounds_spec, geom_var_names)

    # Sample pool in geometry space.
    u = sobol_u01(args.pool, len(geom_var_names), seed=args.seed)
    X_geom = lows + (highs - lows) * u
    X_geom = apply_discreteness(X_geom, geom_var_names, meta)

    # Apply design constraints first (cheap).
    ok_design = filter_design_constraints(X_geom, geom_var_names, design_exprs)
    X_geom_d = X_geom[ok_design]
    if X_geom_d.shape[0] == 0:
        raise SystemExit("No candidates satisfy design constraints. Loosen constraints or increase bounds/pool.")

    # Convert valid geometry candidates into the feature table expected by the model.
    feature_rows = build_feature_rows(X_geom_d, geom_var_names, feature_columns, fixed_features)
    Xs_df = pd.DataFrame(feature_rows, columns=feature_columns)

    Y = np.asarray(model.predict(Xs_df))

    n_candidates = X_geom_d.shape[0]
    if Y.ndim != 2 or Y.shape[0] != n_candidates or Y.shape[1] != len(target_columns):
        raise SystemExit(f"Unexpected surrogate output shape {Y.shape}; expected (N,{len(target_columns)}).")

    # Flatten model outputs into named arrays for scoring and predicted constraints.
    pred_flat = {name: Y[:, i] for i, name in enumerate(target_columns)}

    ok_pred = filter_predicted_constraints(pred_flat, pred_exprs)
    scores, _ = score_prediction_dict(pred_flat, scoring, spec_path.parent)

    X_final = X_geom_d[ok_pred]
    scores_final = scores[ok_pred]

    if X_final.shape[0] == 0:
        raise SystemExit("No candidates satisfy predicted constraints. Loosen constraints or increase pool.")

    # Choose the best feasible candidates after applying the diversity cut.
    k = min(args.k, X_final.shape[0])

    chosen_idx = diverse_topk(
        X_final, scores_final, k=k, min_dist=min_dist,
        lows=lows, highs=highs
    )
    X_out = X_final[chosen_idx]

    # Write the selected candidates as a sweep file for geometry generation.
    sweep_yaml = build_sweep_yaml(sweep_base, X_out, geom_var_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(sweep_yaml, output_file, sort_keys=False)

    print(f"Wrote {output_path} with {len(sweep_yaml['variants'])} variants "
          f"(sampled={X_geom.shape[0]}, design_ok={X_geom_d.shape[0]}, feasible={X_final.shape[0]}).")


if __name__ == "__main__":
    main()
