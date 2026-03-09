#!/usr/bin/env python3
"""
propose_bo.py 

Reads a BO spec (bounds/constraints + sweep_base), samples candidate geometries,
scores them using a trained LightGBM surrogate (point predictions), and writes
a sweep YAML compatible with hcal_generator/geometries/sweep_geometries.py.

- Geometry knobs may be split between:
    * bounds        -> optimized/proposed variables
    * fixed_features -> fixed surrogate inputs
    * sweep_base.constants -> fixed geometry written into output YAML
- Surrogate expects features:
    gun_energy_GeV,
    seg1_layers, seg2_layers, seg3_layers,
    t_absorber_seg1/2/3, t_scin_seg1/2/3
  
- Variants written to YAML contain ONLY valid geometry keys (no extra bookkeeping keys),
  so hcal_generator/geometries/generate_hcal.py won’t see unknown --set parameters.

Example:
  python3 surrogate/propose_bo.py \
    --model model/lgbm_surrogate.joblib \
    --spec  geometries/sweeps/bo_spec.yaml \
    --out   geometries/sweeps/sweep_bo001.yaml \
    --pool  20000 \
    --k     32 \
    --seed  0
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import yaml

# Check if PyTorch is available for Sobol sampling
_TORCH_OK = False
try:
    import torch  # type: ignore
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


# -----------------------------
# Surrogate schema 
# -----------------------------
SURROGATE_FEATURES = [
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

SURROGATE_TARGETS = [
    "detection_efficiency",
    "eff_lo",
    "eff_hi",
    "energy_resolution",
]

# Geometry knobs we propose 
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


# -----------------------------
# Sampling helpers
# -----------------------------
def sobol_u01(n: int, d: int, seed: int) -> np.ndarray:
    if _TORCH_OK:
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return eng.draw(n).cpu().numpy()
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def parse_bounds(bounds_spec: Dict[str, Any], var_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
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
     X2 = X.copy()
     for j, (name, m) in enumerate(zip(var_names, meta)):
         t = str(m.get("type", "float")).lower()
         step = float(m.get("step", 1.0))
         if t in ("int", "integer"):
             X2[:, j] = np.round(X2[:, j] / step) * step
             X2[:, j] = np.round(X2[:, j]).astype(int)
         elif t == "discrete":
             X2[:, j] = np.round(X2[:, j] / step) * step
         # else float, do nothing
     return X2 


def normalize01(X: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    denom = np.maximum(highs - lows, 1e-12)
    return (X - lows) / denom


def diverse_topk(X: np.ndarray, scores: np.ndarray, k: int, min_dist: float, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
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

    # pad if needed
    if len(chosen) < k:
        for idx in idx_sorted:
            ii = int(idx)
            if ii not in chosen:
                chosen.append(ii)
            if len(chosen) >= k:
                break
    return np.array(chosen[:k], dtype=int)


# -----------------------------
# Constraint + scoring helpers
# -----------------------------
def safe_eval_expr(expr: str, local_vars: Dict[str, Any]) -> bool:
    return bool(eval(expr, {"__builtins__": {}}, local_vars))


def filter_design_constraints(X: np.ndarray, var_names: List[str], exprs: List[str]) -> np.ndarray:
    if not exprs:
        return np.ones((X.shape[0],), dtype=bool)
    ok = np.ones((X.shape[0],), dtype=bool)
    for i in range(X.shape[0]):
        loc = {var_names[j]: float(X[i, j]) for j in range(len(var_names))}
        # also expose ints nicely for layer vars
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


def score_candidates(pred: Dict[str, np.ndarray], scoring: Dict[str, Any]) -> np.ndarray:
    mode = str(scoring.get("mode", "metric")).lower()
    maximize = bool(scoring.get("maximize", True))

    if mode == "tradeoff":
        expr = scoring.get("expr")
        if not expr:
            raise SystemExit("scoring.mode=tradeoff requires scoring.expr")
        scores = np.zeros((len(next(iter(pred.values()))),), dtype=float)
        for i in range(scores.shape[0]):
            loc = {k: float(v[i]) for k, v in pred.items()}
            scores[i] = float(eval(expr, {"__builtins__": {}}, loc))
        if not maximize:
            scores = -scores
        return scores

    metric = str(scoring.get("metric", "eff_lo"))
    if metric not in pred:
        raise SystemExit(f"Unknown scoring.metric '{metric}'. Available: {list(pred.keys())}")
    scores = pred[metric].astype(float).copy()
    if not maximize:
        scores = -scores
    return scores


# -----------------------------
# Mapping geometry -> surrogate features
# -----------------------------
def geom_to_surrogate_features(X_geom: np.ndarray, geom_var_names: List[str], fixed_features: Dict[str, Any]) -> np.ndarray:
    """
    X_geom columns correspond to GEOM_VARS in cm for thickness (seg thicknesses).
    Convert to surrogate feature matrix in SURROGATE_FEATURES order.
    Assumption: t_absorber_seg* and t_scin_seg* are in cm, convert to mm for surrogate.
    """
    n = X_geom.shape[0]
    Xs = np.zeros((n, len(SURROGATE_FEATURES)), dtype=float)
    feat_idx = {f: i for i, f in enumerate(SURROGATE_FEATURES)}
    geom_idx = {g: i for i, g in enumerate(geom_var_names)}

    # fixed features
    for k, v in (fixed_features or {}).items():
        if k not in feat_idx:
            raise SystemExit(f"fixed_features contains unknown surrogate feature '{k}'")
        Xs[:, feat_idx[k]] = float(v)


    # Fill any proposed geometry vars that are also surrogate features
    for g in geom_var_names:
        if g in feat_idx:
            Xs[:, feat_idx[g]] = X_geom[:, geom_idx[g]].astype(float)

    return Xs


# -----------------------------
# YAML emission
# -----------------------------
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


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Propose next geometries (Option A) and write a sweep YAML.")
    ap.add_argument("--model", required=True, help="Path to trained surrogate .joblib (MultiOutputRegressor LGBM).")
    ap.add_argument("--spec", required=True, help="Path to bo_spec.yaml (e.g. ../hcal_generator/geometries/sweeps/bo_spec.yaml).")
    ap.add_argument("--out", required=True, help="Output sweep YAML path (e.g. ../hcal_generator/geometries/sweeps/sweep_bo001.yaml).")
    ap.add_argument("--pool", type=int, default=20000, help="Number of random/Sobol candidates to sample.")
    ap.add_argument("--k", type=int, default=32, help="Number of variants to output.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load spec
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    sweep_base = spec.get("sweep_base")
    if not isinstance(sweep_base, dict):
        raise SystemExit("bo_spec.yaml must contain a 'sweep_base' mapping.")

    bounds_spec = spec.get("bounds")
    if not isinstance(bounds_spec, dict):
        raise SystemExit("bo_spec.yaml must contain a 'bounds' mapping.")

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

    # Determine which geometry variables we propose.
    # Allow only a subset of GEOM_VARS to be optimized; the rest may come from fixed_features.
    geom_var_names = [k for k in GEOM_VARS if k in bounds_spec]

    unknown_bounds = [k for k in bounds_spec.keys() if k not in GEOM_VARS]
    if unknown_bounds:
        raise SystemExit(
            f"Unknown bounds keys not recognized as geometry variables: {unknown_bounds}"
        )

    missing_for_surrogate = [
        k for k in SURROGATE_FEATURES
        if k != "gun_energy_GeV"
        and k not in geom_var_names
        and k not in fixed_features
    ]

    if missing_for_surrogate:
        raise SystemExit(
            "Some surrogate features are neither proposed in bounds nor provided in fixed_features: "
            f"{missing_for_surrogate}"
        )
    
    # Parse bounds
    lows, highs, meta = parse_bounds(bounds_spec, geom_var_names)

    # Sample pool in geometry space
    u = sobol_u01(args.pool, len(geom_var_names), seed=args.seed)
    X_geom = lows + (highs - lows) * u
    X_geom = apply_discreteness(X_geom, geom_var_names, meta)

    # Apply design constraints first (cheap)
    ok_design = filter_design_constraints(X_geom, geom_var_names, design_exprs)
    X_geom_d = X_geom[ok_design]
    if X_geom_d.shape[0] == 0:
        raise SystemExit("No candidates satisfy design constraints. Loosen constraints or increase bounds/pool.")

    # Build surrogate features + predict
    model = joblib.load(args.model)

    Xs = geom_to_surrogate_features(X_geom_d, geom_var_names, fixed_features)

    import pandas as pd
    Xs_df = pd.DataFrame(Xs, columns=SURROGATE_FEATURES)

    Y = np.asarray(model.predict(Xs_df))

    if Y.ndim != 2 or Y.shape[1] != len(SURROGATE_TARGETS):
        raise SystemExit(f"Unexpected surrogate output shape {Y.shape}; expected (N,{len(SURROGATE_TARGETS)}).")

    pred = {name: Y[:, i] for i, name in enumerate(SURROGATE_TARGETS)}

    # Apply predicted constraints
    ok_pred = filter_predicted_constraints(pred, pred_exprs)
    X_final = X_geom_d[ok_pred]
    pred_final = {k: v[ok_pred] for k, v in pred.items()}

    if X_final.shape[0] == 0:
        raise SystemExit("No candidates satisfy predicted constraints. Loosen constraints or increase pool.")

    # Score + choose top-k with diversity
    scores = score_candidates(pred_final, scoring)
    k = min(args.k, X_final.shape[0])

    chosen_idx = diverse_topk(
        X_final, scores, k=k, min_dist=min_dist,
        lows=lows, highs=highs
    )
    X_out = X_final[chosen_idx]

    # Emit sweep YAML
    sweep_yaml = build_sweep_yaml(sweep_base, X_out, geom_var_names)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(sweep_yaml, f, sort_keys=False)

    print(f"Wrote {args.out} with {len(sweep_yaml['variants'])} variants "
          f"(sampled={X_geom.shape[0]}, design_ok={X_geom_d.shape[0]}, feasible={X_final.shape[0]}).")


if __name__ == "__main__":
    main()