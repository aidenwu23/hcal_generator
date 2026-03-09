# aggregator.py
# Scans hcal_generator/data/processed/<geomID>/<runID> for (meta.json, performance.json),
# joins geometry parameters from hcal_generator/geometries/generated/<geomID>/geometry.json,
# and emits a run-level training table (CSV) in the specified directory.
#
# Usage:
#   python3 surrogate/aggregator.py --processed-root data/processed --out csv_data/training.csv
#
# Output columns: geometry_id, run_id, nLayers, seg1_layers, seg2_layers, seg3_layers,
#   t_absorber_seg1/2/3, t_scin_seg1/2/3, t_spacer,
#   gun_energy_GeV, detect_threshold_GeV,
#   detection_efficiency, eff_lo, eff_hi, energy_resolution

import argparse, json
from pathlib import Path
import pandas as pd


def _extract(meta_p: Path, perf_p: Path, geometry_root: Path) -> dict:
    # meta.json provides beam and threshold config; performance.json provides computed metrics
    meta = json.loads(meta_p.read_text())
    perf = json.loads(perf_p.read_text())

    geometry_id = meta.get("geometry_id") or perf.get("geometry_id")

    # Load per-geometry parameters; warn and leave fields None if file is missing
    geom_params = {}
    geom_json_path = geometry_root / geometry_id / "geometry.json"
    if geom_json_path.exists():
        geom_params = json.loads(geom_json_path.read_text())
    else:
        print(f"[WARN] geometry.json not found for {geometry_id}: {geom_json_path}")

    return {
        "geometry_id":          geometry_id,
        "run_id":               meta_p.parent.name,
        # Geometry features
        "nLayers":              geom_params.get("nLayers"),
        "seg1_layers":          geom_params.get("seg1_layers"),
        "seg2_layers":          geom_params.get("seg2_layers"),
        "seg3_layers":          geom_params.get("seg3_layers"),
        "t_absorber_seg1":      geom_params.get("t_absorber_seg1"),
        "t_absorber_seg2":      geom_params.get("t_absorber_seg2"),
        "t_absorber_seg3":      geom_params.get("t_absorber_seg3"),
        "t_scin_seg1":          geom_params.get("t_scin_seg1"),
        "t_scin_seg2":          geom_params.get("t_scin_seg2"),
        "t_scin_seg3":          geom_params.get("t_scin_seg3"),
        "t_spacer":             geom_params.get("t_spacer"),
        # Beam and threshold config
        "gun_energy_GeV":       meta.get("gun_energy_GeV"),
        "detect_threshold_GeV": meta.get("detect_threshold_GeV"),
        # Performance metrics
        "detection_efficiency": perf.get("detection_efficiency"),
        "eff_lo":               perf.get("eff_lo"),
        "eff_hi":               perf.get("eff_hi"),
        "energy_resolution":    perf.get("energy_resolution"),
    }


def _pairs(processed_root: Path):
    # Expect: processed/<geomID>/<runID>/meta.json alongside performance.json
    for meta_path in processed_root.rglob("meta.json"):
        if meta_path.parent.name in {".ipynb_checkpoints", "__pycache__"}:
            continue
        perf_path = meta_path.with_name("performance.json")
        if perf_path.exists():
            yield meta_path, perf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True, help="Path to hcal_generator/data/processed")
    ap.add_argument("--geometry-root", default=None, help="Path to hcal_generator/geometries/generated (inferred if omitted)")
    ap.add_argument("--out", default="training.csv", help="Output CSV filename")
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    # Default: two levels up from data/processed, then into geometries/generated
    geometry_root = Path(args.geometry_root) if args.geometry_root else processed_root.parent.parent / "geometries" / "generated"

    rows = []
    for mp, pp in _pairs(processed_root):
        try:
            rows.append(_extract(mp, pp, geometry_root))
        except Exception as e:
            print(f"[WARN] {mp.parent}: {e}")

    if not rows:
        raise SystemExit("No (meta.json, performance.json) pairs found.")

    df = pd.DataFrame(rows)
    
    thickness_cols = ["t_absorber_seg1", "t_absorber_seg2", "t_absorber_seg3", "t_scin_seg1", "t_scin_seg2", "t_scin_seg3", "t_spacer"]
    df[thickness_cols] = df[thickness_cols].apply(lambda s: pd.to_numeric(s.astype(str).str.replace("*cm", "", regex=False).str.strip(), errors="coerce"))
    
    preferred = [
        "geometry_id", "run_id",
        "nLayers",
        "seg1_layers", "seg2_layers", "seg3_layers",
        "t_absorber_seg1", "t_absorber_seg2", "t_absorber_seg3",
        "t_scin_seg1", "t_scin_seg2", "t_scin_seg3",
        "t_spacer",
        "gun_energy_GeV", "detect_threshold_GeV",
        "detection_efficiency", "eff_lo", "eff_hi",
        "energy_resolution",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    out_p = Path(args.out)
    df.to_csv(out_p, index=False)
    print(f"Wrote {len(df)} rows -> {out_p.resolve()}")


if __name__ == "__main__":
    main()
