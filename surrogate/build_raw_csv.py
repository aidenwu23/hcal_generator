# build_raw_csv.py
# Scans hcal_optimizer/data/processed/<geomID>/<runID> for (meta.json, calibration.json, performance.json),
# joins geometry parameters from hcal_optimizer/geometries/generated/<geomID>/geometry.json,
# and emits a run-level training table (CSV) in the specified directory.
#
# Usage:
#   python3 surrogate/build_raw_csv.py --processed-root data/processed --out surrogate/iterations/1-3_GeV/iteration_2/training_raw_2.csv
#
# Output columns: geometry_id, run_id, gun_particle, nLayers, seg1_layers, seg2_layers, seg3_layers,
#   t_absorber_seg1/2/3, t_scin_seg1/2/3, t_spacer,
#   kinetic_energy_GeV, total_energy_GeV, muon_threshold_GeV,
#   detection_efficiency, energy_resolution
"""
python3 surrogate/build_raw_csv.py \
  --processed-root data/processed \
  --out surrogate/campaigns/data.csv
"""

import argparse, json
from pathlib import Path
import pandas as pd

from simulation.helpers.geometry_index import eval_geometry_length_mm


def _extract(meta_p: Path, calibration_p: Path, perf_p: Path, geometry_root: Path) -> dict:
    # meta.json provides beam config, calibration.json provides threshold, and performance.json provides metrics.
    meta = json.loads(meta_p.read_text())
    calibration = json.loads(calibration_p.read_text())
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
        "gun_particle":         meta.get("gun_particle"),
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
        "kinetic_energy_GeV":   meta.get("kinetic_energy_GeV"),
        "total_energy_GeV":     meta.get("total_energy_GeV"),
        "muon_threshold_GeV":   calibration.get("muon_threshold_GeV"),
        # Performance metrics
        "detection_efficiency": perf.get("detection_efficiency"),
        "energy_resolution":    perf.get("energy_resolution"),
    }


def _pairs(processed_root: Path):
    # Expect: processed/<geomID>/<runID>/meta.json alongside calibration.json and performance.json.
    for meta_path in processed_root.rglob("meta.json"):
        if meta_path.parent.name in {".ipynb_checkpoints", "__pycache__"}:
            continue
        calibration_path = meta_path.with_name("calibration.json")
        perf_path = meta_path.with_name("performance.json")
        if calibration_path.exists() and perf_path.exists():
            yield meta_path, calibration_path, perf_path


def _geometry_thickness_cm(value: object) -> float | None:
    # Keep the training CSV thickness columns in centimeters.
    if value is None:
        return None
    return eval_geometry_length_mm(value) / 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True, help="Path to hcal_optimizer/data/processed")
    ap.add_argument("--geometry-root", default=None, help="Path to hcal_optimizer/geometries/generated (inferred if omitted)")
    ap.add_argument("--out", default="training.csv", help="Output CSV filename")
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    # Default: two levels up from data/processed, then into geometries/generated
    geometry_root = Path(args.geometry_root) if args.geometry_root else processed_root.parent.parent / "geometries" / "generated"

    rows = []
    for mp, cp, pp in _pairs(processed_root):
        try:
            rows.append(_extract(mp, cp, pp, geometry_root))
        except Exception as e:
            print(f"[WARN] {mp.parent}: {e}")

    if not rows:
        raise SystemExit("No (meta.json, calibration.json, performance.json) triples found.")

    df = pd.DataFrame(rows)
    
    thickness_cols = ["t_absorber_seg1", "t_absorber_seg2", "t_absorber_seg3", "t_scin_seg1", "t_scin_seg2", "t_scin_seg3", "t_spacer"]
    for column_name in thickness_cols:
        if column_name in df.columns:
            df[column_name] = df[column_name].map(_geometry_thickness_cm)
    
    preferred = [
        "geometry_id", "run_id", "gun_particle",
        "nLayers",
        "seg1_layers", "seg2_layers", "seg3_layers",
        "t_absorber_seg1", "t_absorber_seg2", "t_absorber_seg3",
        "t_scin_seg1", "t_scin_seg2", "t_scin_seg3",
        "t_spacer",
        "kinetic_energy_GeV", "total_energy_GeV", "muon_threshold_GeV",
        "detection_efficiency",
        "energy_resolution",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    out_p = Path(args.out)
    df.to_csv(out_p, index=False)
    print(f"Wrote {len(df)} rows -> {out_p.resolve()}")


if __name__ == "__main__":
    main()
