"""
Build one run-level CSV row per processed geometry run.

python3 surrogate/build_raw_csv.py \
  --processed-root data/processed \
  --out csv_data/test.csv
"""

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from simulation.helpers.geometry_index import eval_geometry_length_mm


THICKNESS_COLUMNS = [
    "t_absorber_seg1",
    "t_absorber_seg2",
    "t_absorber_seg3",
    "t_scin_seg1",
    "t_scin_seg2",
    "t_scin_seg3",
    "t_spacer",
]

PREFERRED_COLUMNS = [
    "geometry_id",
    "run_id",
    "gun_particle",
    "beam_mode",
    "beam_label",
    "momentum_GeV",
    "spectrum_id",
    "spectrum_x_axis",
    "spectrum_x_min_GeV",
    "spectrum_x_max_GeV",
    "nLayers",
    "seg1_layers",
    "seg2_layers",
    "seg3_layers",
    *THICKNESS_COLUMNS,
    "detection_efficiency",
    "tiles_mean",
    "layers_mean",
]


def _extract(meta_p: Path, calibration_p: Path, perf_p: Path, geometry_root: Path) -> dict:
    # meta.json provides beam config, calibration.json provides threshold, and performance.json provides metrics.
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    json.loads(calibration_p.read_text(encoding="utf-8"))
    perf = json.loads(perf_p.read_text(encoding="utf-8"))

    geometry_id = meta.get("geometry_id") or perf.get("geometry_id")

    # Load per-geometry parameters; warn and leave fields None if file is missing
    geom_params = {}
    geom_json_path = geometry_root / geometry_id / "geometry.json"
    if geom_json_path.exists():
        geom_params = json.loads(geom_json_path.read_text(encoding="utf-8"))
    else:
        print(f"[WARN] geometry.json not found for {geometry_id}: {geom_json_path}")

    return {
        "geometry_id":          geometry_id,
        "run_id":               meta_p.parent.name,
        "gun_particle":         meta.get("gun_particle"),
        "beam_mode":            meta.get("beam_mode"),
        "beam_label":           meta.get("beam_label"),
        "momentum_GeV":         meta.get("momentum_GeV"),
        "spectrum_id":          meta.get("spectrum_id"),
        "spectrum_x_axis":      meta.get("spectrum_x_axis"),
        "spectrum_x_min_GeV":   meta.get("spectrum_x_min_GeV"),
        "spectrum_x_max_GeV":   meta.get("spectrum_x_max_GeV"),
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
        # Performance metrics
        "detection_efficiency": perf.get("detection_efficiency"),
        "tiles_mean":           perf.get("tiles_mean"),
        "layers_mean":          perf.get("layers_mean"),
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

    for row in rows:
        for column_name in THICKNESS_COLUMNS:
            if column_name in row:
                row[column_name] = _geometry_thickness_cm(row[column_name])

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [column_name for column_name in PREFERRED_COLUMNS if any(column_name in row for row in rows)]
    for row in rows:
        for column_name in row:
            if column_name not in fieldnames:
                fieldnames.append(column_name)

    with out_p.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})
    print(f"Wrote {len(rows)} rows -> {out_p.resolve()}")


if __name__ == "__main__":
    main()
