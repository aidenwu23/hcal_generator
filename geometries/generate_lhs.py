#!/usr/bin/env python3
"""Generate a sweep YAML file using Latin Hypercube sampling.
python3 geometries/generate_lhs.py \
  --template geometries/sweeps/template_sweep.yaml \
  --out geometries/sweeps/runs/75_geos.yaml \
  --n 75

"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from geometry_utils import resolve_project_path

LAYER_THICKNESS_BOUNDS = [
    ("t_absorber_seg1", 3.5, 4.5),
    ("t_scin_seg1", 0.3, 0.6),
    ("t_absorber_seg2", 3.5, 4.5),
    ("t_scin_seg2", 0.3, 0.6),
    ("t_absorber_seg3", 3.5, 4.5),
    ("t_scin_seg3", 0.3, 0.6),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LHS sweep YAML")
    parser.add_argument("--template", required=True, help="Template sweep YAML path")
    parser.add_argument("--out", required=True, help="Output sweep YAML path")
    parser.add_argument("--n", type=int, required=True, help="Number of variants")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tag-prefix", default="lhs", help="Variant tag prefix")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    # Try importing scipy.
    try:
        from scipy.stats import qmc  # type: ignore
    except Exception as error:
        raise SystemExit("SciPy is required for Latin Hypercube generation") from error

    # Resolve the template and output paths.
    template_path = resolve_project_path(arguments.template)
    output_path = resolve_project_path(arguments.out)

    if not template_path.exists():
        raise SystemExit(f"Template sweep file not found: {template_path}")

    # Load the template YAML and check that it contains a mapping payload.
    with template_path.open("r", encoding="utf-8") as template_file:
        template_payload = yaml.safe_load(template_file)
    if template_payload is None:
        template_payload = {}
    if not isinstance(template_payload, dict):
        raise SystemExit(f"Template sweep file must contain an object: {template_path}")

    # Generate one Latin Hypercube point per requested variant, then scale it to the
    # configured HCAL thickness bounds.
    sampler = qmc.LatinHypercube(d=len(LAYER_THICKNESS_BOUNDS), seed=arguments.seed)
    unit_sample = sampler.random(n=arguments.n)

    _, lower_bounds, upper_bounds = zip(*LAYER_THICKNESS_BOUNDS)
    scaled_sample = qmc.scale(unit_sample, lower_bounds, upper_bounds)

    variants: list[dict[str, object]] = []
    # Build the YAML variant records from the scaled sample rows.
    for sample_index, sample_row in enumerate(scaled_sample):
        variant: dict[str, object] = {"tag": f"{arguments.tag_prefix}{sample_index:03d}"}
        # Write the sweep thickness values as bare centimeter numbers.
        for (parameter_name, *_), value in zip(LAYER_THICKNESS_BOUNDS, sample_row):
            variant[parameter_name] = round(float(value), 4)
        variants.append(variant)

    output_payload = dict(template_payload)
    output_payload["variants"] = variants

    # Write the output.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(output_payload, output_file, sort_keys=False)

    print(f"Wrote {output_path} with {len(variants)} variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
