#!/usr/bin/env python3
"""
Extend an existing LHS sweep YAML with additional non-overlapping variants.
Example:
python geometries/extend_lhs.py \
  --input geometries/sweeps/finalized_run.yaml \
  --out geometries/sweeps/extension_1.yaml \
  --n 40 \
  --seed 67
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description="Extend an LHS sweep YAML with new variants.")
    parser.add_argument("--input", required=True, help="Existing sweep YAML path")
    parser.add_argument("--out", required=True, help="Output extension YAML path")
    parser.add_argument("--n", type=int, required=True, help="Number of new variants")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tag-prefix", default="", help="Optional replacement tag prefix")
    parser.add_argument(
        "--index-base",
        type=int,
        default=None,
        help="Optional starting index for new tags. Defaults to one after the highest existing index.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.15,
        help="Minimum normalized Euclidean distance from existing and newly accepted points.",
    )
    return parser.parse_args()


def variant_key(variant: dict[str, object]) -> tuple[float, ...]:
    return tuple(round(float(variant[name]), 4) for name, _, _ in LAYER_THICKNESS_BOUNDS)


def parse_existing_tag_indices(variants: list[dict[str, object]], tag_prefix: str) -> list[int]:
    # Recover the numeric suffixes from matching tags so new tags can continue the sequence.
    indices: list[int] = []
    for variant in variants:
        tag_text = str(variant.get("tag", "")).strip()
        if not tag_text.startswith(tag_prefix):
            continue
        suffix = tag_text[len(tag_prefix):]
        if suffix.isdigit():
            indices.append(int(suffix))
    return indices


def normalize_point(point: tuple[float, ...]) -> np.ndarray:
    # Map one parameter point into the unit box before computing distances.
    normalized_values = []
    for value, (_, low, high) in zip(point, LAYER_THICKNESS_BOUNDS):
        normalized_values.append((value - low) / (high - low))
    return np.array(normalized_values, dtype=float)


def is_too_close(
    candidate_point: tuple[float, ...],
    existing_points: list[np.ndarray],
    min_distance: float,
) -> bool:
    if min_distance <= 0.0 or not existing_points:
        return False
    normalized_candidate = normalize_point(candidate_point)
    distances = np.linalg.norm(np.vstack(existing_points) - normalized_candidate, axis=1)
    return bool(np.any(distances < min_distance))


def main() -> int:
    # Load the existing sweep, generate new non-overlapping points, then write the extension YAML.
    arguments = parse_arguments()

    try:
        from scipy.stats import qmc  # type: ignore
    except Exception as error:
        raise SystemExit("SciPy is required for LHS extension generation") from error

    input_path = resolve_project_path(arguments.input)
    output_path = resolve_project_path(arguments.out)

    if not input_path.exists():
        raise SystemExit(f"Input sweep file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as input_file:
        input_payload = yaml.safe_load(input_file)
    if input_payload is None:
        input_payload = {}
    if not isinstance(input_payload, dict):
        raise SystemExit(f"Input sweep file must contain an object: {input_path}")

    existing_variants = input_payload.get("variants", [])
    if not isinstance(existing_variants, list):
        raise SystemExit(f"Input sweep file has invalid variants list: {input_path}")

    tag_prefix = arguments.tag_prefix or str(input_payload.get("tag_prefix", "lhs"))
    existing_indices = parse_existing_tag_indices(existing_variants, tag_prefix)
    if arguments.index_base is None:
        index_base = max(existing_indices, default=-1) + 1
    else:
        index_base = arguments.index_base

    seen_keys = set()
    normalized_points: list[np.ndarray] = []
    # Seed the duplicate and distance checks with the variants already present in the input sweep.
    for variant in existing_variants:
        try:
            existing_key = variant_key(variant)
        except Exception as error:
            raise SystemExit(f"Existing variant is missing one or more LHS parameters: {variant}") from error
        seen_keys.add(existing_key)
        normalized_points.append(normalize_point(existing_key))

    sampler = qmc.LatinHypercube(d=len(LAYER_THICKNESS_BOUNDS), seed=arguments.seed)
    _, lower_bounds, upper_bounds = zip(*LAYER_THICKNESS_BOUNDS)

    new_variants: list[dict[str, object]] = []
    # Keep sampling until we collect the requested number of new parameter points.
    while len(new_variants) < arguments.n:
        remaining = arguments.n - len(new_variants)
        batch_size = max(remaining * 4, 32)  # Arbitrarily chosen oversampling factor to reduce resampling loops.
        unit_sample = sampler.random(n=batch_size)
        scaled_sample = qmc.scale(unit_sample, lower_bounds, upper_bounds)

        for sample_row in scaled_sample:
            candidate_variant: dict[str, object] = {}
            for (parameter_name, *_), value in zip(LAYER_THICKNESS_BOUNDS, sample_row):
                candidate_variant[parameter_name] = round(float(value), 4)

            candidate_key = variant_key(candidate_variant)
            if candidate_key in seen_keys:
                continue
            if is_too_close(candidate_key, normalized_points, arguments.min_distance):
                continue

            candidate_variant["tag"] = f"{tag_prefix}{index_base + len(new_variants):03d}"
            new_variants.append(candidate_variant)
            seen_keys.add(candidate_key)
            normalized_points.append(normalize_point(candidate_key))

            if len(new_variants) >= arguments.n:
                break

    output_payload = dict(input_payload)
    output_payload["tag_prefix"] = tag_prefix
    output_payload["variants"] = new_variants

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(output_payload, output_file, sort_keys=False)

    print(f"Wrote {output_path} with {len(new_variants)} extension variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
