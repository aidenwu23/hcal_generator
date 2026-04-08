#!/usr/bin/env python3
"""Geometry discovery and geometry-derived metadata."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
GEOMETRY_DIRECTORY = PROJECT_DIRECTORY / "geometries"

UNIT_MM: Dict[str, float] = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "um": 0.001,
    "nm": 1e-6,
}

# Convert the DD4hep-style strings used in the geometry files into millimeters so the rest
# of the geometry helpers can work with one consistent depth unit.
def eval_length_mm(value: object) -> float:
    """Evaluate a required DD4hep-style length expression in millimeter."""
    if value is None:
        raise ValueError("Required geometry length is missing.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        expression = value.strip()
        if not expression:
            raise ValueError("Required geometry length is empty.")
        try:
            return float(expression)
        except ValueError:
            safe_locals = {**UNIT_MM, "pi": math.pi}
            try:
                return float(eval(expression, {"__builtins__": {}}, safe_locals))
            except Exception as error:
                raise ValueError(f"Invalid geometry length expression: {value!r}") from error
    raise ValueError(f"Unsupported geometry length value: {value!r}")


# Treat bare numeric geometry metadata values as centimeters and explicit expressions as DD4hep lengths.
def eval_geometry_length_mm(value: object) -> float:
    """Evaluate a geometry metadata length in millimeter."""
    if value is None:
        raise ValueError("Required geometry length is missing.")
    if isinstance(value, (int, float)):
        return float(value) * 10.0
    if isinstance(value, str):
        expression = value.strip()
        if not expression:
            raise ValueError("Required geometry length is empty.")
        try:
            return float(expression) * 10.0
        except ValueError:
            return eval_length_mm(expression)
    raise ValueError(f"Unsupported geometry length value: {value!r}")

# One generated geometry plus the paths and parameter payload that describe it.
@dataclass
class GeometryVariant:
    geometry_id: str
    tag: str
    geometry_directory: Path
    params_path: Path
    xml_path: Path
    spec_path: Path
    params: Dict[str, object]

    @property
    def n_layers(self) -> int:
        return int(self.params.get("nLayers", 0))

    @property
    def side(self) -> str:
        return str(self.params.get("side", "-z"))


# One physical HCAL layer after the three longitudinal segments have been expanded.
@dataclass
class GeometryLayerRow:
    layer_index: int
    absorber_thickness_mm: float
    scintillator_thickness_mm: float
    spacer_thickness_mm: float
    layer_total_thickness_mm: float
    depth_front_mm: float
    depth_mid_mm: float
    depth_back_mm: float
    absorber_depth_mid_mm: float
    scintillator_depth_mid_mm: float


# Compact depth numbers derived from the expanded layer stack.
@dataclass
class GeometryLayerSummary:
    total_depth_mm: float
    total_absorber_thickness_mm: float
    absorber_fraction_by_depth: float


# The repeated layer recipe for one longitudinal HCAL segment.
@dataclass
class GeometrySegmentRecipe:
    segment_index: int
    layer_count: int
    absorber_thickness_mm: float
    scintillator_thickness_mm: float
    spacer_thickness_mm: float


# Ask the sweep script what geometries a set of sweep specs would produce without building
# any further metadata by hand in this helper.
def inspect_geometry_rows(python_bin: str, spec_paths: Iterable[Path]) -> List[Dict[str, object]]:
    """Ask the sweep script for the geometry rows implied by the requested specs."""
    sweep_script = GEOMETRY_DIRECTORY / "sweep_geometries.py"
    if not sweep_script.exists():
        raise FileNotFoundError(f"{sweep_script} not found.")
    command = [python_bin, str(sweep_script), "--dry-run"]
    # Pass every requested spec to the sweep helper so the geometry list matches conductor.
    for spec_path in spec_paths:
        command.extend(["--spec", str(spec_path)])
    result = subprocess.run(
        command,
        cwd=PROJECT_DIRECTORY,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        geometry_rows = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Failed to parse sweep_geometries.py dry-run output") from error
    if not isinstance(geometry_rows, list):
        raise RuntimeError("Expected sweep_geometries.py dry-run to return a JSON list")
    return geometry_rows


# Turn the dry-run sweep rows into strongly typed geometry records with validated paths.
def load_geometry_variants(
    geometry_rows: Iterable[Dict[str, object]],
    *,
    require_geometry_files: bool,
) -> List[GeometryVariant]:
    """Compile geometry metadata from dry-run sweep rows."""
    geometry_variants: List[GeometryVariant] = []
    for geometry_row in geometry_rows:
        # Resolve the geometry file locations first so the later consistency checks compare
        # stable absolute paths rather than mixed relative path strings.
        geometry_directory = _resolve_project_path(str(geometry_row["geometry_directory"]))
        params_path = _resolve_project_path(str(geometry_row["json_path"]))
        xml_path = _resolve_project_path(str(geometry_row["xml_path"]))
        parameters = geometry_row.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"Missing parameters in geometry row: {geometry_row}")
        geometry_id = str(geometry_row.get("geometry_id", "")).strip()
        if not geometry_id:
            raise ValueError(f"Missing geometry_id in geometry row: {geometry_row}")
        if geometry_directory.name != geometry_id:
            raise ValueError(f"Geometry directory does not match geometry_id: {geometry_directory}")
        if params_path.parent != geometry_directory:
            raise ValueError(f"Geometry JSON is outside geometry_directory: {params_path}")
        if xml_path.parent != geometry_directory:
            raise ValueError(f"Geometry XML is outside geometry_directory: {xml_path}")
        if require_geometry_files:
            _validate_geometry_files(geometry_id, params_path, xml_path)
        spec_path = _resolve_project_path(str(geometry_row["spec_path"]))
        # Keep the generated parameter payload attached to the geometry record because the HCAL
        # stack builder uses those values directly instead of reverse-engineering the compact XML.
        geometry_variants.append(
            GeometryVariant(
                geometry_id=geometry_id,
                tag=str(geometry_row.get("tag", "")),
                geometry_directory=geometry_directory,
                params_path=params_path,
                xml_path=xml_path,
                spec_path=spec_path,
                params={str(key): value for key, value in parameters.items()},
            )
        )
    return geometry_variants


# Translate the generated HCAL depth and detector side into the world z-range used by simulation.
def derive_thickness_and_zrange(geometry_variant: GeometryVariant) -> Tuple[float, float, float]:
    """Return (thickness_mm, zmin_mm_world, zmax_mm_world) from geometry params."""
    layer_rows = build_layer_stack(geometry_variant)
    summary = summarize_layer_stack(layer_rows)
    total_thickness = summary.total_depth_mm

    z_face_mm = eval_length_mm(geometry_variant.params.get("zmin"))
    if geometry_variant.side == "-z":
        zmax_world = -z_face_mm
        zmin_world = -(z_face_mm + total_thickness)
    else:
        zmin_world = z_face_mm
        zmax_world = z_face_mm + total_thickness
    return total_thickness, zmin_world, zmax_world


# Resolve project-relative helper paths in one place so the rest of the module can work with
# absolute paths.
def _resolve_project_path(path_text: str) -> Path:
    raw_path = Path(path_text).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (PROJECT_DIRECTORY / raw_path).resolve()


# Make sure the generated geometry files exist and that the JSON payload matches the expected
# geometry identity.
def _validate_geometry_files(geometry_id: str, params_path: Path, xml_path: Path) -> None:
    if not params_path.exists():
        raise FileNotFoundError(f"Params JSON missing: {params_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"Compact XML missing: {xml_path}")
    with params_path.open("r", encoding="utf-8") as parameter_file:
        parameter_payload = json.load(parameter_file)
    file_geometry_id = str(parameter_payload.get("geometry_id", "")).strip()
    if file_geometry_id != geometry_id:
        raise ValueError(f"Geometry ID mismatch in {params_path}")


# Expand the three-segment HCAL description into one physical layer row per built layer.
def build_layer_stack(geometry_variant: GeometryVariant) -> List[GeometryLayerRow]:
    """Expand the generated 3-segment HCAL into one physical row per layer."""
    segment_recipes = _resolve_segment_recipes(geometry_variant)

    layer_rows: List[GeometryLayerRow] = []
    running_depth_mm = 0.0
    layer_index = 0
    # Walk segment by segment so the output rows follow the same longitudinal ordering as the HCAL plugin.
    for segment_recipe in segment_recipes:
        layer_total_thickness_mm = (
            segment_recipe.absorber_thickness_mm
            + segment_recipe.scintillator_thickness_mm
            + 2.0 * segment_recipe.spacer_thickness_mm
        )
        for _ in range(segment_recipe.layer_count):
            # Record the front, middle, and back of each physical layer so later studies can
            # place observables anywhere through the stack.
            depth_front_mm = running_depth_mm
            absorber_depth_mid_mm = depth_front_mm + 0.5 * segment_recipe.absorber_thickness_mm
            scintillator_front_mm = (
                depth_front_mm
                + segment_recipe.absorber_thickness_mm
                + segment_recipe.spacer_thickness_mm
            )
            scintillator_depth_mid_mm = (
                scintillator_front_mm + 0.5 * segment_recipe.scintillator_thickness_mm
            )
            depth_back_mm = depth_front_mm + layer_total_thickness_mm
            depth_mid_mm = 0.5 * (depth_front_mm + depth_back_mm)

            layer_rows.append(
                GeometryLayerRow(
                    layer_index=layer_index,
                    absorber_thickness_mm=segment_recipe.absorber_thickness_mm,
                    scintillator_thickness_mm=segment_recipe.scintillator_thickness_mm,
                    spacer_thickness_mm=segment_recipe.spacer_thickness_mm,
                    layer_total_thickness_mm=layer_total_thickness_mm,
                    depth_front_mm=depth_front_mm,
                    depth_mid_mm=depth_mid_mm,
                    depth_back_mm=depth_back_mm,
                    absorber_depth_mid_mm=absorber_depth_mid_mm,
                    scintillator_depth_mid_mm=scintillator_depth_mid_mm,
                )
            )

            running_depth_mm = depth_back_mm
            layer_index += 1
    return layer_rows


# Reduce the expanded stack to the geometry-level depth numbers used by later analysis steps.
def summarize_layer_stack(layer_rows: List[GeometryLayerRow]) -> GeometryLayerSummary:
    """Reduce the layer rows to the geometry-level depth scalars."""
    if not layer_rows:
        raise ValueError("Geometry has no resolved layers.")

    total_depth_mm = layer_rows[-1].depth_back_mm
    total_absorber_thickness_mm = sum(
        layer_row.absorber_thickness_mm for layer_row in layer_rows
    )
    absorber_fraction_by_depth = (
        total_absorber_thickness_mm / total_depth_mm if total_depth_mm > 0.0 else 0.0
    )
    return GeometryLayerSummary(
        total_depth_mm=total_depth_mm,
        total_absorber_thickness_mm=total_absorber_thickness_mm,
        absorber_fraction_by_depth=absorber_fraction_by_depth,
    )


# Recover the three contiguous longitudinal segment recipes from the generated HCAL parameter set.
def _resolve_segment_recipes(geometry_variant: GeometryVariant) -> List[GeometrySegmentRecipe]:
    """Build the three contiguous longitudinal segment recipes used by the HCAL plugin."""
    geometry_parameters = geometry_variant.params
    if geometry_variant.n_layers <= 0:
        raise ValueError("Generated HCAL geometry must define a positive nLayers.")

    # The generated HCAL contract requires exactly three positive segment counts that sum to nLayers.
    segment_layer_counts = [
        int(geometry_parameters.get("seg1_layers", 0) or 0),
        int(geometry_parameters.get("seg2_layers", 0) or 0),
        int(geometry_parameters.get("seg3_layers", 0) or 0),
    ]
    if any(segment_layer_count <= 0 for segment_layer_count in segment_layer_counts):
        raise ValueError(
            "Generated HCAL geometry must define positive seg1_layers, seg2_layers, and seg3_layers."
        )
    if sum(segment_layer_counts) != geometry_variant.n_layers:
        raise ValueError("Segment layer counts must sum to nLayers.")

    # The spacer thickness is the same for every layer across all three segments.
    base_spacer_mm = eval_length_mm(geometry_parameters.get("t_spacer"))

    # Build one repeated layer recipe for each longitudinal segment.
    segment_recipes: List[GeometrySegmentRecipe] = []
    for segment_index, segment_layer_count in enumerate(segment_layer_counts, start=1):
        absorber_thickness_mm = eval_length_mm(
            geometry_parameters.get(f"t_absorber_seg{segment_index}")
        )
        scintillator_thickness_mm = eval_length_mm(
            geometry_parameters.get(f"t_scin_seg{segment_index}")
        )
        segment_recipes.append(
            GeometrySegmentRecipe(
                segment_index=segment_index,
                layer_count=segment_layer_count,
                absorber_thickness_mm=absorber_thickness_mm,
                scintillator_thickness_mm=scintillator_thickness_mm,
                spacer_thickness_mm=base_spacer_mm,
            )
        )
    return segment_recipes
