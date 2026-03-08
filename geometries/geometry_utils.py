#!/usr/bin/env python3
"""Shared geometry helpers used by generate_hcal.py and sweep_geometries.py."""

from __future__ import annotations

import xml.etree.ElementTree as xml_tree
from pathlib import Path
from typing import Dict, Optional

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PATH = "geometries/templates/hcal_template.xml"

# These are flat, single-value parameter names that the generator does not support.
# The per-segment variants (t_absorber_seg1/2/3, t_scin_seg1/2/3, etc.) are required instead
# so that each longitudinal segment can have independent layer thicknesses.
INVALID_PARAMETER_KEYS = {
    "t_absorber",
    "t_scin",
    "t_tape",
    "tapeMaterial",
    "t_spacer_seg1",
    "t_spacer_seg2",
    "t_spacer_seg3",
}


def resolve_project_path(path_text: str) -> Path:
    raw_path = Path(path_text).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (PROJECT_DIRECTORY / raw_path).resolve()


def to_project_relative_text(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIRECTORY))
    except ValueError:
        return str(path)


def find_target_detector(
    root_element: xml_tree.Element,
    detector_name: Optional[str],
    detector_type: Optional[str],
) -> xml_tree.Element:
    detectors_element = root_element.find("detectors")
    if detectors_element is None:
        raise RuntimeError("Template is missing <detectors> section")

    detector_elements = detectors_element.findall("detector")
    if not detector_elements:
        raise RuntimeError("Template has no <detector> entries")

    # Search with the most specific filter first (both name and type), then fall back to
    # name-only or type-only, and finally return the first detector if no filter was given.
    if detector_name and detector_type:
        for detector_element in detector_elements:
            if detector_element.get("name") == detector_name and detector_element.get("type") == detector_type:
                return detector_element

    if detector_name:
        for detector_element in detector_elements:
            if detector_element.get("name") == detector_name:
                return detector_element

    if detector_type:
        for detector_element in detector_elements:
            if detector_element.get("type") == detector_type:
                return detector_element

    return detector_elements[0]


def read_detector_parameters(detector_element: xml_tree.Element) -> Dict[str, str]:
    parameter_values: Dict[str, str] = {}
    for parameter_element in detector_element.findall("parameter"):
        key_text = parameter_element.get("name")
        value_text = parameter_element.get("value")
        if key_text is None or value_text is None:
            continue
        parameter_values[key_text] = value_text
    return parameter_values


def parse_int_value(value_text: str, key_text: str) -> int:
    try:
        return int(float(value_text))
    except ValueError as error:
        raise ValueError(f"{key_text} must be an integer-like value.") from error


def validate_parameter_contract(parameter_values: Dict[str, str]) -> None:
    # Reject any parameter names that belong to the unsupported flat format.
    invalid_keys = sorted(key_text for key_text in parameter_values if key_text in INVALID_PARAMETER_KEYS)
    if invalid_keys:
        joined_keys = ", ".join(invalid_keys)
        raise ValueError(
            "Invalid HCAL parameters: "
            f"{joined_keys}. Use t_spacer, spacerMaterial, "
            "t_absorber_seg1/2/3, and t_scin_seg1/2/3."
        )

    # Confirm that every physical parameter the generator needs is present and non-empty.
    required_keys = [
        "seg1_layers",
        "seg2_layers",
        "seg3_layers",
        "t_spacer",
        "spacerMaterial",
        "t_absorber_seg1",
        "t_absorber_seg2",
        "t_absorber_seg3",
        "t_scin_seg1",
        "t_scin_seg2",
        "t_scin_seg3",
    ]
    missing_keys = [key_text for key_text in required_keys if str(parameter_values.get(key_text, "")).strip() == ""]
    if missing_keys:
        joined_keys = ", ".join(missing_keys)
        raise ValueError(f"Missing required HCAL parameters: {joined_keys}")

    # Check that the three segment layer counts are each positive and sum exactly to nLayers.
    # This enforces the physical constraint that the three segments together cover the full stack.
    layer_count = parse_int_value(str(parameter_values.get("nLayers", "")), "nLayers")
    segment_layer_counts = [
        parse_int_value(str(parameter_values["seg1_layers"]), "seg1_layers"),
        parse_int_value(str(parameter_values["seg2_layers"]), "seg2_layers"),
        parse_int_value(str(parameter_values["seg3_layers"]), "seg3_layers"),
    ]
    if layer_count <= 0:
        raise ValueError("nLayers must be positive.")
    if any(segment_layer_count <= 0 for segment_layer_count in segment_layer_counts):
        raise ValueError("seg1_layers, seg2_layers, and seg3_layers must all be positive.")
    if sum(segment_layer_counts) != layer_count:
        raise ValueError("seg1_layers + seg2_layers + seg3_layers must equal nLayers.")
