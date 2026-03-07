#!/usr/bin/env python3
"""Generate one HCAL geometry XML/JSON pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as xml_tree
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PATH = "geometries/templates/hcal_template.xml"
DEFAULT_OUTPUT_DIRECTORY = "geometries/generated"
RESERVED_PARAMETER_KEYS = {"geometry_id"}
LEGACY_PARAMETER_KEYS = {
    "t_absorber",
    "t_scin",
    "t_tape",
    "tapeMaterial",
    "t_spacer_seg1",
    "t_spacer_seg2",
    "t_spacer_seg3",
}
NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one geometry XML/JSON pair")
    parser.add_argument("--template", "-t", default=DEFAULT_TEMPLATE_PATH, help="Template XML path")
    parser.add_argument("--out", "-o", help="Output XML path")
    parser.add_argument("--outdir", help="Output directory for auto naming")
    parser.add_argument("--tag", help="Optional output tag")
    parser.add_argument("--detector-name", help="Detector name filter")
    parser.add_argument("--detector-type", help="Detector type filter")
    parser.add_argument("--write-json", help="Output JSON parameter path")
    parser.add_argument("--set", "-s", action="append", default=[], help="Parameter override key=value")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved parameters and exit")
    return parser.parse_args()


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


def parse_set_assignment(assignment_text: str) -> Tuple[str, str]:
    if "=" not in assignment_text:
        raise ValueError(f"Invalid --set value: {assignment_text}")
    key_text, value_text = assignment_text.split("=", 1)
    key_text = key_text.strip()
    value_text = value_text.strip()
    if not key_text:
        raise ValueError(f"Invalid --set key: {assignment_text}")
    return key_text, value_text


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


def set_detector_parameter(detector_element: xml_tree.Element, key_text: str, value_text: str) -> None:
    for parameter_element in detector_element.findall("parameter"):
        if parameter_element.get("name") == key_text:
            parameter_element.set("value", value_text)
            return
    new_parameter = xml_tree.SubElement(detector_element, "parameter")
    new_parameter.set("name", key_text)
    new_parameter.set("value", value_text)


def parse_int_value(value_text: str, key_text: str) -> int:
    try:
        return int(float(value_text))
    except ValueError as error:
        raise ValueError(f"{key_text} must be an integer-like value.") from error


def validate_parameter_contract(parameter_values: Dict[str, str]) -> None:
    legacy_keys = sorted(key_text for key_text in parameter_values if key_text in LEGACY_PARAMETER_KEYS)
    if legacy_keys:
        joined_keys = ", ".join(legacy_keys)
        raise ValueError(
            "Legacy HCAL parameters are no longer supported: "
            f"{joined_keys}. Use t_spacer, spacerMaterial, "
            "t_absorber_seg1/2/3, and t_scin_seg1/2/3."
        )

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


def compute_geometry_id(parameter_values: Dict[str, str]) -> str:
    digest_input = json.dumps(parameter_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(digest_input).hexdigest()[:8]


def convert_json_value(value_text: str) -> object:
    value_text = value_text.strip()
    if NUMERIC_PATTERN.match(value_text):
        numeric_value = float(value_text)
        if numeric_value.is_integer():
            return int(numeric_value)
        return numeric_value
    return value_text


def create_json_payload(parameter_values: Dict[str, str], geometry_id: str) -> Dict[str, object]:
    payload: Dict[str, object] = {"geometry_id": geometry_id}
    for key_text, value_text in sorted(parameter_values.items()):
        payload[key_text] = convert_json_value(value_text)
    return payload


def choose_output_paths(arguments: argparse.Namespace, geometry_id: str) -> Tuple[Path, Path]:
    if arguments.out:
        output_xml_path = resolve_project_path(arguments.out)
        geometry_directory = output_xml_path.parent
    else:
        if arguments.outdir:
            geometry_root_directory = resolve_project_path(arguments.outdir)
        else:
            geometry_root_directory = resolve_project_path(DEFAULT_OUTPUT_DIRECTORY)
        geometry_directory = geometry_root_directory / geometry_id
        output_xml_path = geometry_directory / "geometry.xml"

    if arguments.write_json:
        output_json_path = resolve_project_path(arguments.write_json)
    else:
        output_json_path = geometry_directory / "geometry.json"

    return output_xml_path, output_json_path


def indent_xml(element: xml_tree.Element, level: int = 0) -> None:
    indentation = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "
        for child_element in list(element):
            indent_xml(child_element, level + 1)
        if not child_element.tail or not child_element.tail.strip():
            child_element.tail = indentation
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def main() -> None:
    arguments = parse_arguments()

    template_path = resolve_project_path(arguments.template)
    if not template_path.exists():
        raise SystemExit(f"Template XML not found: {template_path}")

    xml_document = xml_tree.parse(template_path)
    root_element = xml_document.getroot()
    detector_element = find_target_detector(root_element, arguments.detector_name, arguments.detector_type)

    parameter_values = read_detector_parameters(detector_element)

    for assignment_text in arguments.set:
        key_text, value_text = parse_set_assignment(assignment_text)
        if key_text in RESERVED_PARAMETER_KEYS:
            continue
        parameter_values[key_text] = value_text

    parameter_values.setdefault("side", "-z")
    validate_parameter_contract(parameter_values)
    geometry_id = compute_geometry_id(parameter_values)
    output_xml_path, output_parameter_path = choose_output_paths(arguments, geometry_id)

    for key_text, value_text in sorted(parameter_values.items()):
        set_detector_parameter(detector_element, key_text, value_text)

    output_parameter_payload = create_json_payload(parameter_values, geometry_id)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    output_parameter_path.parent.mkdir(parents=True, exist_ok=True)

    tag_text = arguments.tag if arguments.tag else geometry_id
    parameter_source_text = to_project_relative_text(output_parameter_path)

    comment_lines = [
        f" Generated by generate_hcal.py on {datetime.now().isoformat(timespec='seconds')}",
        f" GeometryID: {geometry_id}",
        f" Tag: {tag_text}",
        f" Template: {to_project_relative_text(template_path)}",
        f" ParameterSource: {parameter_source_text}",
        " MergedParameters:",
    ]
    comment_lines.extend(f"   - {key_text} = {value_text}" for key_text, value_text in sorted(parameter_values.items()))
    root_element.insert(0, xml_tree.Comment("\n" + "\n".join(comment_lines) + "\n"))

    if arguments.dry_run:
        print("\n".join(comment_lines))
        print(f"OutputXML: {to_project_relative_text(output_xml_path)}")
        print(f"OutputJSON: {to_project_relative_text(output_parameter_path)}")
        return

    indent_xml(root_element)
    with output_parameter_path.open("w", encoding="utf-8") as output_json_file:
        json.dump(output_parameter_payload, output_json_file, indent=2)
    xml_document.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote: {output_xml_path}")
    print(f"GeometryID: {geometry_id}")


if __name__ == "__main__":
    main()
