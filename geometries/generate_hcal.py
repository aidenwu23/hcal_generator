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

from geometry_utils import (
    find_target_detector,
    read_detector_parameters,
    resolve_project_path,
    to_project_relative_text,
    validate_parameter_contract,
    DEFAULT_TEMPLATE_PATH,
)

DEFAULT_OUTPUT_DIRECTORY = "geometries/generated"
RESERVED_PARAMETER_KEYS = {"geometry_id"}
NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
SEGMENT_LENGTH_KEYS = {
    "t_absorber_seg1",
    "t_absorber_seg2",
    "t_absorber_seg3",
    "t_scin_seg1",
    "t_scin_seg2",
    "t_scin_seg3",
}


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


def parse_set_assignment(assignment_text: str) -> tuple[str, str]:
    # Split one --set key=value assignment into its two parts.
    if "=" not in assignment_text:
        raise ValueError(f"Invalid --set value: {assignment_text}")
    key_text, value_text = assignment_text.split("=", 1)
    key_text = key_text.strip()
    value_text = value_text.strip()
    if not key_text:
        raise ValueError(f"Invalid --set key: {assignment_text}")
    return key_text, value_text


def set_detector_parameter(detector_element: xml_tree.Element, key_text: str, value_text: str) -> None:
    # Update an existing parameter or append a new one when it is missing.
    for parameter_element in detector_element.findall("parameter"):
        if parameter_element.get("name") == key_text:
            parameter_element.set("value", value_text)
            return
    new_parameter = xml_tree.SubElement(detector_element, "parameter")
    new_parameter.set("name", key_text)
    new_parameter.set("value", value_text)


# Hash the sorted parameter set to get a deterministic geometry ID.
def compute_geometry_id(parameter_values: dict[str, str]) -> str:
    digest_input = json.dumps(parameter_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(digest_input).hexdigest()[:8]


# Convert bare numeric strings into JSON numbers.
def convert_json_value(value_text: str) -> object:
    value_text = value_text.strip()
    if NUMERIC_PATTERN.match(value_text):
        numeric_value = float(value_text)
        if numeric_value.is_integer():
            return int(numeric_value)
        return numeric_value
    return value_text


# Keep sweep thickness inputs in centimeters and write explicit DD4hep unit expressions.
def normalize_hcal_parameter_value(key_text: str, value_text: str) -> str:
    stripped_value = value_text.strip()
    if key_text in SEGMENT_LENGTH_KEYS and NUMERIC_PATTERN.match(stripped_value):
        return f"{stripped_value}*cm"
    return stripped_value


def create_json_payload(parameter_values: dict[str, str], geometry_id: str) -> dict[str, object]:
    # Build the JSON parameter payload that accompanies the generated XML.
    payload: dict[str, object] = {"geometry_id": geometry_id}
    for key_text, value_text in sorted(parameter_values.items()):
        payload[key_text] = convert_json_value(normalize_hcal_parameter_value(key_text, value_text))
    return payload


def choose_output_paths(arguments: argparse.Namespace, geometry_id: str) -> tuple[Path, Path]:
    # Resolve the XML and JSON output paths for this geometry.
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
    # Apply the repo XML indentation style recursively.
    indentation = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "
        child_elements = list(element)
        for child_element in child_elements:
            indent_xml(child_element, level + 1)
        last_child = child_elements[-1]
        if not last_child.tail or not last_child.tail.strip():
            last_child.tail = indentation
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def main() -> None:
    arguments = parse_arguments()

    # Load the template XML and select the detector element to edit.
    template_path = resolve_project_path(arguments.template)
    if not template_path.exists():
        raise SystemExit(f"Template XML not found: {template_path}")

    xml_document = xml_tree.parse(template_path)
    root_element = xml_document.getroot()
    detector_element = find_target_detector(root_element, arguments.detector_name, arguments.detector_type)

    # Start from the template parameters, then apply CLI overrides.
    parameter_values = read_detector_parameters(detector_element)

    # Skip reserved keys so callers cannot inject a fake geometry ID.
    for assignment_text in arguments.set:
        key_text, value_text = parse_set_assignment(assignment_text)
        if key_text in RESERVED_PARAMETER_KEYS:
            continue
        parameter_values[key_text] = value_text

    # Validate the merged parameters, then choose the output locations.
    parameter_values.setdefault("side", "-z")
    validate_parameter_contract(parameter_values)
    geometry_id = compute_geometry_id(parameter_values)
    output_xml_path, output_parameter_path = choose_output_paths(arguments, geometry_id)

    # Write the resolved parameter values back into the XML.
    for key_text, value_text in sorted(parameter_values.items()):
        set_detector_parameter(
            detector_element,
            key_text,
            normalize_hcal_parameter_value(key_text, value_text),
        )

    # Build the JSON payload and create the output directories.
    output_parameter_payload = create_json_payload(parameter_values, geometry_id)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    output_parameter_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the provenance comment block that is embedded in the XML.
    tag_text = arguments.tag or geometry_id
    parameter_source_text = to_project_relative_text(output_parameter_path)

    comment_lines = [
        f" Generated by generate_hcal.py on {datetime.now().isoformat(timespec='seconds')}",
        f" GeometryID: {geometry_id}",
        f" Tag: {tag_text}",
        f" Template: {to_project_relative_text(template_path)}",
        f" ParameterSource: {parameter_source_text}",
        " MergedParameters:",
    ]
    comment_lines.extend(
        f"   - {key_text} = {normalize_hcal_parameter_value(key_text, value_text)}"
        for key_text, value_text in sorted(parameter_values.items())
    )
    root_element.insert(0, xml_tree.Comment("\n" + "\n".join(comment_lines) + "\n"))

    # Print the resolved outputs without writing files during dry runs.
    if arguments.dry_run:
        print("\n".join(comment_lines))
        print(f"OutputXML: {to_project_relative_text(output_xml_path)}")
        print(f"OutputJSON: {to_project_relative_text(output_parameter_path)}")
        return

    # Write both generated outputs to disk.
    indent_xml(root_element)
    with output_parameter_path.open("w", encoding="utf-8") as output_json_file:
        json.dump(output_parameter_payload, output_json_file, indent=2)
    xml_document.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote: {output_xml_path}")
    print(f"GeometryID: {geometry_id}")


if __name__ == "__main__":
    main()
