#!/usr/bin/env python3
"""Generate geometry variants from one or more sweep YAML files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as xml_tree
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PATH = "geometries/templates/hcal_template.xml"
DEFAULT_GENERATED_OUTPUT_DIRECTORY = "geometries/generated"
LEGACY_PARAMETER_KEYS = {
    "t_absorber",
    "t_scin",
    "t_tape",
    "tapeMaterial",
    "t_spacer_seg1",
    "t_spacer_seg2",
    "t_spacer_seg3",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate geometry sweeps from YAML specifications")
    parser.add_argument("--spec", "-s", nargs="+", required=True, help="Sweep YAML file path(s)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print planned geometry rows as JSON")
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


def load_yaml_object(yaml_path: Path) -> dict:
    if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Unsupported sweep file extension: {yaml_path}")

    try:
        import yaml  # type: ignore
    except Exception as error:
        raise RuntimeError("PyYAML is required to read sweep YAML files") from error

    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        loaded_object = yaml.safe_load(yaml_file)

    if loaded_object is None:
        return {}
    if not isinstance(loaded_object, dict):
        raise ValueError(f"Sweep YAML must contain an object at top level: {yaml_path}")
    for unsupported_key in ("outdir_xml", "outdir_params"):
        if unsupported_key in loaded_object:
            raise ValueError(f"Unsupported sweep key: {unsupported_key}")
    return loaded_object


def sanitize_tag_text(tag_text: str) -> str:
    clean_text = re.sub(r"[^A-Za-z0-9_.-]", "", tag_text.strip().replace(" ", "_"))
    if clean_text:
        return clean_text
    return "sweep"


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


def stringify_geometry_parameters(raw_parameter_map: Dict[str, object]) -> Dict[str, str]:
    geometry_parameters = {
        str(key_text): str(value_object)
        for key_text, value_object in raw_parameter_map.items()
        if str(key_text) != "tag"
    }
    geometry_parameters.setdefault("side", "-z")
    return geometry_parameters


def read_existing_geometry_id(parameter_json_path: Path) -> str:
    if not parameter_json_path.exists():
        raise FileNotFoundError(f"Missing parameter JSON: {parameter_json_path}")
    with parameter_json_path.open("r", encoding="utf-8") as parameter_file:
        payload = json.load(parameter_file)
    geometry_id_value = payload.get("geometry_id")
    if geometry_id_value is None or str(geometry_id_value).strip() == "":
        raise ValueError(f"Missing geometry_id in: {parameter_json_path}")
    return str(geometry_id_value)


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


def build_generate_command(
    template_path: Path,
    generated_output_directory: Path,
    geometry_tag: str,
    detector_name: Optional[str],
    detector_type: Optional[str],
    geometry_parameters: Dict[str, str],
) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_DIRECTORY / "geometries" / "generate_hcal.py"),
        "--template",
        str(template_path),
        "--outdir",
        str(generated_output_directory),
        "--tag",
        geometry_tag,
    ]

    if detector_name:
        command.extend(["--detector-name", str(detector_name)])
    if detector_type:
        command.extend(["--detector-type", str(detector_type)])

    for key_text, value_text in geometry_parameters.items():
        command.extend(["--set", f"{key_text}={value_text}"])

    return command


def inspect_geometry_generation(command: List[str]) -> Dict[str, str]:
    inspected_command = [*command, "--dry-run"]
    result = subprocess.run(
        inspected_command,
        cwd=PROJECT_DIRECTORY,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    geometry_id = ""
    output_xml_path = ""
    output_json_path = ""
    for output_line in result.stdout.splitlines():
        stripped_line = output_line.strip()
        if stripped_line.startswith("GeometryID:"):
            geometry_id = stripped_line.split(":", 1)[1].strip()
        elif stripped_line.startswith("OutputXML:"):
            output_xml_path = stripped_line.split(":", 1)[1].strip()
        elif stripped_line.startswith("OutputJSON:"):
            output_json_path = stripped_line.split(":", 1)[1].strip()

    if geometry_id == "" or output_xml_path == "" or output_json_path == "":
        raise RuntimeError("generate_hcal.py --dry-run did not report expected output paths")
    return {
        "geometry_id": geometry_id,
        "xml_path": output_xml_path,
        "json_path": output_json_path,
    }


def run_geometry_generation(command: List[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_DIRECTORY,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    geometry_id = ""
    for output_line in result.stdout.splitlines():
        stripped_line = output_line.strip()
        if stripped_line.startswith("GeometryID:"):
            geometry_id = stripped_line.split(":", 1)[1].strip()

    if geometry_id == "":
        raise RuntimeError("generate_hcal.py did not report GeometryID")
    return geometry_id


def build_variant_parameter_list(specification: dict) -> tuple[List[Dict[str, object]], str, int]:
    constant_parameters = specification.get("constants", {}) or {}
    if not isinstance(constant_parameters, dict):
        raise ValueError("constants must be an object")

    raw_variant_list = specification.get("variants", []) or []
    if not isinstance(raw_variant_list, list):
        raise ValueError("variants must be a list")

    variant_parameter_list: List[Dict[str, object]] = []
    for raw_variant in raw_variant_list:
        if raw_variant is None:
            raw_variant = {}
        if not isinstance(raw_variant, dict):
            raise ValueError("each variant entry must be an object")
        variant_parameters = dict(constant_parameters)
        variant_parameters.update(raw_variant)
        variant_parameter_list.append(variant_parameters)

    name_text = str(specification.get("name", "sweep"))
    tag_prefix = sanitize_tag_text(str(specification.get("tag_prefix", name_text)))
    index_base = int(specification.get("index_base", 0))
    return variant_parameter_list, tag_prefix, index_base


def build_geometry_rows(specification: dict, sweep_spec_path: Path) -> List[Dict[str, object]]:
    template_path = resolve_project_path(str(specification.get("template", DEFAULT_TEMPLATE_PATH)))
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    detector_name = specification.get("detector_name")
    detector_type = specification.get("detector_type")
    generated_output_directory = resolve_project_path(
        str(specification.get("outdir_generated", DEFAULT_GENERATED_OUTPUT_DIRECTORY))
    )

    variant_parameter_list, tag_prefix, index_base = build_variant_parameter_list(specification)
    geometry_rows: List[Dict[str, object]] = []
    for list_offset, raw_variant_parameters in enumerate(variant_parameter_list):
        variant_index = index_base + list_offset
        variant_name = f"variant{variant_index:03d}"

        requested_tag = raw_variant_parameters.get("tag")
        if requested_tag is None:
            geometry_tag = f"{tag_prefix}_{variant_name}"
        else:
            geometry_tag = sanitize_tag_text(str(requested_tag))

        xml_document = xml_tree.parse(template_path)
        root_element = xml_document.getroot()
        detector_element = find_target_detector(root_element, detector_name, detector_type)
        geometry_parameters = read_detector_parameters(detector_element)
        geometry_parameters.update(stringify_geometry_parameters(raw_variant_parameters))
        validate_parameter_contract(geometry_parameters)

        generation_command = build_generate_command(
            template_path=template_path,
            generated_output_directory=generated_output_directory,
            geometry_tag=geometry_tag,
            detector_name=detector_name,
            detector_type=detector_type,
            geometry_parameters=geometry_parameters,
        )
        generated_layout = inspect_geometry_generation(generation_command)
        output_parameter_path = resolve_project_path(generated_layout["json_path"])
        output_xml_path = resolve_project_path(generated_layout["xml_path"])
        geometry_rows.append(
            {
                "spec_path": str(sweep_spec_path),
                "index": variant_index,
                "variant_name": variant_name,
                "tag": geometry_tag,
                "geometry_id": generated_layout["geometry_id"],
                "geometry_directory": to_project_relative_text(output_xml_path.parent),
                "json_path": to_project_relative_text(output_parameter_path),
                "xml_path": to_project_relative_text(output_xml_path),
                "parameters": geometry_parameters,
                "command": generation_command,
            }
        )
    return geometry_rows


def main() -> None:
    arguments = parse_arguments()
    all_geometry_rows: List[Dict[str, object]] = []

    for sweep_spec_text in arguments.spec:
        sweep_spec_path = resolve_project_path(sweep_spec_text)
        if not sweep_spec_path.exists():
            raise FileNotFoundError(f"Sweep spec not found: {sweep_spec_path}")
        specification = load_yaml_object(sweep_spec_path)
        geometry_rows = build_geometry_rows(specification, sweep_spec_path)
        all_geometry_rows.extend(geometry_rows)

        if arguments.dry_run:
            continue

        generated_count = 0
        for geometry_row in geometry_rows:
            geometry_id = str(geometry_row["geometry_id"])
            output_parameter_path = resolve_project_path(str(geometry_row["json_path"]))
            output_xml_path = resolve_project_path(str(geometry_row["xml_path"]))
            if output_parameter_path.exists() and output_xml_path.exists() and not arguments.overwrite:
                existing_geometry_id = read_existing_geometry_id(output_parameter_path)
                if existing_geometry_id != geometry_id:
                    raise ValueError(f"Geometry ID mismatch in {output_parameter_path}")
                continue
            try:
                generated_geometry_id = run_geometry_generation(list(geometry_row["command"]))
            except subprocess.CalledProcessError as error:
                print(error.stdout)
                raise RuntimeError(f"Generation failed for {geometry_row['tag']}") from error
            if generated_geometry_id != geometry_id:
                raise ValueError(
                    f"Geometry ID mismatch for {geometry_row['tag']}: "
                    f"expected {geometry_id}, got {generated_geometry_id}"
                )
            generated_count += 1

        print(f"Generated {generated_count} geometries for {sweep_spec_path}")

    if arguments.dry_run:
        printable_rows = []
        for geometry_row in all_geometry_rows:
            printable_rows.append(
                {
                    "spec_path": geometry_row["spec_path"],
                    "index": geometry_row["index"],
                    "variant_name": geometry_row["variant_name"],
                    "tag": geometry_row["tag"],
                    "geometry_id": geometry_row["geometry_id"],
                    "geometry_directory": geometry_row["geometry_directory"],
                    "json_path": geometry_row["json_path"],
                    "xml_path": geometry_row["xml_path"],
                    "parameters": geometry_row["parameters"],
                }
            )
        print(json.dumps(printable_rows, indent=2))


if __name__ == "__main__":
    main()
