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

from geometry_utils import (
    compute_geometry_id,
    create_json_payload,
    DEFAULT_TEMPLATE_PATH,
    PROJECT_DIRECTORY,
    find_target_detector,
    read_detector_parameters,
    resolve_geometry_output_paths,
    resolve_project_path,
    to_project_relative_text,
    validate_parameter_contract,
)

DEFAULT_GENERATED_OUTPUT_DIRECTORY = "geometries/generated"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate geometry sweeps from YAML specifications")
    parser.add_argument("--spec", "-s", nargs="+", required=True, help="Sweep YAML file path(s)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs")
    return parser.parse_args()


def load_yaml_object(yaml_path: Path) -> dict:
    # Load one sweep YAML file into a top-level mapping.
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
    return loaded_object


def sanitize_tag_text(tag_text: str) -> str:
    # Keep only filesystem-safe tag characters.
    clean_text = re.sub(r"[^A-Za-z0-9_.-]", "", tag_text.strip().replace(" ", "_"))
    if clean_text:
        return clean_text
    return "sweep"


def stringify_geometry_parameters(raw_parameter_map: dict[str, object]) -> dict[str, str]:
    # Convert sweep parameters into the string-valued form expected by generate_hcal.py.
    geometry_parameters = {
        str(key_text): str(value_object)
        for key_text, value_object in raw_parameter_map.items()
        if str(key_text) != "tag"
    }
    geometry_parameters.setdefault("side", "-z")
    return geometry_parameters


def read_existing_geometry_id(parameter_json_path: Path) -> str:
    # Read the stored geometry ID from an existing generated JSON file.
    if not parameter_json_path.exists():
        raise FileNotFoundError(f"Missing parameter JSON: {parameter_json_path}")
    with parameter_json_path.open("r", encoding="utf-8") as parameter_file:
        payload = json.load(parameter_file)
    geometry_id_value = payload.get("geometry_id")
    if geometry_id_value is None or str(geometry_id_value).strip() == "":
        raise ValueError(f"Missing geometry_id in: {parameter_json_path}")
    return str(geometry_id_value)


def run_generation_command(command: list[str]) -> str:
    # Run generate_hcal.py and return its combined stdout text.
    result = subprocess.run(
        command,
        cwd=PROJECT_DIRECTORY,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


def build_generate_command(
    template_path: Path,
    generated_output_directory: Path,
    geometry_tag: str,
    detector_name: str | None,
    detector_type: str | None,
    geometry_parameters: dict[str, str],
) -> list[str]:
    # Build the generate_hcal.py command for one geometry variant.
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


def run_geometry_generation(command: list[str]) -> str:
    # Run the real geometry generation and read back the reported geometry ID.
    output_text = run_generation_command(command)
    geometry_id = ""
    for line in output_text.splitlines():
        if line.strip().startswith("GeometryID:"):
            geometry_id = line.strip().split(":", 1)[1].strip()
            break
    if not geometry_id:
        raise RuntimeError("generate_hcal.py did not report GeometryID")
    return geometry_id


def build_variant_parameter_list(specification: dict) -> tuple[list[dict[str, object]], str, int]:
    # Read the shared constants and the per-variant override list from the sweep spec.
    constant_parameters = specification.get("constants", {}) or {}
    if not isinstance(constant_parameters, dict):
        raise ValueError("constants must be an object")

    raw_variant_list = specification.get("variants", []) or []
    if not isinstance(raw_variant_list, list):
        raise ValueError("variants must be a list")

    # Merge the shared constants into each variant record.
    variant_parameter_list: list[dict[str, object]] = []
    for raw_variant in raw_variant_list:
        if raw_variant is None:
            raw_variant = {}
        if not isinstance(raw_variant, dict):
            raise ValueError("each variant entry must be an object")
        variant_parameters = dict(constant_parameters)
        variant_parameters.update(raw_variant)
        variant_parameter_list.append(variant_parameters)

    # Resolve the naming scheme used for generated variant tags and indices.
    name_text = str(specification.get("name", "sweep"))
    tag_prefix = sanitize_tag_text(str(specification.get("tag_prefix", name_text)))
    index_base = int(specification.get("index_base", 0))
    return variant_parameter_list, tag_prefix, index_base


def build_geometry_rows(specification: dict, sweep_spec_path: Path) -> list[dict[str, object]]:
    # Resolve the sweep-level template, detector selection, and generated output directory.
    template_path = resolve_project_path(str(specification.get("template", DEFAULT_TEMPLATE_PATH)))
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    detector_name = specification.get("detector_name")
    detector_type = specification.get("detector_type")
    generated_output_directory = resolve_project_path(
        str(specification.get("outdir_generated", DEFAULT_GENERATED_OUTPUT_DIRECTORY))
    )

    variant_parameter_list, tag_prefix, index_base = build_variant_parameter_list(specification)
    geometry_rows: list[dict[str, object]] = []
    # Build one geometry row for each requested variant.
    for list_offset, raw_variant_parameters in enumerate(variant_parameter_list):
        variant_index = index_base + list_offset
        variant_name = f"variant{variant_index:03d}"

        # Use the requested tag when present, otherwise build one from the sweep prefix.
        requested_tag = raw_variant_parameters.get("tag")
        if requested_tag is None:
            geometry_tag = f"{tag_prefix}_{variant_name}"
        else:
            geometry_tag = sanitize_tag_text(str(requested_tag))

        # Read the template defaults, then merge the variant parameters on top.
        xml_document = xml_tree.parse(template_path)
        root_element = xml_document.getroot()
        detector_element = find_target_detector(root_element, detector_name, detector_type)
        geometry_parameters = read_detector_parameters(detector_element)
        geometry_parameters.update(stringify_geometry_parameters(raw_variant_parameters))
        validate_parameter_contract(geometry_parameters)

        # Ask generate_hcal.py for the predicted ID and output paths before writing files.
        generation_command = build_generate_command(
            template_path=template_path,
            generated_output_directory=generated_output_directory,
            geometry_tag=geometry_tag,
            detector_name=detector_name,
            detector_type=detector_type,
            geometry_parameters=geometry_parameters,
        )
        geometry_id = compute_geometry_id(geometry_parameters)
        output_xml_path, output_parameter_path = resolve_geometry_output_paths(
            geometry_id,
            outdir=str(generated_output_directory),
        )
        # Record the full geometry row for later validation and real generation.
        geometry_rows.append(
            {
                "spec_path": str(sweep_spec_path),
                "index": variant_index,
                "variant_name": variant_name,
                "tag": geometry_tag,
                "geometry_id": geometry_id,
                "geometry_directory": to_project_relative_text(output_xml_path.parent),
                "json_path": to_project_relative_text(output_parameter_path),
                "xml_path": to_project_relative_text(output_xml_path),
                "parameters": create_json_payload(geometry_parameters, geometry_id),
                "command": generation_command,
            }
        )
    return geometry_rows


def main() -> None:
    arguments = parse_arguments()

    # Build all requested geometry rows, one sweep spec at a time.
    for sweep_spec_text in arguments.spec:
        sweep_spec_path = resolve_project_path(sweep_spec_text)
        if not sweep_spec_path.exists():
            raise FileNotFoundError(f"Sweep spec not found: {sweep_spec_path}")
        specification = load_yaml_object(sweep_spec_path)
        geometry_rows = build_geometry_rows(specification, sweep_spec_path)

        generated_count = 0
        # Generate each geometry unless the existing outputs are still valid.
        for geometry_row in geometry_rows:
            geometry_id = str(geometry_row["geometry_id"])
            output_parameter_path = resolve_project_path(str(geometry_row["json_path"]))
            output_xml_path = resolve_project_path(str(geometry_row["xml_path"]))

            # Skip regeneration when the on-disk geometry already matches this row.
            if output_parameter_path.exists() and output_xml_path.exists() and not arguments.overwrite:
                existing_geometry_id = read_existing_geometry_id(output_parameter_path)
                if existing_geometry_id != geometry_id:
                    raise ValueError(f"Geometry ID mismatch in {output_parameter_path}")
                continue

            # Run generation and verify the returned geometry ID.
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


if __name__ == "__main__":
    main()
