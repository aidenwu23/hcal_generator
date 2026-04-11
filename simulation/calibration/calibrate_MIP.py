#!/usr/bin/env python3
"""
Run a muon control simulation and measure the MIP calibration from raw hits.
Example:
python3 simulation/calibration/calibrate_MIP.py \
  --spec geometries/sweeps/nhcal.yaml \
  --raw-out data/calib/run_mu_ctrl.edm4hep.root \
  --json-out data/calib/calibration.json \
  --plots-out data/calib/landau_plots.root \
  --n-events 10000 \
  --alpha 0.5
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from simulation.helpers.geometry_index import inspect_geometry_rows, load_geometry_variants
from simulation.helpers.run_steps import maybe_run_sweeps

CALIBRATION_MACRO_PATH = PROJECT_DIRECTORY / "simulation" / "calibration" / "calibrate_MIP.C"
PLOT_MACRO_PATH = PROJECT_DIRECTORY / "analysis" / "simulation" / "plot_landau.C"
DEFAULT_GUN_DIRECTION = "0 0 -1"
DEFAULT_GUN_POSITION = "0 0 0"


def run_command(command: List[str], label: str) -> None:
    quoted = " ".join(shlex.quote(token) for token in command)
    print(f"[{label}] {quoted}")
    subprocess.run(command, check=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a muon control calibration for MIP thresholds")
    parser.add_argument("--spec", required=True, help="Sweep YAML that defines the geometry")
    parser.add_argument("--raw-out", required=True, help="Output raw EDM4hep file")
    parser.add_argument("--json-out", required=True, help="Output calibration JSON file")
    parser.add_argument("--plots-out", default="", help="Optional output ROOT file for Landau plots")

    parser.add_argument("--ddsim", default="ddsim", help="ddsim executable")
    parser.add_argument("--root-bin", default="root", help="ROOT executable")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for sweep generation")
    parser.add_argument("--overwrite-geos", action="store_true", help="Regenerate existing geometry outputs")

    parser.add_argument("--gun-particle", default="mu-", help="Control particle")
    parser.add_argument("--gun-energy", default="10*GeV", help="Gun energy expression")
    parser.add_argument("--gun-direction", default=DEFAULT_GUN_DIRECTION, help="Gun direction string")
    parser.add_argument("--gun-position", default=DEFAULT_GUN_POSITION, help="Gun position string")
    parser.add_argument("--n-events", type=int, default=10000, help="Control event count")
    parser.add_argument("--alpha", type=float, default=0.5, help="Threshold in fractions of one MIP")
    return parser.parse_args()


def resolve_spec_path(spec_text: str) -> Path:
    raw_path = Path(spec_text).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (PROJECT_DIRECTORY / raw_path).resolve()


def resolve_first_geometry_xml(arguments: argparse.Namespace) -> Path:
    spec_path = resolve_spec_path(arguments.spec)
    if not spec_path.exists():
        raise FileNotFoundError(f"Sweep spec not found: {spec_path}")

    maybe_run_sweeps(arguments, [spec_path])

    geometry_rows = inspect_geometry_rows([spec_path])
    geometry_variants = load_geometry_variants(geometry_rows, require_geometry_files=True)
    if not geometry_variants:
        raise ValueError(f"No geometry variants found in {spec_path}")
    return geometry_variants[0].xml_path


def main() -> int:
    arguments = parse_arguments()
    raw_output_path = Path(arguments.raw_out).expanduser().resolve()
    json_output_path = Path(arguments.json_out).expanduser().resolve()
    plots_output_path = Path(arguments.plots_out).expanduser().resolve() if arguments.plots_out else None

    if not CALIBRATION_MACRO_PATH.exists():
        print(f"ERROR: missing macro: {CALIBRATION_MACRO_PATH}", file=sys.stderr)
        return 2
    if plots_output_path is not None and not PLOT_MACRO_PATH.exists():
        print(f"ERROR: missing macro: {PLOT_MACRO_PATH}", file=sys.stderr)
        return 2
    if arguments.n_events <= 0:
        print("ERROR: --n-events must be positive", file=sys.stderr)
        return 2
    if arguments.alpha < 0.0:
        print("ERROR: --alpha must be non-negative", file=sys.stderr)
        return 2

    try:
        compact_xml_path = resolve_first_geometry_xml(arguments)
    except Exception as error:
        print(f"ERROR: failed to resolve geometry XML: {error}", file=sys.stderr)
        return 2

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    if plots_output_path is not None:
        plots_output_path.parent.mkdir(parents=True, exist_ok=True)

    ddsim_command: List[str] = [
        arguments.ddsim,
        "--compactFile",
        str(compact_xml_path),
        "--outputFile",
        str(raw_output_path),
        "--numberOfEvents",
        str(arguments.n_events),
        "--enableGun",
        "--gun.particle",
        arguments.gun_particle,
        "--gun.energy",
        arguments.gun_energy,
        "--gun.direction",
        arguments.gun_direction,
        "--gun.position",
        arguments.gun_position,
        "--part.keepAllParticles",
        "true",
        "--part.minimalKineticEnergy",
        "0*MeV",
        "--part.minDistToParentVertex",
        "0*mm",
    ]

    calibration_macro_call = (
        f'{CALIBRATION_MACRO_PATH}('
        f'"{raw_output_path}","{json_output_path}",{arguments.alpha})'
    )
    calibration_command: List[str] = [arguments.root_bin, "-l", "-b", "-q", calibration_macro_call]

    run_command(ddsim_command, "ddsim")
    run_command(calibration_command, "calibration")

    if plots_output_path is not None:
        plot_macro_call = (
            f'{PLOT_MACRO_PATH}('
            f'"{raw_output_path}","{plots_output_path}")'
        )
        plot_command: List[str] = [arguments.root_bin, "-l", "-b", "-q", plot_macro_call]
        run_command(plot_command, "plot_landau")

    if not json_output_path.exists():
        print(f"ERROR: missing calibration output: {json_output_path}", file=sys.stderr)
        return 3

    with json_output_path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if "mpvs" not in payload or "thresholds" not in payload:
        print("ERROR: calibration JSON missing mpvs or thresholds", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
