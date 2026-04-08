#!/usr/bin/env python3
"""
Run a muon control simulation and measure the MIP calibration.
Example:
python3 simulation/calibration/calibrate_MIP.py \
  --compact-xml geometries/generated/42bd89c3/geometry.xml \
  --raw-out data/raw/42bd89c3/run_mu_ctrl_10k/run_mu_ctrl.edm4hep.root \
  --events-out data/processed/42bd89c3/run_mu_ctrl_10k/events.root \
  --json-out data/processed/42bd89c3/run_mu_ctrl_10k/calibration.json \
  --n-events 10000 \
  --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
CALIBRATION_MACRO_PATH = PROJECT_DIRECTORY / "simulation" / "calibration" / "calibrate_MIP.C"
DEFAULT_GUN_DIRECTION = "0 0 -1"
DEFAULT_GUN_POSITION = "0 0 0"


def run_command(command: List[str], label: str) -> None:
    quoted = " ".join(shlex.quote(token) for token in command)
    print(f"[{label}] {quoted}")
    subprocess.run(command, check=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a muon control calibration for MIP thresholds")
    parser.add_argument("--compact-xml", required=True, help="Path to compact XML geometry")
    parser.add_argument("--raw-out", required=True, help="Output raw EDM4hep file")
    parser.add_argument("--events-out", required=True, help="Output processed events file")
    parser.add_argument("--json-out", required=True, help="Output calibration JSON file")

    parser.add_argument("--ddsim", default="ddsim", help="ddsim executable")
    parser.add_argument("--process-bin", default="./build/bin/process", help="Processor executable")
    parser.add_argument("--root-bin", default="root", help="ROOT executable")

    parser.add_argument("--gun-particle", default="mu-", help="Control particle")
    parser.add_argument("--gun-energy", default="10*GeV", help="Gun energy expression")
    parser.add_argument("--gun-direction", default=DEFAULT_GUN_DIRECTION, help="Gun direction string")
    parser.add_argument("--gun-position", default=DEFAULT_GUN_POSITION, help="Gun position string")
    parser.add_argument("--n-events", type=int, default=10000, help="Control event count")
    parser.add_argument("--alpha", type=float, default=0.5, help="Threshold in fractions of one MIP")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    compact_xml_path = Path(arguments.compact_xml).expanduser().resolve()
    raw_output_path = Path(arguments.raw_out).expanduser().resolve()
    events_output_path = Path(arguments.events_out).expanduser().resolve()
    json_output_path = Path(arguments.json_out).expanduser().resolve()

    if not compact_xml_path.exists():
        print(f"ERROR: missing compact XML: {compact_xml_path}", file=sys.stderr)
        return 2
    if not CALIBRATION_MACRO_PATH.exists():
        print(f"ERROR: missing macro: {CALIBRATION_MACRO_PATH}", file=sys.stderr)
        return 2
    if arguments.n_events <= 0:
        print("ERROR: --n-events must be positive", file=sys.stderr)
        return 2
    if arguments.alpha < 0.0:
        print("ERROR: --alpha must be non-negative", file=sys.stderr)
        return 2

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

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

    process_command: List[str] = [
        arguments.process_bin,
        str(raw_output_path),
        "--out",
        str(events_output_path),
        "--expected-pdg",
        "-13",
    ]

    macro_call = (
        f'{CALIBRATION_MACRO_PATH}('
        f'"{events_output_path}","{json_output_path}",{arguments.alpha})'
    )
    macro_command: List[str] = [arguments.root_bin, "-l", "-b", "-q", macro_call]

    run_command(ddsim_command, "ddsim")
    run_command(process_command, "process")
    run_command(macro_command, "macro")

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
