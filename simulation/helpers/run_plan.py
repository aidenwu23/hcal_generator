#!/usr/bin/env python3
"""Run planning and deterministic run identity."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry_index import GeometryVariant

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"

PARTICLE_PDG: Dict[str, int] = {
    "neutron": 2112,
    "proton": 2212,
    "kaon0L": 130,
    "pi-": -211,
    "pi+": 211,
    "pion-": -211,
    "pion+": 211,
    "pion0": 111,
    "pi0": 111,
    "mu-": -13,
    "mu+": 13,
    "electron": 11,
    "e-": 11,
    "positron": -11,
    "e+": -11,
    "photon": 22,
    "gamma": 22,
}

# Translate the conductor particle names into the PDG codes expected by the processor.
def lookup_pdg(particle: str) -> Optional[int]:
    """Translate a human-readable particle name into a PDG code when possible."""
    particle_name = particle.strip()
    if particle_name in PARTICLE_PDG:
        return PARTICLE_PDG[particle_name]
    return PARTICLE_PDG.get(particle_name.lower())


# One fully expanded simulation run, including all file paths that later execution steps need.
@dataclass
class RunPlan:
    geometry_variant: GeometryVariant
    gun_particle: str
    momentum_GeV: float
    gun_direction: str
    gun_position: str
    seed: Optional[int]
    n_events: int
    run_id: str
    run_id_int: int
    raw_path: Path
    events_path: Path
    meta_path: Path
    calibration_path: Path
    performance_path: Path
    expected_pdg: Optional[int]


# Per-run bookkeeping that records status, timing, and any failure message after execution.
@dataclass
class RunRecord:
    plan: RunPlan
    status: str
    error: Optional[str] = None
    ddsim_seconds: Optional[float] = None
    process_seconds: Optional[float] = None
    performance_seconds: Optional[float] = None
    meta_seconds: Optional[float] = None


# Build a deterministic run identity from the geometry, beam settings, and processing extras
# so the same physical configuration always maps to the same run id.
def compute_run_id(
    geometry_id: str,
    particle: str,
    energy: float,
    seed: Optional[int],
    event_count: int,
    extra_tokens: Sequence[str],
) -> Tuple[str, int]:
    seed_token = "noseed" if seed is None else str(seed)
    payload = "|".join(
        [
            geometry_id,
            particle,
            f"{energy:.6f}",
            seed_token,
            str(event_count),
            *extra_tokens,
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    run_id = f"run{digest}"
    run_id_int = int(digest[:8], 16) & 0x7FFFFFFF
    return run_id, run_id_int


# Expand the geometry list, energies, and seeds into the concrete run list that conductor executes.
def build_run_plans(
    args: argparse.Namespace,
    geometry_variants: List[GeometryVariant],
    extra_process_flags: Sequence[str],
) -> List[RunPlan]:
    run_plans: List[RunPlan] = []
    seed_values: List[Optional[int]] = list(args.seeds) if args.seeds is not None else [None]
    requested_particles = [str(particle).strip() for particle in args.gun_particle if str(particle).strip()]
    momentum_values = list(args.gun_momentum)
    for geometry_variant in geometry_variants:
        # Include the processor extras, geometry tag, and detector side in the run identity so
        # distinct physical or processing configurations do not collide onto the same run id.
        run_id_tokens = list(extra_process_flags) + [geometry_variant.tag or "", geometry_variant.side]
        gun_direction = geometry_variant.params.get("gun.direction", args.gun_direction)
        gun_position = geometry_variant.params.get("gun.position", args.gun_position)

        # Sweep over every requested particle
        for particle in requested_particles:
            expected_pdg = lookup_pdg(particle)
            # Per particle, sweep all requested momenta
            for momentum_gev in momentum_values:
                # Per momentum, sweep every requested seed value
                for seed in seed_values:
                    run_id, run_id_int = compute_run_id(
                        geometry_variant.geometry_id,
                        particle,
                        momentum_gev,
                        seed,
                        args.events,
                        run_id_tokens,
                    )
                    # Keep the raw EDM4hep file and the processed outputs in their standard campaign locations.
                    raw_output_directory = DATA_DIRECTORY / "raw" / geometry_variant.geometry_id
                    processed_output_directory = DATA_DIRECTORY / "processed" / geometry_variant.geometry_id / run_id
                    run_plans.append(
                        RunPlan(
                            geometry_variant=geometry_variant,
                            gun_particle=particle,
                            momentum_GeV=momentum_gev,
                            gun_direction=gun_direction,
                            gun_position=gun_position,
                            seed=seed,
                            n_events=args.events,
                            run_id=run_id,
                            run_id_int=run_id_int,
                            raw_path=raw_output_directory / f"{run_id}.edm4hep.root",
                            events_path=processed_output_directory / "events.root",
                            meta_path=processed_output_directory / "meta.json",
                            calibration_path=processed_output_directory / "calibration.json",
                            performance_path=processed_output_directory / "performance.json",
                            expected_pdg=expected_pdg,
                        )
                    )
    return run_plans
