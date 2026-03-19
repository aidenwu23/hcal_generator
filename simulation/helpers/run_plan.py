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

PARTICLE_REST_MASS_GEV: Dict[str, float] = {
    "neutron": 0.9395654205,
    "proton": 0.9382720882,
    "kaon0l": 0.497611,
    "kaon0L": 0.497611,
    "pi-": 0.13957039,
    "pi+": 0.13957039,
    "pion-": 0.13957039,
    "pion+": 0.13957039,
    "pion0": 0.1349768,
    "pi0": 0.1349768,
    "mu-": 0.1056583755,
    "mu+": 0.1056583755,
    "electron": 0.00051099895,
    "e-": 0.00051099895,
    "positron": 0.00051099895,
    "e+": 0.00051099895,
    "photon": 0.0,
    "gamma": 0.0,
}


# Translate the conductor particle names into the PDG codes expected by the processor.
def lookup_pdg(particle: str) -> Optional[int]:
    """Translate a human-readable particle name into a PDG code when possible."""
    particle_name = particle.strip()
    if particle_name in PARTICLE_PDG:
        return PARTICLE_PDG[particle_name]
    return PARTICLE_PDG.get(particle_name.lower())


# Translate the conductor particle names into the rest-mass energy needed for kinetic-energy input.
def lookup_rest_mass_gev(particle: str) -> Optional[float]:
    """Translate a human-readable particle name into a rest-mass energy in GeV when possible."""
    particle_name = particle.strip()
    if particle_name in PARTICLE_REST_MASS_GEV:
        return PARTICLE_REST_MASS_GEV[particle_name]
    return PARTICLE_REST_MASS_GEV.get(particle_name.lower())


# One fully expanded simulation run, including all file paths that later execution steps need.
@dataclass
class RunPlan:
    geometry_variant: GeometryVariant
    gun_particle: str
    gun_energy_GeV: float
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
    use_kinetic_energy = args.gun_kinetic_energy is not None
    energy_values = list(args.gun_kinetic_energy) if use_kinetic_energy else list(args.gun_energy)
    for geometry_variant in geometry_variants:
        # Include the processor extras, geometry tag, and detector side in the run identity so
        # distinct physical or processing configurations do not collide onto the same run id.
        run_id_tokens = list(extra_process_flags) + [geometry_variant.tag or "", geometry_variant.side]
        gun_direction = geometry_variant.params.get("gun.direction", args.gun_direction)
        gun_position = geometry_variant.params.get("gun.position", args.gun_position)

        # Sweep over every requested particle
        for particle in requested_particles:
            expected_pdg = lookup_pdg(particle)
            rest_mass_gev = None
            if use_kinetic_energy:
                rest_mass_gev = lookup_rest_mass_gev(particle)
                if rest_mass_gev is None:
                    raise ValueError(f"No rest mass configured for particle '{particle}' used with --gun-kinetic-energy.")
            # Per particle, sweep all requested energies
            for energy in energy_values:
                total_energy_gev = energy + rest_mass_gev if rest_mass_gev is not None else energy
                # Per energy, sweep every requested seed value
                for seed in seed_values:
                    run_id, run_id_int = compute_run_id(
                        geometry_variant.geometry_id,
                        particle,
                        total_energy_gev,
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
                            gun_energy_GeV=total_energy_gev,
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
