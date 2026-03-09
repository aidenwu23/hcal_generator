"""
Build a geometry display for one geometry ID and overlay MC particle segments
from the first N events of that geometry's first non-muon-control raw run.

Example:
python visuals/visualize.py --geometry-id 81c3da7d -n 1
"""

import argparse
import math
from pathlib import Path
import subprocess

import ROOT

try:
    import uproot
except ImportError:
    uproot = None

ROOT.gROOT.SetBatch(True)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
GEOMETRY_ROOT = PROJECT_ROOT / "geometries" / "generated"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "visuals"
MIN_TRACK_LENGTH = 1e-6  # Treat nearly identical endpoints as zero-length segments.
CUT_FRACTION = 0.5  # Show half of each slab in the cutaway view.
CUTAWAY_TRANSPARENCY = 70
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 900
GEOMETRY_KEY = "default"
FULL_DISPLAY_NAME = "event_display"
CUTAWAY_DISPLAY_NAME = "event_display_cutaway"


def parse_args():
    # Parse the CLI arguments for one geometry display job.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry-id",
        required=True,
        help="Geometry ID used to resolve the XML and first raw run.",
    )
    parser.add_argument(
        "-n",
        "--num-events",
        type=int,
        default=10,
        help="Number of events to overlay from the start of the run.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output ROOT file. Defaults to visuals/<geometry_id>_display_with_particles.root.",
    )
    parser.add_argument(
        "--charged-only",
        action="store_true",
        help="Only draw charged MC particle segments.",
    )
    parser.add_argument(
        "--min-energy",
        type=float,
        default=0.0,
        help="Minimum MC particle kinetic energy in GeV to draw.",
    )
    return parser.parse_args()


def scale_position_for_display(position):
    return [coordinate * 0.1 for coordinate in position]


def get_particle_charge(pdg_code):
    particle = ROOT.TDatabasePDG.Instance().GetParticle(int(pdg_code))
    if particle:
        return particle.Charge()

    fallback_charges = {
        -2212: -3.0,
        -211: -3.0,
        -13: -3.0,
        -11: 3.0,
        11: -3.0,
        13: 3.0,
        211: 3.0,
        2212: 3.0,
    }
    return fallback_charges.get(int(pdg_code), 0.0)


def get_kinetic_energy(momentum_x, momentum_y, momentum_z, mass):
    momentum_squared = (
        momentum_x * momentum_x
        + momentum_y * momentum_y
        + momentum_z * momentum_z
    )
    total_energy = math.sqrt(max(0.0, momentum_squared + mass * mass))
    return total_energy - mass


def get_track_color(pdg_code):
    if pdg_code == 22:
        return ROOT.kYellow + 1
    if pdg_code == 2112:
        return ROOT.kGray + 2
    if abs(pdg_code) == 11:
        return ROOT.kAzure + 1
    if pdg_code == 2212:
        return ROOT.kRed + 1
    return ROOT.kSpring + 5


def get_track_length(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def resolve_geometry_xml_path(geometry_id):
    xml_path = GEOMETRY_ROOT / geometry_id / "geometry.xml"
    if not xml_path.exists():
        raise RuntimeError(f"Missing geometry XML for {geometry_id}: {xml_path}")

    return xml_path


def resolve_raw_run_path(geometry_id):
    # Pick the first raw run for this geometry and skip the muon control file.
    raw_directory = RAW_DATA_ROOT / geometry_id
    if not raw_directory.exists():
        raise RuntimeError(f"Missing raw data directory for {geometry_id}: {raw_directory}")

    run_paths = sorted(raw_directory.glob("run*.edm4hep.root"))
    run_paths = [path for path in run_paths if path.name != "run_mu_ctrl.edm4hep.root"]
    if not run_paths:
        raise RuntimeError(f"No non-muon-control raw runs found for {geometry_id} in {raw_directory}")

    return run_paths[0]


def resolve_output_path(geometry_id, output_argument):
    if output_argument:
        return Path(output_argument)

    return DEFAULT_OUTPUT_ROOT / f"{geometry_id}_display_with_particles.root"


def build_particle_entries(
    pdg_codes,
    vertex_x,
    vertex_y,
    vertex_z,
    endpoint_x,
    endpoint_y,
    endpoint_z,
    mass_values,
    momentum_x,
    momentum_y,
    momentum_z,
):
    # Convert one event of parallel MC arrays into particle segment records.
    particles = []

    for index, pdg_code in enumerate(pdg_codes):
        particle_pdg = int(pdg_code)
        particles.append(
            {
                "pdg": particle_pdg,
                "start": scale_position_for_display(
                    [vertex_x[index], vertex_y[index], vertex_z[index]]
                ),
                "end": scale_position_for_display(
                    [endpoint_x[index], endpoint_y[index], endpoint_z[index]]
                ),
                "is_charged": get_particle_charge(particle_pdg) != 0.0,
                "kinetic_energy_GeV": get_kinetic_energy(
                    momentum_x[index],
                    momentum_y[index],
                    momentum_z[index],
                    mass_values[index],
                ),
            }
        )

    return particles


def load_mc_particles_with_uproot(input_file, num_events):
    # Load the requested events in one pass, then flatten them into particle segments.
    particles = []

    with uproot.open(input_file) as root_file:
        tree = root_file["events"]
        event_count = min(num_events, tree.num_entries)

        pdg_codes = tree["MCParticles.PDG"].array(library="np", entry_stop=event_count)
        vertex_x = tree["MCParticles.vertex.x"].array(
            library="np", entry_stop=event_count
        )
        vertex_y = tree["MCParticles.vertex.y"].array(
            library="np", entry_stop=event_count
        )
        vertex_z = tree["MCParticles.vertex.z"].array(
            library="np", entry_stop=event_count
        )
        endpoint_x = tree["MCParticles.endpoint.x"].array(
            library="np", entry_stop=event_count
        )
        endpoint_y = tree["MCParticles.endpoint.y"].array(
            library="np", entry_stop=event_count
        )
        endpoint_z = tree["MCParticles.endpoint.z"].array(
            library="np", entry_stop=event_count
        )
        mass_values = tree["MCParticles.mass"].array(
            library="np", entry_stop=event_count
        )
        momentum_x = tree["MCParticles.momentum.x"].array(
            library="np", entry_stop=event_count
        )
        momentum_y = tree["MCParticles.momentum.y"].array(
            library="np", entry_stop=event_count
        )
        momentum_z = tree["MCParticles.momentum.z"].array(
            library="np", entry_stop=event_count
        )

    for event_index in range(event_count):
        particles.extend(
            build_particle_entries(
                pdg_codes[event_index],
                vertex_x[event_index],
                vertex_y[event_index],
                vertex_z[event_index],
                endpoint_x[event_index],
                endpoint_y[event_index],
                endpoint_z[event_index],
                mass_values[event_index],
                momentum_x[event_index],
                momentum_y[event_index],
                momentum_z[event_index],
            )
        )

    return particles


def load_leaf_values(tree, leaf_name):
    leaf = tree.GetLeaf(leaf_name)
    if not leaf:
        raise RuntimeError(f"Could not find leaf {leaf_name}")

    return [leaf.GetValue(index) for index in range(leaf.GetNdata())]


def load_mc_particles_with_root(input_file, num_events):
    root_file = ROOT.TFile.Open(str(input_file), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open {input_file}")

    try:
        tree = root_file.Get("events")
        if tree is None:
            raise RuntimeError("Could not find the 'events' tree")

        particles = []
        event_count = min(num_events, tree.GetEntries())

        # Read the requested events and merge all truth particle segments.
        for event_index in range(event_count):
            tree.GetEntry(event_index)

            particles.extend(
                build_particle_entries(
                    load_leaf_values(tree, "MCParticles.PDG"),
                    load_leaf_values(tree, "MCParticles.vertex.x"),
                    load_leaf_values(tree, "MCParticles.vertex.y"),
                    load_leaf_values(tree, "MCParticles.vertex.z"),
                    load_leaf_values(tree, "MCParticles.endpoint.x"),
                    load_leaf_values(tree, "MCParticles.endpoint.y"),
                    load_leaf_values(tree, "MCParticles.endpoint.z"),
                    load_leaf_values(tree, "MCParticles.mass"),
                    load_leaf_values(tree, "MCParticles.momentum.x"),
                    load_leaf_values(tree, "MCParticles.momentum.y"),
                    load_leaf_values(tree, "MCParticles.momentum.z"),
                )
            )

        return particles
    finally:
        root_file.Close()


def load_mc_particles(input_file, num_events):
    # Prefer uproot for simple array access, then fall back to ROOT.
    if uproot is not None:
        return load_mc_particles_with_uproot(input_file, num_events)

    return load_mc_particles_with_root(input_file, num_events)


def passes_filters(particle, charged_only, min_energy_GeV):
    start = particle["start"]
    end = particle["end"]
    if get_track_length(start, end) <= MIN_TRACK_LENGTH:
        return False
    if charged_only and not particle["is_charged"]:
        return False
    if particle["kinetic_energy_GeV"] < min_energy_GeV:
        return False

    return True


def build_tracks(particles):
    # Build TGeoTrack objects so the geometry carries the MC segments.
    tracks = []

    for index, particle in enumerate(particles):
        start = particle["start"]
        end = particle["end"]
        track = ROOT.TGeoTrack(index, particle["pdg"])
        track.SetLineColor(get_track_color(particle["pdg"]))
        track.SetLineWidth(2)
        track.AddPoint(start[0], start[1], start[2], 0.0)
        track.AddPoint(end[0], end[1], end[2], 1.0)
        tracks.append(track)

    return tracks


def build_polylines(particles):
    # Build plain 3D lines so JSROOT can draw the MC segments reliably.
    polylines = []

    for particle in particles:
        start = particle["start"]
        end = particle["end"]
        polyline = ROOT.TPolyLine3D(2)
        polyline.SetPoint(0, start[0], start[1], start[2])
        polyline.SetPoint(1, end[0], end[1], end[2])
        polyline.SetLineColor(get_track_color(particle["pdg"]))
        polyline.SetLineWidth(2)
        polylines.append(polyline)

    return polylines


def build_geometry_root(xml_path, output_file):
    # Run geoConverter to materialize the compact XML as a ROOT geometry file.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "geoConverter",
        "-compact2tgeo",
        "-input",
        str(xml_path),
        "-output",
        str(output_file),
    ]
    subprocess.run(command, check=True)


def clear_existing_tracks(geometry):
    tracks = geometry.GetListOfTracks()
    if tracks:
        tracks.Delete()


def get_hcal_volume(geometry):
    top_volume = geometry.GetTopVolume()
    if top_volume is None or top_volume.GetNdaughters() == 0:
        raise RuntimeError("Could not find the top detector placement")

    return top_volume.GetNode(0).GetVolume()


def capture_cutaway_state(hcal_volume):
    # Save the original slab sizes and node placements before the cutaway edit.
    original_dimensions = {}
    original_translations = []

    for index in range(hcal_volume.GetNdaughters()):
        node = hcal_volume.GetNode(index)
        volume = node.GetVolume()
        shape = volume.GetShape()
        volume_name = volume.GetName()

        if volume_name not in original_dimensions:
            original_dimensions[volume_name] = (
                shape.GetDX(),
                shape.GetDY(),
                shape.GetDZ(),
            )

        translation = node.GetMatrix().GetTranslation()
        original_translations.append((translation[0], translation[1], translation[2]))

    return original_dimensions, original_translations


def apply_x_cutaway(hcal_volume):
    # Shrink each slab in x and shift it left to expose the detector interior.
    unique_volumes = {}

    for index in range(hcal_volume.GetNdaughters()):
        node = hcal_volume.GetNode(index)
        volume = node.GetVolume()
        shape = volume.GetShape()
        volume_name = volume.GetName()

        if volume_name not in unique_volumes:
            unique_volumes[volume_name] = volume
            cut_dx = shape.GetDX() * CUT_FRACTION
            shape.SetBoxDimensions(cut_dx, shape.GetDY(), shape.GetDZ())
            volume.SetTransparency(CUTAWAY_TRANSPARENCY)

        matrix = node.GetMatrix()
        matrix.SetDx(-unique_volumes[volume_name].GetShape().GetDX())


def restore_cutaway_state(hcal_volume, original_dimensions, original_translations):
    # Restore the slab sizes and node placements after writing the cutaway canvas.
    restored_volumes = set()

    for index in range(hcal_volume.GetNdaughters()):
        node = hcal_volume.GetNode(index)
        volume = node.GetVolume()
        volume_name = volume.GetName()

        if volume_name not in restored_volumes:
            restored_volumes.add(volume_name)
            dx, dy, dz = original_dimensions[volume_name]
            volume.GetShape().SetBoxDimensions(dx, dy, dz)
            volume.SetTransparency(0)

        x_value, y_value, z_value = original_translations[index]
        matrix = node.GetMatrix()
        matrix.SetTranslation(x_value, y_value, z_value)


def write_geometry_with_tracks(output_file, tracks, polylines):
    # Store the geometry, the track objects, and both display canvases in one ROOT file.
    output_root_file = ROOT.TFile.Open(str(output_file), "UPDATE")
    if not output_root_file or output_root_file.IsZombie():
        raise RuntimeError(f"Could not open {output_file} for update")

    try:
        geometry = output_root_file.Get(GEOMETRY_KEY)
        if geometry is None:
            raise RuntimeError(f"Could not find the '{GEOMETRY_KEY}' geometry")

        clear_existing_tracks(geometry)
        for track in tracks:
            geometry.AddTrack(track)

        geometry.Write(GEOMETRY_KEY, ROOT.TObject.kOverwrite)

        # Keep both canvases in local scope until the file update is complete.
        full_canvas = ROOT.TCanvas(
            FULL_DISPLAY_NAME,
            FULL_DISPLAY_NAME,
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
        )
        geometry.GetTopVolume().Draw()
        for polyline in polylines:
            polyline.Draw("same")
        full_canvas.Write(FULL_DISPLAY_NAME, ROOT.TObject.kOverwrite)

        # Write a cutaway view so the line segments remain visible inside the slab.
        hcal_volume = get_hcal_volume(geometry)
        original_dimensions, original_translations = capture_cutaway_state(hcal_volume)
        apply_x_cutaway(hcal_volume)
        try:
            cutaway_canvas = ROOT.TCanvas(
                CUTAWAY_DISPLAY_NAME,
                CUTAWAY_DISPLAY_NAME,
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
            )
            geometry.GetTopVolume().Draw()
            for polyline in polylines:
                polyline.Draw("same")
            cutaway_canvas.Write(CUTAWAY_DISPLAY_NAME, ROOT.TObject.kOverwrite)
        finally:
            restore_cutaway_state(hcal_volume, original_dimensions, original_translations)
    finally:
        output_root_file.Close()


def main():
    # Resolve the inputs, build the geometry file, then overlay the MC particles.
    args = parse_args()
    geometry_id = args.geometry_id
    raw_run_path = resolve_raw_run_path(geometry_id)
    xml_path = resolve_geometry_xml_path(geometry_id)
    output_path = resolve_output_path(geometry_id, args.output)

    build_geometry_root(xml_path, output_path)
    particles = load_mc_particles(raw_run_path, args.num_events)
    filtered_particles = [
        particle
        for particle in particles
        if passes_filters(particle, args.charged_only, args.min_energy)
    ]
    tracks = build_tracks(filtered_particles)
    polylines = build_polylines(filtered_particles)
    write_geometry_with_tracks(output_path, tracks, polylines)
    print(
        f"Wrote {len(polylines)} MC particle lines from {raw_run_path.name} "
        f"into {output_path}"
    )


if __name__ == "__main__":
    main()
