# hcal_optimizer

Geometry optimizer for a generic layered calorimeter (HCal). Uses Geant4 simulations via DD4hep/ddsim to evaluate candidate geometries, trains a LightGBM surrogate on the observed results, and uses Bayesian optimization to propose improved geometry candidates.

*important* In this setup, `ddsim --gun.energy` appears to map to particle momentum in the run logs rather than total energy. The active simulation pipeline therefore uses `momentum_GeV` naming to avoid ambiguity.

## Notes and Assumptions

- Run metadata and raw CSVs store both `kinetic_energy_GeV` and `total_energy_GeV`.
- Compact training CSVs are keyed by `(geometry_id, kinetic_energy_GeV)`.
- BO scoring is evaluated across exact `kinetic_energy_GeV` points from `geometries/sweeps/bo_spec.yaml`.
- Energy resolution is normalized to `total_energy_GeV` for now.
- Sweep geometry thickness values are stored as bare numbers in `cm`.
- Generated geometry XML/JSON files store thicknesses as explicit DD4hep length expressions like `0.4*cm`.
- Downstream Python helpers normalize geometry lengths before using them.

## Setup

**Prerequisites:** DD4hep, Geant4, ROOT, EDM4hep/podio, CMake ≥ 3.16. These are managed via Spack.

```bash
./build.sh                 # build detector plugin and event processor
source setup.sh            # add build outputs to PATH and set library paths
```

**Python dependencies:** `pandas`, `pyyaml`, `scikit-learn`, `lightgbm`, `joblib`, `scipy`

## Pipeline

The optimization loop alternates between two scripts:

```
sweep YAML --> conductor.py --> ddsim --> event processor --> thickness-scaled calibration --> performance metrics

performance metrics --> orchestrator --> train surrogate/select best observed --> propose batch --> set up next sweep yaml
```

### 1. Generate an initial geometry set

Use Latin Hypercube Sampling to create a broad initial training set before the surrogate has been trained:

```bash
python3 geometries/generate_lhs.py --help
```

### 2. Run a simulation campaign

`conductor.py` takes one or more sweep YAMLs, materializes the geometries, simulates each one, scales a measured 4 mm MIP reference by segment scintillator thickness, and writes processed performance outputs.

```bash
python3 conductor.py --help
```

### 3. Train the surrogate and propose the next batch

`orchestrator.py` rebuilds training CSVs from processed results, trains the surrogate, proposes the next geometry batch, and identifies the current best observed geometry.

```bash
python3 orchestrator.py --help
```

Repeat steps 2–3, merging training CSVs across iterations, until the optimum converges.

## Thresholding

Each processed run gets its own `calibration.json`. `conductor.py` writes this file before the performance step by scaling a measured 4 mm scintillator MIP reference to the geometry being simulated.

The reference values live in `simulation/calibration/reference_mip_4mm.json`:

```
reference_thickness_mm = 4.0
reference_mpv_GeV = 0.0006193051107101145
```

For each of the three longitudinal segments, the run-local calibration does:

```
segment_mpv_GeV = reference_mpv_GeV * (segment_scintillator_thickness_mm / reference_thickness_mm)
threshold_GeV = mip_alpha * segment_mpv_GeV
```

`mip_alpha` is set from `conductor.py --mip-alpha` and defaults to `0.5`, so the threshold is half of the segment-scaled MIP MPV unless overridden.

The resulting `calibration.json` stores:

```
alpha
reference_geometry_id
reference_thickness_mm
reference_mpv_GeV
segment_scintillator_thicknesses_mm
mpvs
thresholds
```

`simulation/processing/performance.C` then applies those thresholds event by event:

- Each layer uses the threshold for its segment.
- A cell is counted as fired if `cell_energy >= threshold`.
- A layer is counted as fired if at least one cell in that layer fires.
- An event is counted as detected if at least one tile fires anywhere in the detector.

This means the current detection efficiency is a binary event-level metric:

```
detection_efficiency = detected_event_count / valid_event_count
```

The same pass through the events also records multiplicity information in `performance.json`:

- `tiles_mean`
- `tiles_std`
- `layers_mean`
- `layers_std`

## Geometry parameterization

The detector is a 10-layer segmented HCal with three longitudinal segments. The BO optimizes six continuous parameters (all in cm):

```
| Parameter         | Bounds     | Description                                  |
| `t_absorber_seg1` | [3.5, 4.5] | Absorber layer thickness, front segment      |
| `t_absorber_seg2` | [3.5, 4.5] | Absorber layer thickness, middle segment     |
| `t_absorber_seg3` | [3.5, 4.5] | Absorber layer thickness, back segment       |
| `t_scin_seg1`     | [0.3, 0.6] | Scintillator layer thickness, front segment  |
| `t_scin_seg2`     | [0.3, 0.6] | Scintillator layer thickness, middle segment |
| `t_scin_seg3`     | [0.3, 0.6] | Scintillator layer thickness, back segment   |
```

Fixed parameters: segment layer counts (3 / 3 / 4), spacer thickness (0.05 cm, Air), transverse dimensions (100 × 100 cm), front face position (−20 cm).

**BO objective:** maximize the configured surrogate metric from the active BO spec.

## CSV reference

### Compact (geometry-and-energy) CSV

One row per `geometry_id`, aggregated across repeated runs for each particle.

```
| Column                      | Description                                   |
| `geometry_id`               | 8-character hash of the parameter set         |
| `nLayers`                   | Total layer count                             |
| `seg{1,2,3}_layers`         | Layers per segment                            |
| `t_absorber_seg{1,2,3}`     | Absorber thickness per segment (cm)           |
| `t_scin_seg{1,2,3}`         | Scintillator thickness per segment (cm)       |
| `t_spacer`                  | Spacer/gap thickness (cm)                     |
| `{particle}_efficiency`     | Mean detection efficiency across runs         |
| `{particle}_efficiency_std` | Sample standard deviation across runs         |
| `{particle}_tiles_mean`     | Mean fired-tile count across runs             |
| `{particle}_layers_mean`    | Mean fired-layer count across runs            |
```

### Raw (run-level) CSV

One row per processed run.
```
| Column                  | Description                                  |
| `geometry_id`           | Geometry hash                                |
| `run_id`                | Unique run identifier                        |
| `gun_particle`          | Simulated particle species                   |
| `beam_mode`             | Beam configuration mode from `meta.json`     |
| `beam_label`            | Beam label from `meta.json`                  |
| `momentum_GeV`          | Gun momentum setting for monoenergetic runs  |
| `spectrum_id`           | Spectrum identifier for spectrum runs        |
| `spectrum_x_axis`       | Spectrum x-axis definition                   |
| `spectrum_x_min_GeV`    | Lower spectrum bound                         |
| `spectrum_x_max_GeV`    | Upper spectrum bound                         |
| `nLayers`               | Total layer count                            |
| `seg{1,2,3}_layers`     | Layers per segment                           |
| `t_absorber_seg{1,2,3}` | Absorber thickness per segment (cm)          |
| `t_scin_seg{1,2,3}`     | Scintillator thickness per segment (cm)      |
| `t_spacer`              | Spacer thickness (cm)                        |
| `detection_efficiency`  | Per-run detection efficiency                 |
| `tiles_mean`            | Mean fired-tile count for the run            |
| `layers_mean`           | Mean fired-layer count for the run           |
```

## Visualization

Convert a generated geometry XML to a ROOT-viewable TGeo file:

```bash
geoConverter -compact2tgeo -input geometries/generated/<geometry_id>/geometry.xml -output geometry.root
```

Create a visual for the MC simulated neutron shower given a geometry (requires an events.root):

```bash
python3 visuals/visualize.py --help
```

## Repository structure

```
hcal_optimizer/
--> conductor.py              # Simulation campaign runner
--> orchestrator.py           # Surrogate training and BO proposal runner
--> geometries/
    --> templates/            # DD4hep XML detector template
    --> src/                  # C++ DD4hep detector plugin
    --> generated/            # Materialized geometry XML/JSON files (hash-indexed)
    --> sweeps/               # Sweep YAML specs (bo_spec.yaml, LHS, proposed batches)
--> simulation/
    --> calibration/          # Muon threshold and particle response calibration scripts
    --> helpers/              # Geometry indexing, run planning, and execution steps
    --> processing/           # C++ event reduction and performance macros
--> analysis/
    --> geometry/             # Geometry comparison tools
    --> result_validation/    # Efficiency-vs-threshold scanning and validation
--> surrogate/
    --> csv_data/             # Training CSVs (raw, compact, merged, predictions)
    --> model/                # Saved surrogate model bundles (.joblib)
--> data/
    --> raw/                  # Raw EDM4hep ROOT files from ddsim
    --> processed/            # Per-run outputs (meta.json, calibration.json, performance.json)
    --> manifests/            # Run manifests (JSON and CSV)
--> csv_data/                 # Top-level result summary CSVs
```
