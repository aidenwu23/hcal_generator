# hcal_optimizer

Geometry optimizer for a generic layered calorimeter (HCal). Uses Geant4 simulations via DD4hep/ddsim to evaluate candidate geometries, trains a LightGBM surrogate on the observed results, and uses Bayesian optimization to propose improved geometry candidates. The objective is to maximize combined neutron and kaon0L detection efficiency subject to energy resolution constraints.

*important* In this setup, `ddsim --gun.energy` appears to map to particle momentum in the run logs rather than total energy. The active simulation pipeline therefore uses `momentum_GeV` naming to avoid ambiguity.

## Current results:
### Geometry IDs and identification
Each geometry has 10 total layers divided into 3 segments, with 3/3/4 layers in the front/middle/back segments.
```
        | Trained energy range | Iterations of BO | Notes
04e3fdfb| N/A                  | N/A              | nHCal
c2bbd5d3| 1 GeV                | 3                | Converged
57fc2ba4| 1-3 GeV              | 2                | Converged
```
Results are computed using average across 3000 events * 3 different seeds

### Three geometries and their thicknesses (cm)
```
         |Abs 1 |Scint 1|Abs 2 |Scint 2|Abs 3 |Scint 3|
04e3fdfb:|4.0   |0.4    |4.0   |0.4    |4.0   |0.4    |
c2bbd5d3:|3.5462|0.593  |3.7294|0.5514 |3.7378|0.587  |
57fc2ba4:|3.5503|0.5843 |3.53  |0.5762 |4.0639|0.59   |
```

### Geometry neutron efficiency across 1-3 GeV (max = 1)
```
         |1 GeV |2 GeV  |3 GeV |
04e3fdfb:|0.1043|0.4084 |0.6598|
c2bbd5d3:|0.2846|0.6544 |0.7798|
57fc2ba4:|0.2899|0.6484 |0.7946|
```

### Geometry kaon0L efficiency across 1-3 GeV
```
         |1 GeV |2 GeV  |3 GeV |
04e3fdfb:|0.2259|0.5618 |0.7483|
c2bbd5d3:|0.4878|0.7448 |0.8081|
57fc2ba4:|0.4857|0.7466 |0.8088|
```

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

**BO objective:** maximize `neutron_efficiency + kaon0L_efficiency`
**BO constraints:** `neutron_energy_resolution <= 1.0`, `kaon0L_energy_resolution <= 1.0`

## CSV reference

### Compact (geometry-and-energy) CSV

One row per `(geometry_id, kinetic_energy_GeV)`, averaged across repeated runs at the same geometry, energy, and particle.

```
| Column                             | Description                                   |
| `geometry_id`                      | 8-character hash of the parameter set         |
| `kinetic_energy_GeV`               | Shared energy axis used by the surrogate      |
| `nLayers`                          | Total layer count                             |
| `seg{1,2,3}_layers`                | Layers per segment                            |
| `t_absorber_seg{1,2,3}`            | Absorber thickness per segment (cm)           | 
| `t_scin_seg{1,2,3}`                | Scintillator thickness per segment (cm)       |
| `t_spacer`                         | Spacer/gap thickness (cm)                     |
| `{particle}_efficiency`            | Mean detection efficiency across seeds        |
| `{particle}_efficiency_std`        | Standard deviation of efficiency across seeds |
| `{particle}_energy_resolution`     | Mean energy resolution (sigma/E)              |
| `{particle}_energy_resolution_std` | Standard deviation of energy resolution       |
```

### Raw (run-level) CSV

One row per `(geometry, particle, kinetic_energy_GeV, seed)` combination.
```
| Column                 | Description                     |
| `geometry_id`          | Geometry hash                   |
| `run_id`               | Unique run identifier           |
| `gun_particle`         | Simulated particle species      |
| `kinetic_energy_GeV`   | Gun kinetic energy              |
| `total_energy_GeV`     | Gun total energy                |
| `muon_threshold_GeV`   | Applied detection threshold     |
| `detection_efficiency` | Per-run detection efficiency    |
| `energy_resolution`    | Per-run energy resolution       |
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
