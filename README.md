# hcal_optimizer

Geometry optimizer for a segmented backward-facing hadronic calorimeter (HCal). Uses Geant4 simulations via DD4hep/ddsim to evaluate candidate geometries, trains a LightGBM surrogate on the observed results, and uses Bayesian optimization to propose improved geometry candidates. The objective is to maximize combined neutron and kaon0L detection efficiency subject to energy resolution constraints.

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
sweep YAML --> conductor.py --> ddsim --> event processor --> calibratoin --> performance

performance --> orchestrator --> train surrogate/select best observed --> propose batch --> set up next sweep yaml
```

### 1. Generate an initial geometry set

Use Latin Hypercube Sampling to create a broad initial training set before the surrogate has been trained:

```bash
python3 geometries/generate_lhs.py --help
```

### 2. Run a simulation campaign

`conductor.py` takes one or more sweep YAMLs, materializes the geometries, simulates each one, and writes processed performance outputs.

```bash
python3 conductor.py --help
```

### 3. Train the surrogate and propose the next batch

`orchestrator.py` rebuilds training CSVs from processed results, trains the surrogate, proposes the next geometry batch, and identifies the current best observed geometry.

```bash
python3 orchestrator.py --help
```

Repeat steps 2–3, merging training CSVs across iterations, until the optimum converges.

# Iteration naming

- `training_0` is the initial observed dataset (Latin Hypercube Sampling).
- `proposed_0` is the first surrogate-proposed batch generated from `training_0`.
- `training_N` comes from actual Geant4 results produced by running `proposed_(N-1)`.
- `proposed_N` is generated after retraining on the observed data available through `training_N`.

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
**BO constraints:** `neutron_energy_resolution <= 0.50`, `kaon0L_energy_resolution <= 0.50`

## CSV reference

### Compact (geometry-level) CSV

One row per geometry, averaged across seeds and energies.

```
| Column                             | Description                                   |
| `geometry_id`                      | 8-character hash of the parameter set         |
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

One row per (geometry, particle, seed) combination.
```
| Column                 | Description                     |
| `geometry_id`          | Geometry hash                   |
| `run_id`               | Unique run identifier           |
| `gun_particle`         | Simulated particle species      |
| `gun_energy_GeV`       | Total gun energy                |
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
--> processing/               # C++ EDM4hep event processor
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
