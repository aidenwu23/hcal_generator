Use Latin Hypercube sampling to generate simple (backwards) hadronic calorimeter geometry used to train a surrogate.

Iteration naming:
- `training_0` is the initial observed dataset.
- `proposed_0` is the first surrogate-proposed batch generated from `training_0`.
- `training_N` comes from actual Geant4 results produced by running `proposed_(N-1)`.
- `proposed_N` is generated after retraining on the observed data available through `training_N`.

Also, the following converts a xml file into a viewable root file:
geoConverter -compact2tgeo -input geometries/generated/06529cd5/geometry.xml -output optimized.root
