 python3 conductor.py     --spec geometries/sweeps/sweep000.yaml     --process-bin ./build/bin/process     --ddsim ddsim     --root-bin root     --events-per-run 100     --gun-particle neutron     --gun-energy 5  --process-extra="--start-threshold 1e-2" --seed 67 --overwrite
{
  "detection_efficiency": 0.5,
  "eff_hi": 0.04975185951049943,
  "eff_lo": 0.04975185951049943,
  "energy_resolution": 0.18299408894504118,
  "frac_seg1": 0.21,
  "frac_seg2": 0.17,
  "frac_seg3": 0.06,
  "geometry_id": "d87dd9ea",
  "gun_energy_GeV": 5.0,
  "start_definition": "first_active_layer_above_threshold",
  "start_layer_median": 3.0
}

 python3 conductor.py     --spec geometries/sweeps/sweep000.yaml     --process-bin ./build/bin/process     --ddsim ddsim     --root-bin root     --events-per-run 100     --gun-particle neutron     --gun-energy 5  --process-extra="--start-threshold 5e-3" --seed 67 --overwrite
 {
  "detection_efficiency": 0.5,
  "eff_hi": 0.04975185951049943,
  "eff_lo": 0.04975185951049943,
  "energy_resolution": 0.18299408894504118,
  "frac_seg1": 0.42,
  "frac_seg2": 0.15,
  "frac_seg3": 0.03,
  "geometry_id": "d87dd9ea",
  "gun_energy_GeV": 5.0,
  "start_definition": "first_active_layer_above_threshold",
  "start_layer_median": 1.0
}

