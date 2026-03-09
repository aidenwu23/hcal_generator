#!/usr/bin/env python3

"""
Example usage:
    python3 bayesian_optimization.py \
        --init-training True \
        --rounds 2 \
        --sweep-template geometries/sweeps/template_sweep.yaml \
        --lhs-sweep geometries/sweeps/sweep000.yaml \
        --lhs-variants 5 \
        --tag-prefix lhs \
        --training-csv csv_data/training.csv \
        --model model/lgbm_surrogate.joblib \
        --bo-spec geometries/sweeps/bo_spec.yaml \
        --sweep-yaml geometries/sweeps/sweep_bo001.yaml \
        --processed-root data/processed \
        --pool 20000 \
        --bo-variants 3 \
        --seed 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
import shutil



def run_cmd(cmd):
    cmd = [str(x) for x in cmd]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run iterative surrogate-guided optimization loop.")
    ap.add_argument("--init-training", type=bool, default=False, help="Whether to perform initial training")
    ap.add_argument("--rounds", type=int, default=5, help="Number of BO rounds to run")
    ap.add_argument("--sweep-template", required=True, help="Template sweep YAML path")
    ap.add_argument("--lhs-sweep", required=True, help="Output sweep YAML path")
    ap.add_argument("--lhs-variants", type=int, required=True, help="Number of LHS variants")
    ap.add_argument("--tag-prefix", default="lhs", help="Variant tag prefix")
    ap.add_argument("--training-csv", required=True, default="csv_data/training.csv", help="Master CSV of all evaluated designs")
    ap.add_argument("--model", required=True, help="Path to write surrogate model .joblib")
    ap.add_argument("--bo-spec", help="Path to bo_spec.yaml", default="geometries/sweeps/bo_spec.yaml")
    ap.add_argument("--sweep-yaml", help="Path to sweep.yaml", default="geometries/sweeps/sweep_bo001.yaml")
    ap.add_argument("--processed-root", help="Path to hcal_generator/data/processed")
    #ap.add_argument("--workdir", help="Directory for per-round YAMLs / CSVs", default="bo_workdir")
    ap.add_argument("--pool", type=int, default=20000, help="Number of random/Sobol candidates to sample.")
    ap.add_argument("--bo-variants", type=int, default=5, help="Number of BO variants to output.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    #workdir = Path(args.workdir)
    #workdir.mkdir(parents=True, exist_ok=True)
    #shutil.copy(args.training-csv, args.workdir)
    #shutil.copy(args.model, args.workdir)

    if(args.init_training):
        print("=== Initial training ===")
        run_cmd([
            "python3", "geometries/generate_lhs.py",
            "--template", args.sweep_template,
            "--out", args.lhs_sweep,
            "--n", args.lhs_variants,
            "--seed", args.seed,
            "--tag-prefix", args.tag_prefix
        ])

        run_cmd([
            "python3", "conductor.py",
            "--spec", args.lhs_sweep,
            "--events-per-run", "2000",
            "--seed", args.seed,
            "--overwrite"
        ])

        run_cmd([
            "python3", "surrogate/aggregator.py",
            "--processed-root", args.processed_root,
            "--out", args.training_csv
        ])

    for r in range(args.rounds):
        print(f"\n=== BO round {r:03d} ===")


        # 4. Train surrogate on current master dataset
        run_cmd([
            "python3", "surrogate/train_surrogate.py",
            "--training-csv", args.training_csv,
            "--output-model", args.model
        ])
        
        # 1. Propose next batch
        #sweep_yaml = args.workdir / f"sweep_bo001.yaml" 
        run_cmd([
            "python3", "surrogate/propose_bo.py",
            "--model", args.model,
            "--spec", args.bo_spec,
            "--out", args.sweep_yaml,
            "--pool", str(args.pool),
            "--k", str(args.bo_variants),
            "--seed", str(args.seed + r),
        ])

        # 2. Run the sweep / simulations
        run_cmd([
            "python3", "conductor.py",
            "--spec", args.sweep_yaml,
            "--events-per-run", "2000",
            "--seed", str(67 + r),
            "--overwrite"
        ])

        # 3. Collect this round's results into CSV
        run_cmd([
            "python3", "surrogate/aggregator.py",
            "--processed-root", args.processed_root,
            "--out", args.training_csv
        ])

    #make script that will select the best geoometry parameters and performance 
    

if __name__ == "__main__":
    main()
