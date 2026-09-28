"""Minimal experiment comparing independent vs OT coupling for flow matching.

Trains two small models (2 epochs, 2000 molecules) with and without OT coupling,
samples at 10/25/50/100 Euler steps each, and reports molecule stability.

Usage:
    python run_ot_experiment.py --data data/qm9.pt
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path

import torch


def run_cmd(cmd, desc):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/qm9.pt")
    p.add_argument("--out-dir", default="ot_experiment")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--nf", type=int, default=64, help="smaller network for quick experiment")
    p.add_argument("--layers", type=int, default=4, help="smaller network for quick experiment")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    couplings = ["independent", "ot"]
    steps_list = [10, 25, 50, 100]

    # Train models
    ckpts = {}
    for coupling in couplings:
        ckpt_path = out_dir / f"flow_{coupling}.pt"
        ckpts[coupling] = str(ckpt_path)

        run_cmd([
            sys.executable, "train.py",
            "--model", "flow",
            "--data", a.data,
            "--out", str(ckpt_path),
            "--epochs", str(a.epochs),
            "--limit", str(a.limit),
            "--batch-size", str(a.batch_size),
            "--nf", str(a.nf),
            "--layers", str(a.layers),
            "--coupling", coupling,
            "--seed", str(a.seed),
        ], f"Training flow model with {coupling} coupling")

    # Sample and evaluate
    results = {coupling: {} for coupling in couplings}

    for coupling in couplings:
        for steps in steps_list:
            sample_path = out_dir / f"samples_{coupling}_steps{steps}.pt"
            metrics_path = out_dir / f"metrics_{coupling}_steps{steps}.json"

            run_cmd([
                sys.executable, "sample.py",
                "--ckpt", ckpts[coupling],
                "--out", str(sample_path),
                "--n", str(a.n_samples),
                "--steps", str(steps),
                "--seed", str(a.seed),
            ], f"Sampling from {coupling} model with {steps} Euler steps")

            run_cmd([
                sys.executable, "metrics.py",
                "--samples", str(sample_path),
                "--out", str(metrics_path),
                "--no-novelty",  # skip novelty for speed
            ], f"Evaluating {coupling} model samples ({steps} steps)")

            # Load results
            with open(metrics_path) as f:
                metrics = json.load(f)
            results[coupling][steps] = {
                "atom_stability": metrics["atom_stability"],
                "mol_stability": metrics["mol_stability"],
                "validity": metrics["validity"],
            }

    # Print summary table
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS: Independent vs OT Coupling")
    print("="*80)
    print(f"Training: {a.epochs} epochs, {a.limit} molecules, nf={a.nf}, layers={a.layers}")
    print(f"Sampling: {a.n_samples} molecules per configuration")
    print("="*80)

    print("\nMolecule Stability (higher is better):")
    print("-"*60)
    print(f"{'Steps':<10} {'Independent':<20} {'OT':<20} {'Diff':<10}")
    print("-"*60)
    for steps in steps_list:
        ind = results["independent"][steps]["mol_stability"]
        ot = results["ot"][steps]["mol_stability"]
        diff = ot - ind
        sign = "+" if diff > 0 else ""
        print(f"{steps:<10} {ind:<20.4f} {ot:<20.4f} {sign}{diff:.4f}")

    print("\nAtom Stability (higher is better):")
    print("-"*60)
    print(f"{'Steps':<10} {'Independent':<20} {'OT':<20} {'Diff':<10}")
    print("-"*60)
    for steps in steps_list:
        ind = results["independent"][steps]["atom_stability"]
        ot = results["ot"][steps]["atom_stability"]
        diff = ot - ind
        sign = "+" if diff > 0 else ""
        print(f"{steps:<10} {ind:<20.4f} {ot:<20.4f} {sign}{diff:.4f}")

    print("\nValidity (higher is better):")
    print("-"*60)
    print(f"{'Steps':<10} {'Independent':<20} {'OT':<20} {'Diff':<10}")
    print("-"*60)
    for steps in steps_list:
        ind = results["independent"][steps]["validity"]
        ot = results["ot"][steps]["validity"]
        diff = ot - ind
        sign = "+" if diff > 0 else ""
        print(f"{steps:<10} {ind:<20.4f} {ot:<20.4f} {sign}{diff:.4f}")

    # Save all results
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": vars(a),
            "results": results,
        }, f, indent=2)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
