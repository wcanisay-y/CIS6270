"""Smoke test: run 2-epoch training end-to-end.

This script runs through the full pipeline with minimal data to verify
everything works:
    1. Build dataset cache (limited to 500 sequences)
    2. Train flow model for 2 epochs
    3. Train classifier for 2 epochs
    4. Generate 50 samples
    5. Evaluate

Run with: python smoke_test.py
"""
import subprocess
import sys
from pathlib import Path


def run(cmd, description):
    """Run a command and check for errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD: {cmd}")
    print('='*60 + "\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        sys.exit(1)
    print(f"\nSUCCESS: {description}\n")


def main():
    # Ensure we're in the proteins directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    ckpt_dir = script_dir / "ckpt"
    out_dir = script_dir / "out"

    data_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    cache_path = data_dir / "pfam_esm2.pt"
    flow_ckpt = ckpt_dir / "flow_smoke.pt"
    classifier_ckpt = ckpt_dir / "classifier_smoke.pt"
    samples_path = out_dir / "samples_smoke.pt"
    metrics_path = out_dir / "metrics_smoke.json"

    # Step 1: Build dataset cache (limited)
    if not cache_path.exists():
        run(
            f'python "{script_dir / "data" / "pfam.py"}" '
            f'--out "{cache_path}" --limit 500 --device cpu',
            "Build dataset cache (500 sequences)"
        )
    else:
        print(f"Using existing cache: {cache_path}")

    # Step 2: Train flow model for 2 epochs
    run(
        f'python "{script_dir / "train.py"}" '
        f'--model flow --data "{cache_path}" --out "{flow_ckpt}" '
        f'--epochs 2 --batch-size 16 --layers 2 --n-heads 2',
        "Train flow model (2 epochs)"
    )

    # Step 3: Train classifier for 2 epochs
    run(
        f'python "{script_dir / "train.py"}" '
        f'--model classifier --data "{cache_path}" --out "{classifier_ckpt}" '
        f'--epochs 2 --batch-size 16 --clean-only',
        "Train classifier (2 epochs)"
    )

    # Step 4: Generate samples
    run(
        f'python "{script_dir / "sample.py"}" '
        f'--ckpt "{flow_ckpt}" --out "{samples_path}" '
        f'--n 50 --batch-size 16 --steps 20 --w 1.0 --gate per_residue',
        "Generate 50 samples with per-residue gating"
    )

    # Step 5: Evaluate
    run(
        f'python "{script_dir / "metrics.py"}" '
        f'--samples "{samples_path}" --data "{cache_path}" '
        f'--classifier-ckpt "{classifier_ckpt}" --out "{metrics_path}"',
        "Evaluate samples"
    )

    print("\n" + "="*60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Cache:      {cache_path}")
    print(f"  Flow model: {flow_ckpt}")
    print(f"  Classifier: {classifier_ckpt}")
    print(f"  Samples:    {samples_path}")
    print(f"  Metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
