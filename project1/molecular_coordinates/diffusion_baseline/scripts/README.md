# Modality 1: 3D molecule coordinates — DDPM on QM9

Unconditional 3D molecule generation on [structure-epflai/qm9](https://huggingface.co/datasets/structure-epflai/qm9),
following the EDM recipe (Hoogeboom et al., ICML 2022) with a swappable denoiser.

## Files

| File | What it does |
|---|---|
| `modality1_data.py` | QM9 preprocessing: load, filter, split, pad, centre, optional canonical frame |
| `modality1_diffusion_model_baseline.py` | Denoisers, diffusion process, training, sampling, metrics |
| `modality1_test_baseline.py` | 13 correctness checks — run before any long job |
| `modality1_make_figures.py` | Result figures + `figure_data.csv` |
| `run_ddpm.sbatch` | Slurm job. One script for every arm, selected by environment variables |

## Quick start

```bash
PROJECT=/vast/projects/nzh/wharton/project/yjiang/projects/project1_dm
cd $PROJECT/scripts

$PROJECT/envs/qm9diff/bin/python modality1_test_baseline.py   # seconds
sbatch --test-only run_ddpm.sbatch                            # validate the Slurm request
sbatch run_ddpm.sbatch                                        # MLP baseline, ~30 min on 4 cores
```

Other arms — same script, different environment variables:

```bash
DENOISER=egnn CPUS=8 TIME=5:00:00 sbatch run_ddpm.sbatch   # equivariant denoiser
GAUGE=canonical sbatch run_ddpm.sbatch                     # MLP, inertial-frame gauge
GAUGE=none sbatch run_ddpm.sbatch                          # MLP, QM9's native orientation
EPOCHS=2 LIMIT=2000 sbatch run_ddpm.sbatch                 # smoke run, minutes
```

Figures, once a run finishes:

```bash
python modality1_make_figures.py --results $PROJECT/results/ddpm_mlp
```

## How the model works

**Forward process.** `z_t = α_t·z_0 + σ_t·ε`, variance-preserving, `polynomial_2` schedule,
T=500. Coordinates live on the **zero-centre-of-mass subspace** — noise is projected into it and
the denoiser's coordinate output is projected back. That is what makes the model translation
invariant, and breaking it silently invalidates the likelihood.

**Atom types** ride along as one-hot vectors scaled by 1/4 (EDM's `normalize_factors`). Putting
them on a smaller scale than coordinates makes the denoiser fix rough geometry before atom
identity; EDM's Table 10 shows this is worth ~35 points of molecule stability.

**Denoisers.** Both take `(z_x, z_h, t, mask)` and return `(eps_x, eps_h)`, so the diffusion
process, loss and sampler are shared and any comparison isolates the denoiser alone.

- `mlp` — flattens the padded molecule. **Not** rotation equivariant. Needs `--augment` or
  `--canonical` to cope with orientation.
- `egnn` — messages depend only on `‖xᵢ − xⱼ‖²`, so it is E(3)-equivariant by construction
  (this includes reflections, so it cannot represent chirality; true SE(3) needs GCPNet).

**Metrics** follow EDM: bonds inferred from interatomic distances via bond-length tables, then
atom/molecule stability, validity, uniqueness, novelty. RDKit is optional; without it you still
get the stability numbers.

## Results

`results/ddpm_baseline/` — MLP, 316,904 params, full dataset, 200 epochs, 27 min on 4 CPU cores.

| Metric | Value |
|---|---|
| Atom stability | 0.006 |
| Molecule stability | 0.000 |
| Validity | 0.971 ⚠️ |
| Uniqueness | 0.028 |
| Novelty | 1.000 ⚠️ |

**Validity and novelty are artefacts.** Generated atoms sit ~56 Å apart against 1.12 Å in the
data, so no bond is ever inferred, every atom becomes its own fragment, and RDKit sanitises a
lone `"C"` as valid. The honest signals are atom stability 0.006 and molecule stability 0.000.

**The implementation is not at fault.** Replacing the denoiser with an oracle that returns the
exact noise, and running the same sampler, reproduces the data scale exactly (Rg 2.323 vs 2.32,
atom-type accuracy 1.000). The schedule, subspace projection and ancestral update are correct.

**The MLP is.** A non-equivariant denoiser must learn every shape in every orientation, and with
random rotation augmentation the lowest-loss strategy is to predict the orientation-average — a
spherical blob. The reverse process then multiplies the residual error by α₀/α_T ≈ 294 over 500
steps. The EGNN arm is the comparison this motivates.

## Comparability

These numbers are **not** directly comparable to published EDM/GCDM results:

1. **Split.** This dataset ships a single `train` split that includes the 3,054 molecules which
   failed QM9's geometry consistency check. These scripts exclude those and make their own
   deterministic 80/10/10 split. Published numbers use the Cormorant split (100,000/17,748/13,083).
2. **No charge feature.** EDM carries atomic number as a sixth node feature; this uses one-hot
   types only.

Compare against this baseline by re-running this script with one thing changed, never against a
published number.

## Environment and cluster notes

`$PROJECT/envs/qm9diff` — torch 2.11.0+cu128, datasets 5.0.1, rdkit, numpy. It lives on `/vast`,
not `$HOME`, because home sits near its 250K inode limit and a Python env is tens of thousands of
small files.

- The cluster supports **CUDA 12.8 only**. A bare `pip install torch` pulls a CUDA 13 build that
  fails on the driver, after which training silently falls back to CPU. `cu126` wheels lack
  `sm_100` and fail on B200.
- `b200-mig45` requires **exactly 6 CPUs per GPU** and caps at **2 days**. The submission filter
  rejects anything else, so always `sbatch --test-only` first.
- **`--mem` is rejected** on CPU partitions; memory follows the core count (`genoa-std-mem`:
  5632 MB/CPU). Ask for cores.
- **`MaxCpuPerAccount`** is shared lab-wide. Small requests start while large ones queue — 4
  cores started immediately when 16 had been blocked for hours.
- Cache the dataset on the **login node**; compute nodes have no outbound network and jobs run
  with `HF_DATASETS_OFFLINE=1`.
