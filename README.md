# Project 1

CIS 6270, Fall 2026. Continuous generative modelling across two modalities, with the same
proposed innovation — feasibility-gated guidance — transferred between them.

## Repository structure

```
project1/
  molecules/    Modality 1: QM9 3D coordinates. Flow matching + diffusion on a shared
                 EGNN backbone, classifier-free guidance, feasibility-gated guidance (§4.2-4.5)
  proteins/     Modality 2: Pfam protein sequences in ESM-2 latent space. Flow matching,
                 classifier-free guidance, the same gate transferred to this domain (§4.6)
  benchmarks/   External baselines for §4.5: EDM and GCDM reproduced from the authors'
                 pretrained QM9 checkpoints under one shared evaluator (not our method)
  requirements.txt   dependencies for molecules/ (see Setup)
```

Each of the three directories is self-contained (its own scripts, `run/` or top-level pipeline
script, `figures/`, and either `results/` or `out/`) and has its own README with full detail;
this file gives the setup and the map from paper section to script.

## Setup

Molecules and proteins have separate dependencies (different frameworks: `torch_geometric` +
`rdkit` vs `transformers` + `datasets`), each installed from its own directory:

```bash
pip install -r project1/requirements.txt              # molecules
pip install -r project1/proteins/requirements.txt      # proteins
```

`numpy` must stay below 2 for molecules: the torch 2.2.x wheels are built against the numpy 1.x
C ABI, and with numpy 2.x installed every tensor-to-array conversion fails — training survives
it, but plotting and the RDKit scoring in `metrics.py` do not.

`project1/benchmarks/` reproduces two external models and has its own, heavier setup (conda env,
cloned repos, downloaded checkpoints) — see `project1/benchmarks/README.md`. It is not needed to
reproduce our method, only the external comparison rows in §4.5.

## Running Modality 1

Everything is one script. Run it from `project1/molecules/`:

```bash
cd project1/molecules

SMOKE=1 bash run/run_all.sh    # 2 epochs on 2,000 molecules, ~2 min -- do this first
bash run/run_all.sh            # the real run
```

`SMOKE=1` exercises every stage end to end in about two minutes and is worth running
before committing a GPU, because it catches environment problems that would otherwise
surface hours into training.

**Run the real thing on a GPU.** `train.py` selects `cuda if available else cpu`, and on
CPU a single epoch over the full 100k training split takes about 17 minutes, so 100 epochs
is roughly 29 hours per model.

### What it does

| Stage | Output |
|---|---|
| `train` | three checkpoints: flow matching, diffusion, and the property-net evaluator |
| `trajectory` | `figures/fig_trajectory.png` — one molecule per model at four time points |
| `samples` | every table for results sections 4.2 through 4.5 |

Only the three trainings need a GPU for long. Guidance and the feasibility gate are
inference-time mechanisms — the gate adds no parameters and takes no backward pass — so
the whole `samples` stage is sampling against the checkpoints the `train` stage produced.

Subsets of the pipeline:

```bash
STAGES=samples bash run/run_all.sh       # re-run guidance and the ablation on existing checkpoints
STAGES=trajectory bash run/run_all.sh    # redraw the figure only
```

### Where every result is stored

All paths relative to `project1/molecules/`. Each sampling run writes a pair: a `.pt` of
the generated molecules and a `.json` of the metrics scored from it.

**Inputs and checkpoints**

| Path | What |
|---|---|
| `data/qm9.pt` | cached dataset, 137 MB, already built |
| `data/train_smiles.json` | novelty reference, 97,626 SMILES, auto-built on first scoring |
| `ckpt/flow_s0.pt` | flow-matching baseline |
| `ckpt/diff_s0.pt` | diffusion baseline |
| `ckpt/propeval.pt` | property-net evaluator |

**Figures**

| Path | Feeds |
|---|---|
| `figures/fig_trajectory.png` / `.pdf` | methods §3.7, presentation item 1f |

**§4.2 — flow matching vs diffusion**

| Path | Row |
|---|---|
| `results/42_flow_50.*` | flow, 50 Euler steps |
| `results/42_flow_100.*` | flow, 100 steps |
| `results/42_flow_200.*` | flow, 200 steps |
| `results/42_diff.*` | diffusion, K=1000 |

**§4.3 — guidance works** (gate off throughout; `w` swept)

| Path | Row |
|---|---|
| `results/43_cfg_w0.*` | unguided |
| `results/43_cfg_w1.*` | guided, low |
| `results/43_cfg_w2.*` | guided, default |
| `results/43_cfg_w4.*` | guided, high |

**§4.4 — innovation ablation** (all at `w = GATE_W`, only the gate varies)

| Path | Row | Rules out |
|---|---|---|
| `results/44_none.*` | selected base model, CFG only | — |
| `results/44_global.*` | one gate per molecule | per-atom resolution isn't needed |
| `results/44_shuffled.*` | gate permuted within the molecule | the gate is just a smaller effective `w` |
| `results/44_per_atom.*` | **the proposed method** | — |
| `results/44_penalty_r0.001.*` | additive constraint, HLTF form | the multiplicative form is what matters |
| `results/44_penalty_r0.01.*` | same, stronger | — |
| `results/44_wmatched.*` | run manually, see below | same, from the other direction |

`44_none` and `43_cfg_w2` are the same configuration, so their numbers should match exactly
— a free consistency check on the pipeline.

The `w`-matched control needs a value that only exists after `per_atom` has run. The script
prints the exact command at the end: read `mean_gate` from `results/44_per_atom.json` and
re-run the base model at `w × mean_gate`.

**§4.5 — comparison with recent methods** (10,000 samples, matching the baselines)

| Path | Row |
|---|---|
| `results/45_base.*` | internal base model |
| `results/45_ours.*` | our method |

The three external rows are not regenerated here. EDM and GCDM are reproduced at 10,000
samples with bootstrap intervals in `project1/benchmarks/` (see below); SemlaFlow is quoted
from its paper and must be labelled as quoted.

**What is and isn't tracked by git**

Checkpoints and the generated-sample `.pt` files are gitignored — they are large and
regenerable. The metric `.json` files and everything in `figures/` **are** tracked, because
the paper cites them directly.

To rebuild the QM9 cache from scratch, see `build_cache` in `project1/molecules/data/qm9.py`.

## Running Modality 2

Everything is one script, same shape as Modality 1. Run it from `project1/proteins/`:

```bash
cd project1/proteins

python smoke_test.py    # tiny end-to-end pass, CPU-friendly -- do this first
bash run/run_all.sh      # the real run: dataset, both trainings, sampling, figures
```

`run/run_all.sh` builds the Pfam-ESM2 dataset cache, trains the flow model and the (evaluation
-only) family classifier, samples the guidance ablation (§4.6), scores it, and draws
`figures/fig_training.*` and `figures/fig46_*.*`. Full detail — the dataset, the ablation rows,
file-by-file structure — is in `project1/proteins/README.md`.

## External baseline comparison (§4.5)

`project1/benchmarks/` reproduces two published E(3)-equivariant diffusion models, EDM and
GCDM, from the authors' pretrained QM9 checkpoints, 10,000 unconditional samples each scored by
one shared evaluator with 95% bootstrap intervals. Nothing here is retrained — it exists only
to give Modality 1's §4.5 comparison external rows that are reproduced rather than quoted.
Setup is heavier (clones the two repos, downloads a 2.4 GB checkpoint) and is independent of
molecules/proteins; see `project1/benchmarks/README.md` for reproduction steps.

## Connection to the paper

| Section | What | Produced by |
|---|---|---|
| 4.2 | flow matching vs. diffusion, Modality 1 | `molecules/run/run_all.sh` (`train` + `samples` stages) |
| 4.3 | classifier-free guidance sweep, Modality 1 | `molecules/run/run_all.sh` (`samples` stage) |
| 4.4 | feasibility-gate ablation, Modality 1 (**the innovation**) | `molecules/guidance.py` (`feasibility_gate`, `cfg`); run via `molecules/sample.py --gate per_atom` |
| 4.5 | comparison with recent methods | `molecules/results/45_*` (ours) + `benchmarks/` (EDM, GCDM, reproduced) + SemlaFlow (quoted) |
| 4.6 | feasibility gate transferred to protein sequences (**the innovation**) | `proteins/guidance.py` (`feasibility_gate`, `cfg`); run via `proteins/run/run_all.sh` |

**External code and data adapted, not authored here:**

- EGNN backbone (`molecules/egnn.py`) follows Satorras et al. 2021, *E(n) Equivariant Graph
  Neural Networks*.
- QM9 (`molecules/data/qm9.py`) is loaded via `torch_geometric`; bond-length reference tables in
  `molecules/guidance.py` and stability/validity definitions in `molecules/metrics.py` follow
  Hoogeboom et al. 2022 (EDM).
- The additive-constraint ablation control in `molecules/guidance.py` (`constraint_grad`)
  reproduces the form of Awasthi et al., *HLTF*, ICLR 2026 workshop, for comparison against the
  proposed multiplicative gate — see the docstring there for the exact differences.
- `project1/benchmarks/` runs EDM (github.com/ehoogeboom/e3_diffusion_for_molecules) and GCDM
  (github.com/BioinfoMachineLearning/Bio-Diffusion) from their authors' pretrained checkpoints;
  see `project1/benchmarks/README.md`.
- Modality 2 encodes sequences with the frozen ESM-2 (`facebook/esm2_t6_8M_UR50D`; Lin et al.
  2023, *Evolutionary-scale prediction of atomic-level protein structure with a language
  model*, Science) via HuggingFace `transformers`, and trains on
  `DanielHesslow/SwissProt-Pfam` via `datasets`.

Everything else — the flow-matching and diffusion models, classifier-free guidance, and the
feasibility gate itself (both modalities) — is this project's code.
