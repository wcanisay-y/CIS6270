# Project 1

CIS 6270, Fall 2026. Continuous generative modelling across two modalities.

Modality 1 is QM9 3D molecular coordinates, in `project1/molecules/`: a continuous
flow-matching model and a diffusion model on a shared EGNN backbone, classifier-free
guidance, and the proposed feasibility-gated guidance.

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

The three external rows are not regenerated here. EDM and GCDM live on the `benchmarks`
branch, already reproduced at 10,000 samples with bootstrap intervals; SemlaFlow is quoted
from its paper and must be labelled as quoted.

**What is and isn't tracked by git**

Checkpoints and the generated-sample `.pt` files are gitignored — they are large and
regenerable. The metric `.json` files and everything in `figures/` **are** tracked, because
the paper cites them directly.

## Setup

```bash
pip install -r project1/requirements.txt
```

`numpy` must stay below 2. The torch 2.2.x wheels are built against the numpy 1.x C ABI,
and with numpy 2.x installed every tensor-to-array conversion fails — training survives it,
but plotting and the RDKit scoring in `metrics.py` do not.

To rebuild the QM9 cache from scratch, see `build_cache` in `project1/molecules/data/qm9.py`.

## Other modalities and baselines

On the `benchmarks` branch, not yet merged:

- `project1/benchmarks/` — EDM and GCDM run from the authors' pretrained QM9 checkpoints,
  10,000 samples each under one shared evaluator with bootstrap intervals. These are the
  external comparisons for section 4.5, reproduced rather than quoted.
- `project1/protein_structure/` — Modality 2, sparse SE(3) diffusion over protein Cα
  backbones.
