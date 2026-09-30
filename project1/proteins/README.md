# Protein Flow Matching (Modality 2)

Transfer of feasibility-gated guidance from molecules to protein sequences in ESM-2 latent
space: flow matching over per-residue ESM-2 hidden states, classifier-free guidance toward a
requested Pfam family, and the same feasibility gate as Modality 1, adapted to this domain.

## The innovation: feasibility-gated guidance

Standard CFG applies the same guidance strength to every residue. The innovation modulates it
per-residue by how "resolved" the flow's current endpoint estimate already looks:

```
v_i = v_null_i + w * g_i * (v_cond_i - v_null_i)      g_i = exp(-beta * r_i^2)
```

`r_i` is a feasibility residual read off the endpoint estimate, with no extra network trained:

| | Molecules (`../molecules/guidance.py`) | Proteins (`guidance.py`) |
|--|-----------|----------|
| Feasibility signal | valency (correct # bonds) | ESM-2 decoding confidence |
| Residual `r_i` | \|soft valency − allowed valency\| | 1 − max(softmax(lm_head(z))) |
| Oracle | bond-length table (parameter-free) | frozen ESM-2 `lm_head` (parameter-free) |

Confident residues (`r ≈ 0`, `g ≈ 1`) get full guidance; uncertain ones (`r` large, `g ≈ 0`) are
damped so the unconditional flow resolves basic sequence grammar first. See `guidance.py` for
the implementation and `GATE_MODES` for the ablation controls (`none`, `global`, `shuffled`,
`per_residue`).

## Setup

```bash
cd project1/proteins
pip install -r requirements.txt
```

Uses the frozen encoder/decoder `facebook/esm2_t6_8M_UR50D` (via `transformers`), downloaded
automatically on first use.

## Dataset

[DanielHesslow/SwissProt-Pfam](https://huggingface.co/datasets/DanielHesslow/SwissProt-Pfam)
(via `datasets`), filtered to the top 3 Pfam families and sequences of length 32-150, padded or
center-cropped to 64 residues. Sequences are clustered by k-mer similarity and split by cluster
(80/10/10) so near-duplicate sequences can't straddle train/test. See `data/pfam.py`.

```bash
python data/pfam.py --out data/pfam_esm2.pt --device cuda
```

This encodes every sequence with the frozen ESM-2 and caches the standardized per-residue
latents to `data/pfam_esm2.pt` (gitignored: regenerate it with the command above).

## Running everything

```bash
python smoke_test.py    # tiny end-to-end pass (500 sequences, 2 epochs, CPU-friendly) -- do this first
bash run/run_all.sh      # the real run: dataset, both trainings, sampling, figures
```

`run/run_all.sh` is the one script behind every result below; `smoke_test.py` is a separate,
faster sanity check with a smaller model (not meant to reproduce actual numbers). Subsets:

```bash
STAGES=sample,metrics bash run/run_all.sh    # re-run sampling + evaluation on existing ckpts
STAGES=figures bash run/run_all.sh           # redraw figures/*.png, *.pdf and figure_data.csv
```

### What it does

| Stage | Output |
|---|---|
| `data` | `data/pfam_esm2.pt` — cached ESM-2 latents |
| `train` | `ckpt/flow.pt` (generator), `ckpt/classifier.pt` (evaluator only) |
| `sample` | `out/samples.pt` plus the five `out/abl_*.pt` ablation rows (section 4.6) |
| `metrics` | the matching `out/*.json` metric tables |
| `figures` | `figures/fig_training.*`, `figures/fig46_*.*`, `figures/figure_data.csv` |

The classifier is an evaluation tool only (checks whether a generated sequence decodes to its
requested family) — it never guides generation, same separation as `propeval.pt` in Modality 1.

### Ablation (section 4.6)

All five rows sample from the same `ckpt/flow.pt` at `w=2.0` (`abl_none` at `w=0`), varying only
the gate:

| Run | Gate | Rules out |
|---|---|---|
| `abl_none` | — (`w=0`) | unguided reference |
| `abl_cfg` | `none` | whether CFG alone helps, before the innovation enters |
| `abl_global` | `global` | one gate per sequence — per-residue resolution isn't needed |
| `abl_shuffled` | `shuffled` | gate permuted within the sequence — mean-matched control |
| `abl_gated` | `per_residue` | **the proposed method** |

## File structure

```
proteins/
  data/pfam.py       dataset build: filter, cluster-split, encode with ESM-2
  flow_matching.py   flow model (small transformer over ESM-2 latents)
  guidance.py        CFG + feasibility gating  <- the innovation
  train.py           training (flow model, classifier)
  sample.py          generation, with guidance/gate flags
  metrics.py         family accuracy, uniqueness, novelty, diversity
  plot_results.py    figures + figure_data.csv from out/*.json and out/*.pt
  smoke_test.py      tiny end-to-end pass for a fast environment check
  run/run_all.sh     the whole pipeline (see above)
  scripts/           verification helpers, gitignored, not part of the method
```

## Metrics

| Metric | Description |
|--------|-------------|
| `family_accuracy` | fraction whose predicted family (by the classifier) matches the one requested |
| `uniqueness` | distinct valid sequences / total valid sequences |
| `novelty` | unique sequences absent from the training set / unique sequences |
| `diversity` | 1 − mean pairwise sequence identity (higher = more diverse) |

## Attribution

Adapts the frozen `facebook/esm2_t6_8M_UR50D` encoder/decoder (Lin et al., *Evolutionary-scale
prediction of atomic-level protein structure with a language model*, Science 2023) via
HuggingFace `transformers`, and the `DanielHesslow/SwissProt-Pfam` dataset via `datasets`. The
flow-matching setup, CFG and the feasibility gate itself are this project's code, transferring
the mechanism implemented for Modality 1 in `../molecules/guidance.py`.
