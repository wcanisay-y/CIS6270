# Protein Flow Matching (Modality 2)

Transfer of feasibility-gated guidance from molecules to protein sequences in ESM-2 latent space.

---

## The Innovation: Feasibility-Gated Guidance

Standard CFG applies uniform guidance to all positions. The innovation: **modulate guidance per-position based on feasibility**.

```
v_i = v_null_i + w · g_i · (v_cond_i - v_null_i)
```

where `g_i = exp(-β · r_i²)` and `r_i` measures how unresolved position i is.

### How the Gate Works (matches molecules exactly)

| State | Residual r | Gate g | Guidance |
|-------|-----------|--------|----------|
| Resolved (confident) | r ≈ 0 | g ≈ 1 | **Full** guidance toward target |
| Unresolved (uncertain) | r large | g ≈ 0 | **Damped** - let flow settle first |

### Transfer from Molecules → Proteins

| | Molecules | Proteins |
|--|-----------|----------|
| **Feasibility signal** | Valency (correct # bonds) | ESM decoding confidence |
| **Residual** | \|soft_valency - allowed\| | 1 - max(softmax(lm_head)) |
| **Oracle** | Bond-length table (parameter-free) | Frozen ESM lm_head (parameter-free) |

Both are **zero-cost feasibility oracles** requiring no extra training.

---

## Setup

```powershell
cd CIS6270\project1\proteins
pip install torch transformers datasets numpy scipy
```

---

## Run Experiments

### Step 1: Build Dataset

```powershell
python data/pfam.py --out data/pfam_esm2.pt --device cuda
```

### Step 2: Train Models

```powershell
# Flow model
python train.py --model flow --data data/pfam_esm2.pt --out ckpt/flow.pt --epochs 100 --batch-size 64

# Classifier (for evaluation)
python train.py --model classifier --data data/pfam_esm2.pt --out ckpt/classifier.pt --epochs 50 --clean-only
```

### Step 3: Generate Samples

```powershell
# With feasibility gating (the innovation)
python sample.py --ckpt ckpt/flow.pt --out out/samples.pt --n 1000 --steps 100 --w 2.0 --gate per_residue
```

### Step 4: Evaluate

```powershell
python metrics.py --samples out/samples.pt --data data/pfam_esm2.pt --classifier-ckpt ckpt/classifier.pt
```

---

## Ablations (for Section 4.6)

```powershell
# No guidance
python sample.py --ckpt ckpt/flow.pt --out out/abl_none.pt --n 1000 --steps 100 --w 0.0

# CFG only (no gating)
python sample.py --ckpt ckpt/flow.pt --out out/abl_cfg.pt --n 1000 --steps 100 --w 2.0 --gate none

# Per-residue gating (innovation)
python sample.py --ckpt ckpt/flow.pt --out out/abl_gated.pt --n 1000 --steps 100 --w 2.0 --gate per_residue
```

**Gate modes**: `none`, `global`, `shuffled`, `per_residue`

---

## File Structure

```
proteins/
├── data/pfam.py       # Dataset loading + ESM encoding
├── flow_matching.py   # Flow model (transformer)
├── guidance.py        # CFG + feasibility gating  ← THE INNOVATION
├── train.py           # Training
├── sample.py          # Generation
└── metrics.py         # Evaluation
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| `family_accuracy` | Predicted family matches requested |
| `uniqueness` | Distinct / total |
| `novelty` | Not in training set |
| `diversity` | 1 - mean pairwise identity |
