# Protein Flow Matching (Modality 2)

Transfer of flow matching + guidance innovation from molecules (Modality 1) to protein sequences in ESM-2 latent space.

---

## The Two Innovations Available

### 1. Feasibility-Gated Guidance (Main Innovation)

**Problem**: Standard CFG applies uniform guidance strength to all positions. Positions that are already "resolved" get pushed just as hard as uncertain positions, degrading quality at high guidance.

**Solution**: Per-position gate based on feasibility signal:
```
v_i = v_null_i + w · g_i · (v_cond_i - v_null_i)
```
where `g_i = exp(-β · r_i²)` and `r_i` measures how "unresolved" position i is.

| Modality | Feasibility Signal | What it measures |
|----------|-------------------|------------------|
| Molecules | Valency residual | Does atom have correct # bonds? |
| **Proteins** | 1 - max(softmax(lm_head(z))) | Can ESM confidently decode this residue? |

**Why decoding confidence works**: ESM-2 was trained on millions of real proteins. If a latent vector "looks like" a real amino acid, ESM's lm_head will confidently predict one amino acid. If it's malformed/noisy, the prediction will be uncertain (spread across many AAs). This gives us a free feasibility oracle without training anything extra.

### 2. Optimal Transport (OT) Flow Matching (Optional)

**Problem**: Standard flow matching randomly pairs noise samples with data samples. This can lead to crossing paths during training.

**Solution**: Use mini-batch optimal transport to find the pairing that minimizes total transport cost, leading to straighter paths.

```
Standard: z0[i] paired with z1[i] (random)
OT:       z0[π(i)] paired with z1[i] (optimal permutation π)
```

**Note**: OT and feasibility-gating are orthogonal. You can use either, both, or neither.

---

## Quick Verification (Optional)

To verify the code works before running full experiments:
```powershell
python smoke_test.py
```

---

## Full Experiment Pipeline

### Step 0: Environment Setup

```powershell
cd CIS6270\project1\proteins
pip install torch transformers datasets numpy scipy
```

Verify GPU:
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 1: Build Dataset Cache

Downloads SwissProt-Pfam, selects top 3 families, clusters sequences to prevent homolog leakage, encodes with frozen ESM-2.

```powershell
python data/pfam.py --out data/pfam_esm2.pt --device cuda
```

This takes ~10-20 minutes depending on GPU. Output: `data/pfam_esm2.pt`

### Step 2: Train Flow Model (Standard)

```powershell
python train.py --model flow --data data/pfam_esm2.pt --out ckpt/flow.pt `
    --epochs 100 --batch-size 64 --lr 1e-4 --layers 4 --n-heads 4
```

### Step 2b: Train Flow Model with OT (for comparison)

```powershell
python train.py --model flow --data data/pfam_esm2.pt --out ckpt/flow_ot.pt `
    --epochs 100 --batch-size 64 --lr 1e-4 --layers 4 --n-heads 4 --use-ot
```

### Step 3: Train Family Classifier (for evaluation)

```powershell
python train.py --model classifier --data data/pfam_esm2.pt --out ckpt/classifier.pt `
    --epochs 50 --batch-size 64 --clean-only
```

---

## Generation & Evaluation

### Generate Samples

**No guidance (baseline)**:
```powershell
python sample.py --ckpt ckpt/flow.pt --out out/samples_uncond.pt `
    --n 1000 --steps 100 --w 0.0
```

**CFG only (no gating)**:
```powershell
python sample.py --ckpt ckpt/flow.pt --out out/samples_cfg.pt `
    --n 1000 --steps 100 --w 2.0 --gate none
```

**CFG + Feasibility Gating (the innovation)**:
```powershell
python sample.py --ckpt ckpt/flow.pt --out out/samples_gated.pt `
    --n 1000 --steps 100 --w 2.0 --gate per_residue --beta 1.0
```

### Evaluate

```powershell
python metrics.py --samples out/samples_gated.pt --data data/pfam_esm2.pt `
    --classifier-ckpt ckpt/classifier.pt --out out/metrics_gated.json
```

### Metrics Reported

| Metric | Description |
|--------|-------------|
| `family_accuracy` | Fraction classified as requested Pfam family |
| `uniqueness` | Distinct sequences / total generated |
| `novelty` | Unique sequences not in training set |
| `diversity` | 1 - mean pairwise sequence identity |
| `mean_gate` | Average gate value (diagnostic) |

---

## Ablation Experiments (Required for Paper Section 4.4)

Run all gate modes to isolate the contribution of per-residue gating:

```powershell
# 1. No guidance baseline
python sample.py --ckpt ckpt/flow.pt --out out/abl_w0.pt --n 1000 --steps 100 --w 0.0

# 2. CFG only (gate=none)
python sample.py --ckpt ckpt/flow.pt --out out/abl_cfg.pt --n 1000 --steps 100 --w 2.0 --gate none

# 3. Global gate (average across sequence - tests position-specificity)
python sample.py --ckpt ckpt/flow.pt --out out/abl_global.pt --n 1000 --steps 100 --w 2.0 --gate global

# 4. Shuffled gate (permuted positions - mean-matched control)
python sample.py --ckpt ckpt/flow.pt --out out/abl_shuffled.pt --n 1000 --steps 100 --w 2.0 --gate shuffled

# 5. Per-residue gate (the innovation)
python sample.py --ckpt ckpt/flow.pt --out out/abl_perresidue.pt --n 1000 --steps 100 --w 2.0 --gate per_residue

# Evaluate all
foreach ($f in Get-ChildItem out/abl_*.pt) {
    Write-Host "Evaluating $f"
    python metrics.py --samples $f --data data/pfam_esm2.pt --classifier-ckpt ckpt/classifier.pt
}
```

**What each ablation tests**:
- `none` vs `per_residue`: Does gating help at all?
- `global` vs `per_residue`: Does position-specificity matter?
- `shuffled` vs `per_residue`: Is it the specific positions or just having varying weights?

---

## Guidance Strength Sweep

To find optimal guidance strength w:

```powershell
foreach ($w in 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0) {
    python sample.py --ckpt ckpt/flow.pt --out "out/sweep_w${w}.pt" `
        --n 500 --steps 100 --w $w --gate per_residue
    python metrics.py --samples "out/sweep_w${w}.pt" --data data/pfam_esm2.pt `
        --classifier-ckpt ckpt/classifier.pt
}
```

---

## OT vs Standard Flow Matching Comparison

If comparing OT flow matching:

```powershell
# Generate from both models
python sample.py --ckpt ckpt/flow.pt --out out/standard_fm.pt --n 1000 --steps 100 --w 2.0 --gate per_residue
python sample.py --ckpt ckpt/flow_ot.pt --out out/ot_fm.pt --n 1000 --steps 100 --w 2.0 --gate per_residue

# Evaluate both
python metrics.py --samples out/standard_fm.pt --data data/pfam_esm2.pt --classifier-ckpt ckpt/classifier.pt
python metrics.py --samples out/ot_fm.pt --data data/pfam_esm2.pt --classifier-ckpt ckpt/classifier.pt
```

---

## File Structure

```
proteins/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── smoke_test.py          # Quick verification (optional)
│
├── data/
│   ├── __init__.py
│   └── pfam.py            # Dataset: SwissProt-Pfam → ESM-2 latents
│
├── flow_matching.py       # FlowModel (transformer) + OT coupling
├── guidance.py            # CFG + feasibility gate via ESM lm_head
├── train.py               # Training loop (--use-ot for OT variant)
├── sample.py              # Generation with Euler integration
└── metrics.py             # Evaluation metrics
│
├── ckpt/                  # Saved models (created during training)
├── out/                   # Generated samples & metrics (created during eval)
└── data/                  # Cached dataset (created by pfam.py)
```

---

## Connecting to Paper Sections

| Paper Section | What to Run |
|--------------|-------------|
| 3.6 Transfer to Modality 2 | This entire codebase |
| 4.6 Transfer Results | Generate + evaluate, compare to Modality 1 |
| 4.4 Ablations (if needed) | Gate mode ablations above |

**What stays the same from Modality 1**:
- CFG formulation: `v = v_null + w·g·(v_cond - v_null)`
- Gate formula: `g_i = exp(-β·r_i²)`
- Ablation modes: none, global, shuffled, per_residue

**What changes for Modality 2**:
- Feasibility signal: valency → ESM decoding confidence
- Architecture: EGNN → Transformer (no geometry in latent space)
- Data: QM9 molecules → SwissProt-Pfam proteins

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
scipy>=1.10.0  # for OT coupling
```

---

## Troubleshooting

**Out of memory**: Reduce `--batch-size` to 32 or 16

**Slow training**: Ensure CUDA is being used (`--device cuda`)

**scipy not found**: `pip install scipy` (needed for OT coupling)

**HuggingFace rate limits**: Set `HF_TOKEN` environment variable
