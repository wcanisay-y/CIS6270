#!/bin/bash
#SBATCH -J p2_B
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 16:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaB.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaB.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}
C=ckpt

# `return 0` matters: without it the last missing checkpoint makes the
# function exit non-zero and `set -e` kills the script before anything runs.
have() { for f in "$@"; do [ -f "$C/$f.pt" ] && printf "%s " "$C/$f.pt"; done; return 0; }
ALL=$(have flat dms6 prop6 rand6 soft08 gamma1 hp2 murphy10 ar)
HIER=$(have dms6 prop6 rand6 soft08 gamma1 hp2 murphy10)
echo "arms found: $ALL"

# Stage 1. Every arm at one matched setting. Matched parameters, corpus,
# training budget, sampling precision and forward passes; the only thing
# that differs is the state space. rand6 is the decisive control: a random
# partition with the same group sizes as the chemical one. If rand6 matches
# dms6, the gain is the extra slot, not the chemistry.
echo "########## stage 1: every arm, matched setting ##########"
$PY -m hier.run_generate --ckpt $ALL --length 200 --steps 100 --n $N \
    --batch 24 --temps 1.0 --fam-temps 1.0 --out results/B_main.json

# Stage 2. Quality against diversity. The claim is that two temperatures
# move the whole frontier outward, not that one number improves, so the
# 2-D family-by-identity sweep must be compared with the 1-D flat curve.
echo "########## stage 2: quality-diversity frontier ##########"
$PY -m hier.run_generate --ckpt $C/flat.pt --length 200 --steps 100 --n $N \
    --batch 24 --temps 0.6 0.7 0.8 0.9 1.0 1.1 --out results/B_qd_flat.json
$PY -m hier.run_generate --ckpt $C/dms6.pt --length 200 --steps 100 --n $N \
    --batch 24 --temps 0.7 0.85 1.0 1.15 --fam-temps 0.7 0.85 1.0 \
    --out results/B_qd_dms6.json

# Stage 3. Compute budget.
echo "########## stage 3: NFE sweep ##########"
$PY -m hier.run_generate --ckpt $(have flat dms6 rand6) --length 200 \
    --steps 100 --n $N --batch 24 --temps 1.0 --fam-temps 1.0 \
    --nfe-sweep 25 50 200 --out results/B_nfe.json

# Stage 4a. Does the schedule-following sampler matter? The naive two-phase
# sampler writes every family before any identity, a state the training
# corruption never produces because the family mask is nested in the
# identity mask.
echo "########## stage 4a: scheduled vs out-of-distribution two-phase ######"
$PY -m hier.run_generate --ckpt $C/dms6.pt --length 200 --steps 100 --n $N \
    --batch 24 --temps 1.0 --fam-temps 1.0 --hier-sampler twophase \
    --rho-sweep 0.15 0.3 0.5 --out results/B_twophase.json

# Stage 4b. The commit ranking, as an ablation. Ranking by the clean
# probability and committing irrevocably collapses every arm: measured here
# it produced homopolymer runs of 109 to 113 residues out of 200 and
# diversity of 0.08. Everything else in this file uses the Gumbel ranking.
echo "########## stage 4b: clean-ranking ablation ##########"
$PY -m hier.run_generate --ckpt $(have flat dms6) --length 200 --steps 100 \
    --n $N --batch 24 --temps 1.0 --fam-temps 1.0 --rank clean \
    --out results/B_cleanrank.json

# Stage 5. Controllability: hand the model a family pattern from a real
# protein and measure compliance and foldability.
echo "########## stage 5: family-pattern conditioning ##########"
$PY -m hier.run_generate --ckpt $(have dms6 rand6 soft08) --length 200 \
    --steps 100 --n $N --batch 24 --temps 1.0 --fam-temps 1.0 \
    --conditioned --out results/B_cond.json

echo "########## score ##########"
for f in B_main B_nfe B_qd_flat B_qd_dms6 B_twophase B_cleanrank B_cond; do
  [ -f results/$f.json ] || continue
  echo "--- scoring $f ---"
  $PY -m common.run_score --gen results/$f.json \
      --out results/${f}_scored.json --fold-batch 8 --pppl-n 10
done
echo DONE
