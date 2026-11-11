#!/bin/bash
#SBATCH -J p2_clean
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/clean_cond.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/clean_cond.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

# ===================================================================
# Conditioning, re-measured on a homology-clean pattern set.
#
# The original 48 patterns came from the upstream SwissProt dev split, which
# is NOT homology-clustered: 17 of 48 share more than half their 10-mers
# with a training sequence. The chemical arm gains +10.2 pLDDT on those
# sources and the random arm gains nothing, so a chemically coherent pattern
# appears to act as a retrieval key for a memorised fold. The headline +26.4
# is therefore inflated by an unknown amount.
#
# results/G_clean_patterns.json holds the 96 cleanest dev sequences of
# length >= 200: max 10-mer containment 0.026, mean 0.0075, against 0.37
# mean for the original set.
#
# Both checkpoints run on the SAME patterns, so the chemical-minus-random
# contrast stays paired per source protein.
# ===================================================================
echo "########## clean-set conditioning, 96 patterns ##########"
$PY -m hier.run_generate --ckpt ckpt/dms6.pt ckpt/rand6.pt \
    --length 200 --steps 100 --n 96 --batch 24 \
    --conditioned --pattern-file results/G_clean_patterns.json \
    --rho 0.3 --temps 1.0 --fam-temps 1.0 --rank gumbel \
    --hier-sampler scheduled --out results/G_cond_clean.json

echo "########## score ##########"
$PY -m common.run_score --gen results/G_cond_clean.json \
    --out results/G_cond_clean_scored.json --fold-batch 12 --pppl-n 8

echo DONE
