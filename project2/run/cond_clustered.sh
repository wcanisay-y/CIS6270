#!/bin/bash
#SBATCH -J p2_condclust
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cond_clustered.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cond_clustered.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
PAT=data/clustered/heldout_patterns.json

# ===================================================================
# The 2x2 that settles the leakage question.
#
#               chemical partition   random partition
#   original    A                    B
#   purged      C                    D
#
# All four cells use the SAME 128 held-out patterns, so:
#   (A-B) is the leaky chemical-minus-random contrast
#   (C-D) is the same contrast with every relative of every pattern source
#         removed from training
#   (A-B) - (C-D) is the contribution of having homologs in training
#
# The original checkpoints have never seen these particular patterns as a
# conditioning set, so row 1 is a fresh measurement rather than the +20.22
# being re-read. Unconditional baselines come out of the same runs, which is
# what makes the purged row interpretable despite its smaller corpus.
# ===================================================================
for F in "$PAT" ckpt/clustered/dms6.pt ckpt/clustered/rand6.pt; do
  [ -s "$F" ] || { echo "FATAL: missing $F" >&2; exit 1; }
done
$PY -I -c "import json;d=json.load(open('$PAT'));print('patterns: %d, purge %d (%.2f%%), sources with a relative %d'%(d['n_patterns'],d['n_purged'],100*d['purge_frac'],d['n_patterns_with_relative']))"

echo "########## row 1: ORIGINAL checkpoints, held-out patterns ##########"
$PY -m hier.run_generate --ckpt ckpt/dms6.pt ckpt/rand6.pt \
    --length 200 --steps 100 --n 128 --batch 24 \
    --conditioned --pattern-file $PAT \
    --rho 0.3 --temps 1.0 --fam-temps 1.0 --rank gumbel \
    --hier-sampler scheduled --out results/J_cond_orig.json

echo "########## row 2: PURGED checkpoints, same patterns ##########"
$PY -m hier.run_generate --ckpt ckpt/clustered/dms6.pt ckpt/clustered/rand6.pt \
    --length 200 --steps 100 --n 128 --batch 24 \
    --conditioned --pattern-file $PAT \
    --rho 0.3 --temps 1.0 --fam-temps 1.0 --rank gumbel \
    --hier-sampler scheduled --out results/J_cond_purged.json

echo "########## score both rows ##########"
for F in results/J_cond_orig.json results/J_cond_purged.json; do
  $PY -m common.run_score --gen $F \
      --out ${F%.json}_scored.json --fold-batch 12 --pppl-n 8
done

echo "########## the comparison ##########"
$PY -m analysis.clustered || true

echo DONE
