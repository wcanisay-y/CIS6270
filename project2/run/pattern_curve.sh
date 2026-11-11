#!/bin/bash
#SBATCH -J p2_curve
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/pattern_curve.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/pattern_curve.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

# ===================================================================
# How accurate must a family pattern be, and how accurate is the model's own?
#
# This is the quantitative form of "predict the family label before the exact
# amino acid". The architecture already exists and already loses end to end
# (36.58 self-predicted vs 43.09 for no family channel vs 56.49 for a real
# pattern), so the question is whether the generator side is closable.
#
#   degradation curve  -> the accuracy budget a predictor must hit
#   selfpattern arm    -> what the current predictor is actually worth, with
#                         the identity phase held identical to the oracle arm
#   fam_head accuracy  -> whether the budget is reachable
#
# Patterns are the homology-clean set, so this is not re-measuring the leak.
# ===================================================================
echo "########## chemical partition: curve + self-pattern + diagnostics ####"
$PY -m hier.run_pattern_curve --ckpt ckpt/dms6.pt --partition dms6 \
    --pattern-file results/G_clean_patterns.json \
    --length 200 --steps 100 --n 96 --batch 24 \
    --fracs 0.0 0.05 0.1 0.2 0.3 0.5 0.75 1.0 \
    --out results/I_curve_dms6.json

echo "########## the random-partition control, endpoints only ##########"
# f=0 and f=1 on the random partition bound the curve's control arm: if the
# random partition shows the same curve shape, the curve is about constraint
# strength and not about chemistry.
$PY -m hier.run_pattern_curve --ckpt ckpt/rand6.pt --partition dms6 \
    --random-partition \
    --pattern-file results/G_clean_patterns.json \
    --length 200 --steps 100 --n 96 --batch 24 \
    --fracs 0.0 0.5 1.0 \
    --out results/I_curve_rand6.json

echo "########## score both ##########"
for F in results/I_curve_dms6.json results/I_curve_rand6.json; do
  $PY -m common.run_score --gen $F \
      --out ${F%.json}_scored.json --fold-batch 12 --pppl-n 8
done

echo DONE
