#!/bin/bash
#SBATCH -J p2_axgate
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/argmax_gate.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/argmax_gate.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

# ===================================================================
# Close ask 4 empirically rather than by inference.
#
# The argmax completion repairs the external contact head: precision@L
# goes from 0.067 (at the all-mask null) to 0.737 at 75 % revealed, and the
# repaired external signal then BEATS the free self-attention signal at
# 50-75 % revealed. The obvious next question is whether a gate fed that
# repaired signal beats plain confidence.
#
# We expect not, because the oracle arms already hand the selector a
# deposited X-ray map -- the ceiling no predictor can exceed -- and the rule
# still loses. But "we expect not" is an inference, and the comment asked
# for the measurement, so here it is.
#
# Budgets 25 and 50 only: the completion is at the null below ~25 %
# revealed, so a long budget spends most of its steps in the regime where
# the repair does nothing, and each external call costs 7.88 denoiser
# passes.
# ===================================================================
echo "########## gate on the argmax-completed external map ##########"
$PY -m parallel.run_generate --lengths 200 --steps 25 50 \
    --selectors confidence contact contact_shuffled \
    --n $N --batch 8 --contact-mode external_argmax --contact-size 650 \
    --contact-every 1 --tau-mode quantile --tau-q 0.99 \
    --out results/F_argmax_gate.json

echo "########## the same arms on the plain masked external map ##########"
$PY -m parallel.run_generate --lengths 200 --steps 25 50 \
    --selectors contact contact_shuffled \
    --n $N --batch 8 --contact-mode external --contact-size 650 \
    --contact-every 1 --tau-mode quantile --tau-q 0.99 \
    --out results/F_plainext_gate.json

echo "########## score ##########"
for f in F_argmax_gate F_plainext_gate; do
  $PY -m common.run_score --gen results/$f.json \
      --out results/${f}_scored.json --fold-batch 12 --pppl-n 8
done
echo DONE
