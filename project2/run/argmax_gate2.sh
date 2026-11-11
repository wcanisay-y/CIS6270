#!/bin/bash
#SBATCH -J p2_axg2
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/argmax_gate2.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/argmax_gate2.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

# ===================================================================
# The two configurations the first argmax-gate run did not cover.
#
# That run was a single setting: absorbing decoder, budgets 25 and 50, no
# warm-up, one seed family, 96 sequences per arm. Two gaps follow from our
# own measurements, and both favour the method, so they are the fair test.
#
#  A. THE REMASKING DECODER. It is the decoder DPLM ships and the one that
#     reaches the published 84.3 pLDDT, and it is where the original contact
#     rule was most harmful (-10.73 pooled). The repair has never been tried
#     there.
#
#  B. WARM-UP 0.5 WITH THE REPAIR. The completion is at the all-mask null
#     below about 25 % revealed and only becomes informative above 50 %, so
#     a run with no warm-up spends roughly half its gating decisions on a
#     dead signal. Deferring the gate to 50 % revealed puts every contact
#     call in the regime where the repaired signal exists. This is the
#     best case for the method that our data allow.
# ===================================================================

echo "########## A. repaired map inside the remasking decoder ##########"
$PY -m parallel.run_generate --lengths 200 --steps 25 50 100 \
    --selectors reparam_confidence reparam_contact reparam_contact_shuffled \
    --n $N --batch 8 --contact-mode external_argmax --contact-size 650 \
    --contact-every 1 --tau-mode quantile --tau-q 0.90 --warmup 0.0 \
    --out results/H_argmax_reparam_w0.0.json

echo "########## A2. same, warm-up 0.5 ##########"
$PY -m parallel.run_generate --lengths 200 --steps 25 50 100 \
    --selectors reparam_confidence reparam_contact reparam_contact_shuffled \
    --n $N --batch 8 --contact-mode external_argmax --contact-size 650 \
    --contact-every 1 --tau-mode quantile --tau-q 0.90 --warmup 0.5 \
    --out results/H_argmax_reparam_w0.5.json

echo "########## B. absorbing decoder, repaired map, warm-up 0.5 ##########"
$PY -m parallel.run_generate --lengths 200 --steps 25 50 100 \
    --selectors confidence contact contact_shuffled \
    --n $N --batch 8 --contact-mode external_argmax --contact-size 650 \
    --contact-every 1 --tau-mode quantile --tau-q 0.99 \
    --contact-min-revealed 0.5 \
    --out results/H_argmax_abs_w0.5.json

echo "########## score ##########"
for f in H_argmax_reparam_w0.0 H_argmax_reparam_w0.5 H_argmax_abs_w0.5; do
  $PY -m common.run_score --gen results/$f.json \
      --out results/${f}_scored.json --fold-batch 12 --pppl-n 8
done
echo DONE
