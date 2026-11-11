#!/bin/bash
#SBATCH -J p2_rep
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/revised_reparam.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/revised_reparam.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

# The decisive test for the revised method. The absorbing decoder sits about
# 30 pLDDT below the one DPLM ships, so a rule that helps only there cannot
# matter for a real system. Here the warm-up runs inside the reparameterised
# remasking decoder, where the baseline reaches 80+, using the FREE
# self-attention signal that the deposited-structure measurement showed is
# the only one with information while masked.
for W in 0.0 0.35 0.5 0.75; do
  echo "########## reparam, self signal, warmup=$W ##########"
  $PY -m parallel.run_generate --lengths 200 --steps 25 50 100 200 \
      --selectors reparam_confidence reparam_contact reparam_contact_shuffled \
      --n $N --batch 16 --contact-mode self --contact-size 150 \
      --tau-mode quantile --tau-q 0.90 --warmup $W \
      --out results/C_reparam_w${W}.json
  $PY -m common.run_score --gen results/C_reparam_w${W}.json \
      --out results/C_reparam_w${W}_scored.json --fold-batch 12 --pppl-n 8
done
echo DONE
