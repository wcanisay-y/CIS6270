#!/bin/bash
#SBATCH -J p2_sc
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/scorecheck.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/scorecheck.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

# Is the absorbing baseline a strawman? DPLM's own decoder scores the
# published 83.6 in our harness, but every absorbing arm lands near 55.
# Two candidate causes: (1) we rank commits by DPLM's Gumbel-perturbed
# score, which is its RE-MASKING criterion and may be noise-dominated as a
# commit ranking; (2) irrevocable commitment is simply much worse than
# re-predicting everything each step. This separates them.
echo "########## clean confidence vs gumbel-perturbed, absorbing ##########"
$PY -m parallel.run_generate --lengths 200 --steps 25 200 \
    --selectors confidence random --n 48 --batch 16 \
    --score-mode clean --out results/A_clean.json
$PY -m common.run_score --gen results/A_clean.json \
    --out results/A_clean_scored.json --fold-batch 12 --pppl-n 8
echo DONE
