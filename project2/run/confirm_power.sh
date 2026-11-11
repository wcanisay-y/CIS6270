#!/bin/bash
#SBATCH -J p2_conf
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 16:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/confirm.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/confirm.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

# High-power confirmation of ONE cell. The sweeps use 48 sequences, where
# the smallest detectable effect is about 7 pLDDT, while the largest
# published gain for order selection on this model is 1.65. N=960 brings the
# detectable effect to roughly 1.6, which is the first setting in this study
# able to see the effect it is looking for.
N=${N:-960}
T=${T:-25}
MODE=${MODE:-self}
W=${W:-0.0}
DEC=${DEC:-absorbing}
TAG=${TAG:-conf}
# Fresh seed base: this is a REPLICATION of the cell the sweep
# selected as its maximum, so it must not reuse the sweep's seeds
# (the sweep ran base 0 with offsets up to ~32000).
SEED=${SEED:-500000}

if [ "$DEC" = "reparam" ]; then
  echo "### high-power confirmation, REMASKING decoder, T=$T warmup=$W N=$N ###"
  $PY -m parallel.run_generate --lengths 200 --steps $T \
      --selectors reparam_confidence reparam_contact reparam_contact_shuffled \
      --n $N --batch 16 --contact-mode $MODE --contact-size 150 \
      --tau-mode quantile --tau-q 0.90 --warmup $W --seed $SEED \
      --out results/D_confirm_${TAG}.json
else
  echo "### high-power confirmation, ABSORBING decoder, T=$T warmup=$W N=$N ###"
  $PY -m parallel.run_revised --length 200 --n $N --batch 16 --steps $T \
      --confirm --confirm-mode $MODE --confirm-warmup $W --tau-q 0.90 --seed $SEED \
      --out results/D_confirm_${TAG}.json
fi
$PY -m common.run_score --gen results/D_confirm_${TAG}.json \
    --out results/D_confirm_${TAG}_scored.json --fold-batch 16 --pppl-n 8
echo DONE
