#!/bin/bash
#SBATCH -J p2_cathtrain
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -a 0-2
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cathtrain_%a.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cathtrain_%a.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
CORPUS=data/clustered_cath/train_purged.txt

# ===================================================================
# The clustered-split retrain.
#
# Identical to run/train_all.sh in every hyperparameter -- 20,000 steps,
# batch 256, same seed, same architecture -- and differs ONLY in the training
# corpus, which has every BLAST relative (E <= 1e-3) of the 128 held-out
# pattern sources removed. Keeping the step count fixed rather than the epoch
# count means both runs see the same number of gradient updates; the purged
# run simply has a slightly smaller pool to draw them from.
#
# flat is included as the no-family-channel reference, because the
# conditioning gain is quoted against the two-level arm's own unconditional
# baseline and that baseline has to be re-established on this corpus.
# ===================================================================
if [ ! -s "$CORPUS" ]; then
  echo "FATAL: $CORPUS is missing or empty; run/cath_split.sh must run first" >&2
  exit 1
fi
echo "purged corpus: $(wc -l < $CORPUS) lines"
mkdir -p ckpt/clustered_cath

STEPS=${STEPS:-20000}
BATCH=${BATCH:-256}

case $SLURM_ARRAY_TASK_ID in
 0) ARGS="--arm flat";                                     TAG=flat ;;
 1) ARGS="--arm hier --partition dms6";                    TAG=dms6 ;;
 2) ARGS="--arm hier --partition dms6 --random-partition"; TAG=rand6 ;;
 *) echo "bad task"; exit 1 ;;
esac

echo "=== clustered retrain $SLURM_ARRAY_TASK_ID : $TAG : $ARGS ==="
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -m hier.train $ARGS --steps $STEPS --batch $BATCH \
    --train-file $CORPUS \
    --out ckpt/clustered_cath/$TAG.pt --tag ${TAG}_cath
