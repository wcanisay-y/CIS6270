#!/bin/bash
#SBATCH -J p2_train
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -a 0-8%4
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/train_%a.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/train_%a.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
STEPS=${STEPS:-12000}
BATCH=${BATCH:-256}
PY=./venv2/bin/python

# Arm list. The decisive control is rand6: a random partition with the same
# group sizes as the chemical one. If it matches dms6, the gain is from the
# extra slot and not from the chemistry, and Idea B has no contribution over
# HDLM / MDM-Prime.
case $SLURM_ARRAY_TASK_ID in
 0) ARGS="--arm flat";                                              TAG=flat ;;
 1) ARGS="--arm hier --partition dms6";                             TAG=dms6 ;;
 2) ARGS="--arm hier --partition proposal6";                        TAG=prop6 ;;
 3) ARGS="--arm hier --partition dms6 --random-partition";          TAG=rand6 ;;
 4) ARGS="--arm hier --partition dms6 --soft-xi 0.8";               TAG=soft08 ;;
 5) ARGS="--arm hier --partition dms6 --gamma 1.0";                 TAG=gamma1 ;;
 6) ARGS="--arm hier --partition hp2";                              TAG=hp2 ;;
 7) ARGS="--arm hier --partition murphy10";                         TAG=murphy10 ;;
 8) ARGS="--arm ar";                                                TAG=ar ;;
 *) echo "bad task"; exit 1 ;;
esac

echo "=== arm $SLURM_ARRAY_TASK_ID : $TAG : $ARGS ==="
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -m hier.train $ARGS --steps $STEPS --batch $BATCH \
    --out ckpt/$TAG.pt --tag $TAG
