#!/bin/bash
#SBATCH -J p2_famsig
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -a 0-1
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/famsig_%a.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/famsig_%a.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

# ===================================================================
# Can the family head be trained at all? Two variants against ckpt/dms6.pt.
#
# The measurement that motivates this: on ckpt/dms6.pt the family head's
# argmax accuracy is 0.3634 at EVERY identity-reveal level from 0 to 0.75,
# against a majority-class floor of 0.3633, and its cross-entropy is 1.5716
# nats against a marginal entropy of 1.662. The head is a constant predictor
# that learned the family marginal and 5 % more. Self-generated patterns are
# correspondingly 43.2 % aliphatic against 36.3 % in real patterns.
#
# The suspected cause is the family loss weighting. With gamma=2 the family
# channel is masked with probability t**2 and weighted 1/t**2, which is
# unbiased in expectation (the weighted contribution is L at every t) but has
# variance growing as 1/t**2, reaching 1e6 at the eps=1e-3 end. Worse, the
# informative regime -- few families masked, most identities revealed -- is
# exactly the small-t regime where almost no position is family-masked, so
# the head is trained mostly at t near 1 where everything is masked and the
# correct answer IS the marginal.
#
#   arm 0  gamma=1.0           families masked at the same rate as identities,
#                              so the head gets an identity-head-quality
#                              signal. Gives up "families lead identities" in
#                              training, which is fine for a diagnostic: the
#                              conditioned use case supplies the whole pattern
#                              anyway.
#   arm 1  clip the weight     keeps gamma=2 and the lead, caps 1/p_fam at 50,
#                              trading a little bias for bounded variance.
#
# If family accuracy stays at the floor in both, it is an information limit at
# this scale and "predict the family first" cannot be made to work here. If
# either rises, the published two-level negative is a training artefact and
# the arm has to be re-run before the idea can be dismissed.
#
# Full corpus, so this is isolated from the clustered-split experiment.
# ===================================================================
STEPS=${STEPS:-20000}
BATCH=${BATCH:-256}
mkdir -p ckpt/famsig

case $SLURM_ARRAY_TASK_ID in
 0) ARGS="--arm hier --partition dms6 --gamma 1.0";                TAG=g1 ;;
 1) ARGS="--arm hier --partition dms6 --fam-weight-clip 50";       TAG=clip50 ;;
 *) echo "bad task"; exit 1 ;;
esac

echo "=== family-signal arm $SLURM_ARRAY_TASK_ID : $TAG : $ARGS ==="
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -m hier.train $ARGS --steps $STEPS --batch $BATCH \
    --out ckpt/famsig/$TAG.pt --tag fam_$TAG

echo "=== family prediction accuracy for $TAG ==="
$PY -m hier.run_pattern_curve --ckpt ckpt/famsig/$TAG.pt --partition dms6 \
    --pattern-file results/G_clean_patterns.json \
    --length 200 --steps 100 --n 96 --batch 24 --diag-only \
    --out results/I_famsig_$TAG.json \
    --diag-out results/I_famsig_${TAG}_diag.json
