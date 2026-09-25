#!/bin/bash
# Train a diffusion model on QM9. One script for every arm -- the arm is chosen with environment
# variables, so there is a single place to look and a single thing to keep correct.
#
#   sbatch run_ddpm.sbatch                                   # MLP + rotation augmentation
#   DENOISER=egnn CPUS=8 TIME=5:00:00 sbatch run_ddpm.sbatch # EGNN
#   GAUGE=canonical sbatch run_ddpm.sbatch                   # MLP, inertial-frame gauge
#   GAUGE=none sbatch run_ddpm.sbatch                        # MLP, QM9's native orientation
#   EPOCHS=2 LIMIT=2000 sbatch run_ddpm.sbatch               # quick smoke run
#
# Resuming: checkpoint.pt is written atomically every epoch. Resubmit the same command and
# --resume continues from it, including optimizer and EMA state. Raise EPOCHS to extend a run.
#
# Always validate first -- the submission filter rejects bad CPU/GPU ratios:
#   sbatch --test-only run_ddpm.sbatch
#
#SBATCH --job-name=ddpm
#SBATCH --partition=genoa-std-mem
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --account=nzh-wharton
#SBATCH --output=/vast/projects/nzh/wharton/project/yjiang/projects/project1_dm/logs/%x_%j.out

set -euo pipefail

PROJECT=/vast/projects/nzh/wharton/project/yjiang/projects/project1_dm
ENV="$PROJECT/envs/qm9diff"

DENOISER="${DENOISER:-mlp}"      # mlp | egnn
GAUGE="${GAUGE:-augment}"        # augment | canonical | none   (ignored for egnn)
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-64}"
LR="${LR:-2e-4}"
HIDDEN="${HIDDEN:-256}"
LAYERS="${LAYERS:-4}"
TIMESTEPS="${TIMESTEPS:-500}"
SAMPLES="${SAMPLES:-1000}"
LIMIT="${LIMIT:-}"               # empty = full dataset (130,831 after exclusions)
OUTDIR="${OUTDIR:-$PROJECT/results/ddpm_${DENOISER}}"

# The EGNN is equivariant by construction, so rotating its input teaches it nothing. Only the
# non-equivariant MLP has a gauge to choose.
GAUGE_FLAG=""
if [ "$DENOISER" = "mlp" ]; then
    case "$GAUGE" in
        augment)   GAUGE_FLAG="--augment" ;;
        canonical) GAUGE_FLAG="--canonical" ;;
        none)      GAUGE_FLAG="" ;;
        *) echo "GAUGE must be augment, canonical or none" >&2; exit 1 ;;
    esac
fi

mkdir -p "$PROJECT/logs" "$OUTDIR"

# Pin threads to the allocation. Unset, torch spawns one thread per host core (224) inside the
# cgroup and spends the run context-switching.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HOME="$PROJECT/data"
export HF_DATASETS_OFFLINE=1     # compute nodes have no outbound network; cache is pre-populated
export PYTHONUNBUFFERED=1

cd "$PROJECT/scripts"
echo "host=$(hostname) cores=${SLURM_CPUS_PER_TASK:-?} start=$(date -Is)"
echo "denoiser=$DENOISER gauge=$GAUGE epochs=$EPOCHS out=$OUTDIR"

# Seconds, and worth it: a silent equivariance or masking bug trains to a plausible loss curve
# and produces wrong samples hours later.
"$ENV/bin/python" -u modality1_test_baseline.py

"$ENV/bin/python" -u modality1_diffusion_model_baseline.py \
    --denoiser "$DENOISER" $GAUGE_FLAG \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --hidden "$HIDDEN" \
    --layers "$LAYERS" \
    --timesteps "$TIMESTEPS" \
    --samples "$SAMPLES" \
    ${LIMIT:+--limit "$LIMIT"} \
    --workers 2 \
    --resume \
    --uncharacterized "$PROJECT/data/raw/uncharacterized.txt" \
    --output "$OUTDIR"

echo "end=$(date -Is)"
