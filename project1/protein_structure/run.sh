#!/usr/bin/env bash
# Train, sample, evaluate and plot the Ca-backbone diffusion arm. One script for the whole
# pipeline, so there is a single place to look and a single thing to keep correct.
#
#   bash run.sh                     # interactively, on the GPU this shell already holds
#   sbatch run.sh                   # as a batch job (submit from THIS directory: logs/ is relative)
#   SMOKE=1 bash run.sh             # 2 epochs on 2,000 proteins + 256 samples, a few minutes
#   EPOCHS=500 bash run.sh          # extend the existing run; --resume continues from epoch 200
#   STAGES=eval,figures bash run.sh # rescore and redraw only
#   FORCE=1 bash run.sh             # redo stages whose outputs already exist
#
# Stages whose outputs exist are SKIPPED, so the script is safe to re-run after an interruption
# and safe to run against the finished run of record: training resumes from checkpoint.pt and a
# completed run is a no-op. Nothing is ever deleted; FORCE=1 overwrites in place.
#
# Expect about 3.5 h end to end at the defaults: training 2.2 h, sampling 1.0 h, the rest minutes.
#
#SBATCH --job-name=ca_diff
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

MODULE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/pany3/anaconda3/envs/scleap/bin/python}"

# --- what to run -------------------------------------------------------------------------
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then
    RUN="${RUN:-smoke}"; EPOCHS="${EPOCHS:-2}"; SAMPLES="${SAMPLES:-256}"
else
    RUN="${RUN:-ca_sparse_k16}"; EPOCHS="${EPOCHS:-200}"; SAMPLES="${SAMPLES:-10000}"
fi
BATCH="${BATCH:-128}"
LR="${LR:-1e-4}"
HIDDEN="${HIDDEN:-256}"
LAYERS="${LAYERS:-9}"
K="${K:-16}"
TIMESTEPS="${TIMESTEPS:-1000}"
EMA="${EMA:-0.999}"
SEED="${SEED:-0}"
SAMPLE_BATCH="${SAMPLE_BATCH:-1024}"
WRITE_PDB="${WRITE_PDB:-20}"
STAGES="${STAGES:-test,prepare,train,sample,eval,figures}"
FORCE="${FORCE:-0}"

RUN_DIR="$MODULE/results/$RUN"
SAMPLE_FILE="$RUN_DIR/samples_${SAMPLES}.npz"
# Only the run of record owns the module's figures/ directory; anything else keeps its own, so a
# smoke run cannot overwrite the figures a report cites.
if [ "$RUN" = ca_sparse_k16 ]; then FIG_OUT="${FIG_OUT:-$MODULE/figures}"
else FIG_OUT="${FIG_OUT:-$RUN_DIR/figures}"; fi

# The structure cache is bulky regenerable input data and lives outside the repository.
export PROTEIN_DIFFUSION_ROOT="${PROTEIN_DIFFUSION_ROOT:-/mnt/isilon/tan_lab/pany3/protein_diffusion}"
CACHE="$PROTEIN_DIFFUSION_ROOT/cache"

mkdir -p "$MODULE/logs" "$RUN_DIR" "$FIG_OUT"

stage_wanted() { [[ ",$STAGES," == *",$1,"* ]]; }
have()         { [ "$FORCE" = 0 ] && [ -e "$1" ]; }
say()          { printf '\n== %s\n' "$*"; }

# --- environment -------------------------------------------------------------------------
# Unset, torch spawns one thread per host core inside the cgroup and spends the run context-switching.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
# Node-local /tmp on this cluster has filled and taken down whole sessions; keep scratch on Isilon.
export TMPDIR="${TMPDIR:-/mnt/isilon/tan_lab/pany3/tmp}"
mkdir -p "$TMPDIR"

# gres.conf on some nodes lists devices the node does not have, so CUDA_VISIBLE_DEVICES=<index>
# can resolve to a card this job does not hold. Pass GPU_UUID=GPU-xxxx to pin it by identity.
[ -n "${GPU_UUID:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU_UUID"

say "run=$RUN epochs=$EPOCHS samples=$SAMPLES stages=$STAGES force=$FORCE"
echo "host=$(hostname)  start=$(date -Is)"
echo "python=$PYTHON"
echo "run dir=$RUN_DIR"
echo "figures=$FIG_OUT"
nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,memory.used --format=csv 2>/dev/null \
    || echo "no nvidia-smi; the run will fall back to the CPU and take days"
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}  cuda {torch.cuda.is_available()}  "\
"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

# Warn if an interactive allocation will end before the work can finish. Everything launched from
# an Open OnDemand shell dies with the job, mid-epoch, with no checkpoint for that epoch.
if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
    left=$(( SLURM_JOB_END_TIME - $(date +%s) ))
    need=$(( SMOKE == 1 ? 900 : EPOCHS * 40 + SAMPLES / 3 + 600 ))
    printf 'allocation ends %s (%.1f h left); this run needs roughly %.1f h\n' \
        "$(date -d "@$SLURM_JOB_END_TIME" -Is)" "$(bc -l <<< "$left/3600")" "$(bc -l <<< "$need/3600")"
    [ "$left" -lt "$need" ] && echo "WARNING: the allocation is shorter than the expected runtime." \
        "Use sbatch, or lower EPOCHS/SAMPLES. Training resumes from checkpoint.pt if it is cut short."
fi

cd "$MODULE/scripts"

# --- 1. tests ------------------------------------------------------------------------------
# Seconds, and worth it: a silent masking, dtype or sampler-coefficient bug trains to a plausible
# loss curve and produces wrong structures hours later.
if stage_wanted test; then
    say "tests"
    "$PYTHON" -u test_protein.py
fi

# --- 2. structure cache --------------------------------------------------------------------
# Off by default because it needs outbound network for AlphaFold DB and takes about 45 minutes;
# it is also resumable in shards, so an interrupted fetch continues where it stopped.
if stage_wanted prepare && ! have "$CACHE/test_ca.npz"; then
    say "fetching and packing structures into $CACHE (about 45 min, needs outbound network)"
    "$PYTHON" -u prepare_structures.py 2>&1 | tee "$RUN_DIR/prepare.log"
fi
if [ ! -f "$CACHE/test_ca.npz" ]; then
    echo "no structure cache at $CACHE. Run with STAGES=prepare,... or set PROTEIN_DIFFUSION_ROOT." >&2
    exit 1
fi

# --- 3. training ---------------------------------------------------------------------------
# --resume always: it continues an interrupted run and is a no-op on a finished one, so this
# stage is idempotent and raising EPOCHS extends the existing run rather than restarting it.
if stage_wanted train; then
    say "training $EPOCHS epochs -> $RUN_DIR"
    "$PYTHON" -u train_protein.py \
        --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
        --hidden "$HIDDEN" --layers "$LAYERS" --k "$K" \
        --timesteps "$TIMESTEPS" --ema "$EMA" --seed "$SEED" \
        $([ "$SMOKE" = 1 ] && echo --smoke) \
        --resume --output "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/train.log"
fi

# --- 4. sampling ---------------------------------------------------------------------------
if stage_wanted sample && ! have "$SAMPLE_FILE"; then
    say "sampling $SAMPLES backbones ($TIMESTEPS denoising steps each)"
    "$PYTHON" -u sample_protein.py \
        --checkpoint "$RUN_DIR/checkpoint.pt" \
        --n "$SAMPLES" --batch-size "$SAMPLE_BATCH" --seed "$SEED" \
        --out "$SAMPLE_FILE" --write-pdb "$WRITE_PDB" 2>&1 | tee "$RUN_DIR/sample.log"
fi

# --- 5. evaluation -------------------------------------------------------------------------
# Every metric is computed identically for the samples and for the AlphaFold reference, so the
# reference row of metrics.json is the ceiling rather than a number quoted from a paper.
if stage_wanted eval && ! have "$RUN_DIR/metrics.json"; then
    say "scoring against the AlphaFold test reference"
    "$PYTHON" -u evaluate_protein.py \
        --samples "$SAMPLE_FILE" --out "$RUN_DIR/metrics.json" 2>&1 | tee "$RUN_DIR/eval.log"
fi

# --- 6. figures ----------------------------------------------------------------------------
if stage_wanted figures; then
    say "drawing figures -> $FIG_OUT"
    "$PYTHON" -u plot_results.py --run "$RUN_DIR" --samples "$SAMPLE_FILE" --out "$FIG_OUT"
fi

say "done  end=$(date -Is)"
echo "results:  $RUN_DIR"
echo "figures:  $FIG_OUT/fig0_overview.png"
echo "write-up: $MODULE/results/RESULTS.md"
