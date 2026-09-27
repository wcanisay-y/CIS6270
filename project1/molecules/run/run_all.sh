#!/usr/bin/env bash
# Modality 1 path:
#
#   bash run/run_all.sh          # everything: 3 trainings, the figure, and every 4.2-4.5 table
#   SMOKE=1 bash run/run_all.sh  # 2 epochs on 2,000 molecules, ~2 min, proves the pipeline
#   STAGES=samples bash run/run_all.sh       # re-run guidance + the ablation on existing ckpts
#   STAGES=trajectory bash run/run_all.sh    # redraw the figure only
#
# Three trainings, and only three. Everything downstream of them -- guidance (4.3), the
# innovation ablation (4.4) and the baseline table (4.5) -- is SAMPLING, not training: the
# feasibility gate adds no parameters and takes no backward pass, so one flow checkpoint
# serves every one of those rows. See the `samples` stage below.
#
#   flow_s0.pt    the flow-matching baseline        -> 4.2, and the base model for 4.3/4.4/4.5
#   diff_s0.pt    the diffusion baseline            -> 4.2 only
#   propeval.pt   PropertyNet, --clean-only (t=1)   -> the property-MAE column in 4.3/4.4/4.5
#
# propeval.pt is the EVALUATOR, not part of generation. Guidance here is classifier-free, so
# nothing gradient-based reads this network during sampling; scoring guided samples with a
# regressor that also steered them would be circular.
#
# WHERE OUTPUT GOES (all paths relative to molecules/):
#   data/qm9.pt     the cached dataset, already built     (input, 137 MB)
#   ckpt/*.pt       checkpoints, best validation epoch    (train stage)
#   figures/*       fig_trajectory.png and .pdf           (trajectory stage)
#   results/*.pt    generated samples                     (samples stage)
#   results/*.json  metric tables                         (samples stage)
#
# RUN THIS ON A GPU. train.py selects `cuda if available else cpu`; measured on an M-series
# CPU a single epoch over the full 100k train split takes ~17 min, so 100 epochs is ~29 h per
# model. On an A100 the same run is a couple of hours. Verify with SMOKE=1 locally first.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."      # every path below is relative to molecules/

PYTHON="${PYTHON:-python}"
STAGES="${STAGES:-train,trajectory,samples}"
SEED="${SEED:-0}"

if [ "${SMOKE:-0}" = 1 ]; then
    EPOCHS="${EPOCHS:-2}"; PROP_EPOCHS="${PROP_EPOCHS:-2}"; LIMIT="--limit 2000"
    N="${N:-64}"; STEPS="${STEPS:-20}"; N_FINAL="${N_FINAL:-64}"
else
    EPOCHS="${EPOCHS:-100}"; PROP_EPOCHS="${PROP_EPOCHS:-50}"; LIMIT=""
    N="${N:-2000}"; STEPS="${STEPS:-200}"
    # 4.5 uses 10,000 to match the sample count EDM and GCDM were scored at on the
    # benchmarks branch; comparing 2,000 against their 10,000 would widen our intervals
    # for no reason. The 4.2-4.4 tables stay at N, since those are internal comparisons.
    N_FINAL="${N_FINAL:-10000}"
fi

# TARGET is alpha in Bohr^3; 75 sits in the upper tail of QM9. BETA is the gate sharpness.
TARGET="${TARGET:-75}"; BETA="${BETA:-1.0}"

# Two DIFFERENT uses of the guidance strength, which is easy to confuse:
#
#   WS         4.3 SWEEPS it, gate off, to show controllability:
#              w=0 unguided, then low / default / high.
#   GATE_W     4.4 HOLDS IT FIXED while only the gate varies, so the ablation changes
#              one thing at a time. It must be one of the values in WS, because the
#              gate=none row of 4.4 and the w=GATE_W row of 4.3 are the same run and
#              should produce identical numbers -- a free consistency check.
WS="${WS:-0 1 2 4}"
GATE_W="${GATE_W:-2.0}"

# Batch size. 64 is small for an A100 -- the dense EGNN at N=29 is launch-latency bound there,
# so raising this is the cheapest speedup available. Scale LR with it if you go much higher.
BATCH="${BATCH:-64}"; LR="${LR:-1e-4}"

# Strengths for the additive-constraint control (the HLTF form), swept rather than fixed.
# grad(-sum r^2) measures ~900 at the t=0 noise state, so the exp44 default of rho=1.0 adds
# ~900 to the velocity per Euler step and diverges to NaN. The control has to be tuned as
# carefully as the method it is controlling for, or the ablation is not a fair comparison.
RHOS="${RHOS:-0.001 0.01}"

mkdir -p ckpt results figures
stage() { [[ ",$STAGES," == *",$1,"* ]]; }
say()   { printf '\n== %s\n' "$*"; }

say "stages=$STAGES seed=$SEED epochs=$EPOCHS smoke=${SMOKE:-0}"
"$PYTHON" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# --- 1. the three trainings ----------------------------------------------------------------
# Single seed. exp42 originally looped SEED 0 1; at 36 hours that second seed is not
# affordable, so report single-seed numbers and disclose it under 4.1.3 rather than
# shipping empty tables.
if stage train; then
    say "1/3  flow matching  -> ckpt/flow_s$SEED.pt"
    "$PYTHON" train.py --model flow --out "ckpt/flow_s$SEED.pt" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" $LIMIT

    say "2/3  diffusion      -> ckpt/diff_s$SEED.pt"
    "$PYTHON" train.py --model diff --out "ckpt/diff_s$SEED.pt" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" $LIMIT

    say "3/3  property net   -> ckpt/propeval.pt   (evaluator, t=1 only)"
    "$PYTHON" train.py --model prop --out ckpt/propeval.pt --clean-only --seed "$SEED" \
        --epochs "$PROP_EPOCHS" --batch-size "$BATCH" --lr "$LR" $LIMIT
fi

# --- 2. the trajectory figure --------------------------------------------------------------
# One molecule per model, four time points, both rows started from the same prior draw.
# This is the raw material for the 3.7 overview figure and presentation item 1f.
if stage trajectory; then
    say "trajectory figure -> figures/fig_trajectory.png"
    "$PYTHON" trajectory.py --flow "ckpt/flow_s$SEED.pt" --diff "ckpt/diff_s$SEED.pt" \
        --steps "$STEPS" --seed "$SEED" --out figures/fig_trajectory
fi

# --- 3. samples and metrics (4.2 - 4.5) ----------------------------------------------------
# No training here -- this is where guidance and the innovation actually run. Both live in
# guidance.cfg(), called once per Euler step from sample_flow():
#
#   --w W            v = v_null + W (v_cond - v_null)          classifier-free guidance, 4.3
#   --gate per_atom  the same, with W scaled per atom by a      the innovation, 4.4
#                    feasibility gate read off xhat_1
#   --rho R          adds R * grad(-sum r^2) to the velocity    additive control, 4.4
#
# The null branch CFG interpolates from exists because train.py trained with --cond-drop 0.1.
# The gate itself trains nothing: no parameters, no backward pass.
#
# NOTE: the first metrics.py call builds data/train_smiles.json by running RDKit over the
# 100k-molecule training split for the novelty column. That is a one-off and takes minutes;
# it has not hung.
#
#   4.2  flow at three step counts vs diffusion -- the NFE column decides this comparison
#   4.3  guidance sweep w = 0, 1, 2, 4
#   4.4  the ablation: none / global / shuffled / per_atom, plus the additive rho control
#   4.5  the final base-vs-ours pair, to sit alongside the reproduced EDM and GCDM rows
if stage samples; then
    for S in 50 100 200; do
        say "4.2  flow, $S steps"
        "$PYTHON" sample.py --ckpt "ckpt/flow_s$SEED.pt" --out "results/42_flow_${S}.pt" \
            --n "$N" --steps "$S" --seed "$SEED"
        "$PYTHON" metrics.py --samples "results/42_flow_${S}.pt" --out "results/42_flow_${S}.json"
    done
    say "4.2  diffusion"
    "$PYTHON" sample.py --ckpt "ckpt/diff_s$SEED.pt" --out results/42_diff.pt --n "$N" --seed "$SEED"
    "$PYTHON" metrics.py --samples results/42_diff.pt --out results/42_diff.json

    # 4.3: unguided, then each guided strength. Gate off throughout -- this table is about
    # whether CFG controls the property at all, before the innovation enters.
    for GW in $WS; do
        say "4.3  guidance w=$GW  ($([ "$GW" = 0 ] && echo unguided || echo guided))"
        "$PYTHON" sample.py --ckpt "ckpt/flow_s$SEED.pt" --out "results/43_cfg_w$GW.pt" \
            --n "$N" --steps "$STEPS" --w "$GW" --target "$TARGET" --seed "$SEED"
        "$PYTHON" metrics.py --samples "results/43_cfg_w$GW.pt" --prop-ckpt ckpt/propeval.pt \
            --out "results/43_cfg_w$GW.json"
    done

    for G in none global shuffled per_atom; do
        say "4.4  ablation gate=$G"
        "$PYTHON" sample.py --ckpt "ckpt/flow_s$SEED.pt" --out "results/44_$G.pt" \
            --n "$N" --steps "$STEPS" --w "$GATE_W" --gate "$G" --beta "$BETA" \
            --target "$TARGET" --seed "$SEED"
        "$PYTHON" metrics.py --samples "results/44_$G.pt" --prop-ckpt ckpt/propeval.pt \
            --out "results/44_$G.json"
    done
    for R in $RHOS; do
        say "4.4  additive-constraint control (the HLTF form), rho=$R"
        "$PYTHON" sample.py --ckpt "ckpt/flow_s$SEED.pt" --out "results/44_penalty_r$R.pt" \
            --n "$N" --steps "$STEPS" --w "$GATE_W" --rho "$R" --target "$TARGET" --seed "$SEED"
        "$PYTHON" metrics.py --samples "results/44_penalty_r$R.pt" --prop-ckpt ckpt/propeval.pt \
            --out "results/44_penalty_r$R.json"
    done

    # The w-matched control rules out "the gate merely lowered w". It needs a number that only
    # exists after per_atom has run, so it is printed rather than executed.
    # 4.5: the headline table, at the baselines' sample count. Two rows only -- the internal
    # base model and the full method. The three external rows (EDM, GCDM reproduced on the
    # benchmarks branch; SemlaFlow quoted) are not regenerated here.
    say "4.5  base model, n=$N_FINAL"
    "$PYTHON" sample.py --ckpt "ckpt/flow_s$SEED.pt" --out results/45_base.pt \
        --n "$N_FINAL" --steps "$STEPS" --w "$GATE_W" --gate none \
        --target "$TARGET" --seed "$SEED"
    "$PYTHON" metrics.py --samples results/45_base.pt --prop-ckpt ckpt/propeval.pt \
        --out results/45_base.json

    say "4.5  our method, n=$N_FINAL"
    "$PYTHON" sample.py --ckpt "ckpt/flow_s$SEED.pt" --out results/45_ours.pt \
        --n "$N_FINAL" --steps "$STEPS" --w "$GATE_W" --gate per_atom --beta "$BETA" \
        --target "$TARGET" --seed "$SEED"
    "$PYTHON" metrics.py --samples results/45_ours.pt --prop-ckpt ckpt/propeval.pt \
        --out results/45_ours.json

    say "4.4  w-matched control -- read mean_gate from results/44_per_atom.json, then run:"
    echo "  $PYTHON sample.py --ckpt ckpt/flow_s$SEED.pt --out results/44_wmatched.pt \\"
    echo "      --n $N --steps $STEPS --w \$(python -c \"print($GATE_W * MEAN_GATE)\") \\"
    echo "      --gate none --target $TARGET --seed $SEED"
fi

say "done"
echo "checkpoints  ckpt/"
echo "figures      figures/fig_trajectory.png"
echo "samples      results/*.pt"
echo "metrics      results/*.json"
