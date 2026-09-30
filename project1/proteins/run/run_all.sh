#!/usr/bin/env bash
# Modality 2 path (mirrors molecules/run/run_all.sh):
#
#   bash run/run_all.sh                  # everything: dataset, both trainings, all samples/ablations, figures
#   STAGES=sample,metrics bash run/run_all.sh   # re-run sampling + evaluation on existing checkpoints
#   STAGES=figures bash run/run_all.sh   # redraw figures/*.png and figure_data.csv only
#
# Two trainings: the flow model (generation) and the family classifier (evaluation only, used
# to check whether a generated sequence matches its requested Pfam family). Sampling and the
# feasibility gate are inference-time, same as molecules: one flow checkpoint serves every
# ablation row below.
#
# Section 4.6 rows: abl_cfg (plain CFG, no gate), abl_global / abl_shuffled (gate controls),
# abl_gated (the innovation: per-residue gate), abl_none (unguided reference).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."      # every path below is relative to proteins/

STAGES="${STAGES:-data,train,sample,metrics,figures}"
stage() { [[ ",$STAGES," == *",$1,"* ]]; }
say()   { printf '\n== %s\n' "$*"; }

mkdir -p logs ckpt out figures
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > logs/LATEST
rm -f logs/EXIT_CODE

run() {
    set -e
    say "stages=$STAGES  commit=$(git rev-parse --short HEAD)"
    python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"

    if stage data; then
        say "Step 1  build dataset -> data/pfam_esm2.pt"
        python -u data/pfam.py --out data/pfam_esm2.pt --device cuda
    fi

    if stage train; then
        say "Step 2a  flow model -> ckpt/flow.pt"
        python -u train.py --model flow --data data/pfam_esm2.pt --out ckpt/flow.pt --epochs 100 --batch-size 64
        say "Step 2b  classifier (evaluator) -> ckpt/classifier.pt"
        python -u train.py --model classifier --data data/pfam_esm2.pt --out ckpt/classifier.pt --epochs 50 --clean-only
    fi

    if stage sample; then
        say "Step 3  samples, per-residue gate (the innovation) -> out/samples.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/samples.pt --n 1000 --steps 100 --w 2.0 --gate per_residue
        say "4.6 ablation  no guidance -> out/abl_none.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_none.pt --n 1000 --steps 100 --w 0.0
        say "4.6 ablation  CFG only, no gate -> out/abl_cfg.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_cfg.pt --n 1000 --steps 100 --w 2.0 --gate none
        say "4.6 ablation  per-residue gate (ours) -> out/abl_gated.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_gated.pt --n 1000 --steps 100 --w 2.0 --gate per_residue
        say "4.6 ablation  global gate (control) -> out/abl_global.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_global.pt --n 1000 --steps 100 --w 2.0 --gate global
        say "4.6 ablation  shuffled gate (control) -> out/abl_shuffled.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_shuffled.pt --n 1000 --steps 100 --w 2.0 --gate shuffled
    fi

    if stage metrics; then
        for NAME in samples abl_none abl_cfg abl_gated abl_global abl_shuffled; do
            say "Step 4  evaluate out/$NAME.pt -> out/$NAME.json"
            python -u metrics.py --samples "out/$NAME.pt" --data data/pfam_esm2.pt \
                --classifier-ckpt ckpt/classifier.pt --out "out/$NAME.json"
        done
    fi

    if stage figures; then
        say "Step 5  figures -> figures/*.png, *.pdf, figure_data.csv"
        python -u plot_results.py
    fi
    say "done"
}

( run ) 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date '+%F %T')" "$line"; done | tee "$LOG"
CODE=${PIPESTATUS[0]}
echo "$CODE" > logs/EXIT_CODE
echo "$(date '+%F %T') EXIT_CODE=$CODE" >> "$LOG"
exit "$CODE"
