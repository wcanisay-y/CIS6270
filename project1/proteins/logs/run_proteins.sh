#!/usr/bin/env bash
# Runs project1/proteins of the both_modalities branch as its README lists the steps.
# The protein code itself is not modified. Every command below is the README's, with
# `python -u` for unbuffered logs and `--out` added to metrics.py so each table is saved.
# The two sampling runs marked EXTRA are not in the README: they are the remaining gate
# modes it names, run with the same settings so the ablation has all four rows.
set -uo pipefail

ROOT=/mnt/isilon/tan_lab/pany3/CIS6270-both-modalities
cd "$ROOT/project1/proteins"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
STAGES="${STAGES:-data,train,sample,metrics}"
stage() { [[ ",$STAGES," == *",$1,"* ]]; }
say()   { printf '\n== %s\n' "$*"; }

mkdir -p logs ckpt out
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
        say "Step 2b  classifier -> ckpt/classifier.pt"
        python -u train.py --model classifier --data data/pfam_esm2.pt --out ckpt/classifier.pt --epochs 50 --clean-only
    fi

    if stage sample; then
        say "Step 3  samples, per-residue gate -> out/samples.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/samples.pt --n 1000 --steps 100 --w 2.0 --gate per_residue
        say "Ablation  no guidance -> out/abl_none.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_none.pt --n 1000 --steps 100 --w 0.0
        say "Ablation  CFG only -> out/abl_cfg.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_cfg.pt --n 1000 --steps 100 --w 2.0 --gate none
        say "Ablation  per-residue gate -> out/abl_gated.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_gated.pt --n 1000 --steps 100 --w 2.0 --gate per_residue
        say "EXTRA  global gate -> out/abl_global.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_global.pt --n 1000 --steps 100 --w 2.0 --gate global
        say "EXTRA  shuffled gate -> out/abl_shuffled.pt"
        python -u sample.py --ckpt ckpt/flow.pt --out out/abl_shuffled.pt --n 1000 --steps 100 --w 2.0 --gate shuffled
    fi

    if stage metrics; then
        for NAME in samples abl_none abl_cfg abl_gated abl_global abl_shuffled; do
            say "Step 4  evaluate out/$NAME.pt -> out/$NAME.json"
            python -u metrics.py --samples "out/$NAME.pt" --data data/pfam_esm2.pt \
                --classifier-ckpt ckpt/classifier.pt --out "out/$NAME.json"
        done
    fi
    say "done"
}

( run ) 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date '+%F %T')" "$line"; done | tee "$LOG"
CODE=${PIPESTATUS[0]}
echo "$CODE" > logs/EXIT_CODE
echo "$(date '+%F %T') EXIT_CODE=$CODE" >> "$LOG"
exit "$CODE"
