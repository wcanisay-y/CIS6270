#!/usr/bin/env bash
# Table 4.3: guided vs unguided, otherwise matched. Establishes controllability.
# Reports the target metric (property MAE) and the quality/diversity metrics
# alongside it, so guidance is not judged by the target score alone.
set -e
N=${N:-2000}; TARGET=${TARGET:-75}          # alpha in Bohr^3, upper tail of QM9
python train.py --model prop --out ckpt/propeval.pt --epochs ${EPOCHS:-50} --clean-only
for W in 0 1 2 4; do
  python sample.py --ckpt ckpt/flow_s0.pt --out results/43_cfg_w$W.pt \
    --n $N --w $W --target $TARGET
  python metrics.py --samples results/43_cfg_w$W.pt --prop-ckpt ckpt/propeval.pt \
    --out results/43_cfg_w$W.json
done
