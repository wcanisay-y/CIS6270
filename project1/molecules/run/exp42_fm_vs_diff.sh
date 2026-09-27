#!/usr/bin/env bash
# Table 4.2: flow matching vs diffusion, unguided, matched backbone and data.
set -e
N=${N:-2000}
for SEED in 0 1; do
  python train.py --model flow --out ckpt/flow_s$SEED.pt --seed $SEED --epochs ${EPOCHS:-100}
  python train.py --model diff --out ckpt/diff_s$SEED.pt --seed $SEED --epochs ${EPOCHS:-100}
  # flow at several step counts: the NFE column is where this comparison is decided
  for S in 50 100 200; do
    python sample.py --ckpt ckpt/flow_s$SEED.pt --out results/42_flow_${S}_s$SEED.pt \
      --n $N --steps $S --seed $SEED
    python metrics.py --samples results/42_flow_${S}_s$SEED.pt \
      --out results/42_flow_${S}_s$SEED.json
  done
  python sample.py --ckpt ckpt/diff_s$SEED.pt --out results/42_diff_s$SEED.pt --n $N --seed $SEED
  python metrics.py --samples results/42_diff_s$SEED.pt --out results/42_diff_s$SEED.json
done
