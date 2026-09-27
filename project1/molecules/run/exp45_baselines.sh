#!/usr/bin/env bash
# Table 4.5: the final method against the internal baseline. The three external
# methods (EDM 2022, GCDM 2023, SemlaFlow 2024) are QUOTED from their papers and
# must be labelled as quoted, not reproduced, with the protocol mismatch stated.
set -e
N=${N:-10000}; TARGET=${TARGET:-75}
python sample.py --ckpt ckpt/flow_s0.pt --out results/45_base.pt  --n $N --w 2.0 --gate none     --target $TARGET
python sample.py --ckpt ckpt/flow_s0.pt --out results/45_ours.pt  --n $N --w 2.0 --gate per_atom --target $TARGET
for R in base ours; do
  python metrics.py --samples results/45_$R.pt --prop-ckpt ckpt/propeval.pt --out results/45_$R.json
done
