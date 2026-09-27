#!/usr/bin/env bash
# Table 4.4: the innovation ablation. Every row uses the same checkpoint, the
# same seed, the same step count and the same w; only the gate changes.
#
#   none      Guided Flows CFG, constant strength           (selected base model)
#   rho>0     additive valency-constraint gradient          (lecture / HLTF form)
#   global    one gate per molecule                         (isolates localisation)
#   shuffled  per-atom gate permuted within the molecule    (mean-matched control)
#   per_atom  the proposed method
#
# The w-matched row is the control that rules out "the gate just lowered w":
# read mean_gate from the per_atom run and rerun the base model at w * mean_gate.
set -e
N=${N:-2000}; W=${W:-2.0}; TARGET=${TARGET:-75}; BETA=${BETA:-1.0}
for SEED in 0 1; do
  for G in none global shuffled per_atom; do
    python sample.py --ckpt ckpt/flow_s$SEED.pt --out results/44_${G}_s$SEED.pt \
      --n $N --w $W --gate $G --beta $BETA --target $TARGET --seed $SEED
    python metrics.py --samples results/44_${G}_s$SEED.pt --prop-ckpt ckpt/propeval.pt \
      --out results/44_${G}_s$SEED.json
  done
  python sample.py --ckpt ckpt/flow_s$SEED.pt \
    --out results/44_penalty_s$SEED.pt --n $N --w $W --rho 1.0 \
    --target $TARGET --seed $SEED
  python metrics.py --samples results/44_penalty_s$SEED.pt --prop-ckpt ckpt/propeval.pt \
    --out results/44_penalty_s$SEED.json
done
echo "now read mean_gate from results/44_per_atom_s0.json and run:"
echo "  python sample.py --ckpt ckpt/flow_s0.pt --out results/44_wmatched.pt \\"
echo "    --n $N --w \$(python -c \"print($W*MEAN_GATE)\") --gate none --target $TARGET"
