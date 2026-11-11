#!/bin/bash
#SBATCH -J p2_A2
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaA2.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaA2.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

# Stage 6. The first grid used a fixed coupling threshold of 0.3. A contact
# head on a mostly masked sequence emits probabilities near 0.01, so that
# threshold rejected 0.05 % of pairs: the rule never bound and the selector
# was silently identical to the confidence baseline. These arms set the
# threshold at a quantile of each sequence's own coupling values, so the rule
# rejects a controlled fraction of pairs whatever the signal's scale.
echo "########## stage 6: quantile threshold, 1 % of pairs rejected ##########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled anti_contact --n $N --batch 16 \
    --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --out results/A_q99.json

echo "########## stage 7: quantile threshold, 10 % of pairs rejected ##########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled --n $N --batch 16 \
    --contact-mode self --tau-mode quantile --tau-q 0.90 \
    --out results/A_q90.json

# Stage 8. Apply the rule only where stage 0 showed the signal exists.
echo "########## stage 8: gated to the late trajectory ##########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled --n $N --batch 16 \
    --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --contact-min-revealed 0.5 --out results/A_gated.json

# Stage 9. Length robustness.
echo "########## stage 9: second length ##########"
$PY -m parallel.run_generate --lengths 100 --steps 12 25 50 \
    --selectors confidence contact contact_shuffled random \
    --n $N --batch 16 --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --out results/A_L100.json

echo "########## score ##########"
for f in A_q99 A_q90 A_gated A_L100; do
  $PY -m common.run_score --gen results/$f.json \
      --out results/${f}_scored.json --fold-batch 8 --pppl-n 10
done
echo DONE

# Stage 10. The self-oracle: fold what the baseline produces and hand its
# own true contact map back to the selector on a second pass from the same
# seed. The tightest bound available on what any contact predictor could buy.
echo "########## stage 10: self-oracle ##########"
$PY -m parallel.self_oracle --length 200 --steps 12 25 50 --n $N \
    --batch 16 --out results/A_self_oracle.json
$PY -m common.run_score --gen results/A_self_oracle.json \
    --out results/A_self_oracle_scored.json --fold-batch 8 --pppl-n 10
echo DONE2
