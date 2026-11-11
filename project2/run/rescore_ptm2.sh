#!/bin/bash
#SBATCH -J p2_rescore2
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/rescore2.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/rescore2.err
set -uo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

# Second pass of the pTM re-score. The first attempt (27161676) finished
# C_revised_self and then died because it guessed the generation file names:
# the oracle runs write `*_gen.json`, the reparam runs do not. Inputs are
# therefore listed EXPLICITLY as in:out and all checked before ESMFold is
# loaded, and one bad file no longer takes the rest of the queue down.
#
# Order is by decisiveness: the 960-per-arm replication first, then the new
# fold-non-redundant CATH oracle, then the ad-hoc oracle, then the two
# remasking-decoder endpoints.
JOBS=(
  "results/D_confirm_w035T50.json:results/D_confirm_w035T50_scored2.json"
  "results/A_oracle_cath_gen.json:results/A_oracle_cath_scored2.json"
  "results/A_oracle_pdb_gen.json:results/A_oracle_pdb_scored2.json"
  "results/C_reparam_w0.0.json:results/C_reparam_w0.0_scored2.json"
  "results/C_reparam_w0.75.json:results/C_reparam_w0.75_scored2.json"
)
miss=0
for J in "${JOBS[@]}"; do
  IN="${J%%:*}"
  [ -f "$IN" ] || { echo "MISSING INPUT: $IN"; miss=1; }
done
[ "$miss" -eq 0 ] || { echo "aborting before model load"; exit 2; }

fail=0
for J in "${JOBS[@]}"; do
  IN="${J%%:*}"; OUT="${J##*:}"
  echo "### re-scoring $IN -> $OUT ###"
  if $PY -m common.run_score --gen "$IN" --out "$OUT" \
         --fold-batch 16 --pppl-n 8; then
    echo "OK $OUT"
  else
    echo "FAILED $IN"; fail=1
  fi
done
echo "DONE fail=$fail"
exit "$fail"
