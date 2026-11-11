#!/bin/bash
#SBATCH -J p2_rev
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/revised.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/revised.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

echo "##### stage 1: fold the 48 deposited sequences, compare map sources #####"
# Emits A_oracle_pdb.npz (deposited CB-CB), A_oracle_esmfold.npz (ESMFold
# CB-CB with the atom index fixed) and A_oracle_oxygen.npz (ESMFold O-O,
# which is what the earlier run actually used). Also rescores ESM-2 contact
# quality against the DEPOSITED map instead of a predicted one.
$PY -m parallel.run_pdb_oracle --pdb-file results/A_pdb_oracle_maps.npz \
    --outdir results --batch 4 --quality-n 12

echo "##### stage 2: oracle sampling grid, one run per map source #####"
# Binary maps, so the threshold is ABSOLUTE at 0.5 (reject a pair that is in
# contact). A quantile threshold is meaningless on a 2 %-dense binary matrix.
# The selector list and its order are identical across the three runs so the
# per-cell seeds line up and the confidence arm is directly comparable.
for SRC in pdb esmfold oxygen; do
  echo "----- map source: $SRC -----"
  $PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
      --selectors confidence contact_oracle contact_oracle_shuffled \
      --n $N --batch 16 --tau 0.5 --tau-mode absolute \
      --oracle-file results/A_oracle_${SRC}.npz \
      --out results/A_oracle_${SRC}_gen.json
  $PY -m common.run_score --gen results/A_oracle_${SRC}_gen.json \
      --out results/A_oracle_${SRC}_scored.json --fold-batch 12 --pppl-n 8
done

echo "##### stage 3: the revised method #####"
# Warm-up sweep, trigger-refresh policy sweep, group-size cap, and the free
# self-attention signal, all with cost charged through a measured ratio.
$PY -m parallel.run_revised --length 200 --n $N --batch 16 \
    --steps 25 50 100 --warmups 0.0 0.25 0.5 0.75 \
    --refresh-dist 0.15 --refresh-reveal 0.10 --refresh-max-age 8 \
    --tau-q 0.90 --out results/C_revised.json
$PY -m common.run_score --gen results/C_revised.json \
    --out results/C_revised_scored.json --fold-batch 12 --pppl-n 8

echo DONE
