#!/bin/bash
#SBATCH -J p2_cath
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 10:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cath.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cath.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

# Oracle on the CATH S40 set: one chain per CATH topology, stratified over
# the four classes. n=43 so each map is used exactly once per cell, which
# lets every generated sequence be attributed to a CATH class and the
# mechanism verdict be reported PER FOLD CLASS. Beta-rich folds carry ~60%
# more long-range contacts than alpha folds, so that is where a contact rule
# has the most to work with.
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors confidence contact_oracle contact_oracle_shuffled \
                anti_contact_oracle \
    --n 43 --batch 16 --oracle-file results/A_oracle_cath.npz \
    --tau 0.5 --tau-mode absolute --contact-mode self \
    --out results/A_oracle_cath_gen.json
$PY -m common.run_score --gen results/A_oracle_cath_gen.json \
    --out results/A_oracle_cath_scored.json --fold-batch 16 --pppl-n 8
echo DONE
