#!/bin/bash
#SBATCH -J p2_prof
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/prof_asks.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/prof_asks.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

# ===================================================================
# ASK 4: contact accuracy on the ACTUAL ARGMAX COMPLETIONS.
# The original table fed a masked sequence to the contact head. Filling the
# masks with the generator's argmax hands the head a complete sequence,
# which is in distribution for ESM-2. If that restores precision, the
# "external planners fail on masked input" claim needs heavy qualification
# and the method has a live repair.
# ===================================================================
echo "########## smoke: 2 chains, fail fast before the long arms ##########"
$PY -m parallel.argmax_contacts --oracle-file results/A_oracle_cath.npz \
    --n 2 --reveals 0.50 --out results/F_argmax_smoke.json

echo "########## ask 4a: argmax completion, CATH S40 (fold-non-redundant) #####"
$PY -m parallel.argmax_contacts --oracle-file results/A_oracle_cath.npz \
    --n 43 --reveals 1.0 0.75 0.50 0.25 0.10 \
    --out results/F_argmax_cath.json

echo "########## ask 4b: same on the ad-hoc PDB set, for comparability ########"
$PY -m parallel.argmax_contacts --oracle-file results/A_oracle_pdb.npz \
    --n 48 --reveals 1.0 0.75 0.50 0.25 0.10 \
    --out results/F_argmax_pdb.json

# ===================================================================
# ASK 5: the control ladder on one frozen backbone, at the full budget
# ladder. PUNT and spacing existed but were scored at only the two smallest
# budgets, so neither was ever benchmarked against confidence where it
# matters.
# ===================================================================
echo "########## ask 5: confidence / spacing / PUNT / attention-graph ########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 200 \
    --selectors confidence distance punt contact contact_shuffled random \
    --n $N --batch 16 --contact-mode self \
    --tau-mode quantile --tau-q 0.99 --out results/F_ladder.json

echo "########## score the ladder ##########"
$PY -m common.run_score --gen results/F_ladder.json \
    --out results/F_ladder_scored.json --fold-batch 12 --pppl-n 8

echo DONE
