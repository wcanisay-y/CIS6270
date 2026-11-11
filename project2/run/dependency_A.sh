#!/bin/bash
#SBATCH -J p2_depA
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/dependency_A.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/dependency_A.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
# Part A across the full reveal range, with probe positions forced masked in
# every member of a chunk. Both oracle sets, so the answer does not depend on
# the ad-hoc selection.
$PY -m parallel.run_dependency --part A --reveals 0.10 0.25 0.50 0.75 \
    --n-probe 40 --batch 8 --states 6 \
    --oracle-file results/A_oracle_pdb.npz --out results/E_depA_pdb.json
$PY -m parallel.run_dependency --part A --reveals 0.10 0.25 0.50 0.75 \
    --n-probe 40 --batch 8 --states 6 \
    --oracle-file results/A_oracle_cath.npz --out results/E_depA_cath.json
echo DONE
