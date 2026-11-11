#!/bin/bash
#SBATCH -J p2_dep
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 8:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/dependency.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/dependency.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

echo "### smoke ###"
$PY -m parallel.run_dependency --states 2 --reveals 0.5 --group 8 \
    --n-probe 24 --batch 4 --out results/E_dep_smoke.json

echo "### full ###"
$PY -m parallel.run_dependency --states 12 --reveals 0.25 0.50 0.75 \
    --group 8 16 --n-probe 40 --batch 8 \
    --out results/E_dependency.json
echo DONE
