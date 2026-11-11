#!/bin/bash
#SBATCH -J p2_depA2
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/dependency_A2.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/dependency_A2.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
# Rerun Part A keeping per-chain rows, so AUROC(dependency | true contact)
# can be split by CATH class. This is the linchpin: the oracle helps on
# beta-rich folds (+7.47) and hurts on alpha-rich ones (-4.59), and the
# prediction is that contacts predict the model's own coupling BETTER in
# beta folds. If they do, the whole account is coherent.
$PY -m parallel.run_dependency --part A --reveals 0.10 0.25 0.50 0.75 \
    --n-probe 40 --batch 8 --states 6 \
    --oracle-file results/A_oracle_cath.npz --out results/E_depA_cath2.json
echo DONE
