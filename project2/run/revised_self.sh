#!/bin/bash
#SBATCH -J p2_self
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/revised_self.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/revised_self.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

# Stage 1 of the oracle redo showed, against DEPOSITED structures, that the
# generator's own attention keeps precision@L 0.648 at 50 % masking while a
# separate frozen ESM-2 pass is at its null (0.066 vs 0.069). The free
# signal is roughly ten times the external one in exactly the regime the
# selector runs in. So the warm-up sweep that matters is the one on the FREE
# signal, which the main job runs at a single warm-up only.
$PY -m parallel.run_revised --length 200 --n $N --batch 16 \
    --steps 25 50 100 --warmups 0.0 0.2 0.35 0.5 \
    --refresh-dist 0.15 --refresh-reveal 0.10 --refresh-max-age 8 \
    --tau-q 0.90 --quick --self-sweep \
    --out results/C_revised_self.json
$PY -m common.run_score --gen results/C_revised_self.json \
    --out results/C_revised_self_scored.json --fold-batch 12 --pppl-n 8
echo DONE
