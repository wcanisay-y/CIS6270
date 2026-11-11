#!/bin/bash
#SBATCH -J p2_Arp
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 10:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaArp.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaArp.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

# Idea A inside the decoder DPLM actually ships. The absorbing sampler the
# proposal implies scores about 55 pLDDT in our harness; DPLM's own
# reparameterised decoder scores 83.6, matching its published 84.3. The
# rule therefore has to be tested where the quality is, not only where the
# proposal's framing puts it. Forward passes and the survivor count per
# step are unchanged; only WHICH positions survive changes.
echo "########## contact rule inside the reparameterised decoder ##########"
$PY -m parallel.run_generate --lengths 200 --steps 25 50 100 200 \
    --selectors reparam_confidence reparam_contact reparam_contact_shuffled \
                reparam_anti_contact \
    --n $N --batch 12 --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --out results/A_reparam.json
$PY -m common.run_score --gen results/A_reparam.json \
    --out results/A_reparam_scored.json --fold-batch 12 --pppl-n 10
echo DONE
