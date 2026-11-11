#!/bin/bash
#SBATCH -J p2_Ar
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 16:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaAr.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaAr.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}

# Resume: stages 0, 0b, 1 and 2 already completed and their files are on
# disk. Scoring runs FIRST here, so the headline numbers exist even if a
# later generation stage fails again.
echo "########## score what is already generated ##########"
for f in A_fidelity A_self; do
  [ -f results/${f}_scored.json ] && { echo "skip $f"; continue; }
  echo "--- scoring $f ---"
  $PY -m common.run_score --gen results/$f.json \
      --out results/${f}_scored.json --fold-batch 12 --pppl-n 10
done

echo "########## stage 3: raw-attention coupling ##########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled --n $N --batch 16 \
    --contact-mode rawattn --tau-mode quantile --tau-q 0.99 \
    --out results/A_rawattn.json

echo "########## stage 5: oracle contact maps ##########"
$PY -m parallel.run_generate --lengths 200 --steps 6 12 25 50 \
    --selectors contact_oracle contact_oracle_shuffled anti_contact_oracle \
    --n $N --batch 16 --tau 0.5 \
    --oracle-file results/A_oracle_maps.npz --out results/A_oracle.json

echo "########## stage 6: quantile threshold on predicted contacts ##########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled anti_contact --n $N --batch 16 \
    --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --out results/A_q99.json

echo "########## stage 8: gated to the late trajectory ##########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled --n $N --batch 16 \
    --contact-mode self --tau-mode quantile --tau-q 0.99 \
    --contact-min-revealed 0.5 --out results/A_gated.json

echo "########## stage 4: external ESM-2 650M contacts (the proposal) #######"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 \
    --selectors contact contact_shuffled --n $N --batch 8 \
    --contact-mode external --contact-size 650 --contact-every 1 \
    --tau-mode quantile --tau-q 0.99 --out results/A_ext.json

echo "########## stage 10: self-oracle ##########"
$PY -m parallel.self_oracle --length 200 --steps 12 25 50 --n $N \
    --batch 16 --out results/A_self_oracle.json

echo "########## score the rest ##########"
for f in A_oracle A_q99 A_rawattn A_gated A_ext A_self_oracle; do
  [ -f results/$f.json ] || continue
  echo "--- scoring $f ---"
  $PY -m common.run_score --gen results/$f.json \
      --out results/${f}_scored.json --fold-batch 12 --pppl-n 10
done
echo DONE
