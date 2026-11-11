#!/bin/bash
#SBATCH -J p2_A
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 16:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaA.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/ideaA.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
N=${N:-48}
mkdir -p results

# --------------------------------------------------------------------------
# Stage 0: is the contact signal even there? Contact-prediction quality as a
# function of how masked the input is, scored against the true contacts of
# the full sequence. Published work says ESM-2 needs 20-40 flanking residues
# per side to see a contact at all, which is the opposite of the regime the
# selector runs in early. Nobody has measured this, so we do.
# --------------------------------------------------------------------------
echo "########## stage 0: contact quality vs mask rate ##########"
$PY -m parallel.oracle --mode quality --length 200 --n 12 \
    --out results/A_contact_quality.json

# --------------------------------------------------------------------------
# Stage 0b: true contact maps of natural proteins, for the oracle arm.
# --------------------------------------------------------------------------
echo "########## stage 0b: build oracle contact maps ##########"
$PY -m parallel.oracle --mode oracle --length 200 --n $N \
    --out results/A_oracle_maps.npz

# --------------------------------------------------------------------------
# Stage 1: fidelity check. Our decoders must reproduce published DPLM-150M
# numbers before any comparison means anything. Targets from the literature:
# DPLM-150M at L=200 is pLDDT ~84; confidence/margin ordering ~82; random
# ordering ~40. If we do not see that spread, the harness is wrong.
# --------------------------------------------------------------------------
echo "########## stage 1: fidelity vs published DPLM numbers ##########"
$PY -m parallel.run_generate --lengths 200 --steps 200 500 \
    --selectors confidence random entropy --n $N --batch 16 \
    --contact-mode self --reparam --out results/A_fidelity.json

# --------------------------------------------------------------------------
# Stage 2: the main matched-NFE grid, free signal (generator's own attention
# through the ESM-2 contact-head regression).
# --------------------------------------------------------------------------
echo "########## stage 2: matched-NFE grid, self contacts ##########"
$PY -m parallel.run_generate --lengths 200 --steps 6 12 25 50 100 200 \
    --selectors confidence random entropy contact contact_shuffled \
                anti_contact contact_degree distance punt \
    --n $N --batch 16 --tau 0.3 --tau-sweep 0.1 0.2 0.5 \
    --contact-mode self --out results/A_self.json

# --------------------------------------------------------------------------
# Stage 3: raw symmetrised attention instead of the supervised contact head.
# This is the free signal the recent text-domain methods use. If the contact
# head does not beat it, a second network is not justified.
# --------------------------------------------------------------------------
echo "########## stage 3: raw-attention coupling ##########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 100 \
    --selectors contact contact_shuffled --n $N --batch 16 --tau 0.3 \
    --contact-mode rawattn --out results/A_rawattn.json

# --------------------------------------------------------------------------
# Stage 4: the proposal as written -- a separate ESM-2 650M forward pass per
# step. Doubles cost per step, so it is also reported per total FLOPs.
# The token_dropout ablation matters: ESM-2 rescales embeddings by
# (1-0.12)/(1-mask_ratio), about 89x on an all-mask 200-mer.
# --------------------------------------------------------------------------
echo "########## stage 4: external ESM-2 650M contacts ##########"
$PY -m parallel.run_generate --lengths 200 --steps 12 25 50 \
    --selectors contact contact_shuffled --n $N --batch 8 --tau 0.3 \
    --contact-mode external --contact-size 650 --contact-every 1 \
    --out results/A_ext.json
$PY -m parallel.run_generate --lengths 200 --steps 25 \
    --selectors contact --n $N --batch 8 --tau 0.3 \
    --contact-mode external --contact-size 650 --esm-token-dropout \
    --out results/A_ext_tokdrop.json

# --------------------------------------------------------------------------
# Stage 5: the oracle. Would a CORRECT contact map help at all? If not, no
# improvement in contact prediction can rescue the mechanism.
# --------------------------------------------------------------------------
echo "########## stage 5: oracle contact maps ##########"
$PY -m parallel.run_generate --lengths 200 --steps 6 12 25 50 \
    --selectors contact_oracle contact_oracle_shuffled anti_contact_oracle \
    --n $N --batch 16 --tau 0.3 \
    --oracle-file results/A_oracle_maps.npz --out results/A_oracle.json

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
echo "########## score ##########"
for f in A_fidelity A_self A_rawattn A_ext A_ext_tokdrop A_oracle; do
  echo "--- scoring $f ---"
  $PY -m common.run_score --gen results/$f.json \
      --out results/${f}_scored.json --fold-batch 8 --pppl-n 10
done
echo DONE
