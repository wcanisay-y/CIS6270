#!/bin/bash
#SBATCH -J p2_condcath
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cond_cath.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cond_cath.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
PAT=data/clustered_cath/heldout_patterns.json

# ===================================================================
# The same 2x2 as run/cond_clustered.sh, with CATH S35 domains as the pattern
# sources instead of SwissProt dev sequences.
#
#               chemical partition   random partition
#   original    A                    B
#   CATH-purged C                    D
#
# (A-B) tests whether the conditioning effect holds on a different protein
# distribution at all. (C-D) tests it with every BLAST relative of every CATH
# pattern source removed from training. (A-B) - (C-D) is the memorisation
# contribution on this dataset, comparable with the +2.71 [+0.95, +4.55]
# measured on SwissProt.
#
# The SwissProt comparison numbers, same design, 128 patterns:
#   original +26.09 [+22.89, +29.28]   purged +23.38 [+20.07, +26.68]
# ===================================================================
for F in "$PAT" ckpt/clustered_cath/dms6.pt ckpt/clustered_cath/rand6.pt; do
  [ -s "$F" ] || { echo "FATAL: missing $F" >&2; exit 1; }
done
$PY -I -c "import json;d=json.load(open('$PAT'));print('CATH patterns: %d, with a SwissProt training relative %d (%.1f%%), purged %d (%.2f%% of corpus)'%(d['n_patterns'],d['n_patterns_with_relative'],100*d['n_patterns_with_relative']/d['n_patterns'],d['n_purged'],100*d['purge_frac']))"

echo "########## row 1: ORIGINAL checkpoints, CATH patterns ##########"
$PY -m hier.run_generate --ckpt ckpt/dms6.pt ckpt/rand6.pt \
    --length 200 --steps 100 --n 128 --batch 24 \
    --conditioned --pattern-file $PAT \
    --rho 0.3 --temps 1.0 --fam-temps 1.0 --rank gumbel \
    --hier-sampler scheduled --out results/K_cath_orig.json

echo "########## row 2: CATH-PURGED checkpoints, same patterns ##########"
$PY -m hier.run_generate --ckpt ckpt/clustered_cath/dms6.pt ckpt/clustered_cath/rand6.pt \
    --length 200 --steps 100 --n 128 --batch 24 \
    --conditioned --pattern-file $PAT \
    --rho 0.3 --temps 1.0 --fam-temps 1.0 --rank gumbel \
    --hier-sampler scheduled --out results/K_cath_purged.json

echo "########## score both rows ##########"
for F in results/K_cath_orig.json results/K_cath_purged.json; do
  $PY -m common.run_score --gen $F \
      --out ${F%.json}_scored.json --fold-batch 12 --pppl-n 8
done

echo "########## the comparison ##########"
$PY -m analysis.clustered results/K_cath_orig_scored.json \
    results/K_cath_purged_scored.json \
    data/clustered_cath/heldout_patterns.json || true

echo DONE
