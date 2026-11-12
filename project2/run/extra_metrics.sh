#!/bin/bash
#SBATCH -J p2_extrametrics
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/extra_metrics.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/extra_metrics.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python

# ===================================================================
# Metrics for the conditioning result that are NOT ESMFold confidence.
#
# The whole conditioning claim -- +23.4 pLDDT on SwissProt, +19.9 on CATH S35
# -- rests on pLDDT, pTM, pAE and foldability, which are one ESMFold pass read
# four ways (per-sequence pTM correlates 0.909 with pLDDT). The one metric
# from a different model, ESM-2 pseudo-perplexity, moves 0.20 where pLDDT
# moves 19.91 and on CATH favours the UNCONDITIONAL arm. This job adds the
# axes that can disagree:
#
#   NSR, edit distance   is the pattern a retrieval key or a constraint?
#   Max-ID (blastp)      did any generated sequence come back close to a
#                        training protein, which k-mer novelty cannot see?
#   FPD, MMD             do the generated and natural ESM-2 embedding
#                        DISTRIBUTIONS move together, which no per-sequence
#                        mean can show?
#
# Nothing is generated and nothing is folded: every sequence scored here is
# already stored in the results/*_scored.json the published numbers came from.
# ===================================================================

module load BLAST+/2.14.0-gompi-2022b
echo "blastp: $(command -v blastp)"

# ---- blast databases for the two PURGED corpora -------------------
# Each arm is compared against the corpus IT was trained on, so a purged arm
# must not be scored against sequences that were deleted before it trained.
for D in data/clustered data/clustered_cath; do
  if [ ! -s $D/purgeddb.pin ]; then
    echo "### building $D/purgeddb"
    $PY -I -c "
import sys
src, dst = sys.argv[1], sys.argv[2]
with open(dst,'w') as o:
    for i, line in enumerate(open(src)):
        s=line.strip()
        if s: o.write('>pt%d\n%s\n'%(i,s))
" $D/train_purged.txt $D/train_purged.fasta
    makeblastdb -in $D/train_purged.fasta -dbtype prot -out $D/purgeddb \
        -title purgeddb > $D/makeblastdb_purged.log 2>&1
    tail -2 $D/makeblastdb_purged.log
  fi
done

# scored file | pattern file | ckpt dir (for the family-aware NSR floors) |
# blast db = the training corpus of the arms in that file
RUNS=(
"results/J_cond_orig_scored.json|data/clustered/heldout_patterns.json|ckpt|data/clustered/traindb|J_orig"
"results/J_cond_purged_scored.json|data/clustered/heldout_patterns.json|ckpt/clustered|data/clustered/purgeddb|J_purged"
"results/K_cath_orig_scored.json|data/clustered_cath/heldout_patterns.json|ckpt|data/clustered/traindb|K_orig"
"results/K_cath_purged_scored.json|data/clustered_cath/heldout_patterns.json|ckpt/clustered_cath|data/clustered_cath/purgeddb|K_purged"
)

for R in "${RUNS[@]}"; do
  IFS='|' read -r SCORED PAT CK DB TAG <<< "$R"
  echo "########## $TAG ##########"
  [ -s "$SCORED" ] || { echo "FATAL: missing $SCORED" >&2; exit 1; }

  echo "--- NSR, edit distance, FPD, MMD"
  $PY -m tools.extra_metrics_run --scored "$SCORED" --patterns "$PAT" \
      --ckpt-dir "$CK" --out "results/L_extra_${TAG}.json"

  echo "--- Max-ID against $DB"
  $PY -m tools.max_id --scored "$SCORED" --db "$DB" --threads 8 \
      --out "results/L_maxid_${TAG}.json"
done

echo "########## done ##########"
ls -la results/L_*.json
