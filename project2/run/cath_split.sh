#!/bin/bash
#SBATCH -J p2_cathsplit
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cath_split.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/cath_split.err
set -euo pipefail
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
D=data/clustered_cath
DB=data/clustered/traindb          # same train.fasta, so the db is reusable

# ===================================================================
# Test the conditioning result on a second, classical dataset.
#
# CATH S35: one of the two canonical fold classifications, non-redundant at
# 35 % sequence identity by construction, and made of structural DOMAINS
# rather than full-length UniProt records. 3,876 of its 37,350 domains are in
# the 200-254 length window used throughout this project, with mean length
# 224.3 against 225.8 for the SwissProt candidate pool -- matched length,
# different protein distribution.
#
# The purge is run against the SwissProt TRAINING corpus, because that is
# what the model saw. CATH domains are excised from real PDB entries, so most
# will have SwissProt relatives; removing them is what makes this comparable
# with the SwissProt arm rather than a weaker test.
# ===================================================================
module load BLAST+/2.14.0-gompi-2022b
echo "blastp: $(command -v blastp)"

for F in $D/devcand.txt $D/train.txt ${DB}.pin; do
  [ -s "$F" ] || { echo "FATAL: missing $F" >&2; exit 1; }
done
echo "candidates: $(wc -l < $D/devcand.txt)  training: $(wc -l < $D/train.txt)"

echo "########## sample 128 held-out CATH pattern sources ##########"
$PY -m tools.build_clustered_split select --dir $D --n-patterns 128 --seed 0

echo "########## round 1: CATH patterns vs the SwissProt training corpus ####"
time blastp -query $D/h.fasta -db $DB \
    -outfmt "6 qseqid sseqid pident evalue bitscore qcovhsp scovhsp" \
    -evalue 1 -max_target_seqs 5000 -num_threads 4 -seg yes \
    -out $D/h_vs_train.tsv
wc -l $D/h_vs_train.tsv

echo "########## expand one step (single linkage) ##########"
$PY -m tools.build_clustered_split expand --dir $D --evalue 1e-3
NP=$(grep -c "^>" $D/purge.fasta || true)
echo "purge.fasta has $NP sequences"

if [ "$NP" -gt 0 ] && [ "$NP" -le 8000 ]; then
  echo "########## round 2 ##########"
  time blastp -query $D/purge.fasta -db $DB \
      -outfmt "6 qseqid sseqid pident evalue bitscore qcovhsp scovhsp" \
      -evalue 1 -max_target_seqs 5000 -num_threads 4 -seg yes \
      -out $D/purge_vs_train.tsv
  wc -l $D/purge_vs_train.tsv
  DEPTH=2
else
  echo "skipping round 2: $NP sequences is outside the 1..8000 budget"
  DEPTH=1
fi

echo "########## build the purged corpus (depth $DEPTH) ##########"
$PY -m tools.build_clustered_split finish --dir $D --evalue 1e-3 \
    --depth $DEPTH --max-purge-frac 0.40

echo DONE
