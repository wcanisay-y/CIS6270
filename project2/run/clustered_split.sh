#!/bin/bash
#SBATCH -J p2_split
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/clustered_split.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/clustered_split.err
set -euo pipefail
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
PY=./venv2/bin/python
D=data/clustered

# ===================================================================
# Stage 1 of the clustered-split experiment: find, by ALIGNMENT, which
# training sequences are relatives of the held-out pattern sources.
#
# Why this exists. The +20.22 conditioning result is measured on sources
# selected by 10-mer containment, which is a substring test. It cannot see
# remote homology, so a source with no shared 10-mer may still share a fold
# with many training sequences and the pattern could act as a retrieval key
# for a memorised fold. BLAST is the available alignment search: MMseqs2 and
# CD-HIT are not installed, and running a downloaded MMseqs2 binary was not
# permitted, so the neighbourhood is defined by blastp instead of by a
# proper clustering. That is a weaker instrument and is declared as such:
# single-linkage BLAST neighbourhood, not MMseqs2 clustering.
#
# E <= 1e-3 with soft masking counts as "a relative". That is permissive for
# proteins, which is the safe direction here: over-purging costs training
# data, under-purging leaves the leak in place.
# ===================================================================
module load BLAST+/2.14.0-gompi-2022b
echo "blastp: $(command -v blastp)"
blastp -version | head -1

echo "########## export the corpus the model actually trains on ##########"
$PY -m tools.export_fasta --length 200 --outdir $D

echo "########## sample the held-out pattern sources ##########"
# No homology criterion in the sampling: selecting orphan proteins would
# bias the set towards unusual sequences and would not test the +20.22.
$PY -m tools.build_clustered_split select --dir $D --n-patterns 128 --seed 0

echo "########## build the BLAST database over training ##########"
makeblastdb -in $D/train.fasta -dbtype prot -out $D/traindb \
    -title traindb -parse_seqids > $D/makeblastdb.log 2>&1
tail -3 $D/makeblastdb.log

echo "########## round 1: held-out sources vs training ##########"
time blastp -query $D/h.fasta -db $D/traindb \
    -outfmt "6 qseqid sseqid pident evalue bitscore qcovhsp scovhsp" \
    -evalue 1 -max_target_seqs 5000 -num_threads 4 -seg yes \
    -out $D/h_vs_train.tsv
wc -l $D/h_vs_train.tsv

echo "########## expand the removal set one step (single linkage) ##########"
$PY -m tools.build_clustered_split expand --dir $D --evalue 1e-3
NP=$(grep -c "^>" $D/purge.fasta || true)
echo "purge.fasta has $NP sequences"

if [ "$NP" -gt 0 ] && [ "$NP" -le 8000 ]; then
  echo "########## round 2: those relatives vs training ##########"
  time blastp -query $D/purge.fasta -db $D/traindb \
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
    --depth $DEPTH --max-purge-frac 0.30

echo DONE
