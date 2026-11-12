"""Max-ID: the highest identity between a generated sequence and the corpus
the model that produced it was trained on.

The question this answers is narrow and important. The conditioning gain
survives deleting every BLAST relative of every pattern source (Round 4), but
that controls the PATTERN's homology, not whether a generated sequence happens
to reproduce some training protein. Max-ID asks the second question directly:
align every generated sequence against the whole training corpus and keep the
best identity.

k-mer novelty already in the metrics table is a substring statistic and will
not see a diverged copy; blastp will.

Reads the sequences already stored in a scored JSON -- nothing is generated.
blastp comes from the BLAST+ module, as in run/clustered_split.sh; this script
does not download or install anything.

Usage:
    python -m tools.max_id --scored results/K_cath_purged_scored.json \
        --db data/clustered_cath/traindb --out results/L_maxid_K_purged.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def blast_max_identity(seqs, db, evalue=10.0, threads=4, max_target=50):
    """-> per-sequence best pident (0 where blastp reports no hit at all)."""
    with tempfile.TemporaryDirectory() as td:
        q = os.path.join(td, "q.fasta")
        with open(q, "w") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">g{i}\n{s}\n")
        tab = os.path.join(td, "hits.tsv")
        cmd = ["blastp", "-query", q, "-db", db, "-outfmt",
               "6 qseqid sseqid pident length qlen", "-evalue", str(evalue),
               "-max_target_seqs", str(max_target), "-num_threads",
               str(threads), "-out", tab]
        subprocess.run(cmd, check=True)
        best = np.zeros(len(seqs))
        cov = np.zeros(len(seqs))
        for line in open(tab):
            f = line.split("\t")
            i = int(f[0][1:])
            pid, aln, qlen = float(f[2]), int(f[3]), int(f[4])
            # identity over the QUERY, so a 20-residue perfect patch cannot
            # score 100 %: that is the number people mean by "max identity".
            v = pid * aln / max(qlen, 1)
            if v > best[i]:
                best[i], cov[i] = v, aln / max(qlen, 1)
    return best, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--db", required=True, help="makeblastdb prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    d = json.load(open(args.scored))
    out = {"scored": args.scored, "db": args.db, "cells": []}
    for c in d["cells"]:
        if not c.get("seqs"):
            continue
        best, cov = blast_max_identity(c["seqs"], args.db,
                                       threads=args.threads)
        rec = {"selector": c["selector"], "n": len(best),
               "max_id_mean": float(best.mean()),
               "max_id_max": float(best.max()),
               "max_id_p95": float(np.percentile(best, 95)),
               "frac_no_hit": float((best == 0).mean()),
               "aln_cov_mean": float(cov.mean())}
        out["cells"].append(rec)
        print("  %-22s mean %.1f %%  max %.1f %%  p95 %.1f %%  no hit %.0f %%"
              % (rec["selector"], rec["max_id_mean"], rec["max_id_max"],
                 rec["max_id_p95"], 100 * rec["frac_no_hit"]), flush=True)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
