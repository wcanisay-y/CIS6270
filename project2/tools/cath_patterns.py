"""Build a CATH S35 pattern-source set, to test conditioning off SwissProt.

Why a second dataset. Every conditioning number so far comes from SwissProt
dev sequences, which are drawn from the same corpus the model trains on. That
tests the capacity of the family channel but not whether the effect holds on
a different distribution of proteins.

CATH S35 is the natural classical choice: it is one of the two canonical fold
classifications, it is non-redundant at 35 % sequence identity by
construction, and its entries are structural DOMAINS rather than full-length
UniProt records. Domains are a genuinely different distribution -- no signal
peptides, no linkers, no multi-domain architecture, tighter length range --
so this is a distribution shift and not a re-draw from the same pool.

Output is written in the layout `tools/build_clustered_split.py` expects, so
the same BLAST purge, retrain and 2x2 conditioning pipeline runs unchanged:

    <outdir>/devcand.txt    one candidate sequence per line
    <outdir>/devcand.fasta  the same, as cd<i> records
    <outdir>/train.txt      a copy of the SwissProt training corpus
    <outdir>/train.fasta    the same, as tr<i> records

The training corpus is copied rather than rebuilt so that the `tr<i>` indices
match the SwissProt run exactly and an existing BLAST database can be reused.
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.alphabet import AA20


def read_fasta(path):
    name, buf = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(buf)
            name, buf = line[1:], []
        elif line:
            buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="data/cath_seq/cath-S35.fa")
    ap.add_argument("--length", type=int, default=200,
                    help="a candidate must be at least this long, matching "
                         "the SwissProt run so the two are comparable")
    ap.add_argument("--max-len", type=int, default=254,
                    help="same upper bound as the training length filter")
    ap.add_argument("--cap", type=int, default=4000,
                    help="cap the candidate pool; 128 are sampled from it "
                         "later and a smaller pool keeps the BLAST cheap")
    ap.add_argument("--from-dir", default="data/clustered",
                    help="directory holding the SwissProt train.txt/fasta to "
                         "reuse, so tr<i> indices and the BLAST db still match")
    ap.add_argument("--outdir", default="data/clustered_cath")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ok = set(AA20)
    kept, n_tot, n_short, n_long, n_bad = [], 0, 0, 0, 0
    seen = set()
    for name, seq in read_fasta(a.fasta):
        n_tot += 1
        if len(seq) < a.length:
            n_short += 1
            continue
        if len(seq) > a.max_len:
            n_long += 1
            continue
        if not set(seq) <= ok:
            n_bad += 1
            continue
        if seq in seen:            # S35 is non-redundant but be safe
            continue
        seen.add(seq)
        kept.append((name, seq))

    print("CATH %s: %d domains" % (a.fasta, n_tot))
    print("  shorter than %d          : %d" % (a.length, n_short))
    print("  longer than %d           : %d" % (a.max_len, n_long))
    print("  non-canonical residues   : %d" % n_bad)
    print("  eligible                 : %d" % len(kept))

    if len(kept) > a.cap:
        rng = np.random.default_rng(a.seed)
        idx = sorted(rng.choice(len(kept), a.cap, replace=False).tolist())
        kept = [kept[i] for i in idx]
        print("  capped to                : %d (seed %d)" % (len(kept), a.seed))

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "devcand.txt"), "w") as fh:
        fh.write("\n".join(s for _, s in kept))
    with open(os.path.join(a.outdir, "devcand.fasta"), "w") as fh:
        for i, (_, s) in enumerate(kept):
            fh.write(">cd%d\n%s\n" % (i, s))
    with open(os.path.join(a.outdir, "devcand_ids.txt"), "w") as fh:
        fh.write("\n".join("cd%d\t%s" % (i, n)
                           for i, (n, _) in enumerate(kept)))

    for f in ("train.txt", "train.fasta"):
        src, dst = os.path.join(a.from_dir, f), os.path.join(a.outdir, f)
        if not os.path.exists(src):
            raise SystemExit("missing %s; run tools.export_fasta first" % src)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    print("\nwrote %s/devcand.{txt,fasta} and reused the SwissProt train files"
          % a.outdir)

    ln = np.array([len(s) for _, s in kept], dtype=float)
    print("candidate length: mean %.1f median %.0f min %d max %d"
          % (ln.mean(), np.median(ln), ln.min(), ln.max()))


if __name__ == "__main__":
    main()
