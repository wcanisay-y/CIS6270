"""Write the training corpus and the dev pattern candidates as FASTA.

Step 1 of the clustered-split experiment. The corpus is exactly what
hier/train.py sees (same length filter, same canonical-residue filter), so
the BLAST search that follows is run against the sequences the model was
actually trained on rather than against the raw file.

Dev candidates are the dev sequences long enough to supply a `--length`
pattern, which is the pool G_clean_patterns.json was drawn from by 10-mer
containment. Here they are written out so the same pool can be re-selected
by alignment instead.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_swissprot


def write_fasta(path, seqs, prefix):
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(">%s%d\n%s\n" % (prefix, i, s))
    return len(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200,
                    help="pattern length; dev candidates must be at least this")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--outdir", default="data/clustered")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    tr, va = load_swissprot(min_len=40, max_len=a.max_len - 2)
    cand = [s for s in va if len(s) >= a.length]

    n_tr = write_fasta(os.path.join(a.outdir, "train.fasta"), tr, "tr")
    n_cd = write_fasta(os.path.join(a.outdir, "devcand.fasta"), cand, "cd")

    # The index is the contract between this script and the split builder:
    # BLAST ids are tr<i> / cd<i> into these two lists.
    with open(os.path.join(a.outdir, "train.txt"), "w") as fh:
        fh.write("\n".join(tr))
    with open(os.path.join(a.outdir, "devcand.txt"), "w") as fh:
        fh.write("\n".join(cand))

    print("train    %7d sequences -> train.fasta" % n_tr)
    print("dev cand %7d sequences -> devcand.fasta (length >= %d)"
          % (n_cd, a.length))
    print("dev total %6d" % len(va))


if __name__ == "__main__":
    main()
