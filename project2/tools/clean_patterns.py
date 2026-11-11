"""Select a homology-clean pattern source set for the conditioning experiment.

The original 48 patterns were the first 48 dev sequences of length >= 200.
The upstream SwissProt dev split is not homology-clustered, so 17 of those 48
share more than half their 10-mers with a training sequence and 6 share more
than nine tenths. The conditioning gain is larger on exactly those, and only
for the chemical partition, so the headline number is inflated by an unknown
amount of recall.

This script scans every dev sequence long enough to supply a 200-residue
pattern, measures the largest 10-mer containment of each in the training
corpus the model actually saw, and writes the cleanest N as the replacement
set. Containment is a crude proxy for an alignment, so it is reported per
sequence and kept with the set rather than collapsed to a pass/fail.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_swissprot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=96,
                    help="how many clean patterns to keep")
    ap.add_argument("--max-containment", type=float, default=0.10)
    ap.add_argument("--out", default="results/G_clean_patterns.json")
    a = ap.parse_args()

    K, L = a.k, a.length
    tr, va = load_swissprot(max_n=None)
    cand = [s[:L] for s in va if len(s) >= L]
    print("train %d  dev %d  dev with length >= %d: %d"
          % (len(tr), len(va), L, len(cand)), flush=True)

    kmers = [{s[i:i + K] for i in range(len(s) - K + 1)} for s in cand]
    index = collections.defaultdict(list)
    for i, ks in enumerate(kmers):
        for km in ks:
            index[km].append(i)
    print("indexed %d distinct %d-mers from %d candidates"
          % (len(index), K, len(cand)), flush=True)

    best = np.zeros(len(cand))
    for n, s in enumerate(tr):
        hits = collections.Counter()
        for i in range(len(s) - K + 1):
            for si in index.get(s[i:i + K], ()):
                hits[si] += 1
        for si, h in hits.items():
            c = h / len(kmers[si])
            if c > best[si]:
                best[si] = c
        if (n + 1) % 50000 == 0:
            print("  scanned %d/%d training sequences" % (n + 1, len(tr)),
                  flush=True)

    order = np.argsort(best)
    keep = [int(i) for i in order[: a.n]]
    kept_c = best[keep]
    print("\ncontainment over all %d candidates: mean %.3f  median %.3f"
          % (len(cand), best.mean(), np.median(best)))
    for thr in (0.05, 0.10, 0.25, 0.50, 0.90):
        print("  candidates with containment <= %4.2f : %4d"
              % (thr, int((best <= thr).sum())))
    print("\nkept %d, containment: max %.4f  mean %.4f"
          % (len(keep), kept_c.max(), kept_c.mean()))
    if kept_c.max() > a.max_containment:
        print("  WARNING: the cleanest %d still reach %.3f, above the %.2f "
              "target" % (a.n, kept_c.max(), a.max_containment))

    out = dict(config=vars(a),
               seqs=[cand[i] for i in keep],
               containment=[float(x) for x in kept_c],
               all_containment_summary=dict(
                   n=len(cand), mean=float(best.mean()),
                   median=float(np.median(best)),
                   le_010=int((best <= 0.10).sum()),
                   le_005=int((best <= 0.05).sum())))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
