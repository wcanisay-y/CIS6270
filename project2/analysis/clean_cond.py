"""Conditioning on a homology-clean pattern set, against the original.

The original 48 patterns came from the upstream SwissProt dev split, which
is not homology-clustered: 17 of 48 shared more than half their 10-mers with
a training sequence, and the chemical arm gained +10.2 pLDDT on exactly
those while the random arm gained nothing. `results/G_clean_patterns.json`
replaces them with the 96 cleanest dev sequences (max containment 0.026).

Reports the paired chemical-minus-random contrast per source protein, which
is the comparison that isolates chemistry from information content, and
re-runs the retrieval checks on the new sequences.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.alphabet import AA20, AA_IDX, PARTITIONS, family_table, \
    random_family_table

RNG = np.random.default_rng(0)
NB = 20000


def arms(path):
    d = json.load(open(path))
    out = {}
    for c in d["cells"]:
        s = c.get("selector")
        # run_score writes the file as it folds, so a cell read mid-run may
        # not carry plddt_list yet. Skip those rather than crash.
        if s and "plddt_list" in c:
            out[s] = c
    return out, d


def ci(x):
    bs = np.array([RNG.choice(x, len(x), True).mean() for _ in range(NB)])
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return ci(d)


def main(clean="results/G_cond_clean_scored.json",
         orig="results/B_cond_scored.json",
         patterns="results/G_clean_patterns.json"):
    A, D = arms(clean)
    pf = json.load(open(patterns))
    nat = [s[:200] for s in pf["seqs"]]
    cont = np.asarray(pf["containment"], float)
    print("clean pattern set: %d patterns, containment max %.4f mean %.4f"
          % (len(nat), cont.max(), cont.mean()))

    print("\n=== pLDDT by arm ===")
    print("%-26s %8s %8s %8s %8s" % ("arm", "n", "pLDDT", "pTM", "fold"))
    for k in sorted(A):
        c = A[k]
        pl = np.asarray(c["plddt_list"], float)
        print("%-26s %8d %8.2f %8.4f %8.4f"
              % (k, len(pl), pl.mean(),
                 float(np.mean(c.get("ptm_list") or [np.nan])),
                 c.get("foldability", float("nan"))))

    kc, kr = "dms6|conditioned", "rand6|conditioned"
    uc, ur = "dms6|uncond", "rand6|uncond"
    if kc not in A or kr not in A:
        print("\nconditioned arms missing; nothing further to compare")
        return

    a = np.asarray(A[kc]["plddt_list"], float)
    b = np.asarray(A[kr]["plddt_list"], float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    print("\n=== THE DECISIVE CONTRAST: chemical - random, paired per source ===")
    print("  clean set   %+.2f [%+.2f, %+.2f]   (n=%d)" % (*paired(a, b), n))
    print("  original    +26.58  (contaminated set, n=48)")
    if uc in A:
        u = float(np.mean(A[uc]["plddt_list"]))
        print("\n=== chemical conditioned vs its own unconditional baseline ===")
        print("  conditioned %.2f   unconditional %.2f   gain %+.2f"
              % (a.mean(), u, a.mean() - u))
    if ur in A:
        u2 = float(np.mean(A[ur]["plddt_list"]))
        bb = b.mean()
        print("  random      %.2f   unconditional %.2f   gain %+.2f"
              % (bb, u2, bb - u2))

    print("\n=== does the gain still track containment? (it should not now) ===")
    g = a - b
    c2 = cont[:n]
    if c2.std() > 0:
        r = np.corrcoef(c2, g)[0, 1]
        print("  corr(containment, chemical-random) = %+.3f over %d sources "
              "(containment range %.4f-%.4f)" % (r, n, c2.min(), c2.max()))

    print("\n=== retrieval checks on the clean set ===")
    cnt = np.zeros(20)
    for s in nat[:n]:
        for ch in s:
            cnt[AA_IDX[ch]] += 1
    p = cnt / cnt.sum()
    for tag, lut in ((kc, family_table(PARTITIONS["dms6"])[1]),
                     (kr, random_family_table(PARTITIONS["dms6"], 0)[1])):
        gen = A[tag]["seqs"][:n]
        own = np.mean([np.mean([x == y for x, y in zip(gen[i], nat[i])])
                       for i in range(n)])
        unif = np.mean([np.mean([1.0 / (lut == lut[AA_IDX[ch]]).sum()
                                 for ch in nat[i]]) for i in range(n)])
        modal = {}
        for fm in range(lut.max() + 1):
            mem = np.where(lut == fm)[0]
            modal[fm] = mem[np.argmax(p[mem])]
        mod = np.mean([np.mean([AA_IDX[ch] == modal[lut[AA_IDX[ch]]]
                                for ch in nat[i]]) for i in range(n)])
        print("  %-22s identity to source %.3f | uniform floor %.3f | "
              "modal floor %.3f | novelty %.4f"
              % (tag, own, unif, mod, A[tag].get("novelty_kmer", float("nan"))))

    if os.path.exists(orig):
        O, _ = arms(orig)
        print("\n=== original (contaminated) set, for reference ===")
        for k in (kc, kr, uc, ur):
            if k in O:
                print("  %-22s %.2f" % (k, O[k]["plddt_mean"]))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
