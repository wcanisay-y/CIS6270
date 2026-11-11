"""Did removing the pattern sources' relatives from training change the gain?

Reads the 2x2 produced by run/cond_clustered.sh and reports, on identical
patterns:

    (chemical - random) on the ORIGINAL corpus
    (chemical - random) on the PURGED corpus
    the difference between them = what homologs in training were worth

The within-row contrast is the one that carries the conclusion. The purged
model trains on a smaller corpus, so its absolute pLDDT is not comparable
with the original model's; chemical-minus-random is, because both arms in a
row share the corpus, the step count and the patterns.
"""
import json
import os
import sys

import numpy as np

RNG = np.random.default_rng(0)
NB = 20000


def cells(path):
    d = json.load(open(path))
    return {c["selector"]: c for c in d["cells"] if "plddt_list" in c}, d


def ci(x):
    x = np.asarray(x, float)
    bs = np.array([RNG.choice(x, len(x), True).mean() for _ in range(NB)])
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    return ci(a[:n] - b[:n]), n


def row(A, label):
    kc, kr = "dms6|conditioned", "rand6|conditioned"
    uc, ur = "dms6|uncond", "rand6|uncond"
    missing = [k for k in (kc, kr, uc, ur) if k not in A]
    if missing:
        print("  %s: missing cells %s" % (label, missing))
        return None
    (m, lo, hi), n = paired(A[kc]["plddt_list"], A[kr]["plddt_list"])
    out = dict(
        label=label, n=n,
        chem_cond=float(np.mean(A[kc]["plddt_list"])),
        rand_cond=float(np.mean(A[kr]["plddt_list"])),
        chem_uncond=float(np.mean(A[uc]["plddt_list"])),
        rand_uncond=float(np.mean(A[ur]["plddt_list"])),
        chem_minus_rand=m, lo=lo, hi=hi,
        chem_gain=float(np.mean(A[kc]["plddt_list"])
                        - np.mean(A[uc]["plddt_list"])),
        rand_gain=float(np.mean(A[kr]["plddt_list"])
                        - np.mean(A[ur]["plddt_list"])),
        chem_fold=A[kc].get("foldability"), rand_fold=A[kr].get("foldability"),
        chem_ptm=A[kc].get("ptm_mean"), rand_ptm=A[kr].get("ptm_mean"))
    print("  %-10s n=%3d  chem %.2f  rand %.2f  chem-rand %+.2f [%+.2f, %+.2f]"
          "  own gains %+.2f / %+.2f" % (label, n, out["chem_cond"],
          out["rand_cond"], m, lo, hi, out["chem_gain"], out["rand_gain"]))
    return out


def main(orig="results/J_cond_orig_scored.json",
         purged="results/J_cond_purged_scored.json",
         split="data/clustered/heldout_patterns.json"):
    if os.path.exists(split):
        s = json.load(open(split))
        print("split: %d patterns, %d of them with a training relative, "
              "%d sequences purged (%.2f %% of the corpus), depth %d, "
              "E <= %g" % (s["n_patterns"], s["n_patterns_with_relative"],
                           s["n_purged"], 100 * s["purge_frac"],
                           s["config"]["depth"], s["config"]["evalue"]))
        print("corpus %d -> %d" % (s["n_train_before"], s["n_train_after"]))

    print("\n=== chemical minus random, paired per source ===")
    rows = []
    for path, label in ((orig, "original"), (purged, "purged")):
        if not os.path.exists(path):
            print("  %s: %s not written yet" % (label, path))
            continue
        A, _ = cells(path)
        r = row(A, label)
        if r:
            rows.append((label, A, r))

    if len(rows) < 2:
        print("\nboth rows are needed for the difference of differences")
        return

    (_, Ao, ro), (_, Ap, rp) = rows
    print("\n=== difference of differences: what training homologs were worth "
          "===")
    co = np.asarray(Ao["dms6|conditioned"]["plddt_list"], float) \
        - np.asarray(Ao["rand6|conditioned"]["plddt_list"], float)
    cp = np.asarray(Ap["dms6|conditioned"]["plddt_list"], float) \
        - np.asarray(Ap["rand6|conditioned"]["plddt_list"], float)
    n = min(len(co), len(cp))
    # Paired across rows too: the same source protein supplies the pattern in
    # both, so the per-source difference is the right unit.
    m, lo, hi = ci(co[:n] - cp[:n])
    print("  original %+.2f  purged %+.2f  difference %+.2f [%+.2f, %+.2f]"
          % (ro["chem_minus_rand"], rp["chem_minus_rand"], m, lo, hi))
    if lo <= 0 <= hi:
        print("  -> the interval contains zero: removing every BLAST relative "
              "of the pattern sources did not measurably change the gain, so "
              "memorised homologs are not what produces it")
    else:
        print("  -> the interval excludes zero: homologs in training were "
              "worth %+.2f pLDDT of the gain" % m)

    print("\n=== foldability and pTM, for the record ===")
    print("  %-10s %-18s %-18s" % ("", "chemical", "random"))
    for label, _, r in rows:
        print("  %-10s fold %.3f pTM %.3f  fold %.3f pTM %.3f"
              % (label, r["chem_fold"] or 0, r["chem_ptm"] or 0,
                 r["rand_fold"] or 0, r["rand_ptm"] or 0))

    json.dump(dict(rows=[r for _, _, r in rows],
                   dod=dict(mean=m, lo=lo, hi=hi, n=n)),
              open("results/J_clustered_summary.json", "w"), indent=1)
    print("\nWROTE results/J_clustered_summary.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
