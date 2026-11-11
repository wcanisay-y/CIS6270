"""Put the SwissProt and CATH conditioning experiments side by side.

Both run the identical 2x2 -- chemical and random partitions, original and
homology-purged corpora, 128 pattern sources, patterns identical within a
column -- so the two datasets are directly comparable. The question this
answers is whether the conditioning effect is a property of the SwissProt dev
split or of protein chemistry.

Reads results/{J,K}_clustered_summary.json written by analysis/clustered.py.
"""
import json
import os
import sys


def load(stem, split):
    s = "results/%s_clustered_summary.json" % stem
    if not os.path.exists(s):
        return None
    d = json.load(open(s))
    out = {r["label"]: r for r in d["rows"]}
    out["dod"] = d["dod"]
    if os.path.exists(split):
        out["split"] = json.load(open(split))
    return out


def main():
    sets = [("SwissProt dev", "J", "data/clustered/heldout_patterns.json"),
            ("CATH S35", "K", "data/clustered_cath/heldout_patterns.json")]
    got = [(n, load(s, sp)) for n, s, sp in sets]
    got = [(n, d) for n, d in got if d]
    if not got:
        print("no summaries found yet")
        return

    print("=== how much homology each pattern set has with training ===")
    print("%-16s %10s %12s %10s %8s %8s"
          % ("dataset", "patterns", "with a rel.", "purged", "frac", "depth"))
    for n, d in got:
        s = d.get("split")
        if not s:
            continue
        print("%-16s %10d %11.1f %% %10d %7.2f %% %8d"
              % (n, s["n_patterns"],
                 100 * s["n_patterns_with_relative"] / s["n_patterns"],
                 s["n_purged"], 100 * s["purge_frac"], s["config"]["depth"]))

    print("\n=== chemical minus random, paired per source ===")
    print("%-16s %-28s %-28s %-24s"
          % ("dataset", "original corpus", "purged corpus", "difference"))
    for n, d in got:
        o, p, dd = d.get("original"), d.get("purged"), d["dod"]
        if not (o and p):
            continue
        print("%-16s %+7.2f [%+.2f, %+.2f]      %+7.2f [%+.2f, %+.2f]      "
              "%+6.2f [%+.2f, %+.2f]"
              % (n, o["chem_minus_rand"], o["lo"], o["hi"],
                 p["chem_minus_rand"], p["lo"], p["hi"],
                 dd["mean"], dd["lo"], dd["hi"]))

    print("\n=== absolute cells, for the record ===")
    print("%-16s %-9s %8s %8s %8s %8s"
          % ("dataset", "corpus", "chem", "chem u.", "rand", "rand u."))
    for n, d in got:
        for lab in ("original", "purged"):
            r = d.get(lab)
            if not r:
                continue
            print("%-16s %-9s %8.2f %8.2f %8.2f %8.2f"
                  % (n, lab, r["chem_cond"], r["chem_uncond"],
                     r["rand_cond"], r["rand_uncond"]))

    print("\n=== foldability: the qualitative claim ===")
    print("%-16s %-9s %10s %10s" % ("dataset", "corpus", "chem fold",
                                    "rand fold"))
    for n, d in got:
        for lab in ("original", "purged"):
            r = d.get(lab)
            if not r:
                continue
            print("%-16s %-9s %10.3f %10.3f"
                  % (n, lab, r["chem_fold"] or 0.0, r["rand_fold"] or 0.0))

    if len(got) == 2:
        (n1, d1), (n2, d2) = got
        if d1.get("purged") and d2.get("purged"):
            a = d1["purged"]["chem_minus_rand"]
            b = d2["purged"]["chem_minus_rand"]
            lo1, hi1 = d1["purged"]["lo"], d1["purged"]["hi"]
            lo2, hi2 = d2["purged"]["lo"], d2["purged"]["hi"]
            print("\n=== verdict ===")
            print("  purged %s %+.2f [%+.2f, %+.2f]" % (n1, a, lo1, hi1))
            print("  purged %s %+.2f [%+.2f, %+.2f]" % (n2, b, lo2, hi2))
            overlap = not (hi1 < lo2 or hi2 < lo1)
            if overlap:
                print("  -> the intervals OVERLAP, so the effect does not "
                      "depend on which dataset supplies the patterns: it is "
                      "a property of the chemistry, not of the SwissProt "
                      "dev split")
            else:
                print("  -> the intervals are DISJOINT, so the effect size "
                      "is dataset-dependent and must be quoted per dataset")
            print("  NOTE: these are separate bootstraps on different "
                   "pattern sets, so overlap is an informal check and not a "
                   "test of equality.")


if __name__ == "__main__":
    main()
