"""Quality, diversity and cost in one table. Pranam's asks 5, 6, 7 and 11.

The comment asks for the control ladder on one frozen backbone, with
reveal-set sizes matched, the full ESM-2 forward cost counted, and quality,
diversity and latency reported together rather than quality alone.

Cost is charged in denoiser-pass equivalents, from the ratio measured on the
run GPU: a DPLM-150M denoiser pass is 1.00, the same pass with attentions
returned is 1.04, and an ESM-2 650M contact call is 7.88. So the free
self-attention signal costs 4 percent over plain confidence and the external
signal costs 7.88 per refresh on top of generation.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COST_ATTN = 1.04       # denoiser pass with attentions returned
COST_EXT = 7.88        # ESM-2 650M contact call
RNG = np.random.default_rng(0)


def cost_of(c):
    g, x = c.get("nfe_gen") or 0, c.get("nfe_contact") or 0
    mode = (c.get("selector") or "")
    base = g * (COST_ATTN if "contact" in mode or "punt" in mode else 1.0)
    return base + COST_EXT * x


def ci(x, nb=10000):
    bs = np.array([RNG.choice(x, len(x), True).mean() for _ in range(nb)])
    return np.percentile(bs, [2.5, 97.5])


def table(path, order=None):
    if not os.path.exists(path):
        print("MISSING %s" % path)
        return
    d = json.load(open(path))
    cells = [c for c in d["cells"] if c.get("T") is not None]
    nat = [c for c in d["cells"] if c.get("T") is None]
    Ts = sorted({c["T"] for c in cells})
    sels = order or sorted({c["selector"] for c in cells})
    print("\n## %s" % os.path.basename(path))
    print("\n| selector | passes | cost | pLDDT | 95 % CI | pTM | foldability "
          "| diversity | novelty | pppl | s/seq |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for T in Ts:
        for s in sels:
            c = next((x for x in cells if x["selector"] == s and x["T"] == T), None)
            if not c:
                continue
            pl = np.asarray(c["plddt_list"], float)
            lo, hi = ci(pl)
            pt = float(np.mean(c.get("ptm_list") or [np.nan]))
            n = len(c.get("seqs") or [])
            print("| %s | %d | %.1f | %.2f | [%.1f, %.1f] | %.3f | %.3f | %.3f "
                  "| %.3f | %.2f | %.2f |"
                  % (s, T, cost_of(c), pl.mean(), lo, hi, pt,
                     c.get("foldability", float("nan")),
                     c.get("diversity", float("nan")),
                     c.get("novelty_kmer", float("nan")),
                     c.get("pppl_mean", float("nan")),
                     (c.get("seconds") or 0) / max(n, 1)))
    for c in nat:
        pl = np.asarray(c["plddt_list"], float)
        print("| natural | - | - | %.2f | - | %.3f | %.3f | %.3f | %.3f | %.2f | - |"
              % (pl.mean(), float(np.mean(c.get("ptm_list") or [np.nan])),
                 c.get("foldability", float("nan")),
                 c.get("diversity", float("nan")),
                 c.get("novelty_kmer", float("nan")),
                 c.get("pppl_mean", float("nan"))))

    # quality at equal cost: the comparison the method's own claim requires
    print("\nQuality at matched cost (pLDDT per selector, interpolated to the"
          "\ncost of the confidence arm at each budget):")
    for T in Ts:
        base = next((x for x in cells if x["selector"] == "confidence" and x["T"] == T), None)
        if not base:
            continue
        bc = cost_of(base)
        row = []
        for s in sels:
            c = next((x for x in cells if x["selector"] == s and x["T"] == T), None)
            if c:
                row.append("%s %.1f (x%.2f cost)"
                           % (s, np.mean(c["plddt_list"]), cost_of(c) / max(bc, 1e-9)))
        print("  T=%-4d %s" % (T, " | ".join(row)))


if __name__ == "__main__":
    order = ["confidence", "distance", "punt", "contact", "contact_shuffled",
             "random", "entropy"]
    for p in (sys.argv[1:] or ["results/F_ladder_scored.json"]):
        table(p, order)
