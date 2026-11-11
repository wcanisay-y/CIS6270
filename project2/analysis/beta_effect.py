"""Does the beta-fold contact effect hold up as more than a three-class split?

Section 6.3 of REPORT_REVISED reported a significant beta-minus-alpha
interaction against the shuffled control. Three classes is three looks, and
11 beta chains is thin, so this re-tests the claim the way it would have to
survive to become a method:

  continuous   per-chain effect against per-chain contact density, which is
               the mechanism the class split is only a proxy for. A method
               needs a continuous decision rule, not a fold label, because
               the fold label is not available at design time either.
  polarity     if dense contacts are what the gate exploits, the reversed
               polarity (co-commit contacting positions) should be worse on
               dense chains, not better. A mechanism that only works one way
               round in one direction is a mechanism; one that helps both
               ways is a density confound.
  budget       the effect pooled within chain across budgets, and per budget,
               so a single lucky budget cannot carry it.
  power        how many chains a real effect of this size would need.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLASS = {1: "mainly-alpha", 2: "mainly-beta", 3: "alpha-beta", 4: "few-ss"}
RNG = np.random.default_rng(0)
NB = 20000


def load(scored, npz):
    d = json.load(open(scored))
    cells = [c for c in d["cells"] if c.get("T") is not None]
    z = np.load(npz, allow_pickle=True)
    maps, cath = z["maps"], z["cath"]
    n = maps.shape[0]
    L = maps.shape[1]
    iu = np.triu_indices(L, k=6)
    dens = np.array([maps[i][iu].mean() for i in range(n)])
    cls = np.array([int(c[0]) for c in cath])
    by = {}
    for c in cells:
        pl = np.asarray(c["plddt_list"], float)
        if len(pl) != n:
            continue
        by[(c["selector"], c["T"])] = pl
    Ts = sorted({k[1] for k in by})
    return by, Ts, dens, cls, n


def pooled(by, Ts, a, b, n):
    """Per-chain mean over budgets of (arm a - arm b)."""
    d = np.zeros((len(Ts), n))
    for i, T in enumerate(Ts):
        d[i] = by[(a, T)] - by[(b, T)]
    return d.mean(0)


def boot_mean(x):
    bs = np.array([RNG.choice(x, len(x), replace=True).mean() for _ in range(NB)])
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def boot_corr(x, y):
    r = np.corrcoef(x, y)[0, 1]
    bs = []
    for _ in range(NB):
        i = RNG.integers(0, len(x), len(x))
        if np.std(x[i]) == 0 or np.std(y[i]) == 0:
            continue
        bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return r, *np.percentile(bs, [2.5, 97.5])


def perm_slope(x, y, nperm=20000):
    """Permutation p for the slope of y on x."""
    s = np.polyfit(x, y, 1)[0]
    cnt = sum(abs(np.polyfit(x, RNG.permutation(y), 1)[0]) >= abs(s)
              for _ in range(nperm))
    return s, (cnt + 1) / (nperm + 1)


def main(scored="results/A_oracle_cath_scored2.json",
         npz="results/A_oracle_cath.npz"):
    by, Ts, dens, cls, n = load(scored, npz)
    print("%d chains, budgets %s" % (n, Ts))
    print("contact density: mean %.4f  range %.4f-%.4f" % (dens.mean(), dens.min(), dens.max()))
    for k in sorted(set(cls)):
        print("  %-14s n=%2d  density %.4f" % (CLASS.get(k, k), (cls == k).sum(), dens[cls == k].mean()))

    eff = pooled(by, Ts, "contact_oracle", "contact_oracle_shuffled", n)
    eff_c = pooled(by, Ts, "contact_oracle", "confidence", n)
    has_anti = ("anti_contact_oracle", Ts[0]) in by
    anti = pooled(by, Ts, "anti_contact_oracle", "contact_oracle_shuffled", n) if has_anti else None

    print("\n=== 1. CONTINUOUS: per-chain effect vs per-chain contact density ===")
    for nm, e in (("true - shuffled", eff), ("true - confidence", eff_c)):
        r, lo, hi = boot_corr(dens, e)
        s, p = perm_slope(dens, e)
        print("  %-18s r = %+.3f [%+.3f, %+.3f]   slope %+.1f pLDDT per unit density, perm p = %.4f"
              % (nm, r, lo, hi, s, p))
    if anti is not None:
        r, lo, hi = boot_corr(dens, anti)
        print("  %-18s r = %+.3f [%+.3f, %+.3f]   <-- reversed polarity"
              % ("anti - shuffled", r, lo, hi))

    print("\n=== 2. SPLIT BY DENSITY, ignoring the fold label ===")
    med = np.median(dens)
    hi_m, lo_m = dens > med, dens <= med
    for nm, e in (("true - shuffled", eff), ("true - confidence", eff_c)):
        a, b = e[hi_m], e[lo_m]
        diff = a.mean() - b.mean()
        bs = np.array([RNG.choice(a, len(a), True).mean() - RNG.choice(b, len(b), True).mean()
                       for _ in range(NB)])
        print("  %-18s dense %+.2f (n=%d)   sparse %+.2f (n=%d)   diff %+.2f [%+.2f, %+.2f]"
              % (nm, a.mean(), len(a), b.mean(), len(b), diff, *np.percentile(bs, [2.5, 97.5])))

    print("\n=== 3. PER BUDGET: can one budget carry it? ===")
    print("  %-6s %-22s %-22s" % ("T", "beta - alpha (t-shuf)", "dense - sparse (t-shuf)"))
    for T in Ts:
        d = by[("contact_oracle", T)] - by[("contact_oracle_shuffled", T)]
        ba = d[cls == 2].mean() - d[cls == 1].mean()
        ds = d[hi_m].mean() - d[lo_m].mean()
        print("  %-6d %+-22.2f %+-22.2f" % (T, ba, ds))

    print("\n=== 4. CLASS MEANS, for reference ===")
    for k in sorted(set(cls)):
        m = cls == k
        print("  %-14s n=%2d  true-shuf %+.2f   true-conf %+.2f" % (CLASS.get(k, k), m.sum(), eff[m].mean(), eff_c[m].mean()))

    print("\n=== 5. POWER: what would it take to pin this down? ===")
    sd = eff.std(ddof=1)
    print("  per-chain sd of (true - shuffled), pooled over budgets: %.2f" % sd)
    for target in (11.6, 8.0, 5.0, 3.0):
        nreq = 2 * (2.8 * sd / target) ** 2
        print("  chains per group for 80%% power at a %.1f pLDDT group difference: %.0f" % (target, np.ceil(nreq)))

    print("\n=== 6. THE WHOLE-SET QUESTION: does the gate help on average? ===")
    for nm, e in (("true - shuffled", eff), ("true - confidence", eff_c)):
        print("  %-18s %+.2f [%+.2f, %+.2f] over all %d chains" % (nm, *boot_mean(e), n))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
