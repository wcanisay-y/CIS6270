"""Turn scored result files into comparison tables with uncertainty.

The verdict rests on paired contrasts, not on raw means:
  Idea A  at each NFE budget T, every selector is compared with the
          `confidence` baseline at the SAME T. The two controls that matter
          are contact_shuffled (same group structure, no structure
          information) and distance (trivial sequence spacing). If
          contact only beats confidence by as much as contact_shuffled
          does, the contact map contributed nothing.
  Idea B  every arm is compared with `flat` at matched parameters, data,
          budget and NFE. The control that matters is rand6: a random
          partition of the same group sizes.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def boot_diff(a, b, n=10000, seed=0):
    """Bootstrap CI and two-sided p for mean(a) - mean(b)."""
    a = np.asarray([v for v in a if np.isfinite(v)])
    b = np.asarray([v for v in b if np.isfinite(v)])
    if a.size < 2 or b.size < 2:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    rng = np.random.default_rng(seed)
    da = rng.choice(a, (n, a.size), replace=True).mean(1)
    db = rng.choice(b, (n, b.size), replace=True).mean(1)
    d = da - db
    obs = a.mean() - b.mean()
    # p from the bootstrap distribution of the difference, shifted to zero
    centred = d - d.mean()
    p = float((np.abs(centred) >= abs(obs)).mean())
    return dict(diff=float(obs), lo=float(np.quantile(d, 0.025)),
                hi=float(np.quantile(d, 0.975)), p=p)


def mwu(a, b):
    try:
        from scipy.stats import mannwhitneyu
        a = [v for v in a if np.isfinite(v)]
        b = [v for v in b if np.isfinite(v)]
        if len(a) < 3 or len(b) < 3:
            return np.nan
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def load(paths):
    cells = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  (missing {p})")
            continue
        d = json.load(open(p))
        for c in d["cells"]:
            c["_src"] = os.path.basename(p)
            cells.append(c)
    return cells


METRICS = [("plddt_mean", "pLDDT", "%7.2f"),
           ("ptm_mean", "pTM", "%6.3f"),
           ("pae_mean", "pAE", "%6.2f"),
           ("foldability", "fold%", "%6.3f"),
           ("pppl_mean", "ESM2-pppl", "%9.2f"),
           ("amphipathic_index", "amph", "%6.3f"),
           ("helix_power", "helixP", "%7.3f"),
           ("diversity", "div", "%6.3f"),
           ("novelty_kmer", "nov", "%6.3f"),
           ("uniqueness", "uniq", "%6.3f"),
           ("max_repeat", "rep", "%5.1f"),
           ("buried_hydrophobic", "burHyd", "%7.3f"),
           ("aa_jsd", "aaJSD", "%6.3f")]


def table(cells, key=lambda c: (c.get("selector"), c.get("T")), title=""):
    print(f"\n### {title}")
    hdr = "%-26s" % "cell" + "".join(
        "%*s" % (len(h) + 2, h) for _, h, _ in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for c in sorted(cells, key=lambda z: str(key(z))):
        name = f"{c.get('selector')}/T={c.get('T')}"
        if c.get("tau") not in (None, 0.3):
            name += f"/tau={c['tau']}"
        if c.get("temp") not in (None, 1.0) or c.get("fam_temp") not in (None, 1.0):
            name += f"/t={c.get('temp')},{c.get('fam_temp')}"
        if c.get("rho") not in (None, 0.3):
            name += f"/rho={c.get('rho')}"
        row = "%-26s" % name[:26]
        for k, h, fmt in METRICS:
            v = c.get(k, float("nan"))
            row += "%*s" % (len(h) + 2, (fmt % v) if v is not None
                            and np.isfinite(v) else "-")
        print(row)


def contrasts(cells, baseline_name, group_key, label):
    """Paired contrasts against a baseline within each group."""
    print(f"\n### {label}: pLDDT difference vs `{baseline_name}` "
          f"(bootstrap 95% CI, p from bootstrap / Mann-Whitney)")
    groups = {}
    for c in cells:
        groups.setdefault(group_key(c), []).append(c)
    hdr = ("%-22s %6s %8s %18s %9s %9s" %
           ("selector", "group", "diff", "95% CI", "p_boot", "p_mwu"))
    print(hdr); print("-" * len(hdr))
    for g in sorted(groups, key=str):
        gc = groups[g]
        base = [c for c in gc if c.get("selector") == baseline_name
                or c.get("tag") == baseline_name]
        if not base:
            continue
        bl = base[0].get("plddt_list") or []
        for c in sorted(gc, key=lambda z: str(z.get("selector"))):
            nm = c.get("selector")
            if nm == baseline_name:
                continue
            r = boot_diff(c.get("plddt_list") or [], bl)
            p2 = mwu(c.get("plddt_list") or [], bl)
            star = "" if not np.isfinite(r["p"]) else (
                "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01
                else "*" if r["p"] < 0.05 else "")
            print("%-22s %6s %8.2f %18s %9.4f %9.4f %s"
                  % (str(nm)[:22], str(g), r["diff"],
                     "[%.2f, %.2f]" % (r["lo"], r["hi"]), r["p"], p2, star))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["A", "B"], required=True)
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    cells = load(args.files)
    print(f"loaded {len(cells)} cells")
    if not cells:
        return

    if args.which == "A":
        main_cells = [c for c in cells if c.get("note") != "tau_sweep"]
        table(main_cells, title="Idea A: all cells")
        contrasts([c for c in main_cells if c.get("selector") != "natural"],
                  "confidence", lambda c: c.get("T"),
                  "Idea A, within each NFE budget T")
        sw = [c for c in cells if c.get("note") == "tau_sweep"]
        if sw:
            table(sw, title="Idea A: threshold sweep at T=25")
        print("\n### mechanism diagnostic: coupled pairs actually committed")
        print("%-20s %5s %18s %18s" % ("selector", "T", "coupled_pair_frac",
                                       "within_coupling"))
        for c in sorted(main_cells, key=lambda z: (str(z.get("selector")),
                                                   z.get("T") or 0)):
            if c.get("coupled_pair_frac") is None:
                continue
            print("%-20s %5s %18.3f %18.4f"
                  % (c["selector"], c["T"], c["coupled_pair_frac"],
                     c["within_coupling"]))
        print("\n### cost: generator NFE vs contact-model calls")
        print("%-20s %5s %9s %13s" % ("selector", "T", "nfe_gen",
                                      "nfe_contact"))
        for c in sorted(main_cells, key=lambda z: (str(z.get("selector")),
                                                   z.get("T") or 0)):
            if "nfe_gen" in c:
                print("%-20s %5s %9d %13d" % (c["selector"], c["T"],
                                              c["nfe_gen"],
                                              c.get("nfe_contact", 0)))
    else:
        for c in cells:
            if c.get("tag"):
                c["selector"] = c["tag"] if c.get("kind") == "uncond" \
                    else f"{c['tag']}|{c.get('kind')}"
        table(cells, title="Idea B: all cells")
        uncond = [c for c in cells
                  if c.get("kind") == "uncond" and c.get("temp") == 1.0
                  and c.get("fam_temp") in (None, 1.0)] + \
                 [c for c in cells if c.get("selector") == "natural"]
        table(uncond, title="Idea B: matched setting (temp=1.0, T=100)")
        contrasts(uncond, "flat", lambda c: "matched",
                  "Idea B, matched setting")
        nfe = [c for c in cells if c.get("kind") == "nfe"]
        if nfe:
            contrasts(nfe, "flat|nfe", lambda c: c.get("T"),
                      "Idea B, within each NFE budget")
        cond = [c for c in cells if c.get("kind") == "conditioned"]
        if cond:
            print("\n### controllability: family-pattern compliance")
            for c in cond:
                print("%-24s compliance %.4f  pLDDT %.2f"
                      % (c["tag"], c.get("family_compliance", float("nan")),
                         c.get("plddt_mean", float("nan"))))


if __name__ == "__main__":
    main()
