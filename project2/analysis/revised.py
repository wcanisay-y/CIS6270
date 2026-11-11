"""Tables for the oracle redo and the revised method.

Two reports:

  oracle   pLDDT by (map source, selector, T). The decisive column is the
           true-minus-shuffled difference WITHIN each map source, and the
           decisive comparison across sources is whether that difference
           changes when the map comes from a deposited structure rather
           than from ESMFold, and when CB is the real CB rather than the
           backbone oxygen the earlier run used by mistake.

  revised  pLDDT against COST, where cost charges the external contact
           passes through the ratio measured on the GPU. The method claims
           time at equal quality, so a quality table alone cannot judge it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(p):
    with open(p) as fh:
        return json.load(fh)


def _ci(c):
    lo, hi = c.get("plddt_ci", (float("nan"), float("nan")))
    return "[%.1f, %.1f]" % (lo, hi)


def oracle_report(srcs, outdir="results"):
    lines = ["# Oracle redo: deposited structures vs ESMFold vs the oxygen bug",
             "",
             "Each cell is 48 sequences at length 200, matched forward passes.",
             "`true` hands the selector the contact map of a real protein;",
             "`shuffled` permutes that same matrix, preserving its value",
             "distribution and destroying only the association between",
             "coupling and position. The oracle map is supplied once and never",
             "refreshed, so this measures the mechanism, not contact quality.",
             ""]
    summary = {}
    for name, path in srcs.items():
        if not os.path.exists(path):
            lines += ["## %s" % name, "", "MISSING: %s" % path, ""]
            continue
        cells = [c for c in load(path)["cells"]
                 if c.get("T") is not None]
        by = {}
        for c in cells:
            by[(c["selector"], c["T"])] = c
        Ts = sorted({c["T"] for c in cells})
        lines += ["## map source: %s" % name, "",
                  "| forward passes | confidence | true map | shuffled map "
                  "| true - shuffled |", "|---|---|---|---|---|"]
        diffs = []
        for T in Ts:
            base = by.get(("confidence", T))
            tru = by.get(("contact_oracle", T))
            shf = by.get(("contact_oracle_shuffled", T))
            if not (tru and shf):
                continue
            d = tru["plddt_mean"] - shf["plddt_mean"]
            diffs.append(d)
            lines.append("| %d | %.2f | %.2f %s | %.2f %s | %+.2f |"
                         % (T, base["plddt_mean"] if base else float("nan"),
                            tru["plddt_mean"], _ci(tru),
                            shf["plddt_mean"], _ci(shf), d))
        mean_d = float(np.mean(diffs)) if diffs else float("nan")
        summary[name] = dict(mean_true_minus_shuffled=mean_d,
                             per_T=diffs)
        lines += ["",
                  "Mean true-minus-shuffled across budgets: **%+.2f pLDDT**."
                  % mean_d, ""]
        # does the oracle beat the free baseline at all
        vs = []
        for T in Ts:
            base = by.get(("confidence", T))
            tru = by.get(("contact_oracle", T))
            if base and tru:
                vs.append(tru["plddt_mean"] - base["plddt_mean"])
        if vs:
            summary[name]["mean_true_minus_confidence"] = float(np.mean(vs))
            lines += ["Mean true-map-minus-confidence: **%+.2f pLDDT**."
                      % np.mean(vs), ""]
    lines += ["## Verdict", ""]
    if len(summary) >= 2:
        ks = list(summary)
        lines.append("| map source | true - shuffled | true - confidence |")
        lines.append("|---|---|---|")
        for k in ks:
            lines.append("| %s | %+.2f | %+.2f |"
                         % (k, summary[k]["mean_true_minus_shuffled"],
                            summary[k].get("mean_true_minus_confidence",
                                           float("nan"))))
        lines.append("")
    p = os.path.join(outdir, "A_oracle_redo.md")
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    json.dump(summary, open(os.path.join(outdir, "A_oracle_redo.json"), "w"),
              indent=2)
    print("\n".join(lines))
    return summary


def revised_report(path, outdir="results", tag=""):
    if not os.path.exists(path):
        print("MISSING: %s" % path)
        return {}
    d = load(path)
    # scored files carry a `natural` reference cell with T=None
    cells = [c for c in d["cells"] if c.get("T") is not None]
    ratio = d.get("config", {}).get("_ratio")
    cost = None
    for c in cells:
        if "cost_eq" in c:
            cost = True
    lines = ["# The revised method: warm-up, cached map, trigger refresh", ""]
    rt = load(path).get("cost") or {}
    if rt:
        lines += ["Measured on the run GPU: %.4f s per denoiser call, "
                  "%.4f s per external ESM-2 call, ratio **%.2f**."
                  % (rt.get("gen_s", float('nan')),
                     rt.get("contact_s", float('nan')),
                     rt.get("ratio", float('nan'))), ""]

    def row(c):
        return ("| %s | %s | %.2f | %s | %.0f | %.0f | %.1f | %.2f | %.0f |"
                % (c["selector"], c["contact_mode"] if "contact_mode" in c
                   else c.get("mode", "?"),
                   c.get("warmup", 0.0),
                   ("%d" % c["g_max"]) if c.get("g_max") else "-",
                   c["nfe_gen"], c["nfe_contact"], c["cost_eq"],
                   c["plddt_mean"], c["refreshes"]))

    hdr = ["| selector | signal | warm-up | cap | gen NFE | ESM NFE | "
           "cost | pLDDT | refreshes |", "|---|---|---|---|---|---|---|---|---|"]

    Ts = sorted({c["T"] for c in cells})
    for T in Ts:
        sub = [c for c in cells if c["T"] == T
               and c.get("family") in ("baseline", "warmup", "control",
                                       "free", "free_control")]
        if not sub:
            continue
        lines += ["## T = %d" % T, ""] + hdr
        for c in sorted(sub, key=lambda x: (x.get("family"), x.get("warmup"))):
            lines.append(row(c))
        base = next((c for c in sub if c.get("family") == "baseline"), None)
        if base:
            best = max((c for c in sub if c.get("family") == "warmup"),
                       key=lambda x: x["plddt_mean"], default=None)
            free = next((c for c in sub if c.get("family") == "free"), None)
            lines += [""]
            if best:
                lines.append("Best warm-up arm is %+.2f pLDDT against the "
                             "free confidence baseline, at %.1fx its cost."
                             % (best["plddt_mean"] - base["plddt_mean"],
                                best["cost_eq"] / max(base["cost_eq"], 1e-9)))
            if free:
                lines.append("The free self-attention signal is %+.2f pLDDT "
                             "against the same baseline at %.1fx cost."
                             % (free["plddt_mean"] - base["plddt_mean"],
                                free["cost_eq"] / max(base["cost_eq"], 1e-9)))
            lines.append("")

    for fam, title in (("refresh", "Refresh policy"),
                       ("gmax", "Group-size cap"),
                       ("gmax_baseline", "Group-size cap, baseline")):
        sub = [c for c in cells if c.get("family") == fam]
        if not sub:
            continue
        lines += ["## %s" % title, ""] + hdr
        for c in sub:
            lines.append(row(c) + (" <!-- %s -->" % c.get("tag", "")))
        lines.append("")

    p = os.path.join(outdir, "C_revised_report%s.md" % tag)
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return cells


def reparam_report(warmups, outdir="results"):
    """Warm-up sweep inside DPLM's own remasking decoder.

    This is the one that matters for a real system: the absorbing decoder
    sits about 30 pLDDT lower, so a rule that only helps there is a rule
    about a decoder nobody ships.
    """
    lines = ["# Warm-up inside the remasking decoder (the one DPLM ships)",
             "",
             "Free self-attention signal, quantile threshold at 0.90, 48",
             "sequences per cell. `gated` counts steps that fell back to",
             "confidence because the warm-up had not been reached, so those",
             "steps also made no contact call.",
             ""]
    table = {}
    for w in warmups:
        p = os.path.join(outdir, "C_reparam_w%s_scored.json" % w)
        if not os.path.exists(p):
            lines += ["warmup %s: MISSING (%s)" % (w, p), ""]
            continue
        cells = load(p)["cells"]
        for c in cells:
            # the scored files carry a `natural` reference cell with T=None
            if c.get("T") is None:
                continue
            table[(w, c["selector"], c["T"])] = c
    Ts = sorted({k[2] for k in table})
    sels = ["reparam_confidence", "reparam_contact",
            "reparam_contact_shuffled"]
    lines += ["| warm-up | forward passes | confidence | contact | shuffled "
              "| contact - confidence | contact - shuffled |",
              "|---|---|---|---|---|---|---|"]
    best = None
    for w in warmups:
        for T in Ts:
            row = [table.get((w, s, T)) for s in sels]
            if not all(row):
                continue
            b, ct, sh = [r["plddt_mean"] for r in row]
            d1, d2 = ct - b, ct - sh
            if best is None or d1 > best[0]:
                best = (d1, w, T)
            lines.append("| %s | %d | %.2f | %.2f | %.2f | %+.2f | %+.2f |"
                         % (w, T, b, ct, sh, d1, d2))
    lines.append("")
    if best:
        lines.append("Best cell for the contact rule against the free "
                     "confidence baseline: %+.2f pLDDT at warm-up %s, %d "
                     "forward passes." % (best[0], best[1], best[2]))
        lines.append("")
    p = os.path.join(outdir, "C_reparam_report.md")
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return table


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "revised", "reparam", "all"],
                    default="all")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    if a.mode in ("oracle", "all"):
        oracle_report({
            "deposited CB-CB (X-ray)": "results/A_oracle_pdb_scored.json",
            "ESMFold CB-CB (fixed)": "results/A_oracle_esmfold_scored.json",
            "ESMFold O-O (earlier bug)": "results/A_oracle_oxygen_scored.json",
        }, outdir=a.outdir)
    if a.mode in ("revised", "all"):
        revised_report("results/C_revised_scored.json", outdir=a.outdir)
        if os.path.exists("results/C_revised_self_scored.json"):
            print("\n\n### free-signal warm-up sweep ###\n")
            revised_report("results/C_revised_self_scored.json",
                           outdir=a.outdir, tag="_self")
    if a.mode in ("reparam", "all"):
        reparam_report(["0.0", "0.35", "0.5", "0.75"], outdir=a.outdir)
