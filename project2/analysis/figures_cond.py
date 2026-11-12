"""Figures for the conditioning result -- the one positive finding in project 2.

The six figures in `analysis/figures.py` were all written on 2026-11-06, before
the 2026-11-09 metric fixes, and none of them plots conditioning at all: fig4
and fig6 show the UNCONDITIONAL two-level arms, which is the Idea-B negative.
These two draw the Round 4 and Round 5 numbers (2026-11-11).

  figure 7  the headline contrast with its bit-matched random control, before
            and after deleting the pattern sources' homologs from training, on
            both pattern-source datasets; and what moves in absolute pLDDT.
  figure 8  the accuracy budget -- the gain is gone by 19 % pattern error, and
            nothing available predicts a six-class chemical pattern that well.

Statistics are NOT recomputed here. The panels read the summary JSONs that
`analysis/{clustered,cross_dataset,pattern_curve}.py` write, so a figure cannot
silently disagree with the note beside it. The one exception is the per-point
gain interval in 8A, computed with the same bootstrap helper and seed that
`pattern_curve.py` uses.

Palette, marks and axes follow `analysis/figures.py`: the eight-slot
categorical order assigned in fixed order (blue = chemical arm, orange = the
random-partition control, in both figures), one y-axis per panel, thin marks,
recessive grid, legend plus selective direct labels.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from analysis.figures import C, INK, INK2, MUTED, GRID, FIG, RES, style  # noqa: E402

CHEM, RAND = C[0], C[1]
RNG = np.random.default_rng(0)
NB = 20000


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(x):
    """The bootstrap used by analysis/pattern_curve.py, same seed and draws."""
    x = np.asarray(x, float)
    bs = np.array([RNG.choice(x, len(x), True).mean() for _ in range(NB)])
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def cells(name):
    d = load(name)
    return {c["selector"]: c for c in d["cells"] if "plddt_list" in c} if d \
        else {}


# ---------------------------------------------------------------- figure 7

def fig_conditioning():
    J, K = load("J_clustered_summary.json"), load("K_clustered_summary.json")
    if not J or not K:
        print("fig7: missing J/K summaries")
        return
    sets = [("SwissProt dev", J), ("CATH S35", K)]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))

    # -- A: the paired contrast, original vs purged, both datasets ----------
    ax = axes[0]
    rows, y = [], []
    for ds, S in sets:
        for r in S["rows"]:                      # 'original' then 'purged'
            rows.append((ds, r))
    y = np.arange(len(rows))[::-1]
    for i, (ds, r) in enumerate(rows):
        purged = r["label"] == "purged"
        ax.plot([r["lo"], r["hi"]], [y[i]] * 2, color=CHEM, lw=2.0,
                solid_capstyle="round", zorder=3,
                alpha=1.0 if purged else 0.55)
        ax.plot([r["chem_minus_rand"]], [y[i]], "o", ms=7 if purged else 6,
                mfc=CHEM if purged else "white", mec=CHEM if purged else CHEM,
                mew=1.6, zorder=4)
        ax.annotate(f"{r['chem_minus_rand']:+.1f}",
                    (r["chem_minus_rand"], y[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{ds}, {r['label']}" for ds, r in rows], fontsize=7.5)
    # memorisation: the paired difference of differences, per dataset
    for k, (ds, S) in enumerate(sets):
        d = S["dod"]
        yy = (y[2 * k] + y[2 * k + 1]) / 2
        ax.annotate(f"training homologs\nworth {d['mean']:+.1f} "
                    f"[{d['lo']:+.1f}, {d['hi']:+.1f}]",
                    (0.02, yy), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 0), ha="left",
                    va="center", fontsize=6.6, color=MUTED)
    ax.set_xlim(0, 34)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    style(ax, "chemical minus random, pLDDT (paired per source)", "",
          "Filled = homologs purged from training")

    # -- B: what actually moves, purged corpora ----------------------------
    ax = axes[1]
    bars = []
    for ds, S in sets:
        r = [x for x in S["rows"] if x["label"] == "purged"][0]
        bars.append((f"{ds}\nchemical", r["chem_uncond"], r["chem_cond"],
                     CHEM, r["chem_fold"]))
        bars.append((f"{ds}\nrandom", r["rand_uncond"], r["rand_cond"],
                     RAND, r["rand_fold"]))
    y = np.arange(len(bars))[::-1]
    for i, (lab, u, c, col, fold) in enumerate(bars):
        ax.annotate("", xy=(c, y[i]), xytext=(u, y[i]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color=col,
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=11))
        ax.plot([u], [y[i]], "o", ms=5, mfc="white", mec=col, mew=1.4,
                zorder=4)
        txt = f"{c:.1f}  ({c - u:+.1f})   fold {fold:.2f}"
        if c > u:
            ax.annotate(txt, (c, y[i]), textcoords="offset points",
                        xytext=(10, 0), ha="left", va="center",
                        fontsize=7.4, color=INK2)
        else:                      # leftward arrow: label above, not into the axis
            ax.annotate(txt, (c, y[i]), textcoords="offset points",
                        xytext=(0, 10), ha="center", va="bottom",
                        fontsize=7.4, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=7)
    ax.set_xlim(26, 97)
    ax.set_ylim(-0.7, len(bars) - 0.3)
    style(ax, "ESMFold pLDDT, unconditional to conditioned", "",
          "Open circle = unconditional baseline")
    h = [plt.Line2D([], [], color=CHEM, lw=2.0, label="chemical families"),
         plt.Line2D([], [], color=RAND, lw=2.0,
                    label="random partition, bit-matched")]
    ax.legend(handles=h, loc="lower right", fontsize=7,
              borderaxespad=0.3)

    fig.suptitle("Conditioning on a real protein's six-class chemical pattern",
                 x=0.005, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig7_conditioning.png"))
    fig.savefig(os.path.join(FIG, "fig7_conditioning.pdf"))
    plt.close(fig)
    print("fig7 written")


# ---------------------------------------------------------------- figure 8

def fig_pattern_budget():
    S = load("I_curve_dms6_summary.json")
    R = load("I_curve_rand6_summary.json")
    D = load("I_curve_dms6_diag.json")
    if not S or not R:
        print("fig8: missing curve summaries")
        return
    A = cells("I_curve_dms6_scored.json")
    base = cells("G_cond_clean_scored.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # -- A: the degradation curve, chemical against a random partition -----
    ax = axes[0]
    ax.axvspan(0, S["f_zero"], color=CHEM, alpha=0.06, zorder=0)
    ax.axvline(S["f_half"], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.axvline(S["f_zero"], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)

    for (summ, name, col) in ((S, "chemical families", CHEM),
                              (R, "random partition", RAND)):
        fs = np.asarray(summ["fracs"], float)
        gs = np.asarray(summ["gains"], float)
        ax.plot(fs, gs, "-o", color=col, lw=2.0, ms=5, mec="white", mew=0.9,
                label=name, zorder=3)
    # intervals on the chemical curve, same bootstrap as the note
    if A and base and "dms6|uncond" in base:
        u = float(np.mean(base["dms6|uncond"]["plddt_list"]))
        for k, c in sorted([(c["corrupt_frac"], c) for c in A.values()
                            if c.get("kind") == "corrupt"]):
            g, lo, hi = ci(np.asarray(c["plddt_list"], float) - u)
            ax.plot([k, k], [lo, hi], color=CHEM, lw=1.2, alpha=0.5, zorder=2)

    ax.annotate(f"{S['gains'][0]:+.1f} exact", (0.0, S["gains"][0]),
                textcoords="offset points", xytext=(9, -1), fontsize=7.2,
                color=INK2, va="center")
    ax.annotate(f"{R['gains'][0]:+.1f}", (0.0, R["gains"][0]),
                textcoords="offset points", xytext=(9, -10), fontsize=7.2,
                color=INK2)
    ax.annotate(f"half the gain is gone at {100 * S['f_half']:.1f} % error,\n"
                f"all of it by {100 * S['f_zero']:.1f} %",
                (0.30, 0.62 * S["gains"][0]), fontsize=7.0, color=MUTED,
                va="center")
    ax.set_xlim(-0.03, 1.03)
    style(ax, "fraction of the pattern wrong", "pLDDT gain over unconditional",
          f"Slope {S['slope_per_pct']:.2f} per 1 % error, against "
          f"{R['slope_per_pct']:.2f}")
    ax.legend(loc="upper right", fontsize=7.2)

    # -- B: the budget against what any predictor achieves -----------------
    ax = axes[1]
    ax.axvspan(0, 100 * S["f_zero"], color=CHEM, alpha=0.10, zorder=0)
    ax.axvspan(0, 100 * S["f_half"], color=CHEM, alpha=0.14, zorder=1)
    ax.annotate(f"affordable:\nany gain at all\nbelow {100 * S['f_zero']:.0f} %",
                (100 * S["f_zero"], 2.93), textcoords="offset points",
                xytext=(5, 0), fontsize=6.8, color=MUTED, va="top")

    marks = []
    if D:
        fa = D["family_accuracy"]
        acc = fa["by_reveal"]["0.0"]["acc"]
        marks.append(("the family head, argmax", 100 * (1 - acc)))
        marks.append(("a pattern it samples",
                      100 * (1 - fa["marginal_match_floor"])))
    y = np.arange(len(marks))[::-1] + 1.0
    for i, (lab, v) in enumerate(marks):
        ax.plot([v], [y[i]], "o", ms=8, mfc=RAND, mec="white", mew=1.2,
                zorder=4)
        ax.annotate(f"{lab}: {v:.0f} %", (v, y[i]),
                    textcoords="offset points", xytext=(-11, 0), ha="right",
                    va="center", fontsize=7.4, color=INK2)
        if S["f_zero"] > 0:
            ax.annotate(f"{v / (100 * S['f_zero']):.1f}x the budget",
                        (v, y[i]), textcoords="offset points",
                        xytext=(12, 0), ha="left", va="center", fontsize=6.8,
                        color=MUTED)

    # the direct test: the model's own pattern through the same path
    gain = None
    if A and base and "dms6|uncond" in base and "dms6|selfpattern" in A:
        u = float(np.mean(base["dms6|uncond"]["plddt_list"]))
        gain = ci(np.asarray(A["dms6|selfpattern"]["plddt_list"], float) - u)
    if gain:
        ax.annotate("the direct test -- the model's own pattern through the\n"
                    f"identical conditioning path is worth {gain[0]:+.2f} "
                    f"[{gain[1]:+.2f}, {gain[2]:+.2f}],\n"
                    f"i.e. {100 * gain[0] / S['gains'][0]:.0f} % of the "
                    f"{S['gains'][0]:+.1f} an exact pattern buys",
                    (0.26, 0.04), xycoords="axes fraction", fontsize=6.9,
                    color=INK2, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3.1)
    ax.set_yticks([])
    style(ax, "pattern error rate, %", "",
          "What is affordable, against what is achieved")

    fig.suptitle("A coarse chemical pattern is only useful while it is nearly "
                 "exact", x=0.005, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig8_pattern_budget.png"))
    fig.savefig(os.path.join(FIG, "fig8_pattern_budget.pdf"))
    plt.close(fig)
    print("fig8 written")


# ---------------------------------------------------------------- figure 9

ARMS = [("dms6|conditioned", "chemical,\nconditioned", CHEM),
        ("rand6|conditioned", "random,\nconditioned", RAND),
        ("dms6|uncond", "chemical,\nunconditional", CHEM),
        ("rand6|uncond", "random,\nunconditional", RAND)]


def _sel(name):
    d = load(name)
    return {c["selector"]: c for c in d["cells"]} if d else {}


def fig_independent_metrics():
    """What the metrics that are NOT ESMFold say about the same sequences."""
    S = _sel("L_extra_J_purged.json")
    K = _sel("L_extra_K_purged.json")
    MS = _sel("L_maxid_J_purged.json")
    MO = _sel("L_maxid_J_orig.json")
    R = _sel("M_reftm_K_purged.json")
    RP = _sel("M_reftm_K_purged_perm.json")
    if not S:
        print("fig9: nothing scored yet")
        return

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4))

    # -- A: FPD, the distributional distance to natural -------------------
    ax = axes[0][0]
    y = np.arange(len(ARMS))[::-1]
    for i, (sel, lab, col) in enumerate(ARMS):
        for k, (D, hatch, nm) in enumerate(((S, None, "SwissProt"),
                                            (K, "///", "CATH S35"))):
            v = (D.get(sel) or {}).get("fpd_vs_sources")
            if v is None:
                continue
            ax.barh(y[i] + (0.19 if k == 0 else -0.19), v, height=0.34,
                    color=col, alpha=1.0 if k == 0 else 0.55,
                    hatch=hatch, edgecolor="white", lw=0.8, zorder=3)
            ax.annotate(f"{v:.1f}", (v, y[i] + (0.19 if k == 0 else -0.19)),
                        textcoords="offset points", xytext=(4, 0),
                        va="center", fontsize=6.8, color=INK2)
    cal = (S.get("natural") or {}).get("fpd_vs_sources")
    if cal:
        ax.axvline(cal, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"natural vs natural {cal:.1f}", (cal, y[0] + 0.52),
                    textcoords="offset points", xytext=(5, 0), fontsize=6.6,
                    color=MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([a[1] for a in ARMS], fontsize=7)
    style(ax, "FPD to the pattern sources (lower = more natural)", "",
          "ESM-2 embedding space, not ESMFold")
    h = [plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.9,
                       label="SwissProt"),
         plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.5, hatch="///",
                       label="CATH S35")]
    ax.legend(handles=h, loc="upper right", fontsize=6.8)
    ax.set_xlim(0, 34)
    ax.set_ylim(-0.75, len(ARMS) - 0.15)

    # -- B: NSR against its own floors ------------------------------------
    ax = axes[0][1]
    rows = [(S, "dms6|conditioned", "SwissProt\nchemical", CHEM),
            (S, "rand6|conditioned", "SwissProt\nrandom", RAND),
            (K, "dms6|conditioned", "CATH\nchemical", CHEM),
            (K, "rand6|conditioned", "CATH\nrandom", RAND)]
    rows = [r for r in rows if r[0].get(r[1])]
    y = np.arange(len(rows))[::-1]
    for i, (D, sel, lab, col) in enumerate(rows):
        c = D[sel]
        u = c.get("nsr_floor_uniform_in_family")
        m = c.get("nsr_floor_modal_in_family")
        if u and m:
            ax.plot([u, m], [y[i]] * 2, color=GRID, lw=7,
                    solid_capstyle="butt", zorder=2)
            for v, nm in ((u, "uniform"), (m, "modal")):
                ax.plot([v, v], [y[i] - 0.22, y[i] + 0.22], color=MUTED,
                        lw=1.2, zorder=3)
        ax.plot([c["nsr_lo"], c["nsr_hi"]], [y[i]] * 2, color=col, lw=2.0,
                zorder=4)
        ax.plot([c["nsr"]], [y[i]], "o", ms=6.5, color=col, mec="white",
                mew=1.2, zorder=5)
        ax.annotate(f"{c['nsr']:.3f}", (c["nsr"], y[i]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[2] for r in rows], fontsize=7)
    ax.set_xlim(0.26, 0.50)
    style(ax, "native sequence recovery against the source", "",
          "Grey band = uniform-in-family to modal-in-family floor")

    # -- C: Max-ID, before and after the purge ----------------------------
    ax = axes[1][0]
    sels = ["dms6|conditioned", "rand6|conditioned", "dms6|uncond", "natural"]
    labs = ["chemical,\nconditioned", "random,\nconditioned",
            "chemical,\nunconditional", "natural\nproteins"]
    y = np.arange(len(sels))[::-1]
    for i, sel in enumerate(sels):
        a = (MO.get(sel) or {}).get("max_id_mean")
        b = (MS.get(sel) or {}).get("max_id_mean")
        if a is None or b is None:
            continue
        col = C[2] if sel == "natural" else (CHEM if "dms6" in sel else RAND)
        ax.annotate("", xy=(b, y[i]), xytext=(a, y[i]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color=col,
                                    shrinkA=0, shrinkB=0, mutation_scale=11))
        ax.plot([a], [y[i]], "o", ms=5, mfc="white", mec=col, mew=1.4,
                zorder=4)
        ax.annotate(f"{a:.0f} to {b:.0f} %", (min(a, b), y[i]),
                    textcoords="offset points", xytext=(-8, 0), ha="right",
                    va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlim(-14, 100)
    style(ax, "Max-ID to the training corpus, %", "",
          "Open circle = original corpus, arrow = after the purge")

    # -- D: Ref-TM, the structural check ----------------------------------
    ax = axes[1][1]
    if R:
        order = [("source", "the source slice\nitself (ceiling)", C[2]),
                 ("dms6|conditioned", "chemical,\nconditioned", CHEM),
                 ("rand6|conditioned", "random,\nconditioned", RAND),
                 ("dms6|uncond", "chemical,\nunconditional", CHEM),
                 ("rand6|uncond", "random,\nunconditional", RAND)]
        order = [o for o in order if o[0] in R]
        y = np.arange(len(order))[::-1]
        for i, (sel, lab, col) in enumerate(order):
            c = R[sel]
            ax.plot([c["ref_tm_lo"], c["ref_tm_hi"]], [y[i]] * 2, color=col,
                    lw=2.0, solid_capstyle="round", zorder=3)
            ax.plot([c["ref_tm"]], [y[i]], "o", ms=6.5, color=col,
                    mec="white", mew=1.2, zorder=4)
            ax.annotate(f"{c['ref_tm']:.2f}", (c["ref_tm"], y[i]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7, color=INK2)
            p = (RP.get(sel) or {}).get("permuted_tm")
            if p and np.isfinite(p):
                ax.plot([p], [y[i]], "o", ms=6, mfc="white", mec=col,
                        mew=1.4, zorder=4)
        ax.set_ylim(-0.6, len(order) - 0.35)
        if any((RP.get(o[0]) or {}).get("permuted_tm") for o in order):
            ax.annotate("open circle = aligned to a DIFFERENT source",
                        (0.015, len(order) - 0.55),
                        xycoords=("axes fraction", "data"), va="center",
                        ha="left", fontsize=6.6, color=MUTED)
        ax.axvline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate("same fold", (0.5, y[-1] - 0.4), textcoords="offset points",
                    xytext=(4, 0), fontsize=6.8, color=MUTED, va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([o[1] for o in order], fontsize=7)
        ax.set_xlim(0, 1.0)
        style(ax, "Ref-TM to the source domain's crystal structure", "",
              "CATH S35, homologs purged")
    else:
        ax.axis("off")
        ax.annotate("Ref-TM not scored yet", (0.5, 0.5), ha="center",
                    fontsize=8, color=MUTED, xycoords="axes fraction")

    fig.suptitle("The same sequences, measured by four instruments that are "
                 "not ESMFold", x=0.005, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig9_independent_metrics.png"))
    fig.savefig(os.path.join(FIG, "fig9_independent_metrics.pdf"))
    plt.close(fig)
    print("fig9 written")


if __name__ == "__main__":
    for f in (fig_conditioning, fig_pattern_budget, fig_independent_metrics):
        try:
            f()
        except Exception as e:
            print(f"{f.__name__}: skipped ({type(e).__name__}: {e})")
            raise
