"""Figures for both ideas. Each one is skipped if its input is missing.

Palette: the validated eight-slot categorical order, assigned in fixed order
and never cycled, so a series keeps its hue when another is filtered out.
One y-axis per panel, thin marks, recessive grid, legend plus direct labels.
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
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
     "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e3e2de"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160,
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight",
})


def logx(ax, ticks):
    """Log x-axis with only the ticks we name; matplotlib's minor log labels
    collide with them otherwise."""
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())


def style(ax, xlabel="", ylabel="", title=""):
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=8)


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- figure 1
def fig_contact_quality():
    rows = load("A_contact_quality.json")
    if not rows:
        return
    null = [r for r in rows if r.get("note") == "all_mask_null"]
    rows = [r for r in rows if r.get("note") != "all_mask_null"]
    fr = sorted({r["frac"] for r in rows})
    x = [f * 100 for f in fr]
    srcs = [("ext", "frozen ESM-2 650M\n(the proposed signal)"),
            ("self", "generator's own attention\n(free)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    for ax, key, lab in [(axes[0], "prec_L", "precision@L"),
                         (axes[1], "auroc", "AUROC")]:
        for k, (src, nm) in enumerate(srcs):
            y = [np.mean([r[f"{key}_{src}"] for r in rows if r["frac"] == f])
                 for f in fr]
            ax.plot(x, y, "-o", color=C[k], lw=2.0, ms=5, zorder=3,
                    label=nm.replace("\n", " "))
        if null:
            v = np.mean([null[0][f"{key}_ext"], null[0][f"{key}_self"]])
            ax.axhline(v, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
            ax.annotate("all-mask null", (100, v),
                        textcoords="offset points", xytext=(-2, 6),
                        ha="right", color=MUTED, fontsize=7)
        style(ax, "percent of the sequence masked", lab)
        ax.set_title(lab, loc="left", pad=6, fontsize=9)
    # one shared legend, placed where neither panel has data
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Contact prediction collapses once the sequence is masked",
                 x=0.02, ha="left", y=1.10, fontsize=10.5, color=INK)
    fig.text(0.02, 1.015,
             "natural 200-mers, contacts scored against ESMFold; "
             "the null is an all-mask input, which holds no sequence "
             "information",
             ha="left", fontsize=7.5, color=INK2)
    fig.subplots_adjust(wspace=0.26)
    fig.savefig(os.path.join(FIG, "fig1_contact_quality.png"))
    fig.savefig(os.path.join(FIG, "fig1_contact_quality.pdf"))
    plt.close(fig)
    print("fig1 written")


# ---------------------------------------------------------------- figure 2
SEL_ORDER = ["confidence", "contact", "contact_shuffled", "distance",
             "anti_contact", "random", "punt", "dplm_reparam"]
SEL_LABEL = {"confidence": "confidence (baseline)",
             "contact": "contact-gated (proposed)",
             "contact_shuffled": "contact shuffled (control)",
             "distance": "sequence distance (control)",
             "anti_contact": "contact-seeking (opposite polarity)",
             "random": "random order",
             "punt": "conditional-independence test",
             "contact_oracle": "true contact map (oracle)",
             "dplm_reparam": "DPLM published decoder"}


def _cells(files):
    out = []
    for f in files:
        d = load(f)
        if d:
            out.extend(d["cells"])
    return out


def fig_plddt_vs_nfe():
    cells = _cells(["Am_q99_scored.json", "A_fidelity_scored.json"])
    if not cells:
        return
    cells = [c for c in cells if c.get("note") != "tau_sweep"
             and c.get("plddt_mean") is not None]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    nat = [c for c in cells if c.get("selector") == "natural"]
    if nat:
        ax.axhline(nat[0]["plddt_mean"], color=MUTED, lw=1.2,
                   ls=(0, (4, 3)), zorder=2)
        ax.annotate("natural proteins", (6, nat[0]["plddt_mean"]),
                    textcoords="offset points", xytext=(0, 4),
                    color=MUTED, fontsize=7)
    k = 0
    for sel in SEL_ORDER:
        g = sorted([c for c in cells if c.get("selector") == sel],
                   key=lambda z: z["T"])
        if not g:
            continue
        x = [c["T"] for c in g]
        y = [c["plddt_mean"] for c in g]
        lo = [c["plddt_ci"][0] for c in g]
        hi = [c["plddt_ci"][1] for c in g]
        ax.fill_between(x, lo, hi, color=C[k % 8], alpha=0.12, lw=0, zorder=2)
        ax.plot(x, y, "-o", color=C[k % 8], lw=2.0, ms=4.5,
                label=SEL_LABEL.get(sel, sel), zorder=3)
        k += 1
    logx(ax, [6, 12, 25, 50, 100, 200, 500])
    style(ax, "generator forward passes (log scale)", "ESMFold pLDDT",
          "Quality against compute, length 200")
    ax.legend(loc="lower right", ncol=1)
    fig.savefig(os.path.join(FIG, "fig2_plddt_vs_nfe.png"))
    fig.savefig(os.path.join(FIG, "fig2_plddt_vs_nfe.pdf"))
    plt.close(fig)
    print("fig2 written")


# ---------------------------------------------------------------- figure 3
def fig_contrasts():
    from analysis.report import boot_diff
    cells = _cells(["Am_q99_scored.json"])
    if not cells:
        return
    cells = [c for c in cells if c.get("note") != "tau_sweep"
             and c.get("plddt_list")]
    Ts = sorted({c["T"] for c in cells if c.get("T")})
    sels = [s for s in SEL_ORDER if s not in ("confidence", "dplm_reparam")
            and any(c.get("selector") == s for c in cells)]
    fig, ax = plt.subplots(figsize=(5.4, 0.42 * len(Ts) * len(sels) + 1.0))
    ypos, labels = [], []
    y = 0
    for T in Ts:
        base = [c for c in cells if c["T"] == T
                and c.get("selector") == "confidence"]
        if not base:
            continue
        bl = base[0]["plddt_list"]
        for k, s in enumerate(sels):
            g = [c for c in cells if c["T"] == T and c.get("selector") == s]
            if not g:
                continue
            r = boot_diff(g[0]["plddt_list"], bl)
            col = C[(SEL_ORDER.index(s)) % 8]
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2.0, zorder=3,
                    solid_capstyle="round")
            ax.plot([r["diff"]], [y], "o", color=col, ms=5.5, zorder=4,
                    mec="white", mew=1.0)
            ypos.append(y)
            labels.append(f"T={T}  {SEL_LABEL.get(s, s).split(' (')[0]}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color=INK2, lw=1.0, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7)
    style(ax, "pLDDT difference from the confidence baseline", "",
          "Every selector against the baseline, at matched forward passes")
    fig.savefig(os.path.join(FIG, "fig3_contrasts.png"))
    fig.savefig(os.path.join(FIG, "fig3_contrasts.pdf"))
    plt.close(fig)
    print("fig3 written")


# ---------------------------------------------------------------- figure 4
def fig_ideaB():
    cells = _cells(["B_main_scored.json", "B_qd_flat_scored.json",
                    "B_qd_dms6_scored.json"])
    if not cells:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    groups = {}
    for c in cells:
        if c.get("plddt_mean") is None:
            continue
        tag = c.get("tag") or c.get("selector")
        groups.setdefault(tag, []).append(c)
    k = 0
    for tag in sorted(groups):
        g = [c for c in groups[tag] if np.isfinite(c.get("diversity", np.nan))]
        if not g:
            continue
        g = sorted(g, key=lambda z: z["diversity"])
        ax.plot([c["diversity"] for c in g], [c["plddt_mean"] for c in g],
                "-o", color=C[k % 8], lw=2.0, ms=4.5, label=tag, zorder=3)
        k += 1
    style(ax, "diversity (1 - mean pairwise identity)", "ESMFold pLDDT",
          "Quality against diversity")
    ax.legend(ncol=2, loc="lower left")

    ax = axes[1]
    k = 0
    for tag in sorted(groups):
        g = [c for c in groups[tag]
             if np.isfinite(c.get("amphipathic_index", np.nan))]
        if not g:
            continue
        v = np.mean([c["amphipathic_index"] for c in g])
        ax.barh(k, v, color=C[k % 8], height=0.62, zorder=3)
        ax.annotate(f"{v:.3f}", (v, k), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=7, color=INK2)
        k += 1
    ax.set_yticks(range(k))
    ax.set_yticklabels(sorted(groups), fontsize=7)
    style(ax, "helical amphipathic index", "",
          "Hydrophobic periodicity near 3.6 residues per turn")
    fig.savefig(os.path.join(FIG, "fig4_ideaB.png"))
    fig.savefig(os.path.join(FIG, "fig4_ideaB.pdf"))
    plt.close(fig)
    print("fig4 written")


# ---------------------------------------------------------------- figure 5
def fig_oracle():
    """The decisive control: a true contact map against its own shuffle."""
    cells = _cells(["A_oracle_scored.json"])
    cells = [c for c in cells if c.get("plddt_mean") is not None
             and c.get("T")]
    if not cells:
        return
    series = [("contact_oracle", "true contact map", C[0]),
              ("contact_oracle_shuffled", "same map, shuffled", C[1]),
              ("anti_contact_oracle", "opposite polarity", C[2])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axes[0]
    for name, lab, col in series:
        g = sorted([c for c in cells if c["selector"] == name],
                   key=lambda z: z["T"])
        if not g:
            continue
        x = [c["T"] for c in g]
        ax.fill_between(x, [c["plddt_ci"][0] for c in g],
                        [c["plddt_ci"][1] for c in g], color=col,
                        alpha=0.12, lw=0, zorder=2)
        ax.plot(x, [c["plddt_mean"] for c in g], "-o", color=col, lw=2.0,
                ms=5, label=lab, zorder=3)
    logx(ax, [6, 12, 25, 50])
    style(ax, "generator forward passes", "ESMFold pLDDT",
          "Even a correct map does not help")
    ax.legend(loc="upper left")

    ax = axes[1]
    tr = {c["T"]: c["plddt_mean"] for c in cells
          if c["selector"] == "contact_oracle"}
    sh = {c["T"]: c["plddt_mean"] for c in cells
          if c["selector"] == "contact_oracle_shuffled"}
    Ts = sorted(set(tr) & set(sh))
    d = [tr[t] - sh[t] for t in Ts]
    ax.bar(range(len(Ts)), d, color=[C[0] if v > 0 else C[1] for v in d],
           width=0.6, zorder=3)
    ax.axhline(0, color=INK2, lw=1.0, zorder=4)
    for i, v in enumerate(d):
        ax.annotate(f"{v:+.1f}", (i, v), textcoords="offset points",
                    xytext=(0, 4 if v > 0 else -12), ha="center",
                    fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(Ts)))
    ax.set_xticklabels([str(t) for t in Ts])
    style(ax, "generator forward passes",
          "pLDDT, true map minus shuffled",
          f"Mean effect {np.mean(d):+.2f}")
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(os.path.join(FIG, "fig5_oracle.png"))
    fig.savefig(os.path.join(FIG, "fig5_oracle.pdf"))
    plt.close(fig)
    print("fig5 written")


# ---------------------------------------------------------------- figure 6
def fig_ideaB_main():
    cells = _cells(["B_main_scored.json"])
    cells = [c for c in cells if c.get("plddt_mean") is not None]
    if not cells:
        return
    order = ["flat", "dms6", "rand6", "natural"]
    lab = {"flat": "flat, 20 letters", "dms6": "two-level, chemical",
           "rand6": "two-level, RANDOM groups", "natural": "natural proteins"}
    rows = []
    for k in order:
        g = [c for c in cells
             if (c.get("tag") or c.get("selector")) == k]
        if g:
            rows.append((k, g[0]))
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(rows))[::-1]
    for i, (k, c) in enumerate(rows):
        col = C[3] if k == "natural" else (C[0] if k == "flat" else C[1])
        ci = c.get("plddt_ci", [c["plddt_mean"]] * 2)
        ax.plot([ci[0], ci[1]], [y[i], y[i]], color=col, lw=2.0,
                solid_capstyle="round", zorder=3)
        ax.plot([c["plddt_mean"]], [y[i]], "o", color=col, ms=6, zorder=4,
                mec="white", mew=1.0)
        ax.annotate(f"{c['plddt_mean']:.1f}", (c["plddt_mean"], y[i]),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=7.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([lab[k] for k, _ in rows],
                                         fontsize=8)
    style(ax, "ESMFold pLDDT", "",
          "Matched size, data, budget and forward passes")
    fig.savefig(os.path.join(FIG, "fig6_ideaB.png"))
    fig.savefig(os.path.join(FIG, "fig6_ideaB.pdf"))
    plt.close(fig)
    print("fig6 written")


if __name__ == "__main__":
    for f in (fig_contact_quality, fig_plddt_vs_nfe, fig_contrasts,
              fig_oracle, fig_ideaB_main, fig_ideaB):
        try:
            f()
        except Exception as e:
            print(f"{f.__name__}: skipped ({type(e).__name__}: {e})")
