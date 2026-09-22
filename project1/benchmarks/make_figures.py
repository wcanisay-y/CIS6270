# draw result figures for the qm9 diffusion benchmark (edm vs gcdm).
# reads results/unified_all.json, results/bootstrap.json (95% bootstrap intervals) and
# the native numbers from the run logs.

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BENCH = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BENCH, "results")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

# palette (validated): edm = slot 1 blue, gcdm = slot 2 orange, reference = gray
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"EDM": "#2a78d6", "GCDM": "#eb6834"}
REF = "#898781"
ERR = dict(ecolor=INK2, elinewidth=1, capsize=2.5, capthick=1)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def read_native(log, patterns):
    text = open(os.path.join(RES, log), errors="ignore").read()
    out = {}
    for key, pat in patterns.items():
        m = re.findall(pat, text)
        out[key] = float(m[-1]) if m else None
    return out


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ data
unified = json.load(open(os.path.join(RES, "unified_all.json")))
boot_path = os.path.join(RES, "bootstrap.json")
boot = json.load(open(boot_path)) if os.path.exists(boot_path) else None
edm_native = read_native("edm_eval.log", {
    "novelty": r"Novelty over \d+ unique valid molecules: ([\d.]+)%",
    "test_nll": r"Final test nll (-?[\d.]+)",
})
gcdm_native = read_native("gcdm_eval.log", {
    "novelty": r"Novelty over \d+ unique valid molecules: ([\d.]+)%",
    "test_nll": r"Test negative log-likelihood \(NLL\): (-?[\d.]+)",
})
native_novelty = {"EDM": edm_native["novelty"] / 100, "GCDM": gcdm_native["novelty"] / 100}
test_nll = {"EDM": edm_native["test_nll"], "GCDM": gcdm_native["test_nll"]}
n_ref = len(json.load(open(os.path.join(RES, "qm9_parquet_smiles.json"))))

METRICS = [
    ("atom_stability", "Atom\nstability"),
    ("molecule_stability", "Molecule\nstability"),
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness"),
    ("valid_and_unique", "Valid and\nunique"),
    ("novelty", "Novelty"),
]
MODELS = ["EDM", "GCDM"]


def interval(name, key):
    # (value, lo, hi) from the bootstrap file; falls back to the point estimate without bars
    if boot and name in boot and isinstance(boot[name].get(key), dict):
        b = boot[name][key]
        return b["value"], b["lo"], b["hi"]
    v = native_novelty[name] if key == "novelty_native" else unified[name][key]
    return v, v, v


def ci_note():
    if boot:
        return f"Error bars: 95% bootstrap intervals over the 10,000 samples ({boot['n_boot']:,} resamples)"
    return ""


# ------------------------------------------------------------------ figure 1: sample quality
def draw_quality(ax):
    width, gap = 0.30, 0.06
    xs = range(len(METRICS))
    for j, model in enumerate(MODELS):
        offs = (j - 0.5) * (width + gap)
        vals, los, his = zip(*[interval(model, k) for k, _ in METRICS])
        bars = ax.bar([x + offs for x in xs], vals, width=width, color=COLORS[model], zorder=3, label=model)
        if boot:
            ax.errorbar([x + offs for x in xs], vals,
                        yerr=[[v - lo for v, lo in zip(vals, los)], [hi - v for v, hi in zip(vals, his)]],
                        fmt="none", zorder=5, **ERR)
        for b, v, hi in zip(bars, vals, his):
            ax.text(b.get_x() + b.get_width() / 2, hi + 0.014, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7, color=INK2, zorder=6,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
    # qm9 data as a reference tick across each metric group (none for novelty)
    span = width + gap / 2
    for x, (k, _) in zip(xs, METRICS):
        if unified["QM9 data"][k] is None:
            continue
        v, lo, hi = interval("QM9 data", k)
        if boot:
            ax.plot([x, x], [lo, hi], color=REF, linewidth=1, zorder=4)
        ax.plot([x - span, x + span], [v, v], color=REF, linewidth=2, solid_capstyle="round", zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab in METRICS])
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Fraction of 10,000 samples")
    ax.set_title("Sample quality on QM9, shared evaluator")
    handles = [Patch(color=COLORS[m], label=m) for m in MODELS]
    handles.append(Line2D([0], [0], color=REF, linewidth=2, label="QM9 data (reference)"))
    ax.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.30), handlelength=1.4)
    ax.text(0.0, -0.36, ci_note(), transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED)
    style(ax)


# ------------------------------------------------------------------ figure 2: novelty vs reference set
def draw_novelty(ax):
    ys = {"EDM": 1, "GCDM": 0}
    for model in MODELS:
        a, a_lo, a_hi = interval(model, "novelty_native")
        b, b_lo, b_hi = interval(model, "novelty")
        y = ys[model]
        ax.plot([a, b], [y, y], color=COLORS[model], linewidth=2, zorder=2, solid_capstyle="round")
        if boot:
            for v, lo, hi in ((a, a_lo, a_hi), (b, b_lo, b_hi)):
                ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="none", zorder=2.5,
                            ecolor=COLORS[model], elinewidth=1, capsize=3, capthick=1)
        ax.scatter([a], [y], s=70, facecolor=SURFACE, edgecolor=COLORS[model], linewidth=2, zorder=3)
        ax.scatter([b], [y], s=70, facecolor=COLORS[model], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(a + 0.012, y + 0.18, f"{a:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.text(b - 0.012, y + 0.18, f"{b:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(MODELS)
    ax.set_ylim(-1.45, 1.7)
    ax.set_xlim(0.40, 0.75)
    ax.set_xlabel("Novelty (fraction of unique valid samples not in reference)")
    ax.set_title("Novelty depends on the reference set")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK2,
               markeredgewidth=1.6, markersize=8, label="Own training split (native evaluator)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=INK2, markeredgecolor=INK2,
               markersize=8, label=f"Full QM9 parquet, {n_ref:,} SMILES (shared)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1,
              frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1, borderpad=0.4)
    if boot:
        ax.text(1.0, -0.36, "Error bars: 95% bootstrap intervals", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)


# ------------------------------------------------------------------ figure 3: test nll
def draw_nll(ax):
    ys = {"EDM": 1, "GCDM": 0}
    for model in MODELS:
        v, y = test_nll[model], ys[model]
        passes = boot["nll"][model]["passes"] if boot else [v]
        if len(passes) > 1:
            ax.plot([min(passes), max(passes)], [y, y], color=COLORS[model], linewidth=1, zorder=2)
            ax.scatter(passes, [y] * len(passes), s=28, facecolor=SURFACE, edgecolor=COLORS[model],
                       linewidth=1.2, zorder=3)
            label = f"{v:.1f}  (mean of {len(passes)} passes, {min(passes):.1f} to {max(passes):.1f})"
        else:
            label = f"{v:.1f}  (single pass)"
        ax.scatter([v], [y], s=90, color=COLORS[model], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(v, y + 0.22, label, ha="left" if model == "GCDM" else "right", va="bottom",
                fontsize=8, color=INK2)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(MODELS)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(-190, -95)
    ax.set_xlabel("Test negative log-likelihood (lower is better)")
    ax.set_title("Held-out likelihood, each model's own evaluator")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.annotate("", xy=(-186, -0.55), xytext=(-150, -0.55),
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
    ax.text(-168, -0.5, "better", ha="center", va="bottom", fontsize=8, color=MUTED)
    if boot:
        ax.text(1.0, -0.36, "Open markers: individual passes over the test split", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=MUTED)


fig, ax = plt.subplots(figsize=(8.4, 3.8))
draw_quality(ax)
save(fig, "fig1_sample_quality")

fig, ax = plt.subplots(figsize=(5.4, 2.9))
draw_novelty(ax)
save(fig, "fig2_novelty_reference")

fig, ax = plt.subplots(figsize=(5.4, 2.3))
draw_nll(ax)
save(fig, "fig3_test_nll")

# combined overview for slides
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.7, wspace=0.3)
draw_quality(fig.add_subplot(gs[0, :]))
draw_novelty(fig.add_subplot(gs[1, 0]))
draw_nll(fig.add_subplot(gs[1, 1]))
fig.suptitle("QM9 unconditional generation: EDM vs GCDM, 10,000 samples each", fontsize=12,
             fontweight="semibold", color=INK, y=0.98)
save(fig, "fig_overview")

# table twin of every plotted value
keys = [k for k, _ in METRICS] + ["novelty_native"]
with open(os.path.join(OUT, "figure_data.csv"), "w") as f:
    f.write("model,metric,value,ci_lo,ci_hi\n")
    for m in MODELS + ["QM9 data"]:
        for k in keys:
            if m == "QM9 data" and k.startswith("novelty"):
                continue
            v, lo, hi = interval(m, k)
            f.write(f"{m},{k},{v:.4f},{lo:.4f},{hi:.4f}\n")
    for m in MODELS:
        f.write(f"{m},test_nll,{test_nll[m]:.2f},,\n")
print("bootstrap file:", "found" if boot else "missing")
print("wrote:", sorted(os.listdir(OUT)))
