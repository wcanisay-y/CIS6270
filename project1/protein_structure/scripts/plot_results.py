#!/usr/bin/env python3
"""ARCH 5. Result figures for the Ca diffusion arm.

Reads the run's history.json (training curves), metrics.json (the aggregate table),
checkpoint.pt (the run's own hyperparameters) and samples_*.npz, recomputes every
per-structure quantity with the scorer from evaluate_protein.py so the figures cannot drift
from the reported table, and writes five standalone figures, a one-page overview, and the
numbers behind them.

Every panel is a function of (ax, ctx), so the overview and the standalone figures draw the
same marks from the same numbers.

    python plot_results.py                          # the 10,000-sample run
    python plot_results.py --run <dir> --out <dir>
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_protein import BOND_LO, BOND_HI, per_structure, reference

# palette (validated): reference = slot 1 blue, generated = slot 2 orange
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PLANE, RING = "#f9f9f7", (11 / 255, 11 / 255, 11 / 255, 0.10)
REF_C, GEN_C = "#2a78d6", "#eb6834"
CHANCE = "#898781"
# one-hue ramps for the chain-direction encoding, blue for reference and orange for generated
REF_RAMP = LinearSegmentedColormap.from_list("ref", ["#86b6ef", "#2a78d6", "#0d366b"])
GEN_RAMP = LinearSegmentedColormap.from_list("gen", ["#f6b295", "#eb6834", "#7a2f11"])

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

BINS = [(16, 40), (41, 60), (61, 80), (81, 100), (101, 128)]
REF_LABEL, GEN_LABEL = "AlphaFold test", "generated"


# ------------------------------------------------------------------ small helpers
def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def column(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return v[np.isfinite(v)]


def binned(rows):
    length = np.array([r["length"] for r in rows])
    table = []
    for lo, hi in BINS:
        sub = [r for r, keep in zip(rows, (length >= lo) & (length <= hi)) if keep]
        table.append(dict(
            bin=f"{lo}-{hi}", n=len(sub),
            validity=float(np.mean([r["valid"] for r in sub])) if sub else np.nan,
            clash_free=float(np.mean([r["clashes"] == 0 for r in sub])) if sub else np.nan,
            clashes=float(np.mean([r["clashes"] for r in sub])) if sub else np.nan,
            radius=float(np.mean([r["radius"] for r in sub])) if sub else np.nan))
    return table


def hist_pair(ax, values, colours, **kw):
    for v, c in zip(values, colours):
        ax.hist(v, color=c, alpha=0.45, histtype="stepfilled", zorder=3, **kw)
        ax.hist(v, color=c, linewidth=1.6, histtype="step", zorder=4, **kw)


def legend_handles():
    return [Line2D([], [], color=c, linewidth=2.4, label=l)
            for l, c in ((REF_LABEL, REF_C), (GEN_LABEL, GEN_C))]


# ------------------------------------------------------------------------- panels
def panel_training(ax, ctx, compact=False):
    """Full 200-epoch curve on a log scale."""
    e, h = ctx["epoch"], ctx["history_curves"]
    for label, y, c, lw in h:
        ax.plot(e, y, color=c, linewidth=lw, label=label, zorder=3, alpha=0.95)
    ax.set_yscale("log")
    ticks = [0.16, 0.2, 0.3, 0.5, 0.8, 1.2, 1.7]
    ax.set_yticks(ticks, minor=False)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylim(0.14, 1.9)
    ax.set_xlim(0, 205)
    ax.set_xlabel("epoch")
    ax.set_ylabel("denoising loss (log scale)")
    ax.set_title("Training converged over 200 epochs" if not compact else "Training loss")
    ax.legend(loc="upper right")
    style(ax)
    if compact:
        best_e, best_v = ctx["best_epoch"], ctx["dev_ema"].min()
        ax.annotate(f"best dev (EMA) {best_v:.4f} at epoch {best_e}\ntrain {ctx['train'][-1]:.4f} vs"
                    f" dev {ctx['dev_raw'][-1]:.4f}\nat epoch 200: no overfitting",
                    xy=(0.92, 0.56), xycoords="axes fraction", ha="right", va="top",
                    fontsize=8, color=INK2, linespacing=1.5)


def panel_training_zoom(ax, ctx, compact=False):
    """Last 100 epochs, linear, with the three end points labelled."""
    e = ctx["epoch"]
    keep = e >= 100
    for label, y, c, lw in ctx["history_curves"]:
        ax.plot(e[keep], y[keep], color=c, linewidth=lw, zorder=3, alpha=0.95)
        ax.annotate(f"{label} {y[-1]:.4f}", xy=(e[-1], y[-1]), xytext=(4, 0),
                    textcoords="offset points", color=c, fontsize=8.5,
                    va="center", fontweight="semibold")
    best_e, best_v = ctx["best_epoch"], ctx["dev_ema"].min()
    ax.scatter([best_e], [best_v], s=34, facecolor=SURFACE, edgecolor=GEN_C,
               linewidth=1.6, zorder=5)
    ax.annotate(f"best dev (EMA) {best_v:.4f}, epoch {best_e}", xy=(best_e, best_v),
                xytext=(-12, 6), textcoords="offset points", color=INK2, fontsize=8,
                ha="right", va="bottom")
    ax.set_xlim(100, 232)
    ax.set_xlabel("epoch")
    ax.set_ylabel("denoising loss")
    ax.set_title("Train and dev sit on top of each other: no overfitting")
    style(ax)


QUALITY_METRICS = [("validity", "validity", None),
                   ("bonds in range", "bond_ok", None),
                   ("clash-free", "clash_free", None),
                   ("right-handed windows", "right_handed", 0.5),
                   ("helical fraction", "helix_fraction", None)]


def panel_quality(ax, ctx, compact=False):
    y = np.arange(len(QUALITY_METRICS))[::-1]
    h = 0.34
    for offset, stats, colour, label in ((h / 2, ctx["ref_stats"], REF_C, f"{REF_LABEL} (n={ctx['ref_stats']['n']:,})"),
                                         (-h / 2, ctx["gen_stats"], GEN_C, f"{GEN_LABEL} (n={ctx['gen_stats']['n']:,})")):
        values = [stats[key] for _, key, _ in QUALITY_METRICS]
        ax.barh(y + offset, values, height=h - 0.02, color=colour, zorder=3,
                label=label, edgecolor=SURFACE, linewidth=1.2)
        for yy, v in zip(y + offset, values):
            ax.annotate(f"{v:.3f}", xy=(v, yy), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8 if compact else 8.5, color=INK2)
    for i, (_, _, chance) in enumerate(QUALITY_METRICS):
        if chance is None:
            continue
        ax.plot([chance, chance], [y[i] - h, y[i] + h], color=CHANCE, linewidth=1.4,
                linestyle=(0, (3, 2)), zorder=4)
        ax.annotate("chance", xy=(chance, y[i] + h), xytext=(0, 3), textcoords="offset points",
                    color=CHANCE, fontsize=8, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _, _ in QUALITY_METRICS], color=INK2)
    ax.set_xlim(0, 1.16)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("fraction of structures (or of windows within one)")
    ax.set_title("Local geometry is solved; clashes cap validity")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02))
    style(ax, axis="x")


def panel_bond(ax, ctx, compact=False):
    hist_pair(ax, (ctx["ref_steps"], ctx["gen_steps"]), (REF_C, GEN_C),
              bins=np.linspace(2.6, 5.0, 90), density=True)
    for x in (BOND_LO, BOND_HI):
        ax.axvline(x, color=CHANCE, linewidth=1.2, linestyle=(0, (3, 2)), zorder=2)
    ax.set_xlim(2.85, 4.75)
    ax.annotate("accepted\nwindow", xy=(3.12, ax.get_ylim()[1] * 0.94), color=CHANCE,
                fontsize=8, ha="left", va="top")
    ax.set_xlabel("consecutive Ca-Ca distance (A)")
    ax.set_ylabel("density")
    ax.set_title("Bond length")
    style(ax)


def panel_clash_hist(ax, ctx, compact=False):
    edges = np.arange(-0.5, 8.6, 1.0)
    for rows, c in ((ctx["ref_rows"], REF_C), (ctx["gen_rows"], GEN_C)):
        v = np.clip(column(rows, "clashes"), 0, 8)
        w = np.full(len(v), 100.0 / len(v))
        ax.hist(v, bins=edges, weights=w, color=c, alpha=0.45, histtype="stepfilled", zorder=3)
        ax.hist(v, bins=edges, weights=w, color=c, linewidth=1.6, histtype="step", zorder=4)
    ax.set_xlabel("clashes per structure (8 = 8 or more)")
    ax.set_ylabel("% of structures")
    ax.set_title("Steric clashes: the failure mode")
    style(ax)


def panel_radius_hist(ax, ctx, compact=False):
    hist_pair(ax, (column(ctx["ref_rows"], "radius"), column(ctx["gen_rows"], "radius")),
              (REF_C, GEN_C), bins=np.linspace(6, 32, 70), density=True)
    ax.set_xlabel("radius of gyration (A)")
    ax.set_ylabel("density")
    ax.set_title("Samples are over-collapsed")
    style(ax)


def panel_helix_hist(ax, ctx, compact=False):
    hist_pair(ax, (column(ctx["ref_rows"], "helix"), column(ctx["gen_rows"], "helix")),
              (REF_C, GEN_C), bins=np.linspace(0, 1, 41), density=True)
    ax.set_xlabel("helical fraction of a structure")
    ax.set_ylabel("density")
    ax.set_title("Helical content is the weakest axis")
    style(ax)


def panel_len_validity(ax, ctx, compact=False):
    x = np.arange(len(BINS))
    ref, gen = ctx["ref_bins"], ctx["gen_bins"]
    ax.plot(x, [r["validity"] for r in gen], color=GEN_C, linewidth=2, marker="o",
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4, label="validity")
    ax.plot(x, [r["clash_free"] for r in gen], color=GEN_C, linewidth=2, marker="s",
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle=(0, (4, 2)),
            zorder=4, alpha=0.75, label="clash-free")
    ax.plot(x, [r["validity"] for r in ref], color=REF_C, linewidth=2, marker="o",
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
            label="validity, AlphaFold")
    for xx, r in zip(x, gen):
        ax.annotate(f"{r['validity']:.2f}", xy=(xx, r["validity"]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8, color=GEN_C)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([r["bin"] for r in gen])
    ax.set_xlabel("chain length (residues)")
    ax.set_ylabel("fraction of structures")
    ax.set_title("Validity falls 2.5x with length")
    ax.legend(loc="lower left", fontsize=8)
    style(ax)


def panel_len_clashes(ax, ctx, compact=False):
    x = np.arange(len(BINS))
    gen = ctx["gen_bins"]
    ax.bar(x, [r["clashes"] for r in gen], width=0.6, color=GEN_C, zorder=3,
           edgecolor=SURFACE, linewidth=1.2)
    for xx, r in zip(x, gen):
        ax.annotate(f"{r['clashes']:.2f}", xy=(xx, r["clashes"]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([r["bin"] for r in gen])
    ax.set_xlabel("chain length (residues)")
    ax.set_ylabel("mean clashes per structure")
    ax.set_title("Clashes accumulate (AlphaFold: 0 at every length)"
                 if not compact else "Clashes accumulate (AlphaFold: 0)")
    style(ax)


def panel_len_radius(ax, ctx, compact=False):
    x = np.arange(len(BINS))
    ref, gen = ctx["ref_bins"], ctx["gen_bins"]
    for label, table, c in ((REF_LABEL, ref, REF_C), (GEN_LABEL, gen, GEN_C)):
        ax.plot(x, [r["radius"] for r in table], color=c, linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3, label=label)
    ax.fill_between(x, [r["radius"] for r in gen], [r["radius"] for r in ref],
                    color=GEN_C, alpha=0.12, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([r["bin"] for r in gen])
    ax.set_xlabel("chain length (residues)")
    ax.set_ylabel("radius of gyration (A)")
    ax.set_title("The collapse gap widens with length")
    ax.legend(loc="lower right", fontsize=8)
    style(ax)


def draw_backbone(ax, coords, length, ramp, label_size=8.5):
    c = coords[:length] - coords[:length].mean(0)
    for k in range(length - 1):
        ax.plot(c[k:k + 2, 0], c[k:k + 2, 1], c[k:k + 2, 2],
                color=ramp(k / max(length - 2, 1)), linewidth=2.1, solid_capstyle="round")
    lim = np.abs(c).max() * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.set_axis_off()
    ax.set_title(f"{length} residues", fontsize=label_size, color=INK2,
                 fontweight="normal", pad=-12)


def stat_tile(ax, value, label, sub, colour):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=PLANE, edgecolor=RING, linewidth=1.0, zorder=1))
    ax.text(0.5, 0.66, value, transform=ax.transAxes, ha="center", va="center",
            fontsize=25, color=colour, fontweight="semibold", zorder=2)
    ax.text(0.5, 0.38, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5, color=INK, zorder=2)
    ax.text(0.5, 0.21, sub, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color=MUTED, zorder=2)


# ------------------------------------------------------------- standalone figures
def fig_training(ctx, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    panel_training(axes[0], ctx)
    panel_training_zoom(axes[1], ctx)
    fig.tight_layout()
    save(fig, out, "fig1_training")


def fig_quality(ctx, out):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    panel_quality(ax, ctx)
    fig.tight_layout()
    save(fig, out, "fig2_sample_quality")


def fig_distributions(ctx, out):
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.3))
    for ax, panel in zip(axes, (panel_bond, panel_clash_hist, panel_radius_hist, panel_helix_hist)):
        panel(ax, ctx)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out, "fig3_geometry_distributions")


def fig_length(ctx, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    for ax, panel in zip(axes, (panel_len_validity, panel_len_clashes, panel_len_radius)):
        panel(ax, ctx)
    fig.tight_layout()
    save(fig, out, "fig4_length_stratified")


def fig_backbones(ctx, out):
    n = max(len(ctx["ref_picks"]), len(ctx["gen_picks"]))
    fig = plt.figure(figsize=(2.1 * n, 4.4))
    for row, (picks, coords, lengths, ramp, title) in enumerate((
            (ctx["ref_picks"], ctx["ref_coords"], ctx["ref_lengths"], REF_RAMP, REF_LABEL),
            (ctx["gen_picks"], ctx["gen_coords"], ctx["gen_lengths"], GEN_RAMP, GEN_LABEL))):
        for col, i in enumerate(picks):
            ax = fig.add_subplot(2, n, row * n + col + 1, projection="3d")
            draw_backbone(ax, coords[i], int(lengths[i]), ramp)
            if col == 0:
                ax.text2D(-0.02, 0.5, title, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=10.5, color=INK, fontweight="semibold")
    fig.suptitle(f"Ca traces, {ctx['gallery_note']} (shade runs N to C terminus)",
                 fontsize=10.5, color=INK, fontweight="semibold", y=0.98)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.01, wspace=0.0, hspace=0.06)
    save(fig, out, "fig5_backbones")


# --------------------------------------------------------------- one-page overview
def fig_overview(ctx, out):
    fig = plt.figure(figsize=(11.5, 15.0))
    gs = fig.add_gridspec(5, 12, height_ratios=[1.05, 2.4, 2.1, 2.1, 3.1],
                          hspace=0.52, wspace=1.5,
                          left=0.065, right=0.975, top=0.925, bottom=0.035)

    m, ref, gen = ctx["meta"], ctx["ref_stats"], ctx["gen_stats"]
    fig.text(0.065, 0.972, "Unconditional Ca-backbone diffusion on SwissProt-Pfam",
             fontsize=17, color=INK, fontweight="semibold", ha="left", va="center")
    fig.text(0.065, 0.950,
             f"{m['params'] / 1e6:.2f}M-parameter sparse SE(3) EGNN, {m['layers']} layers, hidden "
             f"{m['hidden']}, k={m['k']} neighbours  |  {m['n_train']:,} train / {m['n_dev']:,} dev / "
             f"{m['n_test']:,} test chains of 16-128 residues, single Pfam family, AlphaFold DB "
             f"structures\n{m['epochs']} epochs at batch {m['batch']} in {m['train_hours']:.2f} h on one "
             f"A100-40GB  |  {gen['n']:,} backbones sampled with {m['timesteps']} denoising steps  |  "
             f"every metric computed identically for both rows, so the AlphaFold row is the ceiling",
             fontsize=8.8, color=INK2, ha="left", va="top", linespacing=1.65)

    tiles = [(f"{gen['validity']:.3f}", "validity", f"AlphaFold {ref['validity']:.3f}", GEN_C),
             (f"{gen['clash_free']:.3f}", "clash-free", f"AlphaFold {ref['clash_free']:.3f}", GEN_C),
             (f"{gen['bond_ok']:.3f}", "bonds in range", f"AlphaFold {ref['bond_ok']:.3f}", GEN_C),
             (f"{gen['right_handed']:.3f}", "right-handed windows",
              f"chance 0.500, AlphaFold {ref['right_handed']:.3f}", GEN_C)]
    for i, (value, label, sub, colour) in enumerate(tiles):
        stat_tile(fig.add_subplot(gs[0, i * 3:(i + 1) * 3]), value, label, sub, colour)

    panel_training(fig.add_subplot(gs[1, 0:6]), ctx, compact=True)
    panel_quality(fig.add_subplot(gs[1, 6:12]), ctx, compact=True)

    panel_len_validity(fig.add_subplot(gs[2, 0:4]), ctx, compact=True)
    panel_len_clashes(fig.add_subplot(gs[2, 4:8]), ctx, compact=True)
    panel_len_radius(fig.add_subplot(gs[2, 8:12]), ctx, compact=True)

    for i, panel in enumerate((panel_bond, panel_clash_hist, panel_radius_hist, panel_helix_hist)):
        panel(fig.add_subplot(gs[3, i * 3:(i + 1) * 3]), ctx, compact=True)

    n = max(len(ctx["ref_picks"]), len(ctx["gen_picks"]))
    inner = gs[4, :].subgridspec(2, n, hspace=0.10, wspace=0.0)
    for row, (picks, coords, lengths, ramp, title) in enumerate((
            (ctx["ref_picks"], ctx["ref_coords"], ctx["ref_lengths"], REF_RAMP, REF_LABEL),
            (ctx["gen_picks"], ctx["gen_coords"], ctx["gen_lengths"], GEN_RAMP, GEN_LABEL))):
        for col, i in enumerate(picks):
            ax = fig.add_subplot(inner[row, col], projection="3d")
            draw_backbone(ax, coords[i], int(lengths[i]), ramp, label_size=8)
            if col == 0:
                ax.text2D(-0.06, 0.5, title, transform=ax.transAxes, rotation=90, va="center",
                          ha="center", fontsize=10, color=INK, fontweight="semibold")

    fig.text(0.5, 0.012,
             f"Ca traces above: {ctx['gallery_note']}, shade running N to C terminus. "
             "Bonds are solved and handedness is learned; clashes and missing helices are what is left.",
             fontsize=8.5, color=MUTED, ha="center", va="bottom")
    save(fig, out, "fig0_overview")


# ------------------------------------------------------------------------- driver
def build_context(run, samples):
    history = json.load(open(os.path.join(run, "history.json")))
    table = json.load(open(os.path.join(run, "metrics.json")))
    ref_stats = next(r for r in table if r["name"].startswith("AlphaFold"))
    gen_stats = next(r for r in table if r["name"] == "generated")

    d = np.load(samples, allow_pickle=False)
    gen_coords, gen_lengths = d["coords"], d["lengths"]
    ref_coords, _, ref_lengths = reference("test")
    ref_coords, ref_lengths = ref_coords[:ref_stats["n"]], ref_lengths[:ref_stats["n"]]

    print(f"scoring {len(gen_lengths):,} generated and {len(ref_lengths):,} reference structures")
    gen_rows = [per_structure(gen_coords[i], int(gen_lengths[i])) for i in range(len(gen_lengths))]
    ref_rows = [per_structure(ref_coords[i], int(ref_lengths[i])) for i in range(len(ref_lengths))]
    # the figures must reproduce the reported table exactly
    for name, rows, stats in (("generated", gen_rows, gen_stats), ("reference", ref_rows, ref_stats)):
        v = float(np.mean([r["valid"] for r in rows]))
        assert abs(v - stats["validity"]) < 1e-9, f"{name} validity {v} != {stats['validity']}"

    def steps(coords, lengths):
        return np.concatenate([np.linalg.norm(np.diff(coords[i][:int(lengths[i])], axis=0), axis=1)
                               for i in range(len(lengths))])

    def pick(lengths, rows, targets, seed):
        """Clash-free examples near each target length, without repeats.

        An undertrained run has no valid structure at all, so fall back to the whole pool rather
        than failing: a gallery of bad backbones is exactly what you want to look at then.
        """
        rng = np.random.default_rng(seed)
        lengths = np.asarray(lengths)
        valid = np.where(np.array([r["valid"] for r in rows]))[0]
        pool = valid if len(valid) else np.arange(len(rows))
        chosen = []
        for t in targets:
            free = np.setdiff1d(pool, chosen, assume_unique=False)
            if not len(free):
                break
            near = free[np.abs(lengths[free] - t) < 12]
            chosen.append(int(rng.choice(near)) if len(near)
                          else int(free[np.argmin(np.abs(lengths[free] - t))]))
        return chosen, bool(len(valid))

    epoch = np.array([h["epoch"] for h in history])
    train = np.array([h["train_loss"] for h in history])
    dev_raw = np.array([h["valid_loss_raw"] for h in history])
    dev_ema = np.array([h["valid_loss"] for h in history])

    meta = dict(n_train=0, n_dev=0, n_test=int(ref_stats["n"]),
                train_hours=float(sum(h["seconds"] for h in history) / 3600.0),
                epochs=len(history), params=0, layers=0, hidden=0, k=0, batch=0, timesteps=0)
    try:  # the run's own hyperparameters, so the header cannot be stale
        import torch
        ck = torch.load(os.path.join(run, "checkpoint.pt"), map_location="cpu", weights_only=False)
        a = ck["args"]
        meta.update(params=sum(v.numel() for v in ck["model"].values() if hasattr(v, "numel")),
                    layers=a["layers"], hidden=a["hidden"], k=a["k"],
                    batch=a["batch_size"], timesteps=a["timesteps"])
        for split, key in (("train", "n_train"), ("dev", "n_dev")):
            _, _, lens = reference(split)
            meta[key] = len(lens)
    except Exception as exc:  # figures must still draw without torch or the checkpoint
        print(f"  note: run metadata unavailable ({exc})")

    targets = [35, 60, 80, 100, 120]
    ref_picks, _ = pick(ref_lengths, ref_rows, targets, 0)
    gen_picks, gen_any_valid = pick(gen_lengths, gen_rows, targets, 0)
    return dict(
        history=history, epoch=epoch, train=train, dev_raw=dev_raw, dev_ema=dev_ema,
        best_epoch=int(epoch[dev_ema.argmin()]),
        history_curves=[("train", train, INK2, 1.6), ("dev", dev_raw, REF_C, 2.0),
                        ("dev (EMA)", dev_ema, GEN_C, 2.0)],
        ref_stats=ref_stats, gen_stats=gen_stats, ref_rows=ref_rows, gen_rows=gen_rows,
        ref_bins=binned(ref_rows), gen_bins=binned(gen_rows),
        ref_steps=steps(ref_coords, ref_lengths), gen_steps=steps(gen_coords, gen_lengths),
        ref_coords=ref_coords, ref_lengths=ref_lengths,
        gen_coords=gen_coords, gen_lengths=gen_lengths,
        ref_picks=ref_picks, gen_picks=gen_picks,
        gallery_note=("clash-free examples matched on length" if gen_any_valid
                      else "examples matched on length; NO generated structure is clash-free"),
        meta=meta)


def write_csv(ctx, out):
    path = os.path.join(out, "figure_data.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "series", "key", "value"])
        for k, v in (("best_epoch", ctx["best_epoch"]), ("train_final", ctx["train"][-1]),
                     ("dev_raw_final", ctx["dev_raw"][-1]), ("dev_ema_final", ctx["dev_ema"][-1]),
                     ("dev_ema_best", ctx["dev_ema"].min()),
                     ("train_hours", ctx["meta"]["train_hours"])):
            w.writerow(["fig1_training", "run", k, f"{v:.6g}"])
        for key in ("n", "validity", "bond_mean", "bond_sd", "bond_ok", "clash_free",
                    "clashes_per_structure", "radius_mean", "right_handed", "helix_fraction"):
            w.writerow(["fig2_sample_quality", REF_LABEL, key, ctx["ref_stats"][key]])
            w.writerow(["fig2_sample_quality", GEN_LABEL, key, ctx["gen_stats"][key]])
        for series, tab in ((REF_LABEL, ctx["ref_bins"]), (GEN_LABEL, ctx["gen_bins"])):
            for r in tab:
                for key in ("n", "validity", "clash_free", "clashes", "radius"):
                    w.writerow(["fig4_length_stratified", f"{series} {r['bin']}", key,
                                r[key] if key == "n" else f"{r[key]:.4f}"])
    print("  wrote figure_data.csv")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    module = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(module, "results", "ca_sparse_k16"))
    ap.add_argument("--samples", default=None)
    ap.add_argument("--out", default=os.path.join(module, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    samples = args.samples or os.path.join(args.run, "samples_10000.npz")

    ctx = build_context(args.run, samples)
    print("drawing")
    fig_overview(ctx, args.out)
    fig_training(ctx, args.out)
    fig_quality(ctx, args.out)
    fig_distributions(ctx, args.out)
    fig_length(ctx, args.out)
    fig_backbones(ctx, args.out)
    write_csv(ctx, args.out)


if __name__ == "__main__":
    main()
