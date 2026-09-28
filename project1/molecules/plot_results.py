"""Draw the result figures for the modality-1 run: training, then sections 4.2 to 4.5.

Follows the palette and conventions of benchmarks/make_figures.py in the benchmarks branch, so
these figures sit next to the EDM/GCDM ones without a visual seam.

Reads    results/*.json                   the metric tables metrics.py wrote
         results/*.pt, ckpt/propeval.pt   the samples, rescored per molecule for the intervals
         the log that logs/LATEST names   training curves: train.py prints them and keeps none
Writes   figures/fig_training, fig42_nfe, fig43_guidance, fig44_ablation, fig45_headline
         (.png and .pdf), figures/figure_data.csv, and the cache results/per_sample.npz

    python plot_results.py
    python plot_results.py --only fig44_ablation fig45_headline

The JSONs hold point estimates only, so the error bars are 95% bootstrap intervals over
molecules, with the settings of benchmarks/bootstrap_eval.py. Every run of one size is
resampled with the same indices: runs that share a seed start from the same prior draws, and
resampling them together is what lets a difference between two rows carry its own interval.
Each aggregate recomputed from the per-molecule values is checked against its JSON before
anything is drawn, so the figures cannot drift from the reported tables.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# palette, copied from benchmarks/make_figures.py so the two sets of figures match
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # slot 1 and slot 2
REF = "#898781"                          # every control and reference row
# guidance strength is ordered: one hue stepped light to dark, with the unguided run in grey
W_COLOURS = {0: REF, 1: "#86b6ef", 2: "#2a78d6", 4: "#0d366b"}

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

N_BOOT, SEED, CI = 1000, 0, (2.5, 97.5)      # as benchmarks/bootstrap_eval.py
FIGURES = ("fig_training", "fig42_nfe", "fig43_guidance", "fig44_ablation", "fig45_headline")

METRICS = [("prop_mae", "Property MAE", "Bohr$^3$, lower is better"),
           ("validity", "Validity", "fraction of molecules"),
           ("mol_stability", "Molecule stability", "fraction of molecules"),
           ("atom_stability", "Atom stability", "fraction of atoms")]
FLOW_RUNS, DIFF_RUN = ("42_flow_50", "42_flow_100", "42_flow_200"), "42_diff"
WS = (0, 1, 2, 4)
ABLATION = [("44_none", "No gate"),
            ("44_global", "Global gate"),
            ("44_shuffled", "Shuffled gate"),
            ("44_per_atom", "Per-atom gate (ours)"),
            ("44_penalty_r0.001", "Additive penalty, ρ = 0.001"),
            ("44_penalty_r0.01", "Additive penalty, ρ = 0.01"),
            ("44_wmatched", "No gate, w-matched")]
HEADLINE = [("45_base", "Base: no gate"), ("45_ours", "Ours: per-atom gate")]


# ------------------------------------------------------------------ data
def parse_log(path):
    curves, key = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.search(r"\b(flow|diff|prop): [\d.]+M parameters", line)
        if m:
            key = m.group(1)
            curves[key] = ([], [])
        m = re.search(r"epoch\s+\d+\s+train ([\d.]+)\s+val ([\d.]+)", line)
        if m and key:
            curves[key][0].append(float(m.group(1)))
            curves[key][1].append(float(m.group(2)))
    return {k: (np.array(tr), np.array(va)) for k, (tr, va) in curves.items()}


def build_per_sample(names, prop_ckpt, data, results):
    """Per-molecule atom count, stable-atom count, validity and predicted property for each
    run, plus the true property over the training split as the reference distribution."""
    import torch

    from data.qm9 import H_SCALE, QM9Data
    from guidance import VALENCE
    from metrics import bond_orders, to_smiles
    from sample import load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ck = load(prop_ckpt, device)
    mu, sd = ck["prop_mean"].item(), ck["prop_std"].item()
    out = {"qm9_train/alpha": (QM9Data(data).splits["train"][3] * sd + mu).numpy()}
    for name in names:
        s = torch.load(results / f"{name}.pt", weights_only=False)
        x, h, mask = (s[k].to(device) for k in ("x", "h", "mask"))
        types = (h / H_SCALE).argmax(-1)
        order = bond_orders(x, types, mask)
        ok = (order.sum(-1).float() == torch.tensor(VALENCE, device=device)[types]) * mask
        with torch.no_grad():
            pred = net(x, h, torch.ones(len(x), device=device), mask) * sd + mu
        n, types, order = mask.sum(-1).long().tolist(), types.cpu(), order.cpu()
        valid = [to_smiles(types[i], order[i], n[i]) is not None for i in range(len(x))]
        out.update({f"{name}/atoms": np.array(n), f"{name}/stable": ok.sum(-1).cpu().numpy(),
                    f"{name}/valid": np.array(valid), f"{name}/pred": pred.cpu().numpy()})
        print(f"  scored {name}: {len(x):,} molecules")
    return out


def aggregates(ps, name, idx, target):
    """The metrics.py aggregates over the molecules `idx` of one run."""
    atoms, stable = ps[f"{name}/atoms"][idx], ps[f"{name}/stable"][idx]
    out = {"atom_stability": stable.sum() / atoms.sum(),
           "mol_stability": (stable == atoms).mean(),
           "validity": ps[f"{name}/valid"][idx].mean()}
    if target is not None:                       # the unguided 4.2 rows request no property
        out["prop_mae"] = np.abs(ps[f"{name}/pred"][idx] - target).mean()
    return out


def bootstrap(ps, runs):
    """{run: {metric: (value, the N_BOOT resampled values)}}"""
    rng = np.random.default_rng(SEED)
    index, out = {}, {}
    for name, r in runs.items():
        n, target = r["n_samples"], r["config"]["target"]
        if n not in index:
            index[n] = rng.integers(0, n, size=(N_BOOT, n))
        point = aggregates(ps, name, slice(None), target)
        draws = [aggregates(ps, name, i, target) for i in index[n]]
        out[name] = {k: (float(v), np.array([d[k] for d in draws])) for k, v in point.items()}
    return out


def build_context(results, log, prop_ckpt, data):
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(results.glob("*.json"))}
    cache = results / "per_sample.npz"
    ps = dict(np.load(cache)) if cache.exists() else {}

    def drifted(name):
        if f"{name}/atoms" not in ps:
            return True
        agg = aggregates(ps, name, slice(None), runs[name]["config"]["target"])
        return any(abs(v - runs[name][k]) > 1e-4 * max(1.0, abs(v)) for k, v in agg.items())

    todo = [n for n in runs if drifted(n)]
    if todo:
        print(f"scoring {len(todo)} runs per molecule")
        ps.update(build_per_sample(todo, prop_ckpt, data, results))
        # the figures must reproduce the reported tables exactly
        assert not any(drifted(n) for n in todo), "per-molecule values disagree with the JSON"
        np.savez_compressed(cache, **ps)
    return dict(runs=runs, ps=ps, boot=bootstrap(ps, runs), curves=parse_log(log))


def interval(ctx, name, key):
    v, draws = ctx["boot"][name][key]
    return (v, *np.percentile(draws, CI))


def difference(ctx, a, b, key):
    """b - a, with the interval of the difference itself rather than of either row."""
    (va, da), (vb, db) = ctx["boot"][a][key], ctx["boot"][b][key]
    return (vb - va, *np.percentile(db - da, CI))


def ci_note(ctx, name, each="row"):
    return (f"Error bars: 95% bootstrap intervals over the {ctx['runs'][name]['n_samples']:,} "
            f"samples of each {each} ({N_BOOT:,} resamples).")


# ------------------------------------------------------------------ small helpers
def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def signed(v):
    return f"{v:+.3f}".replace("-", "−")


def bars(ax, x, y, v, lo, hi, colour, horizontal=False, capsize=2.5):
    err = [np.subtract(v, lo), np.subtract(hi, v)]
    ax.errorbar(x, y, fmt="none", ecolor=colour, elinewidth=1, capsize=capsize, capthick=1,
                zorder=3, **({"xerr": err} if horizontal else {"yerr": err}))


def footer(fig, handles, note):
    """Legend below the panels and the interval note under it, as in the benchmark figures."""
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.0, 0.0),
               handlelength=1.4)
    fig.text(0.008, -0.30 / fig.get_figheight(), note, ha="left", va="top", fontsize=7.5,
             color=MUTED)


# ------------------------------------------------------------------------- panels
def panel_training(ax, ctx, key, title, ylabel):
    train, val = ctx["curves"][key]
    ep = np.arange(1, len(train) + 1)
    best = int(val.argmin())                     # the epoch train.py kept
    ax.plot(ep, train, color=BLUE, linewidth=1.6, label="train", zorder=3)
    ax.plot(ep, val, color=ORANGE, linewidth=1.2, label="validation", alpha=0.85, zorder=3)
    ax.scatter([ep[best]], [val[best]], s=34, facecolor=SURFACE, edgecolor=ORANGE,
               linewidth=1.6, zorder=5,
               label=f"saved: epoch {ep[best]}, validation {val[best]:.4f}")
    if key == "prop":
        ax.set_yscale("log")
    ax.set_xlim(0, len(ep) * 1.02)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", handlelength=1.4)
    style(ax)


def panel_nfe(ax, ctx, key, title, unit):
    for names, colour, label in ((FLOW_RUNS, BLUE, "flow matching, Euler steps"),
                                 ((DIFF_RUN,), ORANGE, "diffusion, DDPM ancestral")):
        x = [ctx["runs"][n]["nfe"] for n in names]
        v, lo, hi = zip(*[interval(ctx, n, key) for n in names])
        bars(ax, x, v, v, lo, hi, colour)
        ax.plot(x, v, color=colour, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4, label=label)
        ax.annotate(f"{v[-1]:.3f}", xy=(x[-1], v[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 1000])
    ax.set_xticklabels(["50", "100", "200", "1,000"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(38, 1900)
    ax.set_xlabel("network evaluations per sample (log scale)")
    ax.set_ylabel(unit)
    ax.set_title(title)
    style(ax)


def panel_property(fig, spec, ctx):
    """One row per distribution on a shared density scale, so that narrower reads as taller."""
    target = ctx["runs"]["43_cfg_w0"]["config"]["target"]
    rows = [("QM9 training split", REF, ctx["ps"]["qm9_train/alpha"])]
    rows += [("unguided, w = 0" if w == 0 else f"guided, w = {w}", W_COLOURS[w],
              ctx["ps"][f"43_cfg_w{w}/pred"]) for w in WS]
    bins = np.linspace(45, 105, 61)
    top = max(np.histogram(v, bins=bins, density=True)[0].max() for _, _, v in rows) * 1.12
    inner = spec.subgridspec(len(rows), 1, hspace=0.10)
    for i, (label, colour, v) in enumerate(rows):
        ax = fig.add_subplot(inner[i])
        ax.hist(v, bins=bins, density=True, color=colour, alpha=0.45, histtype="stepfilled",
                zorder=3)
        ax.hist(v, bins=bins, density=True, color=colour, linewidth=1.6, histtype="step",
                zorder=4)
        ax.axvline(target, color=INK, linewidth=1.0, zorder=5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.tick_params(length=0, labelbottom=i == len(rows) - 1)
        ax.spines["left"].set_visible(False)
        outside = 1 - np.mean((v >= bins[0]) & (v <= bins[-1]))
        ax.text(0.0, 0.70, label, transform=ax.transAxes, fontsize=8, color=INK2)
        ax.text(1.0, 0.70, f"mean {v.mean():.1f}, sd {v.std():.1f}"
                + (f", {outside:.1%} off scale" if outside > 0.005 else ""),
                transform=ax.transAxes, fontsize=8, color=MUTED, ha="right")
        if i == 0:
            ax.set_title("Property of the samples", pad=16)
            ax.annotate(f"target {target:g}", xy=(target, top), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8,
                        color=INK2)
    ax.set_xlabel("polarizability (Bohr$^3$)\n"
                  "true for QM9, predicted by the evaluator for samples")


def panel_sweep(ax, ctx, keys, labels=None):
    """Metrics against guidance strength, in the row colours of panel_property."""
    for key in keys:
        v, lo, hi = zip(*[interval(ctx, f"43_cfg_w{w}", key) for w in WS])
        ax.plot(WS, v, color=AXIS, linewidth=2, zorder=2)
        for j, w in enumerate(WS):
            # on the 0-1 axis the intervals are shorter than a marker; caps would only poke out
            bars(ax, [w], [v[j]], [v[j]], [lo[j]], [hi[j]], INK2, capsize=0 if labels else 2.5)
            ax.scatter([w], [v[j]], s=70, facecolor=W_COLOURS[w], edgecolor=SURFACE,
                       linewidth=2, zorder=4)
            if labels is None:
                ax.annotate(f"{v[j]:.2f}", xy=(w, v[j]), xytext=(7, 7),
                            textcoords="offset points", fontsize=8, color=INK2)
        if labels:
            ax.annotate(labels[key], xy=(WS[-1], v[-1]), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_xticks(WS)
    ax.set_xlabel("guidance strength w")
    style(ax)


def panel_rows(axes, ctx, rows, ours, ref):
    """One panel per metric, one row per run. The hairline marks the reference row; a filled
    marker is a row whose difference from it has a 95% interval that excludes zero."""
    for ax, (key, title, unit) in zip(axes, METRICS):
        ax.axvline(interval(ctx, ref, key)[0], color=AXIS, linewidth=0.8, zorder=1)
        for y, (name, _) in enumerate(rows):
            v, lo, hi = interval(ctx, name, key)
            colour = BLUE if name == ours else REF
            _, dlo, dhi = difference(ctx, ref, name, key)
            solid = name == ref or dlo > 0 or dhi < 0
            bars(ax, [v], [y], [v], [lo], [hi], colour, horizontal=True)
            ax.scatter([v], [y], s=70, marker="D" if name == ref else "o", zorder=4,
                       linewidth=2, facecolor=colour if solid else SURFACE,
                       edgecolor=SURFACE if solid else colour)
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.margins(x=0.14)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        style(ax, axis="x")
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for _, label in rows])


def rows_footer(fig, ctx, ref_label, name):
    footer(fig, [
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=REF, markeredgecolor=SURFACE,
               markersize=8, label=ref_label),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=INK2, markeredgecolor=INK2,
               markersize=8, label="differs from the reference"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=SURFACE,
               markeredgecolor=INK2, markeredgewidth=1.6, markersize=8,
               label="does not: the 95% interval of the paired difference includes zero"),
    ], ci_note(ctx, name) + " Differences are tested one at a time, without a correction "
                            "for the number of rows.")


# ------------------------------------------------------------- standalone figures
def fig_training(ctx, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    for ax, args in zip(axes, (("flow", "Flow matching", "flow-matching loss"),
                               ("diff", "Diffusion", "noise-prediction loss"),
                               ("prop", "Property evaluator", "squared error (log scale)"))):
        panel_training(ax, ctx, *args)
    fig.suptitle("Training and validation loss of the three networks, seed 0", fontsize=12,
                 fontweight="semibold", color=INK, y=1.0)
    fig.tight_layout()
    save(fig, out, "fig_training")


def fig_nfe(ctx, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    for ax, (key, title, unit) in zip(axes, METRICS[:0:-1]):
        panel_nfe(ax, ctx, key, title, unit)
    fig.suptitle("4.2  Unguided sample quality against sampling cost", fontsize=12,
                 fontweight="semibold", color=INK, y=1.0)
    fig.tight_layout()
    footer(fig, axes[0].get_legend_handles_labels()[0], ci_note(ctx, DIFF_RUN, "point"))
    save(fig, out, "fig42_nfe")


def fig_guidance(ctx, out):
    fig = plt.figure(figsize=(11.4, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.12], wspace=0.30, left=0.04,
                          right=0.99, top=0.80, bottom=0.17)
    panel_property(fig, gs[0], ctx)

    ax = fig.add_subplot(gs[1])
    panel_sweep(ax, ctx, ["prop_mae"])
    ax.set_ylim(0, interval(ctx, "43_cfg_w0", "prop_mae")[0] * 1.18)
    ax.set_ylabel("Bohr$^3$, lower is better")
    ax.set_title("Property MAE", pad=16)

    ax = fig.add_subplot(gs[2])
    panel_sweep(ax, ctx, ["atom_stability", "validity", "mol_stability"],
                {"atom_stability": "atom stability", "validity": "validity",
                 "mol_stability": "molecule stability"})
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.3, 6.6)
    ax.set_ylabel("fraction of atoms or molecules")
    ax.set_title("Sample quality", pad=16)

    rmse = np.sqrt(ctx["curves"]["prop"][1].min()) * ctx["ps"]["qm9_train/alpha"].std()
    fig.text(0.04, 0.0, ci_note(ctx, "43_cfg_w0", "setting") + " Gate off throughout. "
             f"Evaluator error on the QM9 validation split: RMSE {rmse:.2f} Bohr$^3$.",
             ha="left", va="top", fontsize=7.5, color=MUTED)
    fig.suptitle("4.3  Classifier-free guidance toward a polarizability of "
                 f"{ctx['runs']['43_cfg_w0']['config']['target']:g} Bohr$^3$", fontsize=12,
                 fontweight="semibold", color=INK, y=0.97)
    save(fig, out, "fig43_guidance")


def fig_ablation(ctx, out):
    w = {name: ctx["runs"][name]["config"]["w"] for name, _ in ABLATION}
    rows = [(name, f"{label} (w = {w[name]:.2f})" if name == "44_wmatched" else label)
            for name, label in ABLATION]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.4, 3.5), sharey=True)
    panel_rows(axes, ctx, rows, "44_per_atom", "44_none")
    fig.suptitle(f"4.4  Ablation at guidance strength w = {w['44_none']:g}", fontsize=12,
                 fontweight="semibold", color=INK, y=1.0)
    fig.tight_layout()
    rows_footer(fig, ctx, "reference: plain guidance, no gate", "44_none")
    save(fig, out, "fig44_ablation")


def fig_headline(ctx, out):
    (base, _), (ours, _) = HEADLINE
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.4, 2.7), sharey=True)
    panel_rows(axes, ctx, HEADLINE, ours, base)
    for ax, (key, title, _) in zip(axes, METRICS):
        d, lo, hi = difference(ctx, base, ours, key)
        ax.set_title(title, pad=19)
        ax.text(0.5, 1.04, f"ours − base {signed(d)}  [{signed(lo)}, {signed(hi)}]",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=INK2)
    fig.suptitle(f"4.5  Per-atom gate against the base model, w = "
                 f"{ctx['runs'][base]['config']['w']:g}, "
                 f"{ctx['runs'][base]['n_samples']:,} samples each", fontsize=12,
                 fontweight="semibold", color=INK, y=1.0)
    fig.tight_layout()
    rows_footer(fig, ctx, "reference: base model", base)
    save(fig, out, "fig45_headline")


def write_csv(ctx, out):
    # table twin of every plotted value
    with (out / "figure_data.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "run", "metric", "value", "ci_lo", "ci_hi"])
        for key, (train, val) in ctx["curves"].items():
            best = int(val.argmin())
            for k, v in (("saved_epoch", best + 1), ("saved_validation_loss", val[best]),
                         ("final_train_loss", train[-1]), ("final_validation_loss", val[-1])):
                w.writerow(["fig_training", key, k, f"{v:.6g}", "", ""])
        for name in ctx["runs"]:
            figure = {"42": "fig42_nfe", "43": "fig43_guidance", "44": "fig44_ablation",
                      "45": "fig45_headline"}[name[:2]]
            for key in ctx["boot"][name]:
                w.writerow([figure, name, key] + [f"{v:.4f}" for v in interval(ctx, name, key)])
        guided = [f"43_cfg_w{w}" for w in WS[1:]]
        for figure, ref, rows in (("fig43_guidance", "43_cfg_w0", guided),
                                  ("fig44_ablation", "44_none", [n for n, _ in ABLATION[1:]]),
                                  ("fig45_headline", "45_base", [n for n, _ in HEADLINE[1:]])):
            for name in rows:
                for key, _, _ in METRICS:
                    w.writerow([figure, f"{name} - {ref}", key]
                               + [f"{v:.4f}" for v in difference(ctx, ref, name, key)])
    print("  wrote figure_data.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--log", default=None, help="run log; default is the one logs/LATEST names")
    p.add_argument("--prop-ckpt", default="ckpt/propeval.pt")
    p.add_argument("--data", default="data/qm9.pt")
    p.add_argument("--only", nargs="+", choices=FIGURES, default=FIGURES)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ctx = build_context(a.results, a.log or Path("logs/LATEST").read_text().strip(),
                        a.prop_ckpt, a.data)
    print("drawing")
    for name, draw in zip(FIGURES, (fig_training, fig_nfe, fig_guidance, fig_ablation,
                                    fig_headline)):
        if name in a.only:
            draw(ctx, a.out)
    write_csv(ctx, a.out)


if __name__ == "__main__":
    main()
