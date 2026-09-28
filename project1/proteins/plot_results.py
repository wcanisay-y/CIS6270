"""Draw the result figures for the modality-2 protein run: training, then section 4.6.

Follows the palette and conventions of benchmarks/make_figures.py in the benchmarks branch and
of molecules/plot_results.py in the molecules branch, so the three sets of figures match.

Reads    out/*.json                       the metric tables metrics.py wrote
         out/*.pt, data/pfam_esm2.pt,     the samples and the real sequences, rescored per
         ckpt/classifier.pt, ckpt/flow.pt sample; the flow model replays the gate
         the log that logs/LATEST names   training curves: train.py prints them and keeps none
Writes   figures/fig_training, fig46_guidance, fig46_ablation, fig46_gate (.png and .pdf),
         figures/figure_data.csv, and the cache out/per_sample.npz

    python plot_results.py
    python plot_results.py --only fig46_ablation

Family accuracy, validity, uniqueness and novelty come out at 1.000 for every guided run, so
they cannot separate the gate from its controls. The ablation therefore plots four quantities
that are not saturated, each computed per sample with the functions of metrics.py and
guidance.py: the classifier's probability for the requested family, diversity, the decoding
confidence the gate itself is built on, and identity to the nearest training sequence.
Diversity is taken over all pairs of samples; metrics.py estimates the same mean from 10,000
random pairs, and the two agree to within 0.001.

Error bars are 95% bootstrap intervals over samples, with the settings of
benchmarks/bootstrap_eval.py. Runs that share a seed start from the same prior draws and the
same requested families, so all runs are resampled with the same indices and a difference
between two rows carries its own interval. Every aggregate recomputed here is checked against
its JSON before anything is drawn, and the replayed sampler must reproduce the saved samples
exactly, so the figures cannot drift from the reported tables.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# palette, copied from benchmarks/make_figures.py so the sets of figures match
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # slots 1 to 3
REF = "#898781"                                           # every control and reference row
RAMP = ["#86b6ef", "#2a78d6", "#0d366b"]                  # one hue, light to dark: an order
HEAT = LinearSegmentedColormap.from_list("heat", [SURFACE, "#cde2fb", "#86b6ef", "#2a78d6",
                                                  "#104281"])

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
FIGURES = ("fig_training", "fig46_guidance", "fig46_ablation", "fig46_gate")

UNGUIDED, PLAIN, OURS, REAL = "abl_none", "abl_cfg", "abl_gated", "real_test"
ABLATION = [("abl_cfg", "No gate"),
            ("abl_global", "Global gate"),
            ("abl_shuffled", "Shuffled gate"),
            ("abl_gated", "Per-residue gate (ours)")]
METRICS = [("family_prob", "Family confidence", "probability of the requested family"),
           ("diversity", "Diversity", "1 − mean pairwise identity"),
           ("decoding", "Decoding confidence", "mean top probability per residue"),
           ("nearest", "Nearest training sequence", "sequence identity")]
SATURATED = [("family_accuracy", "family accuracy"), ("valid_rate", "validity"),
             ("uniqueness", "uniqueness"), ("novelty", "novelty")]
GATES = {"per_residue": "abl_gated", "global": "abl_global"}     # gate mode -> its run
AA = "ACDEFGHIKLMNPQRSTVWYXBUZO"             # decoded letters, as small integers


# ------------------------------------------------------------------ data
def parse_log(path):
    curves, key = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.search(r"\b(flow|classifier): [\d.]+M parameters", line)
        if m:
            key = m.group(1)
            curves[key] = ([], [])
        m = re.search(r"epoch\s+\d+\s+train ([\d.]+)\s+val ([\d.]+)", line)
        if m and key:
            curves[key][0].append(float(m.group(1)))
            curves[key][1].append(float(m.group(2)))
    return {k: (np.array(tr), np.array(va)) for k, (tr, va) in curves.items()}


def tokens(sequences):
    """[n, longest] small integers, -1 past the end of a sequence."""
    out = np.full((len(sequences), max(map(len, sequences))), -1, np.int8)
    for i, s in enumerate(sequences):
        out[i, :len(s)] = [AA.index(c) for c in s]
    return out


def identity(a, b):
    """[len(a), len(b)] sequence identity, as metrics.sequence_identity defines it: matches
    over the shorter of the two lengths."""
    la, lb = (a >= 0).sum(1), (b >= 0).sum(1)
    out = np.zeros((len(a), len(b)), np.float32)
    for i in range(0, len(a), 100):                       # in blocks, to bound the memory
        block = a[i:i + 100, None, :] == b[None, :, :]
        block &= (a[i:i + 100, None, :] >= 0) & (b[None, :, :] >= 0)
        out[i:i + 100] = block.sum(-1) / np.maximum(np.minimum(la[i:i + 100, None], lb[None]), 1)
    return out


def build_per_sample(names, out_dir, data_path, clf_ckpt, flow_ckpt, runs):
    """Per-sample values for each run and for the real test split, and the gate replayed over
    the sampling trajectory. The replay mirrors sample.py, which returns only the mean gate."""
    import torch

    from data.pfam import PfamData
    from flow_matching import sample_prior
    from guidance import cfg, get_decoder
    from metrics import decode_sequences, mean_pairwise_identity, sequence_identity
    from sample import load, sample_family

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PfamData(data_path)
    clf, _ = load(clf_ckpt, device)
    pool_strings = sorted(data.train_sequences)
    pool = tokens(pool_strings)
    out = {"meta/families": np.array(data.family_names)}

    def score(name, z, family, z_mean, z_std):
        seqs = decode_sequences(z, z_mean, z_std, device)
        tok = tokens(seqs)
        with torch.no_grad():
            prob = torch.softmax(clf(z.to(device)), -1).cpu()
            top = torch.softmax(get_decoder(device).decode_logits(z.to(device), z_mean, z_std),
                                -1).max(-1).values.mean(-1).cpu()
        near = identity(tok, pool)
        # the vectorised identity must be the one metrics.py defines
        rng = np.random.RandomState(0)
        for i, j in zip(rng.randint(len(seqs), size=20), rng.randint(len(pool), size=20)):
            assert abs(near[i, j] - sequence_identity(seqs[i], pool_strings[j])) < 1e-6
        valid = np.array([s.count("X") < len(s) * 0.5 for s in seqs])
        pairs = mean_pairwise_identity([s for s, v in zip(seqs, valid) if v])
        out.update({f"{name}/family": family.numpy(), f"{name}/pred": prob.argmax(-1).numpy(),
                    f"{name}/family_prob": prob[torch.arange(len(family)), family].numpy(),
                    f"{name}/decoding": top.numpy(), f"{name}/nearest": near.max(1),
                    f"{name}/tokens": tok, f"{name}/valid": valid,
                    f"{name}/novel": np.array([s not in data.train_sequences for s in seqs]),
                    f"{name}/pairs": np.float64(pairs)})
        print(f"  scored {name}: {len(seqs):,} sequences")

    for name in names:
        s = torch.load(out_dir / f"{name}.pt", weights_only=False)
        score(name, s["z"], s["family"], float(s["z_mean"]), float(s["z_std"]))
        # metrics.py estimates diversity from 10,000 random pairs; its own function must agree
        assert abs(out[f"{name}/pairs"] - runs[name]["mean_pairwise_identity"]) < 1e-9, name
    z, family = data.splits["test"]
    score(REAL, z, family, float(data.z_mean), float(data.z_std))

    for mode, name in GATES.items():
        saved = torch.load(out_dir / f"{name}.pt", weights_only=False)
        c = saved["config"]
        torch.manual_seed(c["seed"])
        model, ck = load(flow_ckpt, device)
        z_mean, z_std = float(ck["z_mean"]), float(ck["z_std"])
        gates, finals, families, done = [], [], [], 0
        while done < c["n"]:
            b = min(c["batch_size"], c["n"] - done)
            fam = sample_family(ck["family_hist"], b, device)
            z = sample_prior(b, device)
            steps = []
            for i in range(c["steps"]):
                t = torch.full((b,), i / c["steps"], device=device)
                with torch.no_grad():
                    v, g = cfg(model, z, t, fam, c["w"], mode, c["beta"], z_mean, z_std)
                steps.append(g.squeeze(-1).cpu())
                z = z + v / c["steps"]
            gates.append(torch.stack(steps))
            finals.append(z.cpu())
            families.append(fam.cpu())
            done += b
        assert torch.equal(torch.cat(finals), saved["z"]), f"replay of {mode} left the samples"
        g = torch.cat(gates, 1).numpy()                                   # [steps, n, L]
        flat = g.reshape(len(g), -1) if mode == "per_residue" else g.mean(-1)
        fam = torch.cat(families).numpy()
        out.update({f"gate/{mode}/mean": g.mean((1, 2)),
                    f"gate/{mode}/quantiles": np.percentile(flat, [10, 50, 90], axis=1),
                    f"gate/{mode}/beta": np.float64(c["beta"])})
        if mode == "per_residue":
            pick = [0, len(g) // 2, len(g) - 1]
            out.update({"gate/times": np.array(pick) / len(g),
                        "gate/values": g[pick].reshape(3, -1).astype(np.float32),
                        "gate/profile": np.stack([g[-1][fam == f].mean(0)
                                                  for f in range(len(data.family_names))])})
        print(f"  replayed the {mode} gate: {g.shape[1]:,} samples, {len(g)} steps")

    with torch.no_grad():
        z, _ = data.splits["test"]
        top = torch.softmax(get_decoder(device).decode_logits(
            z.to(device), float(data.z_mean), float(data.z_std)), -1).max(-1).values
    out["gate/real"] = np.float64(torch.exp(-(1 - top) ** 2).mean().item())
    return out


def aggregates(ps, name):
    """The metrics.py aggregates that can be recomputed from the per-sample values."""
    valid, tok = ps[f"{name}/valid"], ps[f"{name}/tokens"]
    unique = {t.tobytes() for t in tok[valid]}
    novel = {t.tobytes() for t in tok[valid & ps[f"{name}/novel"]]}
    return {"family_accuracy": (ps[f"{name}/pred"] == ps[f"{name}/family"]).mean(),
            "valid_rate": valid.mean(),
            "uniqueness": len(unique) / max(valid.sum(), 1),
            "novelty": len(novel) / max(len(unique), 1),
            "mean_pairwise_identity": float(ps[f"{name}/pairs"])}


def bootstrap(ps, names):
    """{run: {metric: (value, the N_BOOT resampled values)}}. Diversity is a statistic of
    pairs, so a resample is a vector of counts c and its mean identity is c'Mc over the pairs
    of two different samples."""
    rng = np.random.default_rng(SEED)
    n = len(ps[f"{names[0]}/family"])
    counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n)
                       for _ in range(N_BOOT)]).astype(np.float32)
    out = {}
    for name in names:
        out[name] = {k: (float(ps[f"{name}/{k}"].mean()), counts @ ps[f"{name}/{k}"] / n)
                     for k in ("family_prob", "decoding", "nearest")}
        m = identity(ps[f"{name}/tokens"], ps[f"{name}/tokens"])
        np.fill_diagonal(m, 0)
        pairs = lambda c: ((c @ m) * c).sum(1) / (c.sum(1) ** 2 - (c ** 2).sum(1))
        valid = ps[f"{name}/valid"].astype(np.float32)
        every = float(pairs(valid[None])[0])
        # metrics.py estimates this mean from 10,000 random pairs; all pairs are used here so
        # that the value and its interval come from one statistic
        assert abs(every - float(ps[f"{name}/pairs"])) < 2e-3, name
        out[name]["diversity"] = (1 - every, 1 - pairs(counts * valid))
    return out


def build_context(out_dir, log, data_path, clf_ckpt, flow_ckpt):
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir.glob("abl_*.json"))}
    cache = out_dir / "per_sample.npz"
    ps = dict(np.load(cache)) if cache.exists() else {}

    def drifted(name):
        if f"{name}/family" not in ps or "gate/real" not in ps:
            return True
        return any(abs(v - runs[name][k]) > 1e-6 for k, v in aggregates(ps, name).items())

    if any(drifted(n) for n in runs):
        print(f"scoring {len(runs)} runs per sample")
        ps = build_per_sample(list(runs), out_dir, data_path, clf_ckpt, flow_ckpt, runs)
        # the figures must reproduce the reported tables exactly
        assert not any(drifted(n) for n in runs), "per-sample values disagree with the JSON"
        np.savez_compressed(cache, **ps)
    return dict(runs=runs, ps=ps, boot=bootstrap(ps, [n for n, _ in ABLATION]),
                curves=parse_log(log), families=[str(f) for f in ps["meta/families"]])


def interval(ctx, name, key):
    v, draws = ctx["boot"][name][key]
    return (v, *np.percentile(draws, CI))


def difference(ctx, a, b, key):
    """b - a, with the interval of the difference itself rather than of either row."""
    (va, da), (vb, db) = ctx["boot"][a][key], ctx["boot"][b][key]
    return (vb - va, *np.percentile(db - da, CI))


def real(ctx, key):
    if key == "diversity":
        return 1 - float(ctx["ps"][f"{REAL}/pairs"])
    return float(ctx["ps"][f"{REAL}/{key}"].mean())


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


def bars(ax, x, y, v, lo, hi, colour):
    ax.errorbar(x, y, xerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none", ecolor=colour,
                elinewidth=1, capsize=2.5, capthick=1, zorder=3)


def footer(fig, handles, note):
    """Legend below the panels and the interval note under it, as in the benchmark figures."""
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.0, 0.0),
               handlelength=1.4)
    fig.text(0.008, -0.30 / fig.get_figheight(), note, ha="left", va="top", fontsize=7.5,
             color=MUTED)


# ------------------------------------------------------------------------- panels
def panel_training(ax, ctx, key, title, ylabel, log=False):
    train, val = ctx["curves"][key]
    ep = np.arange(1, len(train) + 1)
    best = int(val.argmin())                     # the epoch train.py kept
    ax.plot(ep, train, color=BLUE, linewidth=1.6, label="train", zorder=3)
    ax.plot(ep, val, color=ORANGE, linewidth=1.2, label="validation", alpha=0.85, zorder=3)
    ax.scatter([ep[best]], [val[best]], s=34, facecolor=SURFACE, edgecolor=ORANGE,
               linewidth=1.6, zorder=5,
               label=f"saved: epoch {ep[best]}, validation {val[best]:.4f}")
    if log:
        ax.set_yscale("log")
    ax.set_xlim(0, len(ep) * 1.02)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", handlelength=1.4)
    style(ax)


def panel_confusion(fig, spec, ctx, title):
    """Requested against predicted family, one grid per guidance strength. Every cell carries
    its value, so the colour needs no scale of its own."""
    fam = ctx["families"]
    w = ctx["runs"][PLAIN]["config"]["w"]
    inner = spec.subgridspec(1, 2, wspace=0.12)
    for k, (name, label) in enumerate(((UNGUIDED, "no guidance, w = 0"),
                                       (PLAIN, f"guided, w = {w:g}"))):
        want, got = ctx["ps"][f"{name}/family"], ctx["ps"][f"{name}/pred"]
        m = np.array([[np.mean(got[want == i] == j) for j in range(len(fam))]
                      for i in range(len(fam))])
        ax = fig.add_subplot(inner[k])
        ax.imshow(m, cmap=HEAT, vmin=0, vmax=1, aspect="equal")
        for i in range(len(fam)):
            for j in range(len(fam)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color=SURFACE if m[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(fam)), labels=fam, fontsize=7.5)
        ax.set_yticks(range(len(fam)), labels=fam if k == 0 else [""] * len(fam), fontsize=7.5)
        ax.set_xticks(np.arange(-0.5, len(fam)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(fam)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)     # the gap between the cells
        ax.tick_params(which="both", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("predicted family")
        if k == 0:
            ax.set_ylabel("requested family")
        ax.text(0.5, 1.04, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color=INK2)
    # the grids are square and do not fill the cell, so the title is set on the cell itself,
    # at the height the other panels' titles take
    box = spec.get_position(fig)
    fig.text((box.x0 + box.x1) / 2, box.y1 + 16 / 72 / fig.get_figheight(), title, ha="center",
             va="baseline", fontsize=10.5, fontweight="semibold", color=INK)


def panel_ridges(fig, spec, rows, bins, title, xlabel, marks=()):
    """One row per distribution on a shared density scale, so that narrower reads as taller.
    The top third of each row is kept free for its label, wherever the peak falls."""
    top = max(np.histogram(v, bins=bins, density=True)[0].max() for _, _, v in rows) * 1.55
    inner = spec.subgridspec(len(rows), 1, hspace=0.10)
    for i, (label, colour, v) in enumerate(rows):
        ax = fig.add_subplot(inner[i])
        ax.hist(v, bins=bins, density=True, color=colour, alpha=0.45, histtype="stepfilled",
                zorder=3)
        ax.hist(v, bins=bins, density=True, color=colour, linewidth=1.6, histtype="step",
                zorder=4)
        for x, _ in marks:
            ax.axvline(x, color=INK, linewidth=1.0, zorder=5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.tick_params(length=0, labelbottom=i == len(rows) - 1)
        ax.spines["left"].set_visible(False)
        ax.text(0.0, 0.78, label, transform=ax.transAxes, fontsize=8, color=INK2)
        ax.text(1.0, 0.78, f"mean {np.mean(v):.2f}, sd {np.std(v):.2f}",
                transform=ax.transAxes, fontsize=8, color=MUTED, ha="right")
        if i == 0:
            ax.set_title(title, pad=16)
            for x, text in marks:
                ax.annotate(text, xy=(x, top), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlabel(xlabel)


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
            bars(ax, [v], [y], [v], [lo], [hi], colour)
            ax.scatter([v], [y], s=70, marker="D" if name == ref else "o", zorder=4,
                       linewidth=2, facecolor=colour if solid else SURFACE,
                       edgecolor=SURFACE if solid else colour)
        ax.set_title(title, pad=19)
        ax.text(0.5, 1.04, f"real test sequences: {real(ctx, key):.3f}", ha="center",
                transform=ax.transAxes, va="bottom", fontsize=8, color=INK2)
        ax.set_xlabel(unit)
        ax.margins(x=0.16)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        style(ax, axis="x")
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for _, label in rows])


def panel_gate_time(ax, ctx):
    ps = ctx["ps"]
    mean = ps["gate/per_residue/mean"]
    t = np.arange(len(mean)) / len(mean)
    # The two modes have the same mean by construction; what differs is how far single
    # residues, or single sequences, sit from it.
    handles = []
    for mode, colour, alpha, label in (
            ("per_residue", BLUE, 0.18, "per-residue gate, over residues"),
            ("global", INK2, 0.30, "global gate, over sequences")):
        lo, _, hi = ps[f"gate/{mode}/quantiles"]
        ax.fill_between(t, lo, hi, color=colour, alpha=alpha, linewidth=0, zorder=2)
        handles.append(Patch(facecolor=colour, alpha=alpha, label=label))
    gap = np.abs(mean - ps["gate/global/mean"]).max()
    ax.plot(t, mean, color=INK, linewidth=2, zorder=3)
    handles.append(Line2D([0], [0], color=INK, linewidth=2,
                          label="mean gate, equal in both modes" if gap < 5e-4
                          else f"mean gate (modes within {gap:.3f})"))
    floor = float(np.exp(-ps["gate/per_residue/beta"]))
    for y, text, va in ((floor, f"lowest possible gate, {floor:.2f}", "bottom"),
                        (float(ps["gate/real"]), f"real test latents, {float(ps['gate/real']):.2f}",
                         "bottom")):
        ax.axhline(y, color=AXIS, linewidth=0.8, zorder=1)
        ax.text(0.99, y + 0.008, text, ha="right", va=va, fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("sampling time t")
    ax.set_ylabel("gate value")
    ax.set_title("The gate over the trajectory", pad=16)
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.13), handlelength=1.4)
    style(ax)


def panel_gate_position(ax, ctx):
    profile = ctx["ps"]["gate/profile"]
    position = np.arange(1, profile.shape[1] + 1)
    for row, colour, name in zip(profile, (BLUE, ORANGE, AQUA), ctx["families"]):
        ax.plot(position, row, color=colour, linewidth=2, zorder=3, label=name)
    ax.set_xlim(0, profile.shape[1] + 1)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("residue position")
    ax.set_ylabel("mean gate at the last step")
    ax.set_title("The gate along the sequence", pad=16)
    ax.legend(loc="upper right", handlelength=1.4, title="requested family", title_fontsize=8)
    style(ax)


# ------------------------------------------------------------- standalone figures
def fig_training(ctx, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.4))
    panel_training(axes[0], ctx, "flow", "Flow matching", "flow-matching loss")
    panel_training(axes[1], ctx, "classifier", "Family classifier (evaluator)",
                   "cross-entropy (log scale)", log=True)
    fig.suptitle("Training and validation loss of the two networks, seed 0", fontsize=12,
                 fontweight="semibold", color=INK, y=1.0)
    fig.tight_layout()
    save(fig, out, "fig_training")


def fig_guidance(ctx, out):
    ps = ctx["ps"]
    w = ctx["runs"][PLAIN]["config"]["w"]
    fig = plt.figure(figsize=(11.4, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.22, left=0.06, right=0.99,
                          top=0.80, bottom=0.17)
    panel_confusion(fig, gs[0], ctx, "Requested against predicted family")
    rows = lambda key: [("real test sequences", REF, ps[f"{REAL}/{key}"]),
                        ("no guidance, w = 0", RAMP[0], ps[f"{UNGUIDED}/{key}"]),
                        (f"guided, w = {w:g}", RAMP[1], ps[f"{PLAIN}/{key}"])]
    panel_ridges(fig, gs[1], rows("nearest"), np.linspace(0, 1, 41),
                 "Nearest training sequence", "sequence identity")
    panel_ridges(fig, gs[2], rows("decoding"), np.linspace(0.2, 0.8, 41),
                 "Decoding confidence", "mean top probability per residue")
    n = ctx["runs"][PLAIN]["n_samples"]
    fig.text(0.06, 0.0, f"Gate off throughout. {n:,} samples per setting, "
             f"{len(ps[f'{REAL}/family']):,} real test sequences. Families are predicted by "
             "the classifier, sequences decoded by the frozen ESM-2 head.", ha="left",
             va="top", fontsize=7.5, color=MUTED)
    fig.suptitle("4.6  Classifier-free guidance toward a requested Pfam family", fontsize=12,
                 fontweight="semibold", color=INK, y=0.97)
    save(fig, out, "fig46_guidance")


def fig_ablation(ctx, out):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.4, 2.9), sharey=True)
    panel_rows(axes, ctx, ABLATION, OURS, PLAIN)
    fig.suptitle(f"4.6  Ablation at guidance strength w = "
                 f"{ctx['runs'][PLAIN]['config']['w']:g}", fontsize=12,
                 fontweight="semibold", color=INK, y=1.0)
    fig.tight_layout()
    flat = []
    for key, label in SATURATED:
        v = [ctx["runs"][name][key] for name, _ in ABLATION]
        flat.append(f"{label} {min(v):.3f}" + ("" if min(v) == max(v) else f" to {max(v):.3f}"))
    footer(fig, [
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=REF, markeredgecolor=SURFACE,
               markersize=8, label="reference: plain guidance, no gate"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=INK2, markeredgecolor=INK2,
               markersize=8, label="differs from the reference"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=SURFACE,
               markeredgecolor=INK2, markeredgewidth=1.6, markersize=8,
               label="does not: the 95% interval of the paired difference includes zero"),
    ], ci_note(ctx, PLAIN) + " Differences are tested one at a time. In every row: "
       + ", ".join(flat) + ".")
    save(fig, out, "fig46_ablation")


def fig_gate(ctx, out):
    ps = ctx["ps"]
    fig = plt.figure(figsize=(11.4, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.1], wspace=0.28, left=0.06, right=0.99,
                          top=0.80, bottom=0.17)
    panel_gate_time(fig.add_subplot(gs[0]), ctx)
    rows = [(f"t = {t:.2f}", colour, v)
            for t, colour, v in zip(ps["gate/times"], RAMP, ps["gate/values"])]
    panel_ridges(fig, gs[1], rows, np.linspace(0.3, 1.0, 36), "Gate values of all residues",
                 "gate value")
    panel_gate_position(fig.add_subplot(gs[2]), ctx)
    c = ctx["runs"][OURS]["config"]
    fig.text(0.06, 0.0, f"Replay of the {c['n']:,} saved samples at w = {c['w']:g} and "
             f"β = {c['beta']:g}; it reproduces them exactly. Bands run from the 10th to "
             "the 90th percentile. "
             f"sample.py logs the mean over the trajectory, {ctx['runs'][OURS]['mean_gate']:.3f}.",
             ha="left", va="top", fontsize=7.5, color=MUTED)
    fig.suptitle("4.6  The feasibility gate during sampling", fontsize=12,
                 fontweight="semibold", color=INK, y=0.97)
    save(fig, out, "fig46_gate")


def write_csv(ctx, out):
    # table twin of every plotted value
    ps = ctx["ps"]
    with (out / "figure_data.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "run", "metric", "value", "ci_lo", "ci_hi"])
        for key, (train, val) in ctx["curves"].items():
            best = int(val.argmin())
            for k, v in (("saved_epoch", best + 1), ("saved_validation_loss", val[best]),
                         ("final_train_loss", train[-1]), ("final_validation_loss", val[-1])):
                w.writerow(["fig_training", key, k, f"{v:.6g}", "", ""])
        fam = ctx["families"]
        for name in (UNGUIDED, PLAIN):
            want, got = ps[f"{name}/family"], ps[f"{name}/pred"]
            for i, a in enumerate(fam):
                for j, b in enumerate(fam):
                    w.writerow(["fig46_guidance", name, f"requested_{a}_predicted_{b}",
                                f"{np.mean(got[want == i] == j):.4f}", "", ""])
        for name in (REAL, UNGUIDED, PLAIN):
            for key in ("nearest", "decoding"):
                w.writerow(["fig46_guidance", name, f"{key}_mean",
                            f"{ps[f'{name}/{key}'].mean():.4f}", "", ""])
        for name, _ in ABLATION:
            for key, _, _ in METRICS:
                w.writerow(["fig46_ablation", name, key]
                           + [f"{v:.4f}" for v in interval(ctx, name, key)])
        # every row against plain guidance, then the gate against its two controls
        pairs = [(PLAIN, name) for name, _ in ABLATION[1:]]
        pairs += [(name, OURS) for name, _ in ABLATION[1:] if name != OURS]
        for ref, name in pairs:
            for key, _, _ in METRICS:
                w.writerow(["fig46_ablation", f"{name} - {ref}", key]
                           + [f"{v:.4f}" for v in difference(ctx, ref, name, key)])
        for key, _, _ in METRICS:
            w.writerow(["fig46_ablation", REAL, key, f"{real(ctx, key):.4f}", "", ""])
        for mode in GATES:
            lo, mid, hi = ps[f"gate/{mode}/quantiles"]
            for i in range(0, len(mid), 10):
                w.writerow(["fig46_gate", mode, f"gate_t{i / len(mid):.2f}", f"{mid[i]:.4f}",
                            f"{lo[i]:.4f}", f"{hi[i]:.4f}"])
        w.writerow(["fig46_gate", REAL, "gate_mean", f"{float(ps['gate/real']):.4f}", "", ""])
    print("  wrote figure_data.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("out"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--log", default=None, help="run log; default is the one logs/LATEST names")
    p.add_argument("--data", default="data/pfam_esm2.pt")
    p.add_argument("--classifier-ckpt", default="ckpt/classifier.pt")
    p.add_argument("--flow-ckpt", default="ckpt/flow.pt")
    p.add_argument("--only", nargs="+", choices=FIGURES, default=FIGURES)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ctx = build_context(a.results, a.log or Path("logs/LATEST").read_text().strip(), a.data,
                        a.classifier_ckpt, a.flow_ckpt)
    print("drawing")
    for name, draw in zip(FIGURES, (fig_training, fig_guidance, fig_ablation, fig_gate)):
        if name in a.only:
            draw(ctx, a.out)
    write_csv(ctx, a.out)


if __name__ == "__main__":
    main()
