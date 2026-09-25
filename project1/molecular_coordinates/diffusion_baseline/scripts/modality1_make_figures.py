"""Draw result figures for the modality-1 QM9 diffusion baseline.

Follows the palette and conventions of benchmarks/make_figures.py in the benchmarks branch, so
these figures sit next to the EDM/GCDM ones without a visual seam.

Reads from --results:  metrics.json, samples.xyz, qm9_reference_stats.json
Writes to --out:       fig*.png, fig*.pdf, figure_data.csv

    python modality1_make_figures.py --results ../results/ddpm_baseline
"""
import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# palette, copied from benchmarks/make_figures.py so the two sets of figures match
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLOR = "#2a78d6"          # slot 1 blue, as EDM uses in the benchmark figures
ACCENT = "#eb6834"               # slot 2 orange
REF = "#898781"

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

ATOMS = ["H", "C", "N", "O", "F"]
METRICS = [("atom_stability", "Atom\nstability"), ("molecule_stability", "Molecule\nstability"),
           ("validity", "Validity"), ("uniqueness", "Uniqueness"), ("novelty", "Novelty")]


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def read_xyz(path):
    mols, lines, i = [], path.read_text().splitlines(), 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        rows = [l.split() for l in lines[i + 2:i + 2 + n]]
        mols.append(([r[0] for r in rows],
                     np.array([[float(v) for v in r[1:4]] for r in rows])))
        i += 2 + n
    return mols


def geometry(mols):
    nn, rg, types = [], [], []
    for syms, xyz in mols:
        types.extend(ATOMS.index(s) for s in syms)
        if len(xyz) < 2:
            continue
        d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn.extend(d.min(1).tolist())
        rg.append(float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean())))
    return np.array(nn), np.array(rg), np.bincount(types, minlength=len(ATOMS))


# figure 1: the headline metrics, against the QM9 data row as the achievable ceiling
def draw_quality(ax, metrics, note="artefact\nsee Fig. 3"):
    xs = np.arange(len(METRICS))
    vals = [metrics.get(k, 0.0) for k, _ in METRICS]
    bars = ax.bar(xs, vals, width=0.52, color=MODEL_COLOR, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK2, zorder=6)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in METRICS])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(f"Fraction of {metrics['n_molecules']:,} samples")
    ax.set_title("Sample quality, MLP denoiser on QM9")
    # The two bars that look like successes are artefacts; mark them rather than let a reader
    # take them at face value.
    for i, (k, _) in enumerate(METRICS):
        if k in ("validity", "novelty"):
            ax.text(i, 0.06, note, ha="center", va="bottom", fontsize=7,
                    color=ACCENT, zorder=7, fontweight="semibold")
    style(ax)


# figure 2: training curves. Included to show the model did fit something -- the failure is not
# a failure to optimize.
def draw_training(ax, history):
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_loss"] for h in history], color=MODEL_COLOR, linewidth=1.6,
            label="train", zorder=3)
    ax.plot(ep, [h["valid_loss"] for h in history], color=ACCENT, linewidth=1.2,
            label="validation", alpha=0.85, zorder=3)
    floor = 8.0   # E||eps||^2 for a model that predicts zero: 3 coordinate + 5 type dimensions
    ax.axhline(floor, color=REF, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(ep) * 0.5, floor - 0.3, "predict-zero floor", ha="center", va="top",
            fontsize=7.5, color=MUTED)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Denoising loss")
    ax.set_title("Training converged")
    ax.set_ylim(0, 8.6)
    ax.legend(loc="center right", handlelength=1.4)
    style(ax)


# figure 3: the diagnosis. Nearest-neighbour distance is what decides whether any bond is
# inferred at all, and therefore what every downstream metric rests on.
def draw_geometry(ax, gen_nn, ref):
    edges = np.array(ref["nn_edges"])
    centres = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    real = np.array(ref["nn_hist"], dtype=float)
    real /= real.sum()
    gen = np.histogram(np.clip(gen_nn, 0, edges[-1]), bins=len(centres), range=(0, edges[-1]))[0]
    gen = gen.astype(float) / max(gen.sum(), 1)
    ax.bar(centres, real, width=width, color=REF, alpha=0.75, zorder=3, label="QM9 data")
    ax.bar(centres, gen, width=width, color=MODEL_COLOR, alpha=0.8, zorder=4, label="generated")
    ax.axvline(1.8, color=ACCENT, linewidth=1.2, linestyle="--", zorder=5)
    ax.text(2.05, max(real.max(), gen.max()) * 1.0, "1.8 A: longest bond inferred",
            fontsize=7.5, color=ACCENT, va="top", ha="left")
    ax.set_xlim(0, edges[-1])
    # Everything beyond the last edge is piled into the final bin, so say so rather than let the
    # spike read as a real mode at 8 A -- the generated mean is 56 A.
    top = max(real.max(), gen.max())
    ax.annotate(f"all distances > {edges[-1]:.0f} A\n(generated mean {gen_nn.mean():.0f} A)",
                xy=(edges[-1] - width / 2, gen[-1]), xytext=(edges[-1] * 0.60, top * 0.80),
                fontsize=7.5, color=MODEL_COLOR, ha="center",
                arrowprops=dict(arrowstyle="->", color=MODEL_COLOR, linewidth=1))
    ax.set_xlabel("Nearest-neighbour distance (A)")
    ax.set_ylabel("Fraction of atoms")
    ax.set_title("Why every metric collapses")
    ax.legend(loc="center left", handlelength=1.4, bbox_to_anchor=(0.30, 0.42))
    style(ax)


# figure 4: composition. A model that learned nothing about chemistry emits the five elements at
# roughly equal rates; QM9 is dominated by hydrogen and carbon.
def draw_composition(ax, gen_counts, ref):
    xs = np.arange(len(ATOMS))
    real = np.array(ref["composition"])
    gen = gen_counts / max(gen_counts.sum(), 1)
    ax.bar(xs - 0.2, real, width=0.38, color=REF, zorder=3, label="QM9 data")
    ax.bar(xs + 0.2, gen, width=0.38, color=MODEL_COLOR, zorder=3, label="generated")
    ax.axhline(1 / len(ATOMS), color=ACCENT, linewidth=1.1, linestyle="--", zorder=4)
    ax.text(len(ATOMS) - 0.55, 1 / len(ATOMS) + 0.012, "uniform (1/5)", fontsize=7.5,
            color=ACCENT, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(ATOMS)
    ax.set_xlabel("Element")
    ax.set_ylabel("Fraction of atoms")
    ax.set_title("Atom types are near-uniform")
    ax.legend(loc="upper right", handlelength=1.4)
    style(ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (args.results / "figures")
    out.mkdir(parents=True, exist_ok=True)

    bundle = json.loads((args.results / "metrics.json").read_text())
    metrics, history, config = bundle["metrics"], bundle["history"], bundle["config"]
    ref = json.loads((args.results / "qm9_reference_stats.json").read_text())
    gen_nn, gen_rg, gen_counts = geometry(read_xyz(args.results / "samples.xyz"))

    fig, ax = plt.subplots(figsize=(6.6, 3.6)); draw_quality(ax, metrics)
    save(fig, out, "fig1_sample_quality")
    fig, ax = plt.subplots(figsize=(5.6, 3.2)); draw_training(ax, history)
    save(fig, out, "fig2_training")
    fig, ax = plt.subplots(figsize=(5.6, 3.2)); draw_geometry(ax, gen_nn, ref)
    save(fig, out, "fig3_geometry")
    fig, ax = plt.subplots(figsize=(5.0, 3.2)); draw_composition(ax, gen_counts, ref)
    save(fig, out, "fig4_composition")

    # Overview: what the run was and how it trained. The geometry and composition diagnostics
    # stay as standalone figures -- the MLP arm is a floor, not a result to dwell on.
    fig = plt.figure(figsize=(12.4, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.15], wspace=0.30)
    draw_quality(fig.add_subplot(gs[0, 0]), metrics, note="artefact")
    draw_training(fig.add_subplot(gs[0, 1]), history)
    panel = fig.add_subplot(gs[0, 2]); panel.axis("off")
    limit = config.get("limit", "None")
    lines = [
        "Model setup",
        "",
        f"  denoiser        {config['denoiser']}, {bundle['parameters']:,} params",
        f"  layers x width  {config['layers']} x {config['hidden']}",
        f"  data            QM9, {'130,831' if str(limit) == 'None' else limit} molecules",
        f"  split           80 / 10 / 10, seed {config['seed']}",
        f"  diffusion       DDPM, T={config['timesteps']}, polynomial_2",
        f"  parameterised   epsilon-prediction, L2",
        f"  training        {config['epochs']} epochs, batch {config['batch_size']}",
        f"  optimiser       AdamW, lr {config['lr']}, EMA {config['ema']}",
        f"  augmentation    random SO(3) per batch",
        f"  hardware        4 CPU cores, 27 min",
        "",
        "Validity and novelty are artefacts: no atom pair",
        "lands close enough to bond, so every atom is its",
        'own fragment and RDKit sanitises a lone "C".',
    ]
    panel.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8, color=INK2,
               family="monospace", transform=panel.transAxes)
    fig.suptitle("QM9 unconditional generation: MLP denoiser baseline", fontsize=12,
                 fontweight="semibold", color=INK, y=1.02)
    save(fig, out, "fig_overview")

    with (out / "figure_data.csv").open("w") as f:
        f.write("group,key,value\n")
        for k, _ in METRICS:
            f.write(f"metric,{k},{metrics.get(k, float('nan')):.6f}\n")
        f.write(f"geometry,generated_nn_mean,{gen_nn.mean():.4f}\n")
        f.write(f"geometry,real_nn_mean,{ref['nn_mean']:.4f}\n")
        f.write(f"geometry,generated_rg_mean,{gen_rg.mean():.4f}\n")
        f.write(f"geometry,real_rg_mean,{ref['rg_mean']:.4f}\n")
        f.write(f"geometry,generated_frac_bonded,{(gen_nn < 1.8).mean():.4f}\n")
        f.write(f"geometry,real_frac_bonded,{ref['frac_bonded']:.4f}\n")
        for i, a in enumerate(ATOMS):
            f.write(f"composition,generated_{a},{gen_counts[i]/max(gen_counts.sum(),1):.4f}\n")
            f.write(f"composition,real_{a},{ref['composition'][i]:.4f}\n")
        f.write(f"training,final_train_loss,{history[-1]['train_loss']:.4f}\n")
        f.write(f"training,final_valid_loss,{history[-1]['valid_loss']:.4f}\n")

    print(f"wrote {len(sorted(out.iterdir()))} files to {out}")
    for p in sorted(out.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
