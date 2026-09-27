"""Render the inference-time generation trajectory: one molecule per model, four time points.

    python trajectory.py --flow ckpt/flow_s0.pt --diff ckpt/diff_s0.pt

Row 1 is the flow model integrating the learned velocity from t=0 to t=1. Row 2 is the
diffusion reverse chain from k=K down to k=0. Both rows are drawn from the SAME prior
sample (same seed, same atom count), so the two processes can be read against each other
rather than being two unrelated pictures.

The integration here mirrors sample_flow and sample_diffusion in sample.py exactly. It is
duplicated rather than imported because those functions return only the final state, and a
trajectory needs the intermediate ones; if the samplers in sample.py change, change these too.

Writes <out>.png and <out>.pdf. This is the raw material for the 3.7 overview figure.
"""
import argparse

import torch

from data.qm9 import ATOMS, H_SCALE, MAX_N
from flow_matching import sample_prior, zero_com
from guidance import cfg
from metrics import bond_orders
from sample import load

# CPK colours and rough relative sizes, indexed like ATOMS = [H, C, N, O, F].
COLOURS = ["#E8E8E8", "#3B3B3B", "#3050F8", "#FF0D0D", "#90E050"]
# Deliberately small: at eight panels on one page, markers large enough to read individually
# cover the bonds between them, and the bonds are the part that shows structure emerging.
SIZES = [22, 55, 55, 55, 48]


@torch.no_grad()
def flow_trajectory(model, mask, cond, steps, w, gate, beta, n_frames=4):
    """Euler integration of dx/dt = v_theta, keeping n_frames evenly spaced states.

    Frames are captured BEFORE each step, so frame 0 is the prior draw at t=0, and the
    final frame is appended after the loop at t=1.
    """
    x, h = sample_prior(mask, mask.device)
    dt = 1.0 / steps
    # n_frames - 1 captures inside the loop at t = 0, 1/3, 2/3; the last is the finished sample
    grab = {round(i * steps / (n_frames - 1)) for i in range(n_frames - 1)}
    frames = []
    for i in range(steps):
        t = torch.full((mask.shape[0],), i * dt, device=mask.device)
        if i in grab:
            frames.append((f"t = {i * dt:.2f}", x.clone(), h.clone()))
        vx, vh, _ = cfg(model, x, h, t, mask, cond, w, gate, beta, H_SCALE)
        x = zero_com(x + dt * vx, mask)
        h = (h + dt * vh) * mask[..., None]
    frames.append(("t = 1.00", x, h))
    return frames


@torch.no_grad()
def diffusion_trajectory(model, mask, cond, w, n_frames=4):
    """DDPM ancestral chain from k=K to k=0, keeping n_frames evenly spaced states.

    Labelled by k/K so the row reads left to right as noise -> molecule, matching the flow
    row above it even though the two time variables run in opposite directions.
    """
    x, h = sample_prior(mask, mask.device)
    b, K = mask.shape[0], model.K
    off, on = torch.zeros(b, device=mask.device), torch.ones(b, device=mask.device)
    grab = {K - round(i * K / (n_frames - 1)) for i in range(n_frames - 1)}
    frames = []
    for k in range(K, 0, -1):
        if k in grab:
            frames.append((f"k/K = {k / K:.2f}", x.clone(), h.clone()))
        t = torch.full((b,), k / K, device=mask.device)
        ex, eh = model.field(x, h, t, mask, cond, off)
        if w != 0:
            cx, ch = model.field(x, h, t, mask, cond, on)
            ex, eh = ex + w * (cx - ex), eh + w * (ch - eh)
        sigma = (1 - model.abar[k]).sqrt()
        mx = (x - model.betas[k] * ex / sigma) / model.alphas[k].sqrt()
        mh = (h - model.betas[k] * eh / sigma) / model.alphas[k].sqrt()
        if k > 1:
            zx, zh = sample_prior(mask, mask.device)
            s = model.post_var[k].sqrt()
            x, h = mx + s * zx, mh + s * zh
        else:
            x, h = mx, mh
        x = zero_com(x, mask)
        h = h * mask[..., None]
    frames.append(("k/K = 0.00", x, h))
    return frames


def draw(ax, x, h, mask, limit):
    """One panel: atoms coloured by element, plus any bond the distance table infers.

    Early frames are nearly pure noise, so the inferred bonds there are spurious by
    construction. That is the point of the figure -- structure has not formed yet -- so they
    are drawn faintly rather than suppressed.
    """
    n = int(mask[0].sum())
    coords = x[0, :n].cpu()
    types = (h[0, :n] / H_SCALE).argmax(-1).cpu()
    order = bond_orders(x[:, :n], types[None], mask[:, :n])[0].cpu()

    for i in range(n):
        for j in range(i + 1, n):
            if order[i, j] > 0:
                seg = coords[[i, j]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                        color="#888888", linewidth=1.0, alpha=0.5, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=[COLOURS[t] for t in types], s=[SIZES[t] for t in types],
               edgecolors="#222222", linewidths=0.4, depthshade=False, zorder=2)

    # One shared limit across every panel, so the expansion from a compact N(0,1) prior to a
    # molecule spanning several Angstrom is a real change and not per-panel autoscaling.
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow", default="ckpt/flow_s0.pt")
    p.add_argument("--diff", default="ckpt/diff_s0.pt")
    p.add_argument("--out", default="figures/fig_trajectory")
    p.add_argument("--steps", type=int, default=200, help="Euler steps (flow only)")
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--n-atoms", type=int, default=None,
                   help="atoms in the shown molecule; default draws from the QM9 size histogram")
    p.add_argument("--w", type=float, default=0.0, help="CFG strength; 0 is unguided")
    p.add_argument("--gate", default="none",
                   choices=["none", "global", "shuffled", "per_atom"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--target", type=float, default=None, help="property value, original units")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow, fck = load(a.flow, device)
    diff, _ = load(a.diff, device)

    # One molecule, one size, one prior draw, shared by both rows. Seeding immediately before
    # each trajectory is what makes the two rows start from the identical noise sample.
    torch.manual_seed(a.seed)
    if a.n_atoms is None:
        n_atoms = int(torch.multinomial(fck["n_hist"], 1).item())
    else:
        n_atoms = a.n_atoms
    mask = (torch.arange(MAX_N, device=device)[None] < n_atoms).float()

    mu, sd = fck["prop_mean"].item(), fck["prop_std"].item()
    target = mu if a.target is None else a.target
    cond = torch.full((1,), (target - mu) / sd, device=device)
    print(f"molecule with {n_atoms} atoms, w={a.w}, gate={a.gate}, device={device}")

    torch.manual_seed(a.seed)
    rows = [("Flow matching", flow_trajectory(flow, mask, cond, a.steps, a.w, a.gate,
                                              a.beta, a.frames))]
    torch.manual_seed(a.seed)
    rows.append(("Diffusion", diffusion_trajectory(diff, mask, cond, a.w, a.frames)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The prior is a unit Gaussian and QM9 coordinates are raw Angstrom spanning ~7 A, so the
    # trajectory EXPANDS. max (not a percentile) keeps every atom inside the frame.
    limit = max(float(x[0, :n_atoms].abs().max()) for _, fr in rows for _, x, _ in fr) * 1.05

    fig = plt.figure(figsize=(3.1 * a.frames, 3.4 * len(rows)))
    for r, (name, frames) in enumerate(rows):
        for c, (label, x, h) in enumerate(frames):
            ax = fig.add_subplot(len(rows), a.frames, r * a.frames + c + 1, projection="3d")
            draw(ax, x, h, mask, limit)
            ax.set_title(label, fontsize=10, pad=0)
            if c == 0:
                ax.text2D(-0.08, 0.5, name, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=12, fontweight="bold")

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=COLOURS[i],
                          markeredgecolor="#222222", label=s) for i, s in enumerate(ATOMS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(ATOMS), frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Inference-time generation trajectory, one {n_atoms}-atom molecule per model"
                 + ("" if a.w == 0 else f"  (w={a.w}, gate={a.gate})"), fontsize=13)
    fig.tight_layout(rect=(0.01, 0.045, 1, 0.96))

    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"wrote {a.out}.{ext}")


if __name__ == "__main__":
    main()
