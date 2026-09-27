#!/usr/bin/env python3
"""Train the sparse SE(3) Ca-backbone diffusion model on the SwissProt-Pfam cap-128 subset.

Separate from the QM9 baseline: it imports only sparse_egnn.py and reads only the packed Ca
tensors written by prepare_structures.py.

    python train_protein.py --smoke                 # 2 epochs on 2,000 proteins, minutes
    python train_protein.py --epochs 200            # the costed run
    python train_protein.py --resume                # continue from the last checkpoint
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sparse_egnn import (N_RESIDUE_TYPES, TYPE_SCALE, Diffusion, ProteinDenoiser, remove_mean)

# The structure cache is bulky input data and stays outside the repository; override the root
# with PROTEIN_DIFFUSION_ROOT if it moves. Runs land in the module's own results/ directory.
ROOT = Path(os.environ.get("PROTEIN_DIFFUSION_ROOT", "/mnt/isilon/tan_lab/pany3/protein_diffusion"))
CACHE = ROOT / "cache"
RUNS = Path(__file__).resolve().parent.parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RES = 128


def load_split(split, limit=None):
    data = np.load(CACHE / f"{split}_ca.npz", allow_pickle=False)
    coords = torch.from_numpy(data["coords"]).float()
    types = torch.from_numpy(data["types"].astype(np.int64))
    lengths = torch.from_numpy(data["lengths"].astype(np.int64))
    if limit:
        coords, types, lengths = coords[:limit], types[:limit], lengths[:limit]
    mask = (torch.arange(MAX_RES)[None] < lengths[:, None]).float()[..., None]
    coords = remove_mean(coords, mask)                     # centre every protein once, up front
    one_hot = torch.zeros(len(types), MAX_RES, N_RESIDUE_TYPES)
    valid = types.clamp(min=0)
    one_hot.scatter_(2, valid[..., None], 1.0)
    one_hot = one_hot * mask
    return coords, one_hot, lengths, mask


def ema_update(shadow, model, decay):
    with torch.no_grad():
        for s, p in zip(shadow.parameters(), model.parameters()):
            s.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for s, p in zip(shadow.buffers(), model.buffers()):
            s.copy_(p)


def batches(n, size, shuffle, generator=None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, size):
        yield order[i:i + size]


def loss_on(model, diffusion, coords, one_hot, lengths, mask, index):
    x = coords[index].to(DEVICE, non_blocking=True)
    h = one_hot[index].to(DEVICE, non_blocking=True) * TYPE_SCALE
    m = mask[index].to(DEVICE, non_blocking=True)
    lg = lengths[index].to(DEVICE, non_blocking=True)
    t_index = torch.randint(1, diffusion.T + 1, (len(index),), device=DEVICE)
    z_x, z_h, eps_x, eps_h = diffusion.noise(x, h, m, t_index)
    t = t_index.float() / diffusion.T
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_x, pred_h = model(z_x, z_h, t, m, lg)
        denominator = m.sum().clamp(min=1)
        loss = (((pred_x.float() - eps_x) ** 2).sum() / (denominator * 3)
                + ((pred_h.float() - eps_h) ** 2).sum() / (denominator * N_RESIDUE_TYPES))
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    # Lets the loop be exercised on a partial cache before the full fetch finishes, and lets a
    # later arm train on a different packed subset without editing the loader.
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--dev-split", default="dev")
    ap.add_argument("--output", type=Path, default=RUNS / "ca_sparse_k16")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.limit = 2, 2000

    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    coords, one_hot, lengths, mask = load_split(args.train_split, args.limit)
    dev = load_split(args.dev_split, 2000 if args.smoke else None)
    # One global scale, as the baseline does, so the data matches the unit-variance prior.
    coord_std = coords[mask.expand_as(coords).bool()].std().item()
    coords = coords / coord_std
    dev = (dev[0] / coord_std,) + dev[1:]
    length_histogram = torch.bincount(lengths, minlength=MAX_RES + 1).float()
    print(f"train {len(coords):,} proteins | dev {len(dev[0]):,} | coord_std {coord_std:.3f} A "
          f"| mean length {lengths.float().mean():.1f}")

    model = ProteinDenoiser(args.hidden, args.layers, args.k).to(DEVICE)
    shadow = ProteinDenoiser(args.hidden, args.layers, args.k).to(DEVICE)
    shadow.load_state_dict(model.state_dict())
    for p in shadow.parameters():
        p.requires_grad_(False)
    diffusion = Diffusion(args.timesteps, DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"denoiser: sparse EGNN, {args.layers} layers, hidden {args.hidden}, k={args.k}, "
          f"{parameters:,} parameters")

    start_epoch, history = 1, []
    checkpoint = args.output / "checkpoint.pt"
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"]); shadow.load_state_dict(state["ema"])
        optimiser.load_state_dict(state["optimiser"])
        start_epoch, history = state["epoch"] + 1, state["history"]
        print(f"resumed from epoch {state['epoch']}")

    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started, total, seen = time.perf_counter(), 0.0, 0
        for index in batches(len(coords), args.batch_size, True, generator):
            loss = loss_on(model, diffusion, coords, one_hot, lengths, mask, index)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimiser.step()
            ema_update(shadow, model, args.ema)
            total += loss.item() * len(index); seen += len(index)
        train_loss = total / seen
        elapsed = time.perf_counter() - started

        # Both are reported on purpose. The EMA shadow starts at the random initialisation and
        # has an effective window of 1/(1-decay) updates, so for the first several epochs its dev
        # loss lags far behind and looks alarming while nothing is wrong. The raw model's dev loss
        # is the one to read early; the EMA's is the one that matters at the end.
        shadow.eval(); model.eval()
        valid = {}
        with torch.no_grad():
            for label, net in (("ema", shadow), ("raw", model)):
                total_v, seen_v = 0.0, 0
                for index in batches(len(dev[0]), args.batch_size, False):
                    total_v += loss_on(net, diffusion, dev[0], dev[1], dev[2], dev[3],
                                       index).item() * len(index)
                    seen_v += len(index)
                valid[label] = total_v / seen_v
        valid_loss = valid["ema"]

        history.append(dict(epoch=epoch, train_loss=train_loss, valid_loss=valid_loss,
                            valid_loss_raw=valid["raw"], seconds=elapsed,
                            proteins_per_second=seen / elapsed))
        print(f"epoch {epoch:4d}/{args.epochs}  train {train_loss:.4f}  "
              f"dev(raw) {valid['raw']:.4f}  dev(ema) {valid_loss:.4f}  "
              f"{elapsed:.1f} s  {seen/elapsed:.0f} prot/s", flush=True)
        torch.save(dict(model=model.state_dict(), ema=shadow.state_dict(),
                        optimiser=optimiser.state_dict(), epoch=epoch, history=history,
                        args=vars(args) | {"output": str(args.output)},
                        coord_std=coord_std, length_histogram=length_histogram),
                   checkpoint)
        json.dump(history, open(args.output / "history.json", "w"), indent=1)
    print("training finished")


if __name__ == "__main__":
    main()
