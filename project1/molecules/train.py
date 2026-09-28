"""Training loop. One file, three networks, selected by --model.

    python train.py --model flow --epochs 100 --out ckpt/flow.pt
    python train.py --model diff --epochs 100 --out ckpt/diff.pt
    python train.py --model prop --clean-only --out ckpt/propeval.pt  # target metric

Model selection is on validation loss, which is a weak proxy for sample quality;
for the flow and diffusion models the paper should report both this and a
periodic molecule-stability check.
"""
import argparse
import time

import torch

from diffusion import DiffusionModel
from flow_matching import FlowModel, PropertyNet
from data.qm9 import QM9Data


def build(name, args):
    if name == "flow":
        return FlowModel(args.nf, args.layers, args.cond_drop, args.coupling)
    if name == "diff":
        return DiffusionModel(args.nf, args.layers, args.cond_drop, args.steps)
    if name == "prop":
        return PropertyNet(args.nf, max(args.layers - 2, 2))
    raise ValueError(name)


def run_epoch(model, loader, device, opt=None, clean_only=False):
    total, nb = 0.0, 0
    for x, h, mask, cond in loader:
        x, h, mask, cond = (v.to(device) for v in (x, h, mask, cond))
        if isinstance(model, PropertyNet):
            loss = model.loss(x, h, mask, cond, clean_only)
        else:
            loss = model.loss(x, h, mask, cond)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        total, nb = total + loss.item(), nb + 1
    return total / max(nb, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["flow", "diff", "prop"], required=True)
    p.add_argument("--data", default="data/qm9.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--nf", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--cond-drop", type=float, default=0.1)
    p.add_argument("--coupling", choices=["independent", "ot"], default="independent",
                   help="noise coupling for flow matching: independent (default) or ot (minibatch OT)")
    p.add_argument("--steps", type=int, default=1000, help="diffusion K")
    p.add_argument("--clean-only", action="store_true", help="prop net at t=1 only")
    p.add_argument("--limit", type=int, default=None, help="subset for debugging")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = QM9Data(a.data, limit=a.limit, seed=a.seed)
    tr = data.loader("train", a.batch_size)
    va = data.loader("val", a.batch_size, shuffle=False)

    model = build(a.model, a).to(device)
    n_par = sum(q.numel() for q in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"{a.model}: {n_par/1e6:.2f}M parameters on {device}")

    best, t0 = float("inf"), time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tl = run_epoch(model, tr, device, opt, a.clean_only)
        model.eval()
        with torch.no_grad():
            vl = run_epoch(model, va, device, None, a.clean_only)
        flag = ""
        if vl < best:
            best = vl
            torch.save({"model": a.model, "state": model.state_dict(),
                        "args": vars(a), "epoch": ep, "val": vl,
                        "prop_mean": data.prop_mean, "prop_std": data.prop_std,
                        "n_hist": data.n_hist, "params": n_par}, a.out)
            flag = " *"
        print(f"epoch {ep:4d}  train {tl:.4f}  val {vl:.4f}{flag}")
    print(f"best val {best:.4f}, {time.time()-t0:.0f}s, saved to {a.out}")


if __name__ == "__main__":
    main()