"""Training loop for protein flow matching.

    python train.py --model flow --epochs 100 --out ckpt/flow.pt
    python train.py --model classifier --clean-only --out ckpt/classifier.pt

Model selection is on validation loss.
"""
import argparse
import time

import torch

from flow_matching import FlowModel, FamilyClassifier
from data.pfam import PfamData


def build(name, args):
    if name == "flow":
        return FlowModel(args.layers, args.n_heads, args.cond_drop)
    if name == "classifier":
        return FamilyClassifier()
    raise ValueError(name)


def run_epoch(model, loader, device, opt=None, clean_only=False):
    total, nb = 0.0, 0
    for z, family in loader:
        z, family = z.to(device), family.to(device)
        if isinstance(model, FamilyClassifier):
            loss = model.loss(z, family, clean_only)
        else:
            loss = model.loss(z, family)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        total, nb = total + loss.item(), nb + 1
    return total / max(nb, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["flow", "classifier"], required=True)
    p.add_argument("--data", default="data/pfam_esm2.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--cond-drop", type=float, default=0.1)
    p.add_argument("--clean-only", action="store_true", help="classifier at t=1 only")
    p.add_argument("--limit", type=int, default=None, help="subset for debugging")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PfamData(a.data, limit=a.limit, seed=a.seed)
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
            torch.save({
                "model": a.model,
                "state": model.state_dict(),
                "args": vars(a),
                "epoch": ep,
                "val": vl,
                "z_mean": data.z_mean,
                "z_std": data.z_std,
                "family_hist": data.family_hist,
                "family_names": data.family_names,
                "n_families": data.n_families,
                "params": n_par
            }, a.out)
            flag = " *"
        print(f"epoch {ep:4d}  train {tl:.4f}  val {vl:.4f}{flag}")
    print(f"best val {best:.4f}, {time.time()-t0:.0f}s, saved to {a.out}")


if __name__ == "__main__":
    main()
