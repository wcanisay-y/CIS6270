"""Train one arm of the Idea-B comparison. Matched size, data and budget.

Usage:
  python -m hier.train --arm flat                      --out ckpt/flat.pt
  python -m hier.train --arm hier --partition dms6      --out ckpt/dms6.pt
  python -m hier.train --arm hier --partition dms6 --random-partition
  python -m hier.train --arm hier --partition dms6 --soft-xi 0.8
  python -m hier.train --arm ar                        --out ckpt/ar.pt
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.alphabet import AA20, AA_IDX, PARTITIONS, family_table, \
    random_family_table
from common.data import load_swissprot
from hier.model import MDM, ARLM, ID_MASK, N_AA, count_params, loss_flat, \
    loss_hier


def encode(seqs, max_len):
    """-> int8 array (N, max_len) of aa indices, 255 for pad."""
    out = np.full((len(seqs), max_len), 255, dtype=np.uint8)
    for i, s in enumerate(seqs):
        a = np.frombuffer(s.encode(), dtype=np.uint8)
        idx = np.array([AA_IDX[c] for c in s], dtype=np.uint8)
        out[i, :len(s)] = idx
    return out


def batches(arr, bs, rng, device, max_len):
    n = arr.shape[0]
    while True:
        order = rng.permutation(n)
        for i in range(0, n - bs + 1, bs):
            sel = arr[order[i:i + bs]]
            x = torch.from_numpy(sel.astype(np.int64)).to(device)
            pad = x == 255
            x = x.masked_fill(pad, 0)
            yield x, pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["flat", "hier", "ar"], required=True)
    ap.add_argument("--partition", default="dms6")
    ap.add_argument("--random-partition", action="store_true")
    ap.add_argument("--partition-seed", type=int, default=0)
    ap.add_argument("--soft-xi", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--no-bind", action="store_true")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tag = args.tag or os.path.basename(args.out).replace(".pt", "")

    tr, va = load_swissprot(min_len=40, max_len=args.max_len - 2,
                            max_n=args.max_n)
    Xtr, Xva = encode(tr, args.max_len), encode(va, args.max_len)
    print(f"[{tag}] train {len(tr)} val {len(va)} tokens/epoch "
          f"{int((Xtr != 255).sum())}", flush=True)

    fam_lut = None
    n_fam = 0
    if args.arm == "hier":
        base = PARTITIONS[args.partition]
        if args.random_partition:
            names, lut = random_family_table(base, args.partition_seed)
        else:
            names, lut = family_table(base)
        n_fam = len(names)
        fam_lut = torch.as_tensor(lut, dtype=torch.long, device=dev)
        print(f"[{tag}] partition {args.partition} "
              f"{'RANDOMISED' if args.random_partition else 'chemical'} "
              f"k={n_fam} sizes={[int((lut == i).sum()) for i in range(n_fam)]}",
              flush=True)

    if args.arm == "ar":
        model = ARLM(args.d, args.layers, args.heads, args.max_len)
    else:
        model = MDM(args.d, args.layers, args.heads, args.max_len,
                    n_fam=n_fam, fam_lut=(lut if args.arm == "hier" else None))
    model = model.to(dev)
    print(f"[{tag}] params {count_params(model)/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.98))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(args.warmup, 1)
        p = (s - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    def step_loss(x, pad, gen=None):
        if args.arm == "flat":
            return loss_flat(model, x, pad, gen=gen)
        if args.arm == "hier":
            return loss_hier(model, x, fam_lut, n_fam, pad,
                             gamma=args.gamma, soft_xi=args.soft_xi,
                             bind=not args.no_bind, gen=gen)
        bos = torch.full((x.shape[0], 1), N_AA, dtype=torch.long, device=dev)
        inp = torch.cat([bos, x[:, :-1]], 1)
        logits = model(inp)
        ce = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
        keep = ~pad
        n = keep.sum().clamp_min(1)
        return (ce * keep).sum() / n, {"ce_id": float((ce * keep).sum().item()
                                                      / n.item())}

    @torch.no_grad()
    def evaluate(nb=20):
        model.eval()
        g = torch.Generator(device=dev).manual_seed(1234)
        tot, parts = 0.0, {}
        it = batches(Xva, min(args.batch, 128), np.random.default_rng(7), dev,
                     args.max_len)
        for _ in range(nb):
            x, pad = next(it)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(dev == "cuda")):
                l, d = step_loss(x, pad, gen=g)
            tot += float(l)
            for k, v in d.items():
                parts[k] = parts.get(k, 0.0) + v
        model.train()
        return tot / nb, {k: v / nb for k, v in parts.items()}

    it = batches(Xtr, args.batch, rng, dev, args.max_len)
    hist = []
    t0 = time.time()
    best = float("inf")
    for s in range(1, args.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(s)
        x, pad = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(dev == "cuda")):
            loss, d = step_loss(x, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 200 == 0:
            print(f"[{tag}] step {s}/{args.steps} loss {float(loss):.4f} "
                  f"{d} lr {lr_at(s):.2e} {time.time()-t0:.0f}s", flush=True)
        if s % args.eval_every == 0 or s == args.steps:
            vl, vd = evaluate()
            hist.append({"step": s, "train": float(loss), "val": vl, **vd})
            print(f"[{tag}] EVAL step {s} val {vl:.4f} {vd}", flush=True)
            if vl < best:
                best = vl
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "args": vars(args),
                "n_fam": n_fam,
                "fam_lut": (lut.tolist() if args.arm == "hier" else None),
                "params": count_params(model),
                "history": hist,
                "best_val": best,
                "seconds": time.time() - t0}, args.out)
    with open(args.out.replace(".pt", ".json"), "w") as f:
        json.dump({"tag": tag, "args": vars(args), "history": hist,
                   "best_val": best, "params": count_params(model),
                   "seconds": time.time() - t0}, f, indent=2)
    print(f"[{tag}] DONE best_val {best:.4f} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
