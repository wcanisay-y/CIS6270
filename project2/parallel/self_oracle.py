"""The strongest oracle: a contact map of a sequence this model actually makes.

The natural-protein oracle asks "does a realistic contact prior help?".
This asks the sharper question: take the sequence the confidence baseline
produces from a given seed, fold it, and hand its OWN true contact map back
to the selector for a second pass from the same seed. If the mechanism is
sound, knowing where this trajectory's contacts will be should help more
than knowing where some other protein's are.

It is still not a perfect oracle -- changing the order changes the sequence,
so the map belongs to the first pass's output rather than the second's --
but it is the closest thing available without circularity, and it bounds
what any contact predictor could deliver.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel.sample import load_dplm, generate_absorbing
from parallel.oracle import cb_contacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--steps", type=int, nargs="+", default=[12, 25, 50])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_dplm(device=dev, half=True, need_attn=False)
    L, N = args.length, args.n
    out = {"config": vars(args), "cells": []}

    for T in args.steps:
        # pass 1: the baseline, and the seeds we will reuse
        seqs, seeds = [], []
        for b0 in range(0, N, args.batch):
            B = min(args.batch, N - b0)
            sd = args.seed + 100000 + 1000 * T + b0
            s, _ = generate_absorbing(tok, model, L, B, T,
                                      selector="confidence", seed=sd,
                                      device=dev)
            seqs.extend(s)
            seeds.append((b0, B, sd))
        out["cells"].append(dict(L=L, T=T, selector="confidence",
                                 tau=None, seqs=list(seqs),
                                 nfe_gen=T, nfe_contact=0,
                                 note="self_oracle_pass1"))
        print(f"T={T} pass1 done", flush=True)

        # fold them to get their own contact maps
        from common import metrics as M
        fold = M.ESMFoldScorer(device=dev)
        r = fold.fold(seqs, batch_size=8)
        maps = torch.from_numpy(
            np.stack([cb_contacts(cb) for cb in r["cb"]])).float()
        del fold
        torch.cuda.empty_cache()
        print(f"T={T} folded, contact density {float(maps.mean()):.4f}, "
              f"pass1 pLDDT {float(np.mean(r['plddt'])):.2f}", flush=True)
        tok, model = load_dplm(device=dev, half=True, need_attn=False)

        # pass 2: same seeds, now selecting against those true maps
        for sel in ("contact_oracle", "contact_oracle_shuffled",
                    "anti_contact_oracle"):
            s2 = []
            for (b0, B, sd) in seeds:
                fc = maps[b0:b0 + B]
                s, _ = generate_absorbing(tok, model, L, B, T, selector=sel,
                                          tau=0.5, seed=sd, device=dev,
                                          fixed_coupling=fc)
                s2.extend(s)
            out["cells"].append(dict(L=L, T=T, selector=sel + "_self",
                                     tau=0.5, seqs=s2, nfe_gen=T,
                                     nfe_contact=0, note="self_oracle"))
            print(f"T={T} {sel} done | {s2[0][:40]}", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"))
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
