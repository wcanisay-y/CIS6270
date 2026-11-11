"""Generate the Idea-A grid: every selector at every NFE budget.

All absorbing-family cells with the same T use exactly T generator forward
passes, so pLDDT differences between selectors at fixed T are differences in
WHICH positions were committed together, not in how much compute was spent.
Contact refreshes are counted separately and reported as extra FLOPs.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel.sample import load_dplm, generate_absorbing, generate_reparam
from parallel.contacts import ExternalContacts, SelfContacts, \
    RawAttentionContacts

SELECTORS = ["confidence", "random", "entropy", "contact", "contact_shuffled",
             "anti_contact", "contact_degree", "distance", "punt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="airkingbd/dplm_150m")
    ap.add_argument("--lengths", type=int, nargs="+", default=[200])
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[6, 12, 25, 50, 100, 200])
    ap.add_argument("--selectors", nargs="+", default=SELECTORS)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--tau-sweep", type=float, nargs="*", default=[])
    ap.add_argument("--contact-mode", default="self",
                    choices=["self", "external", "rawattn",
                             "external_argmax"])
    ap.add_argument("--esm-token-dropout", action="store_true",
                    help="keep ESM-2's token_dropout rescaling on the "
                         "contact call (ablation; off by default)")
    ap.add_argument("--contact-size", type=int, default=650)
    ap.add_argument("--contact-every", type=int, default=1)
    ap.add_argument("--dist-d", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--oracle-file", default=None,
                    help="npz of true contact maps; enables the "
                         "contact_oracle selectors")
    ap.add_argument("--no-resample", action="store_true",
                    help="disable DPLM's anti-collapse repeat resampling "
                         "in the reparam reference arm")
    ap.add_argument("--reparam", action="store_true",
                    help="also run DPLM's published decoder at each T")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="reparam arms: consult contacts only once this "
                         "fraction of the sequence is kept, and make no "
                         "contact call before that")
    ap.add_argument("--tau-mode", default="absolute",
                    choices=["absolute", "quantile"])
    ap.add_argument("--tau-q", type=float, default=0.99,
                    help="with --tau-mode quantile: reject pairs above this "
                         "quantile of the sequence's own coupling values")
    ap.add_argument("--contact-min-revealed", type=float, default=0.0,
                    help="fall back to confidence selection until this "
                         "fraction of the sequence is written")
    ap.add_argument("--score-mode", default="gumbel",
                    choices=["gumbel", "clean"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Any mode that reads the generator's own attention maps needs eager
    # attention: from transformers 4.56 the SDPA path returns None for
    # attentions and every downstream call fails on it.
    need_attn = args.contact_mode in ("self", "rawattn") or any(
        s.startswith("reparam_") and s != "reparam_confidence"
        for s in args.selectors)
    tok, model = load_dplm(args.model, device=dev, half=True,
                           need_attn=need_attn)
    print(f"loaded {args.model} on {dev}", flush=True)

    oracle = None
    if args.oracle_file:
        z = np.load(args.oracle_file)
        oracle = torch.from_numpy(z["maps"]).float()
        print(f"oracle maps {tuple(oracle.shape)} density "
              f"{float(oracle.mean()):.4f}", flush=True)

    needs_predicted = any(
        (s.startswith(("contact", "anti_contact"))
         or s.startswith("reparam_") and s != "reparam_confidence")
        and "oracle" not in s for s in args.selectors)
    provider = None
    if needs_predicted:
        if args.contact_mode == "self":
            provider = SelfContacts(150, device=dev)
        elif args.contact_mode == "rawattn":
            provider = RawAttentionContacts(device=dev)
        else:
            provider = ExternalContacts(
                args.contact_size, device=dev,
                token_dropout=args.esm_token_dropout)
        print(f"contact provider: {args.contact_mode}", flush=True)

    cells = []
    for L in args.lengths:
        for T in args.steps:
            for sel in args.selectors:
                cells.append(dict(L=L, T=T, selector=sel, tau=args.tau))
            if args.reparam:
                cells.append(dict(L=L, T=T, selector="dplm_reparam",
                                  tau=None))
        for tau in args.tau_sweep:
            cells.append(dict(L=L, T=25, selector="contact", tau=tau,
                              note="tau_sweep"))

    out = {"config": vars(args), "cells": []}
    t_start = time.time()
    for ci, c in enumerate(cells):
        L, T, sel = c["L"], c["T"], c["selector"]
        seqs, diags = [], []
        for b0 in range(0, args.n, args.batch):
            B = min(args.batch, args.n - b0)
            seed = args.seed + 1000 * ci + b0
            fc = None
            if "oracle" in sel:
                if oracle is None:
                    raise SystemExit("oracle selector needs --oracle-file")
                idx = [(b0 + k) % oracle.shape[0] for k in range(B)]
                fc = oracle[idx]
            if sel == "dplm_reparam" or sel.startswith("reparam_"):
                rsel = ("confidence" if sel == "dplm_reparam"
                        else sel[len("reparam_"):])
                s, d = generate_reparam(tok, model, L, B, T,
                                        temperature=args.temperature,
                                        seed=seed, device=dev,
                                        resample=not args.no_resample,
                                        selector=rsel,
                                        contact_provider=provider,
                                        contact_mode=args.contact_mode,
                                        contact_every=args.contact_every,
                                        tau=c["tau"] or args.tau,
                                        tau_mode=args.tau_mode,
                                        tau_q=args.tau_q,
                                        warmup=args.warmup)
            else:
                s, d = generate_absorbing(
                    tok, model, L, B, T, selector=sel,
                    temperature=args.temperature, tau=c["tau"] or args.tau,
                    dist_d=args.dist_d, contact_provider=provider,
                    contact_mode=args.contact_mode,
                    contact_every=args.contact_every, seed=seed, device=dev,
                    fixed_coupling=fc, score_mode=args.score_mode,
                    contact_min_revealed=args.contact_min_revealed,
                    tau_mode=args.tau_mode, tau_q=args.tau_q)
            seqs.extend(s)
            diags.append(d)
        rec = dict(c)
        rec["seqs"] = seqs
        rec["nfe_gen"] = int(np.mean([d["nfe_gen"] for d in diags]))
        rec["nfe_contact"] = int(np.mean([d["nfe_contact"] for d in diags]))
        rec["seconds"] = float(np.sum([d["seconds"] for d in diags]))
        wc = [v for d in diags for v in d.get("within_coupling", [])]
        cpf = [v for d in diags for v in d.get("coupled_pair_frac", [])]
        rec["within_coupling"] = float(np.mean(wc)) if wc else None
        rec["coupled_pair_frac"] = float(np.mean(cpf)) if cpf else None
        rec["gated_steps"] = int(np.mean([d.get("gated_steps", 0)
                                          for d in diags]))
        rec["contact_min_revealed"] = args.contact_min_revealed
        rec["tau_mode"] = args.tau_mode
        rec["tau_q"] = args.tau_q
        tu = [v for d in diags for v in d.get("tau_used", [])]
        rec["tau_used"] = float(np.mean(tu)) if tu else None
        out["cells"].append(rec)
        print(f"[{ci+1}/{len(cells)}] L={L} T={T} {sel} tau={c['tau']} "
              f"nfe={rec['nfe_gen']}+{rec['nfe_contact']} "
              f"{rec['seconds']:.1f}s coupled_pairs={rec['coupled_pair_frac']} "
              f"| {seqs[0][:40]}", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f)
    print(f"ALL DONE {time.time()-t_start:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
