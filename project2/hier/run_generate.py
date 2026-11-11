"""Generate from trained Idea-B checkpoints: unconditional, temperature grid,
and family-pattern-conditioned.

NFE accounting: the flat arm spends `steps` forward passes; the two-level arm
spends the SAME `steps`, split rho on families and 1-rho on identities, so the
comparison is matched-NFE. The AR arm necessarily spends L passes and is
reported with that number rather than pretended to be matched.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.alphabet import AA20, AA_IDX
from common.data import load_swissprot
from hier.model import MDM, ARLM
from hier.sample import sample_flat, sample_hier, sample_ar, \
    sample_hier_conditioned, family_compliance, sample_hier_scheduled


def load_arm(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    if a["arm"] == "ar":
        m = ARLM(a["d"], a["layers"], a["heads"], a["max_len"])
    else:
        m = MDM(a["d"], a["layers"], a["heads"], a["max_len"],
                n_fam=ck["n_fam"],
                fam_lut=(np.array(ck["fam_lut"]) if ck["n_fam"] else None))
    m.load_state_dict(ck["model"])
    return m.eval().to(device), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0])
    ap.add_argument("--fam-temps", type=float, nargs="+", default=[1.0])
    ap.add_argument("--rho-sweep", type=float, nargs="*", default=[])
    ap.add_argument("--conditioned", action="store_true")
    ap.add_argument("--pattern-file", default=None,
                    help="json with a `seqs` list to take family patterns "
                         "from, instead of the first --n dev sequences. Used "
                         "for the homology-clean set, since the upstream dev "
                         "split is not homology-clustered and the chemical "
                         "arm gains specifically on contaminated sources.")
    ap.add_argument("--rank", default="gumbel", choices=["gumbel", "clean"],
                    help="commit ranking; clean collapses the sampler and is "
                         "kept only for the ablation")
    ap.add_argument("--hier-refresh", action="store_true",
                    help="extra forward pass per step after families are "
                         "written; doubles NFE")
    ap.add_argument("--hier-sampler", default="scheduled",
                    choices=["scheduled", "twophase"],
                    help="scheduled follows the training corruption schedule; "
                         "twophase reveals all families then all identities, "
                         "which is out of distribution")
    ap.add_argument("--nfe-sweep", type=int, nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"config": vars(args), "cells": []}

    nat = None
    if args.conditioned:
        if args.pattern_file:
            with open(args.pattern_file) as fh:
                pf = json.load(fh)
            pool = [s for s in pf["seqs"] if len(s) >= args.length][:args.n]
            nat = [s[:args.length] for s in pool]
            out["pattern_file"] = args.pattern_file
            out["pattern_containment"] = pf.get("containment")
            print("patterns: %d from %s (max containment %.4f)"
                  % (len(nat), args.pattern_file,
                     max(pf.get("containment") or [float("nan")])), flush=True)
        else:
            tr, va = load_swissprot(max_n=20000)
            pool = [s for s in va if len(s) >= args.length][:args.n]
            nat = [s[:args.length] for s in pool]

    for path in args.ckpt:
        m, ck = load_arm(path, dev)
        a = ck["args"]
        arm, tag = a["arm"], os.path.basename(path).replace(".pt", "")
        lut = np.array(ck["fam_lut"]) if ck["n_fam"] else None
        print(f"=== {tag} ({arm}) params {ck['params']/1e6:.2f}M "
              f"best_val {ck['best_val']:.4f}", flush=True)

        jobs = []
        if arm == "ar":
            for t in args.temps:
                jobs.append(dict(kind="uncond", temp=t, nfe=args.length))
        elif arm == "flat":
            for t in args.temps:
                jobs.append(dict(kind="uncond", temp=t, steps=args.steps,
                                 nfe=args.steps))
            for T in args.nfe_sweep:
                jobs.append(dict(kind="nfe", temp=1.0, steps=T, nfe=T))
        else:
            for tf in args.fam_temps:
                for t in args.temps:
                    jobs.append(dict(kind="uncond", temp=t, fam_temp=tf,
                                     steps=args.steps, rho=args.rho,
                                     nfe=args.steps))
            for T in args.nfe_sweep:
                jobs.append(dict(kind="nfe", temp=1.0, fam_temp=1.0,
                                 steps=T, rho=args.rho, nfe=T))
            for r in args.rho_sweep:
                jobs.append(dict(kind="rho", temp=1.0, fam_temp=1.0,
                                 steps=args.steps, rho=r, nfe=args.steps))
            if args.conditioned and nat:
                jobs.append(dict(kind="conditioned", temp=1.0,
                                 steps=args.steps, nfe=args.steps))

        for ji, j in enumerate(jobs):
            seqs, fams = [], []
            t0 = time.time()
            for b0 in range(0, args.n, args.batch):
                B = min(args.batch, args.n - b0)
                seed = 1000 * ji + b0
                if j["kind"] == "conditioned":
                    pat = np.stack([lut[[AA_IDX[c] for c in s]]
                                    for s in nat[b0:b0 + B]])
                    s = sample_hier_conditioned(
                        m, pat, j["steps"], t_id=j["temp"], device=dev,
                        seed=seed, rank=args.rank)
                    fams.append(pat)
                elif arm == "ar":
                    s = sample_ar(m, B, args.length, temperature=j["temp"],
                                  device=dev, seed=seed)
                elif arm == "flat":
                    s = sample_flat(m, B, args.length, j["steps"],
                                    temperature=j["temp"], device=dev,
                                    seed=seed, rank=args.rank)
                elif args.hier_sampler == "scheduled":
                    s, fp, nfe_used = sample_hier_scheduled(
                        m, B, args.length, j["steps"],
                        t_fam=j["fam_temp"], t_id=j["temp"],
                        gamma=a.get("gamma", 2.0), device=dev, seed=seed,
                        refresh=args.hier_refresh, rank=args.rank)
                    fams.append(fp)
                    j["nfe"] = nfe_used
                else:
                    s, fp = sample_hier(m, B, args.length, j["steps"],
                                        t_fam=j["fam_temp"], t_id=j["temp"],
                                        rho=j["rho"], device=dev, seed=seed)
                    fams.append(fp)
                seqs.extend(s)
            rec = dict(arm=arm, tag=tag, selector=f"{tag}|{j['kind']}",
                       L=args.length, T=j.get("steps"), nfe=j["nfe"],
                       temp=j["temp"], fam_temp=j.get("fam_temp"),
                       rho=j.get("rho"), kind=j["kind"], seqs=seqs,
                       hier_sampler=args.hier_sampler, rank=args.rank,
                       seconds=time.time() - t0,
                       best_val=ck["best_val"], params=ck["params"])
            if j["kind"] == "conditioned" and fams:
                pat = np.concatenate(fams)[:len(seqs)]
                rec["family_compliance"] = family_compliance(seqs, pat, lut)
            out["cells"].append(rec)
            print(f"  {j['kind']:12s} T={j.get('steps')} temp={j['temp']} "
                  f"famT={j.get('fam_temp')} rho={j.get('rho')} "
                  f"nfe={j['nfe']} {time.time()-t0:.0f}s "
                  f"compl={rec.get('family_compliance')} | {seqs[0][:40]}",
                  flush=True)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            json.dump(out, open(args.out, "w"))
        del m
        torch.cuda.empty_cache()
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
