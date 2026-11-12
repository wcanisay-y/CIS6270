"""Score the stored conditioning sequences on metrics that are not ESMFold.

No generation and no folding happen here. Every conditioning run already wrote
its sequences into `results/*_scored.json`, so this re-scores exactly the
sequences the published pLDDT numbers were computed on -- a different question
asked of the same objects, not a new experiment.

For each cell:

    NSR, edit distance   against the pattern's own source protein, paired per
                         source, with the compositional chance floor beside
                         them. The conditioned arms SHOULD sit near the floor:
                         if they do not, the pattern is a retrieval key.
    FPD, MMD             between that cell's ESM-2 embedding distribution and
                         two references -- the 128 pattern sources, and the
                         natural cell the run already scored.

The source reconstruction below repeats hier/run_generate.py exactly
(`[s[:length] for s in patterns["seqs"] if len(s) >= length][:n]`), so index i
of a cell's sequences and index i of `nat` are the same site-for-site pair the
paired pLDDT contrasts use.

Usage:
    python -m tools.extra_metrics_run \
        --scored results/K_cath_purged_scored.json \
        --patterns data/clustered_cath/heldout_patterns.json \
        --out results/L_extra_K_cath_purged.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.alphabet import (AA_IDX, PARTITIONS, family_table,
                             random_family_table)
from common.extra_metrics import (ESM2Embedder, boot_ci, composition_floor,
                                  frechet, levenshtein, mmd_rbf, nsr)


def arm_lut(selector, ckpt_json):
    """The family partition an arm was trained with, from its checkpoint
    sidecar -- `dms6` is the chemical table, `rand6` the seeded random one."""
    if not os.path.exists(ckpt_json):
        return None
    a = json.load(open(ckpt_json))["args"]
    base = PARTITIONS[a["partition"]]
    if a.get("random_partition"):
        return random_family_table(base, a["partition_seed"])[1]
    return family_table(base)[1]


def family_floors(srcs, lut):
    """The two NSR floors a CONDITIONED arm has to beat to mean anything.

    uniform: the arm is told the family and picks uniformly inside it.
    modal:   it always writes that family's most common residue in the
             sources. The modal floor is the higher of the two and is the one
             that matters -- an arm below it is not even exploiting the
             partition's marginal.
    """
    if lut is None:
        return None, None
    lut = np.asarray(lut)
    p = np.zeros(20)
    for s in srcs:
        for ch in s:
            if ch in AA_IDX and AA_IDX[ch] < 20:
                p[AA_IDX[ch]] += 1
    p = p / max(p.sum(), 1)
    unif = float(np.mean([np.mean([1.0 / (lut == lut[AA_IDX[ch]]).sum()
                                   for ch in s if ch in AA_IDX
                                   and AA_IDX[ch] < 20]) for s in srcs]))
    modal = {}
    for fm in range(int(lut.max()) + 1):
        mem = np.where(lut == fm)[0]
        modal[fm] = mem[int(np.argmax(p[mem]))]
    mod = float(np.mean([np.mean([AA_IDX[ch] == modal[lut[AA_IDX[ch]]]
                                  for ch in s if ch in AA_IDX
                                  and AA_IDX[ch] < 20]) for s in srcs]))
    return unif, mod


def sources(pattern_file, length, n):
    """Exactly hier/run_generate.py's pattern pool, so indices line up."""
    with open(pattern_file) as fh:
        pf = json.load(fh)
    pool = [s for s in pf["seqs"] if len(s) >= length][:n]
    return [s[:length] for s in pool]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None,
                    help="directory holding <tag>.json sidecars, used for the "
                         "family-aware NSR floors")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip FPD/MMD, which are the only part needing a GPU")
    args = ap.parse_args()

    d = json.load(open(args.scored))
    cfg = d["config"]
    L, N = cfg["length"], cfg["n"]
    nat = sources(args.patterns, L, N)
    print(f"{os.path.basename(args.scored)}: length {L}, n {N}, "
          f"{len(nat)} pattern sources", flush=True)

    cells = [c for c in d["cells"] if c.get("seqs")]
    out = {"scored": args.scored, "patterns": args.patterns,
           "length": L, "n": N, "n_sources": len(nat), "cells": []}

    # ---- sequence-level, CPU ------------------------------------------
    for c in cells:
        seqs = c["seqs"]
        k = min(len(seqs), len(nat))
        rec = {"selector": c["selector"], "n": k,
               "plddt_mean": c.get("plddt_mean"),
               "pppl_mean": c.get("pppl_mean"),
               "pae_mean": c.get("pae_mean"),
               "ptm_mean": c.get("ptm_mean"),
               "foldability": c.get("foldability")}
        if k and c["selector"] != "natural":
            v = [nsr(seqs[i], nat[i]) for i in range(k)]
            e = [levenshtein(seqs[i], nat[i]) for i in range(k)]
            rec["nsr"], rec["nsr_lo"], rec["nsr_hi"] = boot_ci(v)
            rec["edit"], rec["edit_lo"], rec["edit_hi"] = boot_ci(e)
            rec["nsr_floor"] = composition_floor(nat[:k], seqs[:k])
            rec["edit_norm"] = rec["edit"] / L
            if args.ckpt_dir and c.get("tag"):
                u, m = family_floors(
                    nat[:k],
                    arm_lut(c["selector"],
                            os.path.join(args.ckpt_dir, c["tag"] + ".json")))
                rec["nsr_floor_uniform_in_family"] = u
                rec["nsr_floor_modal_in_family"] = m
        print("  %-22s NSR %s  edit %s" % (
            rec["selector"],
            "n/a   " if "nsr" not in rec else f"{rec['nsr']:.3f}",
            "n/a" if "edit" not in rec else f"{rec['edit']:.1f}"), flush=True)
        out["cells"].append(rec)

    # ---- distributional, needs the GPU --------------------------------
    if not args.no_embed:
        emb = ESM2Embedder(device=args.device)
        E = {c["selector"]: emb(c["seqs"]) for c in cells}
        E["__sources__"] = emb(nat)
        refs = [("sources", "__sources__")]
        if "natural" in E:
            refs.append(("natural", "natural"))
        for rec in out["cells"]:
            X = E[rec["selector"]]
            for name, key in refs:
                if key == rec["selector"]:
                    continue
                rec[f"fpd_vs_{name}"] = frechet(X, E[key])
                rec[f"mmd_vs_{name}"] = mmd_rbf(X, E[key])
            print("  %-22s FPD(src) %8.2f  MMD(src) %7.4f" % (
                rec["selector"], rec.get("fpd_vs_sources", float("nan")),
                rec.get("mmd_vs_sources", float("nan"))), flush=True)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
