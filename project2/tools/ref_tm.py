"""Ref-TM and Ref-RMSD: does a conditioned sequence adopt its SOURCE's fold?

This is the metric the conditioning result most needs and has never had. Every
number behind "+19.9 pLDDT" is ESMFold's confidence in its own prediction.
Confidence is not correctness, and a model can be confident about a fold that
has nothing to do with the protein whose chemical pattern it was handed.

Ref-TM answers that directly, because the CATH arm's pattern sources are
structural domains with deposited coordinates:

    fold the generated sequence with ESMFold
    structurally align it to the SOURCE domain's crystal structure
    report TM-score normalised by the generated chain, and RMSD

TM >= 0.5 is the conventional same-fold threshold; ~0.17 is what unrelated
structures of this size score. The controls make the number readable:

    dms6|conditioned   the claim
    rand6|conditioned  the bit-matched random partition -- if THIS reaches the
                       same Ref-TM, the chemistry is not what is working
    *|uncond           the null: same model, no pattern at all
    source             the source's own slice folded and compared with its own
                       crystal structure, i.e. the ceiling ESMFold can reach
                       at this length, which bounds everything above it

Reads structures downloaded by tools/fetch_cath_domains.py. Run this module by
path with -I so the download directory is never on the import path.

Usage:
    ./venv2/bin/python -I tools/ref_tm.py \
        --scored results/K_cath_purged_scored.json \
        --patterns data/clustered_cath/heldout_patterns.json \
        --domains data/cath_domains --out results/M_reftm_K_purged.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

THREE2ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y", "MSE": "M",
}


def read_pdb_ca(path):
    """-> (n,3) CA coordinates and the one-letter sequence beside them.

    A deliberately small parser: ATOM/HETATM lines, CA atoms, first altloc
    only, one entry per (chain, resseq, icode). Downloaded files are data, so
    nothing here imports or executes anything from them.
    """
    seen, xyz, seq = set(), [], []
    for line in open(path, "r", errors="replace"):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        res = line[17:20].strip().upper()
        if res not in THREE2ONE:
            continue
        key = (line[21], line[22:27])
        if key in seen:
            continue
        seen.add(key)
        try:
            xyz.append([float(line[30:38]), float(line[38:46]),
                        float(line[46:54])])
        except ValueError:
            continue
        seq.append(THREE2ONE[res])
    return np.asarray(xyz, float), "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--domains", default="data/cath_domains")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cells", default="dms6|conditioned,rand6|conditioned,"
                                       "dms6|uncond,rand6|uncond")
    ap.add_argument("--skip-source-control", action="store_true")
    args = ap.parse_args()

    from tmtools import tm_align
    from common.metrics import ESMFoldScorer

    d = json.load(open(args.scored))
    L, N = d["config"]["length"], d["config"]["n"]
    with open(args.patterns) as fh:
        pf = json.load(fh)
    pool = [s for s in pf["seqs"] if len(s) >= L][:N]
    nat = [s[:L] for s in pool]

    idx = json.load(open(os.path.join(args.domains, "index.json")))
    doms = idx["domains"]
    refs = []
    for dom in doms[:len(nat)]:
        p = os.path.join(args.domains, str(dom) + ".pdb")
        if dom and os.path.exists(p):
            ca, sq = read_pdb_ca(p)
            refs.append((dom, ca, sq) if len(ca) >= 30 else None)
        else:
            refs.append(None)
    have = [i for i, r in enumerate(refs) if r is not None]
    print(f"{len(have)} of {len(nat)} sources have a usable structure",
          flush=True)

    scorer = ESMFoldScorer()
    want = [c.strip() for c in args.cells.split(",") if c.strip()]
    cells = [c for c in d["cells"] if c["selector"] in want and c.get("seqs")]
    if not args.skip_source_control:
        cells.append({"selector": "source", "seqs": nat})

    out = {"scored": args.scored, "domains": args.domains,
           "n_with_structure": len(have), "cells": []}
    for c in cells:
        seqs = c["seqs"]
        tms, rms, lens = [], [], []
        for b0 in range(0, min(len(seqs), len(refs)), args.batch):
            chunk = [seqs[i] for i in range(b0, min(b0 + args.batch,
                                                    len(seqs), len(refs)))]
            f = scorer.fold(chunk, batch_size=len(chunk))
            for k, ca in enumerate(f["ca"]):
                i = b0 + k
                if refs[i] is None:
                    continue
                _, rca, rseq = refs[i]
                try:
                    r = tm_align(np.asarray(ca, float), rca,
                                 seqs[i][:len(ca)], rseq)
                except Exception as e:                        # noqa: BLE001
                    print(f"  tm_align failed at {i}: {e}", flush=True)
                    continue
                tms.append(float(r.tm_norm_chain1))
                rms.append(float(r.rmsd))
                lens.append(len(rca))
        tms, rms = np.asarray(tms), np.asarray(rms)
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(tms, len(tms), True).mean()
                       for _ in range(20000)]) if len(tms) else np.array([0.])
        rec = {"selector": c["selector"], "n": int(len(tms)),
               "ref_tm": float(tms.mean()) if len(tms) else float("nan"),
               "ref_tm_lo": float(np.percentile(bs, 2.5)),
               "ref_tm_hi": float(np.percentile(bs, 97.5)),
               "ref_tm_frac_ge_0.5": float((tms >= 0.5).mean())
               if len(tms) else float("nan"),
               "ref_tm_max": float(tms.max()) if len(tms) else float("nan"),
               "ref_rmsd": float(rms.mean()) if len(rms) else float("nan"),
               "ref_len_mean": float(np.mean(lens)) if lens else float("nan")}
        out["cells"].append(rec)
        print("  %-22s Ref-TM %.3f [%.3f, %.3f]  >=0.5 %.1f %%  RMSD %.1f A"
              % (rec["selector"], rec["ref_tm"], rec["ref_tm_lo"],
                 rec["ref_tm_hi"], 100 * rec["ref_tm_frac_ge_0.5"],
                 rec["ref_rmsd"]), flush=True)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
