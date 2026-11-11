"""Score a generation file: pLDDT, ESM-2 pseudo-perplexity, periodicity, set stats.

Works for both experiments: the input is any JSON with a "cells" list whose
entries carry a "seqs" field. ESMFold is loaded, every cell folded, then
ESMFold is freed before ESM-2 is loaded, so peak memory stays near 10 GB.
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import metrics as M
from common.data import load_swissprot, natural_aa_frequency


def boot_ci(x, n=2000, alpha=0.05, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, x.size), replace=True).mean(1)
    return (float(np.quantile(bs, alpha / 2)),
            float(np.quantile(bs, 1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold-batch", type=int, default=8)
    ap.add_argument("--pppl-n", type=int, default=12,
                    help="sequences per cell for pseudo-perplexity")
    ap.add_argument("--no-fold", action="store_true")
    ap.add_argument("--no-pppl", action="store_true")
    ap.add_argument("--ref-n", type=int, default=120000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = json.load(open(args.gen))
    cells = gen["cells"]
    print(f"{len(cells)} cells, "
          f"{sum(len(c['seqs']) for c in cells)} sequences", flush=True)

    # ---- natural reference, from the same corpus
    tr, va = load_swissprot(max_n=args.ref_n)
    ref_freq = natural_aa_frequency(tr)
    K = 10
    ref_kmers = set()
    for s in tr:
        for i in range(len(s) - K + 1):
            ref_kmers.add(s[i:i + K])
    print(f"reference: {len(tr)} seqs, {len(ref_kmers)} {K}-mers", flush=True)

    # natural baseline row, length-matched to the generated cells
    lens = sorted({len(c["seqs"][0]) for c in cells if c["seqs"]})
    nat_rows = {}
    rng = np.random.default_rng(0)
    for L in lens:
        pool = [s for s in va if len(s) >= L]
        if len(pool) < 48:
            pool = [s for s in tr if len(s) >= L]
        pick = [pool[i][:L] for i in rng.choice(len(pool),
                                                min(48, len(pool)),
                                                replace=False)]
        nat_rows[L] = pick
    cells = cells + [{"L": L, "T": None, "selector": "natural",
                      "tau": None, "seqs": v} for L, v in nat_rows.items()]

    # ---------------------------------------------------------------- fold
    if not args.no_fold:
        t0 = time.time()
        fold = M.ESMFoldScorer(device=dev)
        print(f"esmfold loaded {time.time()-t0:.0f}s", flush=True)
        for ci, c in enumerate(cells):
            t1 = time.time()
            r = fold.fold(c["seqs"], batch_size=args.fold_batch)
            c["plddt_list"] = [float(v) for v in r["plddt"]]
            c["plddt_mean"] = float(np.mean(r["plddt"]))
            c["plddt_ci"] = boot_ci(r["plddt"])
            c["ptm_mean"] = float(np.nanmean(r["ptm"]))
            c["pae_mean"] = float(np.nanmean(r["pae"]))
            c["foldability"] = float(np.nanmean(r["foldable"]))
            c["ptm_list"] = [float(v) for v in r["ptm"]]
            bh = [M.buried_hydrophobic_fraction(s, cb)
                  for s, cb in zip(c["seqs"], r["cb"])]
            c["buried_hydrophobic"] = float(np.nanmean(bh))
            print(f"  fold [{ci+1}/{len(cells)}] {c['selector']} T={c['T']} "
                  f"plddt {c['plddt_mean']:.2f} "
                  f"CI{tuple(round(v,2) for v in c['plddt_ci'])} "
                  f"({time.time()-t1:.0f}s)", flush=True)
            json.dump({"cells": cells, "config": gen.get("config")},
                      open(args.out, "w"))
        del fold
        gc.collect(); torch.cuda.empty_cache()

    # --------------------------------------------------- pseudo-perplexity
    if not args.no_pppl:
        esm = M.ESM2Scorer(device=dev)
        print("esm2 650M loaded", flush=True)
        for ci, c in enumerate(cells):
            sub = c["seqs"][:args.pppl_n]
            vals = [esm.pseudo_perplexity(s) for s in sub]
            c["pppl_mean"] = float(np.mean(vals))
            c["pppl_ci"] = boot_ci(vals)
            print(f"  pppl [{ci+1}/{len(cells)}] {c['selector']} T={c['T']} "
                  f"{c['pppl_mean']:.3f}", flush=True)
            json.dump({"cells": cells, "config": gen.get("config")},
                      open(args.out, "w"))
        del esm
        gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------ cheap metrics
    for c in cells:
        s = c["seqs"]
        per = [M.periodicity_summary(x) for x in s]
        for k in ("helix_power", "strand_power", "amphipathic_index"):
            v = [p[k] for p in per]
            c[k] = float(np.nanmean(v))
            c[k + "_ci"] = boot_ci(v)
        for sc in ("kd", "cornette"):
            c[f"amphipathic_index_{sc}"] = float(np.nanmean(
                [M.amphipathic_index(x, scale=sc) for x in s]))
        c["hmoment"] = float(np.nanmean([M.hydrophobic_moment(x) for x in s]))
        ws = [M.amphipathic_window_stats(x) for x in s]
        for k in ("amph_window_frac", "amph_window_max"):
            v = [w[k] for w in ws]
            c[k] = float(np.nanmean(v))
            c[k + "_ci"] = boot_ci(v)
        # Composition-matched control. Shuffling a sequence keeps its amino
        # acid composition and destroys its hydrophobic patterning, so the
        # gap between a cell and its own shuffle is the part of the
        # periodicity that is patterning rather than composition. Natural
        # proteins sit at about 0.11 against 0.07 for their shuffles.
        import random as _r
        rg = _r.Random(0)
        sh = [''.join(rg.sample(x, len(x))) for x in s]
        vs = [M.amphipathic_window_stats(x)["amph_window_frac"] for x in sh]
        c["amph_window_frac_shuffled"] = float(np.nanmean(vs))
        c["amph_patterning"] = c["amph_window_frac"] - c[
            "amph_window_frac_shuffled"]
        c["identity"] = M.pairwise_identity(s)
        c["diversity"] = 1.0 - c["identity"] if np.isfinite(c["identity"]) \
            else float("nan")
        c["uniqueness"] = len(set(s)) / max(len(s), 1)
        c["novelty_kmer"] = M.kmer_novelty(s, ref_kmers, k=K)
        c["aa_jsd"] = M.aa_composition_jsd(s, ref_freq)
        c["entropy"] = M.shannon_entropy(s)
        c["max_repeat"] = float(np.mean([M.max_repeat_run(x) for x in s]))

    json.dump({"cells": cells, "config": gen.get("config")},
              open(args.out, "w"))
    print(f"WROTE {args.out}", flush=True)

    hdr = f"{'selector':18s} {'T':>5s} {'pLDDT':>7s} {'pppl':>7s} " \
          f"{'amph':>6s} {'div':>6s} {'nov':>6s} {'rep':>5s}"
    print(hdr); print("-" * len(hdr))
    for c in sorted(cells, key=lambda z: (str(z["selector"]),
                                          z["T"] or 0)):
        print(f"{str(c['selector']):18s} {str(c['T']):>5s} "
              f"{c.get('plddt_mean', float('nan')):7.2f} "
              f"{c.get('pppl_mean', float('nan')):7.2f} "
              f"{c['amphipathic_index']:6.3f} {c['diversity']:6.3f} "
              f"{c['novelty_kmer']:6.3f} {c['max_repeat']:5.1f}")


if __name__ == "__main__":
    main()
