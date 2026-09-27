#!/usr/bin/env python3
"""ARCH 4. Protein metrics for generated Ca backbones, replacing the QM9 chemistry evaluator.

Bond-order tables, valence stability and SMILES are meaningless for a Ca trace, so nothing from
the QM9 scorer is reused. Every metric here is computed identically for generated samples and for
the AlphaFold reference, so the reference row is the ceiling.

    python evaluate_protein.py --samples <run>/samples_10000.npz --out <run>/metrics.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

# bulky input data, kept outside the repository; override the root with PROTEIN_DIFFUSION_ROOT
CACHE = Path(os.environ.get("PROTEIN_DIFFUSION_ROOT",
                            "/mnt/isilon/tan_lab/pany3/protein_diffusion")) / "cache"
AA = list("ACDEFGHIKLMNPQRSTVWY")
BOND_LO, BOND_HI = 3.0, 4.6      # a generous window around the 3.80-3.85 A observed in data
CLASH = 3.0                      # two non-adjacent Ca atoms closer than this cannot coexist
HELIX_I3 = (4.6, 6.4)            # Ca(i)-Ca(i+3) in an alpha helix
HELIX_I4 = (5.4, 7.0)            # Ca(i)-Ca(i+4) in an alpha helix


def per_structure(coords, length):
    c = coords[:length]
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    distance = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    index = np.arange(length)
    non_adjacent = np.abs(index[:, None] - index[None, :]) > 1
    clashes = int((distance[non_adjacent] < CLASH).sum() // 2)
    # Handedness: the normalised triple product over four consecutive atoms. Real proteins are
    # strongly biased because their helices are right-handed.
    chirality = np.nan
    if length >= 4:
        u, v, w = c[1:-2] - c[:-3], c[2:-1] - c[1:-2], c[3:] - c[2:-1]
        triple = np.einsum("ij,ij->i", np.cross(v, w), u)
        norm = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1)
        good = norm > 1e-6
        if good.any():
            chirality = float((triple[good] / norm[good] > 0).mean())
    helix = np.nan
    if length >= 5:
        d3 = np.linalg.norm(c[3:] - c[:-3], axis=1)
        d4 = np.linalg.norm(c[4:] - c[:-4], axis=1)
        n = min(len(d3), len(d4))
        helix = float((((d3[:n] > HELIX_I3[0]) & (d3[:n] < HELIX_I3[1]))
                       & ((d4[:n] > HELIX_I4[0]) & (d4[:n] < HELIX_I4[1]))).mean())
    return dict(
        bond_mean=float(step.mean()), bond_sd=float(step.std()),
        bond_ok=float(((step > BOND_LO) & (step < BOND_HI)).mean()),
        clashes=clashes,
        valid=bool(((step > BOND_LO) & (step < BOND_HI)).all() and clashes == 0),
        radius=float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())),
        chirality=chirality, helix=helix, length=int(length))


def score(coords, types, lengths, name):
    rows = [per_structure(coords[i], int(lengths[i])) for i in range(len(lengths))]
    composition = np.zeros(len(AA))
    for i in range(len(lengths)):
        t = types[i][:int(lengths[i])]
        composition += np.bincount(t[t >= 0], minlength=len(AA))
    composition = composition / max(composition.sum(), 1)
    def mean(key):
        values = np.array([r[key] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else float("nan")
    out = dict(
        name=name, n=len(rows),
        validity=float(np.mean([r["valid"] for r in rows])),
        bond_mean=mean("bond_mean"), bond_sd=mean("bond_sd"), bond_ok=mean("bond_ok"),
        clash_free=float(np.mean([r["clashes"] == 0 for r in rows])),
        clashes_per_structure=mean("clashes"),
        radius_mean=mean("radius"), right_handed=mean("chirality"), helix_fraction=mean("helix"),
        composition={a: float(p) for a, p in zip(AA, composition)})
    return out, rows


def reference(split="test"):
    data = np.load(CACHE / f"{split}_ca.npz", allow_pickle=False)
    return data["coords"], data["types"], data["lengths"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=None)
    ap.add_argument("--reference-split", default="test")
    ap.add_argument("--reference-n", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = []
    rc, rt, rl = reference(args.reference_split)
    n = min(args.reference_n, len(rl))
    results.append(score(rc[:n], rt[:n], rl[:n], f"AlphaFold {args.reference_split}")[0])
    if args.samples:
        d = np.load(args.samples, allow_pickle=False)
        results.append(score(d["coords"], d["types"], d["lengths"], "generated")[0])

    keys = ["n", "validity", "bond_mean", "bond_sd", "bond_ok", "clash_free",
            "clashes_per_structure", "radius_mean", "right_handed", "helix_fraction"]
    width = max(len(r["name"]) for r in results)
    print(f"{'':{width}} " + " ".join(f"{k:>12}" for k in keys))
    for r in results:
        print(f"{r['name']:{width}} " + " ".join(
            f"{r[k]:12.4f}" if isinstance(r[k], float) else f"{r[k]:12d}" for k in keys))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
