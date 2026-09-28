"""Reference rows and sanity checks for the protein run. Not part of the teammate's code.

Run as `PYTHONPATH=. python logs/diagnostics.py` from project1/proteins. Uses the functions in metrics.py and sample.py unchanged, reads
data/pfam_esm2.pt, ckpt/classifier.pt and out/abl_*.pt, and writes nothing.
"""
from collections import Counter

import numpy as np
import torch

from data.pfam import PfamData
from metrics import classify_family, decode_sequences, mean_pairwise_identity, sequence_identity
from sample import load

dev = "cuda" if torch.cuda.is_available() else "cpu"
data = PfamData("data/pfam_esm2.pt")
clf, _ = load("ckpt/classifier.pt", dev)
zm, zs = float(data.z_mean), float(data.z_std)
print("families:", data.family_names, "| training share:", [round(float(v), 3) for v in data.family_hist])


def describe(name, z, fam):
    seqs = decode_sequences(z, zm, zs, dev)
    acc = (classify_family(z, clf, dev).cpu() == fam).float().mean().item()
    x = np.mean([s.count("X") / max(len(s), 1) for s in seqs])
    print(f"{name:22s} n={len(seqs):5d}  family_accuracy={acc:.3f}  "
          f"diversity={1 - mean_pairwise_identity(seqs):.4f}  X fraction={x:.3f}  "
          f"uniqueness={len(set(seqs)) / len(seqs):.3f}")
    return seqs


def nearest(seqs, pool, k=200):
    pick = np.random.RandomState(0).choice(len(seqs), min(k, len(seqs)), replace=False)
    return np.array([max(sequence_identity(seqs[i], p) for p in pool) for i in pick])


real_train = describe("REAL train", *data.splits["train"])
real_test = describe("REAL test", *data.splits["test"])
cache = torch.load("data/pfam_esm2.pt", weights_only=False)
stored = [cache["sequences"][i] for i in cache["test_idx"].tolist()]
print(f"decoder round trip on real test latents: identity to the stored sequence "
      f"{np.mean([sequence_identity(a, b) for a, b in zip(real_test, stored)]):.4f}")

gen = {}
for name in ("abl_none", "abl_cfg", "abl_global", "abl_shuffled", "abl_gated"):
    s = torch.load(f"out/{name}.pt", weights_only=False)
    gen[name] = describe(name, s["z"], s["family"])

pool = sorted(set(real_train))
aa = "ACDEFGHIKLMNPQRSTVWYX"
comp = lambda seqs: np.array([Counter("".join(seqs))[a] for a in aa], float) / sum(map(len, seqs))
print(f"nearest training sequence: REAL test      identity {nearest(real_test, pool).mean():.3f}")
for name, seqs in gen.items():
    nn = nearest(seqs, pool)
    print(f"nearest training sequence: {name:14s} identity {nn.mean():.3f} (max {nn.max():.3f})   "
          f"composition distance to real train {0.5 * np.abs(comp(seqs) - comp(real_train)).sum():.3f}")
print("example REAL test :", real_test[0])
for name in ("abl_none", "abl_cfg", "abl_gated"):
    print(f"example {name:10s}:", gen[name][0])
