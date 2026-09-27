"""QM9 -> dense padded tensors.

Representation (both model families use the same one):
    x    [B, N, 3]  atom coordinates in Angstrom, center of mass removed
    h    [B, N, 5]  one-hot atom type over (H, C, N, O, F), scaled by H_SCALE
    mask [B, N]     1 for real atoms, 0 for padding
    prop [B]        the conditioning property (polarizability alpha), standardized

Coordinates are NOT rescaled: they live on the zero-CoM subspace of R^{N x 3},
which is where the prior must also live (see zero_com in guidance.py).
"""
import argparse
from pathlib import Path

import torch

ATOMS = ["H", "C", "N", "O", "F"]
Z2IDX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
MAX_N = 29          # QM9 with explicit hydrogens
H_SCALE = 1.0       # one-hot magnitude; keeps h and x on comparable scales
PROP_IDX = 1        # PyG QM9 target index 1 = alpha (Bohr^3)
SPLIT = (100000, 18000)   # train, val; remainder is test


def _center(x, mask):
    n = mask.sum(-1, keepdim=True).clamp_min(1)
    com = (x * mask[..., None]).sum(-2, keepdim=True) / n[..., None]
    return (x - com) * mask[..., None]


def build_cache(root, out):
    """One-off: read PyG QM9 and write dense tensors to `out`."""
    from torch_geometric.datasets import QM9

    ds = QM9(root)
    m = len(ds)
    x = torch.zeros(m, MAX_N, 3)
    h = torch.zeros(m, MAX_N, len(ATOMS))
    mask = torch.zeros(m, MAX_N)
    prop = torch.zeros(m)
    for i, g in enumerate(ds):
        n = g.z.numel()
        x[i, :n] = g.pos
        h[i, :n, :] = torch.nn.functional.one_hot(
            torch.tensor([Z2IDX[int(z)] for z in g.z]), len(ATOMS)
        ).float()
        mask[i, :n] = 1.0
        prop[i] = g.y[0, PROP_IDX]
    x = _center(x, mask)
    torch.save({"x": x, "h": h, "mask": mask, "prop": prop}, out)
    print(f"cached {m} molecules to {out}")


class QM9Data:
    """Holds the three splits plus the statistics sampling needs."""

    def __init__(self, cache, limit=None, seed=0):
        d = torch.load(cache)
        m = len(d["mask"])
        perm = torch.randperm(m, generator=torch.Generator().manual_seed(seed))
        if limit:                      # small-scale debugging runs
            perm = perm[:limit]
            m = limit
        n_tr = min(SPLIT[0], int(0.77 * m))
        n_va = min(SPLIT[1], int(0.14 * m))
        idx = {"train": perm[:n_tr],
               "val": perm[n_tr:n_tr + n_va],
               "test": perm[n_tr + n_va:]}
        self.splits = {}
        for k, ix in idx.items():
            self.splits[k] = (d["x"][ix], d["h"][ix] * H_SCALE,
                              d["mask"][ix], d["prop"][ix])
        tr_prop = self.splits["train"][3]
        self.prop_mean, self.prop_std = tr_prop.mean(), tr_prop.std() # property statistics for standardization
        for k, (a, b, c, p) in self.splits.items():
            self.splits[k] = (a, b, c, (p - self.prop_mean) / self.prop_std)
        # size distribution, needed to sample n before generating
        counts = self.splits["train"][2].sum(-1).long()
        self.n_hist = torch.bincount(counts, minlength=MAX_N + 1).float()
        self.n_hist /= self.n_hist.sum()

    def loader(self, split, batch_size, shuffle=True):
        ts = torch.utils.data.TensorDataset(*self.splits[split])
        return torch.utils.data.DataLoader(ts, batch_size=batch_size,
                                           shuffle=shuffle, drop_last=shuffle)

    def sample_mask(self, b, device):
        """Draw molecule sizes from the training histogram, return a mask."""
        n = torch.multinomial(self.n_hist, b, replacement=True).to(device)
        ar = torch.arange(MAX_N, device=device)[None, :]
        return (ar < n[:, None]).float()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="build the QM9 tensor cache")
    p.add_argument("--root", default="data/qm9_raw")
    p.add_argument("--out", default="data/qm9.pt")
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    build_cache(a.root, a.out)