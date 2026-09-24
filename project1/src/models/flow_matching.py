#!/usr/bin/env python3
"""Flow matching baseline on QM9 3D molecular coordinates.

Run:
    python qm9_flow.py --smoke                 # 500 molecules, 200 steps, ~1 min on CPU
    python qm9_flow.py --epochs 50             # real run (GPU strongly recommended)
    python qm9_flow.py --mode sample --ckpt outputs/ckpt.pt

Install: pip install torch datasets numpy
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Constants for QM9
ATOM_TYPES = ["H", "C", "N", "O", "F"]   # the only 5 elements in QM9
N_TYPES = len(ATOM_TYPES)
MAX_ATOMS = 29                           # biggest QM9 molecule; everything pads up to this
FEAT_DIM = 3 + N_TYPES                   # per atom: 3 coords + 5 one-hot type channels
TYPE_SCALE = 0.25                        # shrink one-hots so types do not dominate the loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Data
def load_qm9(limit=None, seed=0):
    """QM9 -> padded tensors.

    Returns:
      x1        [N, 29, 8]  float  - 3 coords + 5 one-hot type channels per atom
      mask      [N, 29]     bool   - True where a real atom sits, False for padding
      coord_std scalar             - the number we divided coords by (needed to undo it)
    """
    from datasets import load_dataset

    # only two columns matter; dropping the rest makes the row loop much faster
    ds = load_dataset("structure-epflai/qm9", split="train")
    ds = ds.select_columns(["atomic_symbols", "pos"])
    if limit is not None:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    print(f"[data] {len(ds)} molecules, converting to padded tensors...")

    type_index = {s: i for i, s in enumerate(ATOM_TYPES)}
    n = len(ds)
    x = np.zeros((n, MAX_ATOMS, FEAT_DIM), dtype=np.float32)   # start all-zero = all padding
    mask = np.zeros((n, MAX_ATOMS), dtype=bool)

    for i, row in enumerate(ds):
        symbols = row["atomic_symbols"]                 # e.g. ["C","H","H","H","H"]
        pos = np.asarray(row["pos"], dtype=np.float32)  # [k, 3] angstroms
        k = len(symbols)
        if k > MAX_ATOMS:
            raise ValueError(f"molecule {i} has {k} atoms, more than MAX_ATOMS={MAX_ATOMS}")
        x[i, :k, :3] = pos                              # fill the first k slots
        for j, s in enumerate(symbols):
            x[i, j, 3 + type_index[s]] = 1.0            # one-hot the element
        mask[i, :k] = True                              # mark those k slots as real

    x = torch.from_numpy(x)
    mask = torch.from_numpy(mask)

    # move every molecule so its atoms average to the origin
    # (position in space is not part of the molecule; don't make the model learn it)
    x[..., :3] = center(x[..., :3], mask)

    # divide coords by one global std so data looks like N(0,1), same scale as the prior
    coord_std = x[..., :3][mask].std().item()
    x[..., :3] /= coord_std

    # shrink one-hots: unscaled they have bigger variance than coords and hog the loss
    x[..., 3:] *= TYPE_SCALE

    # belt and braces: zero everything in padded slots
    x = x * mask[..., None]

    sizes = mask.sum(1)
    print(f"[data] atoms per molecule: min {sizes.min()}, mean {sizes.float().mean():.1f}, "
          f"max {sizes.max()}   coord_std {coord_std:.3f} A")
    return x, mask, coord_std


def center(coords, mask):
    """Move each molecule so its real atoms average to the origin.
    coords [B, A, 3], mask [B, A]. Padded slots are ignored and come back zero.
    """
    m = mask[..., None].float()
    mean = (coords * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1)
    return (coords - mean) * m


def sample_prior(n, mask):
    """Draw X_0: Gaussian noise shaped like the data.
    - must be centered too, because the data is centered
      (if noise and data live in different spaces, the flow has nothing sensible to learn)
    - padded slots are zeroed
    """
    z = torch.randn(n, MAX_ATOMS, FEAT_DIM, device=mask.device)
    z[..., :3] = center(z[..., :3], mask)
    return z * mask[..., None]


def random_rotate(x, mask):
    """Randomly rotate each molecule. Data augmentation. A rotated molecule is the same molecule. The network does not know that,
    so we show it rotated copies until it stops caring about orientation.
    """
    b = x.shape[0]
    a = torch.randn(b, 3, 3, device=x.device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    # force det = +1: a mirror image is a DIFFERENT molecule, so no reflections
    det = torch.linalg.det(q)
    q = torch.cat([q[:, :, :1] * det[:, None, None], q[:, :, 1:]], dim=2)
    coords = torch.bmm(x[..., :3], q)
    out = torch.cat([coords, x[..., 3:]], dim=-1)
    return out * mask[..., None]


# 2. Model
def timestep_embedding(t, dim):
    #Turn the scalar t into a vector of sines and cosines.Networks read a rich vector far better than one raw number. t: [B] -> [B, dim].
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None] * 1000.0
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class AtomTransformer(nn.Module):
    """The velocity field v_theta. In: (X_t, t, mask). Out: a velocity per atom.
    - a transformer over atoms, with NO positional encoding
    - so atoms are treated as a set: shuffle the input, the output shuffles the same way
      (atoms have no natural order, unlike words in a sentence)
    - t is embedded once and added to every atom
    """

    def __init__(self, width=256, layers=6, heads=8):
        super().__init__()
        self.width = width
        self.embed = nn.Linear(FEAT_DIM, width)
        self.time = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(width, FEAT_DIM)
        self.skip = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, mask):
        temb = self.time(timestep_embedding(t, self.width))       # [B, W]
        h = self.embed(x) + temb[:, None, :]                      # [B, A, W]
        h = self.encoder(h, src_key_padding_mask=~mask)     # ~mask: True means "ignore this slot"
        # output = (something) * x + (learned correction); easier than learning v from scratch
        v = self.skip(temb)[:, :, None] * x + self.out(h)
        v = torch.cat([center(v[..., :3], mask), v[..., 3:]], dim=-1)  # keep velocity centered too
        return v * mask[..., None]


# 3. Flow matching loss
def flow_matching_loss(model, x1, mask, augment=True):
    if augment:
        x1 = random_rotate(x1, mask)
    x0 = sample_prior(len(x1), mask)                              # X_0: noise
    t = torch.rand(len(x1), device=x1.device)                     # a random time per molecule
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1      # X_t: a point on the line
    target = x1 - x0                                              # U_t: the velocity to predict
    pred = model(xt, t, mask)
    sq = (pred - target).pow(2) * mask[..., None]                 # zero out padding
    # divide by REAL atoms, not 29 x batch, or the loss depends on how much padding you got
    return sq.sum() / (mask.sum() * FEAT_DIM)


# 4. Training
def train(x1, mask, epochs, batch_size, lr, width, layers, out_dir, val_frac=0.05):
    n_val = max(1, int(val_frac * len(x1)))
    perm = torch.randperm(len(x1))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = TensorDataset(x1[train_idx], mask[train_idx])
    val_x, val_mask = x1[val_idx].to(DEVICE), mask[val_idx].to(DEVICE)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = AtomTransformer(width=width, layers=layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M   device: {DEVICE}")

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        total, steps = 0.0, 0
        for xb, mb in loader:
            xb, mb = xb.to(DEVICE), mb.to(DEVICE)
            loss = flow_matching_loss(model, xb, mb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1

        # val loss is noisy (new random t and X_0 every time), so average 8 draws
        model.eval()
        with torch.no_grad():
            v = sum(flow_matching_loss(model, val_x, val_mask, augment=False).item()
                    for _ in range(8)) / 8
        print(f"epoch {epoch+1:3d}  train {total/max(steps,1):.4f}  val {v:.4f}")

        if v < best:                                  # keep the best-on-validation weights
            best = v
            torch.save({"model": model.state_dict(), "width": width,
                        "layers": layers, "val_loss": v, "epoch": epoch + 1},
                       out_dir / "ckpt.pt")
    print(f"best validation loss {best:.4f}  ->  {out_dir/'ckpt.pt'}")
    return model


# 5. Sampling: Euler integration of the learned velocity field
@torch.no_grad()
def sample(model, n_atoms, steps=100):
    """Generate molecules: start at noise, take `steps` small Euler steps to t=1.
    n_atoms [B]: how many atoms each generated molecule should have.
    """
    b = len(n_atoms)
    ar = torch.arange(MAX_ATOMS, device=DEVICE)[None, :]
    mask = ar < n_atoms.to(DEVICE)[:, None]
    x = sample_prior(b, mask)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=DEVICE)
        x = x + dt * model(x, t, mask)
    return x, mask


def decode(x, mask, coord_std):
    """Undo the preprocessing so the output is readable.
    - multiply coords back by coord_std to get angstroms
    - argmax the 5 type channels to pick an element
    """
    coords = (x[..., :3] * coord_std).cpu().numpy()
    types = x[..., 3:].argmax(dim=-1).cpu().numpy()
    mask = mask.cpu().numpy()
    out = []
    for i in range(len(coords)):
        k = int(mask[i].sum())
        out.append(([ATOM_TYPES[j] for j in types[i, :k]], coords[i, :k]))
    return out


def write_xyz(mols, path):
    with open(path, "w") as f:
        for symbols, coords in mols:
            f.write(f"{len(symbols)}\ngenerated\n")
            for s, c in zip(symbols, coords):
                f.write(f"{s} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")


# 6. Minimal sanity metrics (NOT the full stability/validity protocol)
def quick_metrics(mols):
    """Rough check: does this look like a molecule or a random cloud of dots?

    NOT the real metrics. Before the paper you need bond inference from distances,
    valency checks, and an RDKit validity parse.
    """
    nn_dists, clashes, total = [], 0, 0
    for symbols, coords in mols:
        if len(symbols) < 2:
            continue
        d = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)  # all pairwise distances
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        nn_dists.append(nearest)
        clashes += int((nearest < 0.7).sum())     # no two atoms are ever this close in reality
        total += len(symbols)
    nn = np.concatenate(nn_dists)
    return {
        "mean_nearest_neighbor_dist": float(nn.mean()),   # real QM9 sits around 1.0-1.1 A
        "frac_atoms_in_clash": clashes / max(total, 1),
        "n_molecules": len(mols),
    }


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "sample", "train-sample"], default="train-sample")
    p.add_argument("--smoke", action="store_true", help="tiny subset, few epochs, CPU friendly")
    p.add_argument("--limit", type=int, default=None, help="use only this many molecules")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--ckpt", type=Path, default=None)
    args = p.parse_args()

    if args.smoke:
        args.limit, args.epochs, args.width, args.layers = 500, 3, 64, 2
        args.batch_size, args.samples, args.sample_steps = 32, 8, 20

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    x1, mask, coord_std = load_qm9(limit=args.limit, seed=args.seed)
    print(f"loaded {len(x1)} molecules  coord_std={coord_std:.3f}  "
          f"atoms/molecule mean={mask.sum(1).float().mean():.1f}")

    if args.mode in ("train", "train-sample"):
        model = train(x1, mask, args.epochs, args.batch_size, args.lr,
                      args.width, args.layers, args.out)
    if args.mode == "train":
        return

    ck = torch.load(args.ckpt or args.out / "ckpt.pt", map_location=DEVICE, weights_only=True)
    model = AtomTransformer(width=ck["width"], layers=ck["layers"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    # pick molecule sizes by copying sizes seen in the training data
    sizes = mask.sum(1)
    n_atoms = sizes[torch.randint(len(sizes), (args.samples,))]

    x, m = sample(model, n_atoms, steps=args.sample_steps)
    mols = decode(x, m, coord_std)
    write_xyz(mols, args.out / "samples.xyz")

    print("generated:", quick_metrics(mols))
    ref = decode(x1[:args.samples].to(DEVICE) / 1.0, mask[:args.samples].to(DEVICE), coord_std)
    print("real data:", quick_metrics(ref))
    print(f"wrote {args.out/'samples.xyz'}")


if __name__ == "__main__":
    main()
