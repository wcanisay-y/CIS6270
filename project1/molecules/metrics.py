"""Evaluation on FINAL generated molecules only.

    atom stability     fraction of atoms whose inferred valency matches the
                       element's allowed valency
    molecule stability fraction of molecules in which every atom is stable
    validity           fraction whose inferred bond graph gives an RDKit-
                       sanitizable SMILES
    uniqueness         distinct canonical SMILES, over valid molecules
    novelty            valid SMILES absent from the training set
    property MAE       |requested - predicted| for the guidance target metric

Bonds are inferred from interatomic distances against a reference length table
with margins (Hoogeboom et al. 2022). Novelty on QM9 should be read with care:
the dataset is a near-exhaustive enumeration of its chemical space, so a high
novelty score partly indicates off-distribution generation.
"""
import argparse
import json
from pathlib import Path

import torch

from guidance import VALENCE, _B1, _B2, _B3
from data.qm9 import ATOMS, H_SCALE

MARGIN = (0.10, 0.05, 0.03)     # Angstrom, for single / double / triple


def bond_orders(x, types, mask):
    """[B,N,N] integer bond orders from pairwise distances."""
    dev = x.device
    b1, b2, b3 = (torch.tensor(t, device=dev) for t in (_B1, _B2, _B3))
    n = x.shape[1]
    pair = mask[:, :, None] * mask[:, None, :] * (1 - torch.eye(n, device=dev)[None])
    d = torch.cdist(x, x)
    ti, tj = types[:, :, None].expand(-1, -1, n), types[:, None, :].expand(-1, n, -1)
    r1, r2, r3 = b1[ti, tj], b2[ti, tj], b3[ti, tj]
    o = (d < r1 + MARGIN[0]).long()
    o = torch.where((r2 > 0) & (d < r2 + MARGIN[1]), torch.full_like(o, 2), o)
    o = torch.where((r3 > 0) & (d < r3 + MARGIN[2]), torch.full_like(o, 3), o)
    o = torch.where(d < r1 + MARGIN[0], o, torch.zeros_like(o))
    return o * pair.long()


def stability(x, h, mask):
    types = (h / H_SCALE).argmax(-1)
    order = bond_orders(x, types, mask)
    val = order.sum(-1).float()
    allowed = torch.tensor(VALENCE, device=x.device)[types]
    atom_ok = (val == allowed).float() * mask
    n_atoms = mask.sum()
    return (atom_ok.sum() / n_atoms).item(), \
           ((atom_ok.sum(-1) == mask.sum(-1)).float().mean()).item(), \
           types, order


def to_smiles(types, order, n):
    """RDKit canonical SMILES, or None if the graph does not sanitize."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    m = Chem.RWMol()
    for i in range(n):
        m.AddAtom(Chem.Atom(ATOMS[int(types[i])]))
    for i in range(n):
        for j in range(i + 1, n):
            o = int(order[i, j])
            if o:
                m.AddBond(i, j, bt[o])
    try:
        mol = m.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def smiles_set(x, h, mask):
    _, _, types, order = stability(x, h, mask)
    ns = mask.sum(-1).long()
    out = []
    for i in range(len(x)):
        out.append(to_smiles(types[i], order[i], int(ns[i])))
    return out


def evaluate(samples, train_smiles=None, prop_net=None, device="cpu", prop_std=None):
    x, h, mask = samples["x"].to(device), samples["h"].to(device), samples["mask"].to(device)
    atom, mol, _, _ = stability(x, h, mask)
    smi = smiles_set(x, h, mask)
    valid = [s for s in smi if s is not None]
    res = {"n_samples": len(x),
           "atom_stability": atom,
           "mol_stability": mol,
           "validity": len(valid) / max(len(smi), 1),
           "uniqueness": len(set(valid)) / max(len(valid), 1),
           "nfe": samples.get("nfe"),
           "mean_gate": samples.get("mean_gate")}
    if train_smiles is not None:
        res["novelty"] = sum(s not in train_smiles for s in set(valid)) / max(len(set(valid)), 1)
    if prop_net is not None:
        with torch.no_grad():
            t = torch.ones(len(x), device=device)
            pred = prop_net(x, h, t, mask)
        mae = (pred - samples["cond"]).abs().mean().item()
        res["prop_mae_standardized"] = mae
        if prop_std is not None:
            res["prop_mae"] = mae * float(prop_std)      # original units (Bohr^3)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--data", default="data/qm9.pt", help="for novelty")
    p.add_argument("--train-smiles", default="data/train_smiles.json")
    p.add_argument("--prop-ckpt", default=None, help="clean-only PropertyNet")
    p.add_argument("--no-novelty", action="store_true")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = torch.load(a.samples, weights_only=False)

    train_smiles = None
    if not a.no_novelty:
        cache = Path(a.train_smiles)
        if cache.exists():
            train_smiles = set(json.loads(cache.read_text()))
        else:
            from data.qm9 import QM9Data
            d = QM9Data(a.data)
            xt, ht, mt, _ = d.splits["train"]
            print(f"building training SMILES set from {len(xt)} molecules (one-off)")
            s = [v for v in smiles_set(xt, ht, mt) if v is not None]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(sorted(set(s))))
            train_smiles = set(s)

    prop_net, prop_std = None, None
    if a.prop_ckpt:
        from sample import load
        prop_net, ck = load(a.prop_ckpt, device)
        prop_std = ck["prop_std"]

    res = evaluate(samples, train_smiles, prop_net, device, prop_std)
    res["config"] = samples.get("config")
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()