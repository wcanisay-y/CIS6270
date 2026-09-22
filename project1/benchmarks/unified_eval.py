# score generated qm9 molecules (xyz files) with one shared set of metrics.
# reuses edm's bond-order rules and rdkit molecule builder so numbers are
# comparable to the edm/gcdm papers; novelty is measured against the local
# huggingface qm9 parquet instead of each repo's own copy of qm9.

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BENCH_DIR, "edm"))

from configs.datasets_config import get_dataset_info  # noqa: E402
from qm9.analyze import check_stability  # noqa: E402
from qm9.rdkit_functions import build_molecule, mol2smiles  # noqa: E402

# local copy of the structure-epflai/qm9 parquet: $QM9_PARQUET, else data/qm9/data/ next to this
# script (where setup.sh puts it), else the original absolute path used for the reported results
QM9_PARQUET = (
    os.environ.get("QM9_PARQUET")
    or next(iter(sorted(glob.glob(os.path.join(BENCH_DIR, "data", "qm9", "data", "*.parquet")))), None)
    or "/mnt/isilon/tan_lab/pany3/data/qm9/data/train-00000-of-00001-baa918c342229731.parquet"
)
DATASET_INFO = get_dataset_info("qm9", remove_h=False)
ATOM_ENCODER = DATASET_INFO["atom_encoder"]

Molecule = Tuple[np.ndarray, np.ndarray]  # (positions n x 3, atom type ids n)


def read_xyz(path: str) -> List[Molecule]:
    # handles one or many molecules per file
    mols = []
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        atoms = [ln.split() for ln in lines[i + 2 : i + 2 + n]]
        types = np.array([ATOM_ENCODER[a[0]] for a in atoms])
        pos = np.array([[float(v) for v in a[1:4]] for a in atoms])
        mols.append((pos, types))
        i += 2 + n
    return mols


def load_generated(sample_dir: str) -> List[Molecule]:
    files = sorted(glob.glob(os.path.join(sample_dir, "**", "*.txt"), recursive=True))
    files += sorted(glob.glob(os.path.join(sample_dir, "**", "*.xyz"), recursive=True))
    mols = []
    for fp in files:
        mols.extend(read_xyz(fp))
    return mols


def load_reference(n: int = None, seed: int = 0) -> List[Molecule]:
    df = pd.read_parquet(QM9_PARQUET, columns=["atomic_symbols", "pos"])
    if n is not None:
        df = df.sample(n=n, random_state=seed)
    return [
        (np.stack(p).astype(np.float64), np.array([ATOM_ENCODER[s] for s in sym]))
        for sym, p in zip(df["atomic_symbols"], df["pos"])
    ]


def to_smiles(mol: Molecule):
    # same procedure as edm: infer bonds from distances, keep largest fragment
    rd = build_molecule(torch.tensor(mol[0]), torch.tensor(mol[1]), DATASET_INFO)
    if mol2smiles(rd) is None:
        return None
    frags = Chem.rdmolops.GetMolFrags(rd, asMols=True)
    return mol2smiles(max(frags, default=rd, key=lambda m: m.GetNumAtoms()))


def reference_smiles(cache: str) -> set:
    if os.path.exists(cache):
        with open(cache) as f:
            return set(json.load(f))
    smiles = {s for s in map(to_smiles, load_reference()) if s is not None}
    with open(cache, "w") as f:
        json.dump(sorted(smiles), f)
    return smiles


def evaluate(mols: List[Molecule], ref_smiles: set) -> dict:
    n_atoms = n_stable_atoms = n_stable_mols = 0
    for pos, types in mols:
        mol_stable, stable_atoms, total = check_stability(pos, types, DATASET_INFO)
        n_stable_mols += int(mol_stable)
        n_stable_atoms += stable_atoms
        n_atoms += total

    valid = [s for s in map(to_smiles, mols) if s is not None]
    unique = set(valid)
    novel = unique - ref_smiles
    return {
        "n_molecules": len(mols),
        "atom_stability": n_stable_atoms / n_atoms,
        "molecule_stability": n_stable_mols / len(mols),
        "validity": len(valid) / len(mols),
        "uniqueness": len(unique) / max(len(valid), 1),
        "valid_and_unique": len(unique) / len(mols),
        "novelty": len(novel) / max(len(unique), 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=[], help="name=dir pairs of generated xyz files")
    parser.add_argument("--reference_n", type=int, default=10000, help="qm9 molecules scored as the data row")
    parser.add_argument("--out", default=os.path.join(BENCH_DIR, "results", "unified_metrics.json"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ref = reference_smiles(os.path.join(os.path.dirname(args.out), "qm9_parquet_smiles.json"))
    print(f"reference: {len(ref)} unique qm9 smiles from local parquet")

    results = {"QM9 data": evaluate(load_reference(args.reference_n), set())}
    results["QM9 data"]["novelty"] = None  # not meaningful for the dataset itself
    for pair in args.samples:
        name, sample_dir = pair.split("=", 1)
        results[name] = evaluate(load_generated(sample_dir), ref)

    for name, r in results.items():
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
