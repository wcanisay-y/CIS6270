"""QM9 preprocessing for CIS 6270 Project 1, modality 1 (3D molecule coordinates).

Adapted from src/data/qm9.py in wcanisay-y/CIS6270@benchmarks. The public API is unchanged
(ATOM_VOCAB, make_splits, filter_by_size, subsample, make_collate_fn, prepare_qm9), so existing
code keeps working. Four things were changed, each for a reason checked against the dataset:

  1. Source is structure-epflai/qm9. It is a byte-identical clone of yairschiff/qm9 (same parquet
     sha256), so this is a rename, not a data change.
  2. Bulk column reads replace per-row indexing. ds[i] in a loop over 133,885 rows takes minutes;
     reading the three columns at once takes seconds.
  3. Optional exclusion of the 3,054 molecules that failed QM9's geometry consistency check.
     They are present in this dataset and absent from every published EDM/GCDM number.
  4. The collate function returns a float mask shaped [B, N, 1] and one-hot atom types, which is
     what an equivariant denoiser consumes.

The "charges" column is deliberately never read: it holds Mulliken PARTIAL charges (floats), not
atomic numbers. EDM's "charges" means atomic number, so mixing them up silently corrupts atom
types. Atom identity comes from "atomic_symbols" only.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

ATOM_VOCAB: Dict[str, int] = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
ATOM_SYMBOLS: List[str] = ["H", "C", "N", "O", "F"]
DATASET_ID = "structure-epflai/qm9"
QM9_MAX_ATOMS = 29
N_UNCHARACTERIZED = 3054


def load_qm9_raw(split: str = "train"):
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split=split)


def drop_uncharacterized(ds, uncharacterized_txt: Path):
    """Remove the molecules QM9 flags as geometrically inconsistent.

    uncharacterized.txt lists 1-based GDB indices. Row i is taken to be GDB index i+1, which holds
    because the dataset was built from sorted xyz filenames; spot-checked on rows 0-2 (CH4, NH3,
    H2O) and the last row, not proven for all 133,885.

    Only the FIRST token of each line is read, which is what EDM does. Splitting on every token
    instead picks up "133885" out of the header sentence ("List of molecules among the 133885
    GDB9 molecules...") and silently drops the last molecule as well -- 3,055 removed instead of
    3,054. Verified against the real file.
    """
    excluded = set()
    for line in Path(uncharacterized_txt).read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            excluded.add(int(parts[0]) - 1)
    if len(excluded) != N_UNCHARACTERIZED:
        raise ValueError(f"{uncharacterized_txt} lists {len(excluded)} indices, "
                         f"expected {N_UNCHARACTERIZED}; wrong file?")
    keep = [i for i in range(len(ds)) if i not in excluded]
    print(f"excluded {len(ds) - len(keep)} uncharacterized molecules, {len(keep)} remain")
    return ds.select(keep)


def make_splits(ds, test_size: float = 0.1, val_size: float = 0.1, seed: int = 1):
    split1 = ds.train_test_split(test_size=test_size, seed=seed)
    train_val, test_ds = split1["train"], split1["test"]
    rel_val_size = val_size / (1.0 - test_size)
    split2 = train_val.train_test_split(test_size=rel_val_size, seed=seed)
    return split2["train"], split2["test"], test_ds


def filter_by_size(ds, max_atoms: Optional[int] = None):
    if max_atoms is None:
        return ds
    return ds.filter(lambda x: x["num_atoms"] <= max_atoms)


def subsample(ds, n_molecules: Optional[int] = None, seed: int = 0):
    if n_molecules is None or n_molecules >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n_molecules))


def canonicalize(pos: torch.Tensor) -> torch.Tensor:
    """Rotate one zero-centroid molecule into its own inertial frame.

    Why this exists: a denoiser without built-in rotation equivariance has to learn each shape
    separately in every orientation it is shown. Random rotation augmentation makes that worse,
    not better -- the lowest-loss strategy becomes predicting the orientation-averaged answer,
    which is a spherical blob. Fixing the frame removes the nuisance instead of asking the
    network to absorb it, and costs no parameters.

    The frame is the eigenbasis of M = sum_i x_i x_i^T ordered by descending eigenvalue. Two
    details matter:
      * Eigenvectors are only defined up to sign, so the sign of each axis is pinned by the atom
        with the largest projection onto it. Without this, two copies of the same molecule can
        land in mirrored frames and the model sees noise.
      * det(V) is forced to +1 by flipping the LAST axis (the least informative one). A frame
        with det = -1 is a reflection, which would silently map molecules onto their mirror
        images and destroy chirality.

    Degenerate case: molecules with rotational symmetry have repeated eigenvalues, so the frame
    is genuinely ambiguous. QM9 geometries are DFT-relaxed and rarely exactly symmetric, but the
    ambiguity is real and worth stating rather than hiding.
    """
    gram = pos.T @ pos
    _, vectors = torch.linalg.eigh(gram)          # ascending eigenvalue
    vectors = vectors.flip(-1)                     # descending: principal axis first
    projection = pos @ vectors
    anchor = projection.abs().argmax(0)            # the atom that pins each axis
    signs = torch.sign(projection[anchor, torch.arange(3)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    vectors = vectors * signs
    if torch.det(vectors) < 0:
        vectors[:, 2] = -vectors[:, 2]             # keep a rotation, never a reflection
    return pos @ vectors


def to_tensors(ds, max_atoms: int, canonical: bool = False) -> Dict[str, torch.Tensor]:
    """Read the whole split into padded, center-of-mass-centred tensors.

    Returns pos [M, N, 3] float32 in Angstrom, atom_type [M, N] long, mask [M, N, 1] float32.
    Padded slots hold zeros and mask 0, so they can never contribute to a sum or a mean.
    """
    symbols = ds["atomic_symbols"]
    positions = ds["pos"]
    counts = ds["num_atoms"]
    total = len(counts)

    pos = torch.zeros(total, max_atoms, 3, dtype=torch.float32)
    atom_type = torch.zeros(total, max_atoms, dtype=torch.long)
    mask = torch.zeros(total, max_atoms, 1, dtype=torch.float32)

    kept = 0
    for syms, xyz, count in zip(symbols, positions, counts):
        if count > max_atoms or any(s not in ATOM_VOCAB for s in syms):
            continue
        coords = torch.tensor(np.asarray(xyz, dtype=np.float32))
        coords = coords - coords.mean(0, keepdim=True)
        if canonical:
            coords = canonicalize(coords)
        pos[kept, :count] = coords
        atom_type[kept, :count] = torch.tensor([ATOM_VOCAB[s] for s in syms], dtype=torch.long)
        mask[kept, :count] = 1.0
        kept += 1
    if kept != total:
        print(f"note: skipped {total - kept} molecules (too large or unexpected element)")
    return {"pos": pos[:kept], "atom_type": atom_type[:kept], "mask": mask[:kept]}


class MoleculeDataset(Dataset):
    """Indexable view over the padded tensors, so DataLoader needs no collate function."""

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.pos = tensors["pos"]
        self.atom_type = tensors["atom_type"]
        self.mask = tensors["mask"]

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, i: int):
        one_hot = torch.zeros(self.pos.shape[1], len(ATOM_SYMBOLS))
        one_hot[torch.arange(self.pos.shape[1]), self.atom_type[i]] = 1.0
        return self.pos[i], one_hot * self.mask[i], self.mask[i]


def make_collate_fn(max_atoms: int):
    """Kept for compatibility with the original module; pads a list of per-molecule dicts."""

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        pos = torch.zeros(bsz, max_atoms, 3, dtype=torch.float32)
        atom_type = torch.zeros(bsz, max_atoms, dtype=torch.long)
        mask = torch.zeros(bsz, max_atoms, 1, dtype=torch.float32)
        for i, ex in enumerate(batch):
            n = int(ex["num_atoms"])
            pos[i, :n] = ex["pos"]
            atom_type[i, :n] = ex["atom_type"]
            mask[i, :n] = 1.0
        return {"pos": pos, "atom_type": atom_type, "mask": mask}

    return collate


def size_histogram(mask: torch.Tensor, max_atoms: int) -> torch.Tensor:
    """Empirical distribution over molecule size, used to draw N when sampling."""
    sizes = mask.sum((1, 2)).long()
    counts = torch.bincount(sizes, minlength=max_atoms + 1).float()
    return counts / counts.sum()


def compute_property_stats(ds, prop_name: str) -> Tuple[float, float]:
    values = np.array(ds[prop_name], dtype=np.float64)
    return float(values.mean()), float(values.std())


def normalize_property(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def denormalize_property(norm_value: float, mean: float, std: float) -> float:
    return norm_value * std + mean


def prepare_qm9(
    max_atoms: Optional[int] = None,
    n_molecules: Optional[int] = None,
    seed: int = 1,
    uncharacterized_txt: Optional[Path] = None,
    canonical: bool = False,
) -> Dict:
    ds = load_qm9_raw()
    if uncharacterized_txt is not None:
        ds = drop_uncharacterized(ds, uncharacterized_txt)
    ds = subsample(ds, n_molecules=n_molecules, seed=seed)
    ds = filter_by_size(ds, max_atoms=max_atoms)
    resolved_max_atoms = max_atoms or QM9_MAX_ATOMS
    train_ds, val_ds, test_ds = make_splits(ds, seed=seed)

    train_tensors = to_tensors(train_ds, resolved_max_atoms, canonical)
    return {
        "train": MoleculeDataset(train_tensors),
        "valid": MoleculeDataset(to_tensors(val_ds, resolved_max_atoms, canonical)),
        "test": MoleculeDataset(to_tensors(test_ds, resolved_max_atoms, canonical)),
        "max_atoms": resolved_max_atoms,
        "size_distribution": size_histogram(train_tensors["mask"], resolved_max_atoms),
        "train_tensors": train_tensors,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the preprocessed QM9 tensors.")
    parser.add_argument("--max_atoms", type=int, default=None)
    parser.add_argument("--n_molecules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--uncharacterized", type=Path, default=None)
    parser.add_argument("--canonical", action="store_true")
    args = parser.parse_args()

    data = prepare_qm9(args.max_atoms, args.n_molecules, args.seed, args.uncharacterized,
                       args.canonical)
    print(f"train: {len(data['train'])}")
    print(f"valid: {len(data['valid'])}")
    print(f"test:  {len(data['test'])}")
    pos, one_hot, mask = data["train"][0]
    print("\nsample 0")
    print("num_atoms:", int(mask.sum()))
    print("atom_type:", one_hot[: int(mask.sum())].argmax(-1).tolist())
    print("centre of mass:", pos.sum(0).tolist())
