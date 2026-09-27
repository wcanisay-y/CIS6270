# preprocess qm9 3d molecule dataset from huggingface

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

ATOM_VOCAB: Dict[str, int] = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

# local copy of structure-epflai/qm9 (identical to yairschiff/qm9)
QM9_PARQUET = "/mnt/isilon/tan_lab/pany3/data/qm9/data/*.parquet"


def load_qm9_raw(split: str = "train") -> Dataset:
    return load_dataset("parquet", data_files={"train": QM9_PARQUET}, split=split)


def make_splits(
    ds: Dataset, test_size: float = 0.1, val_size: float = 0.1, seed: int = 1
) -> Tuple[Dataset, Dataset, Dataset]:
    split1 = ds.train_test_split(test_size=test_size, seed=seed)
    train_val, test_ds = split1["train"], split1["test"]
    rel_val_size = val_size / (1.0 - test_size)
    split2 = train_val.train_test_split(test_size=rel_val_size, seed=seed)
    return split2["train"], split2["test"], test_ds


def filter_by_size(ds: Dataset, max_atoms: Optional[int] = None) -> Dataset:
    if max_atoms is None:
        return ds
    return ds.filter(lambda x: x["num_atoms"] <= max_atoms)


def subsample(ds: Dataset, n_molecules: Optional[int] = None, seed: int = 0) -> Dataset:
    if n_molecules is None or n_molecules >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n_molecules))


def to_tensor_sample(sample: Dict) -> Dict:
    pos = torch.tensor(sample["pos"], dtype=torch.float32)
    pos = pos - pos.mean(dim=0, keepdim=True)
    atom_type = torch.tensor(
        [ATOM_VOCAB[s] for s in sample["atomic_symbols"]], dtype=torch.long
    )
    return {"pos": pos, "atom_type": atom_type, "num_atoms": sample["num_atoms"]}


def build_tensor_dataset(ds: Dataset) -> List[Dict]:
    return [to_tensor_sample(ds[i]) for i in range(len(ds))]


def compute_property_stats(ds: Dataset, prop_name: str) -> Tuple[float, float]:
    values = np.array(ds[prop_name], dtype=np.float64)
    return float(values.mean()), float(values.std())


def normalize_property(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def denormalize_property(norm_value: float, mean: float, std: float) -> float:
    return norm_value * std + mean


def make_collate_fn(max_atoms: int):
    # adds padding to qm9 samples (which have variable atom counts) for constant batch dimensions
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        pos = torch.zeros(bsz, max_atoms, 3, dtype=torch.float32)
        atom_type = torch.zeros(bsz, max_atoms, dtype=torch.long)
        mask = torch.zeros(bsz, max_atoms, dtype=torch.bool)
        for i, ex in enumerate(batch):
            n = ex["num_atoms"]
            pos[i, :n] = ex["pos"]
            atom_type[i, :n] = ex["atom_type"]
            mask[i, :n] = True
        return {"pos": pos, "atom_type": atom_type, "mask": mask}
    return collate


def prepare_qm9(
    max_atoms: Optional[int] = None,
    n_molecules: Optional[int] = None,
    seed: int = 1,
) -> Dict:
    ds = load_qm9_raw()
    ds = subsample(ds, n_molecules=n_molecules, seed=seed)
    ds = filter_by_size(ds, max_atoms=max_atoms)
    train_ds, val_ds, test_ds = make_splits(ds, seed=seed)
    resolved_max_atoms = max_atoms or int(max(ds["num_atoms"]))

    return {
        "train_tensors": build_tensor_dataset(train_ds),
        "val_tensors": build_tensor_dataset(val_ds),
        "test_tensors": build_tensor_dataset(test_ds),
        "max_atoms": resolved_max_atoms,
        "collate_fn": make_collate_fn(resolved_max_atoms),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_atoms", type=int, default=None)
    parser.add_argument("--n_molecules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    data = prepare_qm9(max_atoms=args.max_atoms, n_molecules=args.n_molecules, seed=args.seed)

    print(f"train: {len(data['train_tensors'])}")
    print(f"val:   {len(data['val_tensors'])}")
    print(f"test:  {len(data['test_tensors'])}")

    sample = data["train_tensors"][0]
    print("\nsample:")
    print("num_atoms:", sample["num_atoms"])
    print("atom_type:", sample["atom_type"])
    print("pos:\n", sample["pos"])