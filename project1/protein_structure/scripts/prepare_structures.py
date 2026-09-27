#!/usr/bin/env python3
"""Build Ca point clouds for the SwissProt-Pfam cap-128 subset from AlphaFold DB.

Self-contained and independent of the QM9 scripts in ../../molecular_coordinates. Nothing here
imports or writes anything under that tree.

  filter -> deduplicate -> fetch AlphaFold PDB by UniProt accession -> parse Ca -> pack

Resumable: structures are cached in shards of SHARD_SIZE, and a re-run skips finished shards.

Usage:
    python prepare_structures.py                       # all three splits
    python prepare_structures.py --splits train --limit 2000    # quick check
"""
import argparse
import ast
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

DATA = os.environ.get("SWISSPROT_PFAM", "/mnt/isilon/tan_lab/pany3/data/swissprot_pfam/data")
# bulky input data, kept outside the repository; override the root with PROTEIN_DIFFUSION_ROOT
OUT = os.path.join(os.environ.get("PROTEIN_DIFFUSION_ROOT",
                                  "/mnt/isilon/tan_lab/pany3/protein_diffusion"), "cache")
MAX_RES = 128                 # node cap; see the costed plan
MIN_RES = 16                  # below this a Ca cloud carries almost no fold information
MIN_PLDDT = 70.0              # mean confidence floor; measured families sit at 85-97
SHARD_SIZE = 2000
WORKERS = 8                   # EBI is a shared service; do not raise this casually

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA)}
THREE2ONE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
             "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
             "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}


def load_subset(split, limit=None):
    """Rows of one split that fit the cap, carry exactly one Pfam family, and are unique.

    The integer `labels` column does NOT agree with the shipped idx_mapping.json (checked on
    20,000 rows: zero matches), so family identity is read from `labels_str` only.
    """
    df = pd.read_parquet(f"{DATA}/{split}-00000-of-00001.parquet",
                         columns=["id", "seq", "labels_str"])
    fams = df["labels_str"].map(ast.literal_eval)
    keep = (df["seq"].str.len().between(MIN_RES, MAX_RES)) & (fams.map(len) == 1)
    sub = df[keep].copy()
    sub["family"] = [f[0].replace("Pfam:", "") for f in fams[keep]]
    before = len(sub)
    sub = sub.drop_duplicates(subset="seq", keep="first")   # 28% of the cap-128 train rows
    print(f"[{split}] {before:,} rows pass the filter, {len(sub):,} after exact deduplication")
    if limit:
        sub = sub.head(limit)
    return sub.reset_index(drop=True)


def fetch(accession, retries=3):
    """AlphaFold PDB text for one accession, or None if the entry is absent."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                    f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=30) as r:
                meta = json.load(r)
            with urllib.request.urlopen(meta[0]["pdbUrl"], timeout=60) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None            # genuinely not in the database, do not retry
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def parse_ca(pdb):
    """Ca coordinates, residue type indices and per-residue pLDDT, in chain order."""
    coords, types, plddt = [], [], []
    for line in pdb.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        residue = line[17:20].strip()
        if residue not in THREE2ONE:
            continue
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        types.append(AA_IDX[THREE2ONE[residue]])
        plddt.append(float(line[60:66]))
    if not coords:
        return None
    return np.asarray(coords, np.float32), np.asarray(types, np.int8), np.asarray(plddt, np.float32)


def one(row):
    """(accession, family, coords, types, plddt) or a string explaining the rejection."""
    accession, sequence, family = row
    pdb = fetch(accession)
    if pdb is None:
        return "absent"
    parsed = parse_ca(pdb)
    if parsed is None:
        return "no_ca"
    coords, types, plddt = parsed
    # AlphaFold models the full UniProt sequence. A length disagreement means the dataset row and
    # the model are different sequences, so the labels would not describe the structure.
    if len(coords) != len(sequence):
        return "length_mismatch"
    if float(plddt.mean()) < MIN_PLDDT:
        return "low_confidence"
    return (accession, family, coords, types, plddt)


def build_split(split, limit=None):
    sub = load_subset(split, limit)
    rows = list(zip(sub["id"], sub["seq"], sub["family"]))
    os.makedirs(f"{OUT}/{split}", exist_ok=True)
    n_shards = (len(rows) + SHARD_SIZE - 1) // SHARD_SIZE
    started = time.perf_counter()
    for shard in range(n_shards):
        path = f"{OUT}/{split}/shard_{shard:04d}.npz"
        if os.path.exists(path):
            continue
        chunk = rows[shard * SHARD_SIZE:(shard + 1) * SHARD_SIZE]
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(one, chunk))
        good = [r for r in results if not isinstance(r, str)]
        reasons = {}
        for r in results:
            if isinstance(r, str):
                reasons[r] = reasons.get(r, 0) + 1
        # Pad into the dense layout the trainer consumes.
        n = len(good)
        coords = np.zeros((n, MAX_RES, 3), np.float32)
        types = np.full((n, MAX_RES), -1, np.int8)
        plddt = np.zeros((n, MAX_RES), np.float32)
        lengths = np.zeros(n, np.int16)
        for i, (_, _, c, t, p) in enumerate(good):
            k = len(c)
            coords[i, :k], types[i, :k], plddt[i, :k], lengths[i] = c, t, p, k
        np.savez_compressed(path, coords=coords, types=types, plddt=plddt, lengths=lengths,
                            accessions=np.array([g[0] for g in good]),
                            families=np.array([g[1] for g in good]))
        done = shard + 1
        rate = (done * SHARD_SIZE) / (time.perf_counter() - started)
        left = (n_shards - done) * SHARD_SIZE / max(rate, 1e-9)
        print(f"[{split}] shard {done}/{n_shards}: kept {n}/{len(chunk)} "
              f"{reasons if reasons else ''} | {rate:.1f} struct/s | {left/60:.0f} min left",
              flush=True)
    # One packed file per split.
    shards = sorted(f"{OUT}/{split}/{f}" for f in os.listdir(f"{OUT}/{split}")
                    if f.startswith("shard_"))
    parts = [np.load(s, allow_pickle=False) for s in shards]
    packed = {k: np.concatenate([p[k] for p in parts]) for k in
              ("coords", "types", "plddt", "lengths", "accessions", "families")}
    np.savez_compressed(f"{OUT}/{split}_ca.npz", **packed)
    print(f"[{split}] packed {len(packed['lengths']):,} structures -> {OUT}/{split}_ca.npz")
    return len(packed["lengths"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    total = {s: build_split(s, args.limit) for s in args.splits}
    print("\ndone:", total)
