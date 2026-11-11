"""SwissProt sequence corpus (local HF cache) with a length-filtered split."""
import json
import os
import numpy as np

# Plain-text cache of the HF SwissProt-Pfam parquet, built once by
# tools/cache_swissprot.py. Reading it needs no parquet engine, which keeps
# this module free of the pyarrow/numpy version coupling.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWISSPROT_TRAIN = os.path.join(_HERE, "data", "swissprot_train.txt")
SWISSPROT_DEV = os.path.join(_HERE, "data", "swissprot_dev.txt")

from .alphabet import AA20, AA_IDX


def load_swissprot(min_len=40, max_len=254, max_n=None, seed=0,
                   train_file=None, dev_file=None):
    """-> (train list[str], val list[str]).

    Keeps sequences of canonical residues only, within [min_len, max_len].
    The upstream dev split is used for validation so the split is not ours.

    `train_file` / `dev_file` override the default corpus paths. This is how
    the homology-purged corpus is fed in: the clustered-split experiment
    trains on data/clustered/train_purged.txt, which is the standard training
    set minus every BLAST neighbour of the held-out pattern proteins. The
    filtering and capping below are identical either way, so a purged run
    differs from the standard run only in which sequences exist.
    """
    ok = set(AA20)

    def prep(path, cap):
        with open(path) as f:
            s = f.read().split("\n")
        out = [x for x in s if min_len <= len(x) <= max_len and set(x) <= ok]
        if cap is not None and len(out) > cap:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(out), cap, replace=False)
            out = [out[i] for i in idx]
        return out

    tr = prep(train_file or SWISSPROT_TRAIN, max_n)
    va = prep(dev_file or SWISSPROT_DEV,
              2000 if max_n is None else min(2000, max_n // 10))
    return tr, va


def natural_aa_frequency(seqs):
    cnt = np.zeros(20)
    for s in seqs:
        for c in s:
            if c in AA_IDX:
                cnt[AA_IDX[c]] += 1
    return cnt / cnt.sum()


def build_reference(out_path, k=10, max_n=120000):
    """Cache the k-mer set and aa frequency of the training corpus."""
    tr, va = load_swissprot(max_n=max_n)
    km = set()
    for s in tr:
        for i in range(len(s) - k + 1):
            km.add(s[i:i + k])
    freq = natural_aa_frequency(tr)
    np.savez_compressed(out_path, freq=freq, n_train=len(tr), n_val=len(va))
    with open(out_path + ".kmers.txt", "w") as f:
        f.write("\n".join(sorted(km)))
    return len(tr), len(va), len(km)


def load_reference(path, k=10):
    z = np.load(path)
    with open(path + ".kmers.txt") as f:
        km = set(f.read().split("\n"))
    return z["freq"], km
