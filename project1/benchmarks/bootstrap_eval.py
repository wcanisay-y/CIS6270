# bootstrap confidence intervals for the sample-quality metrics of the qm9 benchmark.
# per-molecule quantities are computed once with the shared evaluator's functions;
# resampling then works on those indicators only (never on the molecules themselves), so
# resampling with replacement cannot create spurious duplicate smiles.

import csv
import glob
import json
import os
import pickle
import sys

import numpy as np

BENCH = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BENCH, "results")
sys.path.insert(0, BENCH)
from unified_eval import (DATASET_INFO, check_stability, load_generated, load_reference,  # noqa: E402
                          reference_smiles, to_smiles)

N_BOOT, SEED, CI = 1000, 0, (2.5, 97.5)
SAMPLES = {
    "EDM": ("edm/outputs/edm_qm9/eval/analyzed_molecules", "edm/qm9/temp/qm9_smiles.pickle"),
    "GCDM": ("gcdm/output/QM9/Unconditional/gcdm_model_1", "gcdm/data/EDM/QM9/QM9_smiles.pickle"),
}


def per_molecule(mols):
    rows = []
    for pos, types in mols:
        mol_stable, stable_atoms, n_atoms = check_stability(pos, types, DATASET_INFO)
        rows.append((n_atoms, stable_atoms, int(mol_stable), to_smiles((pos, types))))
    return rows


def indicators(rows, shared_ref, native_ref):
    n = len(rows)
    n_atoms = np.array([r[0] for r in rows], dtype=float)
    stable_atoms = np.array([r[1] for r in rows], dtype=float)
    mol_stable = np.array([r[2] for r in rows], dtype=float)
    valid = np.array([r[3] is not None for r in rows], dtype=float)
    first = np.zeros(n)          # first occurrence of a valid smiles == one unique molecule
    novel_shared = np.zeros(n)
    novel_native = np.zeros(n)
    seen = set()
    for i, r in enumerate(rows):
        s = r[3]
        if s is None or s in seen:
            continue
        seen.add(s)
        first[i] = 1
        novel_shared[i] = float(s not in shared_ref)
        novel_native[i] = float(native_ref is not None and s not in native_ref)
    return dict(n_atoms=n_atoms, stable_atoms=stable_atoms, mol_stable=mol_stable, valid=valid,
                first=first, novel_shared=novel_shared, novel_native=novel_native)


def metrics(ind, idx):
    n = len(idx)
    n_valid = ind["valid"][idx].sum()
    n_unique = ind["first"][idx].sum()
    out = {
        "atom_stability": ind["stable_atoms"][idx].sum() / ind["n_atoms"][idx].sum(),
        "molecule_stability": ind["mol_stable"][idx].mean(),
        "validity": n_valid / n,
        "uniqueness": n_unique / max(n_valid, 1),
        "valid_and_unique": n_unique / n,
        "novelty": ind["novel_shared"][idx].sum() / max(n_unique, 1),
        "novelty_native": ind["novel_native"][idx].sum() / max(n_unique, 1),
    }
    return out


def bootstrap(ind, rng):
    n = len(ind["valid"])
    point = metrics(ind, np.arange(n))
    draws = {k: [] for k in point}
    for _ in range(N_BOOT):
        m = metrics(ind, rng.integers(0, n, size=n))
        for k, v in m.items():
            draws[k].append(v)
    return {k: {"value": float(point[k]),
                "lo": float(np.percentile(draws[k], CI[0])),
                "hi": float(np.percentile(draws[k], CI[1])),
                "se": float(np.std(draws[k], ddof=1))} for k in point}


def main():
    rng = np.random.default_rng(SEED)
    shared_ref = reference_smiles(os.path.join(RES, "qm9_parquet_smiles.json"))
    out = {"n_boot": N_BOOT, "seed": SEED, "ci_percentiles": CI, "n_reference_shared": len(shared_ref)}

    for name, (sample_dir, pickle_path) in SAMPLES.items():
        with open(os.path.join(BENCH, pickle_path), "rb") as f:
            native_ref = set(pickle.load(f))
        rows = per_molecule(load_generated(os.path.join(BENCH, sample_dir)))
        with open(os.path.join(RES, f"permol_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_atoms", "stable_atoms", "mol_stable", "smiles"])
            w.writerows(rows)
        ind = indicators(rows, shared_ref, native_ref)
        out[name] = bootstrap(ind, rng)
        out[name]["n_reference_native"] = len(native_ref)
        print(name, {k: (round(v["value"], 4), round(v["lo"], 4), round(v["hi"], 4))
                     for k, v in out[name].items() if isinstance(v, dict)}, flush=True)

    rows = per_molecule(load_reference(10000))
    ind = indicators(rows, set(), None)
    out["QM9 data"] = bootstrap(ind, rng)
    for k in ("novelty", "novelty_native"):
        out["QM9 data"][k] = None
    print("QM9 data", {k: (round(v["value"], 4), round(v["lo"], 4), round(v["hi"], 4))
                       for k, v in out["QM9 data"].items() if isinstance(v, dict)}, flush=True)

    # likelihood: gcdm logged five test passes; edm logged one
    passes = []
    metrics_csv = sorted(glob.glob(os.path.join(BENCH, "gcdm/logs/mol_gen_eval/runs/*/csv/version_0/metrics.csv")))[-1]
    with open(metrics_csv) as f:
        for r in csv.DictReader(f):
            if r.get("test/loss"):
                passes.append(float(r["test/loss"]))
    out["nll"] = {"GCDM": {"passes": passes, "mean": float(np.mean(passes)), "min": min(passes),
                           "max": max(passes), "std": float(np.std(passes, ddof=1))}}
    import re
    edm_log = open(os.path.join(RES, "edm_eval.log"), errors="ignore").read()
    edm_nll = float(re.findall(r"Final test nll (-?[\d.]+)", edm_log)[-1])
    out["nll"]["EDM"] = {"passes": [edm_nll], "mean": edm_nll, "min": edm_nll, "max": edm_nll, "std": None}

    with open(os.path.join(RES, "bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES, "bootstrap_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "value", "ci_lo", "ci_hi", "bootstrap_se"])
        for name in ("EDM", "GCDM", "QM9 data"):
            for k, v in out[name].items():
                if isinstance(v, dict):
                    w.writerow([name, k, f"{v['value']:.4f}", f"{v['lo']:.4f}", f"{v['hi']:.4f}", f"{v['se']:.4f}"])
    print("nll:", out["nll"])


if __name__ == "__main__":
    main()
