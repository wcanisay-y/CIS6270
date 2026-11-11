"""Build the oracle's ground truth from CATH, the field-standard structural set.

Why this exists. The first oracle drew 48 chains from an ad-hoc RCSB query
(X-ray, <=2.0 A, pairwise sequence identity < 0.30). That is reasonable but
it is not a recognised benchmark, and sequence identity below 30 percent
does NOT imply fold-level non-redundancy: two chains can share a CATH
topology while being 20 percent identical, so an "independent" set can be
quietly dominated by a few popular folds.

CATH fixes both problems. We take the official S40 non-redundant set and
keep ONE chain per distinct CATH topology (C.A.T), which is non-redundancy
at the fold level, and we stratify across all four CATH classes so that
mainly-alpha, mainly-beta, alpha-beta and few-secondary-structure folds are
all represented. The class labels also let the oracle result be reported per
fold class, which the ad-hoc set could not do: beta-rich folds carry far
more long-range contacts, so if contact gating helps anywhere it should help
there, and that is a test the earlier set could not run.

Only chains whose CATH domains all share one topology are used, so the
label of a 200-residue window is unambiguous.

CLF 2.0 columns: domain C A T H S O L I D length resolution
Domain id 1oaiA00 -> pdb 1oai, author chain A, domain 00.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel.pdb_oracle import (UA, cb_contacts, _atom_site_rows,
                                 best_window, chain_to_arrays, fetch_cif,
                                 _ident)

CLASS_NAME = {1: "mainly-alpha", 2: "mainly-beta", 3: "alpha-beta",
              4: "few-secondary-structures"}


def parse_backbone_auth(text):
    """Like pdb_oracle.parse_backbone but keyed on the AUTHOR chain id.

    CATH identifies chains by the author id, not label_asym_id, so the
    oracle must index the same way or the domain's chain will not be found.
    """
    chains = {}
    for idx, f in _atom_site_rows(text):
        try:
            if f[idx["group_PDB"]] != "ATOM":
                continue
            if "pdbx_PDB_model_num" in idx and \
                    f[idx["pdbx_PDB_model_num"]] != "1":
                continue
            alt = f[idx["label_alt_id"]] if "label_alt_id" in idx else "."
            if alt not in (".", "?", "A"):
                continue
            atom = f[idx["label_atom_id"]].strip('"')
            if atom not in ("N", "CA", "C", "CB"):
                continue
            seq = f[idx["label_seq_id"]]
            if seq in (".", "?"):
                continue
            key = "auth_asym_id" if "auth_asym_id" in idx else "label_asym_id"
            asym = f[idx[key]]
            comp = f[idx["label_comp_id"]]
            xyz = np.array([float(f[idx["Cartn_x"]]),
                            float(f[idx["Cartn_y"]]),
                            float(f[idx["Cartn_z"]])], dtype=np.float64)
        except (KeyError, IndexError, ValueError):
            continue
        res = chains.setdefault(asym, {}).setdefault(int(seq),
                                                     {"comp": comp})
        res.setdefault(atom, xyz)
    return chains


def load_cath(dom_list, s40_list):
    s40 = set()
    with open(s40_list) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                s40.add(ln)
    doms = []
    with open(dom_list) as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            f = ln.split()
            if len(f) < 12:
                continue
            d = f[0]
            try:
                doms.append(dict(dom=d, pdb=d[:4], chain=d[4],
                                 C=int(f[1]), A=int(f[2]), T=int(f[3]),
                                 H=int(f[4]), length=int(f[10]),
                                 res=float(f[11]), s40=d in s40))
            except ValueError:
                continue
    return doms


def select_chains(doms, n_want, res_max=2.5, min_dom_len=60):
    """One chain per CATH topology, single-topology chains only, S40 members,
    stratified round-robin across the four CATH classes."""
    by_chain = {}
    for d in doms:
        by_chain.setdefault((d["pdb"], d["chain"]), []).append(d)
    cands = []
    for (pdb, ch), ds in by_chain.items():
        tops = {(d["C"], d["A"], d["T"]) for d in ds}
        if len(tops) != 1:
            continue                      # multi-fold chain: ambiguous label
        if not any(d["s40"] for d in ds):
            continue                      # not in the non-redundant set
        res = min(d["res"] for d in ds)
        if not (0 < res <= res_max):
            continue                      # NMR/EM rows carry 999.0
        if max(d["length"] for d in ds) < min_dom_len:
            continue
        cands.append(dict(pdb=pdb, chain=ch, topo=tops.pop(), res=res,
                          tot=sum(d["length"] for d in ds)))
    # best resolution first within each class, then round-robin by class
    per_class = {c: [] for c in (1, 2, 3, 4)}
    seen_topo = set()
    for c in sorted(cands, key=lambda x: (x["res"], -x["tot"])):
        t = c["topo"]
        if t in seen_topo:
            continue
        seen_topo.add(t)
        if t[0] in per_class:
            per_class[t[0]].append(c)
    out, i = [], 0
    while len(out) < n_want * 4 and any(per_class.values()):
        for c in (1, 2, 3, 4):
            if per_class[c]:
                out.append(per_class[c].pop(0))
        i += 1
        if i > 10000:
            break
    return out


def build(dom_list, s40_list, n_want=48, L=200, cache_dir="data/pdb_cif",
          out=None, max_ident=0.30, res_max=2.5, verbose=True):
    doms = load_cath(dom_list, s40_list)
    if verbose:
        print("CATH: %d domains, %d in S40"
              % (len(doms), sum(d["s40"] for d in doms)), flush=True)
    cands = select_chains(doms, n_want, res_max=res_max)
    if verbose:
        print("candidate single-topology S40 chains: %d across %d topologies"
              % (len(cands), len({c["topo"] for c in cands})), flush=True)
    keep, seqs, maps, labels, nfail = [], [], [], [], 0
    gly_tot = 0
    for c in cands:
        if len(keep) >= n_want:
            break
        try:
            p = fetch_cif(c["pdb"], cache_dir)
            text = open(p).read()
            chains = parse_backbone_auth(text)
            ch = chains.get(c["chain"])
            if ch is None:
                nfail += 1
                continue
            w = best_window(ch, L)
            if w is None:
                nfail += 1
                continue
            seq, cb, ca, ngly = chain_to_arrays(ch, w)
            if len(seq) != L:
                nfail += 1
                continue
            if any(_ident(seq, s) >= max_ident for s in seqs):
                continue
            m = cb_contacts(cb)
            keep.append(c)
            seqs.append(seq)
            maps.append(m)
            gly_tot += ngly
            labels.append((c["topo"][0], c["topo"][1], c["topo"][2]))
            if verbose:
                print("  [%2d] %s%s  %-24s res %.2f  contacts/res %.2f"
                      % (len(keep), c["pdb"], c["chain"],
                         CLASS_NAME[c["topo"][0]], c["res"],
                         m.sum() / L), flush=True)
        except Exception as e:                      # noqa: BLE001
            nfail += 1
            if verbose and nfail < 10:
                print("  skip %s%s: %s" % (c["pdb"], c["chain"], e),
                      flush=True)
    maps = np.stack(maps).astype(np.float32)
    lab = np.array(labels, dtype=np.int32)
    if verbose:
        print("kept %d maps, density %.4f, %d failures, %d virtual Gly CB"
              % (len(keep), float(maps.mean()), nfail, gly_tot), flush=True)
        for c in (1, 2, 3, 4):
            n = int((lab[:, 0] == c).sum())
            if n:
                d = float(maps[lab[:, 0] == c].mean())
                print("   class %d %-26s n=%2d  contact density %.4f"
                      % (c, CLASS_NAME[c], n, d), flush=True)
    if out:
        np.savez_compressed(
            out, maps=maps, seqs=np.array(seqs),
            pdb=np.array(["%s%s" % (c["pdb"], c["chain"]) for c in keep]),
            cath=lab, res=np.array([c["res"] for c in keep]))
        print("wrote %s" % out, flush=True)
    return seqs, maps, lab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom-list",
                    default="data/cath_raw/cath-domain-list.txt")
    ap.add_argument("--s40",
                    default="data/cath_raw/cath-dataset-nonredundant-S40.list")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--res-max", type=float, default=2.5)
    ap.add_argument("--cache", default="data/pdb_cif")
    ap.add_argument("--out", default="results/A_oracle_cath.npz")
    a = ap.parse_args()
    build(a.dom_list, a.s40, n_want=a.n, L=a.length, cache_dir=a.cache,
          out=a.out, res_max=a.res_max)
