"""Oracle contact maps from EXPERIMENTAL structures, not from ESMFold.

Why this exists. The first oracle run took its "true" contact map from
ESMFold coordinates. ESMFold's trunk is ESM-2, the contact head being
tested is ESM-2's, and the quality metric is ESMFold pLDDT, so the whole
argument ran inside one model family. This module removes that circularity
by building the map from deposited X-ray coordinates.

It also fixes an atom-indexing error in the earlier run. ESMFold's
`positions` output is in atom14 order, which is
    N, CA, C, O, CB, ...
so index 3 is the backbone OXYGEN and CB is index 4. The earlier oracle
took index 3, so what it called a CB-CB map was an O-O map. Glycine has no
atom14 CB at all, which is why a virtual CB is needed for both sources.

Selection criteria, all enforced here:
  X-ray, resolution <= 2.0 A, one polymer instance in the deposit,
  protein only, a contiguous stretch of L standard residues with no chain
  break (consecutive label_seq_id AND consecutive CA-CA < 4.5 A), and
  sequence-level deduplication against everything already accepted.

Downloaded mmCIF files are untrusted input: they are parsed as text and
never executed, and they are written to their own directory.
"""
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CIF_URL = "https://files.rcsb.org/download/%s.cif"

# A plain client name. Never send a user's address to a third-party service.
UA = {"User-Agent": "protein-mdm-study/1.0"}


# --------------------------------------------------------------- geometry
def virtual_cb(n, ca, c):
    """Standard virtual CB from backbone N, CA, C (trRosetta/AF convention)."""
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return (-0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca)


def cb_contacts(cb, thresh=8.0, min_sep=6):
    """Binary contact map from CB coordinates, |i-j| >= min_sep."""
    d = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    c = (d < thresh).astype(np.float32)
    n = c.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    c[np.abs(ii - jj) < min_sep] = 0.0
    return c


# ------------------------------------------------------------ mmCIF parse
def _atom_site_rows(text):
    """Yield dicts for the _atom_site loop of an mmCIF file.

    Minimal on purpose: the atom_site loop in RCSB-distributed mmCIF has one
    record per line with no embedded whitespace in the fields we read.
    """
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            cols = []
            j = i + 1
            while j < n and lines[j].lstrip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if cols and cols[0].startswith("_atom_site."):
                idx = {c.split(".", 1)[1]: k for k, c in enumerate(cols)}
                while j < n:
                    ln = lines[j]
                    s = ln.strip()
                    if not s or s.startswith("#") or s.startswith("loop_") \
                            or s.startswith("_") or s.startswith("data_"):
                        break
                    f = s.split()
                    if len(f) >= len(cols):
                        yield idx, f
                    j += 1
                return
            i = j
        else:
            i += 1


def parse_backbone(text):
    """-> {asym_id: {seq_id: {"comp": str, "N"/"CA"/"C"/"CB": xyz}}}"""
    chains = {}
    for idx, f in _atom_site_rows(text):
        try:
            if f[idx["group_PDB"]] != "ATOM":
                continue
            if "pdbx_PDB_model_num" in idx and f[idx["pdbx_PDB_model_num"]] != "1":
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
            asym = f[idx["label_asym_id"]]
            comp = f[idx["label_comp_id"]]
            xyz = np.array([float(f[idx["Cartn_x"]]),
                            float(f[idx["Cartn_y"]]),
                            float(f[idx["Cartn_z"]])], dtype=np.float64)
        except (KeyError, IndexError, ValueError):
            continue
        res = chains.setdefault(asym, {}).setdefault(int(seq), {"comp": comp})
        res.setdefault(atom, xyz)
    return chains


def best_window(chain, L, max_ca_gap=4.5):
    """Longest break-free run of standard residues; return a window of L.

    Requires consecutive label_seq_id, N/CA/C present (CB too unless GLY),
    and consecutive CA-CA distance below max_ca_gap so that numbering gaps
    which hide a real break are caught.
    """
    keys = sorted(chain)
    ok = []
    for k in keys:
        r = chain[k]
        if r["comp"] not in AA3TO1:
            continue
        if not all(a in r for a in ("N", "CA", "C")):
            continue
        if r["comp"] != "GLY" and "CB" not in r:
            continue
        ok.append(k)
    runs, cur = [], []
    for k in ok:
        if cur and k == cur[-1] + 1:
            d = np.linalg.norm(chain[k]["CA"] - chain[cur[-1]]["CA"])
            if d > max_ca_gap:
                runs.append(cur)
                cur = [k]
                continue
            cur.append(k)
        else:
            if cur:
                runs.append(cur)
            cur = [k]
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= L]
    if not runs:
        return None
    run = max(runs, key=len)
    off = (len(run) - L) // 2
    return run[off:off + L]


def chain_to_arrays(chain, window):
    """-> (seq, cb, ca, n_gly_virtual)"""
    seq, cb, ca, ngly = [], [], [], 0
    for k in window:
        r = chain[k]
        seq.append(AA3TO1[r["comp"]])
        ca.append(r["CA"])
        if r["comp"] == "GLY" or "CB" not in r:
            cb.append(virtual_cb(r["N"], r["CA"], r["C"]))
            ngly += 1
        else:
            cb.append(r["CB"])
    return "".join(seq), np.stack(cb), np.stack(ca), ngly


# ------------------------------------------------------------------ fetch
def search_entries(rows=600, res_max=2.0, lo=210, hi=500):
    q = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "exptl.method", "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": res_max}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute":
                    "rcsb_entry_info.deposited_polymer_entity_instance_count",
                "operator": "equals", "value": 1}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_monomer_count_maximum",
                "operator": "range", "value": {"from": lo, "to": hi}}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute":
                    "rcsb_entry_info.selected_polymer_entity_types",
                "operator": "exact_match", "value": "Protein (only)"}},
        ]},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                      "direction": "asc"}]},
    }
    req = urllib.request.Request(
        SEARCH_URL, data=json.dumps(q).encode(),
        headers={"Content-Type": "application/json", **UA})
    with urllib.request.urlopen(req, timeout=120) as fh:
        d = json.loads(fh.read().decode())
    return [x["identifier"] for x in d.get("result_set", [])]


def fetch_cif(pdb_id, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, "%s.cif" % pdb_id.upper())
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    req = urllib.request.Request(CIF_URL % pdb_id.upper(), headers=UA)
    with urllib.request.urlopen(req, timeout=120) as fh:
        data = fh.read()
    with open(p, "wb") as out:
        out.write(data)
    return p


def _ident(a, b):
    """Fraction of identical positions over the shorter length (ungapped)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def build(n_want=48, L=200, cache_dir=None, out=None, max_ident=0.30,
          rows=600, verbose=True):
    """Collect n_want non-redundant experimental maps of length L."""
    cache_dir = cache_dir or "data/pdb_cif"
    ids = search_entries(rows=rows)
    kept = {"pdb": [], "seq": [], "cb": [], "ca": [], "ngly": []}
    tried = failed = 0
    for pid in ids:
        if len(kept["pdb"]) >= n_want:
            break
        tried += 1
        try:
            path = fetch_cif(pid, cache_dir)
            with open(path, "r", errors="replace") as fh:
                text = fh.read()
            chains = parse_backbone(text)
        except Exception as e:
            failed += 1
            if verbose:
                print("  %s download/parse failed: %s" % (pid, e), flush=True)
            continue
        got = None
        for asym in sorted(chains):
            w = best_window(chains[asym], L)
            if w is not None:
                got = (asym, w)
                break
        if got is None:
            continue
        seq, cb, ca, ngly = chain_to_arrays(chains[got[0]], got[1])
        if len(seq) != L:
            continue
        if any(_ident(seq, s) > max_ident for s in kept["seq"]):
            continue
        kept["pdb"].append("%s_%s" % (pid, got[0]))
        kept["seq"].append(seq)
        kept["cb"].append(cb)
        kept["ca"].append(ca)
        kept["ngly"].append(ngly)
        if verbose:
            m = cb_contacts(cb)
            print("  %-8s %s density %.4f gly_virtual %d  (%d/%d)"
                  % (kept["pdb"][-1], "len %d" % L, m.mean(), ngly,
                     len(kept["pdb"]), n_want), flush=True)
    maps = np.stack([cb_contacts(c) for c in kept["cb"]])
    ca_maps = np.stack([cb_contacts(c) for c in kept["ca"]])
    if verbose:
        print("tried %d entries, %d download/parse failures, kept %d"
              % (tried, failed, len(kept["pdb"])), flush=True)
        print("CB-CB density %.4f   CA-CA density %.4f"
              % (maps.mean(), ca_maps.mean()), flush=True)
    if out:
        np.savez_compressed(
            out, maps=maps, ca_maps=ca_maps,
            seqs=np.array(kept["seq"]), pdb=np.array(kept["pdb"]),
            cb=np.stack(kept["cb"]), ca=np.stack(kept["ca"]),
            ngly=np.array(kept["ngly"]))
        if verbose:
            print("wrote %s" % out, flush=True)
    return kept, maps


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--cache", default="data/pdb_cif")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(n_want=a.n, L=a.length, cache_dir=a.cache, out=a.out, rows=a.rows)
