"""Download the deposited structure of each CATH S35 pattern source.

Needed for Ref-TM: the conditioning experiment hands the model a real domain's
chemical pattern, so for the CATH arm -- and only for that arm -- the protein
whose pattern was used has a crystal structure. That makes the decisive
question answerable: does a conditioned sequence adopt its SOURCE's fold, or
merely some fold ESMFold is confident about?

SwissProt sources cannot be done this way; they are full-length UniProt
records with no deposited structure guaranteed, which is a second reason the
CATH replication is the better-controlled of the two experiments.

Files land in their own directory and are parsed by our own code; nothing from
the download is executed.

Usage:
    python -m tools.fetch_cath_domains --out data/cath_domains
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API = "http://www.cathdb.info/version/v4_4_0/api/rest/id/%s.pdb"
UA = "protein-structure-benchmark/1.0"
ID_RE = re.compile(r"cath\|[\d_]+\|(\w+)/")


def domain_ids(ids_file, patterns_file):
    """The CATH domain id behind each held-out pattern, in pattern order."""
    rows = [l.rstrip("\n").split("\t") for l in open(ids_file)]
    names = [r[1] if len(r) > 1 else r[0] for r in rows]
    with open(patterns_file) as fh:
        pf = json.load(fh)
    out = []
    for i in pf["dev_indices"]:
        m = ID_RE.search(names[i])
        out.append(m.group(1) if m else None)
    return out, pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/clustered_cath/devcand_ids.txt")
    ap.add_argument("--patterns",
                    default="data/clustered_cath/heldout_patterns.json")
    ap.add_argument("--out", default="data/cath_domains")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    doms, pf = domain_ids(args.ids, args.patterns)

    # the pattern pool is rebuilt exactly as hier/run_generate.py builds it,
    # so index i here is index i of every cell's generated sequences
    keep = [(d, s) for d, s in zip(doms, pf["seqs"])
            if len(s) >= args.length]
    print(f"{len(keep)} of {len(doms)} sources are >= {args.length} long",
          flush=True)

    ok, miss = 0, []
    for i, (dom, _) in enumerate(keep):
        if dom is None:
            miss.append(i)
            continue
        dst = os.path.join(args.out, dom + ".pdb")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            ok += 1
            continue
        req = urllib.request.Request(API % dom, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            if b"ATOM" not in body:
                raise ValueError("no ATOM records")
            with open(dst, "wb") as fh:
                fh.write(body)
            ok += 1
        except Exception as e:                                # noqa: BLE001
            print(f"  {dom}: {type(e).__name__}: {e}", flush=True)
            miss.append(i)
        time.sleep(args.sleep)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(keep)}", flush=True)

    index = {"domains": [d for d, _ in keep], "n_ok": ok, "missing": miss,
             "length": args.length}
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"downloaded {ok}, missing {len(miss)}; index written", flush=True)


if __name__ == "__main__":
    main()
