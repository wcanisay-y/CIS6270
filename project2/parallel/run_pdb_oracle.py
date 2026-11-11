"""Redo the oracle and the contact-quality table against EXPERIMENTAL structures.

Three things the original study could not do, because its ground truth came
from ESMFold and its "CB" was really the backbone oxygen:

  1 agreement  How close is an ESMFold-derived contact map to the deposited
               one, for the same protein? This is the size of the
               circularity the original argument carried.
  2 quality    ESM-2 contact precision against mask rate, scored against
               the deposited map instead of a predicted one.
  3 maps       Emit three oracle files over the SAME 48 sequences:
                 pdb      deposited CB-CB  (new ground truth)
                 esmfold  ESMFold CB-CB    (correct atom index)
                 oxygen   ESMFold O-O      (reproduces the earlier error)
               so the sampling grid can be run on each and the two
               confounds separated.

Needs a GPU. Run via sbatch on gpuq.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel.pdb_oracle import cb_contacts
from parallel.oracle import _auroc


def _prec_at_L(score, label, L, min_sep=6):
    iu = np.triu_indices(label.shape[0], k=min_sep)
    y = label[iu] > 0.5
    s = score[iu]
    return float(y[np.argsort(-s)[:L]].mean())


def _jaccard(a, b, min_sep=6):
    iu = np.triu_indices(a.shape[0], k=min_sep)
    x, y = a[iu] > 0.5, b[iu] > 0.5
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-file", default="results/A_pdb_oracle_maps.npz")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--quality-n", type=int, default=12)
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 0.9])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(a.pdb_file)
    seqs = [str(s) for s in z["seqs"]]
    pdb_maps = z["maps"]
    pdb_ids = [str(x) for x in z["pdb"]]
    L = pdb_maps.shape[-1]
    print("loaded %d experimental maps, L=%d, density %.4f"
          % (len(seqs), L, pdb_maps.mean()), flush=True)

    # ---------------------------------------------------------- 1. fold
    from common import metrics as M
    fold = M.ESMFoldScorer(device=dev)
    r = fold.fold(seqs, batch_size=a.batch)
    esm_maps = np.stack([cb_contacts(c) for c in r["cb"]])
    oxy_maps = np.stack([cb_contacts(c) for c in r["o_atom"]])
    plddt = np.asarray(r["plddt"])
    del fold
    torch.cuda.empty_cache()
    print("ESMFold on the same sequences: pLDDT mean %.2f min %.2f"
          % (plddt.mean(), plddt.min()), flush=True)
    print("density  pdb %.4f  esmfold %.4f  oxygen %.4f"
          % (pdb_maps.mean(), esm_maps.mean(), oxy_maps.mean()), flush=True)

    # --------------------------------------------------- 2. map agreement
    rows = []
    for i in range(len(seqs)):
        rows.append(dict(
            pdb=pdb_ids[i], plddt=float(plddt[i]),
            esm_prec=_prec_at_L(esm_maps[i], pdb_maps[i], L),
            oxy_prec=_prec_at_L(oxy_maps[i], pdb_maps[i], L),
            esm_jac=_jaccard(esm_maps[i], pdb_maps[i]),
            oxy_jac=_jaccard(oxy_maps[i], pdb_maps[i]),
            esm_oxy_jac=_jaccard(esm_maps[i], oxy_maps[i])))
    agree = {k: float(np.mean([x[k] for x in rows]))
             for k in ("plddt", "esm_prec", "oxy_prec", "esm_jac",
                       "oxy_jac", "esm_oxy_jac")}
    print("AGREEMENT vs deposited: ESMFold CB P@L %.3f Jaccard %.3f | "
          "oxygen P@L %.3f Jaccard %.3f | ESMFold-vs-oxygen Jaccard %.3f"
          % (agree["esm_prec"], agree["esm_jac"], agree["oxy_prec"],
             agree["oxy_jac"], agree["esm_oxy_jac"]), flush=True)

    # ------------------------------------- 3. contact quality vs mask rate
    from transformers import AutoTokenizer, EsmForMaskedLM
    from parallel.contacts import ExternalContacts, SelfContacts

    tok = AutoTokenizer.from_pretrained("airkingbd/dplm_150m")
    dplm = EsmForMaskedLM.from_pretrained(
        "airkingbd/dplm_150m", attn_implementation="eager").eval().to(dev)
    ext = ExternalContacts(650, device=dev, half=False)
    slf = SelfContacts(150, device=dev)

    qn = min(a.quality_n, len(seqs))
    qrows = []

    null = torch.full((1, L + 2), tok.mask_token_id, dtype=torch.long,
                      device=dev)
    null[0, 0] = tok.cls_token_id
    null[0, -1] = tok.eos_token_id
    c_ne = ext(null)[0].float().cpu().numpy()
    o_n = dplm(input_ids=null, output_attentions=True)
    c_ns = slf.from_attentions(null, o_n.attentions)[0].float().cpu().numpy()
    iu = np.triu_indices(L, k=6)
    nn = []
    for i in range(qn):
        y = pdb_maps[i][iu] > 0.5
        nn.append((_auroc(c_ne[iu], y), _auroc(c_ns[iu], y),
                   float(y[np.argsort(-c_ne[iu])[:L]].mean()),
                   float(y[np.argsort(-c_ns[iu])[:L]].mean())))
    qrows.append(dict(frac=-1.0, note="all_mask_null",
                      auroc_ext=float(np.mean([x[0] for x in nn])),
                      auroc_self=float(np.mean([x[1] for x in nn])),
                      prec_L_ext=float(np.mean([x[2] for x in nn])),
                      prec_L_self=float(np.mean([x[3] for x in nn]))))
    print("ALL-MASK NULL  AUROC ext %.3f self %.3f | P@L ext %.3f self %.3f"
          % (qrows[0]["auroc_ext"], qrows[0]["auroc_self"],
             qrows[0]["prec_L_ext"], qrows[0]["prec_L_self"]), flush=True)

    for frac in a.fracs:
        acc = []
        for i in range(qn):
            ids = tok(seqs[i], return_tensors="pt")["input_ids"].to(dev)
            g = torch.Generator().manual_seed(i)
            nm = int(round(frac * L))
            idx = torch.randperm(L, generator=g)[:nm] + 1
            m = ids.clone()
            m[0, idx] = tok.mask_token_id
            ce = ext(m)[0].float().cpu().numpy()
            oo = dplm(input_ids=m, output_attentions=True)
            cs = slf.from_attentions(m, oo.attentions)[0].float().cpu().numpy()
            y = pdb_maps[i][iu] > 0.5
            acc.append((_auroc(ce[iu], y), _auroc(cs[iu], y),
                        float(y[np.argsort(-ce[iu])[:L]].mean()),
                        float(y[np.argsort(-cs[iu])[:L]].mean())))
        qrows.append(dict(frac=float(frac),
                          auroc_ext=float(np.mean([x[0] for x in acc])),
                          auroc_self=float(np.mean([x[1] for x in acc])),
                          prec_L_ext=float(np.mean([x[2] for x in acc])),
                          prec_L_self=float(np.mean([x[3] for x in acc]))))
        print("mask %4.2f  AUROC ext %.3f self %.3f | P@L ext %.3f self %.3f"
              % (frac, qrows[-1]["auroc_ext"], qrows[-1]["auroc_self"],
                 qrows[-1]["prec_L_ext"], qrows[-1]["prec_L_self"]),
              flush=True)

    # ----------------------------------------------------- 4. emit maps
    os.makedirs(a.outdir, exist_ok=True)
    for name, mp in (("pdb", pdb_maps), ("esmfold", esm_maps),
                     ("oxygen", oxy_maps)):
        p = os.path.join(a.outdir, "A_oracle_%s.npz" % name)
        np.savez_compressed(p, maps=mp.astype(np.float32),
                            seqs=np.array(seqs), pdb=np.array(pdb_ids),
                            plddt=plddt)
        print("wrote %s  density %.4f" % (p, mp.mean()), flush=True)

    json.dump(dict(agreement=agree, per_protein=rows, quality=qrows,
                   natural_plddt=float(plddt.mean()),
                   density=dict(pdb=float(pdb_maps.mean()),
                                esmfold=float(esm_maps.mean()),
                                oxygen=float(oxy_maps.mean()))),
              open(os.path.join(a.outdir, "A_pdb_oracle_analysis.json"), "w"),
              indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
