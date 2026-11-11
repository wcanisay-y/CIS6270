"""Oracle contact maps and contact-quality diagnostics for Idea A.

Two questions the main grid cannot separate:

  Q1  Would a CORRECT contact map help the selector at all?
      Answered by folding natural proteins, taking their true CB-CB contact
      maps, and running the greedy selector with those maps as the coupling.
      If even a correct map does not beat confidence-only selection, the
      mechanism itself is wrong and no improvement to contact prediction
      can rescue it.

  Q2  Can ESM-2 recover a contact map from a mostly-masked sequence?
      Answered by masking a natural sequence at a range of rates, predicting
      contacts from the masked input, and scoring those predictions against
      the true contacts of the full sequence (AUROC and precision@L).
      The selector needs this to work at HIGH mask fractions, because that
      is where the most positions are committed per step.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cb_contacts(cb, thresh=8.0, min_sep=6):
    """Binary contact map from CB coordinates, |i-j| >= min_sep."""
    d = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    c = (d < thresh).astype(np.float32)
    n = c.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    c[np.abs(ii - jj) < min_sep] = 0.0
    return c


def build_oracle(n, L, out, device="cuda", batch=8):
    """Fold n natural sequences of length L; save their true contact maps."""
    from common.data import load_swissprot
    from common import metrics as M
    tr, va = load_swissprot(max_n=40000)
    pool = [s for s in va if len(s) >= L] or [s for s in tr if len(s) >= L]
    rng = np.random.default_rng(0)
    pick = [pool[i][:L] for i in rng.choice(len(pool), min(n, len(pool)),
                                            replace=False)]
    fold = M.ESMFoldScorer(device=device)
    r = fold.fold(pick, batch_size=batch)
    maps = np.stack([cb_contacts(cb) for cb in r["cb"]])
    np.savez_compressed(out, maps=maps, seqs=np.array(pick),
                        plddt=r["plddt"])
    print(f"oracle: {maps.shape} maps, mean contact density "
          f"{maps.mean():.4f}, natural pLDDT {np.mean(r['plddt']):.2f} "
          f"-> {out}", flush=True)
    return maps


def _auroc(scores, labels):
    """AUROC without sklearn: rank-based (Mann-Whitney) estimator."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def run_quality(L=200, n=12, device="cuda", out=None,
                fracs=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0)):
    """Q2: contact-prediction quality as a function of how masked the input is."""
    from transformers import AutoTokenizer, EsmForMaskedLM
    from common.data import load_swissprot
    from common import metrics as M
    from parallel.contacts import ExternalContacts, SelfContacts

    tr, va = load_swissprot(max_n=40000)
    pool = [s for s in va if len(s) >= L] or [s for s in tr if len(s) >= L]
    rng = np.random.default_rng(1)
    seqs = [pool[i][:L] for i in rng.choice(len(pool), n, replace=False)]

    fold = M.ESMFoldScorer(device=device)
    r = fold.fold(seqs, batch_size=4)
    true = [cb_contacts(cb) for cb in r["cb"]]
    del fold
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained("airkingbd/dplm_150m")
    dplm = EsmForMaskedLM.from_pretrained(
        "airkingbd/dplm_150m", attn_implementation="eager").eval().to(device)
    ext = ExternalContacts(650, device=device, half=False)
    slf = SelfContacts(150, device=device)

    iu = np.triu_indices(L, k=6)
    rows = []

    # ALL-MASK NULL. Every residue masked, cls/eos kept. By construction
    # this map holds no sequence information, so it is the floor that any
    # masked-input prediction must beat. Reporting it next to every masked
    # number is one forward pass and it is the control that decides whether
    # a contact signal exists at all.
    null = torch.full((1, L + 2), tok.mask_token_id, dtype=torch.long,
                      device=device)
    null[0, 0] = tok.cls_token_id
    null[0, -1] = tok.eos_token_id
    c_null_ext = ext(null)[0].float().cpu().numpy()
    o_null = dplm(input_ids=null, output_attentions=True)
    c_null_self = slf.from_attentions(null, o_null.attentions)[0] \
        .float().cpu().numpy()
    nrows = []
    for si in range(len(seqs)):
        y = true[si][iu] > 0.5
        nrows.append(dict(
            auroc_ext=_auroc(c_null_ext[iu], y),
            auroc_self=_auroc(c_null_self[iu], y),
            prec_L_ext=float(y[np.argsort(-c_null_ext[iu])[:L]].mean()),
            prec_L_self=float(y[np.argsort(-c_null_self[iu])[:L]].mean())))
    print("ALL-MASK NULL   AUROC ext %.3f self %.3f | P@L ext %.3f self %.3f"
          % (np.mean([x["auroc_ext"] for x in nrows]),
             np.mean([x["auroc_self"] for x in nrows]),
             np.mean([x["prec_L_ext"] for x in nrows]),
             np.mean([x["prec_L_self"] for x in nrows])), flush=True)
    rows.append(dict(frac=-1.0, seq=-1, note="all_mask_null",
                     auroc_ext=float(np.mean([x["auroc_ext"] for x in nrows])),
                     auroc_self=float(np.mean([x["auroc_self"] for x in nrows])),
                     prec_L_ext=float(np.mean([x["prec_L_ext"] for x in nrows])),
                     prec_L_self=float(np.mean([x["prec_L_self"] for x in nrows])),
                     density=float(np.mean([(true[i][iu] > 0.5).mean()
                                            for i in range(len(seqs))]))))
    for frac in fracs:
        for si, s in enumerate(seqs):
            ids = tok(s, return_tensors="pt")["input_ids"].to(device)
            g = torch.Generator().manual_seed(si)
            nm = int(round(frac * L))
            idx = torch.randperm(L, generator=g)[:nm] + 1
            m = ids.clone()
            m[0, idx] = tok.mask_token_id
            with torch.no_grad():
                ce = ext(m)[0].float().cpu().numpy()
                o = dplm(input_ids=m, output_attentions=True)
                cs = slf.from_attentions(m, o.attentions)[0].float().cpu().numpy()
            y = true[si][iu] > 0.5
            rows.append(dict(frac=float(frac), seq=si,
                             auroc_ext=_auroc(ce[iu], y),
                             auroc_self=_auroc(cs[iu], y),
                             prec_L_ext=float(y[np.argsort(-ce[iu])[:L]].mean()),
                             prec_L_self=float(y[np.argsort(-cs[iu])[:L]].mean()),
                             density=float(y.mean())))
        g = [x for x in rows if x["frac"] == frac]
        if not g:
            continue
        print("mask %4.2f  AUROC ext %.3f self %.3f | P@L ext %.3f self %.3f "
              "| density %.4f"
              % (frac, np.mean([x["auroc_ext"] for x in g]),
                 np.mean([x["auroc_self"] for x in g]),
                 np.mean([x["prec_L_ext"] for x in g]),
                 np.mean([x["prec_L_self"] for x in g]),
                 np.mean([x["density"] for x in g])), flush=True)
    if out:
        json.dump(rows, open(out, "w"), indent=2)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "quality"], required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.mode == "oracle":
        build_oracle(a.n, a.length, a.out, device=dev)
    else:
        run_quality(L=a.length, n=min(a.n, 12), device=dev, out=a.out)
