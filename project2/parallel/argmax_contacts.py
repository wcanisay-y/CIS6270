"""Contact accuracy on the ARGMAX COMPLETION, not on the masked sequence.

Pranam's first ask. Our published masking table fed a partially masked
sequence straight to the contact head and found a collapse from precision@L
0.775 to 0.087 at a single quarter masked, which we attributed to masks
being out of distribution for ESM-2 (trained at a 15 percent mask rate).

If that diagnosis is right, the fix is immediate and costs one forward pass
the sampler has already made: fill every masked position with the
generator's own argmax, and hand the contact head a COMPLETE sequence it has
seen the like of in training. This module measures whether that works.

Three inputs per masking level, scored against the same deposited map:

  masked     the partially masked sequence (what the original study used)
  argmax     masks replaced by DPLM's argmax (what the comment proposes)
  null       the all-mask input, as the floor

and two heads: a separate frozen ESM-2 650M pass (`ext`, what the proposal
specifies) and DPLM's own attention through the same head regression
(`self`, free).

`recovery` is the fraction of masked positions whose argmax equals the true
residue; it bounds how much the completion can be trusted and is reported
alongside, because a completion that is wrong everywhere can still produce a
self-consistent contact map.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel.sample import load_dplm, banned_ids
from parallel.contacts import ExternalContacts, SelfContacts

SEP = 6          # minimum sequence separation for a scored pair


def auroc(score, label):
    """Rank-based AUROC; returns nan if one class is absent."""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def score_map(pred, true, L):
    """precision@L and AUROC of `pred` against `true`, pairs at |i-j| >= SEP."""
    iu = np.triu_indices(L, k=SEP)
    p, t = pred[iu], true[iu].astype(bool)
    k = min(L, len(p))
    top = np.argsort(-p)[:k]
    return float(t[top].mean()), auroc(p, t)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="airkingbd/dplm_150m")
    ap.add_argument("--oracle-file", default="results/A_oracle_cath.npz")
    ap.add_argument("--n", type=int, default=43)
    ap.add_argument("--reveals", type=float, nargs="+",
                    default=[1.0, 0.75, 0.50, 0.25, 0.10])
    ap.add_argument("--contact-size", type=int, default=650)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32 throughout: predict_contacts mixes fp32 attentions with the head's
    # linear layer and dies on a halved model.
    tok, dplm = load_dplm(a.model, device=dev, half=False, need_attn=True)
    ext = ExternalContacts(a.contact_size, device=dev, half=False,
                           token_dropout=False)
    slf = SelfContacts(150, device=dev)
    ban = sorted(banned_ids(tok))
    mask_id = tok.mask_token_id

    z = np.load(a.oracle_file, allow_pickle=True)
    maps, seqs = z["maps"], [str(s) for s in z["seqs"]]
    n = min(a.n, len(seqs))
    print("%d chains from %s" % (n, a.oracle_file), flush=True)

    rows = []
    t0 = time.time()
    for ci in range(n):
        s, true = seqs[ci], maps[ci]
        L = len(s)
        ids0 = tok(s, return_tensors="pt")["input_ids"].to(dev)
        if ids0.shape[1] != L + 2:
            print("  skip %d: tokenised length mismatch" % ci, flush=True)
            continue
        g = torch.Generator().manual_seed(a.seed + ci)
        perm = torch.randperm(L, generator=g)

        for rev in a.reveals:
            n_keep = int(round(rev * L))
            ids = ids0.clone()
            hid = perm[n_keep:] + 1               # never touches cls/eos
            ids[0, hid] = mask_id

            # ---- the masked input, as the original study fed it
            c_ext_m = ext(ids)[0].float().cpu().numpy()
            out_m = dplm(input_ids=ids, output_attentions=True)
            c_slf_m = slf.from_attentions(
                ids, out_m.attentions)[0].float().cpu().numpy()

            # ---- the argmax completion, as the comment proposes
            lg = out_m.logits.float()
            lg[..., ban] = -float("inf")
            arg = lg.argmax(-1)
            ids_a = ids.clone()
            ids_a[0, hid] = arg[0, hid]
            c_ext_a = ext(ids_a)[0].float().cpu().numpy()
            out_a = dplm(input_ids=ids_a, output_attentions=True)
            c_slf_a = slf.from_attentions(
                ids_a, out_a.attentions)[0].float().cpu().numpy()

            rec = float((ids_a[0, hid] == ids0[0, hid]).float().mean()) \
                if hid.numel() else 1.0
            row = dict(chain=ci, reveal=rev, L=L, n_masked=int(hid.numel()),
                       recovery=rec)
            for nm, cm in (("ext_masked", c_ext_m), ("ext_argmax", c_ext_a),
                           ("self_masked", c_slf_m), ("self_argmax", c_slf_a)):
                p, au = score_map(cm, true, L)
                row[nm + "_pL"], row[nm + "_auroc"] = p, au
            rows.append(row)
        if (ci + 1) % 5 == 0:
            print("  %d/%d chains (%.1f min)"
                  % (ci + 1, n, (time.time() - t0) / 60), flush=True)

    # ---- the all-mask floor, one per chain length
    nulls = []
    for ci in range(min(n, 12)):
        s, true = seqs[ci], maps[ci]
        L = len(s)
        ids = tok(s, return_tensors="pt")["input_ids"].to(dev)
        ids[0, 1:L + 1] = mask_id
        c_ext = ext(ids)[0].float().cpu().numpy()
        o = dplm(input_ids=ids, output_attentions=True)
        c_slf = slf.from_attentions(ids, o.attentions)[0].float().cpu().numpy()
        pe, ae = score_map(c_ext, true, L)
        ps, as_ = score_map(c_slf, true, L)
        nulls.append(dict(chain=ci, ext_pL=pe, ext_auroc=ae,
                          self_pL=ps, self_auroc=as_))

    out = dict(config=vars(a), rows=rows, nulls=nulls)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    # ---- table
    print("\n%-8s %-9s | %-16s | %-16s | %s"
          % ("reveal", "recovery", "ext  masked->argmax",
             "self masked->argmax", "(precision@L)"), flush=True)
    for rev in a.reveals:
        sub = [r for r in rows if r["reveal"] == rev]
        if not sub:
            continue
        f = lambda k: np.nanmean([r[k] for r in sub])
        print("%-8.2f %-9.3f | %6.3f -> %6.3f | %6.3f -> %6.3f"
              % (rev, f("recovery"), f("ext_masked_pL"), f("ext_argmax_pL"),
                 f("self_masked_pL"), f("self_argmax_pL")), flush=True)
    if nulls:
        print("%-8s %-9s | %6.3f            | %6.3f"
              % ("null", "-", np.nanmean([x["ext_pL"] for x in nulls]),
                 np.nanmean([x["self_pL"] for x in nulls])), flush=True)
    print("\nAUROC")
    for rev in a.reveals:
        sub = [r for r in rows if r["reveal"] == rev]
        if not sub:
            continue
        f = lambda k: np.nanmean([r[k] for r in sub])
        print("%-8.2f %-9s | %6.3f -> %6.3f | %6.3f -> %6.3f"
              % (rev, "", f("ext_masked_auroc"), f("ext_argmax_auroc"),
                 f("self_masked_auroc"), f("self_argmax_auroc")), flush=True)
    print("\nwrote %s in %.1f min" % (a.out, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
