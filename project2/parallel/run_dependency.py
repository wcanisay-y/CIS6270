"""Is a pairwise CONTACT map the right object for parallel unmasking?

Part A, on NATURAL sequences with DEPOSITED contact maps: measure the
model's own interventional dependency D[i,j] and ask how well a true
contact map predicts it. If a ground-truth contact map cannot identify the
pairs the model actually couples, the gating rule was reading the wrong
graph and no improvement in contact prediction can fix it.

Part B, on GENERATED states: measure the factorization error of the group a
rule selects. This is the quantity parallel decoding gets wrong and the
quantity the rule exists to reduce, measured directly rather than inferred
from pLDDT. Rules compared:

    confidence     no gating (the free baseline)
    pair_contact   the original pairwise rule on the self-attention map
    pair_shuffled  same, structure destroyed (control)
    pair_dep       the same pairwise rule on MEASURED dependency (oracle
                   signal: isolates "wrong graph" from "wrong rule")
    dpp_dep        groupwise determinantal rule on measured dependency
    dpp_contact    groupwise determinantal rule on the contact map

If pair_dep and dpp_dep also fail to beat confidence, the decorrelation
axis is empty regardless of signal or grouping.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.alphabet import AA20
from parallel.sample import load_dplm, banned_ids
from parallel.contacts import SelfContacts
from parallel.dependency import (make_state, factorization_error,
                                 dependency_matrix, psd_kernel,
                                 select_confidence, select_pairwise,
                                 select_dpp)
from parallel.selectors import shuffle_coupling


def auroc(score, label):
    """Rank-based AUROC; label in {0,1}."""
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label).ravel().astype(bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def self_map(model, slf, ids):
    """DPLM's own attentions through the ESM-2 contact head -> (B, L, L)."""
    o = model(input_ids=ids, output_attentions=True)
    return slf.from_attentions(ids, o.attentions).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="airkingbd/dplm_150m")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--oracle-file", default="results/A_oracle_pdb.npz")
    ap.add_argument("--reveals", type=float, nargs="+",
                    default=[0.25, 0.50, 0.75])
    ap.add_argument("--n-probe", type=int, default=40,
                    help="masked positions over which D is measured")
    ap.add_argument("--group", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tau-q", type=float, default=0.90)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="both",
                    choices=["A", "B", "both"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = a.length
    tok, model = load_dplm(a.model, device=dev, half=False, need_attn=True)
    aa_ids = [tok.convert_tokens_to_ids(c) for c in AA20]
    slf = SelfContacts(150, device=dev)
    mask_id = tok.mask_token_id
    rng = np.random.default_rng(a.seed)
    out = {"config": vars(a), "partA": [], "partB": []}
    t0 = time.time()

    # ---------------- Part A: is CONTACT the right graph? ----------------
    if a.part == "B":
        out["partA"] = None
    z = np.load(a.oracle_file, allow_pickle=True)
    maps, seqs = z["maps"], [str(s) for s in z["seqs"]]
    print("### Part A: true contact map vs the model's own dependency ###",
          flush=True)
    for rv in (a.reveals if a.part in ("A", "both") else []):
        rows = []
        for b0 in range(0, min(len(seqs), a.states * a.batch), a.batch):
            chunk = seqs[b0:b0 + a.batch]
            if not chunk:
                break
            B = len(chunk)
            enc = tok(chunk, return_tensors="pt", add_special_tokens=True)
            ids = enc["input_ids"].to(dev)
            torch.manual_seed(a.seed + b0)
            # Probe positions must be masked in EVERY member of the chunk,
            # so choose them FIRST and force them masked, then mask the rest
            # at random to hit the target reveal rate. Drawing independent
            # masks per sequence instead makes the commonly-masked set shrink
            # like (1-rv)^B, which empties above about 25 percent revealed
            # and silently drops those rows.
            n_probe = min(a.n_probe, int(L * (1 - rv)))
            if n_probe < 8:
                continue
            pos = sorted(rng.choice(L, size=n_probe,
                                    replace=False).tolist())
            pt = torch.as_tensor(pos, device=dev)
            keep = torch.zeros(B, L, dtype=torch.bool, device=dev)
            n_keep = int(round(L * rv))
            free = torch.ones(L, dtype=torch.bool, device=dev)
            free[pt] = False
            fidx = free.nonzero().flatten()
            for b in range(B):
                perm = fidx[torch.randperm(fidx.numel(), device=dev)]
                keep[b, perm[:min(n_keep, fidx.numel())]] = True
            body = ids[:, 1:L + 1].clone()
            body[~keep] = mask_id
            ids[:, 1:L + 1] = body
            D, _ = dependency_matrix(tok, model, ids, pos, aa_ids, L,
                                     device=dev)
            Dn = D.cpu().numpy()
            tm = maps[b0:b0 + B][:, pos][:, :, pos]
            sc = self_map(model, slf, ids)[:, pos][:, :, pos] \
                .cpu().numpy()
            iu = np.triu_indices(len(pos), k=1)
            for b in range(B):
                d = Dn[b][iu]
                lab = tm[b][iu] > 0.5
                r = {"reveal": rv, "pdb": str(z["pdb"][b0 + b]),
                     "auroc_true_contact_predicts_dep": auroc(d, lab),
                     "dep_mean_contact": float(d[lab].mean()) if lab.any() else float("nan"),
                     "dep_mean_noncontact": float(d[~lab].mean()),
                     "contact_frac": float(lab.mean()),
                     "dep_mean": float(d.mean())}
                sep = np.abs(np.subtract.outer(np.array(pos), np.array(pos)))[iu]
                r["auroc_seqdist"] = auroc(-sep.astype(float), lab)
                hi = d > np.quantile(d, 0.90)
                r["auroc_seqdist_predicts_dep"] = auroc(
                    -sep.astype(float), hi)
                r["auroc_contact_predicts_hidep"] = auroc(lab * 1.0, hi)
                r["spearman_dep_seqdist"] = float(np.corrcoef(
                    np.argsort(np.argsort(d)),
                    np.argsort(np.argsort(-sep.astype(float))))[0, 1])
                if sc is not None:
                    r["auroc_selfmap_predicts_dep"] = auroc(
                        sc[b][iu], d > np.quantile(d, 0.90))
                    r["auroc_selfmap_predicts_contact"] = auroc(sc[b][iu], lab)
                rows.append(r)
        if rows:
            agg = {k: float(np.nanmean([r[k] for r in rows]))
                   for k in rows[0] if k not in ("pdb",)}
            agg["n"] = len(rows)
            agg["rows"] = rows        # keep per-chain rows: the per-CATH-class
                                      # breakdown needs them
            out["partA"].append(agg)
            print("  reveal %.2f  n=%d  AUROC(dep | true contact)=%.3f  "
                  "dep in-contact %.4f vs non-contact %.4f  "
                  "AUROC(seqdist->contact)=%.3f  "
                  "AUROC(seqdist->hi-dep)=%.3f"
                  % (rv, agg["n"], agg["auroc_true_contact_predicts_dep"],
                     agg["dep_mean_contact"], agg["dep_mean_noncontact"],
                     agg["auroc_seqdist"],
                     agg["auroc_seqdist_predicts_dep"]), flush=True)

    # ------------- Part B: does any rule cut the real error? -------------
    print("### Part B: factorization error by selection rule ###", flush=True)
    for rv in (a.reveals if a.part in ("B", "both") else []):
        for k in a.group:
            acc = {}
            for st in range(a.states):
                ids = make_state(tok, model, L, a.batch, rv, aa_ids,
                                 seed=a.seed + 1000 * st, device=dev)
                maskpos = (ids[:, 1:L + 1] == mask_id)
                allmask = maskpos.all(0).nonzero().flatten()
                if allmask.numel() < k + 4:
                    continue
                sel = allmask[torch.randperm(
                    allmask.numel(), device=dev)[:a.n_probe]].sort().values
                pos = sel.cpu().numpy().tolist()
                D, base = dependency_matrix(tok, model, ids, pos, aa_ids, L,
                                            device=dev)
                conf_all = base.max(-1).values                   # (B, L)
                Cself = self_map(model, slf, ids)        # (B, L, L)
                groups = {}
                for b in range(a.batch):
                    cand = sel
                    conf = conf_all[b]
                    Dfull = torch.zeros(L, L, device=dev)
                    Dfull[sel.unsqueeze(1), sel.unsqueeze(0)] = D[b]
                    Cs = Cself[b]
                    tq = torch.quantile(Cs[sel][:, sel].flatten().float(),
                                        a.tau_q)
                    dq = torch.quantile(D[b].flatten().float(), a.tau_q)
                    Csh = shuffle_coupling(Cs)
                    kd = psd_kernel(D[b], gamma=a.gamma)
                    kc = psd_kernel(Cs[sel][:, sel], gamma=a.gamma)
                    picks = {
                        "confidence": select_confidence(conf, cand, k),
                        "pair_contact": select_pairwise(conf, cand, k, Cs,
                                                        float(tq)),
                        "pair_shuffled": select_pairwise(conf, cand, k, Csh,
                                                         float(tq)),
                        "pair_dep": select_pairwise(conf, cand, k, Dfull,
                                                    float(dq)),
                        "dpp_dep": sel[select_dpp(conf[sel],
                                                  torch.arange(len(pos),
                                                               device=dev),
                                                  k, kd, lam=a.lam)],
                        "dpp_contact": sel[select_dpp(
                            conf[sel],
                            torch.arange(len(pos), device=dev),
                            k, kc, lam=a.lam)],
                    }
                    for nm, p in picks.items():
                        p = p.flatten()[:k]
                        if p.numel() < k:
                            p = torch.cat([p, -torch.ones(
                                k - p.numel(), dtype=torch.long,
                                device=dev)])
                        groups.setdefault(nm, []).append(p)
                for nm, gl in groups.items():
                    g = torch.stack(gl)
                    fe = factorization_error(tok, model, ids, g, aa_ids, L,
                                             device=dev)
                    acc.setdefault(nm, {"tv": [], "flip": [], "conf": []})
                    acc[nm]["tv"].append(fe["tv_mean"])
                    acc[nm]["flip"].append(fe["flip_rate"])
                    valid = g >= 0
                    cg = torch.gather(conf_all, 1, g.clamp_min(0))
                    acc[nm]["conf"].append(
                        float((cg * valid).sum() / valid.sum().clamp_min(1)))
            if not acc:
                continue
            rec = {"reveal": rv, "group": k,
                   "rules": {nm: {"tv": float(np.mean(v["tv"])),
                                  "flip": float(np.mean(v["flip"])),
                                  "conf": float(np.mean(v["conf"])),
                                  "n_states": len(v["tv"])}
                             for nm, v in acc.items()}}
            out["partB"].append(rec)
            base_tv = rec["rules"]["confidence"]["tv"]
            print("  reveal %.2f group %d" % (rv, k), flush=True)
            for nm, v in sorted(rec["rules"].items(),
                                key=lambda x: x[1]["tv"]):
                print("     %-14s TV %.4f (%+.1f%% vs confidence)  "
                      "flip %.3f  conf %.3f"
                      % (nm, v["tv"], 100 * (v["tv"] / base_tv - 1),
                         v["flip"], v["conf"]), flush=True)
            json.dump(out, open(a.out, "w"), indent=1)
    json.dump(out, open(a.out, "w"), indent=1)
    print("done in %.1f min -> %s" % ((time.time() - t0) / 60, a.out),
          flush=True)


if __name__ == "__main__":
    main()
