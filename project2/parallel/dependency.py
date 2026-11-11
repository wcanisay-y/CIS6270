"""Does PAIRWISE CONTACT probability measure the thing parallel decoding
gets wrong? Measure the error directly instead of inferring it from pLDDT.

When a masked-diffusion decoder commits a set S of positions in one step it
replaces the joint p(x_S | ctx) with the product of marginals
prod_i p(x_i | ctx). The error of that substitution is the TOTAL CORRELATION
of S under the MODEL,

    TC(S) = sum_i H(x_i | ctx) - H(x_S | ctx)      >= 0,

which is (a) a property of the model's predictive distribution, not of the
protein's geometry, and (b) groupwise by construction -- three positions can
be pairwise almost independent and still jointly determined.

A contact map is a double proxy: geometry stands in for dependency, and
pairs stand in for the set. This module measures the real quantity.

  factorization_error   leave-one-out TV shift inside a committed group:
                        for each j in S, how far does p(x_j) move once the
                        REST of S is written? Zero iff the group factorises.
                        This is exactly what the gating rule exists to
                        reduce, and it needs no folding to evaluate.

  dependency_matrix     D[i,j] = TV( p(x_j | ctx), p(x_j | ctx, x_i := v) ),
                        the interventional dependency the decoder actually
                        pays for. Comparing D with the contact map answers
                        whether contacts are even the right graph.

  group selectors       `dpp` is the groupwise generalisation of the
                        pairwise rejection rule: a determinantal objective
                        scores a SET by log det of its dependency kernel,
                        which is small exactly when the set is internally
                        dependent, and unlike a pairwise threshold it sees
                        three-way redundancy. `dep` applies the original
                        greedy rule to measured dependency instead of
                        contacts, isolating "wrong graph" from "wrong rule".
"""
import time

import numpy as np
import torch

from .sample import banned_ids, commit_schedule
from .selectors import greedy_decoupled, shuffle_coupling


def _aa_probs(logits_row, aa_ids):
    """(.., V) logits -> (.., 20) renormalised amino-acid probabilities."""
    p = torch.softmax(logits_row.float(), dim=-1)[..., aa_ids]
    return p / p.sum(-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def make_state(tok, model, L, B, reveal_frac, aa_ids, seed=0, device="cuda",
               steps=10):
    """Run the absorbing decoder partway to get a realistic mid-generation
    state: `reveal_frac` of positions written by confidence, rest masked."""
    torch.manual_seed(seed)
    ban = banned_ids(tok)
    mask_id = tok.mask_token_id
    ids = torch.full((B, L + 2), mask_id, dtype=torch.long, device=device)
    ids[:, 0] = tok.cls_token_id
    ids[:, -1] = tok.eos_token_id
    target = int(round(L * reveal_frac))
    if target == 0:
        return ids
    per = max(1, target // steps)
    written = 0
    while written < target:
        n_t = min(per, target - written)
        out = model(input_ids=ids)
        lg = out.logits.float()
        lg[..., ban] = -float("inf")
        body = lg[:, 1:L + 1]
        prob = torch.softmax(body, dim=-1)
        conf, tokidx = prob.max(-1)
        for b in range(B):
            m = (ids[b, 1:L + 1] == mask_id).nonzero().flatten()
            if m.numel() == 0:
                continue
            k = min(n_t, m.numel())
            pick = m[conf[b][m].topk(k).indices]
            ids[b, pick + 1] = tokidx[b][pick]
        written += n_t
    return ids


@torch.no_grad()
def factorization_error(tok, model, ids, groups, aa_ids, L, device="cuda"):
    """Leave-one-out TV shift inside each committed group.

    groups: LongTensor (B, k) of positions (0-based in the body) to commit
    together. Returns dict with the mean TV shift, the max over the group,
    and the token-flip rate (how often the independently sampled token stops
    being the mode once the rest of the group is written).

    Cost: 1 + k forward passes. No folding.
    """
    ban = banned_ids(tok)
    mask_id = tok.mask_token_id
    B, k = groups.shape
    out = model(input_ids=ids)
    lg = out.logits.float()
    lg[..., ban] = -float("inf")
    base = _aa_probs(lg[:, 1:L + 1], aa_ids)                    # (B, L, 20)
    bidx = torch.arange(B, device=device)
    # independent (parallel) draw for every member of the group
    gi = groups.clamp_min(0)
    pg = base[bidx.unsqueeze(1), gi]                            # (B, k, 20)
    drawn = pg.argmax(-1)                                       # modal token
    tok_ids = torch.as_tensor(aa_ids, device=device)[drawn]     # (B, k)

    tv = torch.zeros(B, k, device=device)
    flip = torch.zeros(B, k, device=device)
    for j in range(k):
        ids_j = ids.clone()
        for jj in range(k):
            if jj == j:
                continue
            ok = groups[:, jj] >= 0
            if ok.any():
                ids_j[bidx[ok], groups[ok, jj] + 1] = tok_ids[ok, jj]
        ids_j[bidx, gi[:, j] + 1] = mask_id
        o = model(input_ids=ids_j)
        l2 = o.logits.float()
        l2[..., ban] = -float("inf")
        cond = _aa_probs(l2[:, 1:L + 1], aa_ids)
        pj_cond = cond[bidx, gi[:, j]]                           # (B, 20)
        pj_base = base[bidx, gi[:, j]]
        tv[:, j] = 0.5 * (pj_cond - pj_base).abs().sum(-1)
        flip[:, j] = (pj_cond.argmax(-1) != drawn[:, j]).float()
    valid = (groups >= 0).float()
    n = valid.sum().clamp_min(1)
    return {"tv_mean": float((tv * valid).sum() / n),
            "tv_max": float((tv * valid).max()),
            "flip_rate": float((flip * valid).sum() / n),
            "tv_per_group": (tv * valid).sum(1).div(
                valid.sum(1).clamp_min(1)).cpu().numpy(),
            "nfe": 1 + k}


@torch.no_grad()
def dependency_matrix(tok, model, ids, pos, aa_ids, L, device="cuda"):
    """D[b,i,j] = TV( p(x_j|ctx), p(x_j|ctx, x_i := modal) ) for i,j in pos.

    One forward pass per intervened position i, so len(pos) + 1 in total.
    Only masked positions are meaningful as j.
    """
    ban = banned_ids(tok)
    mask_id = tok.mask_token_id
    B = ids.shape[0]
    m = len(pos)
    out = model(input_ids=ids)
    lg = out.logits.float()
    lg[..., ban] = -float("inf")
    base = _aa_probs(lg[:, 1:L + 1], aa_ids)
    pt = torch.as_tensor(pos, device=device)
    aat = torch.as_tensor(aa_ids, device=device)
    modal = aat[base[:, pt].argmax(-1)]                          # (B, m)
    D = torch.zeros(B, m, m, device=device)
    for a in range(m):
        ids_a = ids.clone()
        ids_a[:, pt[a] + 1] = modal[:, a]
        o = model(input_ids=ids_a)
        l2 = o.logits.float()
        l2[..., ban] = -float("inf")
        cond = _aa_probs(l2[:, 1:L + 1], aa_ids)
        D[:, a, :] = 0.5 * (cond[:, pt] - base[:, pt]).abs().sum(-1)
        D[:, a, a] = 0.0
    D = 0.5 * (D + D.transpose(1, 2))      # symmetrise: dependency is mutual
    return D, base


# ----------------------------------------------------------------------
# group selection rules
# ----------------------------------------------------------------------
def select_confidence(conf, cand, n):
    return cand[conf[cand].topk(min(n, cand.numel())).indices]


def select_pairwise(conf, cand, n, K, tau):
    """The original rule: greedy admission, reject a candidate coupled above
    tau to anything already admitted."""
    return greedy_decoupled(cand, conf, K, tau, n)


def psd_kernel(M, gamma=1.0, eps=1e-6):
    """Turn a dependency matrix into a PSD DPP kernel.

    K = I + gamma * M_sym, projected onto the PSD cone by eigenvalue
    clipping. det(K_S) shrinks as the members of S become mutually
    dependent, which is the property the determinant contributes and a
    pairwise threshold does not have.
    """
    M = 0.5 * (M + M.transpose(-1, -2)).double()
    n = M.shape[-1]
    K = torch.eye(n, device=M.device, dtype=torch.float64) + gamma * M
    w, V = torch.linalg.eigh(K)
    w = w.clamp_min(eps)
    return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)


def select_dpp(conf, cand, n, K, lam=1.0, eps=1e-9):
    """Groupwise rule: greedy MAP of  sum_{i in S} log conf_i + lam*logdet K_S.

    logdet is the groupwise term a pairwise threshold cannot express: it
    falls when the set is internally dependent in ANY order, including
    three-way redundancy that every individual pair survives. The marginal
    gain of adding i to the chosen set C is the Schur complement
    log( K_ii - K_iC K_CC^-1 K_Ci ), so each greedy step costs one solve.
    K must be PSD -- build it with psd_kernel().
    """
    cand = cand.flatten()
    nc = int(cand.numel())
    n = int(min(n, nc))
    if n <= 0:
        return cand[:0]
    Ks = K[cand][:, cand].double()
    Ks = 0.5 * (Ks + Ks.T)
    lc = conf[cand].double().clamp_min(eps).log()
    chosen = []
    for _ in range(n):
        if chosen:
            ci = torch.as_tensor(chosen, device=Ks.device)
            A = Ks[ci][:, ci]
            B = Ks[ci][:, :]                              # (|C|, nc)
            sol = torch.linalg.solve(
                A + eps * torch.eye(len(chosen), device=Ks.device,
                                    dtype=torch.float64), B)
            schur = Ks.diagonal() - (B * sol).sum(0)
        else:
            schur = Ks.diagonal().clone()
        gain = lc + lam * schur.clamp_min(eps).log()
        if chosen:
            gain[torch.as_tensor(chosen, device=Ks.device)] = -float("inf")
        chosen.append(int(gain.argmax()))
    return cand[torch.as_tensor(chosen, device=Ks.device)]
