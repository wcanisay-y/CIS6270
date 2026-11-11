"""Generation from the flat, two-level and AR arms at matched NFE.

Sampling is done from float64 probabilities throughout. Zheng et al.
(arXiv:2409.02908) showed that float32 Gumbel categorical sampling silently
lowers the effective temperature and reduces diversity in masked diffusion
models, which would make any quality-diversity claim an artifact of
numerical precision rather than of the method.
"""
import numpy as np
import torch

from .model import ID_MASK, N_AA
from common.alphabet import AA20


def _cat64(logits, temperature, rank="gumbel"):
    """Sample a token and return the score used to rank commits.

    rank="gumbel" reproduces DPLM's rule: perturb the logits with Gumbel
    noise, take the argmax as the token and the perturbed log-softmax as the
    score. This is not a stylistic choice. Ranking commits by the clean
    probability and committing irrevocably collapses the sampler onto the
    single most predictable residue: measured on DPLM it produced
    polyleucine with 131 of 200 positions identical, and on these arms it
    produced homopolymer runs of 109 to 113 residues. The perturbation is
    what stops that.

    rank="clean" keeps the old behaviour for the ablation.
    """
    lg = logits.double() / max(temperature, 1e-6)
    if rank == "gumbel":
        u = torch.rand_like(lg)
        g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
        conf, out = (lg + g).log_softmax(-1).max(-1)
        return out, conf
    p = torch.softmax(lg, dim=-1)
    flat = p.reshape(-1, p.shape[-1])
    out = torch.multinomial(flat, 1).reshape(p.shape[:-1])
    conf = torch.softmax(logits.double(), -1).gather(
        -1, out.unsqueeze(-1)).squeeze(-1)
    return out, conf


def _sched(n, steps):
    cum = [int(round(n * (i + 1) / steps)) for i in range(steps)]
    cum[-1] = n
    out, prev = [], 0
    for c in cum:
        out.append(max(c - prev, 0)); prev = c
    return out


def _commit_topk(conf, avail, k):
    """Indices of the k most confident available positions, per row.

    The sentinel must be -inf, not -1. Under the Gumbel ranking the scores
    are log-probabilities and therefore negative, so a sentinel of -1 is
    LARGER than many real scores: unavailable positions get selected,
    available ones are skipped, and the sequence finishes with masked
    positions still in it.
    """
    c = conf.masked_fill(~avail, float("-inf"))
    k = min(k, int(avail.sum(1).min().item()) if avail.any() else 0)
    if k <= 0:
        return None
    return c.topk(k, dim=1).indices


@torch.no_grad()
def sample_flat(model, B, L, steps, temperature=1.0, device="cuda", seed=0,
                rank="gumbel"):
    torch.manual_seed(seed)
    ids = torch.full((B, L), ID_MASK, dtype=torch.long, device=device)
    for k in _sched(L, steps):
        logits, _ = model(ids)
        tok, conf = _cat64(logits, temperature, rank)
        avail = ids == ID_MASK
        sel = _commit_topk(conf, avail, k)
        if sel is None:
            continue
        rows = torch.arange(B, device=device)[:, None]
        ids[rows, sel] = tok[rows, sel]
    return _decode(ids)


def _decode(ids):
    """Tokens to strings, failing loudly if anything was left masked."""
    arr = ids.cpu().numpy()
    if (arr >= N_AA).any():
        raise RuntimeError(
            f"{int((arr >= N_AA).sum())} positions were never committed; "
            "the commit schedule did not cover the sequence")
    return [''.join(AA20[i] for i in row) for row in arr]


@torch.no_grad()
def sample_hier(model, B, L, steps, t_fam=1.0, t_id=1.0, rho=0.3,
                bind=True, device="cuda", seed=0, rank="gumbel"):
    """Two-phase: write the family pattern, then the identities.

    rho is the share of the NFE budget spent on the family phase, so the
    total number of forward passes equals `steps` exactly and matches the
    flat arm.
    """
    torch.manual_seed(seed)
    K = model.n_fam
    ids = torch.full((B, L), ID_MASK, dtype=torch.long, device=device)
    fam = torch.full((B, L), K, dtype=torch.long, device=device)
    s_fam = max(1, int(round(steps * rho)))
    s_id = max(1, steps - s_fam)
    rows = torch.arange(B, device=device)[:, None]

    for k in _sched(L, s_fam):
        _, fl = model(ids, fam, bind=bind)
        tok, conf = _cat64(fl, t_fam, rank)
        sel = _commit_topk(conf, fam == K, k)
        if sel is None:
            continue
        fam[rows, sel] = tok[rows, sel]

    for k in _sched(L, s_id):
        il, _ = model(ids, fam, bind=bind)
        tok, conf = _cat64(il, t_id, rank)
        sel = _commit_topk(conf, ids == ID_MASK, k)
        if sel is None:
            continue
        ids[rows, sel] = tok[rows, sel]
    return _decode(ids), fam.cpu().numpy()


@torch.no_grad()
def sample_hier_conditioned(model, fam_pattern, steps, t_id=1.0, bind=True,
                            device="cuda", seed=0, rank="gumbel"):
    """Controllability: the family pattern is given, the model fills identities."""
    torch.manual_seed(seed)
    fam = torch.as_tensor(fam_pattern, dtype=torch.long, device=device)
    if fam.ndim == 1:
        fam = fam[None]
    B, L = fam.shape
    ids = torch.full((B, L), ID_MASK, dtype=torch.long, device=device)
    rows = torch.arange(B, device=device)[:, None]
    for k in _sched(L, steps):
        il, _ = model(ids, fam, bind=bind)
        tok, conf = _cat64(il, t_id, rank)
        sel = _commit_topk(conf, ids == ID_MASK, k)
        if sel is None:
            continue
        ids[rows, sel] = tok[rows, sel]
    return _decode(ids)


@torch.no_grad()
def sample_ar(model, B, L, temperature=1.0, device="cuda", seed=0,
              rank="clean"):
    """NFE = L by construction; reported as such."""
    torch.manual_seed(seed)
    ids = torch.full((B, 1), N_AA, dtype=torch.long, device=device)  # BOS
    for _ in range(L):
        logits = model(ids)[:, -1]
        tok, _ = _cat64(logits, temperature, "clean")
        ids = torch.cat([ids, tok[:, None]], dim=1)
    return _decode(ids[:, 1:])


def family_compliance(seqs, fam_pattern, fam_lut):
    """Fraction of positions whose written identity is in the asked family."""
    fp = np.asarray(fam_pattern)
    if fp.ndim == 1:
        fp = fp[None].repeat(len(seqs), 0)
    ok = tot = 0
    for s, row in zip(seqs, fp):
        for c, f in zip(s, row):
            tot += 1
            ok += int(fam_lut[AA20.index(c)] == f)
    return ok / max(tot, 1)


@torch.no_grad()
def sample_hier_scheduled(model, B, L, steps, t_fam=1.0, t_id=1.0,
                          gamma=2.0, bind=True, device="cuda", seed=0,
                          refresh=False, rank="gumbel"):
    """Two-level sampling that follows the TRAINING corruption schedule.

    The naive two-phase sampler -- reveal every family, then every identity --
    visits states the model was never trained on. Training couples the two
    channels: at noise level t a position's identity is masked with
    probability t and its family with probability t**gamma, and the family
    mask is nested inside the identity mask. The state "all identities
    masked, half the families revealed" has no t that produces it, because
    t = 1 forces t**gamma = 1 as well.

    This sampler walks t from 1 down to 0 and, at each step, reveals however
    many families and identities the schedule calls for at that t. Families
    still run ahead of identities, by exactly the margin the model saw in
    training, so the family pattern is written first without ever leaving
    the training distribution.
    """
    torch.manual_seed(seed)
    K = model.n_fam
    ids = torch.full((B, L), ID_MASK, dtype=torch.long, device=device)
    fam = torch.full((B, L), K, dtype=torch.long, device=device)
    rows = torch.arange(B, device=device)[:, None]
    nfe = 0

    for s in range(1, steps + 1):
        t = 1.0 - s / steps
        want_id_masked = int(round(t * L))
        want_fam_masked = int(round((t ** gamma) * L))
        il, fl = model(ids, fam, bind=bind)
        nfe += 1

        # families first: they lead the identities at every t
        cur = int((fam[0] == K).sum())
        k = cur - want_fam_masked
        if k > 0:
            tok, conf = _cat64(fl, t_fam, rank)
            sel = _commit_topk(conf, fam == K, k)
            if sel is not None:
                fam[rows, sel] = tok[rows, sel]
                if refresh and bind:
                    # A second pass per step re-reads the identity logits
                    # under the families just written. More correct, but it
                    # doubles the forward passes, so it is off by default and
                    # the count is reported either way.
                    il, _ = model(ids, fam, bind=bind)
                    nfe += 1
                elif bind:
                    # keep NFE matched: re-apply the binding restriction to
                    # the logits we already have
                    known = fam < K
                    allow = torch.ones_like(il, dtype=torch.bool)
                    allow[known] = model.memb[fam[known]]
                    il = il.masked_fill(~allow, -1e4)

        cur = int((ids[0] == ID_MASK).sum())
        k = cur - want_id_masked
        if k > 0:
            tok, conf = _cat64(il, t_id, rank)
            sel = _commit_topk(conf, ids == ID_MASK, k)
            if sel is not None:
                ids[rows, sel] = tok[rows, sel]

    if (ids == ID_MASK).any():                # finish any remainder
        il, _ = model(ids, fam, bind=bind)
        nfe += 1
        tok, _ = _cat64(il, t_id, rank)
        ids = torch.where(ids == ID_MASK, tok, ids)
    return _decode(ids), fam.cpu().numpy(), nfe
