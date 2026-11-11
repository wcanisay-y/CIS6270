"""Which masked positions to unmask together (Idea A).

A masked diffusion denoiser gives per-position conditional marginals
p(x_i | x_t), never the joint. Committing two coupled positions in the same
step therefore samples them independently and loses their correlation. Each
selector below is a different answer to "which set is safe to fill at once".

Every selector commits EXACTLY n_t positions per step, so the number of
network forward passes is identical across selectors. That is what makes the
comparison a matched-NFE comparison and not a compute comparison.

    confidence       top-n by denoiser confidence            (MaskGIT / DPLM)
    random           uniform                                 (control)
    entropy          lowest predictive entropy
    margin           largest top1-top2 gap
    contact          ours: greedy, reject a position coupled to one already
                     accepted, coupling = ESM-2 contact probability
    contact_shuffled identical but the contact matrix is permuted
                     (the decisive control: same group structure, no structure
                     information)
    distance         coupling = sequence separation <= d
                     (control: do contacts beat trivial spacing)
    punt             conditional-independence test from the denoiser itself
                     (prior work: PUNT)
"""
import numpy as np
import torch

SELECTORS = ("confidence", "random", "entropy", "margin", "contact",
             "contact_shuffled", "distance", "punt")


# ----------------------------------------------------------------- helpers
def _topk_idx(score, cand, n):
    """Indices (into the full length) of the n highest-scoring candidates."""
    s = score[cand]
    k = min(n, cand.numel())
    order = torch.argsort(s, descending=True)[:k]
    return cand[order]


def greedy_decoupled(cand, score, coupling, tau, n, generator=None):
    """Greedy maximal set of weakly-coupled positions, padded to exactly n.

    Walk candidates in descending confidence. Accept one if its coupling to
    every already-accepted position is below tau. If that yields fewer than n,
    top up with the rejected candidates in ascending order of their coupling
    to the accepted set, so the step still commits n positions and NFE stays
    matched across selectors.

    cand     (m,) long   candidate positions
    score    (L,) float  confidence per position
    coupling (L, L) float  symmetric coupling in [0, 1]
    """
    m = cand.numel()
    n = int(min(n, m))
    if n <= 0:
        return cand[:0]
    order = cand[torch.argsort(score[cand], descending=True)]
    if coupling is None:
        return order[:n]
    sub = coupling[order][:, order]              # (m, m) in confidence order
    accepted = torch.zeros(m, dtype=torch.bool, device=cand.device)
    # running max coupling of each candidate to the accepted set
    maxc = torch.zeros(m, dtype=sub.dtype, device=cand.device)
    for j in range(m):
        if accepted.sum() >= n:
            break
        if maxc[j] < tau:
            accepted[j] = True
            maxc = torch.maximum(maxc, sub[:, j])
    sel = order[accepted]
    if sel.numel() < n:                          # top-up, least coupled first
        rest = order[~accepted]
        rmax = maxc[~accepted]
        extra = rest[torch.argsort(rmax)][: n - sel.numel()]
        sel = torch.cat([sel, extra])
    return sel


# --------------------------------------------------------------- selectors
def choose(name, *, cand, logits_pos, n, conf=None, coupling=None, tau=0.3,
           d=8, rng=None, L=None, punt_state=None):
    """Return the positions to commit this step (exactly min(n, len(cand))).

    `conf` is the confidence used for ranking. It must be the clean
    probability of the token that will actually be written, which is the
    MaskGIT convention; passing it in rather than recomputing a max-prob
    here keeps the ranking and the committed token consistent.
    """
    probs = torch.softmax(logits_pos.float(), dim=-1)          # (L, V)
    top2 = probs.topk(2, dim=-1).values
    if conf is None:
        conf = top2[:, 0]
    if name == "confidence":
        return _topk_idx(conf, cand, n)
    if name == "margin":
        return _topk_idx(top2[:, 0] - top2[:, 1], cand, n)
    if name == "entropy":
        ent = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
        return _topk_idx(-ent, cand, n)
    if name == "random":
        perm = torch.randperm(cand.numel(), generator=rng,
                              device=cand.device)
        return cand[perm[: min(n, cand.numel())]]
    if name in ("contact", "contact_shuffled"):
        if coupling is None:
            raise ValueError(f"{name} needs a coupling matrix")
        return greedy_decoupled(cand, conf, coupling, tau, n)
    if name == "distance":
        Lx = L if L is not None else logits_pos.shape[0]
        ar = torch.arange(Lx, device=cand.device)
        cp = (ar[:, None] - ar[None, :]).abs().le(d).float()
        cp.fill_diagonal_(0.0)
        return greedy_decoupled(cand, conf, cp, 0.5, n)
    if name in ("anti_contact", "anti_contact_oracle"):
        if coupling is None:
            raise ValueError(f"{name} needs a coupling matrix")
        return greedy_coupled(cand, conf, coupling, n)
    if name == "contact_degree":
        return degree_order(cand, coupling, n, descending=True)
    if name == "punt":
        if punt_state is None:
            raise ValueError("punt needs punt_state")
        return greedy_decoupled(cand, conf, punt_state, tau, n)
    raise ValueError(f"unknown selector {name}")


def shuffle_coupling(cp, rng=None):
    """Permute a coupling matrix, preserving its value distribution.

    The same symmetric permutation is applied to rows and columns, so the
    matrix stays a valid symmetric coupling with an unchanged spectrum and
    unchanged marginal statistics -- only the association between coupling
    and position identity is destroyed.
    """
    n = cp.shape[0]
    perm = torch.randperm(n, generator=rng, device=cp.device)
    return cp[perm][:, perm]


# ===================================================================
# Polarity.
#
# The literature prescribes OPPOSITE things from the same signal.
#   Yang, Yim, Barzilay & Jaakkola (arXiv:2312.02447, MLSB@NeurIPS 2023)
#   hypothesised that coupled residues should be decoded TOGETHER, on the
#   grounds that the denoiser would be jointly more confident about them.
#   They tried biasing the order by coupling strength and report it did not
#   work, and they list "residues close in space should decode together" as
#   an open problem.
#   Contact-order work in inverse folding decodes CONTACTING residues FIRST,
#   because those positions are the most constrained.
#   The proposal here does the reverse of the first: it decodes mutually
#   NON-contacting residues together, to limit the error from sampling
#   coupled positions independently.
#
# These are testable alternatives, not a settled question, so both
# directions are implemented and both are run.
# ===================================================================
def greedy_coupled(cand, score, coupling, n):
    """Opposite polarity: commit a mutually COUPLED set.

    Seed with the most confident candidate, then repeatedly add whichever
    remaining candidate has the highest coupling to the accepted set.
    Commits exactly n positions, so NFE still matches.
    """
    m = cand.numel()
    n = int(min(n, m))
    if n <= 0:
        return cand[:0]
    order = cand[torch.argsort(score[cand], descending=True)]
    if coupling is None:
        return order[:n]
    sub = coupling[order][:, order]
    accepted = torch.zeros(m, dtype=torch.bool, device=cand.device)
    accepted[0] = True
    link = sub[:, 0].clone()
    for _ in range(n - 1):
        link_masked = link.masked_fill(accepted, -float("inf"))
        j = int(torch.argmax(link_masked).item())
        accepted[j] = True
        link = torch.maximum(link, sub[:, j])
    return order[accepted]


def degree_order(cand, coupling, n, descending=True):
    """Commit by contact DEGREE, ignoring confidence.

    descending=True reproduces the "most constrained first" prescription of
    the contact-order inverse-folding work; False is its mirror.
    """
    n = int(min(n, cand.numel()))
    if n <= 0:
        return cand[:0]
    if coupling is None:
        return cand[:n]
    deg = coupling[cand][:, cand].sum(1)
    idx = torch.argsort(deg, descending=descending)[:n]
    return cand[idx]


SELECTORS = SELECTORS + ("anti_contact", "contact_degree",
                         "contact_oracle", "contact_oracle_shuffled",
                         "anti_contact_oracle")
