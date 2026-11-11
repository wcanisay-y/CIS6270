"""Flat and two-level masked diffusion models over protein sequences (Idea B).

Both arms share one backbone, one parameter budget and one objective family,
so the only difference is the state space.

FLAT (baseline, MDLM/absorbing):
    a position is either an amino acid or [MASK].
    loss = E_t (1/t) * sum_{masked} -log p(a_i | x_t)

TWO-LEVEL (the proposal):
    a position is in one of three states
        both masked        family unknown, identity unknown
        family revealed    family known,  identity unknown
        revealed           both known
    Identity determines family, so "family masked but identity revealed" is
    not a reachable state and is never constructed.
    The family is masked with probability p_fam = p_id ** gamma, gamma >= 1,
    so the family channel is revealed earlier than the identity channel --
    the asymmetric schedule the proposal asks for.
    loss = E_t [ (1/p_id) sum_{id masked} -log p(a_i | .)
               + (1/p_fam) sum_{fam masked} -log p(f_i | .) ]

Binding: when the family is known, the identity head's logits are restricted
to that family's members. HDLM (arXiv:2510.08632) found a hard restriction
was its worst setting, so bindingness is a parameter (soft_xi < 1 trains with
occasionally wrong family labels) rather than an axiom.

NOTE ON THE OBJECTIVE. Each level uses the MDLM continuous-time weighting
applied to its own schedule. That is a per-level bound, not HDLM's exact
hierarchical CTMC ELBO, so validation losses are NOT comparable across arms
as likelihoods. Arms are compared on generation quality and on masked
cross-entropy at a fixed grid of mask rates, both of which are comparable.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_AA = 20
ID_MASK = 20            # identity-masked symbol
FAM_MASK_OFFSET = 1     # family-masked symbol is index K


class Block(nn.Module):
    def __init__(self, d, heads, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class MDM(nn.Module):
    """One backbone, two heads. `n_fam=0` gives the flat baseline."""

    def __init__(self, d=512, layers=12, heads=8, max_len=256, n_fam=0,
                 drop=0.0, fam_lut=None):
        super().__init__()
        self.n_fam = n_fam
        self.max_len = max_len
        self.id_emb = nn.Embedding(N_AA + 1, d)          # 20 aa + id-mask
        if n_fam:
            self.fam_emb = nn.Embedding(n_fam + 1, d)    # families + fam-mask
            self.register_buffer(
                "fam_lut", torch.as_tensor(fam_lut, dtype=torch.long))
            # (n_fam, 20) boolean membership, for the binding restriction
            memb = torch.zeros(n_fam, N_AA, dtype=torch.bool)
            for a in range(N_AA):
                memb[int(fam_lut[a]), a] = True
            self.register_buffer("memb", memb)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop)
                                     for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.id_head = nn.Linear(d, N_AA)
        self.fam_head = nn.Linear(d, n_fam) if n_fam else None
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def trunk(self, ids, fam=None, pad_mask=None):
        B, L = ids.shape
        x = self.id_emb(ids) + self.pos(
            torch.arange(L, device=ids.device))[None]
        if self.n_fam:
            x = x + self.fam_emb(fam)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        return self.ln_f(x)

    def forward(self, ids, fam=None, pad_mask=None, bind=True):
        h = self.trunk(ids, fam, pad_mask)
        id_logits = self.id_head(h)
        fam_logits = self.fam_head(h) if self.n_fam else None
        if self.n_fam and bind:
            known = fam < self.n_fam
            allow = torch.ones_like(id_logits, dtype=torch.bool)
            allow[known] = self.memb[fam[known]]
            id_logits = id_logits.masked_fill(~allow, -1e4)
        return id_logits, fam_logits


# --------------------------------------------------------------- corruption
def corrupt_flat(aa, t, gen=None):
    """aa (B, L) in [0,20); -> ids with ID_MASK, and the masked indicator."""
    p = t[:, None]
    m = torch.rand(aa.shape, device=aa.device, generator=gen) < p
    ids = aa.masked_fill(m, ID_MASK)
    return ids, m


def corrupt_hier(aa, fam_lut, n_fam, t, gamma=2.0, soft_xi=1.0, gen=None):
    """Build the three-state corruption.

    Returns ids, fam, id_masked, fam_masked. `soft_xi` < 1 replaces a
    revealed family label by a wrong one with probability 1 - soft_xi, which
    is HDLM's soft hierarchy applied at training time.
    """
    p_id = t[:, None]
    p_fam = p_id ** gamma
    u = torch.rand(aa.shape, device=aa.device, generator=gen)
    id_masked = u < p_id
    fam_masked = u < p_fam                     # nested, so fam_masked => id_masked
    ids = aa.masked_fill(id_masked, ID_MASK)
    fam_true = fam_lut[aa]
    fam = fam_true.clone()
    fam[fam_masked] = n_fam                    # the family-masked symbol
    if soft_xi < 1.0:
        shown = ~fam_masked
        flip = (torch.rand(aa.shape, device=aa.device, generator=gen)
                < (1.0 - soft_xi)) & shown
        if flip.any():
            wrong = torch.randint(0, n_fam, (int(flip.sum()),),
                                  device=aa.device, generator=gen)
            fam[flip] = wrong
    return ids, fam, id_masked, fam_masked


# -------------------------------------------------------------------- losses
def loss_flat(model, aa, pad_mask=None, eps=1e-3, gen=None):
    B = aa.shape[0]
    t = torch.rand(B, device=aa.device, generator=gen) * (1 - eps) + eps
    ids, m = corrupt_flat(aa, t, gen)
    if pad_mask is not None:
        m = m & ~pad_mask
    logits, _ = model(ids, pad_mask=pad_mask)
    ce = F.cross_entropy(logits.transpose(1, 2), aa, reduction="none")
    w = (1.0 / t)[:, None]
    n = m.sum().clamp_min(1)
    return (ce * m * w).sum() / n, {"ce_id": float(
        (ce * m).sum().item() / n.item())}


def loss_hier(model, aa, fam_lut, n_fam, pad_mask=None, gamma=2.0,
              soft_xi=1.0, eps=1e-3, bind=True, gen=None):
    B = aa.shape[0]
    t = torch.rand(B, device=aa.device, generator=gen) * (1 - eps) + eps
    ids, fam, idm, fm = corrupt_hier(aa, fam_lut, n_fam, t, gamma, soft_xi,
                                     gen)
    if pad_mask is not None:
        idm = idm & ~pad_mask
        fm = fm & ~pad_mask
    id_logits, fam_logits = model(ids, fam, pad_mask=pad_mask, bind=bind)
    ce_id = F.cross_entropy(id_logits.transpose(1, 2), aa, reduction="none")
    fam_true = fam_lut[aa]
    ce_fam = F.cross_entropy(fam_logits.transpose(1, 2), fam_true,
                             reduction="none")
    p_id = t[:, None]
    p_fam = p_id ** gamma
    n_id = idm.sum().clamp_min(1)
    n_fam_m = fm.sum().clamp_min(1)
    l_id = (ce_id * idm / p_id).sum() / n_id
    l_fam = (ce_fam * fm / p_fam).sum() / n_fam_m
    return l_id + l_fam, {"ce_id": float((ce_id * idm).sum().item()
                                         / n_id.item()),
                          "ce_fam": float((ce_fam * fm).sum().item()
                                          / n_fam_m.item())}


# ------------------------------------------------------------- AR baseline
class ARLM(nn.Module):
    """Causal transformer at matched size. DiMA reports AR beats from-scratch
    discrete diffusion by a wide margin on SwissProt at 35M, so this arm is
    required for an honest comparison."""

    def __init__(self, d=512, layers=12, heads=8, max_len=256, drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(N_AA + 1, d)       # +1 = BOS
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop)
                                     for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_AA)
        self.max_len = max_len
        self.apply(MDM._init)

    def forward(self, ids):
        B, L = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        cm = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool),
                        diagonal=1)
        for blk in self.blocks:
            h = blk.ln1(x)
            a, _ = blk.attn(h, h, h, attn_mask=cm, need_weights=False)
            x = x + a
            x = x + blk.mlp(blk.ln2(x))
        return self.head(self.ln_f(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
