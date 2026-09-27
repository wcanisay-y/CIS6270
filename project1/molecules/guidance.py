"""Guidance for the flow model, and the proposed innovation.

BASELINE GUIDANCE (lecture 3.4; Ho & Salimans 2022; Zheng et al. 2023, ref [8]).
Classifier-free guidance applies the same generic field equation to whatever
the network outputs, so for flow matching it acts on the velocity:

    v_CFG = v_theta(x, t, null) + w [ v_theta(x, t, c) - v_theta(x, t, null) ]

Guidance here is classifier-free only. Reward guidance through a learned
property network is deliberately not implemented: it needs a second trained
model, doubles the failure surface, and its strength schedule kappa(t) is an
unconstrained design choice for flow matching. constraint_grad below supplies
the one gradient term the ablation needs, analytically and without training.

THE INNOVATION (feasibility-gated guidance).
Both rules above apply the same strength to every atom. Atoms whose local
geometry is already chemically consistent get pushed exactly as hard as atoms
that are still infeasible, which is why molecule stability degrades faster than
atom stability as guidance strengthens. We replace the scalar strength with a
per-atom gate read off the endpoint estimate the flow model already provides:

    xhat_1 = x_t + (1-t) v_theta          (lecture 3.4, endpoint guidance)
    Vhat_i = soft valency of atom i in xhat_1
    r_i    = Vhat_i - V(z_i)              constraint residual
    g_i    = exp(-beta r_i^2)             gate in (0, 1]
    v_i    = v_null,i + w g_i (v_cond,i - v_null,i)

g_i depends only on interatomic distances and atom-type probabilities, so it is
E(3)-invariant; an invariant scalar times an equivariant vector stays
equivariant. No extra network is trained: the bond-length table is the same one
the stability metric uses, and the gate needs no backward pass.

RELATION TO PRIOR WORK ON SOFT VALENCY.
Awasthi et al. (HLTF, ICLR 2026 workshop) define a soft degree
deg_i = sum_j <omega, x^(ij)> over GENERATED bond-type probabilities, an
expected max valence d_max_bar = sum_c alpha_c d_max(c) over soft atom types,
and a valence energy ReLU(deg_i - d_max_bar)^2 added to the sampler as an
annealed energy gradient. Three differences here:

  1. Their x^(ij) is a generative variable; this model generates coordinates
     only, so the soft bond order is produced from geometry (a softmax over
     distance to reference bond lengths) rather than from predicted logits.
  2. Their penalty is one-sided. QM9 molecule stability requires an exact
     valency match, so an under-bonded atom is as unstable as an over-bonded
     one; the default here is two-sided. Pass one_sided=True to reproduce
     their form.
  3. Theirs is ADDITIVE, with a time-annealed weight eta(t) = eta_0 (1-t)^gamma.
     The gate is MULTIPLICATIVE and state-dependent, and modifies only the
     magnitude of the guidance, not its direction. Their reported result that
     the additive form trades valence failures for geometry failures is the
     motivation for trying the multiplicative one.
"""
import torch

from flow_matching import zero_com

GATE_MODES = ("none", "global", "shuffled", "per_atom")

# Reference bond lengths in Angstrom, indexed (H, C, N, O, F); 0 = no such bond.
_B1 = [[0.74, 1.09, 1.01, 0.96, 0.92],
       [1.09, 1.54, 1.47, 1.43, 1.35],
       [1.01, 1.47, 1.45, 1.40, 1.36],
       [0.96, 1.43, 1.40, 1.48, 1.42],
       [0.92, 1.35, 1.36, 1.42, 1.42]]
_B2 = [[0.00, 0.00, 0.00, 0.00, 0.00],
       [0.00, 1.34, 1.29, 1.20, 0.00],
       [0.00, 1.29, 1.25, 1.21, 0.00],
       [0.00, 1.20, 1.21, 1.21, 0.00],
       [0.00, 0.00, 0.00, 0.00, 0.00]]
_B3 = [[0.00, 0.00, 0.00, 0.00, 0.00],
       [0.00, 1.20, 1.16, 1.13, 0.00],
       [0.00, 1.16, 1.10, 0.00, 0.00],
       [0.00, 1.13, 0.00, 0.00, 0.00],
       [0.00, 0.00, 0.00, 0.00, 0.00]]
VALENCE = [1.0, 4.0, 3.0, 2.0, 1.0]

# soft-valency hyperparameters
TAU = 0.10          # bond-length tolerance, Angstrom
C0 = 0.28           # effective "no bond" offset
TYPE_TEMP = 0.20    # temperature converting continuous h into type probabilities


def _tables(device):
    b = [torch.tensor(t, device=device) for t in (_B1, _B2, _B3)]
    return b, [(t > 0).float() for t in b], torch.tensor(VALENCE, device=device)


def endpoint(x, h, vx, vh, t):
    """xhat_1 = x_t + (1-t) v ; the flow model's guess at the finished molecule."""
    f = (1.0 - t).view(-1, 1, 1)
    return x + f * vx, h + f * vh


def valency_residual(x1, h1, mask, h_scale=1.0, one_sided=False):
    """Per-atom (soft valency - allowed valency), differentiable in x1 and h1.

    Bond order is a softmax over {no bond, single, double, triple} driven by how
    close the interatomic distance sits to the reference length for the two
    (soft) atom types. Atom types enter as probabilities, so a state whose type
    channels are still ambiguous produces a correspondingly soft valency.

    one_sided=True reproduces the HLTF form, penalising over-valence only.
    """
    bonds, avail, val = _tables(x1.device)
    n = x1.shape[1]
    eye = torch.eye(n, device=x1.device)[None]
    pair = mask[:, :, None] * mask[:, None, :] * (1 - eye)

    p = torch.softmax(h1 / (h_scale * TYPE_TEMP), dim=-1)            # [B,N,5]
    # cdist has a NaN gradient at zero distance (the diagonal), so build it here
    diff = x1[:, :, None, :] - x1[:, None, :, :]
    d = (diff.pow(2).sum(-1) + 1e-8).sqrt()                          # [B,N,N]

    w_none = torch.exp(torch.tensor(-(C0 ** 2) / (2 * TAU ** 2), device=x1.device))
    num, den = 0.0, w_none
    for order in (1, 2, 3):
        a = torch.einsum("bia,bjc,ac->bij", p, p, avail[order - 1])
        mu = torch.einsum("bia,bjc,ac->bij", p, p, bonds[order - 1]) / (a + 1e-6)
        w = a * torch.exp(-((d - mu) ** 2) / (2 * TAU ** 2))
        num, den = num + order * w, den + w
    order_soft = (num / den) * pair                                  # [B,N,N]
    r = order_soft.sum(-1) - (p * val).sum(-1)
    if one_sided:
        r = torch.relu(r)                     # over-valence only (HLTF form)
    return r * mask                                                  # [B,N]


def feasibility_gate(x1, h1, mask, mode="per_atom", beta=1.0, h_scale=1.0,
                     one_sided=False):
    """g in (0,1], shape [B,N,1]. No gradients are taken through the gate:
    it sets how hard to push, it is not an objective being ascended."""
    if mode == "none":
        return torch.ones_like(mask)[..., None]
    with torch.no_grad():
        r = valency_residual(x1, h1, mask, h_scale, one_sided)
        g = torch.exp(-beta * r ** 2) * mask
        if mode == "global":                       # one gate per molecule
            mean = g.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
            g = mean.expand_as(g) * mask
        elif mode == "shuffled":                   # mean-matched control
            key = torch.rand_like(g) + (1 - mask) * 10.0
            g = torch.gather(g, 1, key.argsort(dim=1))
        elif mode != "per_atom":
            raise ValueError(f"unknown gate mode {mode}")
    return (g * mask)[..., None]


def cfg(model, x, h, t, mask, cond, w, gate_mode="none", beta=1.0, h_scale=1.0,
        one_sided=False):
    """Classifier-free guided velocity, optionally feasibility-gated.

    Returns (vx, vh, gate) where gate is reported so runs can log the effective
    guidance strength; the w-matched control in the ablation needs it.
    """
    b = x.shape[0]
    off = torch.zeros(b, device=x.device)
    vx_u, vh_u = model.field(x, h, t, mask, cond, off)
    if w == 0:
        return vx_u, vh_u, torch.ones_like(mask)[..., None]
    on = torch.ones(b, device=x.device)
    vx_c, vh_c = model.field(x, h, t, mask, cond, on)
    if gate_mode == "none":
        g = torch.ones_like(mask)[..., None]
    else:
        x1, h1 = endpoint(x, h, vx_c, vh_c, t)     # gate read off the conditional endpoint
        g = feasibility_gate(x1, h1, mask, gate_mode, beta, h_scale, one_sided)
    vx = vx_u + w * g * (vx_c - vx_u)
    vh = vh_u + w * g * (vh_c - vh_u)
    return zero_com(vx, mask), vh * mask[..., None], g


def constraint_grad(x, h, t, mask, one_sided=False, h_scale=1.0):
    """grad_x of -sum_i r_i^2, the ADDITIVE form of the valency constraint.

    This is the lecture's soft-constraint term (R_lambda - sum_j rho_j h_j^2)
    and HLTF's valence energy, evaluated on the endpoint estimate. It exists
    only so the ablation can compare the additive mechanism against the
    multiplicative gate on the same residual; it is not part of the proposed
    method. No network is involved, so nothing extra is trained.
    """
    with torch.enable_grad():
        xs, hs = x.detach().requires_grad_(True), h.detach().requires_grad_(True)
        r = valency_residual(xs, hs, mask, h_scale, one_sided)
        energy = -(r ** 2 * mask).sum()
        gx, gh = torch.autograd.grad(energy, [xs, hs])
    return zero_com(gx.detach(), mask), gh.detach() * mask[..., None]