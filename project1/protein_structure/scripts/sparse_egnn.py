#!/usr/bin/env python3
"""Sparse SE(3)-equivariant denoiser and the diffusion process for Ca backbone generation.

Independent of the QM9 baseline in ../../molecular_coordinates/diffusion_baseline. The diffusion
recipe is deliberately the same one (polynomial_2 variance-preserving schedule, zero-centre-of-mass
subspace, epsilon regression, residue one-hots on a smaller scale than coordinates) so the protein
arm and the QM9 arm stay comparable. Four things had to change for proteins, and each is marked
ARCH below, because none of them is a hyperparameter:

  ARCH 1  Sparse edges. The QM9 layer materialises a dense [B, N, N, 2*hidden+1] pair tensor.
          At 128 nodes that OOMs above batch 32 on a 40 GB card and runs about 15x slower than
          it needs to. Real Ca graphs average 9 neighbours within 8 A and 16 within 10 A, so the
          edges are built as k nearest spatial neighbours plus the two backbone bonds.

  ARCH 2  Chirality. Messages that see only squared distances are invariant to reflection, so
          such a model cannot tell a right-handed helix from a left-handed one, and every real
          protein helix is right-handed. A normalised scalar triple product over four consecutive
          Ca atoms is a pseudoscalar: invariant under rotation, sign-flipped under reflection.
          Feeding it in as a node scalar makes the network SE(3)-equivariant rather than E(3).

  ARCH 3  Chain order. A protein is a sequence, not a set. The QM9 node features carry only atom
          type and time, so nothing says residue i+1 must sit 3.8 A from residue i. Backbone
          edges carry an explicit flag and residue index enters through a sinusoidal encoding.

  ARCH 4  Evaluation. Bond-order tables, valence stability and SMILES do not apply to a Ca trace.
          See evaluate_protein.py.
"""
import math

import torch
from torch import nn

N_RESIDUE_TYPES = 20
TYPE_SCALE = 0.25          # EDM's normalize_factor for categorical features
POS_DIM = 16               # sinusoidal residue-index encoding width


# --------------------------------------------------------------------------- geometry helpers
def remove_mean(x, mask):
    """Project coordinates onto the zero-centre-of-mass subspace, per protein."""
    count = mask.sum(1, keepdim=True).clamp(min=1)
    mean = (x * mask).sum(1, keepdim=True) / count
    return (x - mean) * mask


def sample_zero_com_noise(shape, mask, device):
    return remove_mean(torch.randn(shape, device=device), mask)


def positional_encoding(n_nodes, device):
    """Sinusoidal encoding of residue index, shaped [n_nodes, POS_DIM]."""
    position = torch.arange(n_nodes, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, POS_DIM, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / POS_DIM))
    out = torch.zeros(n_nodes, POS_DIM, device=device)
    out[:, 0::2] = torch.sin(position * div)
    out[:, 1::2] = torch.cos(position * div)
    return out


def chirality_feature(x, mask):
    """ARCH 2. Normalised scalar triple product over four consecutive Ca atoms.

    For residue i the vectors are u = x[i]-x[i-1], v = x[i+1]-x[i], w = x[i+2]-x[i+1], and the
    feature is det[u, v, w] / (|u| |v| |w|). A rotation leaves it unchanged because det(R) = 1;
    a reflection negates it because det(R) = -1. Residues without the full window get zero.
    """
    batch, n_nodes, _ = x.shape
    pad = torch.zeros(batch, 1, 3, device=x.device, dtype=x.dtype)
    prev = torch.cat([pad, x[:, :-1]], 1)                 # x[i-1]
    nxt = torch.cat([x[:, 1:], pad], 1)                   # x[i+1]
    nxt2 = torch.cat([x[:, 2:], pad, pad], 1)             # x[i+2]
    u, v, w = x - prev, nxt - x, nxt2 - nxt
    triple = (torch.cross(v, w, dim=-1) * u).sum(-1, keepdim=True)
    norm = u.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True) * w.norm(dim=-1, keepdim=True)
    chi = triple / (norm + 1e-6)
    # The window needs i-1, i+1 and i+2 to all be real residues.
    valid = mask.clone()
    valid[:, 0] = 0
    valid[:, -2:] = 0
    shifted = torch.cat([mask[:, 2:], torch.zeros(batch, 2, 1, device=mask.device)], 1)
    return chi * valid * shifted


def build_edges(x, mask, lengths, k=16):
    """ARCH 1 + ARCH 3. Flat edge lists over a batch packed as [B*N] nodes.

    Returns (src, dst, is_backbone). Backbone pairs are removed from the spatial candidates and
    added back explicitly, so no edge is counted twice and the flag is unambiguous.
    """
    batch, n_nodes, _ = x.shape
    device = x.device
    offset = (torch.arange(batch, device=device) * n_nodes)[:, None, None]

    dist = torch.cdist(x.float(), x.float())
    pair_valid = (mask.squeeze(-1)[:, :, None] * mask.squeeze(-1)[:, None, :]).bool()
    dist = dist.masked_fill(~pair_valid, float("inf"))
    index = torch.arange(n_nodes, device=device)
    adjacent = (index[:, None] - index[None, :]).abs() <= 1        # self and both backbone bonds
    dist = dist.masked_fill(adjacent[None], float("inf"))

    k_use = min(k, max(n_nodes - 3, 1))
    value, neighbour = dist.topk(k_use, dim=-1, largest=False)
    finite = torch.isfinite(value) & mask.squeeze(-1).bool()[:, :, None]
    src_spatial = (index[None, :, None].expand(batch, n_nodes, k_use) + offset)[finite]
    dst_spatial = (neighbour + offset)[finite]

    # Backbone edges i -> i+1 and i+1 -> i, only where both residues exist.
    node = torch.arange(n_nodes - 1, device=device)
    both = (mask.squeeze(-1)[:, :-1] * mask.squeeze(-1)[:, 1:]).bool()
    a = (node[None, :].expand(batch, n_nodes - 1) + offset.squeeze(-1))[both]
    b = a + 1
    src_bb = torch.cat([a, b])
    dst_bb = torch.cat([b, a])

    src = torch.cat([src_spatial, src_bb])
    dst = torch.cat([dst_spatial, dst_bb])
    is_bb = torch.cat([torch.zeros(src_spatial.numel(), device=device),
                       torch.ones(src_bb.numel(), device=device)])[:, None]
    return src, dst, is_bb


# --------------------------------------------------------------------------- the denoiser
class SparseEGNNLayer(nn.Module):
    """Same message, attention and coordinate update as the QM9 EGNN layer, gathered over an
    edge list instead of a dense pair tensor. The extra edge input is the backbone flag."""

    def __init__(self, hidden):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.node = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
        self.coord = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 1, bias=False))
        self.attention = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        nn.init.xavier_uniform_(self.coord[-1].weight, gain=0.001)   # start near the identity

    def forward(self, h, x, src, dst, is_bb, node_mask):
        difference = x[src] - x[dst]
        square = (difference ** 2).sum(-1, keepdim=True)
        message = self.edge(torch.cat([h[src], h[dst], square, is_bb], dim=-1))
        message = message * self.attention(message)
        weight = self.coord(message)
        # The 1e-8 under the square root is load-bearing: d(sqrt)/dx is infinite at zero and
        # coincident points would make the backward pass NaN.
        contribution = difference / ((square + 1e-8).sqrt() + 1.0) * weight
        # Every cast back to h.dtype and x.dtype is load-bearing under autocast. The linear
        # layers emit bfloat16, the masks are float32, and bfloat16 * float32 promotes to
        # float32. Without these casts the hidden state silently changes dtype after the first
        # layer and the next index_add_ dies with a scalar-type mismatch. Accumulating the
        # residual stream in float32 is also the numerically better choice.
        shift = torch.zeros_like(x).index_add_(0, src, contribution.to(x.dtype))
        aggregate = torch.zeros_like(h).index_add_(0, src, message.to(h.dtype))
        h = h + self.node(torch.cat([h, aggregate], dim=-1)).to(h.dtype) * node_mask.to(h.dtype)
        x = x + shift * node_mask.to(x.dtype)
        return h, x


class ProteinDenoiser(nn.Module):
    """Predicts the noise on Ca coordinates and on scaled residue one-hots."""

    def __init__(self, hidden=256, layers=9, k=16, use_chirality=True):
        super().__init__()
        self.k = k
        # use_chirality=False reproduces the reflection-invariant E(3) behaviour of the QM9
        # denoiser. It exists so the ablation can be measured, not as a recommended setting.
        self.use_chirality = use_chirality
        # residue one-hots, time, the chirality pseudoscalar, and the residue-index encoding
        self.embed = nn.Linear(N_RESIDUE_TYPES + 1 + 1 + POS_DIM, hidden)
        self.layers = nn.ModuleList(SparseEGNNLayer(hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, N_RESIDUE_TYPES)

    def forward(self, z_x, z_h, t, mask, lengths):
        batch, n_nodes, _ = z_x.shape
        src, dst, is_bb = build_edges(z_x, mask, lengths, self.k)
        chi = chirality_feature(z_x, mask) if self.use_chirality else torch.zeros_like(mask)
        pos = positional_encoding(n_nodes, z_x.device)[None].expand(batch, -1, -1)
        time = t.view(-1, 1, 1).expand(-1, n_nodes, 1)
        features = torch.cat([z_h, time, chi, pos], dim=-1) * mask

        h = (self.embed(features) * mask).reshape(batch * n_nodes, -1)
        x = z_x.reshape(batch * n_nodes, 3)
        node_mask = mask.reshape(batch * n_nodes, 1)
        for layer in self.layers:
            h, x = layer(h, x, src, dst, is_bb, node_mask)

        x = x.view(batch, n_nodes, 3)
        h = h.view(batch, n_nodes, -1)
        # The coordinate output is a displacement; re-projecting keeps it in the zero-CoM subspace.
        return remove_mean(x - z_x, mask), self.head(h) * mask


# --------------------------------------------------------------------------- diffusion
def polynomial_schedule(timesteps, s=1e-5, power=2.0):
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alphas2 = (1 - (steps / timesteps) ** power) ** 2
    alphas2 = torch.clamp(alphas2, min=0.0)
    return ((1 - 2 * s) * alphas2 + s).float()


class Diffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.T = timesteps
        alphas2 = polynomial_schedule(timesteps).to(device)
        self.alpha = alphas2.sqrt()
        self.sigma = (1 - alphas2).sqrt()

    def noise(self, x, h, mask, t_index):
        a = self.alpha[t_index].view(-1, 1, 1)
        s = self.sigma[t_index].view(-1, 1, 1)
        eps_x = sample_zero_com_noise(x.shape, mask, x.device)
        eps_h = torch.randn_like(h) * mask
        return (a * x + s * eps_x) * mask, (a * h + s * eps_h) * mask, eps_x, eps_h

    def reverse_step(self, x, h, eps_x, eps_h, mask, t_index, add_noise=True):
        """One ancestral transition from t_index to t_index - 1.

        Exposed separately because the only test with teeth is a distributional one: composing
        the forward marginal q(z_t | x0) with this transition must reproduce q(z_s | x0) exactly.
        An oracle reconstruction test does NOT catch coefficient errors here, because an oracle
        that knows x0 re-injects the true signal at every step and converges anyway.
        """
        s_index = t_index - 1
        a_t, s_t = self.alpha[t_index], self.sigma[t_index]
        a_s, s_s = self.alpha[s_index], self.sigma[s_index]
        alpha_t_given_s = a_t / a_s.clamp(min=1e-12)
        sigma2_t_given_s = (s_t ** 2 - alpha_t_given_s ** 2 * s_s ** 2).clamp(min=0)
        scale = sigma2_t_given_s / (alpha_t_given_s * s_t.clamp(min=1e-12))
        mu_x = x / alpha_t_given_s - scale * eps_x
        mu_h = h / alpha_t_given_s - scale * eps_h
        std = sigma2_t_given_s.sqrt() * s_s / s_t.clamp(min=1e-12)
        mu_x = remove_mean(mu_x, mask)
        if add_noise:
            mu_x = mu_x + std * sample_zero_com_noise(mu_x.shape, mask, mu_x.device)
            mu_h = mu_h + std * torch.randn_like(mu_h)
        return mu_x * mask, mu_h * mask

    @torch.no_grad()
    def sample(self, model, lengths, n_nodes, device, progress=None, eps_fn=None):
        """Ancestral sampling for the variance-preserving process.

        The posterior q(z_s | z_t, x0) for s < t has
            alpha_{t|s} = alpha_t / alpha_s,   sigma^2_{t|s} = sigma_t^2 - alpha_{t|s}^2 sigma_s^2
            mu  = z_t / alpha_{t|s} - (sigma^2_{t|s} / (alpha_{t|s} sigma_t)) * eps_hat
            std = sigma_{t|s} * sigma_s / sigma_t
        Both ratios are easy to invert by accident, and getting either wrong still yields finite
        zero-CoM output, so it cannot be caught by a smoke test. test_protein.py pins this down by
        running the loop with an oracle that returns the exact noise: a correct sampler then
        reconstructs the input, and a wrong one does not.

        eps_fn overrides the model, which is what the oracle test uses.
        """
        batch = len(lengths)
        mask = (torch.arange(n_nodes, device=device)[None] < lengths[:, None]).float()[..., None]
        x = sample_zero_com_noise((batch, n_nodes, 3), mask, device)
        h = torch.randn(batch, n_nodes, N_RESIDUE_TYPES, device=device) * mask
        for t_index in range(self.T, 0, -1):
            t = torch.full((batch,), t_index / self.T, device=device)
            if eps_fn is not None:
                eps_x, eps_h = eps_fn(x, h, t, mask, lengths, t_index)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    eps_x, eps_h = model(x, h, t, mask, lengths)
                eps_x, eps_h = eps_x.float(), eps_h.float()
            x, h = self.reverse_step(x, h, eps_x, eps_h, mask, t_index)
            if progress and t_index % 100 == 0:
                progress(t_index)
        # z_0 -> x0. The final noise term is dropped on purpose so the returned sample is the
        # posterior mean rather than one more noisy draw.
        t0 = torch.zeros(batch, device=device)
        if eps_fn is not None:
            eps_x, eps_h = eps_fn(x, h, t0, mask, lengths, 0)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eps_x, eps_h = model(x, h, t0, mask, lengths)
            eps_x, eps_h = eps_x.float(), eps_h.float()
        a_0, s_0 = self.alpha[0], self.sigma[0]
        x = remove_mean((x - s_0 * eps_x) / a_0.clamp(min=1e-12), mask) * mask
        h = ((h - s_0 * eps_h) / a_0.clamp(min=1e-12)) * mask
        return x, h, mask
