"""E(n)-equivariant GNN (Satorras et al. 2021), dense/masked, fully connected.

One backbone serves the flow model, the diffusion model and the property
network, so the Modality 1 comparison is not confounded by architecture.

Equivariance: coordinate updates are built only from difference vectors
(x_i - x_j) scaled by functions of ||x_i - x_j||, so they rotate with the input.
Node features are built only from distances, so they are invariant.
"""
import torch
from torch import nn


def mlp(i, o, nf):
    return nn.Sequential(nn.Linear(i, nf), nn.SiLU(), nn.Linear(nf, o))


class EGNNLayer(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.edge = mlp(2 * nf + 1, nf, nf)
        self.att = nn.Sequential(nn.Linear(nf, 1), nn.Sigmoid())
        self.coord = mlp(nf, 1, nf)
        self.node = mlp(2 * nf, nf, nf)

    def forward(self, h, x, pair_mask):
        diff = x[:, :, None, :] - x[:, None, :, :]          # [B,N,N,3]
        d2 = (diff ** 2).sum(-1, keepdim=True)              # [B,N,N,1]
        n = h.shape[1]
        hi = h[:, :, None].expand(-1, -1, n, -1)
        hj = h[:, None].expand(-1, n, -1, -1)
        m = self.edge(torch.cat([hi, hj, d2], -1))
        m = m * self.att(m) * pair_mask[..., None]
        # normalising by (d + 1) keeps the coordinate update stable at short range;
        # the epsilon keeps d/dx sqrt finite on the diagonal, where d2 = 0
        upd = diff / ((d2 + 1e-8).sqrt() + 1.0) * self.coord(m)
        x = x + (upd * pair_mask[..., None]).sum(2) / max(n - 1, 1)
        h = h + self.node(torch.cat([h, m.sum(2)], -1))
        return h, x


class EGNN(nn.Module):
    """Returns (dx, h_out): an equivariant displacement and invariant features."""

    def __init__(self, in_dim, nf=128, layers=6):
        super().__init__()
        self.embed = nn.Linear(in_dim, nf)
        self.layers = nn.ModuleList(EGNNLayer(nf) for _ in range(layers))
        self.out = nn.Linear(nf, nf)

    def forward(self, x, h, mask):
        n = h.shape[1]
        eye = torch.eye(n, device=h.device)[None]
        pair_mask = mask[:, :, None] * mask[:, None, :] * (1 - eye)
        x0 = x
        h = self.embed(h) * mask[..., None]
        for layer in self.layers:
            h, x = layer(h, x, pair_mask)
            h = h * mask[..., None]
            x = x * mask[..., None]
        return (x - x0) * mask[..., None], self.out(h) * mask[..., None]