"""Diffusion model.

Parametrization: EGNN

Predicts noise prediction epsilon.
"""
import torch
from torch import nn

from flow_matching import masked_mse, sample_prior, Field

class DiffusionModel(nn.Module):
    """x_k = sqrt(abar_k) x_0 + sqrt(1-abar_k) eps ;  target is eps."""

    def __init__(self, nf=128, layers=6, cond_drop=0.1, steps=1000):
        super().__init__()
        self.field = Field(nf, layers)
        self.cond_drop = cond_drop
        self.K = steps
        betas = torch.cat([torch.zeros(1), torch.linspace(1e-4, 0.02, steps)])
        alphas = 1.0 - betas
        abar = alphas.cumprod(0)
        prev = torch.cat([torch.ones(1), abar[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.register_buffer("post_var", betas * (1 - prev) / (1 - abar).clamp_min(1e-20))

    def loss(self, x0, h0, mask, cond):
        b = x0.shape[0]
        k = torch.randint(1, self.K + 1, (b,), device=x0.device)
        a = self.abar[k].view(b, 1, 1)
        ex, eh = sample_prior(mask, x0.device)      # eps, zero-CoM on coordinates
        xk = a.sqrt() * x0 + (1 - a).sqrt() * ex
        hk = a.sqrt() * h0 + (1 - a).sqrt() * eh
        on = (torch.rand(b, device=x0.device) > self.cond_drop).float()
        px, ph = self.field(xk, hk, k.float() / self.K, mask, cond, on)
        return masked_mse(px, ex, mask) + masked_mse(ph, eh, mask)
