"""Flow matching model.

Parametrization: EGNN

Predicts velocity v_theta.
"""
import torch
from torch import nn

from egnn import EGNN

N_TYPES = 5     # number of atom types
COND_DIM = 2    # conditions on a property: (property_value, is_present); (0, 0) gives v_unconditioned in CFG

def zero_com(v, mask):
    # translate non-masked atoms to have zero center-of-mass
    n = mask.sum(-1, keepdim=True).clamp_min(1)[..., None]
    mean = (v * mask[..., None]).sum(-2, keepdim=True) / n
    return (v - mean) * mask[..., None]


def masked_mse(pred, target, mask):
    # mse only for non-masked atoms, averaged over all atoms and batch
    diff = (pred - target) ** 2 * mask[..., None]
    return diff.sum() / (mask.sum() * pred.shape[-1]).clamp_min(1)


class Field(nn.Module):
    """
    input (x, h, t, c) -> vector field parameters with input (x, h). Shared by both models.
        x: atom positions
        h: atom features (one-hot atom type)
        t: time (0-1)
        c: condition (property value, is_present)
    """

    def __init__(self, nf=128, layers=6):
        super().__init__()
        self.net = EGNN(N_TYPES + 1 + COND_DIM, nf, layers)
        self.head = nn.Linear(nf, N_TYPES)

    def forward(self, x, h, t, mask, cond, cond_on):
        b, n, _ = h.shape
        t = t.view(b, 1, 1).expand(b, n, 1)
        c = (cond * cond_on).view(b, 1, 1).expand(b, n, 1) # set null if cond_on=0
        on = cond_on.view(b, 1, 1).expand(b, n, 1)
        feats = torch.cat([h, t, c, on], -1)
        dx, hout = self.net(x, feats, mask)
        return zero_com(dx, mask), self.head(hout) * mask[..., None]


def sample_prior(mask, device):
    # returns noisy samples from prob. dist. at t=0 (x0, h0) for flow matching and t=1 (x1, h1) for diffusion
    b, n = mask.shape
    x0 = zero_com(torch.randn(b, n, 3, device=device), mask)            # random 3D coords with zero center-of-mass
    h0 = torch.randn(b, n, N_TYPES, device=device) * mask[..., None]    # random one-hot atom type features (masked atoms are zero)
    return x0, h0


class FlowModel(nn.Module):
    """
    Uses linear time interpolation to define flow: x_t = (1-t) x_0 + t x_1
    target velocity is x_1 - x_0.
    """

    def __init__(self, nf=128, layers=6, cond_drop=0.1):
        super().__init__()
        self.field = Field(nf, layers)
        self.cond_drop = cond_drop

    def loss(self, x1, h1, mask, cond):
        b = x1.shape[0]
        x0, h0 = sample_prior(mask, x1.device)  # noisy data samples
        t = torch.rand(b, device=x1.device)
        tt = t.view(b, 1, 1)
        xt = (1 - tt) * x0 + tt * x1
        ht = (1 - tt) * h0 + tt * h1
        on = (torch.rand(b, device=x1.device) > self.cond_drop).float()
        vx, vh = self.field(xt, ht, t, mask, cond, on)
        return masked_mse(vx, x1 - x0, mask) + masked_mse(vh, h1 - h0, mask)


class PropertyNet(nn.Module):
    """Predicts the standardized property from a (possibly noisy) state.

    Used twice: as the differentiable reward for gradient guidance (trained on
    intermediate states, t ~ U[0,1]), and as the evaluation regressor for the
    section 4.3 target metric (trained with --clean-only, i.e. t = 1).
    """

    def __init__(self, nf=128, layers=4):
        super().__init__()
        self.net = EGNN(N_TYPES + 1, nf, layers)
        self.head = nn.Linear(nf, 1)

    def forward(self, x, h, t, mask):
        b, n, _ = h.shape
        feats = torch.cat([h, t.view(b, 1, 1).expand(b, n, 1)], -1)
        _, hout = self.net(x, feats, mask)
        pooled = (self.head(hout).squeeze(-1) * mask).sum(-1)
        return pooled / mask.sum(-1).clamp_min(1)

    def loss(self, x1, h1, mask, cond, clean_only=False):
        b = x1.shape[0]
        if clean_only:
            return ((self(x1, h1, torch.ones(b, device=x1.device), mask) - cond) ** 2).mean()
        x0, h0 = sample_prior(mask, x1.device)
        t = torch.rand(b, device=x1.device)
        tt = t.view(b, 1, 1)
        xt, ht = (1 - tt) * x0 + tt * x1, (1 - tt) * h0 + tt * h1
        return ((self(xt, ht, t, mask) - cond) ** 2).mean()