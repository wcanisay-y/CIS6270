"""Flow matching model.

Parametrization: EGNN

Predicts velocity v_theta.
"""
import torch
from torch import nn
import numpy as np
from scipy.optimize import linear_sum_assignment

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


def ot_coupling(x0, h0, x1, h1, mask):
    """Reorder noise samples (x0, h0) to minimize transport cost to data (x1, h1).

    Uses minibatch optimal transport: computes pairwise cost matrix based on
    squared Euclidean distance on coordinates, then solves the assignment problem.

    Args:
        x0: noise coordinates [B, N, 3]
        h0: noise features [B, N, N_TYPES]
        x1: data coordinates [B, N, 3]
        h1: data features [B, N, N_TYPES]
        mask: atom mask [B, N]

    Returns:
        Reordered (x0, h0) so each noise sample is paired with nearby data sample.

    Note on extensions:
        The current implementation uses plain squared distance on coordinates.
        To additionally optimize over:

        1. Atom permutation: For each (i,j) pair, solve a secondary assignment
           problem over atoms within the molecules. This requires computing
           atom-to-atom costs (considering both position and type), running
           linear_sum_assignment for each pair, and using the permuted cost.
           Complexity: O(B^2 * N^3) for Hungarian on atoms.

        2. Rotation alignment: For each (i,j) pair, find the optimal rotation
           using Kabsch algorithm (SVD of cross-covariance matrix) before
           computing distance. This requires: centering both point clouds,
           computing H = x0^T @ x1, SVD to get U, V, and rotation R = V @ U^T.
           Handle reflections by checking det(R) and flipping if needed.
           Complexity: O(B^2 * N) for Kabsch per pair.

        Combining both: For each (i,j), iterate: (1) find optimal rotation given
        current atom assignment, (2) find optimal atom assignment given rotation.
        This alternating optimization converges but may find local minima.
    """
    b = x0.shape[0]
    device = x0.device

    # Compute pairwise squared distance cost matrix [B, B]
    # Cost(i, j) = sum over atoms of ||x0[i, a, :] - x1[j, a, :]||^2, weighted by mask
    # We need to handle different molecule sizes via masking

    # Expand for pairwise computation: [B, 1, N, 3] vs [1, B, N, 3]
    x0_exp = x0.unsqueeze(1)  # [B, 1, N, 3]
    x1_exp = x1.unsqueeze(0)  # [1, B, N, 3]

    # Squared distances per atom: [B, B, N]
    sq_dist = ((x0_exp - x1_exp) ** 2).sum(-1)  # [B, B, N]

    # For masking: we use the intersection of masks (atoms valid in both molecules)
    # mask_i AND mask_j for each pair
    mask0_exp = mask.unsqueeze(1)  # [B, 1, N]
    mask1_exp = mask.unsqueeze(0)  # [1, B, N]
    pair_mask = mask0_exp * mask1_exp  # [B, B, N]

    # Sum over atoms, normalize by number of valid atoms in the pair
    n_valid = pair_mask.sum(-1).clamp_min(1)  # [B, B]
    cost = (sq_dist * pair_mask).sum(-1) / n_valid  # [B, B], averaged squared distance

    # Solve assignment problem using scipy
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)

    # Reorder noise samples according to assignment
    # col_ind[i] tells us which noise sample should be paired with data sample i
    # We want: new_x0[i] = old_x0[col_ind[i]]
    perm = torch.tensor(col_ind, device=device, dtype=torch.long)
    x0_reordered = x0[perm]
    h0_reordered = h0[perm]

    return x0_reordered, h0_reordered


class FlowModel(nn.Module):
    """
        Uses linear time interpolation to define flow: x_t = (1-t) x_0 + t x_1
        target velocity is x_1 - x_0.

    Args:
        coupling: 'independent' (default) or 'ot' for minibatch optimal transport
    """

    def __init__(self, nf=128, layers=6, cond_drop=0.1, coupling='independent'):
        super().__init__()
        self.field = Field(nf, layers)
        self.cond_drop = cond_drop
        self.coupling = coupling

    def loss(self, x1, h1, mask, cond):
        b = x1.shape[0]
        x0, h0 = sample_prior(mask, x1.device)  # noisy data samples

        # Apply OT coupling if requested
        if self.coupling == 'ot':
            x0, h0 = ot_coupling(x0, h0, x1, h1, mask)

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