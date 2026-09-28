"""Flow matching model for protein sequences in ESM-2 latent space.

Parametrization: Small Transformer (no equivariance needed for latents)

Predicts velocity v_theta. Flow: x_t = (1-t)x0 + t*x1, target = x1 - x0.

Supports two flow matching variants:
- Standard: random pairing of noise and data (default)
- OT (Optimal Transport): mini-batch OT coupling for straighter paths
"""
import torch
from torch import nn
import math

from data.pfam import SEQ_LEN, HIDDEN_DIM, N_FAMILIES

COND_DIM = 64   # family embedding dimension


def mse_loss(pred, target):
    """MSE loss averaged over all dimensions."""
    return ((pred - target) ** 2).mean()


def ot_coupling(z0, z1):
    """Mini-batch optimal transport coupling.

    Given a batch of noise samples z0 and data samples z1,
    find the permutation of z0 that minimizes total transport cost.
    This leads to straighter flow paths and faster convergence.

    Uses the Hungarian algorithm via linear_sum_assignment.
    For large batches, falls back to greedy matching for speed.
    """
    from scipy.optimize import linear_sum_assignment

    b = z0.shape[0]
    if b == 1:
        return z0, z1

    # Flatten to [B, L*D] for distance computation
    z0_flat = z0.view(b, -1)
    z1_flat = z1.view(b, -1)

    # Compute pairwise squared distances
    # cost[i,j] = ||z0[i] - z1[j]||^2
    with torch.no_grad():
        cost = torch.cdist(z0_flat, z1_flat, p=2).pow(2)
        cost_np = cost.cpu().numpy()

        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # Permute z0 to match optimal coupling
        perm = torch.tensor(col_ind, device=z0.device)

    # Return permuted z0 (matched to z1)
    z0_matched = z0[torch.argsort(perm)]
    return z0_matched, z1


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional/time embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, dim, n_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class Field(nn.Module):
    """
    Vector field for flow matching on protein latents.

    Input:
        z: [B, L, 320] ESM-2 latent states
        t: [B] time in [0, 1]
        family: [B] family index (0 to N_FAMILIES-1)
        cond_on: [B] whether to condition (1) or use null (0) for CFG

    Output:
        v: [B, L, 320] velocity field
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, n_layers=4, n_heads=4, cond_dim=COND_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Family embedding (index -> vector)
        # Use N_FAMILIES + 1 to have a null embedding at index N_FAMILIES
        self.family_emb = nn.Embedding(N_FAMILIES + 1, cond_dim)

        # Project input + conditioning to hidden dim
        self.input_proj = nn.Linear(hidden_dim + cond_dim, hidden_dim)

        # Positional embedding for sequence
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z, t, family, cond_on):
        """
        Args:
            z: [B, L, 320] latent states
            t: [B] time
            family: [B] family index
            cond_on: [B] float, 1.0 for conditional, 0.0 for null

        Returns:
            v: [B, L, 320] velocity
        """
        b, l, d = z.shape

        # Time embedding -> broadcast to sequence
        t_emb = self.time_emb(t)  # [B, D]
        z = z + t_emb[:, None, :]  # Add time to all positions

        # Family embedding with null option
        # When cond_on=0, use null embedding (index N_FAMILIES)
        null_idx = torch.full_like(family, N_FAMILIES)
        family_idx = torch.where(cond_on.bool(), family, null_idx)
        fam_emb = self.family_emb(family_idx)  # [B, cond_dim]
        fam_emb = fam_emb[:, None, :].expand(b, l, -1)  # [B, L, cond_dim]

        # Concatenate and project
        x = torch.cat([z, fam_emb], dim=-1)  # [B, L, D + cond_dim]
        x = self.input_proj(x)  # [B, L, D]

        # Add positional embedding
        x = x + self.pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.norm(x)
        v = self.output_proj(x)

        return v


def sample_prior(batch_size, device):
    """Sample from prior distribution at t=0: standard Gaussian."""
    return torch.randn(batch_size, SEQ_LEN, HIDDEN_DIM, device=device)


class FlowModel(nn.Module):
    """
    Flow matching model for protein latents.

    Uses linear time interpolation: x_t = (1-t) x_0 + t x_1
    Target velocity is x_1 - x_0.

    Supports two variants:
    - use_ot=False (default): Random pairing of noise and data
    - use_ot=True: Optimal transport coupling for straighter paths
    """

    def __init__(self, n_layers=4, n_heads=4, cond_drop=0.1, use_ot=False):
        super().__init__()
        self.field = Field(n_layers=n_layers, n_heads=n_heads)
        self.cond_drop = cond_drop
        self.use_ot = use_ot

    def loss(self, z1, family):
        """
        Compute flow matching loss.

        Args:
            z1: [B, L, D] clean latent states (target)
            family: [B] family indices

        Returns:
            loss: scalar MSE loss
        """
        b = z1.shape[0]
        device = z1.device

        # Sample from prior
        z0 = sample_prior(b, device)

        # Apply OT coupling if enabled (straighter paths)
        if self.use_ot:
            z0, z1 = ot_coupling(z0, z1)

        # Random time
        t = torch.rand(b, device=device)
        tt = t.view(b, 1, 1)

        # Interpolate
        zt = (1 - tt) * z0 + tt * z1

        # Condition dropout for CFG
        cond_on = (torch.rand(b, device=device) > self.cond_drop).float()

        # Predict velocity
        v_pred = self.field(zt, t, family, cond_on)

        # Target velocity
        v_target = z1 - z0

        return mse_loss(v_pred, v_target)


class FamilyClassifier(nn.Module):
    """Simple classifier for Pfam family from ESM-2 latents.

    Used for evaluation: check if generated sequences are classified
    as the requested family.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, n_families=N_FAMILIES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_families),
        )

    def forward(self, z, t=None):
        """
        Args:
            z: [B, L, D] latent states
            t: [B] time (unused, for interface compatibility)

        Returns:
            logits: [B, n_families]
        """
        # Mean pool over sequence
        pooled = z.mean(dim=1)  # [B, D]
        return self.net(pooled)

    def loss(self, z, family, clean_only=False):
        """Classification loss."""
        b = z.shape[0]
        device = z.device

        if clean_only:
            # Only train on clean data (t=1)
            logits = self.forward(z)
        else:
            # Train on noisy data too
            z0 = sample_prior(b, device)
            t = torch.rand(b, device=device)
            tt = t.view(b, 1, 1)
            zt = (1 - tt) * z0 + tt * z
            logits = self.forward(zt, t)

        return nn.functional.cross_entropy(logits, family)
