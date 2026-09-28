"""Guidance for protein flow matching with feasibility gating.

BASELINE GUIDANCE (CFG):
    v_CFG = v_theta(z, t, null) + w * [v_theta(z, t, c) - v_theta(z, t, null)]

THE INNOVATION (feasibility-gated guidance):
The mechanism transfers from molecules: instead of per-atom valency residuals,
we use per-residue decoding confidence from ESM's frozen lm_head.

    z_hat_1 = z_t + (1-t) * v_theta     (endpoint estimate)
    logits = lm_head(z_hat_1)           (decode to amino acid logits)
    max_prob_i = max(softmax(logits_i)) (confidence per residue)
    r_i = 1 - max_prob_i                (residual: low confidence = high residual)
    g_i = exp(-beta * r_i^2)            (gate in (0, 1])
    v_i = v_null_i + w * g_i * (v_cond_i - v_null_i)

Confident residue = resolved. The gate applies per-residue exactly as
valency gating does per-atom in the molecules code.

GATE MODES (ablation):
    none       - no gating, standard CFG (g=1 everywhere)
    global     - average gate over sequence, same for all residues
    shuffled   - permute gates randomly (mean-matched control)
    per_residue - full position-specific gating (the innovation)
"""
import torch

GATE_MODES = ("none", "global", "shuffled", "per_residue")


class ESMDecoder:
    """Wrapper for ESM-2's lm_head to decode latents to amino acid probs.

    Loads the frozen lm_head from ESM-2 and applies it to latent states.
    The lm_head projects from hidden_dim (320) to vocab_size (33 for ESM-2).
    """

    def __init__(self, device="cuda"):
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForMaskedLM

        self._tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model = model.to(self.device).eval()

        # Extract lm_head (the decoding layer)
        self._lm_head = model.lm_head
        for p in self._lm_head.parameters():
            p.requires_grad_(False)

        self._model = model

    def decode_logits(self, z, z_mean, z_std):
        """Decode standardized latents to amino acid logits.

        Args:
            z: [B, L, D] standardized latent states
            z_mean: float, mean used for standardization
            z_std: float, std used for standardization

        Returns:
            logits: [B, L, vocab_size] unnormalized log-probs
        """
        self._load()

        # Unstandardize to original ESM scale
        z_orig = z * z_std + z_mean

        # Apply lm_head
        with torch.no_grad():
            logits = self._lm_head(z_orig)

        return logits

    def decode_to_sequence(self, z, z_mean, z_std):
        """Decode latents to amino acid sequences via argmax.

        Args:
            z: [B, L, D] standardized latent states
            z_mean: float
            z_std: float

        Returns:
            sequences: list of str, length B
        """
        self._load()

        logits = self.decode_logits(z, z_mean, z_std)
        token_ids = logits.argmax(dim=-1)  # [B, L]

        sequences = []
        for i in range(token_ids.shape[0]):
            # Decode tokens to string
            seq = self._tokenizer.decode(token_ids[i], skip_special_tokens=True)
            # Remove spaces that tokenizer might add
            seq = seq.replace(" ", "")
            sequences.append(seq)

        return sequences


# Global decoder instance (lazy loaded)
_decoder = None


def get_decoder(device="cuda"):
    global _decoder
    if _decoder is None or _decoder.device != device:
        _decoder = ESMDecoder(device)
    return _decoder


def endpoint(z, v, t):
    """z_hat_1 = z_t + (1-t) * v; the flow model's guess at the finished latent."""
    f = (1.0 - t).view(-1, 1, 1)
    return z + f * v


def feasibility_residual(z1, z_mean, z_std, device="cuda"):
    """Per-residue feasibility residual: r_i = 1 - max_prob_i.

    Higher residual means lower confidence (less "resolved").

    Args:
        z1: [B, L, D] endpoint estimate (standardized)
        z_mean, z_std: standardization stats

    Returns:
        r: [B, L] residuals in [0, 1]
    """
    decoder = get_decoder(device)
    logits = decoder.decode_logits(z1, z_mean, z_std)  # [B, L, V]
    probs = torch.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values  # [B, L]
    return 1.0 - max_probs


def feasibility_gate(z1, z_mean, z_std, mode="per_residue", beta=1.0, device="cuda"):
    """Gate g in (0, 1], shape [B, L, 1].

    No gradients are taken through the gate: it sets how hard to push,
    it is not an objective being ascended.

    Args:
        z1: [B, L, D] endpoint estimate (standardized)
        mode: one of GATE_MODES
        beta: gate sharpness

    Returns:
        g: [B, L, 1] gate values
    """
    b, l, d = z1.shape

    if mode == "none":
        return torch.ones(b, l, 1, device=z1.device)

    with torch.no_grad():
        r = feasibility_residual(z1, z_mean, z_std, device)  # [B, L]
        g = torch.exp(-beta * r ** 2)

        if mode == "global":
            # One gate per sequence (mean over residues)
            mean = g.mean(dim=-1, keepdim=True)
            g = mean.expand_as(g)

        elif mode == "shuffled":
            # Permute gates randomly (mean-matched control)
            key = torch.rand_like(g)
            g = torch.gather(g, 1, key.argsort(dim=1))

        elif mode != "per_residue":
            raise ValueError(f"unknown gate mode: {mode}")

    return g[..., None]  # [B, L, 1]


def cfg(model, z, t, family, w, gate_mode="none", beta=1.0, z_mean=0.0, z_std=1.0):
    """Classifier-free guided velocity, optionally feasibility-gated.

    Args:
        model: FlowModel
        z: [B, L, D] current latent state
        t: [B] current time
        family: [B] target family indices
        w: guidance strength
        gate_mode: one of GATE_MODES
        beta: gate sharpness
        z_mean, z_std: standardization stats for decoding

    Returns:
        v: [B, L, D] guided velocity
        g: [B, L, 1] gate values (for logging)
    """
    b = z.shape[0]
    device = z.device

    # Unconditional velocity (cond_on = 0)
    cond_off = torch.zeros(b, device=device)
    v_null = model.field(z, t, family, cond_off)

    if w == 0:
        return v_null, torch.ones(b, z.shape[1], 1, device=device)

    # Conditional velocity (cond_on = 1)
    cond_on = torch.ones(b, device=device)
    v_cond = model.field(z, t, family, cond_on)

    # Compute gate
    if gate_mode == "none":
        g = torch.ones(b, z.shape[1], 1, device=device)
    else:
        # Endpoint estimate using conditional velocity
        z1_hat = endpoint(z, v_cond, t)
        g = feasibility_gate(z1_hat, z_mean, z_std, gate_mode, beta, device)

    # Apply gated CFG
    v = v_null + w * g * (v_cond - v_null)

    return v, g
