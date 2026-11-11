"""Pairwise coupling from ESM-2 contact prediction on a partially masked sequence.

Two providers, differing in cost:

  external  a separate frozen ESM-2 forward pass (what the proposal asks for).
            Honest but expensive: ESM-2 650M costs ~4x a DPLM-150M step, so
            refreshing every step would make the "fewer model calls" claim
            vacuous. Refresh frequency is therefore a first-class parameter
            and we report generator NFE and total FLOPs separately.

  self      DPLM's own attention maps fed through the contact-head regression
            of the ESM-2 checkpoint DPLM was initialised from. Costs nothing
            extra, because the forward pass has already happened. Whether the
            regression transfers across DPLM's fine-tuning is an empirical
            question that validate_transfer() answers.

The contact head expects tokens laid out exactly as [cls] + residues + [eos]
with no padding, and returns (B, L, L) over the residues only.
"""
import torch

ESM2_FOR_HEAD = {150: "facebook/esm2_t30_150M_UR50D",
                 650: "facebook/esm2_t33_650M_UR50D",
                 8:   "facebook/esm2_t6_8M_UR50D"}


class ExternalContacts:
    """Frozen ESM-2, full forward pass, attention-based contact head.

    token_dropout MUST be considered here. ESM-2's embedding layer zeroes
    mask embeddings and then rescales every embedding by
    (1 - 0.12) / (1 - observed_mask_ratio). On a sequence that is almost
    entirely masked -- which is exactly the state the selector runs in at
    the first steps -- that factor is enormous (about 89x for a 200-residue
    all-mask input) and saturates the attention maps the contact head reads.
    The default here disables it, which is the sensible choice for this use;
    `token_dropout=True` reproduces the stock behaviour for the ablation.
    """

    def __init__(self, size=650, device="cuda", half=False,
                 token_dropout=False):
        from transformers import EsmForMaskedLM
        name = ESM2_FOR_HEAD[size]
        # attn_implementation="eager" is REQUIRED: from transformers 4.56 the
        # SDPA path returns None for attentions, and predict_contacts then
        # fails trying to stack them.
        self.model = EsmForMaskedLM.from_pretrained(
            name, attn_implementation="eager").eval().to(device)
        if not token_dropout:
            self.model.config.token_dropout = False
            self.model.esm.embeddings.token_dropout = False
        if half:
            # The contact head mixes fp32 attentions with its own linear
            # layer, so it must stay fp32 even when the trunk is halved.
            self.model = self.model.half()
            self.model.esm.contact_head.float()
        self.device = device
        self.size = size
        self.token_dropout = token_dropout
        self.calls = 0

    @torch.no_grad()
    def __call__(self, tokens):
        """tokens (B, L+2) with [cls]/[eos]; -> (B, L, L) float32 in [0,1].

        The head strips cls/eos itself, so the result is already indexed by
        residue position. cls/eos must never be masked: that drives the
        observed mask ratio to 1 and the token_dropout rescale divides by
        zero, returning an all-NaN map.
        """
        am = torch.ones_like(tokens)
        self.calls += 1
        return self.model.predict_contacts(tokens, am).float()


class SelfContacts:
    """DPLM's own attentions + the ESM-2 contact-head regression weights."""

    def __init__(self, size=150, device="cuda"):
        from transformers import EsmForMaskedLM
        from transformers.models.esm.modeling_esm import EsmContactPredictionHead
        name = ESM2_FOR_HEAD[size]
        src = EsmForMaskedLM.from_pretrained(name)
        cfg = src.config
        self.head = EsmContactPredictionHead(
            in_features=cfg.num_hidden_layers * cfg.num_attention_heads,
            bias=True).eval().to(device)
        self.head.load_state_dict(src.esm.contact_head.state_dict())
        del src
        self.device = device
        self.calls = 0

    @torch.no_grad()
    def from_attentions(self, tokens, attentions):
        """attentions: tuple of (B, H, L+2, L+2) per layer -> (B, L, L)."""
        if attentions is None:
            raise RuntimeError(
                "model returned no attentions: load it with "
                "attn_implementation='eager' (the SDPA path returns None "
                "from transformers 4.56)")
        am = torch.ones_like(tokens).to(attentions[0].dtype)
        a = torch.stack(attentions, dim=1).float()
        a = a * am.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        a = a * am.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        self.calls += 1
        return self.head(tokens, a).float()


@torch.no_grad()
def validate_transfer(seqs, device="cuda"):
    """Do DPLM attentions + ESM-2 head reproduce ESM-2's own contacts?

    Reports Spearman correlation of the two contact maps and precision@L of
    each against the other, on natural sequences. Also reports how both decay
    as the input is progressively masked, which is the regime the selector
    actually runs in.
    """
    import numpy as np
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer, EsmForMaskedLM

    tok = AutoTokenizer.from_pretrained("airkingbd/dplm_150m")
    dplm = EsmForMaskedLM.from_pretrained(
        "airkingbd/dplm_150m", attn_implementation="eager").eval().to(device)
    ext = ExternalContacts(650, device=device, half=False)
    slf = SelfContacts(150, device=device)

    rows = []
    for frac in (0.0, 0.25, 0.5, 0.75, 0.9):
        for s in seqs:
            ids = tok(s, return_tensors="pt")["input_ids"].to(device)
            L = len(s)
            g = torch.Generator(device="cpu").manual_seed(0)
            nm = int(round(frac * L))
            idx = torch.randperm(L, generator=g)[:nm] + 1
            m = ids.clone()
            m[0, idx] = tok.mask_token_id
            c_ext = ext(m)[0].cpu().numpy()
            out = dplm(input_ids=m, output_attentions=True)
            c_slf = slf.from_attentions(m, out.attentions)[0].cpu().numpy()
            iu = np.triu_indices(L, k=6)
            r = spearmanr(c_ext[iu], c_slf[iu]).correlation
            k = L
            te = set(map(tuple, np.array(iu).T[np.argsort(-c_ext[iu])[:k]]))
            ts = set(map(tuple, np.array(iu).T[np.argsort(-c_slf[iu])[:k]]))
            rows.append(dict(mask_frac=frac, L=L, spearman=float(r),
                             topL_overlap=len(te & ts) / max(k, 1),
                             ext_mean=float(c_ext[iu].mean()),
                             self_mean=float(c_slf[iu].mean())))
    return rows


class RawAttentionContacts:
    """Symmetrised raw attention from the generator itself, no contact head.

    This is the signal the recent text-domain methods use (attention averaged
    over the last fraction of layers, symmetrised, thresholded). It costs
    nothing beyond the forward pass the sampler already makes, so it is the
    right free-signal baseline to beat: if the supervised contact head does
    not beat raw attention, paying for a second network is unjustified.
    """

    def __init__(self, last_frac=0.3, device="cuda"):
        self.last_frac = last_frac
        self.device = device
        self.calls = 0

    @torch.no_grad()
    def from_attentions(self, tokens, attentions):
        if attentions is None:
            raise RuntimeError(
                "model returned no attentions: load it with "
                "attn_implementation='eager' (the SDPA path returns None "
                "from transformers 4.56)")
        n = len(attentions)
        k = max(1, int(round(self.last_frac * n)))
        a = torch.stack(attentions[-k:], dim=1).float()      # (B,k,H,L+2,L+2)
        a = a.mean(dim=(1, 2))                               # (B,L+2,L+2)
        a = 0.5 * (a + a.transpose(-1, -2))
        a = a[:, 1:-1, 1:-1]                                 # drop cls/eos
        # rescale to [0,1] per sequence so one threshold means the same thing
        B = a.shape[0]
        flat = a.reshape(B, -1)
        lo = flat.min(1).values[:, None, None]
        hi = flat.max(1).values[:, None, None].clamp_min(lo + 1e-8)
        self.calls += 1
        return ((a - lo) / (hi - lo)).float()
