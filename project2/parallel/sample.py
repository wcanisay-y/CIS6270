"""Masked-diffusion sampling from DPLM with a pluggable unmasking-set rule.

Two sampler families:

  absorbing  a position, once written, is final. At step t the schedule says
             how many positions must be committed; the SELECTOR says which.
             This is the family Idea A is about, and every selector in it
             uses exactly T forward passes of the generator, so comparisons
             are matched-NFE by construction.

  reparam    DPLM's published decoder (Zheng et al. reparameterised decoding,
             "reparam-uncond-deterministic-linear"): every position is
             re-predicted each step and the lowest-confidence
             L*(1 - t/T) are returned to mask, so earlier writes can be
             revised. Kept as the published reference point.
"""
import json
import math
import time

import numpy as np
import torch

from .selectors import choose, shuffle_coupling

# selector aliases that use the greedy decoupling rule but a different
# source for the coupling matrix
_GREEDY_ALIASES = {"contact_oracle": "contact",
                   "contact_oracle_shuffled": "contact_shuffled",
                   "anti_contact_oracle": "anti_contact"}


BANNED = ("mask_token_id", "pad_token_id", "cls_token_id", "eos_token_id",
          "unk_token_id")


def load_dplm(name="airkingbd/dplm_150m", device="cuda", half=True,
              need_attn=False):
    from transformers import AutoTokenizer, EsmForMaskedLM
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"attn_implementation": "eager"} if need_attn else {}
    model = EsmForMaskedLM.from_pretrained(name, **kw).eval().to(device)
    if half:
        model = model.half()
    return tok, model


def banned_ids(tok):
    ids = set()
    for a in BANNED:
        v = getattr(tok, a, None)
        if v is not None:
            ids.add(int(v))
    x = tok.convert_tokens_to_ids("X")
    if x is not None and x >= 0:
        ids.add(int(x))
    for t in (".", "-", "<null_1>", "B", "U", "Z", "O"):
        v = tok.convert_tokens_to_ids(t)
        if v is not None and v >= 0:
            ids.add(int(v))
    return sorted(ids)


def commit_schedule(L, T):
    """How many new positions to commit at each of T steps (sums to L)."""
    cum = [int(round(L * (t + 1) / T)) for t in range(T)]
    cum[-1] = L
    out, prev = [], 0
    for c in cum:
        out.append(max(c - prev, 0))
        prev = c
    return out


@torch.no_grad()
def generate_absorbing(tok, model, L, B, T, selector="confidence",
                       temperature=1.0, tau=0.3, dist_d=8,
                       contact_provider=None, contact_mode="external",
                       contact_every=1, shuffle_contacts=False,
                       punt_thresh=0.1, seed=0, device="cuda",
                       collect_diag=True, fixed_coupling=None,
                       score_mode="gumbel", contact_min_revealed=0.0,
                       tau_mode="absolute", tau_q=0.99):
    """Absorbing-state MDM sampling. Returns (sequences, diagnostics).

    tau_mode controls how the coupling threshold is set.
      absolute  tau is a raw coupling value.
      quantile  tau is the tau_q quantile of THIS sequence's own coupling
                values, so the rule rejects a fixed fraction of pairs no
                matter what scale the coupling source produces.
    The second mode is what makes the comparison meaningful. A supervised
    contact head on a mostly masked sequence emits probabilities around
    0.01, so any fixed threshold near 0.3 never binds and the selector
    silently degenerates into the confidence baseline. A quantile threshold
    also puts the contact head, raw attention and a binary oracle map on the
    same footing, which a raw threshold cannot do.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    ban = banned_ids(tok)
    mask_id = tok.mask_token_id

    ids = torch.full((B, L + 2), mask_id, dtype=torch.long, device=device)
    ids[:, 0] = tok.cls_token_id
    ids[:, -1] = tok.eos_token_id
    sched = commit_schedule(L, T)
    # An oracle run supplies the coupling up front and never refreshes it:
    # it answers "would a correct contact map help?", separately from
    # "can ESM-2 recover the contact map from a mostly-masked sequence?".
    if fixed_coupling is not None:
        fixed_coupling = fixed_coupling.to(device)
        if selector.endswith("_shuffled") or shuffle_contacts:
            fixed_coupling = torch.stack(
                [shuffle_coupling(c) for c in fixed_coupling])
    coupling = fixed_coupling

    # coupled_pair_frac: of the pairs actually committed together, what
    # fraction exceed the coupling threshold. This is the mechanism check --
    # if a selector cannot keep it low at small T, it cannot help there.
    diag = {"coupled_pair_frac": [], "within_coupling": [], "nfe_gen": 0,
            "nfe_contact": 0, "sched": sched}
    t0 = time.time()

    for step, n_t in enumerate(sched):
        need_attn = (selector.startswith(("contact", "anti_contact")) and
                     fixed_coupling is None and
                     contact_mode in ("self", "rawattn") and
                     step % contact_every == 0)
        out = model(input_ids=ids, output_attentions=need_attn)
        diag["nfe_gen"] += 1
        logits = out.logits.float()
        logits[..., ban] = -float("inf")

        # Token proposal and the confidence used for ranking.
        #   gumbel  DPLM's own rule: perturb the logits with Gumbel noise,
        #           take the argmax as the token and the perturbed
        #           log-softmax as the score. DPLM's appendix reports that
        #           unperturbed top-k collapses to near-constant sequences,
        #           so this is the setting a fair baseline must use.
        #   clean   MaskGIT's rule: sample at `temperature`, rank by the
        #           sampled token's clean probability.
        if score_mode == "gumbel":
            gum = -torch.log(-torch.log(
                torch.rand_like(logits) + 1e-8) + 1e-8)
            pert = logits + gum
            conf, samp = pert.log_softmax(-1).max(-1)
        else:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            flat = probs.view(-1, probs.shape[-1])
            samp = torch.multinomial(flat, 1).view(B, L + 2)
            conf = torch.softmax(logits, dim=-1).gather(
                -1, samp.unsqueeze(-1)).squeeze(-1)

        # refresh coupling when the selector needs it
        if (selector.startswith(("contact", "anti_contact"))
                and fixed_coupling is None
                and step % contact_every == 0):
            if contact_mode in ("self", "rawattn"):
                coupling = contact_provider.from_attentions(ids, out.attentions)
            elif contact_mode == "external_argmax":
                # Pranam's ask 4, applied inside the sampler. A frozen ESM-2
                # trained at a 15 % mask rate collapses to its all-mask null
                # on a heavily masked input, but recovers almost completely
                # when the masks are filled with the denoiser's own argmax
                # (precision@L 0.067 -> 0.737 at 75 % revealed, deposited
                # labels, parallel/argmax_contacts.py). This hands the head a
                # complete sequence instead of a masked one. The denoiser
                # pass has already happened, so the only extra cost is the
                # ESM-2 call itself, which is 7.88 denoiser passes.
                ids_c = ids.clone()
                mpos = ids_c == mask_id
                if mpos.any():
                    ids_c[mpos] = logits.argmax(-1)[mpos]
                coupling = contact_provider(ids_c)
                diag["nfe_contact"] += 1
            else:
                coupling = contact_provider(ids)
                diag["nfe_contact"] += 1
            if selector.endswith("_shuffled") or shuffle_contacts:
                coupling = torch.stack([shuffle_coupling(c) for c in coupling])

        # per-sequence threshold
        taus = None
        if coupling is not None and tau_mode == "quantile":
            iu = torch.triu_indices(L, L, offset=1, device=coupling.device)
            flat = coupling[:, iu[0], iu[1]]
            taus = torch.quantile(flat.float(), tau_q, dim=1)

        punt_cp = None
        if selector == "punt":
            punt_cp, extra = _punt_coupling(
                tok, model, ids, logits, samp, conf, n_t, ban, punt_thresh)
            diag["nfe_gen"] += extra

        # The contact signal only exists once enough of the sequence is
        # written: measured on natural 200-mers, the frozen ESM-2 650M head
        # is at its all-mask null by 25 % masking and the generator's own
        # attentions hold up to about 50 %. Below `contact_min_revealed`
        # we therefore fall back to plain confidence rather than select on
        # a coupling matrix that carries no information.
        revealed = 1.0 - float((ids[:, 1:L + 1] == mask_id).float().mean())
        eff_sel = selector
        if (contact_min_revealed > 0.0 and revealed < contact_min_revealed
                and selector.startswith(("contact", "anti_contact"))):
            eff_sel = "confidence"
        diag.setdefault("gated_steps", 0)
        diag["gated_steps"] += int(eff_sel != selector)

        for b in range(B):
            cand = (ids[b, 1:L + 1] == mask_id).nonzero(as_tuple=True)[0]
            if cand.numel() == 0:
                continue
            cp = None
            if coupling is not None:
                cp = coupling[b]
            if selector == "punt":
                cp = punt_cp[b]
            tau_b = float(taus[b]) if taus is not None else tau
            sel = choose(_GREEDY_ALIASES.get(eff_sel, eff_sel),
                         cand=cand, logits_pos=logits[b, 1:L + 1],
                         n=n_t, conf=conf[b, 1:L + 1], coupling=cp,
                         tau=tau_b, d=dist_d, L=L,
                         punt_state=cp if eff_sel == "punt" else None)
            if (collect_diag and cp is not None and sel.numel() > 1
                    and eff_sel == selector):
                sub = cp[sel][:, sel]
                iu = torch.triu_indices(sel.numel(), sel.numel(), offset=1)
                vals = sub[iu[0], iu[1]]
                diag["within_coupling"].append(float(vals.mean()))
                diag["coupled_pair_frac"].append(
                    float((vals >= tau_b).float().mean()))
                diag.setdefault("tau_used", []).append(float(tau_b))
            ids[b, sel + 1] = samp[b, sel + 1]

    diag["seconds"] = time.time() - t0
    seqs = [tok.decode(ids[b, 1:L + 1]).replace(" ", "") for b in range(B)]
    return seqs, diag


@torch.no_grad()
def _punt_coupling(tok, model, ids, logits, samp, conf, n_t, ban, thresh):
    """PUNT-style conditional-independence coupling (reimplementation).

    One extra forward pass: speculatively write the top-n_t confidence set,
    re-run the denoiser, and measure how far each member's marginal moved.
    A pair is treated as coupled when both members move, which is the
    observable signature of a violated conditional-independence assumption.
    This follows the idea of testing independence with the denoiser itself
    rather than an external signal; it is our reimplementation, not the
    authors' code.
    """
    B, Lp2 = ids.shape
    L = Lp2 - 2
    mask_id = tok.mask_token_id
    p_before = torch.softmax(logits[:, 1:L + 1], -1)
    spec = ids.clone()
    chosen = []
    for b in range(B):
        cand = (ids[b, 1:L + 1] == mask_id).nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            chosen.append(cand)
            continue
        k = min(max(n_t, 1), cand.numel())
        sel = cand[torch.argsort(conf[b, cand + 1], descending=True)[:k]]
        spec[b, sel + 1] = samp[b, sel + 1]
        chosen.append(sel)
    out2 = model(input_ids=spec)
    lg2 = out2.logits.float()
    lg2[..., ban] = -float("inf")
    p_after = torch.softmax(lg2[:, 1:L + 1], -1)
    tv = 0.5 * (p_after - p_before).abs().sum(-1)        # (B, L)
    move = (tv / max(thresh, 1e-8)).clamp(0, 1)          # normalised shift
    cp = move.unsqueeze(2) * move.unsqueeze(1)           # (B, L, L)
    for b in range(B):
        cp[b].fill_diagonal_(0.0)
    return cp, 1


@torch.no_grad()
def _resample_repeats(model, ids, tokens, scores, ban, mask_id, L,
                      ratio=0.25):
    """DPLM's anti-collapse step, reimplemented from dplm.py `resample`.

    Any token whose frequency in a sequence exceeds L*ratio is masked and
    resampled conditioned on the rest. Without it DPLM's decoder collapses
    to near-constant sequences such as poly-arginine, which still score a
    high pLDDT -- so leaving it out would hand us an unfairly weak baseline.
    It costs one extra forward pass on the affected rows, which we count.
    """
    rows, masks = [], []
    for i in range(tokens.shape[0]):
        seq = tokens[i, 1:L + 1]
        vals, cnts = torch.unique(seq, return_counts=True)
        hot = vals[cnts > L * ratio]
        if hot.numel() == 0:
            continue
        m = torch.zeros_like(tokens[i], dtype=torch.bool)
        m[1:L + 1] = torch.isin(seq, hot)
        rows.append(i)
        masks.append(m)
    if not rows:
        return tokens, scores, 0
    inp = tokens.clone()
    for i, m in zip(rows, masks):
        inp[i] = inp[i].masked_fill(m, mask_id)
    out = model(input_ids=inp[rows])
    lg = out.logits.float()
    lg[..., ban] = -float("inf")
    gum = -torch.log(-torch.log(torch.rand_like(lg) + 1e-8) + 1e-8)
    sc, tk = (lg + gum).log_softmax(-1).max(-1)
    for k, (i, m) in enumerate(zip(rows, masks)):
        tokens[i] = torch.where(m, tk[k], tokens[i])
        scores[i] = torch.where(m, sc[k], scores[i])
    return tokens, scores, 1


@torch.no_grad()
def generate_reparam(tok, model, L, B, T, temperature=1.0, seed=0,
                     device="cuda", resample=True, resample_ratio=0.25,
                     selector="confidence", contact_provider=None,
                     contact_mode="self", contact_every=1, tau=0.3,
                     tau_mode="quantile", tau_q=0.99, warmup=0.0):
    """DPLM's published decoder, reimplemented to match dplm.py.

    `selector` extends it with the Idea-A rule. DPLM keeps the highest
    scoring L*(t/T) positions each step and returns the rest to mask, so the
    set that survives a step IS the set written in parallel at that step.
    Applying the decoupling rule to the surviving set is therefore the
    faithful transfer of the proposal into the decoder DPLM actually ships,
    which in our harness scores 27 pLDDT above any absorbing sampler. The
    number of survivors is unchanged, so the schedule and the forward-pass
    count are untouched.
    """
    torch.manual_seed(seed)
    ban = banned_ids(tok)
    mask_id = tok.mask_token_id
    ids = torch.full((B, L + 2), mask_id, dtype=torch.long, device=device)
    ids[:, 0] = tok.cls_token_id
    ids[:, -1] = tok.eos_token_id
    scores = torch.zeros(B, L + 2, device=device)
    nonspecial = torch.zeros(B, L + 2, dtype=torch.bool, device=device)
    nonspecial[:, 1:L + 1] = True
    xt_neq_x0 = nonspecial.clone()
    nfe_extra = 0
    nfe_contact = 0
    coupling = None
    diag_pairs = []
    t0 = time.time()
    gated = 0
    for step in range(T):
        # Warm-up. In this decoder the kept fraction is (step+1)/T by
        # construction, so that is the revealed fraction the contact signal
        # has to work with. Below `warmup` the step falls back to plain
        # confidence and no contact call is made at all, which is what makes
        # the warm-up a saving rather than only a safeguard.
        active = (selector != "confidence"
                  and (step + 1) / T >= warmup)
        gated += int(selector != "confidence" and not active)
        need_attn = (active and
                     contact_mode in ("self", "rawattn") and
                     step % contact_every == 0)
        out = model(input_ids=ids, output_attentions=need_attn)
        logits = out.logits.float()
        logits[..., ban] = -float("inf")
        # gumbel_argmax: perturb logits, then take argmax log-softmax
        gum = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        pert = logits + gum
        cur_scores, cur_tokens = pert.log_softmax(-1).max(-1)
        if resample:
            cur_tokens, cur_scores, extra = _resample_repeats(
                model, ids, cur_tokens, cur_scores, ban, mask_id, L,
                resample_ratio)
            nfe_extra += extra

        new_ids = ids.clone()
        new_scores = scores.clone()
        upd = nonspecial
        new_ids[upd] = cur_tokens[upd]
        new_scores[upd] = cur_scores[upd]

        if active and step % contact_every == 0:
            if contact_mode in ("self", "rawattn"):
                coupling = contact_provider.from_attentions(ids,
                                                            out.attentions)
            elif contact_mode == "external_argmax":
                # Same repair as in the absorbing decoder: hand the frozen
                # head a complete sequence rather than a masked one. The
                # remasking decoder keeps every position written and masks
                # the lowest-scoring ones back, so there are usually fewer
                # masks here than in the absorbing case, but the call is
                # identical.
                ids_c = ids.clone()
                mpos = ids_c == mask_id
                if mpos.any():
                    ids_c[mpos] = logits.argmax(-1)[mpos]
                coupling = contact_provider(ids_c)
                nfe_contact += 1
            else:
                coupling = contact_provider(ids)
                # a separate ESM-2 pass is NOT a denoiser pass; counting it
                # in nfe_extra reported it as generator NFE and hid the
                # overhead the method has to pay for
                nfe_contact += 1
            if selector.endswith("_shuffled"):
                coupling = torch.stack([shuffle_coupling(c)
                                        for c in coupling])

        rate = 1.0 - (step + 1) / T
        cutoff = (nonspecial.sum(1, keepdim=True).float() * rate).long()
        s = new_scores.masked_fill(~nonspecial, 1000.0)
        srt = s.sort(-1).values
        thr = srt.gather(-1, cutoff.clamp(max=s.shape[1] - 1))
        lowest_k = s < thr

        if active and coupling is not None:
            # Re-choose WHICH positions survive, keeping how many unchanged.
            keep_n = int(nonspecial[0].sum().item()) - int(cutoff[0].item())
            if keep_n > 1:
                base = _GREEDY_ALIASES.get(selector, selector)
                for b in range(B):
                    cand = nonspecial[b].nonzero(as_tuple=True)[0] - 1
                    cp = coupling[b]
                    if tau_mode == "quantile":
                        iu2 = torch.triu_indices(L, L, offset=1,
                                                 device=cp.device)
                        tb = float(torch.quantile(
                            cp[iu2[0], iu2[1]].float(), tau_q))
                    else:
                        tb = tau
                    sel = choose(base, cand=cand,
                                 logits_pos=logits[b, 1:L + 1], n=keep_n,
                                 conf=new_scores[b, 1:L + 1], coupling=cp,
                                 tau=tb, L=L, punt_state=cp)
                    km = torch.ones(L + 2, dtype=torch.bool, device=device)
                    km[sel + 1] = False
                    km = km & nonspecial[b]
                    lowest_k[b] = km
                    if sel.numel() > 1:
                        sub = cp[sel][:, sel]
                        iu3 = torch.triu_indices(sel.numel(), sel.numel(),
                                                 offset=1)
                        diag_pairs.append(float(
                            (sub[iu3[0], iu3[1]] >= tb).float().mean()))

        masked_to_noise = ((~xt_neq_x0) & lowest_k) | (xt_neq_x0 & lowest_k)
        new_ids[masked_to_noise] = mask_id
        new_scores[masked_to_noise] = -math.inf
        xt_neq_x0 = lowest_k
        ids, scores = new_ids, new_scores
    seqs = [tok.decode(ids[b, 1:L + 1]).replace(" ", "") for b in range(B)]
    return seqs, {"nfe_gen": T + nfe_extra, "nfe_contact": nfe_contact,
                  "nfe_resample": nfe_extra,
                  "coupled_pair_frac": diag_pairs,
                  "within_coupling": [],
                  "gated_steps": gated,
                  "warmup": warmup,
                  "seconds": time.time() - t0}
