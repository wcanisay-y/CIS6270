"""The revised method: adaptive, cached, warm-started contact gating.

Implements "Revised Method and Research Hypothesis" as written:

  * frozen denoiser, frozen ESM-2; no ground truth, no reference structure,
    no online validation signal during generation
  * generation starts with confidence-only unmasking, and the contact map is
    initialised by the ESM-2 contact head only after a warm-up
  * masked positions are ranked by denoiser confidence; a group is grown by
    rejecting a position whose contact score against any already-accepted
    position exceeds a threshold
  * an explicit group-size cap as a second control on parallelism
  * the map is CACHED, and refreshed when any of three triggers fires:
      distribution  mean total-variation change in the denoiser's
                    per-position amino-acid distributions since the last
                    refresh, over positions still masked
      reveal        fraction of the sequence newly revealed since the last
                    refresh
      age           steps since the last refresh
    a refresh REPLACES the cached map, as the document specifies
  * every threshold is a fixed argument, set on validation sequences and
    frozen before the test run

What the method claims, and therefore what is measured here: generation
TIME at comparable quality, after charging for contact-prediction overhead.
That is a different axis from the original proposal, which claimed better
proteins, so the accounting is the result. `nfe_gen` and `nfe_contact` are
counted separately and combined through a MEASURED cost ratio rather than
an assumed one, because an ESM-2 650M pass is several times a DPLM-150M
pass and a method that refreshes often can be slower while making fewer
denoiser calls.
"""
import time

import torch

from .sample import banned_ids, commit_schedule
from .selectors import choose, shuffle_coupling

TRIGGERS = ("distribution", "reveal", "age", "init")


def _aa_dist(logits, keep):
    """Per-position distribution over the kept (amino-acid) vocabulary."""
    return torch.softmax(logits[..., keep].float(), dim=-1)


@torch.no_grad()
def generate_adaptive(
        tok, model, L, B, T,
        selector="contact",
        contact_provider=None,
        contact_mode="external",
        aa_ids=None,
        warmup=0.5,
        refresh_dist=0.15,
        refresh_reveal=0.10,
        refresh_max_age=8,
        g_max=None,
        tau_q=0.99,
        shuffle_contacts=False,
        fixed_coupling=None,
        seed=0,
        device="cuda",
        max_steps_factor=4.0):
    """Adaptive contact-gated absorbing sampler. -> (seqs, diag).

    selector      "confidence" (baseline, no contacts at all), "contact",
                  or "contact_shuffled" (control: same groups, no structure)
    contact_mode  "external" a separate frozen ESM-2 pass, charged for
                  "self"     the generator's own attentions through the
                             ESM-2 contact head, free
                  "rawattn"  symmetrised raw attention, free
    warmup        fraction of the sequence that must be revealed before
                  contacts are consulted at all; 0.0 reproduces the
                  original proposal
    g_max         maximum positions committed per step, or None
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    ban = banned_ids(tok)
    mask_id = tok.mask_token_id
    keep = aa_ids if aa_ids is not None else None

    ids = torch.full((B, L + 2), mask_id, dtype=torch.long, device=device)
    ids[:, 0] = tok.cls_token_id
    ids[:, -1] = tok.eos_token_id

    sched = commit_schedule(L, T)
    uses_contacts = selector.startswith("contact")
    free_signal = contact_mode in ("self", "rawattn")

    coupling = None
    if fixed_coupling is not None:
        coupling = fixed_coupling.to(device)
        if shuffle_contacts or selector.endswith("_shuffled"):
            coupling = torch.stack([shuffle_coupling(c) for c in coupling])

    diag = {"nfe_gen": 0, "nfe_contact": 0, "steps": 0,
            "refreshes": 0, "trigger_counts": {k: 0 for k in TRIGGERS},
            "gated_steps": 0, "within_coupling": [], "coupled_pair_frac": [],
            "refresh_at_revealed": [], "dist_change_trace": [],
            "group_sizes": [], "warmup": warmup, "g_max": g_max,
            "selector": selector, "contact_mode": contact_mode}

    ref_dist = None          # AA distributions at the last refresh
    ref_revealed = 0.0       # revealed fraction at the last refresh
    age = 0
    remaining = L
    step = 0
    hard_cap = int(max_steps_factor * T) + L  # guards against a stuck loop
    t0 = time.time()

    while remaining > 0 and step < hard_cap:
        n_t = sched[step] if step < len(sched) else sched[-1]
        n_t = max(1, n_t)
        if g_max is not None:
            n_t = min(n_t, int(g_max))
        n_t = min(n_t, remaining)

        need_attn = uses_contacts and free_signal and fixed_coupling is None
        out = model(input_ids=ids, output_attentions=need_attn)
        diag["nfe_gen"] += 1
        logits = out.logits.float()
        logits[..., ban] = -float("inf")

        # DPLM's own proposal rule: Gumbel-perturbed logits, score = the
        # perturbed log-softmax. Unperturbed top-k collapses.
        gum = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        conf, samp = (logits + gum).log_softmax(-1).max(-1)

        revealed = 1.0 - float((ids[:, 1:L + 1] == mask_id).float().mean())
        active = uses_contacts and revealed >= warmup

        # ---------------------------------------------- refresh decision
        if active and fixed_coupling is None:
            cur = _aa_dist(logits[:, 1:L + 1], keep) if keep is not None \
                else torch.softmax(logits[:, 1:L + 1], dim=-1)
            masked_now = (ids[:, 1:L + 1] == mask_id)
            trigger = None
            if coupling is None or ref_dist is None:
                trigger = "init"
            else:
                # mean total variation over positions still masked: the
                # change in what the denoiser believes, which is the signal
                # the method uses as a proxy for the map going stale
                tv = 0.5 * (cur - ref_dist).abs().sum(-1)
                mm = masked_now.float()
                denom = mm.sum().clamp_min(1.0)
                dchange = float((tv * mm).sum() / denom)
                diag["dist_change_trace"].append(dchange)
                if dchange > refresh_dist:
                    trigger = "distribution"
                elif revealed - ref_revealed > refresh_reveal:
                    trigger = "reveal"
                elif age >= refresh_max_age:
                    trigger = "age"
            if trigger is not None:
                if free_signal:
                    coupling = contact_provider.from_attentions(
                        ids, out.attentions)
                else:
                    coupling = contact_provider(ids)
                    diag["nfe_contact"] += 1
                if shuffle_contacts or selector.endswith("_shuffled"):
                    coupling = torch.stack(
                        [shuffle_coupling(c) for c in coupling])
                diag["refreshes"] += 1
                diag["trigger_counts"][trigger] += 1
                diag["refresh_at_revealed"].append(revealed)
                ref_dist = cur
                ref_revealed = revealed
                age = 0
            else:
                age += 1
        elif active and fixed_coupling is not None:
            pass  # oracle: map supplied once, never refreshed
        else:
            age += 1

        eff_sel = selector if (active and coupling is not None) \
            else "confidence"
        diag["gated_steps"] += int(eff_sel != selector)

        taus = None
        if coupling is not None and eff_sel != "confidence":
            iu = torch.triu_indices(L, L, offset=1, device=coupling.device)
            flat = coupling[:, iu[0], iu[1]]
            taus = torch.quantile(flat.float(), tau_q, dim=1)

        for b in range(B):
            cand = (ids[b, 1:L + 1] == mask_id).nonzero(as_tuple=True)[0]
            if cand.numel() == 0:
                continue
            cp = coupling[b] if (coupling is not None
                                 and eff_sel != "confidence") else None
            tau_b = float(taus[b]) if taus is not None else 1e9
            sel = choose("contact" if eff_sel.startswith("contact")
                         else eff_sel,
                         cand=cand, logits_pos=logits[b, 1:L + 1],
                         n=n_t, conf=conf[b, 1:L + 1], coupling=cp,
                         tau=tau_b, d=8, L=L, punt_state=None)
            if cp is not None and sel.numel() > 1:
                sub = cp[sel][:, sel]
                iu2 = torch.triu_indices(sel.numel(), sel.numel(), offset=1)
                vals = sub[iu2[0], iu2[1]]
                diag["within_coupling"].append(float(vals.mean()))
                diag["coupled_pair_frac"].append(
                    float((vals >= tau_b).float().mean()))
            ids[b, sel + 1] = samp[b, sel + 1]
            diag["group_sizes"].append(int(sel.numel()))

        remaining = int((ids[:, 1:L + 1] == mask_id).any(0).sum())
        step += 1

    diag["steps"] = step
    diag["seconds"] = time.time() - t0
    diag["unfinished"] = int((ids[:, 1:L + 1] == mask_id).sum())
    seqs = [tok.decode(ids[b, 1:L + 1]).replace(" ", "") for b in range(B)]
    return seqs, diag


@torch.no_grad()
def measure_cost_ratio(tok, model, contact_provider, L=200, B=4,
                       device="cuda", reps=6):
    """Measure what each kind of call actually costs on this GPU.

    The method's whole claim is net time, so none of this may be assumed.
    Three quantities, because three arms pay differently:

      gen_s       a denoiser pass, attentions not requested
      gen_attn_s  a denoiser pass WITH output_attentions=True, which the
                  free-signal arms need and which is not free
      con_s       a separate frozen ESM-2 650M contact pass

    Returns (gen_s, gen_attn_s, con_s, ratio, attn_factor) where
    ratio = con_s / gen_s and attn_factor = gen_attn_s / gen_s.
    """
    ids = torch.full((B, L + 2), tok.mask_token_id, dtype=torch.long,
                     device=device)
    ids[:, 0] = tok.cls_token_id
    ids[:, -1] = tok.eos_token_id
    sync = (torch.cuda.synchronize if device.startswith("cuda")
            else (lambda: None))
    for _ in range(2):
        model(input_ids=ids)
        model(input_ids=ids, output_attentions=True)
        contact_provider(ids)
    sync()
    t = time.time()
    for _ in range(reps):
        model(input_ids=ids)
    sync()
    gen_s = (time.time() - t) / reps
    t = time.time()
    for _ in range(reps):
        model(input_ids=ids, output_attentions=True)
    sync()
    gen_attn_s = (time.time() - t) / reps
    t = time.time()
    for _ in range(reps):
        contact_provider(ids)
    sync()
    con_s = (time.time() - t) / reps
    return (gen_s, gen_attn_s, con_s, con_s / max(gen_s, 1e-9),
            gen_attn_s / max(gen_s, 1e-9))
