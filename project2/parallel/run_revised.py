"""Test the revised method: warm-up + cached, trigger-refreshed contact gating.

The revised method claims generation TIME at comparable quality, after
charging for contact prediction. So every cell reports three costs:

    nfe_gen       denoiser forward passes
    nfe_contact   external ESM-2 650M passes
    cost_eq       nfe_gen + ratio * nfe_contact, ratio MEASURED on this GPU

and the comparison that decides the method is quality against cost_eq, not
against nfe_gen. Four questions, one arm family each:

  warm-up     does deferring contacts until the sequence is half written
              rescue a rule that fails when applied from the first step?
              (`contact_min_revealed` existed in the old code but was never
              once set in any run, so this was never tested.)
  refresh     does trigger-based refresh beat a fixed interval at equal cost?
  free signal does the generator's own attention, which needs no cache and
              no second network, match the external head? If it does, the
              caching contribution has nothing to buy.
  control     contact_shuffled at the same warm-up: same group structure,
              no structural information.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.alphabet import AA20
from parallel.sample import load_dplm
from parallel.adaptive import generate_adaptive, measure_cost_ratio
from parallel.contacts import ExternalContacts, SelfContacts, \
    RawAttentionContacts


def build_confirm_arms(T, mode, warmup):
    """Exactly three arms for a high-power confirmation of one cell.

    The sweeps run 48 sequences per cell, where the smallest effect
    detectable at 80 percent power is about 7 pLDDT. The largest published
    gain for unmasking-order selection on this model is 1.65, so a sweep
    cell cannot resolve the effect it is looking for. This builds the
    minimal arm set for one cell so it can be run at around a thousand
    sequences, where the detectable effect falls to roughly 1.6.
    """
    return [
        dict(T=T, selector="confidence", mode="none", warmup=0.0,
             family="baseline", tag="confirm"),
        dict(T=T, selector="contact", mode=mode, warmup=warmup,
             family="warmup", tag="confirm"),
        dict(T=T, selector="contact_shuffled", mode=mode, warmup=warmup,
             family="control", tag="confirm"),
    ]


def build_arms(steps, warmups, quick=False, self_sweep=False):
    """Arm grid. self_sweep moves the warm-up sweep onto the FREE signal.

    The deposited-structure measurement says the generator's own attention
    is the only signal with information in the regime the selector runs in,
    so that is the sweep worth running at full width; the external arm then
    appears once, as the cost comparison.
    """
    arms = []
    if self_sweep:
        for T in steps:
            arms.append(dict(T=T, selector="confidence", mode="none",
                             warmup=0.0, family="baseline"))
            for w in warmups:
                arms.append(dict(T=T, selector="contact", mode="self",
                                 warmup=w, family="warmup"))
                arms.append(dict(T=T, selector="contact_shuffled",
                                 mode="self", warmup=w, family="control"))
            arms.append(dict(T=T, selector="contact", mode="rawattn",
                             warmup=0.35, family="free_rawattn"))
        return arms
    for T in steps:
        arms.append(dict(T=T, selector="confidence", mode="none",
                         warmup=0.0, family="baseline"))
        for w in warmups:
            arms.append(dict(T=T, selector="contact", mode="external",
                             warmup=w, family="warmup"))
        arms.append(dict(T=T, selector="contact_shuffled", mode="external",
                         warmup=0.5, family="control"))
        arms.append(dict(T=T, selector="contact", mode="self",
                         warmup=0.5, family="free"))
        arms.append(dict(T=T, selector="contact_shuffled", mode="self",
                         warmup=0.5, family="free_control"))
    if quick:
        return arms
    T0 = steps[len(steps) // 2]
    # refresh policy at fixed warm-up: eager, default, lazy, and a
    # fixed-interval equivalent that ignores the distribution signal
    for rd, ra, tag in ((0.05, 4, "eager"), (0.15, 8, "default"),
                        (0.50, 16, "lazy"), (9.9, 1, "every_step"),
                        (9.9, 8, "fixed_age8")):
        arms.append(dict(T=T0, selector="contact", mode="external",
                         warmup=0.5, refresh_dist=rd, refresh_max_age=ra,
                         family="refresh", tag=tag))
    # group-size cap
    for gm in (8, 16):
        arms.append(dict(T=T0, selector="contact", mode="external",
                         warmup=0.5, g_max=gm, family="gmax", tag="g%d" % gm))
        arms.append(dict(T=T0, selector="confidence", mode="none",
                         warmup=0.0, g_max=gm, family="gmax_baseline",
                         tag="g%d" % gm))
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="airkingbd/dplm_150m")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--warmups", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--refresh-dist", type=float, default=0.15)
    ap.add_argument("--refresh-reveal", type=float, default=0.10)
    ap.add_argument("--refresh-max-age", type=int, default=8)
    ap.add_argument("--tau-q", type=float, default=0.90)
    ap.add_argument("--contact-size", type=int, default=650)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--confirm", action="store_true",
                    help="high-power confirmation of ONE cell: three arms "
                         "only (baseline, contact, shuffled control) at "
                         "--confirm-mode and --confirm-warmup")
    ap.add_argument("--confirm-mode", default="self",
                    choices=["self", "rawattn", "external"])
    ap.add_argument("--confirm-warmup", type=float, default=0.0)
    ap.add_argument("--self-sweep", action="store_true",
                    help="sweep the warm-up on the generator's own attention "
                         "(the only signal with information while masked) "
                         "instead of on a separate frozen ESM-2 pass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # eager attention is REQUIRED for the self/rawattn arms: the SDPA path
    # returns None for attentions from transformers 4.56. One model serves
    # every arm, and the extra cost of asking for attentions is measured
    # below and charged only to the arms that need them.
    tok, model = load_dplm(a.model, device=dev, need_attn=True)
    aa_ids = [tok.convert_tokens_to_ids(c) for c in AA20]
    assert all(i is not None and i >= 0 for i in aa_ids), aa_ids

    ext = ExternalContacts(a.contact_size, device=dev, half=False)
    slf = SelfContacts(150, device=dev)
    raw = RawAttentionContacts(device=dev)
    providers = {"external": ext, "self": slf, "rawattn": raw, "none": None}

    gen_s, gen_attn_s, con_s, ratio, attn_factor = measure_cost_ratio(
        tok, model, ext, L=a.length, B=min(a.batch, 8), device=dev)
    print("MEASURED cost: denoiser %.4f s, denoiser+attn %.4f s (x%.2f), "
          "ESM-2 %d contact %.4f s (x%.2f vs denoiser)"
          % (gen_s, gen_attn_s, attn_factor, a.contact_size, con_s, ratio),
          flush=True)

    if a.confirm:
        if len(a.steps) != 1:
            raise SystemExit("--confirm takes exactly one --steps value")
        arms = build_confirm_arms(a.steps[0], a.confirm_mode,
                                  a.confirm_warmup)
    else:
        arms = build_arms(a.steps, a.warmups, quick=a.quick,
                          self_sweep=a.self_sweep)
    out = {"config": vars(a),
           "cost": dict(gen_s=gen_s, gen_attn_s=gen_attn_s, contact_s=con_s,
                        ratio=ratio, attn_factor=attn_factor), "cells": []}
    t_start = time.time()
    for ci, arm in enumerate(arms):
        T = arm["T"]
        seqs, diags = [], []
        for b0 in range(0, a.n, a.batch):
            B = min(a.batch, a.n - b0)
            s, d = generate_adaptive(
                tok, model, a.length, B, T,
                selector=arm["selector"],
                contact_provider=providers[arm["mode"]],
                contact_mode=arm["mode"] if arm["mode"] != "none" else "external",
                aa_ids=aa_ids,
                warmup=arm["warmup"],
                refresh_dist=arm.get("refresh_dist", a.refresh_dist),
                refresh_reveal=arm.get("refresh_reveal", a.refresh_reveal),
                refresh_max_age=arm.get("refresh_max_age", a.refresh_max_age),
                g_max=arm.get("g_max"),
                tau_q=a.tau_q,
                seed=a.seed + 1000 * ci + b0, device=dev)
            seqs.extend(s)
            diags.append(d)
        rec = dict(arm)
        rec["seqs"] = seqs
        for k in ("nfe_gen", "nfe_contact", "steps", "refreshes",
                  "gated_steps", "unfinished"):
            rec[k] = float(np.mean([d[k] for d in diags]))
        rec["seconds"] = float(np.sum([d["seconds"] for d in diags]))
        # Cost in denoiser-pass equivalents. An arm reading the generator's
        # own attentions pays attn_factor per denoiser pass; an arm calling a
        # separate ESM-2 pays `ratio` per contact call. Charging only
        # nfe_contact would make the free signal look free when it is not.
        gen_mult = attn_factor if arm["mode"] in ("self", "rawattn") else 1.0
        rec["gen_mult"] = gen_mult
        rec["cost_eq"] = gen_mult * rec["nfe_gen"] + ratio * rec["nfe_contact"]
        tc = {}
        for d in diags:
            for k, v in d["trigger_counts"].items():
                tc[k] = tc.get(k, 0) + v
        rec["trigger_counts"] = tc
        for k in ("within_coupling", "coupled_pair_frac", "group_sizes",
                  "refresh_at_revealed", "dist_change_trace"):
            vals = [v for d in diags for v in d.get(k, [])]
            rec[k] = float(np.mean(vals)) if vals else None
        out["cells"].append(rec)
        print("[%d/%d] T=%d %s/%s warm=%.2f%s nfe=%.0f+%.0f cost=%.1f "
              "steps=%.0f refresh=%.1f %s %.1fs | %s"
              % (ci + 1, len(arms), T, arm["selector"], arm["mode"],
                 arm["warmup"],
                 (" " + arm["tag"]) if arm.get("tag") else "",
                 rec["nfe_gen"], rec["nfe_contact"], rec["cost_eq"],
                 rec["steps"], rec["refreshes"], rec["trigger_counts"],
                 rec["seconds"], seqs[0][:36]), flush=True)
        json.dump(out, open(a.out, "w"))
    print("all arms done in %.1f min -> %s"
          % ((time.time() - t_start) / 60, a.out), flush=True)


if __name__ == "__main__":
    main()
