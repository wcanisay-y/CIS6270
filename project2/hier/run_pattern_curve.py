"""How accurate must a family pattern be, and how accurate is a predicted one?

This answers the proposal "learn the family channel in the diffusion model and
predict the family label before the exact amino acid". That architecture is
already trained (ckpt/dms6.pt) and already measured end to end, and it LOSES:
36.58 pLDDT self-predicting families against 43.09 for a flat 20-letter model
with no family channel, while the same model given a REAL pattern reaches
56.49. So the channel works as a receiver and fails as a generator, and the
open question is quantitative: how good does a predicted pattern have to be?

Three measurements, all on one frozen checkpoint:

  1. DEGRADATION CURVE. Corrupt a fraction f of a real pattern's positions to
     a different random family and condition on the result. This is the
     accuracy budget: it says what pattern error rate still buys anything.
     f = 0 is the oracle pattern, f = 1 is a fully random pattern, and the
     published random-partition control sits at the f = 1 end.

  2. SELF-PATTERN ARM. Let the model generate a family pattern on its own
     (the family phase of the two-level sampler), then feed that pattern back
     through the SAME conditioning path used for real patterns. The identity
     phase is then identical to the oracle arm, so the gap is purely pattern
     quality -- which the end-to-end number conflates with sampler effects.

  3. FAMILY PREDICTION ACCURACY. Teacher-forced: how well does fam_head
     recover the true family at the identity mask rates the sampler actually
     runs at, against the majority-class floor. Read together with the curve,
     this says whether the gap in 2 is closable.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.alphabet import AA20, AA_IDX, PARTITIONS, family_table, \
    random_family_table
from hier.model import ID_MASK
from hier.run_generate import load_arm
from hier.sample import sample_hier_conditioned, family_compliance, \
    _cat64, _sched, _commit_topk


@torch.no_grad()
def generate_family_pattern(model, B, L, steps, t_fam=1.0, device="cuda",
                            seed=0, rank="gumbel"):
    """The family phase alone: start from nothing, write a family pattern.

    This is the generator side of the proposal. Identities stay masked, so
    the model writes families from the same state the conditioned arm starts
    from, and the resulting pattern is directly substitutable for a real one.
    """
    torch.manual_seed(seed)
    K = model.n_fam
    ids = torch.full((B, L), ID_MASK, dtype=torch.long, device=device)
    fam = torch.full((B, L), K, dtype=torch.long, device=device)
    rows = torch.arange(B, device=device)[:, None]
    for k in _sched(L, steps):
        _, fl = model(ids, fam, bind=True)
        tok, conf = _cat64(fl, t_fam, rank)
        sel = _commit_topk(conf, fam == K, k)
        if sel is None:
            continue
        fam[rows, sel] = tok[rows, sel]
    if (fam == K).any():
        _, fl = model(ids, fam, bind=True)
        tok, _ = _cat64(fl, t_fam, rank)
        fam = torch.where(fam == K, tok, fam)
    return fam.cpu().numpy()


def corrupt_pattern(pat, f, n_fam, rng):
    """Replace a fraction f of positions with a DIFFERENT random family.

    Forcing a change rather than resampling uniformly makes f the true error
    rate: at f = 1 every position is wrong, which is the honest right-hand
    end of the curve.
    """
    out = pat.copy()
    B, L = out.shape
    k = int(round(f * L))
    if k == 0:
        return out
    for b in range(B):
        pos = rng.choice(L, k, replace=False)
        for p in pos:
            alt = rng.integers(0, n_fam - 1)
            out[b, p] = alt if alt < out[b, p] else alt + 1
    return out


@torch.no_grad()
def family_accuracy(model, seqs, lut, n_fam, reveal_fracs, device, seed=0):
    """Teacher-forced family recovery at several identity-reveal fractions."""
    rng = np.random.default_rng(seed)
    L = len(seqs[0])
    aa = torch.as_tensor(
        np.stack([[AA_IDX[c] for c in s] for s in seqs]), dtype=torch.long,
        device=device)
    true_fam = torch.as_tensor(lut, dtype=torch.long, device=device)[aa]
    # majority-class floor on this very set
    cnt = np.bincount(true_fam.cpu().numpy().ravel(), minlength=n_fam)
    floor = float(cnt.max() / cnt.sum())
    p = cnt / cnt.sum()
    out = {"majority_floor": floor,
           "marginal_match_floor": float((p ** 2).sum()),
           "by_reveal": {}}
    for rf in reveal_fracs:
        keep = torch.as_tensor(
            np.stack([rng.random(L) < rf for _ in seqs]), device=device)
        ids = torch.where(keep, aa, torch.full_like(aa, ID_MASK))
        fam = torch.full_like(aa, n_fam)          # every family masked
        _, fl = model(ids, fam, bind=True)
        pred = fl.argmax(-1)
        acc = float((pred == true_fam).float().mean())
        # accuracy restricted to positions whose identity is hidden, which is
        # where the sampler actually has to predict
        hid = ~keep
        acc_hidden = float((pred[hid] == true_fam[hid]).float().mean()) \
            if bool(hid.any()) else float("nan")
        # The sampler does not commit families uniformly: it takes the most
        # confident positions first. Accuracy over the top decile of
        # confidence is therefore the number that governs its early commits,
        # and uniform accuracy understates it.
        conf = fl.log_softmax(-1).max(-1).values
        k = max(1, int(0.1 * conf.numel()))
        top = conf.ravel().topk(k).indices
        acc_top10 = float((pred.ravel()[top] == true_fam.ravel()[top])
                          .float().mean())
        out["by_reveal"][str(rf)] = dict(acc=acc, acc_hidden=acc_hidden,
                                         acc_top10conf=acc_top10)
    return out


def pattern_stats(pat, n_fam):
    """Composition, run length and lag-3/4 autocorrelation of a pattern set."""
    comp = np.bincount(pat.ravel(), minlength=n_fam).astype(float)
    comp /= comp.sum()
    runs = []
    for row in pat:
        c = 1
        for i in range(1, len(row)):
            if row[i] == row[i - 1]:
                c += 1
            else:
                runs.append(c); c = 1
        runs.append(c)
    # family 0 is the aliphatic/hydrophobic group in dms6; its periodicity is
    # the signal the proposal's own premise was about
    hyd = (pat == 0).astype(float)
    hyd = hyd - hyd.mean(axis=1, keepdims=True)
    ac = {}
    for lag in (2, 3, 4):
        num = (hyd[:, :-lag] * hyd[:, lag:]).mean()
        den = (hyd * hyd).mean()
        ac["lag%d" % lag] = float(num / den) if den > 0 else float("nan")
    return dict(composition=comp.tolist(), run_mean=float(np.mean(runs)),
                autocorr=ac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/dms6.pt")
    ap.add_argument("--partition", default="dms6")
    ap.add_argument("--random-partition", action="store_true")
    ap.add_argument("--pattern-file", default="results/G_clean_patterns.json")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag-out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, ck = load_arm(a.ckpt, dev)
    tag = os.path.basename(a.ckpt).replace(".pt", "")
    base = PARTITIONS[a.partition]
    if a.random_partition:
        names, lut = random_family_table(base, 0)
    else:
        names, lut = family_table(base)
    n_fam = len(names)
    print("%s: n_fam=%d partition=%s%s" % (tag, n_fam, a.partition,
          " RANDOMISED" if a.random_partition else ""), flush=True)

    pf = json.load(open(a.pattern_file))
    nat = [s[:a.length] for s in pf["seqs"] if len(s) >= a.length][:a.n]
    print("patterns: %d from %s" % (len(nat), a.pattern_file), flush=True)
    real = np.stack([lut[[AA_IDX[c] for c in s]] for s in nat])

    out = dict(cells=[], config=vars(a))
    rng = np.random.default_rng(a.seed)

    # ---- 1. the degradation curve
    for fi, f in enumerate(a.fracs):
        pat = corrupt_pattern(real, f, n_fam, rng)
        err = float((pat != real).mean())
        seqs, t0 = [], time.time()
        for b0 in range(0, len(nat), a.batch):
            B = min(a.batch, len(nat) - b0)
            seqs.extend(sample_hier_conditioned(
                m, pat[b0:b0 + B], a.steps, t_id=1.0, device=dev,
                seed=1000 * fi + b0))
        out["cells"].append(dict(
            arm=ck["args"]["arm"], tag=tag,
            selector="%s|corrupt%.2f" % (tag, f), L=a.length, T=a.steps,
            nfe=a.steps, temp=1.0, kind="corrupt", corrupt_frac=f,
            measured_error=err, seqs=seqs,
            family_compliance=family_compliance(seqs, pat, lut),
            seconds=time.time() - t0, best_val=ck["best_val"],
            params=ck["params"]))
        print("  f=%.2f measured err %.3f  %.0fs  %s"
              % (f, err, time.time() - t0, seqs[0][:40]), flush=True)

    # ---- 2. the self-generated pattern, fed through the same path
    selfpat = []
    for b0 in range(0, len(nat), a.batch):
        B = min(a.batch, len(nat) - b0)
        selfpat.append(generate_family_pattern(
            m, B, a.length, a.steps, device=dev, seed=777 + b0))
    selfpat = np.concatenate(selfpat)[:len(nat)]
    seqs, t0 = [], time.time()
    for b0 in range(0, len(nat), a.batch):
        B = min(a.batch, len(nat) - b0)
        seqs.extend(sample_hier_conditioned(
            m, selfpat[b0:b0 + B], a.steps, t_id=1.0, device=dev,
            seed=5000 + b0))
    out["cells"].append(dict(
        arm=ck["args"]["arm"], tag=tag, selector="%s|selfpattern" % tag,
        L=a.length, T=a.steps, nfe=a.steps, temp=1.0, kind="selfpattern",
        seqs=seqs, family_compliance=family_compliance(seqs, selfpat, lut),
        seconds=time.time() - t0, best_val=ck["best_val"],
        params=ck["params"]))
    print("  selfpattern  %.0fs  %s" % (time.time() - t0, seqs[0][:40]),
          flush=True)

    json.dump(out, open(a.out, "w"))
    print("WROTE %s" % a.out, flush=True)

    # ---- 3. diagnostics: prediction accuracy and pattern realism
    diag = dict(
        family_accuracy=family_accuracy(
            m, nat, lut, n_fam, [0.0, 0.1, 0.25, 0.5, 0.75], dev, a.seed),
        real_pattern_stats=pattern_stats(real, n_fam),
        self_pattern_stats=pattern_stats(selfpat, n_fam),
        # NOT a quality measure: a self-generated pattern is not trying to
        # reproduce THIS protein, so agreement should sit at the chance level
        # for two independent draws from the family marginal. It is recorded
        # only to confirm the self patterns are not copies.
        self_vs_real_agreement=float((selfpat == real).mean()),
        family_names=names)
    dp = a.diag_out or a.out.replace(".json", "_diag.json")
    json.dump(diag, open(dp, "w"), indent=1)
    fa = diag["family_accuracy"]
    print("\nfamily prediction accuracy (majority floor %.3f):"
          % fa["majority_floor"])
    for rf, d in fa["by_reveal"].items():
        print("  identities revealed %4s: all %.3f  hidden-only %.3f  "
              "top-decile confidence %.3f"
              % (rf, d["acc"], d["acc_hidden"], d["acc_top10conf"]))
    print("self pattern vs a real pattern, position agreement: %.3f "
          "(chance for independent draws %.3f -- equality here means the "
          "self patterns are not copies, not that they are bad)"
          % (diag["self_vs_real_agreement"], fa["marginal_match_floor"]))
    print("run length  real %.2f  self %.2f"
          % (diag["real_pattern_stats"]["run_mean"],
             diag["self_pattern_stats"]["run_mean"]))
    print("WROTE %s" % dp)


if __name__ == "__main__":
    main()
