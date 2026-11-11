"""The accuracy budget for a predicted family pattern, against what is achieved.

Puts two numbers side by side:

  REQUIRED -- from the degradation curve, the pattern error rate at which the
  conditioning gain falls to half and to zero. Patterns are the same 96
  sources at every corruption level, so the per-source contrast is paired.

  ACHIEVED -- the family head's own error rate, two ways. Its argmax error is
  1 - accuracy. A pattern it SAMPLES rather than argmaxes has error about
  1 - sum_f p_f^2 if it has only learned the marginal, which is the
  marginal-match floor recorded by the diagnostics.

If REQUIRED is far below ACHIEVED, "predict the family label before the exact
amino acid" cannot work at this scale, and the reason is not the architecture
but that a coarse family pattern is only useful when it is nearly exact.

The selfpattern arm is the direct empirical check on the same conclusion: it
feeds the model's own generated pattern through the identical conditioning
path, so it should land where the curve says its error rate puts it.
"""
import json
import os
import sys

import numpy as np

RNG = np.random.default_rng(0)
NB = 20000


def ci(x):
    x = np.asarray(x, float)
    bs = np.array([RNG.choice(x, len(x), True).mean() for _ in range(NB)])
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def load(path):
    d = json.load(open(path))
    return {c["selector"]: c for c in d["cells"] if "plddt_list" in c}


def crossing(fs, gains, target):
    """First f at which the gain falls below `target`, linearly interpolated."""
    for i in range(1, len(fs)):
        if gains[i] <= target <= gains[i - 1]:
            lo, hi = gains[i - 1], gains[i]
            if hi == lo:
                return fs[i]
            w = (lo - target) / (lo - hi)
            return fs[i - 1] + w * (fs[i] - fs[i - 1])
    return float("nan")


def main(curve="results/I_curve_dms6_scored.json",
         diag="results/I_curve_dms6_diag.json",
         base="results/G_cond_clean_scored.json",
         tag="dms6"):
    A = load(curve)
    corrupt = sorted([(c["corrupt_frac"], c) for k, c in A.items()
                      if c.get("kind") == "corrupt"])
    if not corrupt:
        print("no corruption cells scored yet in %s" % curve)
        return

    uncond = None
    if os.path.exists(base):
        B = load(base)
        k = "%s|uncond" % tag
        if k in B:
            uncond = np.asarray(B[k]["plddt_list"], float)
    if uncond is None:
        print("WARNING: no unconditional baseline found; gains are relative "
              "to the f=1 cell instead")
        uncond = np.asarray(corrupt[-1][1]["plddt_list"], float)

    u = float(uncond.mean())
    print("unconditional baseline: %.2f pLDDT\n" % u)
    print("%-8s %-10s %-22s %-22s" % ("error", "pLDDT", "gain over uncond",
                                      "loss vs perfect pattern"))
    p0 = np.asarray(corrupt[0][1]["plddt_list"], float)
    fs, gains = [], []
    for f, c in corrupt:
        x = np.asarray(c["plddt_list"], float)
        g, glo, ghi = ci(x - u)
        n = min(len(x), len(p0))
        d, dlo, dhi = ci(x[:n] - p0[:n])          # paired per source
        print("%-8.2f %-10.2f %+6.2f [%+.2f, %+.2f]   %+6.2f [%+.2f, %+.2f]"
              % (f, x.mean(), g, glo, ghi, d, dlo, dhi))
        fs.append(f)
        gains.append(g)

    g0 = gains[0]
    f_half = crossing(fs, gains, 0.5 * g0)
    f_zero = crossing(fs, gains, 0.0)
    slope = (gains[1] - gains[0]) / (fs[1] - fs[0]) if len(fs) > 1 else \
        float("nan")
    print("\n=== REQUIRED: the error budget ===")
    print("  gain with a perfect pattern      %+.2f pLDDT" % g0)
    print("  initial slope                    %.2f pLDDT per 1 %% error"
          % (slope / 100))
    print("  error rate that halves the gain  %.3f" % f_half)
    print("  error rate that erases the gain  %.3f" % f_zero)

    print("\n=== ACHIEVED: the family head's own error rate ===")
    if os.path.exists(diag):
        D = json.load(open(diag))
        fa = D["family_accuracy"]
        acc0 = fa["by_reveal"]["0.0"]["acc"]
        floor = fa["majority_floor"]
        mm = fa["marginal_match_floor"]
        print("  argmax accuracy (no context)     %.4f  -> error %.3f"
              % (acc0, 1 - acc0))
        print("  majority-class floor             %.4f" % floor)
        print("  a pattern SAMPLED from a head that only learned the "
              "marginal\n  would match the truth at          %.4f  -> error "
              "%.3f" % (mm, 1 - mm))
        print("  composition, real vs generated   %s vs %s"
              % ([round(x, 3) for x in D["real_pattern_stats"]["composition"]],
                 [round(x, 3) for x in D["self_pattern_stats"]["composition"]]))
        req = f_zero
        print("\n  REQUIRED error <= %.3f to retain anything; ACHIEVED error "
              "%.3f to %.3f" % (req, 1 - acc0, 1 - mm))
        if not np.isnan(req) and (1 - acc0) > req:
            print("  -> the achieved error is %.0fx the budget. Predicting "
                  "the family first cannot work at this scale: the pattern "
                  "helps only while it is nearly exact." % ((1 - acc0) / req))
    else:
        print("  %s not found" % diag)

    print("\n  CAVEAT on reading the budget this way: the curve corrupts "
          "positions\n  UNIFORMLY, while a real predictor's errors are "
          "correlated and may fall\n  on positions that matter less. A head "
          "at 64 % uniform error and a head\n  at 64 % structured error need "
          "not score the same, so the curve gives the\n  budget under the "
          "pessimistic assumption. The selfpattern arm below is\n  the direct "
          "test, because it carries the model's real error structure.")

    sp = A.get("%s|selfpattern" % tag)
    if sp is not None:
        x = np.asarray(sp["plddt_list"], float)
        g, lo, hi = ci(x - u)
        print("\n=== the direct check: the model's own pattern ===")
        print("  selfpattern %.2f pLDDT, gain %+.2f [%+.2f, %+.2f] over "
              "unconditional" % (x.mean(), g, lo, hi))
        print("  perfect pattern gains %+.2f, so the model's own pattern "
              "retains %.0f %% of it" % (g0, 100 * g / g0 if g0 else float("nan")))

    json.dump(dict(fracs=fs, gains=gains, g0=g0, f_half=f_half,
                   f_zero=f_zero, slope_per_pct=slope / 100, uncond=u),
              open("results/I_curve_%s_summary.json" % tag, "w"), indent=1)
    print("\nWROTE results/I_curve_%s_summary.json" % tag)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
