"""Does the argmax-repaired contact map help the gate, in the configurations
the first run did not cover?

The first argmax-gate run was one setting: absorbing decoder, no warm-up,
budgets 25 and 50. Two gaps, both favouring the method:

  remasking decoder   the one DPLM ships, and where the original contact
                      rule was most harmful (-10.73 pooled)
  warm-up 0.5         the repaired signal is at the all-mask null below 25 %
                      revealed and only informative above 50 %, so a run
                      without a warm-up spends half its gating decisions on
                      a signal that does not exist

Reports pLDDT, the shuffled-control contrast (which is what isolates
structural information), cost in denoiser-pass equivalents, and diversity
and max-repeat beside them -- a contact arm that collapses to a homopolymer
can score well on pLDDT, so those columns are not optional here.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COST_EXT = 7.88
RNG = np.random.default_rng(0)
NB = 20000


def cells(p):
    d = json.load(open(p))
    return {(c["selector"], c["T"]): c
            for c in d["cells"] if c.get("T") is not None and "plddt_list" in c}


def boot(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.array([RNG.choice(a, len(a), True).mean()
                  - RNG.choice(b, len(b), True).mean() for _ in range(NB)])
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])


def cost(c):
    return (c.get("nfe_gen") or 0) + COST_EXT * (c.get("nfe_contact") or 0)


def report(path, base, treat, ctrl, label):
    if not os.path.exists(path):
        print("\n### %s\nMISSING %s" % (label, path))
        return None
    C = cells(path)
    Ts = sorted({k[1] for k in C})
    print("\n### %s" % label)
    print("| T | arm | pLDDT | cost | contact calls | diversity | max repeat |")
    print("|---|---|---|---|---|---|---|")
    for T in Ts:
        for s in (base, treat, ctrl):
            c = C.get((s, T))
            if not c:
                continue
            print("| %d | %s | %.2f | %.0f | %d | %.3f | %.1f |"
                  % (T, s, np.mean(c["plddt_list"]), cost(c),
                     c.get("nfe_contact") or 0, c.get("diversity", float("nan")),
                     c.get("max_repeat", float("nan"))))
    pool = {s: np.concatenate([C[(s, T)]["plddt_list"] for T in Ts
                               if (s, T) in C]) for s in (base, treat, ctrl)}
    n = len(pool[base])
    print("\npooled over %s, %d sequences per arm:" % (Ts, n))
    print("  %-26s %.2f" % (base, pool[base].mean()))
    print("  %-26s %.2f" % (treat, pool[treat].mean()))
    print("  %-26s %.2f" % (ctrl, pool[ctrl].mean()))
    r1 = boot(pool[treat], pool[base])
    r2 = boot(pool[treat], pool[ctrl])
    print("  contact - confidence       %+.2f [%+.2f, %+.2f]" % r1)
    print("  contact - shuffled         %+.2f [%+.2f, %+.2f]   <-- isolates structure" % r2)
    mean_cost = np.mean([cost(C[(treat, T)]) / max(cost(C[(base, T)]), 1e-9)
                         for T in Ts if (treat, T) in C and (base, T) in C])
    print("  relative cost of the gate  %.2fx" % mean_cost)
    return dict(label=label, treat=r1, ctrl=r2, cost=mean_cost)


if __name__ == "__main__":
    out = []
    out.append(report("results/H_argmax_reparam_w0.0_scored.json",
                      "reparam_confidence", "reparam_contact",
                      "reparam_contact_shuffled",
                      "A. remasking decoder, repaired map, no warm-up"))
    out.append(report("results/H_argmax_reparam_w0.5_scored.json",
                      "reparam_confidence", "reparam_contact",
                      "reparam_contact_shuffled",
                      "A2. remasking decoder, repaired map, warm-up 0.5"))
    out.append(report("results/H_argmax_abs_w0.5_scored.json",
                      "confidence", "contact", "contact_shuffled",
                      "B. absorbing decoder, repaired map, warm-up 0.5"))
    print("\n\n## SUMMARY: contact minus its own shuffled control")
    print("| configuration | effect | 95 % CI | gate cost |")
    print("|---|---|---|---|")
    for r in out:
        if r:
            print("| %s | %+.2f | [%+.2f, %+.2f] | %.2fx |"
                  % (r["label"], r["ctrl"][0], r["ctrl"][1], r["ctrl"][2],
                     r["cost"]))
    print("\nFor reference, the first run (absorbing, no warm-up, 96/arm):")
    print("  argmax-map contact - confidence  -4.41 [-9.40, +0.60]")
    print("  argmax-map - plain-map contact   -0.60 [-5.43, +4.34]")
