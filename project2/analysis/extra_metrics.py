"""Consolidate the non-ESMFold metrics into one table per dataset.

Reads `results/L_extra_*.json`, `results/L_maxid_*.json` and, where it exists,
`results/M_reftm_*.json`, and prints the comparison the conditioning claim
needs: for each arm, what ESMFold says beside what four independent
instruments say.

The question each column answers:

    pLDDT        ESMFold's confidence -- the axis the whole claim rested on
    NSR, edit    is the pattern a retrieval key? (floors printed beneath)
    Max-ID       did anything come back close to a training protein?
    FPD, MMD     did the generated DISTRIBUTION move toward natural?
    Ref-TM       did it adopt the SOURCE's fold? (CATH only)

Usage:
    python -m analysis.extra_metrics
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")

TAGS = [("J_orig", "SwissProt, original corpus"),
        ("J_purged", "SwissProt, homologs purged"),
        ("K_orig", "CATH S35, original corpus"),
        ("K_purged", "CATH S35, homologs purged")]

LABEL = {"dms6|conditioned": "chemical, conditioned",
         "dms6|uncond": "chemical, unconditional",
         "rand6|conditioned": "RANDOM partition, conditioned",
         "rand6|uncond": "RANDOM partition, unconditional",
         "natural": "natural proteins",
         "source": "the source slice itself"}
ORDER = ["dms6|conditioned", "rand6|conditioned", "dms6|uncond",
         "rand6|uncond", "source", "natural"]


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def by_sel(d):
    return {c["selector"]: c for c in d["cells"]} if d else {}


def fmt(v, spec="%.3f"):
    return "--" if v is None else (spec % v)


def main():
    for tag, title in TAGS:
        E = load(f"L_extra_{tag}.json")
        M = load(f"L_maxid_{tag}.json")
        R = load(f"M_reftm_{tag.replace('K_', 'K_')}.json") \
            or load(f"M_reftm_{tag}.json")
        if not E:
            print(f"\n=== {title}: not scored yet")
            continue
        e, m, r = by_sel(E), by_sel(M), by_sel(R)
        print(f"\n=== {title}")
        print("%-32s %7s %7s %7s %7s %8s %8s %7s"
              % ("arm", "pLDDT", "pppl", "NSR", "Max-ID", "FPD", "MMD",
                 "Ref-TM"))
        for sel in ORDER:
            if sel not in e and sel not in r:
                continue
            c, mc, rc = e.get(sel, {}), m.get(sel, {}), r.get(sel, {})
            print("%-32s %7s %7s %7s %7s %8s %8s %7s" % (
                LABEL.get(sel, sel),
                fmt(c.get("plddt_mean"), "%.1f"),
                fmt(c.get("pppl_mean"), "%.2f"),
                fmt(c.get("nsr")),
                fmt(mc.get("max_id_mean"), "%.1f"),
                fmt(c.get("fpd_vs_sources"), "%.2f"),
                fmt(c.get("mmd_vs_sources"), "%.4f"),
                fmt(rc.get("ref_tm"))))
        cond = e.get("dms6|conditioned", {})
        if cond.get("nsr_floor_uniform_in_family") is not None:
            print("  NSR floors for the chemical arm: uniform-in-family "
                  "%.3f, modal-in-family %.3f, composition %.3f"
                  % (cond["nsr_floor_uniform_in_family"],
                     cond["nsr_floor_modal_in_family"],
                     cond["nsr_floor"]))
        rnd = e.get("rand6|conditioned", {})
        if rnd.get("nsr_floor_uniform_in_family") is not None:
            print("  NSR floors for the random arm:   uniform-in-family "
                  "%.3f, modal-in-family %.3f"
                  % (rnd["nsr_floor_uniform_in_family"],
                     rnd["nsr_floor_modal_in_family"]))
        if r:
            print("  Ref-TM >= 0.5: " + ", ".join(
                "%s %.0f %%" % (LABEL.get(k, k),
                                100 * v.get("ref_tm_frac_ge_0.5", 0))
                for k, v in r.items()))

    print("\nReadings to check against the claim:")
    print("  * NSR at or near its family floor  -> the pattern constrains "
          "rather than retrieves")
    print("  * chemical FPD/MMD well below random -> an axis independent of "
          "ESMFold agrees with the chemistry claim")
    print("  * Ref-TM near the unrelated-pair background (~0.3) -> the pLDDT "
          "gain is confidence, not the source fold")


if __name__ == "__main__":
    main()
