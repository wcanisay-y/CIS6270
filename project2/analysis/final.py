"""Assemble every scored result file into RESULTS.md."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from analysis.report import boot_diff, mwu            # noqa: E402

RES = os.path.join(HERE, "results")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    if isinstance(d, list):          # the contact-quality file is a flat list
        return d
    for c in d.get("cells", []):
        c["_src"] = name
    return d.get("cells", [])


def fmt(v, f="%.2f"):
    try:
        return f % v if v is not None and np.isfinite(v) else "-"
    except Exception:
        return "-"


COLS = [("plddt_mean", "pLDDT", "%.2f"), ("ptm_mean", "pTM", "%.3f"),
        ("pae_mean", "pAE", "%.1f"), ("foldability", "fold", "%.2f"),
        ("pppl_mean", "pppl", "%.2f"),
        ("amphipathic_index", "amph", "%.3f"),
        ("amph_window_frac", "amphW", "%.3f"),
        ("amph_patterning", "amphW-shuf", "%.3f"),
        ("diversity", "div", "%.3f"), ("novelty_kmer", "novel", "%.3f"),
        ("max_repeat", "rep", "%.1f")]


def table(cells, name_fn, title, note=""):
    if not cells:
        return ""
    out = [f"\n### {title}\n"]
    if note:
        out.append(note + "\n")
    out.append("| cell | " + " | ".join(h for _, h, _ in COLS) + " |")
    out.append("|" + "---|" * (len(COLS) + 1))
    for c in cells:
        row = [name_fn(c)] + [fmt(c.get(k), f) for k, _, f in COLS]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def contrast_table(cells, baseline, group_fn, title, label_fn):
    rows = []
    groups = {}
    for c in cells:
        groups.setdefault(group_fn(c), []).append(c)
    for g in sorted(groups, key=lambda z: (z is None, z)):
        gc = groups[g]
        base = [c for c in gc if label_fn(c) == baseline]
        if not base or not base[0].get("plddt_list"):
            continue
        bl = base[0]["plddt_list"]
        for c in gc:
            nm = label_fn(c)
            if nm == baseline or not c.get("plddt_list"):
                continue
            r = boot_diff(c["plddt_list"], bl)
            star = ("***" if r["p"] < 0.001 else "**" if r["p"] < 0.01
                    else "*" if r["p"] < 0.05 else "")
            rows.append((g, nm, r, mwu(c["plddt_list"], bl), star))
    if not rows:
        return ""
    out = [f"\n### {title}\n",
           f"pLDDT difference from `{baseline}`, bootstrap 95% interval.\n",
           "| group | arm | difference | 95% interval | p | |",
           "|---|---|---|---|---|---|"]
    for g, nm, r, p2, star in rows:
        out.append(f"| {g} | {nm} | {r['diff']:+.2f} | "
                   f"[{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['p']:.4f} | {star} |")
    return "\n".join(out) + "\n"


def main():
    doc = ["# Results\n",
           "All pLDDT from ESMFold on a 0-100 scale, 48 sequences per cell, "
           "length 200 unless stated. Intervals are bootstrap 95 percent.\n"]

    # ---------------- Idea A
    doc.append("\n## Idea A: choosing the parallel unmasking set\n")
    fid = load("A_fidelity_scored.json")
    if fid:
        doc.append(table(
            sorted(fid, key=lambda c: (-(c.get("plddt_mean") or 0))),
            lambda c: f"{c['selector']} / T={c['T']}",
            "Harness fidelity",
            "Published DPLM-150M at length 200 is pLDDT 84.30. Our "
            "reimplementation of its decoder returns 83.60 at 200 forward "
            "passes and 87.44 at its default 500, so the harness is "
            "faithful. Every absorbing-sampler arm sits about 30 points "
            "lower, which is the gap between irrevocable commitment and "
            "DPLM's re-predict-and-remask decoder."))

    grid = load("Am_q99_scored.json")
    main_grid = [c for c in grid if c.get("note") != "tau_sweep"]
    if main_grid:
        doc.append(contrast_table(
            [c for c in main_grid if c.get("selector") != "natural"],
            "confidence", lambda c: f"T={c['T']}",
            "Matched forward passes: every selector against the baseline",
            lambda c: c.get("selector")))
        doc.append(table(
            sorted(main_grid, key=lambda c: (str(c.get("selector")),
                                             c.get("T") or 0)),
            lambda c: f"{c['selector']} / T={c['T']}",
            "Full matched-NFE grid"))
        tw = [c for c in grid if c.get("note") == "tau_sweep"]
        if tw:
            doc.append(table(tw, lambda c: f"contact tau={c['tau']}",
                             "Coupling threshold sweep at T=25"))

    for f, title, note in [
        ("A_reparam_scored.json", "The rule inside DPLM's own decoder",
         "Same forward passes and the same number of survivors per step; "
         "only which positions survive changes. This is the decoder DPLM "
         "ships and where its published quality comes from."),
        ("A_self_oracle_scored.json", "Self-oracle",
         "The baseline's own output is folded and its true contact map "
         "handed back for a second pass from the same seed."),
        ("A_clean_scored.json", "Commit ranking: clean probability",
         "Repeats the absorbing arms ranking commits by the clean "
         "probability of the sampled token instead of DPLM's "
         "Gumbel-perturbed score, to check the baseline is not a strawman."),
        ("A_oracle_scored.json", "Oracle: true contact maps",
         "The selector is handed a real protein's true contact map. If a "
         "correct map does not beat confidence ordering, no improvement in "
         "contact prediction can rescue the mechanism."),
        ("A_self_oracle_scored.json", "Self-oracle",
         "The baseline's own output is folded and its true contact map is "
         "handed back for a second pass from the same seed."),
        ("Am_q90_scored.json", "Stronger threshold: top 10 percent of pairs",
         ""),
        ("Am_ext_scored.json",
         "The proposal as written: a separate ESM-2 650M pass per step",
         "Doubles the per-step cost."),
        ("A_selfpartial_scored.json",
         "Secondary record: the non-binding threshold",
         "A fixed coupling threshold of 0.3 rejected 0.05 percent of pairs, "
         "so the rule barely bound. Kept only to document that.")]:
        cells = load(f)
        if not cells:
            continue
        doc.append(table(
            sorted(cells, key=lambda c: (str(c.get("selector")),
                                         c.get("T") or 0)),
            lambda c: f"{c['selector']} / T={c['T']}", title, note))
        base = ("reparam_confidence" if "reparam" in f else "confidence")
        if any(c.get("selector") == base for c in cells):
            doc.append(contrast_table(cells, base, lambda c: f"T={c['T']}",
                                      title + ": contrasts",
                                      lambda c: c.get("selector")))

    cq = load("A_contact_quality.json")
    if isinstance(cq, list) and cq and "frac" in cq[0]:
        doc.append("\n### Contact prediction quality against mask rate\n")
        doc.append("| input | AUROC ext | P@L ext | AUROC self | P@L self |")
        doc.append("|---|---|---|---|---|")
        nul = [r for r in cq if r.get("note") == "all_mask_null"]
        for fr in sorted({r["frac"] for r in cq if r["frac"] >= 0}):
            g = [r for r in cq if r["frac"] == fr]
            doc.append(f"| {fr*100:.0f}% masked | "
                       f"{np.mean([x['auroc_ext'] for x in g]):.3f} | "
                       f"{np.mean([x['prec_L_ext'] for x in g]):.3f} | "
                       f"{np.mean([x['auroc_self'] for x in g]):.3f} | "
                       f"{np.mean([x['prec_L_self'] for x in g]):.3f} |")
        if nul:
            n = nul[0]
            doc.append(f"| all-mask null | {n['auroc_ext']:.3f} | "
                       f"{n['prec_L_ext']:.3f} | {n['auroc_self']:.3f} | "
                       f"{n['prec_L_self']:.3f} |")
        doc.append("")

    # ---------------- Idea B
    doc.append("\n## Idea B: the two-level alphabet\n")
    for f, title, note in [
        ("B_main_scored.json", "Every arm at one matched setting",
         "Matched parameters, corpus, training budget and forward passes. "
         "`rand6` is the control that decides the idea: a random partition "
         "with the same group sizes as the chemical one."),
        ("B_nfe_scored.json", "Against forward passes", ""),
        ("B_qd_flat_scored.json", "Flat arm, temperature sweep", ""),
        ("B_qd_dms6_scored.json", "Two-level arm, family x identity sweep",
         "The claim is that two temperatures move the whole frontier "
         "outward, so this is compared with the flat curve above."),
        ("B_twophase_scored.json", "Out-of-distribution two-phase sampler",
         "Writes every family before any identity, a state the training "
         "corruption never produces."),
        ("B_cleanrank_collapsed_scored.json",
         "Collapsed run, kept as a warning",
         "Ranking commits by clean probability produced homopolymer runs of "
         "109 to 113 residues out of 200 at diversity 0.08, while scoring "
         "pLDDT 84.6 for the flat arm. Not a result, an artefact."),
        ("B_cleanrank_scored.json", "Commit-ranking ablation", ""),
        ("B_cond_scored.json", "Family-pattern conditioning", "")]:
        cells = load(f)
        if not cells:
            continue
        doc.append(table(
            sorted(cells, key=lambda c: str(c.get("tag") or c.get("selector"))),
            lambda c: (f"{c.get('tag', c.get('selector'))}"
                       f"{'' if c.get('temp') in (None, 1.0) else ' t=' + str(c['temp'])}"
                       f"{'' if c.get('fam_temp') in (None, 1.0) else '/' + str(c['fam_temp'])}"
                       f"{'' if not c.get('T') else ' T=' + str(c['T'])}"),
            title, note))
        if any((c.get("tag") == "flat") for c in cells):
            doc.append(contrast_table(
                cells, "flat", lambda c: str(c.get("T")),
                title + ": contrasts against flat",
                lambda c: c.get("tag") or c.get("selector")))
        cond = [c for c in cells if c.get("kind") == "conditioned"]
        if cond:
            doc.append("\n| arm | family compliance | pLDDT |")
            doc.append("|---|---|---|")
            for c in cond:
                doc.append(f"| {c.get('tag')} | "
                           f"{fmt(c.get('family_compliance'), '%.4f')} | "
                           f"{fmt(c.get('plddt_mean'))} |")
            doc.append("")

    txt = "\n".join(doc)
    open(os.path.join(HERE, "RESULTS.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
