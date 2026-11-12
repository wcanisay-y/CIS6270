# Project 2: testing two ideas for improving masked diffusion on proteins

Feasibility study for the two proposals in
`CIS6270-both-modalities/Proposal_Neha.docx` (contact-guided parallel
unmasking) and `Proposal_Yan-2.docx` (a two-level chemical alphabet).

**Start here:** `REPORT.md` is the verdict, with a status block at the top
listing what two later correction rounds changed. `REPORT_METHODS.md` is what
was built and how it was tested. `RESULTS.md` is every table, with the
superseded columns flagged at the top and the later rounds appended as
"Round 2" and "Round 3". `REPORT_REVISED.md` covers the revised Idea-A
method; `ANSWERS_TO_FEEDBACK.md` indexes the reply to the professor's
feedback of 2026-11-10.

## Layout

| path | what |
|---|---|
| `common/` | alphabet and chemical partitions, SwissProt loading, metrics, scoring driver |
| `parallel/` | Idea A: unmasking-set selectors, contact providers, samplers, oracle |
| `hier/` | Idea B: flat and two-level masked diffusion models, training, samplers |
| `analysis/` | result tables, paired contrasts, figures |
| `run/` | the sbatch scripts that produced everything |
| `results/` | generated sequences, scored metrics, and the per-finding notes |
| `figures/` | the figures, PNG and PDF |
| `ckpt/` | trained Idea-B checkpoints |
| `data/` | SwissProt as plain text (see `tools/cache_swissprot.py`) |

## Environment

`venv2/` holds torch 2.7.1 and transformers 4.57.6. The newer torch is
required: transformers refuses `.bin` checkpoints under torch below 2.6, and
DPLM ships only `.bin`. numpy stays below 2, so the corpus is cached as
plain text rather than read from parquet.

## Reproducing

Every result came from a Slurm job in `run/`, in this order:

```bash
sbatch run/download.sh        # model weights
sbatch run/smoke.sh           # end-to-end feasibility gate on one GPU
sbatch run/train_all.sh       # Idea B arms, one per array task
sbatch run/ideaA_main.sh      # Idea A matched-forward-pass grid
sbatch run/ideaA_oracle.sh    # the decisive oracle test
sbatch run/ideaA_reparam.sh   # the same rule inside DPLM's own decoder
sbatch run/scorecheck.sh      # the commit-ranking ablation
sbatch run/ideaB.sh           # Idea B generation and scoring
./venv2/bin/python -m analysis.final     # writes RESULTS.md
./venv2/bin/python -m analysis.figures   # writes figures/fig1-fig6
./venv2/bin/python -m analysis.figures_cond   # writes figures/fig7-fig8
```

## Figures

| figure | what it shows |
|---|---|
| `fig1_contact_quality` | SUPERSEDED (atom14 bug, 2026-11-06 numbers): contact precision against mask rate |
| `fig2_plddt_vs_nfe` | pLDDT against forward passes, all Idea-A selectors |
| `fig3_contrasts` | the Idea-A paired contrasts against confidence |
| `fig4_ideaB`, `fig6_ideaB` | Idea B UNCONDITIONAL: the quality-diversity frontier, and two-level against flat |
| `fig5_oracle` | SUPERSEDED (ad-hoc chain set): the oracle contact-map test |
| **`fig7_conditioning`** | **the positive: chemical minus random before and after the homology purge, on both datasets, and what moves in absolute pLDDT** |
| **`fig8_pattern_budget`** | **the accuracy budget: the gain is gone by 19 % pattern error, against 64-78 % achieved** |

`fig1` and `fig5` predate the 2026-11-09 metric fixes and are kept only so no
analysis silently reuses them; every pTM- or foldability-derived value in
`fig1`-`fig6` predates the batch-maximum fix. `fig7` and `fig8` are drawn from
the summary JSONs the analysis scripts write, so they cannot disagree with the
notes beside them.

## Findings index

| note | what it settles |
|---|---|
| `results/A_fidelity_note.md` | the harness reproduces published DPLM numbers, and the absorbing framing both proposals assume is 30 pLDDT below DPLM's real decoder |
| `results/A_contact_quality_summary.md` | SUPERSEDED (atom14 bug): ESM-2 contact prediction is at its information-free floor by 25 % masking. True on masked input, but the collapse is repairable — see `F_argmax_*.json` and RESULTS.md Round 3 |
| `results/A_oracle_note.md` | SUPERSEDED (chain set not fold-non-redundant): a correct contact map is worth +0.12 pLDDT. On CATH 4.4 S40 it is +1.10 [-2.67, +4.82] n.s., which still does not beat free confidence ordering |
| `results/A_matched_nfe_note.md` | at matched compute the rule matches its own shuffled control and loses to the baseline |
| `results/A_ranking_note.md` | the commit-ranking trap: a 26-point "gain" that is pure mode collapse |
| `results/B_main_note.md` | the two-level arms lose to flat, and chemistry does not beat a random partition |
| `results/B_premise_note.md` | the periodicity deficit the proposal targets does not exist |
| `results/B_conditioning_note.md` | the one positive, and the correction block showing its pattern sources were homology-contaminated |
| `results/G_clean_conditioning_note.md` | the positive re-measured on homology-clean patterns: +20.22 [+16.37, +24.14] against the bit-matched random control, and the only arm with nonzero foldability |
| `results/G_beta_effect_note.md` | the fold-class positive, WITHDRAWN: the mechanism it claimed is absent when tested against per-chain contact density |
| `results/I_pattern_curve_note.md` | the accuracy budget: -1.15 pLDDT per 1 % pattern error against -0.04 for a random partition, the family head pinned at its majority-class floor, and the model's own pattern worth +0.29 (`fig8`) |
| `results/J_clustered_split_note.md` | the gain survives deleting every BLAST relative of every pattern source: +23.38 [+20.07, +26.68], memorisation worth +2.71 (`fig7`) |
| `results/K_cath_dataset_note.md` | it replicates on CATH S35: +19.85 [+16.62, +23.05], memorisation +1.21 n.s., and the random control goes negative on compact domains (`fig7`) |
| `results/EXTERNAL_BASELINES.md` | published numbers and prior art, with provenance |

Two implementation bugs invalidated whole columns of the first study and are
recorded in `REPORT_REVISED.md` section 5 and the top of `RESULTS.md`:
ESMFold's `positions` is atom14 (`N, CA, C, O, CB`), so taking index 3 as CB
made every contact map an O-O map; and HuggingFace ESMFold's `ptm` is a batch
maximum, so every pTM and foldability number predating the fix is invalid.
pLDDT was unaffected by both.

## Caveats that apply throughout

48 sequences per cell at one primary length, against 40 to 100 across
several lengths in published protocols. Within-grid contrasts share every
setting and carry the conclusions; absolute values compare only loosely with
published tables. ESMFold's trunk is ESM-2, so steering generation with an
ESM-2 contact head and scoring with ESMFold shares representation bias;
removing that needs a non-ESM folding model. pLDDT is reported with
repetition and diversity beside it in every table, because on its own it is
reachable by degenerate sequences.

**Power.** Per-sequence pLDDT sd is 17.8, so a 48-sequence cell resolves
only effects above about 7 pLDDT, 192 pooled resolves 3.6, and the largest
published gain for unmasking-order selection on this model class is 1.65,
needing roughly 913 sequences per arm. The negatives here rule out the large
gains both proposals claimed and cannot exclude a 1-2 point effect. A
12-cell sweep reports a +7 maximum under a true null 27 % of the time, so a
sweep's best cell is never quoted as a result; three findings in this project
died when held to that rule.

**Asymmetry worth stating.** The negatives are tested across two structural
sets, two decoder families, five budgets and three map sources. The one
positive, chemical-pattern conditioning, rests on one corpus (SwissProt),
one model pair, one length, one budget and one structure predictor, and the
only quality metric not read off that predictor moves very little. See the
scope section in `REPORT.md`.
