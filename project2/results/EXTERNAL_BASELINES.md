# Published numbers we compare against

Collected from a literature sweep on 2026-11-06. Every row is quoted, not
reproduced by us, except where this project's own runs are marked. Mixing
folding models across tables is invalid: ESMFold, OmegaFold and AlphaFold2
put the random-sequence floor at roughly 22-25, 40 and 44 respectively, so
only rows sharing a folding model are comparable.

## A. Unconditional sequence generation, ESMFold pLDDT, by length

DPLM official repository (github.com/bytedance/dplm), mean pLDDT:

| model | L=100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| DPLM-150M | 73.31 | 84.30 | 84.82 | 86.90 | 81.71 |
| DPLM-650M | 74.00 | 85.61 | 85.91 | 88.16 | 82.58 |
| DPLM-3B   | 77.78 | 86.16 | 87.39 | 90.06 | 87.43 |

DPLM-150M at L=200 is the single number our harness must reproduce. The
DPLM paper's own appendix table disagrees with this repository table
(70.66 vs 73.31 at L=100); the discrepancy is unexplained in the sources.

## B. Decoding-order comparison on DPLM-150M, ESMFold, L=100-500

From the adaptive-order-policy preprint (arXiv:2606.00295), mean pLDDT over
3 seeds. This is the closest published equivalent of our Idea-A grid and
the spread it reports is the calibration target for our implementation.

| unmasking rule | pLDDT |
|---|---|
| random order | 40.04 +/- 0.78 |
| lowest entropy | 55.88 +/- 0.93 |
| highest confidence | 82.20 +/- 0.76 |
| top-1 minus top-2 margin | 82.02 +/- 0.31 |
| learned order policy | 83.85 +/- 0.80 |

Planner work on the same protocol (path-planning preprint arXiv:2502.03540
and PAPL arXiv:2509.23405), L=200-800, 100 sequences per length:

| method | pLDDT | pTM | pAE | foldability % |
|---|---|---|---|---|
| EvoDiff | 31.84 | 0.21 | 24.76 | 0.43 |
| ESM3 | 34.13 | 0.23 | 24.65 | 1.50 |
| ProGen2-medium | 57.94 | 0.38 | 20.81 | 12.75 |
| DPLM-150M | 80.23 | 0.65 | 12.07 | 48.14 |
| DPLM-650M | 79.53 | 0.66 | 11.85 | 49.14 |
| planned decoding, trained planner | 81.48 | 0.72 | 8.97 | 59.40 |
| learned order policy | 86.43 | 0.76 | 9.68 | 54.14 |

Foldability is pLDDT > 80 AND pTM > 0.7 AND pAE < 10. Reporting it matters
because pLDDT alone is high for a confidently predicted single long helix,
and because ESMFold pLDDT above 90 is reachable by optimising against
ESMFold itself.

## C. Parallel-unmasking methods already published

| method | pairwise signal | extra forward passes | venue |
|---|---|---|---|
| MaskGIT confidence top-k | none | 0 | CVPR 2022 |
| entropy-bounded sampler | none | 0 | NeurIPS 2025 |
| DAPD | symmetrised attention, last 30 % of layers | 0 | ICML 2026 |
| DAWN | attention + sink filter | 0 | preprint |
| DEMASK | learned pairwise total-variation head | 0 | preprint |
| PUNT | KL conditional-independence test | ceil(log2 m) | ICLR 2026 |
| path planning, BERT planner | external per-position planner | 1 | preprint |

DAPD's admission rule is the rule Idea A proposes: sort by confidence, admit
a position only if its pairwise score against every already-admitted
position is below a threshold. DAPD takes that score from attention the
forward pass already produced, at zero extra cost.

PUNT's own protein experiment (its appendix) reports mean pLDDT 21-26 for
every sampler it tested, on a membrane-protein model at length 1024. The
ESMFold random-sequence floor is 22-25, so that experiment has no dynamic
range and cannot separate samplers. PUNT also states its advantage is in
denoising steps rather than forward passes, which is precisely what a
matched-NFE comparison removes.

## D. Structure already used to order protein generation

| work | signal | what it orders | venue |
|---|---|---|---|
| Anand et al. | true contact graph, greedy coloring | parallel blocks of non-interacting residues | Nat Commun 13:746 (2022) |
| ESMDiff | ESM3 secondary-structure class | per-position order, beats entropy | ICLR 2025 |
| contact decoding order | true Ca-Ca contacts < 8 A | autoregressive groups, contacting first | bioRxiv Oct 2025 |
| fast non-AR inverse folding | evolutionary couplings | attempted, REPORTED AS FAILED | MLSB@NeurIPS 2023 |

The 2022 result is the closest prior art to Idea A's mechanism: a contact
graph, greedy coloring into independent sets, and parallel sampling within
each set. It differs in using a known fixed backbone rather than a
prediction from the partial state, it is motivated by speed alone, and it
reports no blocked-versus-sequential quality comparison.

The polarity is disputed in this literature. The inverse-folding work
decodes CONTACTING residues together or first; Idea A decodes mutually
NON-contacting residues together. We run both directions.

## E. From-scratch 35M models on SwissProt, ESMFold pLDDT

From the DiMA paper (ICML 2025), which trained every arm from scratch on
SwissProt with sequences 128-254. This is the protocol our Idea-B arms
follow, so these are the right comparison rows for a 38M model.

| arm | pLDDT | ESM-2 pppl | diversity (clusters at 50 % id) | repetition |
|---|---|---|---|---|
| natural dataset | 80.7 | 5.35 | 1.000 | 0.045 |
| random sequences | 24.8 | 21.53 | 1.000 | 0.000 |
| EvoDiff-OADM | 37.1 | 15.77 | 0.986 | 0.006 |
| D3PM | 36.7 | 16.52 | 0.994 | 0.003 |
| discrete flow matching | 37.8 | 15.25 | 0.996 | 0.004 |
| nanoGPT (autoregressive) | 61.0 | 8.18 | 0.900 | 0.228 |
| DPLM recipe | 84.0 | 3.50 | 0.494 | 0.781 |
| DiMA | 83.3 | 4.68 | 0.611 | 0.320 |

Two things to read from this table. A from-scratch masked diffusion model at
this scale lands near pLDDT 37, well below an autoregressive transformer of
the same size at 61. And the DPLM recipe reaches 84 only with a repetition
score of 0.78 and barely half the sequences in distinct clusters, which is
near mode collapse. The real axis at this scale is quality against
diversity, not quality alone.

## Caveats

- These are quoted numbers. Our own runs differ in sample count (48 per
  cell, against 40-100 in the sources) and in using one length for the main
  grid, so absolute values are comparable only loosely. The within-grid
  contrasts, which share every setting, are what our conclusions rest on.
- The DiMA rows are 35M models retrained from scratch, not released
  checkpoints. The released DPLM-650M scores 81.8 in the same paper's
  checkpoint table, not 84.0.
- ESMFold's trunk is ESM-2, so steering generation with an ESM-2 contact
  head and scoring with ESMFold pLDDT shares representation bias end to
  end. This is a real limitation of the Idea-A evaluation and would need a
  non-ESM folding model to remove.
