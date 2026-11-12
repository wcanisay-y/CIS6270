# The conditioning gain survives a homology purge: memorisation is worth 10 % of it

The question. `G_clean_conditioning_note.md` reports chemical-minus-random
conditioning at +20.22 pLDDT on pattern sources selected by 10-mer
containment. Containment is a substring test, so it controls memorised
fragments and not homology: a source can share a fold with hundreds of
training sequences while sharing no 10-mer. Nothing published so far ruled
out the pattern acting as a retrieval key for a memorised fold.

Jobs 27442437 (BLAST purge), 27442438_[0-2] (retrain), 27442439 (the 2x2).
Scripts `tools/{export_fasta,build_clustered_split}.py`,
`run/{clustered_split,retrain_clustered,cond_clustered}.sh`, analysis
`analysis/clustered.py`, data `results/J_cond_{orig,purged}_scored.json`.
Figure `figures/fig7_conditioning.{png,pdf}`, left panel (the contrast before
and after the purge) and right panel (what moves in absolute pLDDT).

## How much homology was there? Almost total

128 held-out pattern sources, sampled from the dev candidates with **no**
homology criterion -- selecting orphan proteins would have biased the set
towards unusual sequences and would not have tested the published number.

| | |
|---|---|
| sources with a BLAST relative in training at E <= 1e-3 | **125 of 128 = 97.7 %** |
| distinct training sequences that are relatives | 19,645 |
| that as a fraction of the corpus | **11.89 %** |
| corpus after the purge | 145,586 (was 165,231) |
| held-out length mean vs candidate pool | 223.1 vs 225.8 |

So the "homology-clean" set of the earlier note, whose maximum 10-mer
containment was 0.026, was almost certainly ~97 % homologous to training by
alignment. The containment filter was measuring the wrong thing.

## The 2x2, identical patterns in every cell

Retrain is identical to the original in every hyperparameter -- 20,000 steps,
batch 256, same seed and architecture -- and differs only in the corpus.

| corpus | chemical cond. | chemical uncond. | random cond. | random uncond. |
|---|---|---|---|---|
| original | 62.48 | 36.67 | 36.39 | 35.67 |
| purged | 60.34 | 35.77 | 36.96 | 35.31 |

| corpus | chemical - random, paired per source | own gains, chem / rand |
|---|---|---|
| original | **+26.09 [+22.89, +29.28]** | +25.81 / +0.72 |
| purged | **+23.38 [+20.07, +26.68]** | +24.58 / +1.66 |

| difference of differences, paired per source | |
|---|---|
| what training homologs were worth | **+2.71 [+0.95, +4.55]** |

## Reading

**The effect is not memorisation.** Removing every detectable relative of
every pattern source from training costs +2.71 pLDDT of a +26.09 effect --
**10.4 %** -- and the gain survives at +23.38 with an interval far from zero.
The interval on the difference excludes zero, so the leak is real and
measurable; it is not the effect.

**The qualitative finding strengthens.** Foldability for the conditioned
chemical arm is 0.172 on the purged corpus against 0.156 on the original,
while the random conditioned arm is 0.000 against 0.008. The conditioned
chemical arm remains the only arm producing foldable sequences, on a model
that has never seen a relative of any pattern it is given.

**The purge cost about a point of absolute quality, as expected.** Every
unconditional baseline drops (36.67 to 35.77 chemical, 35.67 to 35.31
random) on 11.89 % less data, which is why the conclusion rests on the
within-row contrast rather than on absolute pLDDT across rows.

**The two earlier corrections are not the same thing and are not additive.**
On unfiltered patterns the original model gives +26.09 and the purged model
+23.38, so memorisation is worth 2.71. The 10-mer-clean set gave +20.22 on
the original model -- about 6 below +26.09 -- but since actually deleting the
memorisation pathway costs only 2.71, most of that 6 was not memorisation.
The 10-mer-clean set is simply a harder set of patterns. We did not run
10-mer-clean patterns against the purged model, so the two reductions cannot
be combined into a single number.

## What to quote

For the conditioning interface: **+23.38 [+20.07, +26.68] pLDDT over a
bit-matched random partition, on 128 pattern sources none of whose BLAST
relatives were in training.** Memorised homologs account for
+2.71 [+0.95, +4.55] of the unpurged figure.

## Scope, declared

- **Depth 1 only.** 19,645 round-1 relatives exceeded the 8,000-sequence
  budget for a second BLAST round, so relatives-of-relatives remain in
  training. A single-linkage depth-2 purge would be stronger and would
  likely have exceeded the 30 % corpus guard.
- **BLAST, not a clustering.** MMseqs2 and CD-HIT are not installed and
  running a downloaded MMseqs2 binary was not permitted, so the
  neighbourhood is a single-linkage blastp neighbourhood at E <= 1e-3, not
  `mmseqs easy-cluster`. Fold-level homologs undetectable by BLAST remain. A
  structure-based purge with Foldseek would be the stronger instrument.
- One corpus, one model pair, one length, one budget, one structure
  predictor -- unchanged from the earlier note, and still the asymmetry
  against the far better-cornered negatives in this project.
