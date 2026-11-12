# The conditioning result measured by instruments that are not ESMFold

Every number carrying the conditioning claim so far came from one ESMFold
forward pass. pLDDT, pTM, pAE and foldability are that pass read four ways --
per-sequence pTM correlates 0.909 with pLDDT -- and the one quality metric
from a different model, ESM-2 pseudo-perplexity, moved 0.20 where pLDDT moved
19.91 and on CATH mildly favoured the *unconditional* arm. A +23 pLDDT effect
resting on a single predictor's self-confidence was the weakest part of this
project, and it is what the professor's "independent folding" ask (§6 of
`ANSWERS_TO_FEEDBACK.md`) was pointing at.

Job 27496304, `run/extra_metrics.sh`. Scripts `common/extra_metrics.py`,
`tools/{extra_metrics_run,max_id}.py`, analysis `analysis/extra_metrics.py`,
data `results/L_{extra,maxid}_{J,K}_{orig,purged}.json`. Figure
`figures/fig9_independent_metrics.{png,pdf}`.

**Nothing was generated and nothing was re-folded.** Every sequence scored
here is the one already stored in the `*_scored.json` the published pLDDT
numbers came from, so this is a different question asked of the same objects.

## The four arms, purged corpora, side by side

SwissProt dev patterns (left of the slash) and CATH S35 patterns (right):

| arm | pLDDT | pppl | NSR | Max-ID % | FPD | MMD |
|---|---|---|---|---|---|---|
| chemical, conditioned | **60.3 / 53.5** | 14.39 / 17.33 | 0.353 / 0.348 | 11.5 / 12.0 | **12.0 / 13.0** | **0.348 / 0.371** |
| random, conditioned | 37.0 / 33.6 | 15.76 / 16.21 | 0.401 / 0.393 | 10.4 / 10.9 | 26.1 / 24.5 | 0.780 / 0.738 |
| chemical, unconditional | 35.8 / 35.4 | 15.71 / 14.91 | 0.068 / 0.066 | 10.5 / 9.6 | 27.7 / 25.6 | 0.775 / 0.715 |
| random, unconditional | 35.3 / 35.5 | 16.73 / 16.56 | 0.068 / 0.067 | 10.7 / 11.3 | 27.3 / 24.6 | 0.818 / 0.744 |
| natural proteins | 90.1 | 3.50 | -- | 42.1 / 77.9 | 3.0 / 4.4 | -0.002 / 0.031 |

## 1. FPD and MMD: the first independent metrics that agree with the claim

Fréchet distance and kernel MMD between the generated and the natural
distribution in ESM-2 embedding space. These score a SET, not a sequence, so
they see something no per-sequence mean can.

| | SwissProt purged | CATH S35 purged |
|---|---|---|
| unconditional baseline | 27.7 | 25.6 |
| **chemical, conditioned** | **12.0** | **13.0** |
| random, conditioned | 26.1 | 24.5 |
| natural against natural (calibration) | 3.0 | 4.4 |
| **fraction of the distance to natural that the chemical arm closes** | **64 %** | **60 %** |
| the same for the random arm | 5 % | 0.3 % |

MMD, which assumes nothing Gaussian, says the same thing: 0.348 against 0.780
on SwissProt, 0.371 against 0.738 on CATH, with the natural-against-natural
calibration at -0.002 and 0.031, i.e. indistinguishable as it must be.

**This is the corroboration the project did not have.** Conditioning on a
real chemical pattern moves the generated distribution roughly 60 % of the way
from the unconditional model to real proteins; the bit-matched random
partition moves it between 0 and 5 %. Both datasets agree, and the instrument
shares no component with ESMFold's structure module.

## 2. NSR: no retrieval -- and NSR cannot tell the arms apart

Native sequence recovery against the very protein whose pattern was used.
Read against the two floors that matter, not against zero: *uniform*, where
the arm is told the family and picks inside it at random, and *modal*, where
it always writes that family's commonest residue in the sources.

| arm | NSR | uniform floor | modal floor | NSR / modal |
|---|---|---|---|---|
| SwissProt chemical | 0.353 [0.346, 0.360] | 0.306 | 0.391 | 0.90 |
| SwissProt random | 0.401 [0.394, 0.408] | 0.308 | 0.441 | 0.91 |
| CATH chemical | 0.348 | 0.307 | 0.389 | 0.89 |
| CATH random | 0.393 | 0.306 | 0.442 | 0.89 |

Two readings, and the second one matters more.

**No retrieval.** Every conditioned arm sits BELOW its own modal floor. A
model copying the source would be far above it; these are at 89-91 % of the
score for ignoring the source entirely and writing each family's commonest
residue. The unconditional arms are at 0.067, their compositional floor. This
is an alignment-free confirmation of what k-mer novelty said.

**NSR does not discriminate chemistry, and the raw numbers mislead.** The
random arm's NSR is HIGHER (0.401 against 0.353) while it folds 23 pLDDT
worse. The explanation is in the floors: a random partition happens to group
residues so that the modal residue is more dominant (modal floor 0.441 against
0.391), so guessing the mode gets further. Normalised by each arm's own floor
the two are identical to within 0.02 in all four cells. **The chemical gain is
not more native residues in the right places.**

## 3. Max-ID: the purge confirmed by a second, unrelated instrument

Highest identity to the corpus each arm was actually trained on, by blastp,
scored over the query so a short perfect patch cannot reach 100 %. k-mer
novelty is a substring statistic and cannot see a diverged copy; an alignment
can.

| arm | original corpus | after the purge |
|---|---|---|
| chemical, conditioned | 32.2 % (max 79.5) | **11.5 %** |
| random, conditioned | 32.0 % (max 85.0) | 10.4 % |
| chemical, unconditional | 10.5 % | 10.5 % |
| natural dev proteins | 83.0 % | 42.1 % |

On the original corpus a conditioned sequence lands a mean 32 % identity to
training, with a tail reaching 79-85 %. **After the purge that falls to 11.5 %
-- the unconditional arm's value -- while the pLDDT gain stays at +24.6.** The
natural control moves 83 → 42 %, which is the independent check that the purge
removed what it was supposed to.

Round 4 established this by retraining and comparing pLDDT. This establishes
it by aligning the generated sequences themselves against the corpus. Two
instruments with nothing in common agree that the conditioning gain is not
memorisation.

Note the conditioned arms reach 32 % on the original corpus where the
unconditional arms reach 10.5 %. That is the leak Round 4 measured as
+2.71 pLDDT, visible here as sequence identity.

## 4. Pseudo-perplexity: retire it as a discriminator

pppl favours the chemical arm on SwissProt (14.39 against 15.76) and the
random arm on CATH (17.33 against 16.21). It contradicts itself across
datasets while FPD, MMD and pLDDT all agree on both. With natural proteins at
3.50 and every generated arm between 14 and 17, the metric is dominated by
"this is not a real protein" and has little resolution left for which
generated arm is better. Quote it as a declared null, not as evidence either
way.

## 5. Ref-TM and Ref-RMSD

Pending, job 27496556 (`run/ref_tm.sh`). See `results/M_reftm_note.md`.

## 6. scTM and scRMSD: NOT RUN, and why

Self-consistency TM/RMSD needs an inverse-folding model to redesign the
predicted structure. Neither is available here: ProteinMPNN is not on disk and
the only PyPI package of that name is an unvetted third-party wheel that ships
its own weights, and ESM-IF1 (`fair-esm`, which did install) requires
`torch_scatter`/`torch_sparse`, which have no wheel for torch 2.7.1+cu126 and
would need compiling against CUDA.

This is a smaller loss than it looks. scTM exists to judge *de novo* designs
that have no reference structure. These sequences do have one -- they were
conditioned on a real domain's pattern -- so Ref-TM asks the stronger question
directly. scTM would still add "is this sequence designable at all"; it is
declared missing rather than approximated.

## What this changes

The conditioning result now rests on two independent axes rather than one.
FPD and MMD, computed in ESM-2 embedding space on both datasets, say the
chemical pattern moves the generated distribution ~60 % of the way to natural
while its bit-matched control moves it ~0 %. Max-ID says the sequences behind
that gain have no closer relationship to training than unconditional samples
do. NSR says the model is not copying, and says so from below the trivial
modal baseline.

What has NOT changed: the generator is SwissProt-trained in every experiment,
there is still one model pair, one length, one budget, and -- until the Ref-TM
run lands -- no measurement of whether any of this corresponds to the source
protein's actual structure.
