# Predicting the family before the identity: the accuracy budget, and why it cannot be met

Asked 2026-11-11: train the family channel in the diffusion model and predict
the family label before the exact amino acid. That architecture already
exists (`ckpt/dms6.pt`, `hier/sample.py:sample_hier_scheduled`) and already
loses end to end — 36.58 pLDDT self-predicting families against 43.09 for a
flat 20-letter model with no family channel, and 56.49 for the same model
handed a real pattern. So the question is quantitative: how accurate must a
predicted pattern be, and how accurate is the one the model produces?

Jobs 27442330 (`run/pattern_curve.sh`), analysis `analysis/pattern_curve.py`,
data `results/I_curve_{dms6,rand6}_scored.json` and `*_diag.json`. 96
homology-clean patterns, length 200, 100 forward passes, 96 sequences per
cell, paired per source protein.

## 1. The budget: the gain is gone by 19 % pattern error

Chemical partition, corrupting a fraction of positions to a *different*
random family so the fraction is the true error rate:

| pattern error | pLDDT | gain over unconditional | loss vs a perfect pattern |
|---|---|---|---|
| 0.00 | 55.41 | **+18.83 [+15.15, +22.55]** | — |
| 0.05 | 49.65 | +13.07 [+9.83, +16.36] | -5.76 [-9.04, -2.48] |
| 0.10 | 43.23 | +6.65 [+3.79, +9.65] | -12.18 [-15.50, -8.99] |
| 0.20 | 35.51 | -1.08 [-2.66, +0.67] | -19.90 [-23.58, -16.28] |
| 0.30 | 32.56 | -4.02 [-4.95, -3.02] | -22.85 |
| 0.50 | 30.91 | -5.67 [-6.32, -4.96] | -24.50 |
| 0.75 | 31.05 | -5.53 [-6.14, -4.89] | -24.36 |
| 1.00 | 30.20 | **-6.38 [-7.00, -5.70]** | -25.21 |

Slope **-1.15 pLDDT per 1 % of pattern error**. 7.8 % error halves the gain,
18.6 % erases it. The curve flattens from f = 0.5 onward: past half wrong the
pattern is already uninformative and the only remaining cost is the binding
constraint itself.

## 2. A WRONG pattern is actively harmful, and that explains Idea B's deficit

A fully wrong pattern costs **-6.38 pLDDT against generating freely**, because
binding makes a family error unrecoverable: a position labelled aromatic can
never afterwards receive a leucine. The two-level arm's self-predicted
families are near-random (section 4), so this is the mechanism behind the
previously unexplained 6.5-point gap between the two-level arm (36.58) and the
flat baseline (43.09). The family channel does not merely fail to help;
constraining identities to near-random families is worse than having no
channel.

## 3. The control: there is no curve for a random partition

Same experiment on `ckpt/rand6.pt`, a random partition with the same group
sizes (490 bits against the chemical partition's 484 on this corpus):

| pattern error | pLDDT | gain over unconditional |
|---|---|---|
| 0.00 | 36.32 | +1.02 [-0.21, +2.33] |
| 0.50 | 34.35 | -0.96 [-2.08, +0.21] |
| 1.00 | 34.91 | -0.39 [-1.61, +0.91] |

Slope **-0.04** pLDDT per 1 % error, against -1.15 for the chemical
partition. A perfect random-partition pattern is worth +1.02 with an interval
containing zero, so there is nothing to degrade.

This is the control that matters. If conditioning worked by narrowing the
output alphabet, a bit-matched random partition would produce the same curve.
It produces a flat line. **The -1.15 slope is chemical information being
destroyed, not constraint strength being relaxed.** It also re-derives the
published chemical-minus-random result from a different direction: at f = 0
the two arms differ by +17.81, against +20.22 measured directly.

The harm in section 2 is chemistry-specific too: a fully wrong chemical
pattern costs -6.38 while a fully wrong random pattern costs -0.39. Forcing a
buried position to be charged is destructive; forcing it into an arbitrary
group of residues is not.

## 4. What the model's own predictor achieves: nothing

`fam_head` on `ckpt/dms6.pt`, teacher-forced, majority-class floor 0.3633:

| identities revealed | family accuracy | top confidence decile |
|---|---|---|
| 0.00 | 0.3634 | 0.405 |
| 0.10 | 0.3634 | 0.416 |
| 0.25 | 0.3634 | 0.423 |
| 0.50 | 0.3633 | 0.416 |
| 0.75 | 0.3635 | 0.404 |

The head predicts at the majority-class rate and **revealing 75 % of the
identities does not help it at all**. Cross-entropy 1.5716 nats against a
family marginal entropy of 1.662: it learned the marginal and 5 % beyond.
Self-generated patterns are 43.2 % aliphatic against 36.3 % in real patterns,
the signature of a near-constant predictor. `rand6` is the same story
(0.2696 against a 0.2651 floor), so this is the training scheme and not the
partition.

Required error <= 0.186 to retain anything, <= 0.078 for half. Achieved error
0.637 (argmax) to 0.782 (sampled from a marginal-only head). **Three to four
times the budget.**

## 5. The direct test, which removes the main caveat

The curve corrupts positions uniformly, while a real predictor errs in a
correlated way that might fall on positions that matter less. The
`selfpattern` arm settles it: the model generates a family pattern on its own,
then that pattern goes through the *identical* conditioning path a real
pattern uses, so only pattern quality differs.

| pattern | pLDDT | gain over unconditional |
|---|---|---|
| perfect | 55.41 | +18.83 [+15.15, +22.55] |
| **the model's own** | **36.87** | **+0.29 [-1.09, +1.78]** |
| none | 36.58 | — |
| uniform corruption at 0.75 | 31.05 | -5.53 |

The model's own pattern retains **2 %** of the gain, interval spanning zero.

The caveat was real but does not save the idea: at a matched ~78 % error rate
the self-pattern scores 36.87 against 31.05 for uniform corruption, so the
model's errors are indeed ~6 pLDDT more benign than uniform ones — it
produces a plausible-but-wrong pattern rather than a chemically absurd one.
The idea fails with that benefit already included.

## Verdict

Predicting the family before the identity cannot work at this scale, and the
reason is not the architecture, the sampler or the training length. Staying
inside the budget means >= 92 % accuracy on a 6-way chemical class from
sequence context alone with no structure. Structure-conditioned inverse
folding reaches roughly 35-50 % exact residue recovery *with* the backbone
given. A predictor good enough to meet the budget would be a stronger protein
model than the one that would consume its output.

The pattern is useful precisely because it is exact and comes from outside.
That is what makes the conditioning result a controllability result, and it is
why it cannot be bootstrapped.

## 6. Is the family head's failure a training artefact? No — settled

The loss weighting IS pathological. With gamma = 2 the family channel is
masked with probability t^2 and weighted 1/t^2: unbiased, since the expected
weighted contribution is L at every t, but the variance grows as 1/t^2 and
reaches 1e6 at the eps = 1e-3 end. Worse, the head's informative regime is
"few families masked, most identities revealed", which is small t, and at
small t almost no position is family-masked. So the head trains mostly at t
near 1, where everything is masked and the correct answer IS the marginal.

`run/family_signal.sh` (job 27442486) trained two corrected variants on the
full corpus, 20,000 steps each, everything else held fixed.

| arm | best val | ce_fam | nats past the marginal | family accuracy | top decile |
|---|---|---|---|---|---|
| original, gamma = 2 | 8.5019 | 1.5716 | 0.090 | **0.3634** | 0.423 |
| gamma = 1.0 | 7.4964 | 1.4299 | 0.232 | **0.363** | 0.457 |
| gamma = 2, weight clipped at 50 | 7.2478 | ~1.43 | ~0.23 | **0.363** | 0.411 |
| majority-class floor | — | 1.662 | 0 | 0.3633 | — |

Both fixes work on the loss -- validation improves by 1.0 to 1.25 nats and
the family head carries 2.5x the information -- and **neither moves argmax
accuracy off the majority-class floor at any reveal level.** The gain went
into sharper probability mass, not into a different mode: aliphatic remains
the most likely family almost everywhere.

So the earlier suspicion was wrong and is withdrawn. The published two-level
negative is NOT a training artefact. The barrier is informational: the coarse
chemical class at a position is not recoverable from sequence context at this
model scale, even when the head is trained properly to try. Two independent
corrections reach the same floor, which is stronger evidence than one.

This also means the verdict in the previous section stands without
qualification. Nothing in reach of a loss fix closes a 3-4x error gap, and
the measured attempt to fix it closed none of it.
