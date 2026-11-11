# Answers to Pranam's feedback

Group 4 (Yan Pan, Neha Peddinti, Anushka Sheoran), CIS 6270 Project 2.
Feedback of 2026-11-10, answered point by point. Detail lives in
`REPORT.md` and `REPORT_REVISED.md`; this file is the index.

**Read first.** The feedback is written against the outline, so it proposes
experiments that would strengthen the contact-gating method. Most are now
run, and the method does not work: a deposited X-ray contact map handed
straight to the selector does not beat plain confidence ordering, and inside
DPLM's own remasking decoder the rule is strongly harmful. Several asks are
therefore answered as "we ran this, here is the negative result."

One ask changed our minds, in your favour — see §4.

## Status

| # | Ask | Status |
|---|---|---|
| 1 | Contacts don't bound total correlation; Eq. 6 needs assumptions | Measured directly; the link is weaker than assumed (§1) |
| 2 | Define the kernel; adaptive reveal sets may break MDLM dynamics | Your concern confirmed by measurement (§2) |
| 3 | Equations and algorithm consistent | Done; one real inconsistency found (§3) |
| 4 | Contact accuracy on actual argmax completions | **Run. It works, and it overturns a claim of ours** (§4) |
| 5 | Benchmark confidence / spacing / PUNT / attention-graph | Run. Nothing beats confidence (§5) |
| 6 | Match reveal-set sizes | Done throughout: every arm is matched-forward-pass |
| 7 | Count the full ESM-2 forward cost | Done: **7.88x** a denoiser pass (§5) |
| 8 | Shuffled-contact and independent-folding checks | Shuffled: everywhere. Independent folding: not possible here (§6) |
| 9 | Gate-off ablation retains forced neighbour edges | Doesn't apply to our implementation — proved (§7) |
| 10 | Staleness trigger vs revealed-fraction refresh, equal budget | Done: the trigger loses to a fixed interval (§8) |
| 11 | Quality, diversity and latency together | Done (§5) |

## 1. Contacts versus conditional dependence

We measured the model's own interventional dependency,
`D[i,j] = TV(p(x_j|ctx), p(x_j|ctx, x_i:=v))`, and asked how well a true
contact predicts it. Deposited maps, 48 chains:

| revealed | AUROC(dependency \| true contact) | AUROC(sequence distance \| high dependency) |
|---|---|---|
| 10 % | 0.545 | 0.747 |
| 25 % | 0.543 | 0.705 |
| 50 % | 0.671 | 0.803 |
| 75 % | 0.728 | 0.854 |

CATH S40 agrees (0.536 / 0.543 / 0.611 / 0.650). A true contact barely
separates dependent pairs early in sampling — exactly where the gate fires —
and **plain sequence distance predicts the model's coupling better at every
masking level**. That also explains why the trivial `distance` control beat
the contact rule in the original grid.

## 2. The contact-gated transition kernel

Confirmed, quantitatively. In the remasking decoder the harm is linear in
how often the gate fires:

    harm = -21.02 x (fraction of steps the rule fires) + 1.99,  r^2 = 0.991

Extrapolated to a rule that never fires, the effect is +1.99 pLDDT — zero
within noise. A warm-up does not repair the rule, it titrates the dose. That
is what "adaptive reveal sets do not preserve the reverse dynamics" looks
like when measured. We have not written the kernel out formally; we agree
the paper needs it stated as an explicit approximation.

## 3. Consistency between equations and algorithm

One real inconsistency. `tau` is written as a continuous knob, but on a
**binary** oracle map it has only three regimes: `tau<=0` rejects
everything, `0<tau<=1` rejects exactly the true contacts, `tau>1` rejects
nothing. Measured displacement of the committed set:

| tau | 0.0 | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 | 1.001 |
|---|---|---|---|---|---|---|---|
| fraction displaced | 1.000 | 0.140 | 0.147 | 0.135 | 0.111 | 0.133 | 0.000 |

Flat, then a cliff. Tuning `tau` on an oracle arm is meaningless, and the
gate displaces about 13 % of the committed set whenever it binds. On a
predicted map the knob is real, which is why we set it as a quantile of each
sequence's own coupling values.

## 4. Contact accuracy on the argmax completions

**The most valuable item in the feedback. It works, and it overturns one of
our stated findings.**

Filling every mask with the denoiser's argmax hands the frozen head a
complete sequence instead of an out-of-distribution masked one.
precision@L against deposited maps, 48-chain set (CATH S40 agrees):

| revealed | recovery | ESM-2 masked | **ESM-2 on completion** | DPLM self-attention |
|---|---|---|---|---|
| 1.00 | 1.000 | 0.747 | 0.747 | 0.642 |
| 0.75 | 0.447 | 0.067 | **0.737** | 0.623 |
| 0.50 | 0.364 | 0.051 | **0.621** | 0.525 |
| 0.25 | 0.175 | 0.059 | 0.193 | 0.168 |
| 0.10 | 0.098 | 0.054 | 0.068 | 0.070 |

(all-mask null 0.069.)

- The collapse was an **input-representation artifact**, not a property of
  external planners. Our design rule "read the signal off the generator" is
  wrong as stated; the correct rule is "do not hand a masked sequence to a
  model trained at a 15 % mask rate."
- With the completion the external head **beats** the free self signal at
  50–75 % revealed.
- It dies below 25 % revealed, where recovery (0.098) equals the score for
  ignoring context and always writing leucine (0.095).
- Striking: at 75 % revealed more than half the filled residues are wrong,
  yet precision reaches 0.737. The head needs a *plausible* sequence, not
  the right one.

### Does the repaired map make the gate work? No.

Four configurations, 48 sequences per cell. `contact - shuffled` is the
contrast that isolates structural information.

| configuration | contact − confidence | contact − shuffled | gate cost |
|---|---|---|---|
| absorbing, no warm-up (96/arm) | −4.41 [−9.40, +0.60] | — | 8.9x |
| remasking, no warm-up | **−16.76 [−20.87, −12.62]** | −3.39 [−7.52, +0.76] | 6.9x |
| remasking, warm-up 0.5 | −3.41 [−8.00, +1.17] | +1.41 [−3.22, +6.10] | 4.0x |
| absorbing, warm-up 0.5 | −2.86 [−7.17, +1.40] | +0.10 [−4.13, +4.29] | 8.9x |

And directly comparing signal quality at fixed everything else:

| contrast | effect |
|---|---|
| gating on the **repaired** map vs the **plain masked** map | **−0.60 [−5.43, +4.34]** |

**A tenfold better contact map changes the generated sequences by nothing.**
No configuration beats confidence; the best one reaches parity at 4x the
cost, and its structural content contributes +1.41, indistinguishable from a
permuted matrix. The warm-up again acts as a dose control — halving the
contact calls moved the effect from −16.76 to −3.41 — which is the
firing-fraction signature of §2, not a fix.

Why accuracy does not convert: the oracle arms already hand the selector a
**deposited X-ray map**, the ceiling no predictor can exceed, and it still
loses (−4.19 [−7.72, −0.65] on the ad-hoc set; +1.10 n.s. on CATH). What
costs is that a pairwise rule displaces a per-position confidence ordering
already within 1.65 pLDDT of a *trained* optimum (arXiv:2606.00295).
Accuracy only makes the displacement better targeted.

Diversity 0.930–0.941 and max repeat 2.8–12.6 across all arms, so none of
this is mode collapse.

**What we keep.** This becomes the paper's claim 1, restated: P2-Bert's
unexplained published failure (70.80 pLDDT, 8 below no planner at all) is
explained by input representation, not by the planner being external. That
is a better result than the one it replaces, and it came from this feedback.

## 5. The control ladder, with cost and diversity

One frozen backbone (DPLM-150M), length 200, budgets 12/25/50/100/200,
48 sequences per cell, matched reveal-set sizes, free self-attention signal.
Pooled, 240 sequences per arm:

| selector | pLDDT | relative cost | vs confidence |
|---|---|---|---|
| **confidence** | **55.66** | 1.00x | — |
| contact (attention graph) | 55.53 | 1.04x | −0.13 [−3.44, +3.30] |
| distance (spacing, d=8) | 53.99 | 1.00x | −1.67 [−4.92, +1.64] |
| contact_shuffled | 53.64 | 1.04x | −2.02 [−5.31, +1.27] |
| PUNT | 52.35 | **2.08x** | −3.31 [−6.62, +0.03] |
| random | 50.53 | 1.00x | **−5.13 [−8.37, −1.92]** |

Nothing beats confidence. `random` is the only arm significantly different
from it, which confirms the protocol can see a five-point effect — so the
nulls are nulls, not an insensitive measurement. PUNT is the most expensive
arm and among the worst; it had previously been scored only at the two
smallest budgets.

Cost in denoiser-pass equivalents, measured on the run GPU (ask 7): denoiser
1.00, denoiser with attentions 1.04, **ESM-2 650M contact call 7.88**.
Diversity 0.932–0.937 and novelty 0.997–1.000 for every arm, so no arm buys
quality by collapsing. Per-budget table in `analysis/prof_tables.py`.

## 6. Shuffled-contact and independent folding

Shuffled controls are retained everywhere — same matrix under a symmetric
permutation, so value distribution and spectrum are preserved and only the
association with position is destroyed. This control carries most of our
conclusions, because the contact rule frequently loses to it.

Independent folding we cannot run: ESMFold v1 is the only predictor
installed and its trunk *is* ESM-2. Two things limit the damage. Per-sequence
pTM correlates 0.909 with pLDDT, so it is a restatement rather than an
independent axis. More importantly, the decisive negatives use **deposited
X-ray maps** as the steering signal, which are independent of ESM-2
entirely. A genuine check would need OmegaFold or a Boltz-class model; none
is on disk, and we declare this rather than claim a check we did not make.

## 7. Gate-off ablation

Your warning holds for the proposal as written but not for this
implementation, and here is the proof. The distance rule is a separate
selector that builds its own band matrix; it is never added into the contact
coupling. 200 trials each (`tools/gate_off_check.py`):

| check | result |
|---|---|
| contact at `tau > max(coupling)` vs confidence, random matrices | 0/200 differ |
| same, deposited maps | 0/200 differ |
| `distance` (d=8) vs confidence | **200/200 differ** |

The third row shows the first two are not vacuous.

## 8. Staleness trigger versus revealed-fraction refresh

At equal contact-call budgets. The distribution-change trigger has almost no
dynamic range — mean TV change 0.048–0.090 across 28 arms, never reaching
the 0.15 threshold, causing 36 of 422 refreshes (9 %).

| refresh policy | refreshes | cost | pLDDT |
|---|---|---|---|
| fixed interval, age 8, ignores the distribution | 5.0 | 89.4 | **57.23** |
| trigger-based, default thresholds | 5.0 | 89.4 | 56.99 |
| every step | 13.0 | 152.5 | 56.79 |
| lazy | 5.0 | 89.4 | 50.64 |

At identical cost a fixed interval beats the trigger. The spread is inside
noise, but the ask was whether the trigger earns its complexity: it does not.

## Three corrections this round produced

1. **The threshold is not a continuous knob on binary maps** (§3).
2. **The fold-class result is withdrawn.** `REPORT_REVISED.md` §6.3 reported
   a beta-minus-alpha effect of +11.61 with a corroborating r = +0.42
   against contact density. That correlation was computed against
   **class-mean** density, a four-valued relabelling of the class; against
   each chain's own density r = +0.120 [−0.122, +0.356], p = 0.44. Splitting
   at median per-chain density gives −0.05 [−6.80, +6.71]. See
   `results/G_beta_effect_note.md`.
3. **The conditioning pattern set was homology-contaminated, and is
   re-measured.** The SwissProt dev split is not homology-clustered: 17 of
   48 sources shared more than half their 10-mers with training, and the
   chemical arm gained +10.2 pLDDT on exactly those while the random arm
   gained nothing. Rebuilt on 96 patterns with max containment 0.026, the
   chemical-minus-random contrast is **+20.22 [+16.37, +24.14]** against the
   previously quoted +26.58. See `results/G_clean_conditioning_note.md`.

## What we would ask back

The contact-gated method is refuted across two decoder families, three map
sources, four intervention strengths, a ground-truth oracle, and now a
repaired predictor. We think the publishable object is a short
negative-result and diagnostics paper:

1. why external planners fail on masked protein input, and the one-line
   repair — which explains P2-Bert's published failure;
2. the independence assumption in parallel masked-diffusion decoding is
   empirically safe, tested with ground-truth maps;
3. the firing-fraction diagnostic (r² = 0.991) for distinguishing "my
   schedule fixed the mechanism" from "my schedule disabled it";
4. the power critique: per-sequence pLDDT sd is 17.8, so a 48-sequence cell
   resolves only effects above ~7 pLDDT while the largest published gain for
   unmasking-order selection on this model is 1.65 and needs ~913 per arm.

Is that an acceptable Project 2, or would you rather we spend the remaining
time on the conditioning interface? That is the one arm with a large effect
surviving both its own control and a homology-clean rebuild: giving the
model a real protein's six-class chemical pattern and letting it fill
identities is worth **+19.9 pLDDT**, a size- and bit-matched random
partition is worth **+1.0**, and the conditioned arm is the only one of four
that produces any foldable sequence at all (0.083 vs 0.000). What it does
not yet show is that a *hand-written* pattern works — that is the experiment
we would run next.

## Appendix: what was run for this reply

| job | what | outputs |
|---|---|---|
| 27369294 | argmax-completion accuracy on two structural sets; full control ladder | `F_argmax_{cath,pdb}.json`, `F_ladder_scored.json` |
| 27369473 | gating on the repaired vs plain external map, absorbing decoder | `F_{argmax,plainext}_gate_scored.json` |
| 27379873 | gating on the repaired map in the remasking decoder, and with warm-up 0.5 | `H_argmax_*_scored.json` |
| 27369295 | conditioning on the homology-clean pattern set | `G_cond_clean_scored.json` |
| CPU | gate-off ablation; clean-pattern selection; fold-effect re-analysis | `tools/gate_off_check.py`, `G_clean_patterns.json`, `G_beta_effect_note.md` |

New code: `parallel/argmax_contacts.py`, `tools/gate_off_check.py`,
`tools/clean_patterns.py`, `analysis/{beta_effect,clean_cond,prof_tables,argmax_gate}.py`,
and the `external_argmax` contact mode in `parallel/sample.py`.
