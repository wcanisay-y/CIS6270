# Are these two ideas likely to work?

Short answer: **no, not as written.** Both were tested with small demo runs
on standard data against published baselines, and both fail at the point
that decides them. Idea A fails a control that no amount of better
engineering can fix: a deposited X-ray contact map, which is the ceiling for
any contact predictor, does not beat free confidence ordering. Idea B fails
the one comparison that would have given it a contribution over work already
published.

The verdicts rest on controls built to catch measurement artefacts, and that
machinery has since caught three artefacts in our *own* positive findings —
a sweep's best cell, a redundant chain set and a fold-class split — each of
which is recorded below as withdrawn. Two implementation bugs were also
found and fixed, invalidating the first study's contact-accuracy and pTM
columns. See the status block that follows.

Everything below is reproducible from `project2/`. Methods are in
`REPORT_METHODS.md`, every number in `RESULTS.md`, the published
comparisons in `results/EXTERNAL_BASELINES.md`.

---

## Status after two correction rounds, 2026-11-11

This document was written on 2026-11-06. Two later rounds changed numbers in
it, and the changes are folded into the sections below. Read this block first.

| round | what changed |
|---|---|
| 2026-11-09, metric and dataset audit | Two bugs invalidated whole columns. The CB-CB contact maps were actually O-O maps (`positions` is atom14, so index 3 is the backbone oxygen), which invalidated every contact-accuracy number in the first study; and HF ESMFold's `ptm` is a batch maximum, which invalidated every pTM and foldability number. pLDDT was never affected. The ad-hoc 48-chain oracle set was also replaced with CATH 4.4 S40, one chain per topology. |
| 2026-11-10, answers to the professor's feedback | The contact-accuracy collapse turned out to be an input-representation artefact that is repairable, the repaired map still does not make the gate work, the fold-class positive was withdrawn, and the conditioning headline was re-measured on a homology-clean pattern set. Index in `ANSWERS_TO_FEEDBACK.md`. |

Three numbers from the 2026-11-06 text should never be quoted again:

| retired | replacement |
|---|---|
| contact precision@L 0.305 unmasked, 0.027 at 25 % masked | 0.747 unmasked, and the collapse is repairable (below) |
| oracle mean effect +0.12 pLDDT on the ad-hoc 48-chain set | +1.10 [-2.67, +4.82] n.s. on CATH S40 |
| conditioning worth +26.4 pLDDT | **+23.4 [+20.1, +26.7] on a homology-purged corpus** (below) |

**Round 4, 2026-11-11.** Two further experiments, both on the conditioning
result, both reported in full in sections at the end of Idea B:

| question | answer |
|---|---|
| Does the conditioning gain survive deleting the pattern sources' homologs from training? | **Yes.** +23.38 [+20.07, +26.68] after purging, against +26.09 unpurged. Memorised homologs are worth +2.71 [+0.95, +4.55], about 10 %. 125 of 128 sources had a BLAST relative in training, so the earlier "homology-clean" set was clean only of shared substrings. |
| Does it replicate on a second, classical dataset? | **Yes.** CATH S35 domains, same 2×2: **+19.85 [+16.62, +23.05]** purged, overlapping the SwissProt interval. Memorisation there is +1.21 [-0.68, +3.06], smaller as the lower homology (42 % of sources vs 98 %) predicts. |
| Can the model predict the family pattern itself, instead of being given one? | **No.** A perfect pattern is worth +18.83 pLDDT and the model's own is worth +0.29. The error budget is 8 % for half the gain; the family head's error is 64-78 %. Two corrected training variants improve the loss by 1.0-1.25 nats and leave accuracy pinned at the majority-class floor. |

**Round 6, 2026-11-12.** The asymmetry Round 5 declared -- one ESMFold pass
read four ways -- is addressed by re-scoring the SAME stored sequences on
instruments that share nothing with it (`results/L_extra_metrics_note.md`,
`figures/fig9_independent_metrics`):

| metric | what it says |
|---|---|
| **FPD / MMD** in ESM-2 embedding space | **The corroboration that was missing.** The chemical pattern closes 64 % (SwissProt) and 60 % (CATH) of the distance from the unconditional model to natural proteins; the bit-matched random control closes 5 % and 0.3 %. Natural against natural calibrates at 3.0 and -0.002. |
| **Max-ID** (blastp, vs the corpus each arm trained on) | Round 4's purge confirmed by alignment rather than by retraining: a conditioned sequence sits at 32 % identity to the original corpus (tail to 85 %) and at 11.5 %, the unconditional value, once the relatives are deleted -- while keeping +24.6 pLDDT. |
| **NSR** against the source, read against its own floors | No retrieval: every conditioned arm is BELOW its modal-in-family floor. And NSR does not discriminate -- the random arm's higher raw NSR (0.401 vs 0.353) is entirely its higher modal floor; normalised, the two match to within 0.02 in all four cells. |
| **pppl** | Retired as a discriminator: it favours chemical on SwissProt and random on CATH. |
| **scTM / scRMSD** | Not run -- no inverse-folding model is available. Declared, not approximated. |

Figures for this round, added 2026-11-12 (`analysis/figures_cond.py`):
`figures/fig7_conditioning` is the contrast with its control before and after
the purge on both datasets, and `figures/fig8_pattern_budget` is the accuracy
budget against what the family head achieves. Nothing in `fig1`-`fig6` plots
conditioning, and `fig1`/`fig5` rest on retired numbers.

The verdict did not change in either round: neither proposal works as
written. What changed is which evidence carries the verdict.

---

## The harness is trustworthy

Before anything else, the setup reproduces published numbers. DPLM-150M at
length 200 is published at pLDDT 84.30. Our reimplementation of its decoder
returns 83.60 at 200 forward passes and 87.44 at its default 500. Natural
SwissProt proteins of the same length score 90.10. Our from-scratch 38M flat
baseline scores 43.09, just above the published 36-38 range for
from-scratch discrete diffusion at that scale on that corpus.

---

## Idea A: choosing the parallel unmasking set by contacts

### It is not new, and the closest attempt was published as a failure

The admission rule the proposal describes, sort masked positions by
confidence and admit one only if its pairwise coupling to every
already-admitted position stays below a threshold, is the rule of a method
accepted at ICML 2026, which takes the same pairwise signal from attention
the forward pass already produced, at zero extra cost. A 2022 Nature
Communications protein-design paper already builds a contact graph, colours
it greedily into independent sets, and samples each set in parallel. And a
2023 workshop paper from the group behind much of this literature tried
biasing the unmasking order by couplings in a protein masked diffusion
model, reported that it did not work, and listed the proposal's exact idea
as an open problem.

Three separate pieces of prior art is not fatal on its own. The measurements
are.

### The signal collapses where the method needs it, but the collapse is repairable

Nobody had measured how good ESM-2's contact prediction is on a partially
masked sequence. We did. **Both tables below replace the 2026-11-06 version
of this section**, which scored against maps built with the atom14 indexing
bug and so compared predictions with an oxygen-oxygen contact map. These are
scored against deposited X-ray structures, on the ad-hoc 48-chain set;
`results/F_argmax_cath.json` has the fold-non-redundant CATH S40 set, which
agrees on every qualitative point at uniformly lower absolute values
(0.606 unmasked, 0.066 at 75 % revealed, 0.564 repaired).

| revealed | recovery | frozen ESM-2 on the masked input | **ESM-2 on the argmax completion** | generator's own attention |
|---|---|---|---|---|
| 1.00 | 1.000 | 0.747 | 0.747 | 0.642 |
| 0.75 | 0.447 | 0.067 | **0.737** | 0.623 |
| 0.50 | 0.364 | 0.051 | **0.621** | 0.525 |
| 0.25 | 0.175 | 0.059 | 0.193 | 0.168 |
| 0.10 | 0.098 | 0.054 | 0.068 | 0.070 |
| all-mask null | - | 0.069 | - | 0.065 |

Read the second and third columns together. The signal the proposal
specifies, a separate frozen ESM-2 pass on the partially masked sequence, is
at its information-free floor by the time a quarter of the sequence is
missing: 0.067 against a 0.069 null. ESM-2 was trained at a 15 percent mask
rate and has never seen this input.

But that is a property of the *input*, not of external planners. Fill every
mask with the denoiser's argmax first, so the frozen head receives a
complete sequence, and precision returns to 0.737 — and the repaired
external head then **beats** the generator's own attention at 50 to 75
percent revealed. This came out of the professor's feedback and it overturns
the design rule we had drawn from the first measurement. The rule is not
"read the signal off the generator"; it is **"never hand a masked sequence
to a model trained at a 15 percent mask rate."**

Two boundaries on the repair. It dies below 25 percent revealed, where token
recovery (0.098) is no better than ignoring the context and writing leucine
everywhere (0.095). And at 75 percent revealed more than half the filled
residues are still wrong, yet precision reaches 0.737 — the head needs a
*plausible* sequence, not a correct one.

This also explains a published result its own authors left unexplained. P2
(arXiv:2502.03540, ICLR 2025) reports P2-Bert, a separate frozen encoder
used as a planner, at 70.80 pLDDT against 78.79 for no planner at all —
eight points worse — on a 150M MDM at our scale. Input representation
accounts for it.

### Even a correct contact map does not help

This is the result that closes the question. Hand the selector the deposited
X-ray contact map of a real protein and compare it with plain confidence
ordering and with the same map shuffled.

**Dataset correction, 2026-11-09.** The first version of this section used an
ad-hoc 48-chain set that capped pairwise sequence identity at 0.30 but was
not fold-non-redundant: 19 CATH topologies for 48 chains, one fold in 11 of
the 48 entries, and only 2 mainly-alpha chains. It was replaced by CATH 4.4
S40, one chain per topology, class-stratified, 43 maps. **The sign of the
oracle effect depends on which set is used**, so the ad-hoc number is
retired:

| set | true map − confidence | true map − shuffled |
|---|---|---|
| CATH 4.4 S40, 43 topologies (standard) | **+1.10 [−2.67, +4.82] n.s.** | +2.91 [−0.46, +6.31] |
| ad-hoc 48 chains (retired) | −4.19 [−7.72, −0.65] | — |

So the strong claim this section originally made, that an oracle map
*refutes* the mechanism, was an artefact of a redundant chain set. The honest
statement is weaker and still fatal to the proposal: **a deposited X-ray
contact map, the ceiling no contact predictor can exceed, does not beat free
confidence ordering.** Improving contact prediction therefore cannot rescue
the mechanism, which is confirmed directly below. Diversity is 0.93
throughout, so no arm is degenerate.

### And the repaired map confirms it: a tenfold better map changes nothing

With the argmax repair above, the external head goes from 0.067 to 0.737
precision. Feeding that repaired map to the gate, four configurations, 48
sequences per cell:

| configuration | contact − confidence | contact − shuffled | gate cost |
|---|---|---|---|
| absorbing, no warm-up (96/arm) | −4.41 [−9.40, +0.60] | — | 8.9x |
| remasking, no warm-up | **−16.76 [−20.87, −12.62]** | −3.39 [−7.52, +0.76] | 6.9x |
| remasking, warm-up 0.5 | −3.41 [−8.00, +1.17] | +1.41 [−3.22, +6.10] | 4.0x |
| absorbing, warm-up 0.5 | −2.86 [−7.17, +1.40] | +0.10 [−4.13, +4.29] | 8.9x |

And at fixed everything else, gating on the repaired map against gating on
the plain masked map is **−0.60 [−5.43, +4.34]**.

A tenfold more accurate contact map is worth nothing, because the oracle
ceiling is already at parity. What costs is that a pairwise rule displaces a
per-position confidence ordering that is already within 1.65 pLDDT of a
*trained* optimum; accuracy only makes the displacement better targeted.

### And it loses to doing nothing, in both decoders

At matched forward passes, with the threshold set so the rule actually
binds (verified: it excludes all but 0.02 to 0.1 percent of coupled pairs):

| selector | 12 | 25 | 50 | 100 | mean vs baseline |
|---|---|---|---|---|---|
| confidence (baseline, free) | 50.40 | 51.51 | 53.83 | 57.61 | - |
| sequence distance (control, no structure) | 47.51 | 54.12 | 57.12 | 58.05 | **+0.86** |
| contact shuffled (control) | 46.61 | 48.12 | 56.96 | 57.11 | -1.14 |
| contact-gated (proposed) | 45.24 | 50.71 | 50.55 | 57.36 | **-2.38** |
| contact seeking (opposite polarity) | 38.29 | 43.53 | 49.20 | 53.26 | -7.27 |

The proposed rule is the worst of the three "spread the batch out"
variants. A control containing no structural information at all, simply
keeping committed positions far apart in sequence, is the only one that
does not lose to the baseline.

Run exactly as the proposal specifies, with a separate frozen ESM-2 650M
forward pass at every step, the rule gains a mean 1.5 pLDDT over its own
shuffled control and a mean **0.05** over the free confidence baseline,
while roughly doubling the compute per step:

| forward passes | contact (separate ESM-2 pass) | shuffled | confidence baseline |
|---|---|---|---|
| 12 | 46.52 | 43.13 | 50.40 |
| 25 | 54.24 | 54.01 | 51.51 |
| 50 | 55.12 | 54.18 | 53.83 |

Pushing the threshold harder, so the rule rejects the top 10 percent of
each sequence's coupling values rather than the top 1 percent, produces the
cleanest null in the whole study. Real contact map against its own shuffle:
-0.09 at 12 passes, +0.23 at 25, +0.04 at 50. Mean **+0.06 pLDDT**. With
the mechanism switched on as hard as it can be, the structural information
is worth nothing at all.

Inside DPLM's own decoder the result is sharper still, and goes somewhere
unexpected:

| forward passes | baseline | contact rule | shuffled control |
|---|---|---|---|
| 25 | 58.10 | -3.54 | -3.37 |
| 50 | 65.62 | -6.91 | -3.39 |
| 100 | 78.23 | **-14.90** | -5.62 |
| 200 | 80.24 | **-13.79** | **-0.77** |

Using the REAL contact map is worse than using a permuted one. At 200
passes the shuffled matrix costs 0.77 pLDDT while the genuine map costs
13.79. A permuted matrix perturbs the surviving set arbitrarily and later
steps repair it; a genuine contact map perturbs it systematically, pushing
apart exactly the positions that are mutually constrained, and the decoder
cannot repair that. The structural information is not merely useless here,
it is pointed the wrong way.

### The full control ladder, with cost and diversity

Added 2026-11-10. One frozen backbone (DPLM-150M), length 200, budgets
12/25/50/100/200, 48 sequences per cell, matched reveal-set sizes, free
self-attention signal, pooled to 240 sequences per arm:

| selector | pLDDT | relative cost | vs confidence |
|---|---|---|---|
| **confidence** | **55.66** | 1.00x | — |
| contact (attention graph) | 55.53 | 1.04x | −0.13 [−3.44, +3.30] |
| distance (spacing, d=8) | 53.99 | 1.00x | −1.67 [−4.92, +1.64] |
| contact_shuffled | 53.64 | 1.04x | −2.02 [−5.31, +1.27] |
| PUNT | 52.35 | **2.08x** | −3.31 [−6.62, +0.03] |
| random | 50.53 | 1.00x | **−5.13 [−8.37, −1.92]** |

Nothing beats confidence ordering. `random` is the only arm significantly
different from it, which matters as a positive control: the protocol can
see a five-point effect, so the nulls are nulls rather than an insensitive
measurement. PUNT is the most expensive arm and among the worst.

Cost in denoiser-pass equivalents, measured on the run GPU: denoiser 1.00,
denoiser with attentions 1.04, **frozen ESM-2 650M contact call 7.88**.
Diversity 0.932–0.937 and novelty 0.997–1.000 across every arm.

### How large an effect this design can actually see

Added 2026-11-09, and it reframes every negative above. Per-sequence pLDDT
standard deviation is 17.8, so:

| sequences per arm | resolvable effect at 80 % power |
|---|---|
| 48 (one cell) | 7.2 pLDDT |
| 192 (four cells pooled) | 3.6 pLDDT |
| 913 | 1.65 pLDDT |

1.65 is the largest *published* gain for unmasking-order selection on this
model class (arXiv:2606.00295). So the negatives above decisively rule out
the large gains both proposals claimed, and they cannot exclude a 1 to 2
point effect — which is all that is on offer in this literature. Two
corollaries we now apply as rules:

- A 12-cell sweep reports a +7 maximum under a true null 27 percent of the
  time, so a sweep's best cell is never quoted as a result. When we
  replicated our own best cell (warm-up 0.35, T=50, +7.25) at 960 sequences
  per arm on fresh seeds, it came back **+0.65 [−0.96, +2.25] n.s.** — 91
  percent of the effect gone.
- Three findings in this project have now died this way: that sweep cell, the
  ad-hoc oracle set, and the fold-class effect below.

### The one Idea-A positive we reported, and withdrew

On 2026-11-09 we reported that the oracle effect was fold-dependent —
mainly-beta minus mainly-alpha of +11.61 pLDDT [+4.61, +18.68], permutation
p = 0.005, apparently corroborated by r = +0.42 against contact density.
**Withdrawn 2026-11-10** (`results/G_beta_effect_note.md`). The corroborating
correlation had been computed against *class-mean* density, which is a
four-valued relabelling of the class itself. Against each chain's own
density:

| test of the claimed mechanism | result |
|---|---|
| correlation with per-chain contact density | +0.120 [−0.122, +0.356], p = 0.44 |
| split at median per-chain density (true − shuffled) | −0.05 [−6.80, +6.71] |
| stability across budgets 12/25/50/100 | +1.19, −4.66, +25.44, +24.45 |
| reversed polarity, correlation with density | +0.192, i.e. drifts the same way |
| chains per group needed for the effect as estimated | 16; we have 11 beta and 15 alpha |

The class split is significant; the mechanism it was meant to demonstrate is
absent when tested directly. Idea A has no surviving positive result.

### A framing problem worth knowing about regardless

The proposal assumes an absorbing process in which a written residue is
final. DPLM is not that. Its decoder re-predicts every position at every
step and returns the lowest-scoring ones to the mask state, so nothing is
final until the end. In our harness that difference is worth about 30
pLDDT. Any method defined on "which masks may be filled together" is
operating in a decoder family well below the one the field actually uses.

---

## Idea B: the two-level alphabet

### The mechanism is published twice, and the binding rule is contradicted

Two NeurIPS 2025 papers already do this. One writes a coarse class then the
identity, with the coarse class binding on the identity prediction, and
derives the closed-form bound. The other gives every position multiple
slots with per-slot masking and a validity-masked output head. Neither
touches proteins, which leaves the protein instantiation open, but the
mechanism is not the contribution.

Worse for the proposal as written: the first paper's own ablation found that
a hard binding constraint was its **worst** setting, and softening it cut
generative perplexity by 62 percent. The proposal makes binding a rule
enforced in code.

The six chemical families are also not a judgement call. They are, to within
the placement of histidine and cysteine, a six-letter code published in 2002
under the same six group names.

### The one comparison that would have been a contribution comes back negative

Since the mechanism is taken, the contribution had to be that a
chemically structured partition beats an arbitrary one of the same size.
Three arms, 38M parameters each, same backbone, corpus, budget and forward
passes:

| arm | pLDDT | ESM-2 pseudo-perplexity | hydrophobic periodicity |
|---|---|---|---|
| natural proteins | 90.12 | 3.67 | 0.150 |
| **flat, 20 letters** | **43.09** | **13.69** | **0.146** |
| two-level, 6 chemical families | 36.68 | 14.86 | 0.142 |
| two-level, 6 RANDOM groups | 35.41 | 16.37 | 0.139 |

The gap also fails to close with more sampling compute. Across 25, 50, 100
and 200 forward passes the flat arm climbs from 37.2 to 43.9 while the
two-level arms sit flat at 34 to 37:

| arm | 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| flat | 37.15 | 40.38 | 43.09 | 43.88 |
| two-level, chemical | 35.34 | 35.79 | 36.68 | 35.98 |
| two-level, random | 34.16 | 33.85 | 35.41 | 35.65 |

The chemical partition beats a random one by 1.27 pLDDT with overlapping
intervals. The chemistry is real information on the training objective, and
consistently so, but it does not survive into generation quality. And both
two-level arms are 6 to 8 points BELOW the flat baseline: the extra channel
costs more than it returns.

### One part does work, and it is the part that is not the mechanism

Handing the model a family pattern taken from a real protein, and letting it
fill identities underneath, is the only positive result in the study.

**Corrected twice, 2026-11-10.** The original +26.4 pLDDT was measured on 48
patterns drawn from the upstream SwissProt dev split, which is *not*
homology-clustered: 17 of the 48 sources shared more than half their 10-mers
with a training sequence. The contamination was specific to the chemical
channel — the chemical arm gained +10.2 pLDDT on exactly those sources while
the random arm gained nothing — which is the signature of a coherent pattern
acting as a retrieval key for a memorised fold. Re-run on the 96 cleanest dev
sequences (maximum 10-mer containment 0.026 against a mean of 0.37 for the
original set), the headline is smaller and stands:

| arm | n | pLDDT | pTM | foldability | ESM-2 pppl |
|---|---|---|---|---|---|
| natural proteins | 48 | 90.10 | 0.838 | 0.813 | 3.50 |
| **chemical pattern given** | 96 | **56.49** | 0.457 | **0.083** | 14.74 |
| chemical families, unconditional | 96 | 36.58 | 0.201 | 0.000 | 14.94 |
| **random-group pattern given** | 96 | **36.27** | 0.197 | 0.000 | 15.93 |
| random groups, unconditional | 96 | 35.30 | 0.188 | 0.000 | 16.59 |

| contrast | clean set (n=96) | original set (n=48) |
|---|---|---|
| **chemical − random, paired per source** | **+20.22 [+16.37, +24.14]** | +26.58 |
| chemical conditioned − its own unconditional | +19.91 | +26.44 |
| random conditioned − its own unconditional | +0.97 | +1.13 |

Contamination was worth roughly 6 pLDDT of the original 26.4. **Quote +20,
not +26.**

Three things make this more than a large number. The control is matched on
information content, not just on group sizes: a random partition carrying 490
bits against the chemical partition's 484 is worth +0.97, so the gain is the
chemistry and not the act of constraining the output. The comparison is
paired per source protein, so it does not rest on which proteins were drawn.
And it is not retrieval: identity of a generated sequence to its own pattern
source is 0.361, against a 0.305 floor for picking uniformly inside each
family and a 0.397 floor for always writing the family's most common residue
— the model lands *below* the trivial compliant baseline — with k-mer novelty
0.9998 against training.

### How far that +20 has actually been tested

Stated plainly, because the negatives in this report are much better
cornered than this positive is.

**One dataset.** SwissProt only. The patterns come from the dev split of the
same corpus the model was trained on, and the clean set is a re-selection
*within* that split (96 of 560 eligible sequences, chosen for lowest 10-mer
containment), not an independent dataset. Also one model pair (the 38M
from-scratch `dms6`/`rand6` checkpoints), one length (200), one sampling
budget (100 forward passes), and one structure predictor. By contrast the
Idea-A negatives were tested across two structural sets, two decoder
families, five budgets and three map sources. The asymmetry is the honest
weak point of this report.

**Several metrics, but not several independent ones.** pLDDT, pTM, pAE and
foldability all come from a single ESMFold forward pass; per-sequence pTM
correlates 0.909 with pLDDT, and foldability is a threshold on all three. So
that is one axis read four ways, agreeing as it must. What the four readings
do add is a qualitative form the mean does not show: the conditioned chemical
arm is the only one of the four generated arms that produces **any** foldable
sequence at all, 0.083 against 0.000 three times over. A 38M from-scratch
model given a real chemical blueprint crosses a threshold it otherwise never
crosses.

The metrics that are *not* read off ESMFold are weaker, and two of them do
not support the chemistry claim at all:

| independent metric | conditioned chemical | conditioned random | reading |
|---|---|---|---|
| ESM-2 650M pseudo-perplexity | 14.74 | 15.93 | moves 0.20 against its own unconditional arm, while pLDDT moves 19.91 |
| hydrophobic patterning | 0.0207 | 0.0203 | **does not discriminate** — an effect of conditioning, not of chemistry |
| amphipathic index | 0.149 | 0.148 | flat; natural is 0.150 |
| helix / strand power | 0.125 / 0.111 | 0.124 / 0.114 | flat |
| identity to source, novelty, diversity | 0.361 / 0.9998 / 0.931 | 0.074 / 1.000 / 0.926 | retrieval ruled out |

ESM-2 pseudo-perplexity is the one quality metric computed by a different
model, and it barely moves. It is also not fully independent, since ESMFold's
trunk *is* ESM-2 — a circularity we disclose rather than claim to have
controlled. So the correct summary is that **conditioning on a real chemical
pattern is worth about 20 pLDDT of ESMFold confidence, crosses a foldability
threshold nothing else crosses, is not retrieval, and is worth almost nothing
in sequence likelihood.** An independent predictor such as OmegaFold or a
Boltz-class model would settle it; none is on disk.

**UPDATED 2026-11-12 (Round 6).** The paragraph above was written when pLDDT
was the only axis that moved. It no longer is. Re-scoring the same stored
sequences in ESM-2 embedding space gives a second instrument that separates
the arms the same way: Fréchet distance from the generated to the natural
distribution falls from 27.7 (unconditional) to 12.0 with a chemical pattern
and to 26.1 with a bit-matched random one, which is 64 % of the distance to
natural against 5 %; CATH gives 60 % against 0.3 %; MMD agrees without a
Gaussian assumption, and natural-against-natural calibrates at ~0. Max-ID by
blastp independently reproduces Round 4's purge result, and NSR places every
conditioned arm below its own modal-in-family floor, so retrieval is excluded
on an alignment-free basis as well. Two of the three metrics in the table
above that "do not support the chemistry claim" are now joined by two that
do; pseudo-perplexity is retired as a discriminator because it contradicts
itself across datasets. What remains genuinely open is the structural
question, which Ref-TM addresses in `results/M_reftm_note.md`.

This supports the family channel as a CONTROL INTERFACE, where a person
supplies the pattern. It does not support the two-level alphabet as a way to
improve unconditional generation, which is what the proposal is mainly about.

**Still unshown.** That a *hand-written* or consensus pattern works — every
pattern so far was read off a real protein, so this measures the capacity of
the channel, not a design method. And whether the conditioned sequence adopts
its *source protein's* fold or merely some confident fold, which a TM-score
against the known source would settle.

### The gain survives a homology purge: memorisation is worth 10 % of it

Added 2026-11-11, full detail in `results/J_clustered_split_note.md`. The
+20.22 above was measured on sources selected by 10-mer containment, a
substring test that cannot see remote homology. Tested properly: 128 sources
sampled with **no** homology criterion, every BLAST relative at E ≤ 1e-3
deleted from training, and all three arms retrained with every hyperparameter
unchanged.

First, how much homology the containment filter was missing:

| | |
|---|---|
| sources with a BLAST relative in training | **125 of 128 = 97.7 %** |
| training sequences deleted | 19,645 = **11.89 % of the corpus** |

So the earlier "homology-clean" set was clean only of shared substrings. The
2×2, identical patterns in every cell:

| corpus | chemical − random, paired per source | own gains, chem / rand |
|---|---|---|
| original | **+26.09 [+22.89, +29.28]** | +25.81 / +0.72 |
| purged | **+23.38 [+20.07, +26.68]** | +24.58 / +1.66 |
| **difference — what training homologs were worth** | **+2.71 [+0.95, +4.55]** | |

The effect is not memorisation. The leak is real, excludes zero, and accounts
for 10.4 % of the unpurged figure. Foldability for the conditioned chemical
arm is if anything higher on the purged corpus (0.172 against 0.156), and the
random conditioned arm stays at 0.000 — so the only arm producing foldable
sequences does so on a model that has never seen a relative of any pattern it
is handed.

Scope: a depth-1 single-linkage blastp neighbourhood, not an MMseqs2
clustering (neither MMseqs2 nor CD-HIT is installed), so
relatives-of-relatives and fold-level homologs undetectable by BLAST remain
in training.

### And it replicates on CATH S35

Added 2026-11-11, full detail in `results/K_cath_dataset_note.md`. Every
number above uses SwissProt dev sequences as pattern sources, drawn from the
same corpus the model trains on. Repeated with CATH S35 — one of the two
canonical fold classifications, non-redundant at 35 % identity, and composed
of structural *domains* rather than full-length records, at matched length
(mean 224.3 against 225.8).

| dataset | original corpus | purged corpus | memorisation |
|---|---|---|---|
| SwissProt dev | +26.09 [+22.89, +29.28] | **+23.38 [+20.07, +26.68]** | +2.71 [+0.95, +4.55] |
| CATH S35 | +21.06 [+17.86, +24.24] | **+19.85 [+16.62, +23.05]** | +1.21 [−0.68, +3.06] |

The intervals overlap, so the effect is a property of protein chemistry
rather than of the SwissProt dev split. (Separate bootstraps on different
pattern sets, so overlap is an informal check, not a test of equality.)
Memorisation is smaller on CATH exactly as its lower homology predicts — 42 %
of sources have a training relative against SwissProt's 98 % — and its
interval contains zero. Together the two datasets bound memorisation at
roughly 1–3 pLDDT of a 20–26 pLDDT effect.

The CATH run is the better-controlled of the two: the depth-2 single-linkage
expansion ran where SwissProt's exceeded its budget, and the purge cost 4.02 %
of the corpus against 11.89 %, so its purged row is barely handicapped by
lost training data (flat-arm validation 4.6505 against 4.6328 on the full
corpus, versus 4.7536 for the SwissProt purge).

**The random control goes negative on CATH, which is new and corroborating.**
A random-partition pattern scores −1.80 against its own unconditional
baseline (33.87 vs 35.67) where on SwissProt it was +0.72. That is the
degradation-curve mechanism showing up independently: a chemically meaningless
pattern costs something because binding makes a family error unrecoverable,
and it costs more on compact structural domains than on full-length records
with their linkers and disordered regions. The separation widens because the
control gets worse, not because the chemical arm gets better.

Two things to carry forward rather than gloss. CATH foldability for the
conditioned chemical arm is 0.047–0.055 against SwissProt's 0.156–0.172, with
every random arm at 0.000 — so the "only arm that folds at all" claim holds in
form on both datasets but its magnitude is a SwissProt number and should not
be quoted as general. And ESM-2 pseudo-perplexity does not favour the
conditioned arm on CATH either (16.56 conditioned against 14.94
unconditional, slightly worse), so the caveat that the one metric not read
off ESMFold fails to support the effect is unchanged and if anything sharper.

**Quote +19.9 [+16.6, +23.1] on CATH and +23.4 [+20.1, +26.7] on SwissProt,
both with the pattern sources' homologs deleted from training.**

Scope on both: the *generator* is SwissProt-trained throughout. This shows the
pattern source can be independent of training, not that the effect survives a
model trained on a different corpus — and training on CATH is not feasible
here, since 3,876 eligible domains is far too small a corpus for a 38M model
from scratch.

### The model cannot supply the pattern itself

Added 2026-11-11, full detail in `results/I_pattern_curve_note.md`. The
obvious next move is to have the model predict the family pattern and then
fill identities under it — which is the two-level arm, already trained. It
does not work, and now there is a number for why.

| pattern given | pLDDT | gain over unconditional |
|---|---|---|
| perfect (a real protein) | 55.41 | **+18.83 [+15.15, +22.55]** |
| 5 % error | 49.65 | +13.07 |
| 10 % error | 43.23 | +6.65 |
| **the model's own prediction** | **36.87** | **+0.29 [−1.09, +1.78]** |
| none | 36.58 | — |
| fully wrong | 30.20 | **−6.38 [−7.00, −5.70]** |

The slope is −1.15 pLDDT per 1 % of pattern error: 7.8 % error halves the
gain, 18.6 % erases it. The family head's error is 63.7 % by argmax and
78.2 % for a sampled pattern — three to four times the budget — and its
accuracy is 0.3634 against a majority-class floor of 0.3633 at *every*
identity-reveal level from 0 to 0.75.

Three things follow, and two of them were not predictable in advance.

**The curve is chemical, not a constraint effect.** The bit-matched random
partition has slope −0.04 against −1.11, and a perfect random pattern is
worth +1.02 [−0.21, +2.33]. There is no curve to degrade, so the steep slope
is chemical information being destroyed rather than an alphabet restriction
being relaxed.

**A wrong pattern is actively harmful, which explains Idea B's whole
deficit.** Binding makes a family error unrecoverable, so a fully wrong
chemical pattern costs −6.38 against generating freely (a fully wrong
*random* pattern costs only −0.39). The two-level arm's self-predicted
families are near-random, which is the mechanism behind its 6.5-point gap
below the flat baseline — a number reported above without an explanation
until now.

**It is not a training artefact.** The family loss weighting is genuinely
pathological: with γ = 2 the channel is masked with probability t² and
weighted 1/t², whose variance reaches 10⁶ at the ε = 1e-3 end, and the head's
informative regime is the small-t regime where almost nothing is
family-masked. Two corrected variants, γ = 1 and a clipped weight, improve
validation by 1.0 and 1.25 nats and carry 2.5× the information — and leave
argmax accuracy exactly at the floor. The extra information went into sharper
probability mass, not a different mode.

So staying inside the budget needs ≥ 92 % accuracy on a 6-way chemical class
from sequence alone, with no structure, where structure-conditioned inverse
folding reaches 35–50 % exact residue recovery *with* the backbone given. A
predictor that good would be a stronger protein model than the one consuming
its output. **The pattern is useful precisely because it is exact and comes
from outside, which is what makes this a controllability result and why it
cannot be bootstrapped.**

### The motivating diagnosis is wrong

The proposal's decisive test was hydrophobic periodicity, on the premise
that a masked diffusion model "has no mechanism for organising patterning at
that scale and should look flat by comparison". Measured against a
composition-matched shuffle of each sequence, DPLM's generations carry about
two and a half times the hydrophobic patterning of natural proteins, not
less. The two-level arms produce slightly less periodicity than the flat
arm, not more. The test moves the wrong way and the deficit it was meant to
detect is not there.

---

## What is actually worth keeping

Three results from this work are publishable, and none of them is either
original proposal.

1. **Why external planners fail on masked protein input, and the one-line
   repair.** The contact-quality-versus-mask-rate curve with an all-mask
   null, scored against deposited structures, plus the finding that filling
   the masks with the denoiser's argmax restores precision from 0.067 to
   0.737 and lets the external head overtake the generator's own attention.
   This explains P2-Bert's published and unexplained eight-point failure. It
   is the strongest claim in the project and it exists because of the
   professor's feedback; the version of this item written on 2026-11-06 drew
   the opposite design rule from the same curve.
2. **The commit-ranking trap.** Ranking commits by clean confidence and
   committing irrevocably looks 26 pLDDT better and is pure mode collapse:
   polyleucine, 131 of 200 identical residues, pseudo-perplexity 1.27. Any
   paper reporting pLDDT without repetition and diversity beside it can be
   fooled this way.
3. **The decoder-family gap.** The absorbing framing that most of this
   literature assumes sits about 30 pLDDT below the remasking decoder DPLM
   ships. That is a large, simple, checkable fact about where the quality
   in these models comes from.

4. **The chemical family pattern as a conditioning interface.** A pattern
   over six chemical families is worth **+23.4 pLDDT [+20.1, +26.7]** to a
   small model whose training corpus contains no BLAST relative of any
   pattern source, a bit-matched random partition of the same group sizes is
   worth +1.7, and the conditioned arm is the only one of four that produces
   a foldable sequence at all. It is the one piece of either proposal that
   survived its own control, a retrieval check, and a homology purge that
   removed 11.89 % of the corpus — on one corpus, one model and one
   predictor, as the scope section above states.

   Two results make this more than a large number. **Memorisation accounts
   for 10 %** of it and no more (+2.71 [+0.95, +4.55] of +26.09), measured by
   retraining rather than argued. And the gain is **chemical rather than a
   constraint effect**: the degradation slope is −1.15 pLDDT per 1 % pattern
   error for the chemical partition against −0.04 for a bit-matched random
   one, so there is no curve to degrade when the partition carries no
   chemistry.

8. **A coarse chemical pattern is only useful when it is nearly exact, and
   nothing can predict it that well.** The error budget is 7.8 % for half the
   gain and 18.6 % for any of it, while a properly trained family head sits
   at 64 % error and at its majority-class floor regardless of how much
   context it is given. A wrong pattern is worse than no pattern (−6.38),
   because binding makes a family error unrecoverable. This is a general
   statement about coarse-to-fine generation in masked diffusion, not a fact
   about our checkpoint: two independent corrections to the loss improved it
   by 1.0–1.25 nats and moved the accuracy not at all. It also explains the
   unconditional two-level arm's 6.5-point deficit against a flat model,
   which is otherwise an unexplained negative.

5. **The independence assumption in parallel masked-diffusion decoding is
   empirically safe.** P2 states it and leaves it untested (App. C.4,
   coordinates "are thus assumed independent"). We tested it with
   ground-truth maps, which is the strongest available test, and a pairwise
   rule does not beat a per-position one even with the true graph.

6. **The firing-fraction diagnostic.** In the remasking decoder, harm is
   linear in how often the gate fires: `harm = −21.02 x fired_fraction +
   1.99`, r² = 0.991, extrapolating to zero when the rule never fires. This
   distinguishes "my schedule fixed the mechanism" from "my schedule
   disabled it", and it is what showed the proposed warm-up to be a dose
   control rather than a repair. Any paper reporting a schedule that
   improves a gated sampler should be asked for this plot.

7. **The power critique.** Per-sequence pLDDT standard deviation is 17.8, so
   a 48-sequence cell resolves only effects above about 7 pLDDT while the
   largest published gain for unmasking-order selection on this model is
   1.65 and needs roughly 913 sequences per arm. Much of this subfield is
   reporting cells it cannot resolve; three of our own findings died when
   held to it.

If the project needs an ICML-shaped target from here, there are two honest
ones. A negative-result and diagnostics paper built on items 1, 5, 6 and 7,
which has the advantage that item 1 explains a published failure and item 5
closes a stated assumption in the closest prior work. Or a
controllable-design paper built on item 4, which has the advantage of a
positive result with a clean control, and which connects to a thirty-year-old
line of protein design work on specifying binary hydrophobic patterns and
leaving identities open — but which needs the degradation curve, a
hand-written pattern and a second predictor before it is a paper.

The question put back to the professor on 2026-11-10, in
`ANSWERS_TO_FEEDBACK.md`, is which of those two to spend the remaining time
on. It has not been answered yet.
