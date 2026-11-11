# Oracle redo, the revised method, and why published ideas score below benchmark

Four questions, answered in order. Everything here is reproducible from
`project2/`. New code: `parallel/pdb_oracle.py`, `parallel/run_pdb_oracle.py`,
`parallel/adaptive.py`, `parallel/run_revised.py`, `analysis/revised.py`.

---

## 0. Why do these results sit below benchmark when the ideas are published?

Five reasons, in descending order of how much of the gap each explains. Only
the last is a criticism of our implementation.

### The decoder family accounts for about 30 pLDDT of it

The published 84.30 for DPLM-150M at length 200 comes from its
reparameterised remasking decoder, which re-predicts every position at every
step and returns the lowest-scoring ones to the mask state. Our harness
reproduces that: 83.60 at 200 forward passes, 87.44 at the default 500.

Both proposals are defined on an absorbing process, where a written residue
is final. Every absorbing arm in this study lands at 51 to 58. Same weights,
same forward-pass budget, different commitment rule. So most of the distance
to the benchmark is not a defect in the method or the harness. It is the
decoder family the proposals chose, and it is the single most consequential
design decision in either proposal.

### The published methods win on a different axis than we measured

The parallel-decoding papers report accuracy against *steps* on text models,
and their headline claim is speed at retained quality relative to other
parallel decoders. DAPD reports up to 3.87x speedup. That is a claim about
the quality-versus-parallelism frontier, not about beating near-sequential
decoding in absolute quality.

Our grid asked a stricter question: at matched forward passes, does the
gating rule beat free confidence ordering? A method can be the best parallel
decoder available and still lose that comparison, because parallelism trades
quality for time by construction. Reading a speedup result as a quality
result is the category error at the root of both proposals.

### There is almost no headroom over confidence ordering in this model

This is the quantitative answer, and it comes from published numbers rather
than ours. "Adaptive Order Policies for Masked Diffusion" (arXiv:2606.00295)
reports mean pLDDT on DPLM-150M over lengths 100 to 500, three seeds:

| unmasking rule | pLDDT |
|---|---|
| random | 40.04 +/- 0.78 |
| entropy | 55.88 +/- 0.93 |
| margin | 82.02 +/- 0.31 |
| highest confidence | 82.20 +/- 0.76 |
| learned order policy | 83.85 +/- 0.80 |

A policy *trained* to optimise the unmasking order beats plain confidence by
1.65 pLDDT, and by 2.09 in their joint-training setting. Confidence ordering
is already within about two points of a learned optimum. Any heuristic
reordering, contact-gated or otherwise, is competing for that two-point
margin, which is narrower than the confidence interval of a 48-sequence
cell. A null result is therefore the *expected* result for this family of
method on this model, and the published gains for order selection are
themselves small. The arithmetic on their table is ours; the paper makes no
protein-specific claim about headroom.

### For Idea B we were comparing against the wrong benchmark

Our from-scratch 38M flat baseline scores 43.09. The right comparison is
other from-scratch models at that scale on SwissProt, which DiMA (ICML 2025)
reports at 36.7 to 37.8 for discrete diffusion and 61.0 for an
autoregressive transformer of similar size. Our baseline is *above* the
published range for its class.

The two-level arms at 35 to 37 are therefore at the published level for
from-scratch masked diffusion, not below it. The appearance of being far
below benchmark came from comparing a 38M from-scratch model with DPLM, a
150M model pretrained on a much larger corpus. HDLM and MDM-Prime also
report gains in perplexity and FID on text and images, not in a downstream
structural metric, and MDM-Prime's stated advantage is reclaiming idle
sampling steps, which a matched-forward-pass protocol removes by
construction. Our own numbers show the mechanism moving the likelihood
metric while leaving the structural one alone.

### What is genuinely a limitation of our test

We implemented the described rules; we did not run the authors' released
code. Two things keep the conclusion standing anyway. The oracle test is
nearly implementation-free: a correct map, handed directly to the selector,
either helps or it does not. And the learned-policy ceiling above bounds the
achievable gain independently of us.

One real gap, found while doing this work: `contact_min_revealed`, the
warm-up that defers contact gating until enough of the sequence is written,
exists in the sampler and was **never set in any run**. Every Idea-A arm
consulted contacts from the first step, including the regime our own
diagnostic says is information-free. The mechanism verdict from the oracle
stands, but the signal-availability half of the original argument was not
tested properly. That is exactly what the revised method fixes, which is why
it deserved a real test rather than an appeal to the old grid.

---

## 1. The oracle, redone on deposited structures

Two confounds in the first oracle run, one expected and one not.

The expected one was circularity: the "true" map came from ESMFold, whose
trunk is ESM-2, and the scoring metric is ESMFold pLDDT. The unexpected one
was an atom-indexing error. ESMFold's `positions` output is atom14 ordered
`N, CA, C, O, CB`, so index 3 is the backbone oxygen and CB is index 4.
The first run took index 3. Every "CB-CB contact map" in the original study
was an O-O map, and glycine, which has no atom14 CB, was never handled.

### The new ground truth

48 non-redundant deposited chains, X-ray, resolution at most 2.0 angstrom,
one polymer instance, a break-free window of 200 standard residues checked
for consecutive numbering and consecutive CA-CA below 4.5 angstrom, pairwise
sequence identity below 30 percent. CB taken from the structure, virtual CB
from N, CA, C for glycine. Validation: CA-CB distance 1.528 angstrom against
a textbook 1.53, consecutive CA-CA 3.803 angstrom with no break above 3.91.

### Circularity was not the problem; the atom index was

ESMFold folds these 48 sequences at mean pLDDT 89.31. Its contact map,
scored against the deposited map:

| predicted map | precision@L vs deposited | Jaccard vs deposited |
|---|---|---|
| ESMFold CB-CB, correct index | 0.881 | 0.788 |
| ESMFold O-O, the earlier error | 0.532 | 0.315 |

Contact density: deposited 2.25 percent, ESMFold CB-CB 2.20 percent,
ESMFold O-O 1.89 percent. ESMFold-CB against ESMFold-O-O has a Jaccard of
only 0.335.

So an ESMFold-derived CB map is a good stand-in for a deposited one, and the
circularity concern was not material. The atom index was: the map the
original oracle handed its selector shared only about a third of its
contacts with the real one.

### The contact-quality table, rescored against deposited structures

Same measurement as before, now with real labels. Length 200, 12 proteins,
all-mask null as the floor.

| input | AUROC ext | P@L ext | AUROC self | P@L self |
|---|---|---|---|---|
| unmasked | 0.956 | 0.775 | 0.941 | 0.680 |
| 25 % masked | 0.703 | 0.087 | 0.939 | 0.668 |
| 50 % masked | 0.687 | 0.066 | 0.935 | 0.648 |
| 75 % masked | 0.655 | 0.060 | 0.827 | 0.365 |
| 90 % masked | 0.663 | 0.078 | 0.665 | 0.060 |
| all-mask null | 0.658 | 0.069 | 0.713 | 0.065 |

ext is a separate frozen ESM-2 650M pass through its contact head, which is
what the proposal specifies. self is the generator's own attention read
through the same head, at no extra cost.

Both original findings survive, and both are now much sharper.

ESM-2 650M on an unmasked sequence reaches precision@L 0.775, not the 0.305
the oxygen labels reported. The earlier absolute numbers were depressed by
the wrong ground truth. The collapse under masking is real and larger than
we thought: 0.775 to 0.087 at a single quarter masked, which is at the
all-mask null of 0.069. ESM-2 was trained at a 15 percent mask rate and has
never seen this input.

The free signal is not merely better, it is roughly ten times better in the
regime the selector runs in. DPLM's own attention holds precision@L 0.668 at
25 percent masked and 0.648 at 50 percent, against 0.087 and 0.066 for the
separate ESM-2 pass. It gives up a little peak accuracy on a clean sequence,
0.680 against 0.775, and buys near-immunity to masking down to about 20
percent of the sequence revealed. DPLM was trained across the whole masking
range; ESM-2 was not. This is the strongest result in the study and it is
the one that reframes both the original proposal and its revision.

### The oracle on deposited maps

48 deposited maps, one per sequence slot, threshold absolute at 0.5 so the
rule rejects a pair that is genuinely in contact. The rule binds: of the
pairs committed together it leaves 0.04 to 0.25 percent in contact against
a 2.25 percent base rate, so the selector really is avoiding co-committing
contacting positions. No arm is degenerate, diversity is 0.934 and the mean
longest repeat is about three residues.

| forward passes | confidence | true map | shuffled map | true - conf | true - shuf |
|---|---|---|---|---|---|
| 12 | 50.40 | 43.48 | 48.43 | -6.92 | -4.94 |
| 25 | 52.00 | 50.73 | 50.00 | -1.28 | +0.72 |
| 50 | 61.00 | 51.54 | 61.82 | -9.46 | -10.27 |
| 100 | 59.37 | 60.29 | 55.88 | +0.91 | +4.40 |

Paired and pooled over all four budgets, 192 paired samples:

| contrast | effect | 95 % CI |
|---|---|---|
| true minus shuffled | -2.52 | [-6.05, +0.86] |
| true minus confidence | -4.19 | [-7.79, -0.66] |

A deposited contact map does not beat its own shuffled control, and
contact-gated selection is significantly worse than plain confidence at
matched forward passes. The earlier near-zero reading of +0.12 was itself
produced by a map that shared a third of its contacts with the real one.

### How large an effect could this design ever have seen

This is the limitation that matters most, and it applies to the original
study as much as to this one. Per-sequence pLDDT in the absorbing decoder
has a standard deviation of 17.8.

| samples per arm | standard error | smallest effect detectable at 80 % power |
|---|---|---|
| 48, one cell | 2.57 | 7.2 pLDDT |
| 192, pooled over four budgets | 1.28 | 3.6 pLDDT |
| 913 | 0.59 | 1.65 pLDDT |

The largest published gain for unmasking-order selection on this model is
1.65 pLDDT, from a policy trained for the purpose. Detecting that needs
about 913 sequences per arm. At 48 per cell the original grid could only
ever have resolved effects above about 7 pLDDT.

So the negative results are correctly stated only in one direction. They
rule out the large improvement both proposals claimed. They cannot exclude
a one-point effect, and a one-point effect is all that is on offer. The
useful conclusion is not that the mechanism is broken but that the whole
axis is mispriced: the maximum prize is under two pLDDT and costs roughly a
thousand folds per arm to measure, while the single choice of decoder family
is worth about thirty.

---

## 2. Does the revised method make sense?

**Its diagnosis is right, its fix does not work, and the experiments now
explain why the fix cannot work.** The proposal deserved the test it got:
the defect it identifies is real and was genuinely never exercised. But
every arm that fires the rule is worse than not firing it, and the warm-up
that appears to rescue the method does so only by switching the method off.

The document proposes three changes to Idea A. Taken in turn.

### Component 1, the warm-up: correct diagnosis, no benefit

The proposal observes that gating on contacts from the first step consults a
signal that does not exist yet, and defers gating until a fraction of the
sequence is written. That observation is correct, it is the same thing the
masking table in section 1 shows quantitatively, and `contact_min_revealed`
was never set in any run of the original study. So this was a real gap.

In the absorbing decoder with the free signal, four warm-ups by three
budgets, 48 sequences per cell, block bootstrap with the cell as the block:

| warm-up | contact - confidence | contact - shuffled |
|---|---|---|
| 0.00 | +2.03 [-2.24, +6.34] | +0.07 [-4.07, +4.20] |
| 0.20 | +1.01 [-3.03, +5.08] | +0.50 [-3.57, +4.51] |
| 0.35 | +3.06 [-1.35, +7.41] | +4.19 [-0.00, +8.37] |
| 0.50 | -0.92 [-5.15, +3.30] | -1.89 [-6.15, +2.34] |
| **pooled** | **+1.29 [-0.80, +3.38]** | **+0.72 [-1.35, +2.78]** |

Nothing is significant, there is no monotone trend in the warm-up, and the
pooled estimate is consistent with zero against both the free baseline and
the structure-free control.

The single best cell is +7.25 pLDDT, at warm-up 0.35 and 50 forward passes.
That number should not be quoted, and the reason is arithmetic. It is the
maximum of twelve cells whose per-cell standard error on a difference is
3.77. Under a true null the maximum of twelve such cells averages +6.13 and
exceeds +7.32 in **26.9 percent** of trials. A result that a coin flip
produces one time in four is not a result. A replication of exactly that
cell at 960 sequences per arm is running as job 27149380.

### The warm-up in the decoder that actually reaches benchmark

The absorbing decoder sits about 30 pLDDT below DPLM's published number, so
a rule that only survives there is a rule about a decoder nobody ships. Run
inside the remasking decoder, the picture is not ambiguous at all:

| warm-up | rule fires on | contact - confidence | contact - shuffled |
|---|---|---|---|
| 0.00 | 100 % of steps | -18.55 [-21.67, -15.35] | -7.36 [-10.53, -4.19] |
| 0.35 | 66 % of steps | -12.63 [-16.34, -8.88] | -6.84 [-10.85, -2.87] |
| 0.50 | 50 % of steps | -8.79 [-12.63, -4.99] | -3.71 [-7.59, +0.26] |
| 0.75 | 26 % of steps | -2.96 [-6.69, +0.86] | -1.70 [-5.51, +2.11] |
| **pooled** | | **-10.73 [-12.54, -8.91]** | **-4.90 [-6.76, -3.01]** |

The rule is significantly harmful wherever it meaningfully fires, and it is
harmful even against its own shuffled control, which means the structural
information in the map is worse than no information at all.

And the harm is a straight line in how often the rule fires:

```
harm = -21.02 * (fraction of steps the rule fires) + 1.99     r^2 = 0.991
```

Extrapolated to a rule that never fires, the effect is +1.99 pLDDT, which is
zero within noise. **The warm-up does not repair the rule. It titrates the
dose of it, and the quality recovers in exact proportion to how much of the
rule has been removed.** That is the signature of an intervention with no
useful content, and it is not what a working fix looks like: a working fix
would show quality peaking at an intermediate warm-up, not rising
monotonically towards the value it takes when the mechanism is disabled.

### Component 2, the cached map with trigger refresh: sound, and aimed at the wrong target

As engineering this is reasonable, and it has published precedent (see
section 3). Two findings against it here.

The distribution-change trigger has almost no dynamic range. Mean
total-variation change in the per-position amino-acid distributions over
still-masked positions runs 0.048 to 0.090 across all 28 arms, never
reaching the 0.15 threshold the proposal suggests. Pooled over the grid it
fires 36 times against 229 reveal-fraction and 157 cache-age refreshes, so
the signal the proposal is built around accounts for **9 percent** of the
refreshes it triggers; the other two triggers, which are functions of step
count and reveal count rather than of the distribution, do the work.

The five refresh policies are then indistinguishable, and the fixed-interval
arm is the best of them:

| refresh policy | refreshes | cost | pLDDT |
|---|---|---|---|
| fixed interval, age 8, ignores the distribution | 5.0 | 89.4 | 57.23 |
| trigger-based, default thresholds | 5.0 | 89.4 | 56.99 |
| every step | 13.0 | 152.5 | 56.79 |
| eager | 11.7 | 141.9 | 56.40 |
| lazy | 5.0 | 89.4 | 50.64 |

The spread across all five is 6.59 pLDDT against a per-cell standard error
of 2.57, so this is noise. The relevant comparison is the first two rows:
at identical refresh count and identical cost, a fixed interval that never
looks at the distribution scores 57.23 against the trigger's 56.99.
Refreshing every single step, at 1.7x the cost, buys nothing either.

More fundamentally, caching only buys anything for the external signal, and
the external signal is the one that does not work. Measured on the run GPU:

| call | seconds | cost in denoiser passes |
|---|---|---|
| DPLM-150M denoiser | 0.0229 | 1.00 |
| denoiser with attentions returned | 0.0237 | 1.04 |
| ESM-2 650M contact head | 0.1803 | 7.88 |

The free signal, read off the generator's own attentions, costs 4 percent
more than a plain denoiser pass and needs no cache, no second network and no
refresh policy at all. The caching machinery exists to amortise a 7.88x call
whose precision@L at 25 percent masking is 0.087 against a null of 0.069.
Optimising the cost of that call is optimising the cost of a component that
should be deleted.

### Component 3, the group-size cap: no effect

| cap | contact arm | its own confidence baseline | difference |
|---|---|---|---|
| 8 | 57.92 | 54.21 | +3.71 |
| 16 | 52.35 | 56.77 | -4.42 |

Opposite signs at adjacent settings, both inside the noise of a 48-sequence
cell. Nothing here.

### The oracle, now across three map sources

Section 1 gave the deposited-map oracle. With all three map sources scored,
the mechanism verdict has a dose-response of its own, in map quality. Paired
on the map index, 192 paired samples per contrast:

| map source | Jaccard vs deposited | true - shuffled | true - confidence |
|---|---|---|---|
| deposited CB-CB, X-ray | 1.000 | -2.52 [-5.94, +0.93] | **-4.19 [-7.72, -0.65]** |
| ESMFold CB-CB, fixed | 0.788 | -2.26 [-5.42, +0.99] | **-3.63 [-7.12, -0.23]** |
| ESMFold O-O, earlier bug | 0.315 | -0.35 [-3.84, +3.08] | -2.86 [-6.50, +0.85] |

**Repairing the map does not weaken the effect; it strengthens it.** With
either correct map the true-minus-confidence contrast excludes zero
(-4.19 and -3.63); with the degraded oxygen map it does not (-2.86
[-6.50, +0.85]). The original study's near-zero reading was not a lucky
null, it was a map sharing a third of its contacts with the truth.

A caution on how hard to push this. The three points are monotone in map
quality and a regression on them returns r = -0.98, but that is three
points whose intervals overlap heavily: the deposited and ESMFold-CB values
are statistically indistinguishable from each other, and the oxygen point's
interval contains both. So the defensible claim is the qualitative one --
map quality is not the binding constraint, and improving the map does not
move the result toward the proposal's prediction -- not a quantitative
dose-response law. The firing-frequency relationship below is the one with
four well-separated points and is the load-bearing dose-response.

The rejection diagnostics say why map quality is not the lever: the rule
binds about equally hard whatever map it is given. The fraction of
co-committed pairs that are genuinely in contact is 0.00102 (deposited),
0.00128 (ESMFold CB) and 0.00088 (oxygen) against a 2.25 percent base rate,
and the shuffled controls sit at 0.00109 / 0.00111 / 0.00100. The amount of
displacement from confidence ordering is therefore roughly constant across
map sources; what changes is only which positions get displaced.

### Why a better signal does not help: it is displacement, not accuracy

The free self-attention signal is about ten times more informative than a
separate frozen ESM-2 pass in the regime the selector runs in (0.668 against
0.087 precision@L at 25 percent masking). It still does not make the rule
work, and the reason is that the rule's cost does not track its accuracy.

First, reversing the polarity does not rescue it. `anti_contact` co-commits
CONTACTING positions, the opposite of the proposal, and is the polarity used
by Yang et al. and by Layne et al. Paired, across four runs:

| run | anti - contact | anti - confidence | contact - confidence |
|---|---|---|---|
| oxygen-map oracle, absorbing | -3.29 [-6.21, -0.42] | - | - |
| self-signal oracle, absorbing | -0.06 [-4.06, +3.99] | -0.86 [-4.63, +2.96] | -0.80 [-3.33, +1.81] |
| predicted self, absorbing | -6.05 [-9.12, -3.22] | -9.14 [-12.21, -6.24] | -3.08 [-6.81, +0.68] |
| predicted self, remasking | -4.37 [-7.98, -0.81] | -14.16 [-17.71, -10.59] | -9.79 [-13.78, -5.75] |

Both polarities lose to plain confidence, and the reversed one loses by more
in three of four runs. A tempting mechanistic story -- that contacting pairs
are the mutually-determining pairs, so refusing to co-commit them defers the
best-determined positions -- predicts that the reversed polarity should win.
It does not. That story is refuted and should not be repeated.

What survives is simpler. Every pairwise variant we can construct loses to
confidence ordering: the true map (-4.19), its shuffled control (-1.67 by
subtraction), and the reversed polarity (-0.86 to -14.16). The common factor
is not which pairwise rule is applied but that a pairwise rule is applied at
all, displacing a per-position confidence ordering that is already within
1.65 pLDDT of a *trained* optimum (arXiv:2606.00295). The firing-frequency
line, harm = -21.02 x fired_fraction + 1.99 at r^2 = 0.991, is exactly this
statement in quantitative form: the cost is proportional to how much of the
confidence ordering gets overridden, and the extrapolated cost of overriding
none of it is zero.

This is also why the ten-times-better signal changes nothing. Accuracy buys
a better-targeted override, and the override itself is what costs. It is
consistent with P2, whose planner is strictly per-position and which gains
2.2 pLDDT on proteins with the denoiser as its own planner: the per-position
axis has a small amount of headroom on it, and the pairwise axis has none.

### Verdict on item 2

The document is a competent piece of diagnostic work. It correctly
identifies that the original Idea A consulted a signal before that signal
existed, it proposes the right intervention for that defect, and it was
right that the original grid had not tested it. Implemented faithfully and
tested at four warm-ups, two decoders, three map sources and a measured cost
model, it does not work: zero in the decoder where it is harmless, strongly
negative in the decoder that matters, negative against its own
structure-free control, and monotonically less harmful the more of it you
switch off. The one component worth keeping from the whole exercise is not
in the document: use the generator's own attention rather than a separate
frozen network, which is both free and ten times more informative. That
finding deletes the document's main engineering contribution, since there is
then nothing expensive left to cache.

---

## 3. Is the last one a novelty?

**Not as a method. Partly, and defensibly, as a measurement.** The two
mechanisms the revised method is built from are both published, but the
specific gap it aims at is a real one that the closest prior work explicitly
declines to address. The honest contribution of this work is that it closes
that gap, negatively, with evidence the prior work does not have.

### What is already published

| component of the revised method | prior art | relation |
|---|---|---|
| an auxiliary model chooses the unmasking order, training-free, at inference | **P2 / Path Planning**, arXiv:2502.03540, ICLR 2025, Peng et al., with code | same setting, same base model family, and it is the paper that defines the planner/denoiser split |
| the generator itself as the order-selection signal | P2's **Self-Planning** variant | our "free self-attention signal" is the same idea |
| a separate frozen encoder as that signal | P2's **BERT-planning** variant | same idea; see below, it fails there too |
| a learned order policy as the ceiling | arXiv:2606.00295 | high confidence 82.20 +/- 0.76 vs learned policy 83.85 +/- 0.80 |
| cache an auxiliary computation, refresh adaptively when its input has changed enough | **dLLM-Cache**, arXiv:2506.06295 | prompt refresh interval, response refresh interval, adaptive update ratio; up to 9.1x on LLaDA-8B and Dream-7B |

The caching-with-adaptive-refresh component is dLLM-Cache's mechanism
applied to a planner instead of to transformer features. The warm-up is a
step-dependent schedule for when to consult the planner, which P2 does not
have, but it is a scheduling heuristic rather than a mechanism, and the
experiments above show it functions as an off switch.

### What is genuinely not published, and is the proposal's real claim

P2's planner is strictly **per-position**. Its score is indexed by a single
coordinate j, normalised across positions, and the top k are taken
(Eq. 6: `G_phi^j(z, x_t)`; Algorithm 1: `UpdatePos <- Top_k(t)(G~_eta(z, x_t))`).
Nothing in P2 is pairwise. The contact rule is a different object: it is a
**pairwise** constraint on which positions may be committed *together*, and
no amount of per-position reranking expresses it.

And P2 does not merely omit the pairwise problem, it assumes it away in
writing. Appendix C.4: *"The coordinates ... are thus assumed independent
... The hope in making this assumption is that each coordinate ... will be
able to be simulated independently in parallel without introducing
significant error."* Section 2.1 notes that simultaneous denoising of
multiple tokens is what breaks the continuous-time limit, and section 3.2
introduces top-k as "the possibility of changing multiple tokens per
iteration" without revisiting the independence assumption.

So the revised method targets a stated, acknowledged, unmeasured assumption
in the closest prior work. **That is a legitimate novelty claim, and it is
the only one available here.** The problem is the answer. With a perfect
deposited-structure contact map the pairwise constraint is worse than its
own shuffled control and significantly worse than plain confidence, better
maps do more damage, and in DPLM's own decoder the harm is linear in how
often the constraint is applied. The independence assumption P2 hoped was
harmless appears to be harmless, at least against the strongest pairwise
intervention we can construct.

### An independent check that our central measurement is right

P2's own protein ablation corroborates the one strongly positive finding in
this study. On a 150M MDM, the same scale as our DPLM-150M (their Table 5,
pLDDT / foldability):

| method | pLDDT | foldability |
|---|---|---|
| RDM, no planner | 78.79 | 48.57 % |
| P2-Self, denoiser as its own planner | 80.98 | 49.86 % |
| **P2-Bert, separate frozen encoder** | **70.80** | **35.43 %** |
| P2-Train, planner trained for the task | 83.45 | 58.86 % |

A separate frozen encoder used as the planner is **8 pLDDT worse than no
planner at all**, while the generator's own signal is 2.2 better. That is
exactly the asymmetry our deposited-structure measurement predicts and
explains: precision@L at 25 percent masking is 0.087 for a separate frozen
ESM-2 against a 0.069 null, and 0.668 for DPLM's own attention. P2 reports
the failure; we can say why it happens. Note also that P2's headline 22
percent comes from **P2-Train**, a planner trained for the purpose, in line
with the 1.65-pLDDT learned-policy ceiling in arXiv:2606.00295; the
training-free self variant gains 2.2 pLDDT. (Their Table 2 and Table 5
disagree on what should be the same configuration and the paper does not
reconcile them, so we cite Table 5, which matches our model scale.)

### What is publishable here, and what is not

Not publishable as proposed: a parallel-unmasking method gated on predicted
contacts, with or without warm-up and caching. The mechanism is refuted at
the oracle level, the per-position axis it would compete on is occupied by
P2 and by learned order policies, and the cost-saving component optimises
the signal our own measurement says to discard.

Publishable, as a short negative-result and diagnostics paper:

1. **Why external planners fail on masked protein input.** A frozen ESM-2
   650M contact head collapses from precision@L 0.775 to 0.087 at a single
   quarter of the sequence masked, against a 0.069 all-mask null, while the
   generator's own attention read through the same head holds 0.668 at 25
   percent and 0.648 at 50 percent. Labels are deposited X-ray structures,
   not predictions. This explains a published failure (P2-Bert) that its own
   authors report without explaining, and it gives a concrete design rule:
   read the signal off the generator, not off a model trained at a 15
   percent mask rate.
2. **The independence assumption in parallel masked-diffusion decoding is
   empirically safe.** Tested with ground-truth contact maps from deposited
   structures, across two decoder families, three map sources and four
   intervention strengths, with two dose-responses that both run the wrong
   way. This closes an assumption P2 states and leaves open.
3. **A warm-up that works as an off switch.** The r^2 = 0.991 linearity of
   harm in firing frequency is a clean, reusable diagnostic for telling
   "my schedule fixed the mechanism" apart from "my schedule disabled the
   mechanism." Anyone reporting that a warm-up rescued an auxiliary-signal
   method should be asked for this plot.
4. **The power result, which is a criticism of the subfield's protocol.**
   Per-sequence pLDDT has a standard deviation of 17.8, so a 48-sequence
   cell resolves only effects above about 7 pLDDT, while the largest
   published gain for unmasking-order selection on this model is 1.65 and
   needs roughly 913 sequences per arm. Small-gain claims in this area are
   routinely evaluated below the power needed to see them, and a 12-cell
   sweep reports a +7 maximum under a true null more than a quarter of the
   time.

That is a real paper, but it is a negative and methodological one, and it is
not the paper either original proposal set out to write. The honest summary
is that the revised method is a well-reasoned fix to a correctly diagnosed
defect in an idea whose mechanism does not work, and that the most valuable
thing produced along the way was the measurement that explains why nobody's
version of it works.

---

## 4. Replication, a standard dataset, and what the rule should have gated on

Four additions, one of which **reverses a conclusion stated in section 1**.

### 4.1 The sweep's best cell did not replicate

Section 2 flagged +7.25 pLDDT at warm-up 0.35 and 50 forward passes as the
maximum of twelve cells, and predicted that a true null would produce a
maximum that large 26.9 percent of the time. Rerun at 960 sequences per arm
on fresh seeds (base 500,000, so no sample is reused):

| arm | n | pLDDT | pAE |
|---|---|---|---|
| confidence | 960 | 55.96 | 17.94 |
| contact, warm-up 0.35 | 960 | 56.61 | 17.72 |
| shuffled control | 960 | 56.32 | 17.91 |

| contrast | effect | 95 % CI | |
|---|---|---|---|
| contact - confidence | +0.65 | [-0.96, +2.25] | n.s. |
| contact - shuffled | +0.29 | [-1.34, +1.93] | n.s. |
| shuffled - confidence | +0.36 | [-1.25, +1.97] | n.s. |

**91 percent of the sweep's effect vanished.** The half-width is 1.60, as
designed. Note what this does and does not settle: +7.25 is decisively
excluded, but the interval still contains both 0 and the 1.65 that a trained
order policy achieves, so a one-to-two-point effect remains neither shown
nor excluded. The pseudo-perplexity column is not interpretable here --
scoring used `--pppl-n 8`, so it rests on 8 sequences with a CI of
[10.14, 14.61].

### 4.2 The structural dataset was not fold-non-redundant, and it mattered

The first oracle drew 48 chains under a pairwise sequence-identity cap of
0.30. That is not non-redundancy. Mapped onto CATH 4.4:

| | ad-hoc set | CATH S40 set |
|---|---|---|
| chains | 48 | 43 |
| distinct CATH topologies | **19** | **43** |
| folds appearing more than once | 7 | 0 |
| most frequent fold | **3.20.20 in 11 of 48 entries** | - |
| mainly-alpha / mainly-beta / alpha-beta | **2 / 12 / 27** | 16 / 11 / 16 |

Twenty-three percent of the old set was one fold family and an entire
structural class was represented by two chains. The replacement is the
official CATH S40 non-redundant set, one chain per CATH topology,
single-topology chains only so a 200-residue window has an unambiguous fold
label, stratified over the four classes. Validation: 0 non-standard
residues, density 0.0198, mainly-beta 0.0244 against mainly-alpha 0.0150.

### 4.3 On the standard set the oracle's sign reverses

| forward passes | confidence | true map | shuffled | anti | true-conf | true-shuf |
|---|---|---|---|---|---|---|
| 12 | 48.79 | 49.75 | 45.10 | 44.05 | +0.96 | +4.65 |
| 25 | 50.41 | 47.74 | 42.96 | 46.63 | -2.66 | +4.78 |
| 50 | 57.93 | 60.42 | 60.04 | 49.98 | +2.50 | +0.38 |
| 100 | 55.17 | 58.78 | 56.97 | 57.75 | +3.61 | +1.81 |

Pooled, 172 paired samples: true - confidence **+1.10 [-2.67, +4.82]**,
true - shuffled **+2.91 [-0.74, +6.55]**, anti - confidence -3.47
[-7.09, +0.10]. None significant.

**This contradicts section 1**, which reported true - confidence =
-4.19 [-7.72, -0.65] on the ad-hoc set and read it as the oracle refuting
the mechanism. On the field-standard fold-non-redundant set the same
contrast is +1.10 and not significant. The two intervals barely overlap. The
earlier "significant negative" was substantially a property of a set in
which one fold family carried a quarter of the weight, so the effective
sample size for a fold-level claim was far below 48.

The correct statement is therefore weaker in both directions than section 1
claimed: **the oracle result is not robust to the choice of structural
dataset**, and neither the earlier negative nor this positive should be
quoted as settled. What survives unchanged is the remasking-decoder result,
which does not depend on the oracle set.

### 4.4 The effect is fold-dependent, as predicted from contact density

| CATH class | n | contact density | confidence | true map | true - confidence |
|---|---|---|---|---|---|
| mainly-alpha | 15 | 0.0150 | 55.00 | 50.41 | -4.59 [-10.01, +0.71] |
| **mainly-beta** | 11 | 0.0244 | 48.78 | 56.25 | **+7.47 [+0.20, +14.84]** |
| alpha-beta | 16 | 0.0212 | 54.34 | 55.65 | +1.31 [-5.38, +7.88] |

Beta-rich folds carry 63 percent more contacts per residue, and they are
where a true contact map helps; alpha-rich folds are where it hurts. This
was predicted in advance of the measurement, which is why the CATH set was
built with class labels. Caveats: 11 chains, an interval that only just
excludes zero, and one marginal result among three classes tested, so this
is a hypothesis with support, not an established effect.

### 4.5 Why: contacts are the wrong graph, and right too late

A contact map is a double proxy. The error parallel decoding actually incurs
is the total correlation of the co-committed set under the MODEL, so the
relevant object is the model's own interventional dependency
D[i,j] = TV( p(x_j | ctx), p(x_j | ctx, x_i := v) ), measured directly.

**Contacts barely predict it, and sequence distance predicts it better.**
Deposited maps, 48 chains, AUROC for separating contacting from
non-contacting pairs by dependency:

| revealed | AUROC(dep given true contact) | AUROC(seq. distance given high dep) |
|---|---|---|
| 10 % | 0.545 | 0.747 |
| 25 % | 0.543 | 0.705 |
| 50 % | 0.671 | 0.803 |
| 75 % | 0.728 | 0.854 |

The CATH set agrees (0.536, 0.543, 0.611, 0.650). A trivial sequence-distance
rule is a better predictor of the model's coupling than an experimentally
determined contact map, at every masking level. That also explains why the
original grid's `distance` control scored +0.9 against confidence while the
contact rule scored -2.4.

**And the error the rule targets is gone before contacts become useful.**
Leave-one-out TV shift inside a committed group of 8:

| revealed | factorization error (TV) | best rule | the original rule |
|---|---|---|---|
| 25 % | 0.4992 | dpp_dep -60.7 % | pair_contact **+5.0 %** |
| 50 % | 0.0086 | pair_dep +0.0 % | pair_contact +105.2 % |
| 75 % | 0.0096 | dpp_dep -10.9 % | pair_contact +132.9 % |

The error collapses 58-fold between 25 and 50 percent revealed, while
contact informativeness only rises after that. **Contact quality and error
magnitude are anti-aligned in time: there is no phase of generation in which
a contact map is both informative and needed.** This is the deeper reason
the warm-up cannot work -- it moves gating into the regime where the error
has already vanished -- and it is consistent with 4.4, since the fold class
with the most contacts is the one where the alignment is least bad
(AUROC(dep given contact) 0.623 for mainly-beta, 0.577 alpha-beta, 0.564
mainly-alpha, the same rank order as the oracle benefit).

### 4.6 What the rule should have gated on

At 25 percent revealed, group of 8, measured not inferred:

| rule | factorization error | vs confidence | mean confidence of picks |
|---|---|---|---|
| **dpp_dep**, groupwise determinant on measured dependency | 0.1963 | **-60.7 %** | 0.347 |
| pair_dep, pairwise on measured dependency | 0.3589 | -28.1 % | 0.381 |
| dpp_contact | 0.4718 | -5.5 % | 0.399 |
| confidence | 0.4992 | - | 0.403 |
| **pair_contact**, the proposal's rule | 0.5240 | **+5.0 %** | 0.379 |
| pair_shuffled, control | 0.5263 | +5.4 % | 0.385 |

The proposal's rule **increases** the error it exists to reduce and is
indistinguishable from its own shuffled control. Gating on measured
dependency cuts it 28 percent pairwise and 61 percent groupwise. The
determinant is doing real work: in a unit test where a mutually dependent
block of six also has the highest confidence, plain confidence takes all six
and the determinantal rule takes one.

Two caveats. The dependency rules pick lower-confidence positions (0.347
against 0.403), so part of the reduction is a change of target rather than a
better choice among the same targets -- though `pair_contact` pays the same
confidence cost (0.379) and buys nothing. And measuring D costs about 40
extra forward passes, so this is an oracle-style feasibility result, not a
method.

**Correction to the end of section 2.** It said the decorrelation axis is
empty. That is too strong: groupwise dependency gating cuts the real error
by 61 percent. What is empty is the contact-based, pairwise, late-applied
version -- which is exactly the version both proposals specify. Whether a 61
percent error reduction converts into pLDDT is untested and is now the most
promising open experiment in this project. Section 7.2 adds a caveat on the
61 percent figure, found on re-reading the same run.

## 5. Metric audit: the pTM bug, and what survives it

This section exists because of the instruction to check metrics other than
pLDDT. Doing so found a scoring bug of my own, not a property of the method,
and the bug had been producing the only significant positive result in the
whole study.

### 5.1 The bug

HuggingFace's `EsmForProteinFolding` exposes `ptm` from openfold's
`compute_tm`, which ends in `argmax` over the *whole batch* and returns one
scalar per forward call: the best-scoring alignment of the best sequence in
the batch. `common/metrics.py` read that scalar per sequence, so every
member of a fold batch received the batch **maximum**. At `--fold-batch 12`
a 48-sequence cell therefore had 4 distinct pTM values, and pTM was silently
a batch-level extreme-value statistic rather than a per-sequence one. The
foldability flag thresholds on pTM (`pLDDT > 80 and pTM > 0.7 and
pAE < 10`), so it inherited the same corruption.

The fix recomputes pTM per sequence from `ptm_logits`, cropped to each
sequence's own length. `tools/verify_ptm.py` checks the recomputation is
batch-invariant: the same sequence scored at batch sizes 1, 4 and 16 agrees
to 1.2e-3, which is ordinary floating-point nondeterminism across batch
shapes, against the 0.4-wide discrepancies the bug produced.

| | pre-fix | post-fix |
|---|---|---|
| distinct pTM values per 48-sequence cell | 4 | 48 |
| corr(pLDDT, pTM) over 1,440 sequences | 0.183 | **0.909** |
| mean pTM, confidence / contact / shuffled | 0.780 / 0.848 / 0.817 | 0.383 / 0.412 / 0.402 |
| mean pTM, 48 natural length-200 windows | - | 0.838 |

The level correction is the part worth keeping. The buggy pTM put generated
sequences at 0.78-0.85, statistically indistinguishable from real proteins
(0.838) while their pLDDT was 55 against 90. That should have been caught on
sight. Corrected, generated pTM is 0.38-0.41, 16 percent of generated
sequences clear pTM > 0.7, 14 percent clear pLDDT > 80, and mean foldability
is 0.107 against 0.812 for natural windows.

### 5.2 The claim it was supporting, withdrawn

I had reported a significant pTM gain for the contact rule. It came from two
independent errors stacked on the same contrast: the batch-maximum read, and
a bootstrap that pooled 12 contact cells against 3 baseline cells, reusing
each baseline sequence 4 times and so computing an interval on 576 values
when 48 were independent.

| contrast, free-signal sweep | pTM effect | 95 % CI | |
|---|---|---|---|
| pre-fix, pooled as originally run | +0.0674 | [+0.0525, +0.0822] | "significant" |
| post-fix, pooled the same way | +0.0284 | [+0.0036, +0.0536] | still excludes 0 |
| **post-fix, cell-block bootstrap** | **+0.0284** | **[-0.0041, +0.0624]** | **n.s.** |
| post-fix, contact - shuffled, cell-block | +0.0093 | [-0.0216, +0.0402] | n.s. |

The effect halves when pTM is computed correctly and loses significance when
the interval stops double-counting. The contrast that isolates structural
information -- contact against its own shuffled control at matched warm-up
and budget -- is +0.009.

Foldability tells the same story in the clearest possible form. Contact
scores 0.101 against 0.076 for confidence, which is the "+32 percent
relative" I quoted. But the shuffled control scores **0.111**, a larger
relative gain than the treatment. Whatever lifts foldability here is the
change in group structure, not the contact information: a rule that gates on
a permuted matrix does it slightly better.

Per cell, nothing survives either: across 24 (budget, warm-up) contrasts on
corrected pTM, every interval contains zero, and the two foldability
intervals that touch zero do so from below (+0.125 [+0.000, +0.250]).

### 5.3 Consequences for the rest of the report

pTM post-fix correlates 0.909 with pLDDT, so it is close to a restatement of
pLDDT at this quality level rather than an independent axis. The multi-metric
check therefore changes no pLDDT-based conclusion in sections 1 through 4 --
those were computed from `plddt_list`, which the bug never touched -- and it
removes the one metric on which the method appeared to win. The remaining
positives in this project are the conditioning result (section 0) and the
groupwise dependency result (section 4.6), neither of which rests on pTM.

Four files were re-scored for the record alongside this one: the 960-per-arm
replication, the CATH and ad-hoc oracles, and the two remasking-decoder
endpoints (job 27172154, outputs `*_scored2.json`). The originals are kept,
and any pTM or foldability number in an earlier results file or note that
has not been reproduced from a `_scored2` file should be treated as invalid.

## 6. What survives both corrections

This section was written with two of the five re-scored files in. Job
27172154 has since finished (2026-11-10 00:08, no failures) and the
remaining three are reported in section 7. The first two settle the two
questions that mattered, and one of them turns out to be the first positive
result in this study that survives the dataset correction and the metric
correction at the same time.

### 6.1 Provenance of each pTM number

The CATH oracle never needed re-scoring. Its first scoring ran at 19:19,
after `common/metrics.py` was fixed at 18:28, and the re-score reproduces
its pTM lists bit for bit (43 distinct values per 43-sequence cell in both
files). Every CATH number in section 4 was therefore already computed from
per-sequence pTM.

The replication did need it, for a reason worth recording: its scoring job
started at 17:42 and so imported the pre-fix `metrics.py` into a process
that was still running when the file was edited. A module edit does not
reach a running interpreter. Its first scoring finished at 19:35 and is
pre-fix despite the timestamp.

### 6.2 The replication is a tight null on every metric

960 sequences per arm, fresh seeds, length 200, predicted self-attention
map, warm-up 0.35, T=50 -- the cell the sweep had selected as its maximum.

| arm | pLDDT | pTM | foldability |
|---|---|---|---|
| confidence | 55.96 | 0.4106 | 0.099 |
| contact | 56.61 | 0.4170 | 0.111 |
| contact_shuffled | 56.32 | 0.4134 | 0.110 |

| contrast | d pLDDT | d pTM | 95 % CI | d foldability | 95 % CI |
|---|---|---|---|---|---|
| contact - confidence | +0.65 | +0.0064 | [-0.0145, +0.0277] | +0.0125 | [-0.0146, +0.0406] |
| **contact - shuffled** | **+0.29** | **+0.0036** | [-0.0174, +0.0245] | **+0.0010** | [-0.0261, +0.0292] |
| shuffled - confidence | +0.36 | +0.0028 | [-0.0178, +0.0236] | +0.0115 | [-0.0156, +0.0396] |

corr(pLDDT, pTM) on these 2,880 sequences is 0.117 pre-fix and 0.904
post-fix. The comparison that isolates structural information -- contact
against its own shuffled control -- is +0.004 pTM and +0.001 foldability
with intervals of half-width 0.021 and 0.028. Whatever small lift the rule
produces over plain confidence, the permuted matrix produces as well.

### 6.3 The fold dependence is an interaction, and it is significant

> **WITHDRAWN 2026-11-10.** Re-tested against the mechanism it claims, this
> does not hold. The corroborating r = +0.42 was computed against
> CLASS-MEAN contact density, a four-valued relabelling of the class; against
> each chain's own density r = +0.120 [-0.122, +0.356], permutation p = 0.44.
> Splitting the same 43 chains at the median of per-chain density gives
> true-minus-shuffled +2.88 dense vs +2.93 sparse, a difference of -0.05
> [-6.80, +6.71]. The class effect lives in two of four budgets and is
> negative in a third, the reversed polarity drifts the same way with density
> (+0.192), and 11 beta chains is below the 16 per group needed to resolve an
> 11.6-point difference, so the estimate is biased upward. See
> `results/G_beta_effect_note.md` and `analysis/beta_effect.py`. The text
> below is kept as written for the record.

Section 4.4 reported the per-class effect with one interval per class, which
is three looks at the same question. The claim is an interaction -- the
oracle map should help where contacts are dense -- so it is tested as one,
resampling chains (the unit a map is attached to) and pooling the four
budgets within chain.

| contrast | pLDDT | permutation p | pTM | permutation p |
|---|---|---|---|---|
| beta - alpha, true vs confidence | +12.06 [+3.17, +21.01] | 0.023 | +0.108 [+0.008, +0.212] | 0.068 |
| **beta - alpha, true vs shuffled** | **+11.61 [+4.61, +18.68]** | **0.005** | **+0.121 [+0.040, +0.200]** | **0.010** |

Per-chain effect against class contact density: r = +0.42 on pLDDT and
+0.38 on pTM over 42 chains. The class means run in the predicted order on
both metrics (beta +8.28 / +0.089, alpha-beta +3.99 / +0.036, alpha -3.33 /
-0.032 against the shuffled control), and mainly-beta chains also gain
+0.091 foldability.

This is the one positive that survives everything. It is measured against
the shuffled control, so it cannot be an artifact of the set's fold
composition; it is computed from per-sequence pTM, so it is not the batch
maximum; the contrast was specified before measurement, from contact
density; and the ordering agrees with an independent measurement, the
AUROC with which contacts predict the model's own coupling by class
(beta 0.623 > alpha-beta 0.577 > alpha 0.564).

Its scope is narrow and must be stated with it: a deposited contact map, the
absorbing decoder, 11 chains in the beta class, and no demonstration that a
*predicted* map reproduces it -- the 960-per-arm replication with a
predicted map at an unselected fold distribution is the null in 6.2. The
honest summary is that the mechanism has real signal exactly where contact
density is high, and that both proposals apply it in a regime -- predicted
maps, mixed folds, late in sampling -- where that signal is absent.

## 7. Addendum, 2026-11-10

Two items: the three re-scored files that were still folding when section 6
was written, and a caveat on section 4.6 that the section understated. No
new generation run is reported here.

### 7.1 The remasking-decoder negative holds on corrected pTM

Mean over the four budgets (25, 50, 100, 200 forward passes), 48 sequences
per cell, from `C_reparam_w0.0_scored2.json` and
`C_reparam_w0.75_scored2.json`. The confidence baseline is the same cells in
both files.

| warm-up | arm | pLDDT | pTM | foldability |
|---|---|---|---|---|
| - | confidence | 72.24 | 0.510 | 0.292 |
| 0.00 | contact | 53.69 | 0.233 | 0.016 |
| 0.00 | shuffled control | 61.05 | 0.362 | 0.161 |
| 0.75 | contact | 69.29 | 0.487 | 0.318 |
| 0.75 | shuffled control | 70.98 | 0.512 | 0.344 |

| warm-up | contrast | d pLDDT | d pTM | 95 % CI on d pTM |
|---|---|---|---|---|
| 0.00 | contact - confidence | -18.55 | -0.277 | [-0.319, -0.236] |
| 0.00 | contact - shuffled | -7.36 | -0.130 | [-0.171, -0.089] |
| 0.75 | contact - confidence | -2.96 | -0.023 | [-0.078, +0.032] |
| 0.75 | contact - shuffled | -1.69 | -0.025 | [-0.081, +0.030] |

Bootstrap stratified by budget, 10,000 resamples; the pLDDT columns
reproduce section 2 to the second decimal, which checks the script against
the original analysis. The conclusion of section 2 is unchanged on the
corrected metric: where the rule fires on every step it is strongly harmful
and worse than its own shuffled control, and where it fires on a quarter of
the steps nothing is distinguishable from zero. Foldability is the stored
cell fraction and carries no interval, because per-sequence pAE was not
saved and the pAE clause binds in 10 of the 20 distinct cells.

The ad-hoc 48-chain oracle (`A_oracle_pdb_scored2.json`), paired on the map
index: true - confidence is -0.044 pTM [-0.089, +0.001] and true - shuffled
is -0.020 [-0.061, +0.022]. That set is superseded by the CATH set for the
reasons in section 4.2; the numbers are recorded only so that no pre-fix pTM
remains in use.

### 7.2 Caveat on 4.6: the two error measures disagree

Section 4.6 reported the leave-one-out TV shift. The same run
(`results/E_dependency.json`) also recorded the flip rate: how often the
modal token of a group member stops being the mode once the rest of the
group is written. Section 4.6 did not show it. At 25 percent revealed, group
of 8:

| rule | TV shift | flip rate | mean confidence of picks |
|---|---|---|---|
| dpp_dep | 0.196 | **0.406** | 0.347 |
| pair_dep | 0.359 | 0.208 | 0.381 |
| dpp_contact | 0.472 | 0.062 | 0.399 |
| confidence | 0.499 | **0.062** | 0.403 |
| pair_contact | 0.524 | 0.042 | 0.379 |
| pair_shuffled | 0.526 | 0.044 | 0.385 |

The two measures rank the rules in opposite order. Under plain confidence
the members of a group move each other's distributions by 0.50 TV yet keep
the same modal token 94 percent of the time, which is the signature of
sharpening: the group reinforces a choice it has already made. Under
dependency gating the shift is small, but 41 percent of the modal tokens
change. At group of 16 the flip rates are 0.276 against 0.036.

So "cuts the real error by 61 percent" in section 4.6 should read "cuts the
leave-one-out TV shift by 61 percent". Whether sample quality is lost
through TV shift or through mode instability is not established, the two
disagree here, and the measurement rests on 12 mid-generation states (96
group positions at group of 8, 39 flips against 6). The feasibility result
is therefore two-sided, not positive.

The check that should precede the pLDDT experiment needs no folding: on at
least 100 states, score each committed token by its log-probability under
the leave-one-out conditional, and split the TV shift into its same-mode and
mode-changing parts. If dependency gating does not raise that conditional
log-probability, the roughly 1,000-sequence-per-arm pLDDT run is not worth
its cost.
