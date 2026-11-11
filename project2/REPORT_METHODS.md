# What was built and how it was tested

Two proposals for improving masked diffusion models of protein sequences,
evaluated with small demo runs on standard data against published methods.
Everything below is in `project2/`. The environment is `project2/venv2`
(torch 2.7.1); the corpus is SwissProt, cached as plain text in
`project2/data/`.

## Shared setup

**Corpus.** SwissProt via the HuggingFace `SwissProt-Pfam` release, 409,267
sequences. Filtered to canonical residues and length 40-254, which is the
filter the one published from-scratch SwissProt protein-diffusion study
uses, so its baseline table is comparable.

**Folding evaluation.** ESMFold v1. We report mean pLDDT, pTM, mean pAE and
the composite foldability criterion used by the protein decoding-order
papers: pLDDT > 80 and pTM > 0.7 and pAE < 10. Note that this port returns
pLDDT on a 0-1 scale; we rescale to 0-100 to match published tables.

**Sequence evaluation.** Exact masked pseudo-perplexity under frozen ESM-2
650M, mean pairwise identity within a cell, a 10-mer novelty rate against
the training corpus, amino-acid composition divergence from natural, mean
longest homopolymer run, and the hydrophobic-periodicity metrics below.
A natural-sequence row, length-matched and drawn from the held-out split,
is scored identically in every table.

**Hydrophobic periodicity.** Implemented the way the periodicity literature
prescribes rather than as a plain Fourier transform: the least-squares power
spectrum, which fits a sinusoid of each angular frequency to the
hydrophobicity trace of a sliding 18-residue window and records the share of
variance it accounts for. We report the helical band (87.5-107.5 degrees per
residue, i.e. about 3.6 residues per turn), the strand band, and the
amphipathic index, which is the helical band's share of total spectral area.
The primary scale is the Eisenberg consensus, with Kyte-Doolittle and the
purpose-built Cornette scale as robustness checks. Validation: the estimator
recovers the exact angle for a synthetic 3.6-periodic signal, and separates a
designed amphipathic helix (index 0.515) from a scrambled sequence (0.079).

**Uncertainty.** Every cell mean carries a bootstrap 95 percent interval.
Comparisons are paired contrasts within a matched cell, reported as a
difference with its own bootstrap interval and a Mann-Whitney p-value, not
as a comparison of two headline numbers.

## Idea A: choosing the parallel unmasking set

**Base model.** DPLM-150M, the published masked-diffusion protein language
model, loaded as a standard ESM masked LM.

**The matched-NFE protocol.** The schedule fixes how many positions must be
committed at each step, so every selector completes a length-200 sequence in
exactly T forward passes. Selectors differ only in WHICH positions are
committed together. Contact-model calls are counted separately and reported,
because the proposal's specified signal costs a second full network pass per
step.

**Selectors.** confidence (the MaskGIT and DPLM rule, with the Gumbel
perturbation DPLM's own appendix says is needed to stop the decoder
collapsing to near-constant sequences); random; lowest entropy; the proposed
contact-gated greedy rule; a reimplementation of the conditional-independence
test from the published parallel-sampling work; and DPLM's own published
reparameterised decoder with its repeat-resampling step, as the reference
point.

**Controls that decide the result.**
- *contact shuffled*: the same greedy rule on a symmetrically permuted
  contact matrix. Same value distribution, same group sizes, no structural
  information. If the proposed rule only matches this, the contact map
  contributed nothing.
- *sequence distance*: coupling defined as sequence separation below a
  threshold. Tests whether contacts beat trivially spreading the batch out.
- *contact seeking*: the opposite polarity. The published inverse-folding
  work decodes contacting residues together or first; the proposal keeps
  them apart. These are opposite prescriptions from the same signal and both
  are run.
- *oracle*: the true contact map of a real protein, from ESMFold
  coordinates. This separates "the contact map is too noisy" from "the
  mechanism does not help". If a correct map does not beat confidence, no
  improvement in contact prediction can rescue the idea.

**Coupling sources.** The proposal's own: a separate frozen ESM-2 650M pass
per step. A free alternative: the generator's own attention maps read
through the ESM-2 contact-head regression, which costs nothing because the
forward pass has already happened. And raw symmetrised attention with no
contact head, which is the signal the recent text-domain methods use.

**Threshold.** A fixed coupling threshold is the wrong instrument here: a
contact head on a mostly masked sequence emits probabilities near 0.01, so a
threshold of 0.3 rejects 0.05 percent of pairs and the rule silently becomes
the confidence baseline. We therefore also run a quantile threshold, which
rejects a controlled share of each sequence's own coupling values and puts
the contact head, raw attention and a binary oracle map on the same footing.

**Prerequisite measurement.** Before any of that: how good is the contact
prediction on a partially masked sequence? Scored against ESMFold contacts
of the full sequence, across mask rates, with an all-mask null as the floor.
Nobody has published this curve.

## Idea B: the two-level alphabet

**Arms, all 38M parameters, same backbone, same corpus, same budget
(20,000 steps at batch 256), same float64 categorical sampling.**

| arm | what it is |
|---|---|
| flat | standard absorbing masked diffusion, 20 letters |
| dms6 | two-level, the published six-letter chemical code |
| prop6 | two-level, the proposal's own partition (H and C placed differently) |
| rand6 | two-level, RANDOM partition with identical group sizes |
| soft08 | dms6 with a non-binding family (wrong label shown 20 percent of the time) |
| gamma1 | dms6 with a symmetric mask schedule, families no earlier than identities |
| hp2 | two-level with 2 families (hydrophobic/polar) |
| murphy10 | two-level with 10 families |
| ar | autoregressive transformer at matched size |

rand6 is the experiment the idea stands or falls on. The two-level mechanism
itself is already published twice; what is unclaimed is whether a partition
that matches the domain's chemistry beats an arbitrary one of the same size.
soft08 is included because the closest published work found a hard binding
constraint was its worst setting. ar is included because a published
from-scratch comparison at this scale puts an autoregressive transformer
well above every discrete-diffusion arm.

**Sampler.** The two-level arms follow the training corruption schedule:
noise level t runs from 1 to 0, identities are masked with probability t and
families with probability t to the power gamma, nested. Families therefore
lead identities by exactly the margin the model was trained on. The naive
reading of the proposal, write every family then every identity, is out of
distribution: no t produces "all identities masked, half the families
written", because t = 1 forces the family mask rate to 1 as well. Both are
run, so the cost of that mismatch is measured rather than assumed.

**Tests.** Hydrophobic periodicity, which is the proposal's own decisive
test; quality against diversity, swept over family and identity temperatures
for the two-level arms and over the single temperature for flat; quality
against forward passes; family-pattern compliance and foldability when the
pattern is handed to the model.

## Honest limitations

- 48 sequences per cell at one primary length. Published protocols use 40 to
  100 per length across several lengths. Within-grid contrasts share every
  setting and are the basis for the conclusions; absolute values are only
  loosely comparable with published tables.
- ESMFold's trunk is ESM-2. Steering generation with an ESM-2 contact head
  and scoring with ESMFold pLDDT shares representation bias end to end.
  Removing that would need a non-ESM folding model.
- pLDDT alone is known to be reachable by degenerate sequences, which is why
  repetition, diversity and the composite foldability criterion are reported
  beside it in every table.
- The contact-quality curve uses 12 sequences at one length. The direction is
  unambiguous and anchored by the null, but the exact location of the
  collapse needs a proper sweep before it goes in a paper.
