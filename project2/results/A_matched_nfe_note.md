# Matched-forward-pass comparison at the most parallel setting

Length 200, 6 forward passes (33 positions committed per step), 48
sequences per cell, ESMFold pLDDT with bootstrap 95 percent intervals.
Every arm spends exactly 6 generator passes except where noted.

| selector | pLDDT | 95% interval | extra cost |
|---|---|---|---|
| confidence (baseline) | **42.74** | 39.65 - 46.24 | none |
| sequence distance (control) | 41.02 | 37.50 - 44.67 | none |
| contact-gated (proposed) | 40.95 | 37.40 - 44.63 | one ESM-2 pass per step, or free from the generator's own attention |
| contact shuffled (control) | 40.48 | 37.60 - 43.66 | same |
| random order | 39.32 | 36.68 - 42.39 | none |
| conditional-independence test | 37.93 | 35.70 - 40.53 | 2x forward passes |
| contact degree, most constrained first | 36.00 | 34.38 - 37.78 | same |
| contact seeking, opposite polarity | 34.50 | 33.04 - 36.02 | same |
| lowest entropy | 34.09 | 32.67 - 35.71 | none |

Read the top four rows together. The proposed selector scores 40.95. Its
own control, the identical rule on a symmetrically permuted contact matrix
that retains the value distribution and destroys only the association
between coupling and position, scores 40.48. A control that replaces
contacts with nothing more than sequence separation scores 41.02. The three
are inside each other's intervals, and all three are below plain confidence
ordering.

That is the result the controls were built to detect: at this setting the
contact map contributes no usable information, and whatever small effect the
rule has comes from perturbing the confidence ordering rather than from
structure.

Two further readings:

- The polarity question has an answer, and it is not the one either side
  predicted. Grouping contacting residues together scores 34.50, eight
  points below the baseline, so the published hypothesis that coupled
  residues should be decoded together is not supported here either. The
  contact-order prescription, decoding the most constrained residues first,
  scores 36.00. Both directions lose to doing nothing.
- The published conditional-independence method scores 37.93 while spending
  twice the forward passes. Matched on compute rather than on denoising
  steps, it is worse than the free baseline, which is consistent with its
  own paper's note that its advantage is in steps rather than in forward
  passes.

The threshold caveat matters and is handled separately. At a fixed coupling
threshold of 0.3 the rule rejected 0.05 percent of candidate pairs, so it
barely bound at all. A separate run repeats these arms with the threshold
set at a quantile of each sequence's own coupling values, which forces the
rule to reject a controlled share of pairs. That run is what decides whether
the null result above is "the rule does nothing" or "the rule was never
switched on".
