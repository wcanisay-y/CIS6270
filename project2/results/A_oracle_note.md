# The oracle test: would a correct contact map help?

This is the test that decides Idea A independently of how good contact
prediction is. The selector is handed the TRUE contact map of a real
protein, taken from ESMFold coordinates at a CB-CB cutoff of 8 angstroms
with a minimum sequence separation of 6. If the mechanism is sound, a
correct map should beat the same map with its rows and columns permuted,
which preserves the value distribution and destroys only the association
between coupling and position.

Length 200, 48 sequences per cell, matched forward passes.

| forward passes | true contact map | shuffled map | difference |
|---|---|---|---|
| 6 | 41.29 | 37.47 | +3.82 |
| 12 | 45.44 | 45.02 | +0.42 |
| 25 | 51.27 | 50.93 | +0.34 |
| 50 | 55.54 | 59.66 | **-4.12** |

Mean difference across the four budgets: +0.12 pLDDT. The true map wins
narrowly at two budgets, ties at one and loses by four points at the
fourth, and every interval overlaps. There is no effect.

The opposite polarity, committing mutually CONTACTING residues together,
scores 38.33 / 42.14 / 47.04 / 52.89 across the same budgets, three to five
points below the non-contacting rule at every budget. So that direction is
worse, but the non-contacting rule is not better than noise.

None of these arms is degenerate: diversity is 0.93 throughout and the mean
longest repeat run is 2.6 to 4.0 residues, so the numbers are not being
driven by collapse.

This result is what makes the rest of Idea A moot. We had already measured
that ESM-2's contact head is at its all-mask null by 25 percent masking, and
the natural reading of that was "the idea is right but the signal is not
available yet". The oracle removes that defence. Even with a perfect
contact map in hand, choosing mutually non-contacting positions to commit
together does not generate better proteins than simply committing the
positions the model is most confident about. Improving contact prediction
cannot rescue a mechanism that does not help when contact prediction is
exact.
