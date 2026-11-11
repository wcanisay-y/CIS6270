# The two-temperature claim

Idea B's second test: sampling families cold to protect the fold and
identities hot for variety should move the whole quality-diversity curve
outward, not just improve one number. The right comparison is the
two-dimensional attainable set for the two-level arm against the
one-dimensional curve for the flat arm.

Flat arm, single temperature swept 0.6 to 1.1:

| temperature | pLDDT | diversity |
|---|---|---|
| 0.60 | 59.24 | 0.876 |
| 0.70 | 50.42 | 0.903 |
| 0.80 | 43.82 | 0.918 |
| 0.90 | 44.89 | 0.922 |
| 1.00 | 42.02 | 0.928 |
| 1.10 | 35.88 | 0.934 |

Two-level arm, family and identity temperatures swept independently, best
twelve points:

| family T | identity T | pLDDT | diversity |
|---|---|---|---|
| 0.70 | 0.70 | 46.34 | 0.864 |
| 0.70 | 0.85 | 44.93 | 0.885 |
| 0.70 | 1.00 | 44.90 | 0.898 |
| 0.70 | 1.15 | 44.10 | 0.900 |
| 0.85 | 0.70 | 42.93 | 0.892 |
| 0.85 | 1.15 | 40.15 | 0.918 |
| 1.00 | 1.00 | 39.21 | 0.920 |
| 1.00 | 1.15 | 35.68 | 0.927 |

Read at matched diversity, which is the only fair way to read it:

| diversity | flat | two-level | gap |
|---|---|---|---|
| ~0.87 | **59.24** | 46.34 | -12.9 |
| ~0.90 | **50.42** | 44.10 | -6.3 |
| ~0.918 | **43.82** | 40.15 | -3.7 |
| ~0.928 | **42.02** | 35.68 | -6.3 |

The flat arm's one-dimensional curve dominates the two-level arm's
two-dimensional attainable set at every diversity level tested. The second
temperature does give a second dial, and the dial does what it should: at a
fixed family temperature, raising the identity temperature trades quality
for diversity smoothly, so the mechanism works as designed. It simply
starts from a worse place and never catches up.

The claim was that the whole curve moves outward. It moves inward.
