# Oracle redo: deposited structures vs ESMFold vs the oxygen bug

Each cell is 48 sequences at length 200, matched forward passes.
`true` hands the selector the contact map of a real protein;
`shuffled` permutes that same matrix, preserving its value
distribution and destroying only the association between
coupling and position. The oracle map is supplied once and never
refreshed, so this measures the mechanism, not contact quality.

## map source: deposited CB-CB (X-ray)

| forward passes | confidence | true map | shuffled map | true - shuffled |
|---|---|---|---|---|
| 12 | 50.40 | 43.48 [40.2, 47.1] | 48.43 [44.2, 53.1] | -4.94 |
| 25 | 52.00 | 50.73 [45.9, 55.8] | 50.00 [45.9, 53.9] | +0.72 |
| 50 | 61.00 | 51.54 [46.6, 56.4] | 61.82 [56.7, 67.3] | -10.27 |
| 100 | 59.37 | 60.29 [55.2, 65.5] | 55.88 [51.4, 60.6] | +4.40 |

Mean true-minus-shuffled across budgets: **-2.52 pLDDT**.

Mean true-map-minus-confidence: **-4.19 pLDDT**.

## map source: ESMFold CB-CB (fixed)

| forward passes | confidence | true map | shuffled map | true - shuffled |
|---|---|---|---|---|
| 12 | 50.40 | 44.96 [41.9, 48.2] | 46.32 [42.6, 50.2] | -1.36 |
| 25 | 52.00 | 48.65 [44.4, 53.3] | 51.64 [47.2, 56.1] | -3.00 |
| 50 | 61.00 | 52.37 [47.5, 57.5] | 63.05 [57.7, 68.6] | -10.68 |
| 100 | 59.37 | 62.30 [56.9, 67.4] | 56.28 [51.7, 61.1] | +6.01 |

Mean true-minus-shuffled across budgets: **-2.26 pLDDT**.

Mean true-map-minus-confidence: **-3.63 pLDDT**.

## map source: ESMFold O-O (earlier bug)

| forward passes | confidence | true map | shuffled map | true - shuffled |
|---|---|---|---|---|
| 12 | 50.40 | 46.56 [42.5, 51.2] | 45.82 [42.0, 49.5] | +0.74 |
| 25 | 52.00 | 53.01 [47.9, 58.4] | 51.02 [46.7, 55.4] | +1.99 |
| 50 | 61.00 | 52.20 [47.4, 56.7] | 59.09 [53.9, 64.6] | -6.90 |
| 100 | 59.37 | 59.59 [54.6, 64.9] | 56.83 [51.8, 61.9] | +2.76 |

Mean true-minus-shuffled across budgets: **-0.35 pLDDT**.

Mean true-map-minus-confidence: **-2.86 pLDDT**.

## Verdict

| map source | true - shuffled | true - confidence |
|---|---|---|
| deposited CB-CB (X-ray) | -2.52 | -4.19 |
| ESMFold CB-CB (fixed) | -2.26 | -3.63 |
| ESMFold O-O (earlier bug) | -0.35 | -2.86 |

