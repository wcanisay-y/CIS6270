# The revised method: warm-up, cached map, trigger refresh

## T = 25

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| confidence | none | 0.00 | - | 25 | 0 | 25.0 | 53.13 | 0 |
| contact_shuffled | self | 0.00 | - | 25 | 0 | 26.1 | 54.55 | 9 |
| contact_shuffled | self | 0.20 | - | 25 | 0 | 26.1 | 51.41 | 7 |
| contact_shuffled | self | 0.35 | - | 25 | 0 | 26.1 | 49.92 | 6 |
| contact_shuffled | self | 0.50 | - | 25 | 0 | 26.1 | 49.70 | 4 |
| contact | self | 0.00 | - | 25 | 0 | 26.1 | 55.56 | 9 |
| contact | self | 0.20 | - | 25 | 0 | 26.1 | 50.48 | 7 |
| contact | self | 0.35 | - | 25 | 0 | 26.1 | 51.00 | 6 |
| contact | self | 0.50 | - | 25 | 0 | 26.1 | 51.12 | 4 |

Best warm-up arm is +2.42 pLDDT against the free confidence baseline, at 1.0x its cost.

## T = 50

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| confidence | none | 0.00 | - | 50 | 0 | 50.0 | 53.83 | 0 |
| contact_shuffled | self | 0.00 | - | 50 | 0 | 52.2 | 55.07 | 9 |
| contact_shuffled | self | 0.20 | - | 50 | 0 | 52.2 | 55.70 | 7 |
| contact_shuffled | self | 0.35 | - | 50 | 0 | 52.2 | 54.86 | 6 |
| contact_shuffled | self | 0.50 | - | 50 | 0 | 52.2 | 54.58 | 5 |
| contact | self | 0.00 | - | 50 | 0 | 52.2 | 54.57 | 9 |
| contact | self | 0.20 | - | 50 | 0 | 52.2 | 55.56 | 7 |
| contact | self | 0.35 | - | 50 | 0 | 52.2 | 61.08 | 6 |
| contact | self | 0.50 | - | 50 | 0 | 52.2 | 54.07 | 5 |

Best warm-up arm is +7.25 pLDDT against the free confidence baseline, at 1.0x its cost.

## T = 100

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| confidence | none | 0.00 | - | 100 | 0 | 100.0 | 57.83 | 0 |
| contact_shuffled | self | 0.00 | - | 100 | 0 | 104.4 | 61.04 | 12 |
| contact_shuffled | self | 0.20 | - | 100 | 0 | 104.4 | 59.20 | 9 |
| contact_shuffled | self | 0.35 | - | 100 | 0 | 104.4 | 56.63 | 8 |
| contact_shuffled | self | 0.50 | - | 100 | 0 | 104.4 | 63.41 | 6 |
| contact | self | 0.00 | - | 100 | 0 | 104.4 | 60.75 | 12 |
| contact | self | 0.20 | - | 100 | 0 | 104.4 | 61.79 | 9 |
| contact | self | 0.35 | - | 100 | 0 | 104.4 | 55.19 | 8 |
| contact | self | 0.50 | - | 100 | 0 | 104.4 | 56.83 | 6 |

Best warm-up arm is +3.96 pLDDT against the free confidence baseline, at 1.0x its cost.

