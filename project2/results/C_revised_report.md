# The revised method: warm-up, cached map, trigger refresh

## T = 25

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| confidence | none | 0.00 | - | 25 | 0 | 25.0 | 53.13 | 0 |
| contact_shuffled | external | 0.50 | - | 25 | 4 | 56.5 | 51.10 | 4 |
| contact | self | 0.50 | - | 25 | 0 | 25.9 | 48.84 | 4 |
| contact_shuffled | self | 0.50 | - | 25 | 0 | 25.9 | 50.16 | 4 |
| contact | external | 0.00 | - | 25 | 9 | 95.9 | 51.60 | 9 |
| contact | external | 0.25 | - | 25 | 6 | 72.3 | 50.29 | 6 |
| contact | external | 0.50 | - | 25 | 4 | 56.5 | 51.28 | 4 |
| contact | external | 0.75 | - | 25 | 2 | 40.8 | 50.03 | 2 |

Best warm-up arm is -1.54 pLDDT against the free confidence baseline, at 3.8x its cost.
The free self-attention signal is -4.29 pLDDT against the same baseline at 1.0x cost.

## T = 50

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| confidence | none | 0.00 | - | 50 | 0 | 50.0 | 57.57 | 0 |
| contact_shuffled | external | 0.50 | - | 50 | 5 | 89.4 | 52.66 | 5 |
| contact | self | 0.50 | - | 50 | 0 | 51.8 | 56.67 | 5 |
| contact_shuffled | self | 0.50 | - | 50 | 0 | 51.8 | 58.46 | 5 |
| contact | external | 0.00 | - | 50 | 9 | 120.9 | 60.99 | 9 |
| contact | external | 0.25 | - | 50 | 7 | 105.2 | 55.48 | 7 |
| contact | external | 0.50 | - | 50 | 5 | 89.4 | 55.75 | 5 |
| contact | external | 0.75 | - | 50 | 2 | 65.8 | 55.41 | 2 |

Best warm-up arm is +3.42 pLDDT against the free confidence baseline, at 2.4x its cost.
The free self-attention signal is -0.89 pLDDT against the same baseline at 1.0x cost.

## T = 100

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| confidence | none | 0.00 | - | 100 | 0 | 100.0 | 57.41 | 0 |
| contact_shuffled | external | 0.50 | - | 100 | 6 | 147.3 | 62.12 | 6 |
| contact | self | 0.50 | - | 100 | 0 | 103.7 | 57.76 | 6 |
| contact_shuffled | self | 0.50 | - | 100 | 0 | 103.7 | 62.69 | 6 |
| contact | external | 0.00 | - | 100 | 12 | 194.6 | 55.36 | 12 |
| contact | external | 0.25 | - | 100 | 9 | 170.9 | 62.15 | 9 |
| contact | external | 0.50 | - | 100 | 6 | 147.3 | 59.35 | 6 |
| contact | external | 0.75 | - | 100 | 3 | 123.6 | 58.02 | 3 |

Best warm-up arm is +4.74 pLDDT against the free confidence baseline, at 1.7x its cost.
The free self-attention signal is +0.35 pLDDT against the same baseline at 1.0x cost.

## Refresh policy

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| contact | external | 0.50 | - | 50 | 12 | 141.9 | 56.40 | 12 | <!-- eager -->
| contact | external | 0.50 | - | 50 | 5 | 89.4 | 56.99 | 5 | <!-- default -->
| contact | external | 0.50 | - | 50 | 5 | 89.4 | 50.64 | 5 | <!-- lazy -->
| contact | external | 0.50 | - | 50 | 13 | 152.5 | 56.79 | 13 | <!-- every_step -->
| contact | external | 0.50 | - | 50 | 5 | 89.4 | 57.23 | 5 | <!-- fixed_age8 -->

## Group-size cap

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| contact | external | 0.50 | 8 | 50 | 5 | 89.4 | 57.92 | 5 | <!-- g8 -->
| contact | external | 0.50 | 16 | 50 | 5 | 89.4 | 52.35 | 5 | <!-- g16 -->

## Group-size cap, baseline

| selector | signal | warm-up | cap | gen NFE | ESM NFE | cost | pLDDT | refreshes |
|---|---|---|---|---|---|---|---|---|
| confidence | none | 0.00 | 8 | 50 | 0 | 50.0 | 54.21 | 0 | <!-- g8 -->
| confidence | none | 0.00 | 16 | 50 | 0 | 50.0 | 56.77 | 0 | <!-- g16 -->

