# Warm-up inside the remasking decoder (the one DPLM ships)

Free self-attention signal, quantile threshold at 0.90, 48
sequences per cell. `gated` counts steps that fell back to
confidence because the warm-up had not been reached, so those
steps also made no contact call.

| warm-up | forward passes | confidence | contact | shuffled | contact - confidence | contact - shuffled |
|---|---|---|---|---|---|---|
| 0.0 | 25 | 62.03 | 50.28 | 52.05 | -11.76 | -1.78 |
| 0.0 | 50 | 70.74 | 53.01 | 55.08 | -17.73 | -2.07 |
| 0.0 | 100 | 75.45 | 57.42 | 66.90 | -18.03 | -9.48 |
| 0.0 | 200 | 80.75 | 54.07 | 70.17 | -26.68 | -16.11 |
| 0.35 | 25 | 62.03 | 53.33 | 58.94 | -8.71 | -5.62 |
| 0.35 | 50 | 70.74 | 57.72 | 63.99 | -13.02 | -6.27 |
| 0.35 | 100 | 75.45 | 60.69 | 66.82 | -14.77 | -6.13 |
| 0.35 | 200 | 80.75 | 66.71 | 76.06 | -14.04 | -9.35 |
| 0.5 | 25 | 62.03 | 52.65 | 57.46 | -9.39 | -4.81 |
| 0.5 | 50 | 70.74 | 61.55 | 56.87 | -9.19 | +4.67 |
| 0.5 | 100 | 75.45 | 67.18 | 72.09 | -8.27 | -4.91 |
| 0.5 | 200 | 80.75 | 72.42 | 82.22 | -8.33 | -9.80 |
| 0.75 | 25 | 62.03 | 55.38 | 59.47 | -6.66 | -4.09 |
| 0.75 | 50 | 70.74 | 68.01 | 65.54 | -2.73 | +2.47 |
| 0.75 | 100 | 75.45 | 73.48 | 72.57 | -1.97 | +0.91 |
| 0.75 | 200 | 80.75 | 80.27 | 86.35 | -0.48 | -6.07 |

Best cell for the contact rule against the free confidence baseline: -0.48 pLDDT at warm-up 0.75, 200 forward passes.

