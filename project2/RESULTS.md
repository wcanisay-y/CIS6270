# Results

All pLDDT from ESMFold on a 0-100 scale, 48 sequences per cell, length 200 unless stated. Intervals are bootstrap 95 percent.

## READ THIS FIRST: two columns in the tables below are invalid

Generated 2026-11-06. Two bugs were found on 2026-11-09, after which parts of this file are superseded. **pLDDT was never affected by either bug and every pLDDT number here stands.**

| column | status |
|---|---|
| `pTM`, `fold` (foldability) | **INVALID throughout this file.** HuggingFace ESMFold's `ptm` comes from openfold's `compute_tm`, whose final argmax runs over the whole batch, so it returns one scalar per forward call and every sequence in a fold batch received the batch *maximum*. At `--fold-batch 12` a 48-sequence cell has only 4 distinct pTM values. That is why natural proteins and generated sequences both show pTM 0.83-0.95 here while pLDDT differs by 35 points. Corrected per-sequence values are 0.38-0.46 for generated arms and 0.838 for natural, and they correlate 0.909 with pLDDT. `foldability` thresholds on pTM, so it is corrupted too. Corrected numbers live in `*_scored2.json`, in `REPORT_REVISED.md` section 7.1 and in the new sections at the end of this file. |
| contact precision / AUROC in "Contact prediction quality against mask rate" | **SUPERSEDED.** The ground-truth maps were built by taking atom14 index 3 as CB, but atom14 is ordered `N, CA, C, O, CB`, so index 3 is the backbone oxygen. Every "CB-CB contact map" in the first study was an O-O map, sharing only Jaccard 0.315 with deposited contacts where correct CB-CB indexing reaches 0.788. Replaced in the new section at the end of this file. |
| `amph`, `amphW`, `pppl`, `div`, `novel`, `rep`, `identity` | Unaffected. |

Three result sets are also retired as superseded or withdrawn: the ad-hoc 48-chain oracle set (not fold-non-redundant; replaced by CATH 4.4 S40), the fold-class effect (withdrawn, `results/G_beta_effect_note.md`), and the +26.4 conditioning headline (homology-contaminated pattern sources; re-measured at +20.2).

New sections for the 2026-11-09 and 2026-11-10 rounds are appended at the end of this file, under "Round 2" and "Round 3". `ANSWERS_TO_FEEDBACK.md` is the index to Round 3.


## Idea A: choosing the parallel unmasking set


### Harness fidelity

Published DPLM-150M at length 200 is pLDDT 84.30. Our reimplementation of its decoder returns 83.60 at 200 forward passes and 87.44 at its default 500, so the harness is faithful. Every absorbing-sampler arm sits about 30 points lower, which is the gap between irrevocable commitment and DPLM's re-predict-and-remask decoder.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |
| dplm_reparam / T=500 | 87.44 | 0.946 | 7.2 | 0.77 | 4.04 | 0.165 | 0.111 | 0.044 | 0.935 | 0.994 | 4.7 |
| dplm_reparam / T=200 | 83.60 | 0.955 | 10.2 | 0.60 | 4.77 | 0.159 | 0.101 | 0.036 | 0.937 | 0.991 | 5.8 |
| confidence / T=500 | 57.64 | 0.839 | 17.3 | 0.15 | 11.58 | 0.159 | 0.096 | 0.029 | 0.933 | 1.000 | 3.1 |
| confidence / T=200 | 57.18 | 0.847 | 17.8 | 0.10 | 12.46 | 0.146 | 0.076 | 0.010 | 0.935 | 0.999 | 3.5 |
| random / T=200 | 55.23 | 0.835 | 18.2 | 0.12 | 14.06 | 0.151 | 0.095 | 0.038 | 0.935 | 1.000 | 3.0 |
| entropy / T=200 | 53.65 | 0.806 | 19.5 | 0.08 | 13.39 | 0.151 | 0.076 | -0.005 | 0.936 | 1.000 | 3.1 |
| entropy / T=500 | 52.66 | 0.868 | 19.2 | 0.06 | 12.48 | 0.155 | 0.092 | 0.027 | 0.934 | 1.000 | 3.0 |
| random / T=500 | 51.53 | 0.828 | 19.2 | 0.04 | 14.20 | 0.152 | 0.088 | 0.025 | 0.936 | 1.000 | 2.8 |


### Matched forward passes: every selector against the baseline

pLDDT difference from `confidence`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| T=100 | contact | -0.25 | [-7.75, +7.28] | 0.9472 |  |
| T=100 | contact_shuffled | -0.50 | [-7.58, +6.73] | 0.8943 |  |
| T=100 | anti_contact | -4.36 | [-11.57, +2.94] | 0.2428 |  |
| T=100 | distance | +0.44 | [-7.20, +7.99] | 0.9062 |  |
| T=12 | contact | -5.17 | [-11.52, +1.14] | 0.1089 |  |
| T=12 | contact_shuffled | -3.79 | [-9.75, +2.13] | 0.2124 |  |
| T=12 | anti_contact | -12.11 | [-17.70, -6.59] | 0.0001 | *** |
| T=12 | distance | -2.90 | [-9.37, +3.57] | 0.3741 |  |
| T=25 | contact | -0.80 | [-6.59, +4.98] | 0.7877 |  |
| T=25 | contact_shuffled | -3.39 | [-9.48, +2.67] | 0.2750 |  |
| T=25 | anti_contact | -7.98 | [-13.40, -2.51] | 0.0043 | ** |
| T=25 | distance | +2.61 | [-3.84, +9.12] | 0.4303 |  |
| T=50 | contact | -3.28 | [-10.17, +3.49] | 0.3405 |  |
| T=50 | contact_shuffled | +3.13 | [-4.30, +10.66] | 0.4188 |  |
| T=50 | anti_contact | -4.63 | [-11.86, +2.64] | 0.2152 |  |
| T=50 | distance | +3.29 | [-4.31, +11.01] | 0.4050 |  |


### Full matched-NFE grid

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anti_contact / T=12 | 38.29 | 0.449 | 22.7 | 0.00 | 16.08 | 0.142 | 0.070 | 0.016 | 0.936 | 1.000 | 2.8 |
| anti_contact / T=25 | 43.53 | 0.604 | 21.3 | 0.02 | 16.62 | 0.149 | 0.085 | 0.015 | 0.938 | 1.000 | 2.9 |
| anti_contact / T=50 | 49.20 | 0.861 | 19.3 | 0.08 | 15.35 | 0.145 | 0.070 | 0.013 | 0.935 | 1.000 | 2.9 |
| anti_contact / T=100 | 53.26 | 0.937 | 18.2 | 0.10 | 13.81 | 0.155 | 0.093 | 0.016 | 0.935 | 1.000 | 3.1 |
| confidence / T=12 | 50.40 | 0.736 | 19.9 | 0.08 | 14.45 | 0.152 | 0.090 | 0.029 | 0.934 | 0.999 | 3.1 |
| confidence / T=25 | 51.51 | 0.754 | 19.0 | 0.04 | 13.51 | 0.153 | 0.083 | 0.018 | 0.933 | 1.000 | 2.9 |
| confidence / T=50 | 53.83 | 0.798 | 18.7 | 0.10 | 12.38 | 0.152 | 0.092 | 0.019 | 0.932 | 0.999 | 3.3 |
| confidence / T=100 | 57.61 | 0.862 | 17.4 | 0.12 | 12.52 | 0.161 | 0.104 | 0.026 | 0.933 | 1.000 | 3.1 |
| contact / T=12 | 45.24 | 0.620 | 21.3 | 0.02 | 15.58 | 0.145 | 0.076 | -0.002 | 0.934 | 1.000 | 2.9 |
| contact / T=25 | 50.71 | 0.656 | 20.6 | 0.00 | 13.30 | 0.150 | 0.079 | 0.015 | 0.932 | 1.000 | 3.4 |
| contact / T=50 | 50.55 | 0.591 | 20.2 | 0.00 | 12.65 | 0.153 | 0.084 | 0.017 | 0.933 | 1.000 | 3.0 |
| contact / T=100 | 57.36 | 0.834 | 17.2 | 0.15 | 10.57 | 0.152 | 0.082 | 0.015 | 0.936 | 0.998 | 3.6 |
| contact_shuffled / T=12 | 46.61 | 0.553 | 20.9 | 0.00 | 14.76 | 0.156 | 0.091 | 0.025 | 0.934 | 1.000 | 2.8 |
| contact_shuffled / T=25 | 48.12 | 0.735 | 20.2 | 0.04 | 14.18 | 0.144 | 0.073 | 0.014 | 0.935 | 1.000 | 3.4 |
| contact_shuffled / T=50 | 56.96 | 0.856 | 17.9 | 0.08 | 14.28 | 0.149 | 0.078 | 0.009 | 0.934 | 0.999 | 3.1 |
| contact_shuffled / T=100 | 57.11 | 0.930 | 18.1 | 0.12 | 12.18 | 0.149 | 0.077 | 0.018 | 0.934 | 0.999 | 3.2 |
| distance / T=12 | 47.51 | 0.717 | 20.0 | 0.02 | 13.02 | 0.151 | 0.089 | 0.029 | 0.934 | 1.000 | 2.8 |
| distance / T=25 | 54.12 | 0.891 | 18.4 | 0.12 | 14.22 | 0.149 | 0.077 | 0.005 | 0.932 | 1.000 | 3.1 |
| distance / T=50 | 57.12 | 0.826 | 17.5 | 0.15 | 12.80 | 0.151 | 0.092 | 0.026 | 0.935 | 1.000 | 3.0 |
| distance / T=100 | 58.05 | 0.826 | 17.1 | 0.17 | 14.86 | 0.150 | 0.075 | -0.016 | 0.936 | 0.999 | 3.1 |
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.50 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### The rule inside DPLM's own decoder

Same forward passes and the same number of survivors per step; only which positions survive changes. This is the decoder DPLM ships and where its published quality comes from.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |
| reparam_anti_contact / T=25 | 53.08 | 0.523 | 23.7 | 0.02 | 12.51 | 0.154 | 0.100 | 0.042 | 0.934 | 0.999 | 4.1 |
| reparam_anti_contact / T=50 | 57.22 | 0.786 | 22.6 | 0.06 | 8.62 | 0.156 | 0.102 | 0.036 | 0.933 | 0.996 | 6.5 |
| reparam_anti_contact / T=100 | 55.24 | 0.751 | 23.0 | 0.04 | 7.74 | 0.137 | 0.067 | 0.011 | 0.939 | 0.994 | 8.7 |
| reparam_anti_contact / T=200 | 60.03 | 0.652 | 22.9 | 0.02 | 5.14 | 0.145 | 0.087 | 0.012 | 0.938 | 0.983 | 10.7 |
| reparam_confidence / T=25 | 58.10 | 0.916 | 20.5 | 0.15 | 10.92 | 0.147 | 0.077 | 0.025 | 0.937 | 0.994 | 5.4 |
| reparam_confidence / T=50 | 65.62 | 0.923 | 17.8 | 0.19 | 6.07 | 0.146 | 0.099 | 0.029 | 0.936 | 0.993 | 6.3 |
| reparam_confidence / T=100 | 78.23 | 0.948 | 13.2 | 0.46 | 5.14 | 0.147 | 0.074 | 0.011 | 0.936 | 0.995 | 5.6 |
| reparam_confidence / T=200 | 80.24 | 0.946 | 11.6 | 0.52 | 4.06 | 0.165 | 0.147 | 0.072 | 0.938 | 0.992 | 4.2 |
| reparam_contact / T=25 | 54.57 | 0.804 | 21.6 | 0.12 | 11.68 | 0.142 | 0.064 | -0.006 | 0.935 | 0.997 | 4.8 |
| reparam_contact / T=50 | 58.71 | 0.891 | 22.3 | 0.10 | 8.29 | 0.146 | 0.074 | 0.000 | 0.933 | 0.989 | 7.7 |
| reparam_contact / T=100 | 63.32 | 0.919 | 19.2 | 0.25 | 8.63 | 0.150 | 0.099 | 0.044 | 0.939 | 0.992 | 6.5 |
| reparam_contact / T=200 | 66.45 | 0.918 | 17.5 | 0.33 | 6.38 | 0.159 | 0.115 | 0.049 | 0.940 | 0.986 | 9.9 |
| reparam_contact_shuffled / T=25 | 54.74 | 0.713 | 21.6 | 0.06 | 11.33 | 0.153 | 0.102 | 0.027 | 0.932 | 0.997 | 4.8 |
| reparam_contact_shuffled / T=50 | 62.23 | 0.902 | 19.0 | 0.19 | 8.56 | 0.157 | 0.104 | 0.036 | 0.935 | 0.996 | 5.5 |
| reparam_contact_shuffled / T=100 | 72.61 | 0.939 | 16.0 | 0.35 | 6.38 | 0.149 | 0.076 | 0.024 | 0.939 | 0.992 | 7.2 |
| reparam_contact_shuffled / T=200 | 79.48 | 0.945 | 10.9 | 0.56 | 5.07 | 0.153 | 0.092 | 0.024 | 0.938 | 0.988 | 6.4 |


### The rule inside DPLM's own decoder: contrasts

pLDDT difference from `reparam_confidence`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| T=100 | reparam_contact | -14.90 | [-22.50, -7.25] | 0.0000 | *** |
| T=100 | reparam_contact_shuffled | -5.62 | [-13.11, +1.89] | 0.1457 |  |
| T=100 | reparam_anti_contact | -22.98 | [-29.71, -16.30] | 0.0000 | *** |
| T=200 | reparam_contact | -13.79 | [-21.71, -5.25] | 0.0016 | ** |
| T=200 | reparam_contact_shuffled | -0.77 | [-7.98, +6.53] | 0.8369 |  |
| T=200 | reparam_anti_contact | -20.21 | [-27.00, -12.95] | 0.0000 | *** |
| T=25 | reparam_contact | -3.54 | [-11.28, +4.14] | 0.3695 |  |
| T=25 | reparam_contact_shuffled | -3.37 | [-10.67, +4.06] | 0.3752 |  |
| T=25 | reparam_anti_contact | -5.03 | [-12.14, +1.69] | 0.1546 |  |
| T=50 | reparam_contact | -6.91 | [-14.30, +0.55] | 0.0691 |  |
| T=50 | reparam_contact_shuffled | -3.39 | [-10.94, +4.27] | 0.3923 |  |
| T=50 | reparam_anti_contact | -8.40 | [-15.73, -0.95] | 0.0279 | * |


### Self-oracle

The baseline's own output is folded and its true contact map handed back for a second pass from the same seed.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anti_contact_oracle_self / T=12 | 44.95 | 0.529 | 20.9 | 0.00 | 15.62 | 0.151 | 0.084 | 0.019 | 0.934 | 1.000 | 2.8 |
| anti_contact_oracle_self / T=25 | 53.50 | 0.713 | 18.6 | 0.08 | 13.21 | 0.151 | 0.088 | 0.009 | 0.933 | 1.000 | 2.9 |
| anti_contact_oracle_self / T=50 | 56.46 | 0.877 | 17.3 | 0.12 | 11.87 | 0.147 | 0.078 | 0.009 | 0.934 | 0.999 | 3.1 |
| confidence / T=12 | 46.39 | 0.639 | 20.5 | 0.02 | 13.68 | 0.147 | 0.088 | 0.027 | 0.934 | 1.000 | 3.1 |
| confidence / T=25 | 55.79 | 0.827 | 17.9 | 0.08 | 13.06 | 0.157 | 0.098 | 0.041 | 0.933 | 1.000 | 2.9 |
| confidence / T=50 | 55.31 | 0.874 | 17.5 | 0.10 | 12.97 | 0.153 | 0.091 | 0.015 | 0.933 | 0.999 | 2.9 |
| contact_oracle_self / T=12 | 44.44 | 0.550 | 21.5 | 0.00 | 14.28 | 0.146 | 0.076 | 0.009 | 0.934 | 1.000 | 3.0 |
| contact_oracle_self / T=25 | 55.81 | 0.756 | 18.3 | 0.08 | 12.93 | 0.159 | 0.102 | 0.042 | 0.933 | 1.000 | 3.0 |
| contact_oracle_self / T=50 | 54.84 | 0.880 | 18.2 | 0.15 | 13.59 | 0.152 | 0.083 | 0.006 | 0.933 | 0.999 | 2.9 |
| contact_oracle_shuffled_self / T=12 | 45.54 | 0.701 | 20.8 | 0.02 | 13.47 | 0.146 | 0.074 | 0.015 | 0.936 | 1.000 | 2.8 |
| contact_oracle_shuffled_self / T=25 | 48.09 | 0.823 | 19.7 | 0.08 | 14.04 | 0.151 | 0.091 | 0.027 | 0.936 | 1.000 | 3.0 |
| contact_oracle_shuffled_self / T=50 | 57.10 | 0.866 | 17.4 | 0.10 | 13.05 | 0.154 | 0.101 | 0.034 | 0.934 | 1.000 | 3.1 |
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.50 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### Self-oracle: contrasts

pLDDT difference from `confidence`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| T=12 | contact_oracle_self | -1.95 | [-7.15, +3.03] | 0.4429 |  |
| T=12 | contact_oracle_shuffled_self | -0.85 | [-6.19, +4.61] | 0.7536 |  |
| T=12 | anti_contact_oracle_self | -1.44 | [-6.90, +3.73] | 0.5959 |  |
| T=25 | contact_oracle_self | +0.02 | [-7.49, +7.26] | 0.9930 |  |
| T=25 | contact_oracle_shuffled_self | -7.70 | [-14.43, -0.78] | 0.0267 | * |
| T=25 | anti_contact_oracle_self | -2.29 | [-9.00, +4.50] | 0.5086 |  |
| T=50 | contact_oracle_self | -0.47 | [-8.05, +7.17] | 0.9000 |  |
| T=50 | contact_oracle_shuffled_self | +1.80 | [-5.77, +9.30] | 0.6358 |  |
| T=50 | anti_contact_oracle_self | +1.15 | [-6.16, +8.64] | 0.7669 |  |


### Commit ranking: clean probability

Repeats the absorbing arms ranking commits by the clean probability of the sampled token instead of DPLM's Gumbel-perturbed score, to check the baseline is not a strawman.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| confidence / T=25 | 73.86 | 0.630 | 17.1 | 0.00 | 1.27 | 0.094 | 0.002 | -0.041 | 0.283 | 0.734 | 130.9 |
| confidence / T=200 | 76.01 | 0.469 | 20.5 | 0.00 | 1.16 | 0.085 | 0.000 | -0.060 | 0.365 | 0.642 | 143.7 |
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.50 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |
| random / T=25 | 47.44 | 0.747 | 20.4 | 0.06 | 15.14 | 0.150 | 0.081 | 0.011 | 0.936 | 1.000 | 3.2 |
| random / T=200 | 55.40 | 0.825 | 17.6 | 0.17 | 11.38 | 0.153 | 0.084 | 0.027 | 0.934 | 1.000 | 2.8 |


### Commit ranking: clean probability: contrasts

pLDDT difference from `confidence`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| T=200 | random | -20.61 | [-27.92, -12.91] | 0.0000 | *** |
| T=25 | random | -26.42 | [-30.96, -21.63] | 0.0000 | *** |


### Oracle: true contact maps

The selector is handed a real protein's true contact map. If a correct map does not beat confidence ordering, no improvement in contact prediction can rescue the mechanism.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anti_contact_oracle / T=6 | 38.33 | 0.515 | 22.2 | 0.00 | 16.49 | 0.148 | 0.077 | 0.013 | 0.935 | 1.000 | 2.6 |
| anti_contact_oracle / T=12 | 42.14 | 0.485 | 22.1 | 0.00 | 16.06 | 0.143 | 0.073 | 0.008 | 0.934 | 1.000 | 3.0 |
| anti_contact_oracle / T=25 | 47.04 | 0.578 | 21.3 | 0.02 | 14.37 | 0.144 | 0.083 | 0.023 | 0.936 | 1.000 | 2.8 |
| anti_contact_oracle / T=50 | 52.89 | 0.789 | 20.0 | 0.02 | 13.67 | 0.153 | 0.093 | 0.027 | 0.934 | 1.000 | 3.1 |
| contact_oracle / T=6 | 41.29 | 0.434 | 22.3 | 0.00 | 15.91 | 0.151 | 0.094 | 0.022 | 0.935 | 1.000 | 2.8 |
| contact_oracle / T=12 | 45.44 | 0.668 | 20.9 | 0.02 | 15.91 | 0.145 | 0.086 | 0.014 | 0.936 | 1.000 | 2.7 |
| contact_oracle / T=25 | 51.27 | 0.726 | 19.6 | 0.04 | 12.39 | 0.158 | 0.100 | 0.039 | 0.931 | 0.999 | 3.0 |
| contact_oracle / T=50 | 55.54 | 0.837 | 18.2 | 0.12 | 12.52 | 0.153 | 0.096 | 0.036 | 0.933 | 1.000 | 2.9 |
| contact_oracle_shuffled / T=6 | 37.47 | 0.455 | 22.8 | 0.00 | 16.16 | 0.138 | 0.063 | -0.001 | 0.937 | 1.000 | 2.7 |
| contact_oracle_shuffled / T=12 | 45.02 | 0.767 | 21.1 | 0.06 | 15.16 | 0.148 | 0.081 | 0.016 | 0.935 | 1.000 | 3.0 |
| contact_oracle_shuffled / T=25 | 50.93 | 0.778 | 19.4 | 0.06 | 12.71 | 0.148 | 0.088 | 0.019 | 0.935 | 0.999 | 2.8 |
| contact_oracle_shuffled / T=50 | 59.66 | 0.895 | 17.1 | 0.15 | 12.70 | 0.152 | 0.083 | 0.005 | 0.934 | 0.998 | 4.0 |
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.50 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### Self-oracle

The baseline's own output is folded and its true contact map is handed back for a second pass from the same seed.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anti_contact_oracle_self / T=12 | 44.95 | 0.529 | 20.9 | 0.00 | 15.62 | 0.151 | 0.084 | 0.019 | 0.934 | 1.000 | 2.8 |
| anti_contact_oracle_self / T=25 | 53.50 | 0.713 | 18.6 | 0.08 | 13.21 | 0.151 | 0.088 | 0.009 | 0.933 | 1.000 | 2.9 |
| anti_contact_oracle_self / T=50 | 56.46 | 0.877 | 17.3 | 0.12 | 11.87 | 0.147 | 0.078 | 0.009 | 0.934 | 0.999 | 3.1 |
| confidence / T=12 | 46.39 | 0.639 | 20.5 | 0.02 | 13.68 | 0.147 | 0.088 | 0.027 | 0.934 | 1.000 | 3.1 |
| confidence / T=25 | 55.79 | 0.827 | 17.9 | 0.08 | 13.06 | 0.157 | 0.098 | 0.041 | 0.933 | 1.000 | 2.9 |
| confidence / T=50 | 55.31 | 0.874 | 17.5 | 0.10 | 12.97 | 0.153 | 0.091 | 0.015 | 0.933 | 0.999 | 2.9 |
| contact_oracle_self / T=12 | 44.44 | 0.550 | 21.5 | 0.00 | 14.28 | 0.146 | 0.076 | 0.009 | 0.934 | 1.000 | 3.0 |
| contact_oracle_self / T=25 | 55.81 | 0.756 | 18.3 | 0.08 | 12.93 | 0.159 | 0.102 | 0.042 | 0.933 | 1.000 | 3.0 |
| contact_oracle_self / T=50 | 54.84 | 0.880 | 18.2 | 0.15 | 13.59 | 0.152 | 0.083 | 0.006 | 0.933 | 0.999 | 2.9 |
| contact_oracle_shuffled_self / T=12 | 45.54 | 0.701 | 20.8 | 0.02 | 13.47 | 0.146 | 0.074 | 0.015 | 0.936 | 1.000 | 2.8 |
| contact_oracle_shuffled_self / T=25 | 48.09 | 0.823 | 19.7 | 0.08 | 14.04 | 0.151 | 0.091 | 0.027 | 0.936 | 1.000 | 3.0 |
| contact_oracle_shuffled_self / T=50 | 57.10 | 0.866 | 17.4 | 0.10 | 13.05 | 0.154 | 0.101 | 0.034 | 0.934 | 1.000 | 3.1 |
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.50 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### Self-oracle: contrasts

pLDDT difference from `confidence`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| T=12 | contact_oracle_self | -1.95 | [-7.15, +3.03] | 0.4429 |  |
| T=12 | contact_oracle_shuffled_self | -0.85 | [-6.19, +4.61] | 0.7536 |  |
| T=12 | anti_contact_oracle_self | -1.44 | [-6.90, +3.73] | 0.5959 |  |
| T=25 | contact_oracle_self | +0.02 | [-7.49, +7.26] | 0.9930 |  |
| T=25 | contact_oracle_shuffled_self | -7.70 | [-14.43, -0.78] | 0.0267 | * |
| T=25 | anti_contact_oracle_self | -2.29 | [-9.00, +4.50] | 0.5086 |  |
| T=50 | contact_oracle_self | -0.47 | [-8.05, +7.17] | 0.9000 |  |
| T=50 | contact_oracle_shuffled_self | +1.80 | [-5.77, +9.30] | 0.6358 |  |
| T=50 | anti_contact_oracle_self | +1.15 | [-6.16, +8.64] | 0.7669 |  |


### Stronger threshold: top 10 percent of pairs

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| contact / T=12 | 44.38 | 0.618 | 20.6 | 0.00 | 15.19 | 0.149 | 0.077 | 0.000 | 0.935 | 1.000 | 2.8 |
| contact / T=25 | 54.38 | 0.717 | 18.2 | 0.10 | 14.19 | 0.152 | 0.085 | 0.032 | 0.931 | 0.999 | 2.8 |
| contact / T=50 | 56.61 | 0.954 | 17.5 | 0.19 | 12.74 | 0.147 | 0.076 | 0.022 | 0.934 | 1.000 | 3.0 |
| contact_shuffled / T=12 | 44.47 | 0.628 | 21.3 | 0.04 | 14.65 | 0.151 | 0.081 | 0.008 | 0.934 | 1.000 | 3.0 |
| contact_shuffled / T=25 | 54.15 | 0.778 | 18.4 | 0.08 | 12.60 | 0.152 | 0.089 | 0.027 | 0.933 | 1.000 | 2.9 |
| contact_shuffled / T=50 | 56.58 | 0.808 | 17.1 | 0.10 | 13.05 | 0.153 | 0.089 | 0.028 | 0.934 | 1.000 | 3.1 |
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.50 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### The proposal as written: a separate ESM-2 650M pass per step

Doubles the per-step cost.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| contact / T=12 | 46.52 | 0.810 | 20.4 | 0.02 | 14.15 | 0.153 | 0.088 | 0.027 | 0.933 | 1.000 | 2.9 |
| contact / T=25 | 54.24 | 0.830 | 18.1 | 0.08 | 14.28 | 0.148 | 0.072 | 0.017 | 0.932 | 0.999 | 2.9 |
| contact / T=50 | 55.12 | 0.902 | 17.7 | 0.12 | 11.98 | 0.150 | 0.088 | 0.030 | 0.936 | 1.000 | 2.7 |
| contact_shuffled / T=12 | 43.13 | 0.572 | 21.2 | 0.02 | 15.98 | 0.147 | 0.086 | 0.030 | 0.935 | 1.000 | 2.8 |
| contact_shuffled / T=25 | 54.01 | 0.776 | 18.2 | 0.08 | 12.43 | 0.145 | 0.089 | 0.025 | 0.935 | 1.000 | 2.9 |
| contact_shuffled / T=50 | 54.18 | 0.919 | 18.0 | 0.15 | 13.92 | 0.151 | 0.076 | 0.014 | 0.934 | 0.998 | 3.4 |
| natural / T=None | 90.10 | 0.954 | 5.7 | 0.81 | 3.50 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### Secondary record: the non-binding threshold

A fixed coupling threshold of 0.3 rejected 0.05 percent of pairs, so the rule barely bound. Kept only to document that.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anti_contact / T=6 | 34.50 | 0.346 | 23.0 | 0.00 | - | - | 0.068 | 0.006 | - | - | - |
| anti_contact / T=12 | 36.66 | 0.329 | 23.2 | 0.00 | - | - | 0.081 | 0.015 | - | - | - |
| anti_contact / T=25 | - | - | - | - | - | - | 0.078 | 0.007 | - | - | - |
| anti_contact / T=50 | - | - | - | - | - | - | 0.087 | 0.021 | - | - | - |
| anti_contact / T=100 | - | - | - | - | - | - | 0.084 | 0.025 | - | - | - |
| anti_contact / T=200 | - | - | - | - | - | - | 0.086 | 0.020 | - | - | - |
| confidence / T=6 | 42.74 | 0.600 | 21.3 | 0.00 | - | - | 0.078 | 0.003 | - | - | - |
| confidence / T=12 | 46.69 | 0.701 | 21.1 | 0.02 | - | - | 0.079 | 0.005 | - | - | - |
| confidence / T=25 | - | - | - | - | - | - | 0.083 | 0.007 | - | - | - |
| confidence / T=50 | - | - | - | - | - | - | 0.086 | 0.009 | - | - | - |
| confidence / T=100 | - | - | - | - | - | - | 0.081 | 0.009 | - | - | - |
| confidence / T=200 | - | - | - | - | - | - | 0.089 | 0.033 | - | - | - |
| contact / T=6 | 40.95 | 0.597 | 21.8 | 0.00 | - | - | 0.073 | -0.007 | - | - | - |
| contact / T=12 | 42.31 | 0.699 | 21.4 | 0.04 | - | - | 0.080 | 0.017 | - | - | - |
| contact / T=25 | - | - | - | - | - | - | 0.080 | 0.014 | - | - | - |
| contact / T=25 | - | - | - | - | - | - | 0.094 | 0.030 | - | - | - |
| contact / T=25 | - | - | - | - | - | - | 0.074 | 0.005 | - | - | - |
| contact / T=25 | - | - | - | - | - | - | 0.093 | 0.025 | - | - | - |
| contact / T=50 | - | - | - | - | - | - | 0.093 | 0.037 | - | - | - |
| contact / T=100 | - | - | - | - | - | - | 0.091 | 0.031 | - | - | - |
| contact / T=200 | - | - | - | - | - | - | 0.084 | 0.016 | - | - | - |
| contact_degree / T=6 | 36.00 | 0.437 | 22.9 | 0.00 | - | - | 0.072 | 0.023 | - | - | - |
| contact_degree / T=12 | 38.16 | 0.430 | 22.5 | 0.00 | - | - | 0.094 | 0.029 | - | - | - |
| contact_degree / T=25 | - | - | - | - | - | - | 0.065 | 0.008 | - | - | - |
| contact_degree / T=50 | - | - | - | - | - | - | 0.084 | 0.017 | - | - | - |
| contact_degree / T=100 | - | - | - | - | - | - | 0.081 | 0.023 | - | - | - |
| contact_degree / T=200 | - | - | - | - | - | - | 0.092 | 0.032 | - | - | - |
| contact_shuffled / T=6 | 40.48 | 0.525 | 22.0 | 0.00 | - | - | 0.074 | 0.009 | - | - | - |
| contact_shuffled / T=12 | 48.10 | 0.740 | 19.5 | 0.04 | - | - | 0.081 | 0.013 | - | - | - |
| contact_shuffled / T=25 | - | - | - | - | - | - | 0.099 | 0.039 | - | - | - |
| contact_shuffled / T=50 | - | - | - | - | - | - | 0.081 | 0.017 | - | - | - |
| contact_shuffled / T=100 | - | - | - | - | - | - | 0.072 | -0.000 | - | - | - |
| contact_shuffled / T=200 | - | - | - | - | - | - | 0.070 | 0.006 | - | - | - |
| distance / T=6 | 41.02 | 0.589 | 21.7 | 0.00 | - | - | 0.095 | 0.039 | - | - | - |
| distance / T=12 | 52.17 | 0.677 | 19.3 | 0.04 | - | - | 0.088 | 0.029 | - | - | - |
| distance / T=25 | - | - | - | - | - | - | 0.085 | 0.023 | - | - | - |
| distance / T=50 | - | - | - | - | - | - | 0.088 | 0.035 | - | - | - |
| distance / T=100 | - | - | - | - | - | - | 0.087 | 0.007 | - | - | - |
| distance / T=200 | - | - | - | - | - | - | 0.073 | 0.000 | - | - | - |
| entropy / T=6 | 34.09 | 0.284 | 23.2 | 0.00 | - | - | 0.074 | 0.012 | - | - | - |
| entropy / T=12 | 36.97 | 0.310 | 23.1 | 0.00 | - | - | 0.075 | 0.018 | - | - | - |
| entropy / T=25 | - | - | - | - | - | - | 0.092 | 0.026 | - | - | - |
| entropy / T=50 | - | - | - | - | - | - | 0.076 | 0.006 | - | - | - |
| entropy / T=100 | - | - | - | - | - | - | 0.089 | 0.012 | - | - | - |
| entropy / T=200 | - | - | - | - | - | - | 0.098 | 0.032 | - | - | - |
| natural / T=None | - | - | - | - | - | - | 0.078 | 0.017 | - | - | - |
| punt / T=6 | 37.93 | 0.421 | 22.5 | 0.00 | - | - | 0.074 | 0.019 | - | - | - |
| punt / T=12 | 40.25 | 0.614 | 22.3 | 0.02 | - | - | 0.084 | 0.021 | - | - | - |
| punt / T=25 | - | - | - | - | - | - | 0.090 | 0.026 | - | - | - |
| punt / T=50 | - | - | - | - | - | - | 0.095 | 0.030 | - | - | - |
| punt / T=100 | - | - | - | - | - | - | 0.099 | 0.037 | - | - | - |
| punt / T=200 | - | - | - | - | - | - | 0.074 | 0.002 | - | - | - |
| random / T=6 | 39.32 | 0.470 | 22.2 | 0.00 | - | - | 0.091 | 0.030 | - | - | - |
| random / T=12 | 44.54 | 0.595 | 21.6 | 0.02 | - | - | 0.086 | 0.014 | - | - | - |
| random / T=25 | - | - | - | - | - | - | 0.082 | 0.027 | - | - | - |
| random / T=50 | - | - | - | - | - | - | 0.066 | 0.007 | - | - | - |
| random / T=100 | - | - | - | - | - | - | 0.072 | 0.015 | - | - | - |
| random / T=200 | - | - | - | - | - | - | 0.072 | 0.023 | - | - | - |


### Secondary record: the non-binding threshold: contrasts

pLDDT difference from `confidence`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| T=12 | random | -2.15 | [-7.33, +3.12] | 0.4208 |  |
| T=12 | entropy | -9.72 | [-13.98, -5.80] | 0.0000 | *** |
| T=12 | contact | -4.38 | [-9.72, +1.16] | 0.1121 |  |
| T=12 | contact_shuffled | +1.41 | [-4.34, +7.08] | 0.6164 |  |
| T=12 | anti_contact | -10.03 | [-14.44, -5.83] | 0.0000 | *** |
| T=12 | contact_degree | -8.53 | [-13.36, -3.70] | 0.0006 | *** |
| T=12 | distance | +5.49 | [-0.47, +11.51] | 0.0694 |  |
| T=12 | punt | -6.44 | [-11.40, -1.38] | 0.0116 | * |
| T=6 | random | -3.42 | [-7.87, +1.01] | 0.1315 |  |
| T=6 | entropy | -8.65 | [-12.61, -4.97] | 0.0000 | *** |
| T=6 | contact | -1.79 | [-6.73, +3.30] | 0.4788 |  |
| T=6 | contact_shuffled | -2.27 | [-6.92, +2.35] | 0.3336 |  |
| T=6 | anti_contact | -8.25 | [-12.15, -4.61] | 0.0001 | *** |
| T=6 | contact_degree | -6.75 | [-10.74, -2.95] | 0.0011 | ** |
| T=6 | distance | -1.73 | [-6.63, +3.07] | 0.4874 |  |
| T=6 | punt | -4.81 | [-9.09, -0.71] | 0.0254 | * |


### Contact prediction quality against mask rate — SUPERSEDED, see Round 2

Scored against maps built with the atom14 indexing bug, i.e. against an O-O contact map rather than CB-CB. Kept only so that no analysis silently reuses it. Replaced by "Round 2: contact accuracy against deposited structures" at the end of this file.

| input | AUROC ext | P@L ext | AUROC self | P@L self |
|---|---|---|---|---|
| 0% masked | 0.914 | 0.305 | 0.886 | 0.255 |
| 25% masked | 0.700 | 0.027 | 0.879 | 0.247 |
| 50% masked | 0.669 | 0.029 | 0.849 | 0.212 |
| 75% masked | 0.662 | 0.025 | 0.695 | 0.076 |
| 90% masked | 0.643 | 0.020 | 0.701 | 0.035 |
| 100% masked | 0.626 | 0.032 | 0.698 | 0.030 |
| all-mask null | 0.626 | 0.032 | 0.698 | 0.030 |


## Idea B: the two-level alphabet


### Every arm at one matched setting

Matched parameters, corpus, training budget and forward passes. `rand6` is the control that decides the idea: a random partition with the same group sizes as the chemical one.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dms6 T=100 | 36.68 | 0.279 | 22.4 | 0.00 | 14.86 | 0.142 | 0.059 | -0.003 | 0.922 | 1.000 | 3.2 |
| flat T=100 | 43.09 | 0.705 | 20.2 | 0.19 | 13.69 | 0.146 | 0.083 | 0.021 | 0.929 | 0.952 | 2.5 |
| natural | 90.12 | 0.950 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |
| rand6 T=100 | 35.41 | 0.252 | 23.3 | 0.00 | 16.37 | 0.139 | 0.061 | -0.014 | 0.928 | 1.000 | 2.9 |


### Every arm at one matched setting: contrasts against flat

pLDDT difference from `flat`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| 100 | dms6 | -6.41 | [-12.98, -0.38] | 0.0460 | * |
| 100 | rand6 | -7.68 | [-14.37, -1.73] | 0.0166 | * |


### Against forward passes

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dms6 T=100 | 36.68 | 0.279 | 22.4 | 0.00 | 14.86 | 0.142 | 0.059 | -0.003 | 0.922 | 1.000 | 3.2 |
| dms6 T=25 | 35.34 | 0.247 | 22.6 | 0.00 | 16.49 | 0.143 | 0.073 | 0.002 | 0.924 | 1.000 | 2.9 |
| dms6 T=50 | 35.79 | 0.285 | 22.8 | 0.00 | 15.88 | 0.143 | 0.062 | 0.010 | 0.921 | 1.000 | 3.1 |
| dms6 T=200 | 35.98 | 0.273 | 22.8 | 0.00 | 16.13 | 0.146 | 0.068 | -0.010 | 0.924 | 1.000 | 3.0 |
| flat T=100 | 43.09 | 0.705 | 20.2 | 0.19 | 13.69 | 0.146 | 0.083 | 0.021 | 0.929 | 0.952 | 2.5 |
| flat T=25 | 37.15 | 0.565 | 22.1 | 0.02 | 15.78 | 0.139 | 0.062 | 0.000 | 0.931 | 0.992 | 2.8 |
| flat T=50 | 40.38 | 0.551 | 21.3 | 0.06 | 17.70 | 0.148 | 0.083 | 0.020 | 0.929 | 0.994 | 2.9 |
| flat T=200 | 43.88 | 0.829 | 20.3 | 0.12 | 15.77 | 0.146 | 0.077 | 0.019 | 0.925 | 0.953 | 2.7 |
| natural | 90.12 | 0.950 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |
| rand6 T=100 | 35.41 | 0.252 | 23.3 | 0.00 | 16.37 | 0.139 | 0.061 | -0.014 | 0.928 | 1.000 | 2.9 |
| rand6 T=25 | 34.16 | 0.250 | 23.4 | 0.00 | 17.07 | 0.140 | 0.064 | -0.000 | 0.928 | 1.000 | 2.7 |
| rand6 T=50 | 33.85 | 0.245 | 23.2 | 0.00 | 16.42 | 0.147 | 0.077 | 0.010 | 0.928 | 1.000 | 3.0 |
| rand6 T=200 | 35.65 | 0.268 | 23.1 | 0.00 | 16.90 | 0.137 | 0.056 | -0.009 | 0.927 | 1.000 | 2.8 |


### Against forward passes: contrasts against flat

pLDDT difference from `flat`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| 100 | dms6 | -6.41 | [-12.98, -0.38] | 0.0460 | * |
| 100 | rand6 | -7.68 | [-14.37, -1.73] | 0.0166 | * |
| 200 | dms6 | -7.90 | [-14.27, -2.10] | 0.0111 | * |
| 200 | rand6 | -8.23 | [-14.66, -2.23] | 0.0090 | ** |
| 25 | dms6 | -1.81 | [-6.22, +2.00] | 0.3849 |  |
| 25 | rand6 | -2.99 | [-7.52, +1.04] | 0.1639 |  |
| 50 | dms6 | -4.58 | [-9.50, -0.13] | 0.0541 |  |
| 50 | rand6 | -6.53 | [-11.38, -2.24] | 0.0064 | ** |


### Flat arm, temperature sweep

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flat t=0.6 T=100 | 59.24 | 0.831 | 15.7 | 0.31 | 9.78 | 0.146 | 0.074 | 0.025 | 0.876 | 0.857 | 3.6 |
| flat t=0.7 T=100 | 50.42 | 0.753 | 18.8 | 0.17 | 8.24 | 0.146 | 0.083 | 0.026 | 0.903 | 0.929 | 3.2 |
| flat t=0.8 T=100 | 43.82 | 0.637 | 20.3 | 0.10 | 14.61 | 0.136 | 0.060 | 0.001 | 0.918 | 0.954 | 3.2 |
| flat t=0.9 T=100 | 44.89 | 0.671 | 20.1 | 0.08 | 12.64 | 0.144 | 0.073 | 0.023 | 0.922 | 0.953 | 2.8 |
| flat T=100 | 42.02 | 0.737 | 21.0 | 0.12 | 17.30 | 0.151 | 0.095 | 0.029 | 0.928 | 0.973 | 2.9 |
| flat t=1.1 T=100 | 35.88 | 0.491 | 22.5 | 0.04 | 15.78 | 0.141 | 0.059 | 0.002 | 0.934 | 0.993 | 2.6 |
| natural | 90.12 | 0.950 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |



### Two-level arm, family x identity sweep

The claim is that two temperatures move the whole frontier outward, so this is compared with the flat curve above.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dms6 t=0.7/0.7 T=100 | 46.34 | 0.328 | 20.4 | 0.00 | 10.69 | 0.141 | 0.056 | 0.003 | 0.864 | 0.998 | 4.5 |
| dms6 t=0.85/0.7 T=100 | 44.93 | 0.335 | 20.5 | 0.00 | 13.35 | 0.137 | 0.051 | 0.002 | 0.885 | 1.000 | 3.6 |
| dms6/0.7 T=100 | 44.90 | 0.340 | 20.4 | 0.00 | 14.27 | 0.141 | 0.056 | 0.005 | 0.898 | 1.000 | 3.6 |
| dms6 t=1.15/0.7 T=100 | 44.10 | 0.332 | 20.5 | 0.00 | 14.02 | 0.137 | 0.054 | -0.005 | 0.900 | 1.000 | 3.4 |
| dms6 t=0.7/0.85 T=100 | 42.93 | 0.352 | 21.1 | 0.00 | 12.31 | 0.137 | 0.054 | -0.010 | 0.892 | 1.000 | 4.2 |
| dms6 t=0.85/0.85 T=100 | 40.29 | 0.271 | 21.9 | 0.00 | 14.23 | 0.144 | 0.071 | 0.012 | 0.906 | 0.999 | 3.5 |
| dms6/0.85 T=100 | 40.23 | 0.313 | 21.4 | 0.00 | 15.00 | 0.143 | 0.071 | 0.012 | 0.913 | 1.000 | 3.3 |
| dms6 t=1.15/0.85 T=100 | 40.15 | 0.302 | 21.6 | 0.00 | 15.95 | 0.147 | 0.072 | 0.021 | 0.918 | 1.000 | 3.2 |
| dms6 t=0.7 T=100 | 39.65 | 0.279 | 22.1 | 0.00 | 14.81 | 0.140 | 0.064 | 0.010 | 0.915 | 1.000 | 3.7 |
| dms6 t=0.85 T=100 | 38.57 | 0.269 | 22.3 | 0.00 | 13.71 | 0.148 | 0.074 | 0.020 | 0.915 | 1.000 | 3.3 |
| dms6 T=100 | 39.21 | 0.314 | 22.3 | 0.00 | 15.78 | 0.145 | 0.076 | 0.010 | 0.920 | 1.000 | 3.3 |
| dms6 t=1.15 T=100 | 35.68 | 0.254 | 22.8 | 0.00 | 16.12 | 0.146 | 0.073 | 0.003 | 0.927 | 1.000 | 2.8 |
| natural | 90.12 | 0.950 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### Out-of-distribution two-phase sampler

Writes every family before any identity, a state the training corruption never produces.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dms6 T=100 | 37.19 | 0.271 | 22.4 | 0.00 | 15.19 | 0.137 | 0.048 | -0.025 | 0.921 | 1.000 | 2.9 |
| dms6 T=100 | 36.08 | 0.301 | 22.4 | 0.00 | 15.55 | 0.144 | 0.070 | -0.004 | 0.923 | 1.000 | 3.0 |
| dms6 T=100 | 36.66 | 0.266 | 22.6 | 0.00 | 15.91 | 0.146 | 0.072 | 0.010 | 0.920 | 1.000 | 3.0 |
| dms6 T=100 | 35.76 | 0.259 | 22.7 | 0.00 | 15.31 | 0.143 | 0.071 | 0.001 | 0.921 | 1.000 | 3.1 |
| natural | 90.12 | 0.950 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### Collapsed run, kept as a warning

Ranking commits by clean probability produced homopolymer runs of 109 to 113 residues out of 200 at diversity 0.08, while scoring pLDDT 84.6 for the flat arm. Not a result, an artefact.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dms6 T=100 | 34.23 | 0.196 | 25.5 | 0.00 | 1.40 | 0.124 | 0.004 | -0.002 | 0.078 | 0.718 | 109.0 |
| flat T=100 | 84.62 | 0.822 | 7.8 | 0.96 | 2.79 | 0.169 | 0.118 | 0.065 | 0.323 | 0.136 | 2.3 |
| natural | 90.12 | 0.950 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |
| rand6 T=100 | 32.88 | 0.188 | 26.0 | 0.00 | 1.15 | 0.117 | 0.001 | -0.001 | 0.130 | 0.426 | 113.4 |


### Collapsed run, kept as a warning: contrasts against flat

pLDDT difference from `flat`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| 100 | dms6 | -50.39 | [-51.60, -49.11] | 0.0000 | *** |
| 100 | rand6 | -51.75 | [-53.23, -50.17] | 0.0000 | *** |


### Commit-ranking ablation

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dms6 T=100 | 34.23 | 0.196 | 25.5 | 0.00 | 1.40 | 0.124 | 0.004 | -0.002 | 0.078 | 0.718 | 109.0 |
| flat T=100 | 84.62 | 0.822 | 7.8 | 0.96 | 2.79 | 0.169 | 0.118 | 0.065 | 0.323 | 0.136 | 2.3 |
| natural | 90.12 | 0.950 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |


### Commit-ranking ablation: contrasts against flat

pLDDT difference from `flat`, bootstrap 95% interval.

| group | arm | difference | 95% interval | p | |
|---|---|---|---|---|---|
| 100 | dms6 | -50.39 | [-51.60, -49.11] | 0.0000 | *** |


### Family-pattern conditioning — HOMOLOGY-CONTAMINATED, see Round 3

The 48 pattern sources are the first 48 SwissProt dev sequences of length >= 200, and that split is not homology-clustered: 17 of 48 exceed 0.50 10-mer containment against the 165,231-sequence training corpus and 6 exceed 0.90. The chemical arm gains +10.23 [+0.77, +19.18] pLDDT on the contaminated sources while the random arm gains -2.34 [-5.20, +0.38], so the contamination acts through the chemical channel specifically. Re-measured on a clean set in Round 3; the +26.4 headline is retired. pTM and fold in this table are also pre-fix.

| cell | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amphW-shuf | div | novel | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dms6 T=100 | 36.68 | 0.279 | 22.4 | 0.00 | 14.86 | 0.142 | 0.059 | -0.003 | 0.922 | 1.000 | 3.2 |
| dms6 T=100 conditioned | 63.12 | 0.827 | 13.4 | 0.06 | 13.59 | 0.153 | 0.083 | 0.020 | 0.929 | 1.000 | 2.7 |
| natural | 90.12 | 0.950 | 5.7 | 0.81 | 3.67 | 0.150 | 0.078 | 0.017 | 0.929 | 0.476 | 2.9 |
| rand6 T=100 | 35.41 | 0.252 | 23.3 | 0.00 | 16.37 | 0.139 | 0.061 | -0.014 | 0.928 | 1.000 | 2.9 |
| rand6 T=100 conditioned | 36.54 | 0.288 | 23.0 | 0.00 | 15.80 | 0.150 | 0.083 | 0.013 | 0.925 | 1.000 | 2.8 |


| arm | family compliance | pLDDT |
|---|---|---|
| dms6 | 1.0000 | 63.12 |
| rand6 | 1.0000 | 36.54 |

Containment split, `results/B_conditioning_note.md`:

| arm | containment > 0.50 (n=17) | <= 0.50 (n=31) | difference |
|---|---|---|---|
| chemical pattern | 69.73 | 59.50 | +10.23 [+0.77, +19.18] |
| random pattern | 35.03 | 37.37 | -2.34 [-5.20, +0.38] |
| chemical - random | +34.70 | +22.12 | +12.57 [+2.95, +21.78] |


---

# Round 2, 2026-11-09: the revised method, corrected metrics, standard dataset

Jobs 27126932, 27128178, 27128436, 27149380, 27153733, 27153920, 27154072, 27161950, 27172154. Full write-up in `REPORT_REVISED.md`.

### The revised method (warm-up + cached map + group cap), pooled over warm-ups and budgets

`parallel/adaptive.py`, `parallel/run_revised.py`. Four warm-ups x three budgets.

| decoder | contact - confidence | contact - shuffled |
|---|---|---|
| absorbing, free self signal | +1.29 [-0.80, +3.38] | +0.72 [-1.35, +2.78] |
| remasking (reaches benchmark) | **-10.73 [-12.54, -8.91]** | **-4.90 [-6.76, -3.01]** |

### The decisive diagnostic: harm is linear in how often the gate fires

Remasking decoder, 28 arms.

    harm = -21.02 * fired_fraction + 1.99,   r^2 = 0.991

Extrapolates to +1.99 pLDDT, i.e. zero within noise, when the rule never fires. The warm-up titrates the dose rather than repairing the mechanism; a working fix would peak at an intermediate warm-up and none does.

### Cache refresh policy, at matched contact-call budget

Mean TV change across 28 arms is 0.048-0.090 and never reaches the 0.15 trigger threshold, which fires 36 of 422 refreshes (9 %).

| refresh policy | refreshes | cost | pLDDT |
|---|---|---|---|
| fixed interval, age 8, ignores the distribution | 5.0 | 89.4 | **57.23** |
| trigger-based, default thresholds | 5.0 | 89.4 | 56.99 |
| every step | 13.0 | 152.5 | 56.79 |
| lazy | 5.0 | 89.4 | 50.64 |

### Caching is aimed at the wrong target

Cost in denoiser-pass equivalents, measured on the run GPU:

| computation | relative cost |
|---|---|
| denoiser forward pass | 1.00 |
| denoiser with attentions (the free self signal) | 1.04 |
| frozen ESM-2 650M contact call | **7.88** |

### Replication of the sweep's best cell at 960 sequences per arm

The 12-cell warm-up sweep's best cell was +7.25 pLDDT at warm-up 0.35, T=50. Re-run on fresh seeds, 960 per arm:

| contrast | sweep (48/cell) | replication (960/arm) |
|---|---|---|
| contact - confidence | +7.25 | **+0.65 [-0.96, +2.25] n.s.** |

91 % of the effect is gone, as the selection-bias estimate predicted. The interval still contains both 0 and 1.65, so a 1-2 point effect is neither shown nor excluded.

### Oracle on the replacement dataset: CATH 4.4 S40

The ad-hoc 48-chain set capped pairwise identity at 0.30 but was not fold-non-redundant: 19 CATH topologies for 48 chains, fold 3.20.20 in 11 of 48 entries, 2 mainly-alpha chains. Replaced by one chain per topology, class-stratified, 43 maps (`parallel/cath_oracle.py`, `results/A_oracle_cath.npz`).

| contrast | CATH S40 (n=43, standard) | ad-hoc 48 (retired) |
|---|---|---|
| true map - confidence | **+1.10 [-2.67, +4.82] n.s.** | -4.19 [-7.72, -0.65] |
| true map - shuffled | +2.91 [-0.46, +6.31] | — |

The oracle sign depends on the chain set, so "an oracle map refutes the mechanism" was dataset-dependent and is retired. The remasking-decoder negative does not depend on the oracle set and still stands.

### Why contacts are the wrong graph: the model's own dependency

`D[i,j] = TV(p(x_j|ctx), p(x_j|ctx, x_i:=v))`, deposited maps.

| revealed | AUROC(dependency \| true contact), 48-chain | same, CATH | AUROC(sequence distance \| high dependency), 48-chain |
|---|---|---|---|
| 10 % | 0.545 | 0.536 | 0.747 |
| 25 % | 0.543 | 0.543 | 0.705 |
| 50 % | 0.671 | 0.611 | 0.803 |
| 75 % | 0.728 | 0.650 | 0.854 |

A true contact barely separates dependent pairs early in sampling, which is exactly where the gate fires, and plain sequence distance predicts the model's coupling better at every masking level. The factorization error itself collapses 58x between 25 % and 50 % revealed, so contact informativeness and error magnitude are anti-aligned in time.

### Gating on measured dependency instead of contacts, and the caveat that limits it

Factorization error at 25 % revealed, `parallel/dependency.py`:

| rule | change in factorization error | flip rate, group 8 | flip rate, group 16 |
|---|---|---|---|
| confidence | — | 0.062 | 0.036 |
| pairwise contact (the proposal) | **+5.0 %** | — | — |
| its shuffled control | +5.4 % | — | — |
| pairwise measured dependency | -28 % | — | — |
| groupwise dependency (greedy log-det / DPP) | **-61 %** | 0.406 | 0.276 |

The two error measures disagree. The 61 % is a TV-shift reduction; the flip rate, how often a committed token stops being the modal choice once the rest of the group is written, moves sharply the *other* way, on only 12 states. Never quote the 61 % without the flip rate. Costs about 40 extra forward passes, and pLDDT for this arm has not been run.

### pTM corrected per sequence, remasking decoder

From `C_reparam_w*_scored2.json`, mean over budgets 25/50/100/200, 48 per cell. Bootstrap stratified by budget, 10,000 resamples.

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

The pLDDT columns reproduce the original analysis to two decimals, which checks the rescoring script. Verification of batch-invariance in `tools/verify_ptm.py`; per-sequence corr(pLDDT, pTM) goes from 0.147 to 0.802 after the fix.

### Power

Per-sequence pLDDT sd is 17.8.

| sequences per arm | resolvable effect, 80 % power |
|---|---|
| 48 (one cell) | 7.2 |
| 192 (four cells pooled) | 3.6 |
| 913 | 1.65 (the largest published gain for this task) |

A 12-cell sweep reports a +7 maximum under a true null 27 % of the time.


---

# Round 3, 2026-11-10: answers to the professor's feedback

Jobs 27369294, 27369295, 27369473, 27379873, plus CPU analyses. Index in `ANSWERS_TO_FEEDBACK.md`.

### Contact accuracy against deposited structures, and the argmax repair

Replaces "Contact prediction quality against mask rate" above. precision@L against deposited X-ray contacts. `parallel/argmax_contacts.py`, `results/F_argmax_{pdb,cath}.json`.

Ad-hoc 48-chain set:

| revealed | recovery | ext, masked input | ext, argmax completion | self, masked | self, completion |
|---|---|---|---|---|---|
| 1.00 | 1.000 | 0.747 | 0.747 | 0.642 | 0.642 |
| 0.75 | 0.447 | 0.067 | **0.737** | 0.623 | 0.624 |
| 0.50 | 0.364 | 0.051 | **0.621** | 0.525 | 0.526 |
| 0.25 | 0.175 | 0.059 | 0.193 | 0.168 | 0.169 |
| 0.10 | 0.098 | 0.054 | 0.068 | 0.070 | 0.058 |
| all-mask null | — | 0.069 | — | 0.065 | — |

CATH 4.4 S40, fold-non-redundant — same shape, lower absolute values:

| revealed | recovery | ext, masked input | ext, argmax completion | self, masked |
|---|---|---|---|---|
| 1.00 | 1.000 | 0.606 | 0.606 | 0.475 |
| 0.75 | 0.300 | 0.066 | **0.564** | 0.438 |
| 0.50 | 0.253 | 0.056 | 0.426 | 0.358 |
| 0.25 | 0.127 | 0.054 | 0.147 | 0.123 |
| 0.10 | 0.094 | 0.053 | 0.053 | 0.053 |
| all-mask null | — | 0.050 | — | 0.066 |

AUROC, ad-hoc set: 0.703 -> 0.949 at 0.75 revealed, 0.682 -> 0.899 at 0.50, 0.656 -> 0.728 at 0.25.

The collapse on masked input is an input-representation artefact, not a property of external planners: the repaired external head beats the generator's own attention at 50-75 % revealed. It dies below 25 % revealed, where recovery 0.098 equals the 0.095 scored by always writing leucine. At 0.75 revealed more than half the filled residues are wrong and precision still reaches 0.737, so the head needs a plausible sequence rather than a correct one. Implemented as `--contact-mode external_argmax` in `parallel/sample.py`, in both `generate_absorbing` and `generate_reparam`.

### Does the repaired map make the gate work? No. Four configurations

48 sequences per cell. `results/F_{argmax,plainext}_gate_scored.json`, `results/H_argmax_*_scored.json`.

| configuration | contact - confidence | contact - shuffled | gate cost |
|---|---|---|---|
| absorbing, no warm-up (96/arm) | -4.41 [-9.40, +0.60] | — | 8.9x |
| remasking, no warm-up | **-16.76 [-20.87, -12.62]** | -3.39 [-7.52, +0.76] | 6.9x |
| remasking, warm-up 0.5 | -3.41 [-8.00, +1.17] | +1.41 [-3.22, +6.10] | 4.0x |
| absorbing, warm-up 0.5 | -2.86 [-7.17, +1.40] | +0.10 [-4.13, +4.29] | 8.9x |

| contrast | effect |
|---|---|
| gating on the repaired map vs the plain masked map | **-0.60 [-5.43, +4.34]** |

A tenfold better contact map changes the generated sequences by nothing, because the oracle ceiling is already at parity. Halving the contact calls via the warm-up moved the effect from -16.76 to -3.41, which is the firing-fraction signature of Round 2 rather than a fix. Diversity 0.930-0.941, max repeat 2.8-12.6, so no arm is collapsed.

### The full control ladder, per budget

One frozen DPLM-150M, length 200, 48 per cell, matched reveal-set sizes, free self-attention signal. `results/F_ladder_scored.json`.

| selector | 12 | 25 | 50 | 100 | 200 | pooled | cost | vs confidence (pooled, 240/arm) |
|---|---|---|---|---|---|---|---|---|
| **confidence** | 50.40 | 49.52 | 54.88 | 62.15 | 61.34 | **55.66** | 1.00x | — |
| contact (attention graph) | 48.09 | 50.39 | 57.19 | 64.18 | 57.82 | 55.53 | 1.04x | -0.13 [-3.44, +3.30] |
| distance (spacing, d=8) | 48.18 | 54.46 | 54.61 | 58.05 | 54.63 | 53.99 | 1.00x | -1.67 [-4.92, +1.64] |
| contact_shuffled | 47.56 | 50.90 | 55.73 | 56.08 | 57.93 | 53.64 | 1.04x | -2.02 [-5.31, +1.27] |
| PUNT | 39.36 | 48.99 | 54.17 | 58.28 | 60.95 | 52.35 | **2.08x** | -3.31 [-6.62, +0.03] |
| random | 44.37 | 44.32 | 57.07 | 57.63 | 49.25 | 50.53 | 1.00x | **-5.13 [-8.37, -1.92]** |
| natural | — | — | — | — | — | 90.10 | — | — |

`random` is the only arm significantly different from confidence, which is the positive control: the protocol can resolve a five-point effect, so the nulls are nulls. Diversity 0.932-0.937, novelty 0.997-1.000 for every arm. Note the per-budget noise: `contact` is the best arm at T=100 and the second worst at T=200, which is why only the pooled column is quoted.

### Gate-off ablation

200 trials each, `tools/gate_off_check.py`. The concern was that a disabled contact gate might still force neighbour edges through the distance term.

| check | result |
|---|---|
| contact at tau > max(coupling) vs confidence, random matrices | 0/200 differ |
| same, deposited maps | 0/200 differ |
| `distance` (d=8) vs confidence | **200/200 differ** |

The third row shows the first two are not vacuous. In this implementation `distance` is a separate selector with its own band matrix and is never added into the contact coupling.

### tau is not a continuous knob on a binary oracle map

Fraction of the committed set displaced:

| tau | 0.0 | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 | 1.001 |
|---|---|---|---|---|---|---|---|
| fraction displaced | 1.000 | 0.140 | 0.147 | 0.135 | 0.111 | 0.133 | 0.000 |

Three regimes, flat at about 13 % across the interior, then a cliff. Tuning tau on an oracle arm is meaningless; on a predicted map it is real, which is why it is set as a quantile of each sequence's own coupling values.

### The fold-class effect, withdrawn

`analysis/beta_effect.py`, `results/G_beta_effect_note.md`. 43 CATH chains, budgets 12/25/50/100, effects pooled within chain. The withdrawn claim was beta - alpha = +11.61 [+4.61, +18.68], permutation p = 0.005, with r = +0.42 against contact density.

| test of the claimed mechanism | result |
|---|---|
| correlation of (true - shuffled) with per-chain contact density | **+0.120 [-0.122, +0.356]**, permutation p = 0.44 |
| the same against class-mean density, as originally reported | +0.390 |
| median-density split, true - shuffled | **-0.05 [-6.80, +6.71]** |
| median-density split, true - confidence | +6.31 [-1.43, +13.81] |
| reversed polarity, correlation with density | +0.192 [-0.052, +0.462] |

Stability across budgets:

| forward passes | beta - alpha (true - shuffled) | dense - sparse (true - shuffled) |
|---|---|---|
| 12 | +1.19 | +11.02 |
| 25 | -4.66 | -5.68 |
| 50 | +25.44 | +4.81 |
| 100 | +24.45 | -10.34 |

Per-chain sd of (true - shuffled) is 11.49, so 16 chains per group are needed for the effect as estimated and we have 11 mainly-beta against 15 mainly-alpha. Within-class sd of density is 0.00457 against a between-class 0.00400, so the mechanism had room to be tested directly and failed. Withdrawn as a winner's-curse artefact.

### Family-pattern conditioning on a homology-clean set

96 patterns, maximum 10-mer containment 0.0262, mean 0.0075, selected by `tools/clean_patterns.py` from 560 eligible dev sequences whose mean containment is 0.447. Both checkpoints see the same patterns, so the contrast is paired per source protein. `run/clean_conditioning.sh`, `results/G_cond_clean_scored.json`, `analysis/clean_cond.py`.

| arm | n | pLDDT | pTM | pAE | fold | pppl | amph | amphW | amph-pat | helix | strand | div | novel | identity | rep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| natural | 48 | 90.10 | 0.838 | 5.7 | 0.813 | 3.50 | 0.150 | 0.078 | 0.017 | 0.126 | 0.107 | 0.929 | 0.476 | 0.071 | 2.9 |
| dms6 conditioned | 96 | **56.49** | 0.457 | 16.4 | **0.083** | 14.74 | 0.149 | 0.085 | 0.021 | 0.125 | 0.111 | 0.931 | 0.9998 | 0.069 | 2.9 |
| dms6 uncond | 96 | 36.58 | 0.201 | 22.7 | 0.000 | 14.94 | 0.143 | 0.069 | 0.008 | 0.120 | 0.113 | 0.922 | 0.9999 | 0.078 | 3.1 |
| rand6 conditioned | 96 | 36.27 | 0.197 | 22.9 | 0.000 | 15.93 | 0.148 | 0.081 | 0.020 | 0.124 | 0.114 | 0.926 | 1.000 | 0.074 | 2.9 |
| rand6 uncond | 96 | 35.30 | 0.188 | 23.2 | 0.000 | 16.59 | 0.140 | 0.066 | -0.006 | 0.118 | 0.111 | 0.926 | 1.000 | 0.074 | 2.9 |

| contrast | clean set (n=96) | original contaminated set (n=48) |
|---|---|---|
| **chemical - random, paired per source** | **+20.22 [+16.37, +24.14]** | +26.58 |
| chemical conditioned - its own unconditional | +19.91 | +26.44 |
| random conditioned - its own unconditional | +0.97 | +1.13 |

Retrieval checks on the clean set:

| arm | identity to own pattern source | uniform-in-family floor | modal-residue floor | k-mer novelty |
|---|---|---|---|---|
| dms6 conditioned | 0.361 | 0.305 | 0.397 | 0.9998 |
| rand6 conditioned | 0.074 | — | — | 1.000 |

The chemical arm lands below the modal-residue floor, i.e. below the trivial pattern-compliant baseline, so the gain is not recall. The random partition carries 490 bits against the chemical partition's 484 on this corpus, so the control is matched on information content and not only on group sizes.

**What this round's conditioning result does and does not rest on.** One corpus (SwissProt; the clean set is a re-selection within the same dev split, not an independent dataset), one model pair, one length, one budget, one structure predictor. pLDDT, pTM, pAE and foldability all come from a single ESMFold forward pass and per-sequence pTM correlates 0.909 with pLDDT, so they are one axis read four ways; their value is the qualitative form, namely that the conditioned chemical arm is the only generated arm with nonzero foldability. Of the metrics not read off ESMFold, pseudo-perplexity moves only 0.20 against its own unconditional arm while pLDDT moves 19.91, and hydrophobic patterning does not separate the chemical arm (0.021) from the random conditioned arm (0.020) at all — that part of the effect is conditioning per se, not chemistry. ESMFold's trunk is ESM-2, so pseudo-perplexity is not fully independent either. Not run: a hand-written or consensus pattern, the pattern-degradation curve, TM-score against the source fold, and any second structure predictor.

One number not to read: the correlation between residual containment and the gain is +0.209, but containment now spans 0.0000 to 0.0262, so the predictor is degenerate and the figure is noise.


---

# Round 4, 2026-11-11: the clustered-split retrain and the pattern-accuracy budget

Jobs 27442437 (BLAST purge), 27442438_[0-2] (retrain), 27442439 (2x2 conditioning), 27442330 (degradation curve), 27442486_[0-1] (family-signal variants). Notes in `results/J_clustered_split_note.md` and `results/I_pattern_curve_note.md`.

### How much homology the 10-mer containment filter was missing

128 held-out pattern sources sampled from the dev candidates with no homology criterion. blastp against the 165,231-sequence training corpus, E <= 1e-3 counts as a relative.

| quantity | value |
|---|---|
| sources with at least one BLAST relative in training | **125 of 128 = 97.7 %** |
| distinct training sequences that are relatives | 19,645 |
| fraction of the corpus purged | **11.89 %** |
| corpus after the purge | 145,586 (was 165,231) |
| held-out length mean vs candidate pool | 223.1 vs 225.8 |

### The purged-corpus retrain

Identical hyperparameters to `run/train_all.sh` (20,000 steps, batch 256, same seed), differing only in the corpus.

| arm | best val, purged | best val, original |
|---|---|---|
| flat | 4.7536 | 4.6328 |
| dms6 (chemical) | 8.6251 | 8.5019 |
| rand6 (random) | 8.6966 | 8.6741 |

### The 2x2: identical patterns in every cell

| corpus | chemical cond. | chemical uncond. | random cond. | random uncond. |
|---|---|---|---|---|
| original | 62.48 | 36.67 | 36.39 | 35.67 |
| purged | 60.34 | 35.77 | 36.96 | 35.31 |

| corpus | chemical - random, paired per source | own gains, chem / rand | chem fold / pTM | rand fold / pTM |
|---|---|---|---|---|
| original | **+26.09 [+22.89, +29.28]** | +25.81 / +0.72 | 0.156 / 0.546 | 0.008 / 0.203 |
| purged | **+23.38 [+20.07, +26.68]** | +24.58 / +1.66 | 0.172 / 0.521 | 0.000 / 0.204 |

| difference of differences, paired per source | value |
|---|---|
| what training homologs were worth | **+2.71 [+0.95, +4.55]** |

10.4 % of the unpurged effect. The interval excludes zero, so the leak is real; it is not the effect. Foldability for the conditioned chemical arm is higher on the purged corpus, and it remains the only arm producing any foldable sequence.

Scope: depth-1 single-linkage blastp neighbourhood, not an MMseqs2 clustering (neither MMseqs2 nor CD-HIT installed; a downloaded MMseqs2 binary was not permitted to run). 19,645 round-1 relatives exceeded the 8,000 budget for a second round, so relatives-of-relatives remain in training, as do fold-level homologs BLAST cannot see.

### The pattern-accuracy degradation curve

96 homology-clean patterns, 100 forward passes, paired per source. Corruption replaces a position with a *different* random family, so the fraction is the true error rate.

| pattern error | chemical pLDDT | gain over uncond (36.58) | loss vs perfect | random pLDDT | gain over uncond (35.30) |
|---|---|---|---|---|---|
| 0.00 | 55.41 | **+18.83 [+15.15, +22.55]** | — | 36.32 | +1.02 [-0.21, +2.33] |
| 0.05 | 49.65 | +13.07 [+9.83, +16.36] | -5.76 [-9.04, -2.48] | — | — |
| 0.10 | 43.23 | +6.65 [+3.79, +9.65] | -12.18 [-15.50, -8.99] | — | — |
| 0.20 | 35.51 | -1.08 [-2.66, +0.67] | -19.90 [-23.58, -16.28] | — | — |
| 0.30 | 32.56 | -4.02 [-4.95, -3.02] | -22.85 | — | — |
| 0.50 | 30.91 | -5.67 [-6.32, -4.96] | -24.50 | 34.35 | -0.96 [-2.08, +0.21] |
| 0.75 | 31.05 | -5.53 [-6.14, -4.89] | -24.36 | — | — |
| 1.00 | 30.20 | **-6.38 [-7.00, -5.70]** | -25.21 | 34.91 | -0.39 [-1.61, +0.91] |
| **the model's own pattern** | **36.87** | **+0.29 [-1.09, +1.78]** | -18.54 | — | — |
| natural proteins | 90.10 | — | — | 90.10 | — |

| quantity | chemical | random |
|---|---|---|
| slope | **-1.15 pLDDT per 1 % error** | **-0.04** |
| error that halves the gain | 0.078 | — (no gain to halve) |
| error that erases the gain | 0.186 | — |

Three readings. The random partition has no curve, so the chemical slope is chemical information being destroyed and not an alphabet restriction being relaxed. A fully wrong chemical pattern costs -6.38 while a fully wrong random one costs -0.39, so the harm is chemistry-specific too, and it is the mechanism behind the two-level arm's 6.5-point deficit against flat. And the model's own pattern retains 2 % of the gain -- but scores 5.8 pLDDT above uniform corruption at a matched error rate, so its errors are structured more benignly than uniform ones, and the idea fails with that advantage included.

### Family prediction accuracy, and two attempts to fix it

Teacher-forced, `fam_head`, 96 sources. The majority-class floor is the accuracy of always naming the most common family.

| checkpoint | best val | ce_fam | nats past marginal (1.662) | accuracy, any reveal level | best top decile |
|---|---|---|---|---|---|
| dms6 (original, gamma=2) | 8.5019 | 1.5716 | 0.090 | **0.3634** | 0.423 |
| famsig/g1 (gamma=1.0) | 7.4964 | 1.4299 | 0.232 | **0.363** | 0.457 |
| famsig/clip50 (1/p_fam capped at 50) | 7.2478 | ~1.43 | ~0.23 | **0.363** | 0.411 |
| majority-class floor | — | 1.662 | 0 | **0.3633** | — |
| rand6 (random partition) | 8.6741 | — | — | 0.2696 (floor 0.2651) | 0.331 |

Accuracy by reveal level on the original checkpoint, showing the head ignores context entirely:

| identities revealed | 0.00 | 0.10 | 0.25 | 0.50 | 0.75 |
|---|---|---|---|---|---|
| accuracy | 0.3634 | 0.3634 | 0.3634 | 0.3633 | 0.3635 |

Pattern composition, real against model-generated: real [0.363, 0.107, 0.189, 0.112, 0.113, 0.115], generated [0.432, 0.083, 0.165, 0.099, 0.107, 0.115] -- over-weighting the majority class, as a near-constant predictor does. Lag-3 autocorrelation of the hydrophobic family is 0.011 real and 0.001 generated, so neither carries local periodic structure at this granularity.

Both loss corrections improve validation by 1.0 to 1.25 nats and carry 2.5x the information, and neither moves argmax accuracy off the floor. The failure is informational, not a training artefact; the earlier suspicion that it was an artefact is withdrawn.


---

# Round 5, 2026-11-11: the conditioning test on CATH S35

Jobs 27447590 (purge), 27447591_[0-2] (retrain), 27447592 (2x2 conditioning). Note in `results/K_cath_dataset_note.md`. Identical design to Round 4, with CATH S35 domains as pattern sources instead of SwissProt dev sequences.

### The CATH S35 candidate pool

`tools/cath_patterns.py` over `cath-domain-seqs-S35.fa` (latest CATH release, 37,350 domains, non-redundant at 35 % sequence identity).

| filter | domains removed | remaining |
|---|---|---|
| all S35 domains | — | 37,350 |
| shorter than 200 | 28,507 | 8,843 |
| longer than 254 | 4,805 | 4,038 |
| non-canonical residues | 159 | 3,876 |

Candidate length mean 224.3, median 223, range 200-254, against 225.8 mean for the SwissProt candidate pool.

### Homology of each pattern set to the SwissProt training corpus

| | SwissProt dev | CATH S35 |
|---|---|---|
| sources sampled (no homology criterion) | 128 | 128 |
| sources with a BLAST relative at E <= 1e-3 | 125 = **97.7 %** | 54 = **42.2 %** |
| direct relatives | 19,645 | 2,127 |
| depth-2 expansion added | — (over budget) | 4,513 |
| purge depth achieved | 1 | **2** |
| sequences purged | 19,645 = **11.89 %** | 6,640 = **4.02 %** |
| corpus after purge | 145,586 | 158,591 |

### The CATH-purged retrain

| arm | full corpus | CATH-purged (4.02 % removed) | SwissProt-purged (11.89 % removed) |
|---|---|---|---|
| flat | 4.6328 | **4.6505** | 4.7536 |
| dms6 | 8.5019 | **8.5287** | 8.6251 |
| rand6 | 8.6741 | **8.6712** | 8.6966 |

The CATH purge costs 0.02-0.03 nats against the SwissProt purge's 0.10-0.12, in proportion to the corpus removed.

### The 2x2 on CATH patterns

| corpus | chemical cond. | chemical uncond. | random cond. | random uncond. |
|---|---|---|---|---|
| original | 54.93 | 36.67 | **33.87** | 35.67 |
| CATH-purged | 53.47 | 35.44 | **33.62** | 35.49 |

| corpus | chemical - random, paired per source | own gains, chem / rand | chem fold / pTM | rand fold / pTM |
|---|---|---|---|---|
| original | **+21.06 [+17.86, +24.24]** | +18.26 / **-1.80** | 0.055 / 0.497 | 0.000 / 0.193 |
| CATH-purged | **+19.85 [+16.62, +23.05]** | +18.03 / **-1.87** | 0.047 / 0.480 | 0.000 / 0.189 |

| difference of differences, paired per source | value |
|---|---|
| what training homologs were worth, CATH | **+1.21 [-0.68, +3.06]** |
| the same on SwissProt, for comparison | +2.71 [+0.95, +4.55] |

### Cross-dataset summary

| dataset | original corpus | purged corpus | memorisation |
|---|---|---|---|
| SwissProt dev | +26.09 [+22.89, +29.28] | **+23.38 [+20.07, +26.68]** | +2.71 [+0.95, +4.55] |
| CATH S35 | +21.06 [+17.86, +24.24] | **+19.85 [+16.62, +23.05]** | +1.21 [-0.68, +3.06] |

The purged intervals overlap, so the effect does not depend on which dataset supplies the patterns. These are separate bootstraps on different pattern sets, so overlap is an informal check and not a test of equality. Memorisation is smaller on CATH, as its 42 % source homology against SwissProt's 97.7 % predicts, and its interval contains zero; together the two datasets bound memorisation at roughly 1-3 pLDDT of a 20-26 pLDDT effect.

Three readings to carry forward. The random control is **negative** on CATH (-1.80 against its own unconditional baseline, where SwissProt gave +0.72), which independently corroborates the degradation-curve finding that a chemically meaningless pattern costs something because binding makes a family error unrecoverable, and costs more on compact structural domains. Foldability for the conditioned chemical arm is 0.047-0.055 on CATH against 0.156-0.172 on SwissProt, so the "only arm that folds at all" claim holds in form on both datasets but its magnitude is a SwissProt number. And ESM-2 pseudo-perplexity does not favour the conditioned arm on CATH either (16.56 against 14.94 unconditional), leaving the Round 3 caveat about the one non-ESMFold metric unchanged.

Scope: the generator is SwissProt-trained in both experiments, so this establishes that the pattern source can be independent of training, not that the effect survives a model trained on a different corpus. Training on CATH is not feasible here -- 3,876 eligible domains is far too small for a 38M model from scratch.

---

# Round 6, 2026-11-12: metrics that are not ESMFold

Job 27496304 (`run/extra_metrics.sh`), note in `results/L_extra_metrics_note.md`, figure `figures/fig9_independent_metrics`. Nothing was generated or re-folded: these are the sequences already stored in the Round 4 and Round 5 scored files, asked a different question. The motivation is the asymmetry declared at the end of Round 5 -- pLDDT, pTM, pAE and foldability are one ESMFold pass read four ways, and ESM-2 pseudo-perplexity, the only metric from another model, moved 0.20 where pLDDT moved 19.91.

### The four arms on the purged corpora (SwissProt / CATH S35)

| arm | pLDDT | pppl | NSR | Max-ID % | FPD | MMD |
|---|---|---|---|---|---|---|
| chemical, conditioned | **60.3 / 53.5** | 14.39 / 17.33 | 0.353 / 0.348 | 11.5 / 12.0 | **12.0 / 13.0** | **0.348 / 0.371** |
| random, conditioned | 37.0 / 33.6 | 15.76 / 16.21 | 0.401 / 0.393 | 10.4 / 10.9 | 26.1 / 24.5 | 0.780 / 0.738 |
| chemical, unconditional | 35.8 / 35.4 | 15.71 / 14.91 | 0.068 / 0.066 | 10.5 / 9.6 | 27.7 / 25.6 | 0.775 / 0.715 |
| random, unconditional | 35.3 / 35.5 | 16.73 / 16.56 | 0.068 / 0.067 | 10.7 / 11.3 | 27.3 / 24.6 | 0.818 / 0.744 |
| natural proteins | 90.1 | 3.50 | -- | 42.1 / 77.9 | 3.0 / 4.4 | -0.002 / 0.031 |

### FPD and MMD: an independent axis that agrees

Fréchet distance and kernel MMD between generated and natural distributions in ESM-2 embedding space. They score a set rather than a sequence, so they see what no per-sequence mean can. Measured against each arm's own unconditional baseline, the chemical pattern closes **64 %** (SwissProt) and **60 %** (CATH) of the distance to natural proteins; the bit-matched random partition closes **5 %** and **0.3 %**. MMD agrees (0.348 vs 0.780, 0.371 vs 0.738) without assuming anything Gaussian, and natural-against-natural calibrates at 3.0 / -0.002 and 4.4 / 0.031. This is the first metric sharing no component with ESMFold's structure module to confirm the chemical-versus-random separation; hydrophobic patterning and pseudo-perplexity both failed to.

### NSR: no retrieval, and no discrimination either

| arm | NSR | uniform-in-family floor | modal-in-family floor | NSR / modal |
|---|---|---|---|---|
| SwissProt chemical | 0.353 [0.346, 0.360] | 0.306 | 0.391 | 0.90 |
| SwissProt random | 0.401 [0.394, 0.408] | 0.308 | 0.441 | 0.91 |
| CATH chemical | 0.348 | 0.307 | 0.389 | 0.89 |
| CATH random | 0.393 | 0.306 | 0.442 | 0.89 |

Every conditioned arm sits BELOW its own modal floor, which is an alignment-free confirmation that nothing is being retrieved. And the random arm's higher raw NSR is entirely its higher modal floor: normalised by each arm's own floor the two are identical to within 0.02 in all four cells, so **NSR does not discriminate chemistry**. The chemical gain is not more native residues in the right places.

### Max-ID: Round 4's purge confirmed by alignment

| arm | original corpus | after the purge |
|---|---|---|
| chemical, conditioned | 32.2 % (max 79.5) | **11.5 %** |
| random, conditioned | 32.0 % (max 85.0) | 10.4 % |
| chemical, unconditional | 10.5 % | 10.5 % |
| natural dev proteins | 83.0 % | 42.1 % |

blastp against the corpus each arm actually trained on, scored over the query. After the purge a conditioned sequence is no closer to training than an unconditional one, while keeping its +24.6 pLDDT. Round 4 established this by retraining and comparing pLDDT; this establishes it by aligning the sequences themselves, and the natural control moving 83 → 42 % is the check that the purge removed what it was meant to.

### Pseudo-perplexity is retired as a discriminator

It favours the chemical arm on SwissProt (14.39 vs 15.76) and the random arm on CATH (17.33 vs 16.21), contradicting itself where FPD, MMD and pLDDT agree on both datasets. With natural at 3.50 and every generated arm between 14 and 17, it is dominated by "this is not a real protein" and has little resolution left. Report it as a declared null.

### Not run: scTM and scRMSD

Both need an inverse-folding model. ProteinMPNN is not on disk and the only PyPI package of that name is an unvetted third-party wheel shipping its own weights; ESM-IF1 (`fair-esm`, installed) needs `torch_scatter`/`torch_sparse`, which have no wheel for torch 2.7.1+cu126. Declared missing rather than approximated. Ref-TM asks the stronger question for these sequences anyway, because they have a real reference structure.

### Ref-TM: the conditioned sequences adopt the source protein's fold

Jobs 27496556 and 27497977 (`run/ref_tm.sh`, `run/ref_tm_perm.sh`), note in `results/M_reftm_note.md`, figure panel D. The CATH arm's pattern sources are structural domains with deposited coordinates (127 of 128 downloaded by `tools/fetch_cath_domains.py`), so each generated sequence can be folded and structurally aligned to the crystal structure of the domain whose chemical pattern it was given, TM normalised by the generated chain.

| arm | Ref-TM, purged | TM >= 0.5 | RMSD | aligned to a DIFFERENT source |
|---|---|---|---|---|
| the source's own slice (ceiling) | 0.861 [0.835, 0.884] | 95.3 % | 1.7 Å | -- |
| **chemical, conditioned** | **0.596 [0.558, 0.635]** | **64.6 %** | 3.8 Å | **0.303** |
| random partition, conditioned | 0.299 [0.290, 0.309] | 0.8 % | 5.6 Å | 0.288 |
| chemical, unconditional | 0.288 [0.281, 0.294] | 0.0 % | 5.6 Å | -- |
| random partition, unconditional | 0.289 [0.282, 0.295] | 0.0 % | 5.7 Å | -- |

0.5 is the conventional same-fold threshold; an unrelated pair of domains from this set scores 0.29, which is where all three control arms sit. The chemical arm covers 54 % of the distance from that background to the ceiling, the random partition 2 %.

**Three controls, and the effect survives all of them.** (i) The bit-matched random partition is at background. (ii) The permuted pairing — the same generated structures aligned to a *different* source — collapses the chemical arm from 0.596 to 0.303, so what it produces is the source's fold and not a generically domain-like shape. (iii) It is not identity being recognised: the random arm recovers MORE of the source's residues (NSR 0.393 against 0.348) and still scores background.

**Memorisation, measured structurally.** On the unpurged corpus the chemical arm scores 0.607 [0.568, 0.645] with 67.7 % above threshold, against 0.596 and 64.6 % purged, so deleting every BLAST relative of every pattern source costs **0.011 TM of a 0.308 effect over background, under 4 %**. Round 4 bounded memorisation at 10 % of the pLDDT gain, Round 5 at 1-3 pLDDT, and the structural endpoint puts it lower still.

**Declared.** The generated structure is an ESMFold prediction; only the reference is experimental. ESMFold's own training set was not purged and has seen these domains, so this shows the generated sequence carries enough signal for a predictor that knows the fold to recover it, which is weaker than independent physical evidence. The random control bounds how much of the result that can explain, since ESMFold knows the source for those sequences too and they have higher identity to it. Note also that the conditioned arm's pLDDT is only 53.5 while its topology is right, so on this task pLDDT and the foldability flag (0.047) both understate how much structural information the conditioning carries.
