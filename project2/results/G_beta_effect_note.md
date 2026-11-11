# The beta-fold contact effect does not survive its own mechanism test

`REPORT_REVISED.md` §6.3 reported the one positive that appeared to survive
both the dataset correction and the metric correction: on the CATH S40 set,
the beta-minus-alpha contrast for a deposited contact map against its own
shuffled control was +11.61 pLDDT [+4.61, +18.68], permutation p = 0.005,
with +0.121 pTM p = 0.010, and was said to be corroborated by a per-chain
correlation with contact density of r = +0.42.

Re-tested the way it would have to survive to become a method
(`analysis/beta_effect.py`, 43 chains, budgets 12/25/50/100, effects pooled
within chain), it does not hold up.

## 1. The corroborating correlation was against the class label

The reported r = +0.42 was computed against **class-mean** contact density —
a four-valued variable that is a relabelling of the CATH class, not an
independent measurement. Against each chain's **own** density:

| correlation of (true - shuffled) with | r |
|---|---|
| per-chain contact density | **+0.120 [-0.122, +0.356]**, permutation p = 0.44 |
| class-mean contact density (as reported) | +0.390 |

There is no shortage of within-class variation to test on: within-class sd
of density is 0.00457 against a between-class sd of 0.00400. So the density
mechanism can be tested directly, and it fails.

## 2. Splitting on density instead of fold label gives nothing

The mechanism is "a contact map helps where contacts are dense." Split the
43 chains at the median density, ignoring the fold label:

| contrast | dense half (n=21) | sparse half (n=22) | difference |
|---|---|---|---|
| true - shuffled | +2.88 | +2.93 | **-0.05 [-6.80, +6.71]** |
| true - confidence | +4.33 | -1.98 | +6.31 [-1.43, +13.81] |

The contrast that controls for the gate's displacement — true against its
own shuffled map — is exactly zero when the split is made on the mechanism
variable, while the same data split on the fold label gives +11.6.

## 3. The effect is not stable across budgets

| forward passes | beta - alpha (true - shuffled) | dense - sparse (true - shuffled) |
|---|---|---|
| 12 | +1.19 | +11.02 |
| 25 | -4.66 | -5.68 |
| 50 | +25.44 | +4.81 |
| 100 | +24.45 | -10.34 |

The class effect lives entirely in two of four budgets and is negative in
one. The density effect alternates sign. Neither pattern is what a
mechanism looks like.

## 4. The reversed polarity behaves wrongly

If dense contacts are what the gate exploits, the reversed polarity —
co-committing contacting positions — should get *worse* as density rises.
Its correlation with density is **+0.192 [-0.052, +0.462]**, if anything
higher than the true map's +0.120. Both polarities drift the same way, which
is the signature of a density-correlated nuisance rather than a mechanism.

## 5. Power says the estimate is probably inflated

Per-chain sd of (true - shuffled), pooled over budgets, is 11.49.

| group difference | chains per group for 80 % power |
|---|---|
| 11.6 (the reported effect) | 16 |
| 8.0 | 33 |
| 5.0 | 83 |
| 3.0 | 230 |

We have 11 mainly-beta and 15 mainly-alpha chains, below the 16 per group
needed to resolve the effect as estimated. An effect detected at borderline
power is biased upward, which is exactly what happened to the +7.25 warm-up
cell that replicated at +0.65.

## 6. And on the whole set the gate does nothing

| contrast | over all 43 chains |
|---|---|
| true - shuffled | +2.91 [-0.46, +6.31] |
| true - confidence | +1.10 [-2.84, +4.99] |

## Verdict

The beta-fold effect should not be carried forward as a finding. The class
split is significant, but the mechanism it was meant to demonstrate —
benefit rising with contact density — is absent when tested directly, the
correlation offered as corroboration was the class label in disguise, the
effect is unstable across budgets, the reversed polarity moves the wrong
way, and the sample is below the power needed for the effect size claimed.

The most likely reading is a winner's-curse artifact of a three-class split,
the third such artifact found in this project after the +7.25 sweep cell and
the ad-hoc oracle set.

`REPORT_REVISED.md` §6.3 should be amended to say so.
