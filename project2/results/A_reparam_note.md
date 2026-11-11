# Idea A inside the decoder DPLM actually ships

The absorbing framing the proposal assumes scores about 30 pLDDT below
DPLM's own decoder, so the rule has to be tested where the quality is.
DPLM keeps the highest-scoring L*(t/T) positions each step and returns the
rest to mask, which means the set that survives a step IS the set written in
parallel at that step. Applying the decoupling rule to the surviving set is
the faithful transfer. The number of survivors and the forward-pass count
are unchanged; only WHICH positions survive changes.

Length 200, 48 sequences per cell. All arms are diverse (0.93).

| forward passes | baseline | contact rule | shuffled control | opposite polarity |
|---|---|---|---|---|
| 25 | 58.10 | 54.57 | 54.74 | 53.08 |
| 50 | 65.62 | 58.71 | 62.23 | 57.22 |
| 100 | 78.23 | 63.32 | 72.61 | 55.24 |
| 200 | 80.24 | 66.45 | 79.48 | 60.03 |

As a difference from the baseline:

| forward passes | contact rule | shuffled control |
|---|---|---|
| 25 | -3.54 | -3.37 |
| 50 | -6.91 | -3.39 |
| 100 | **-14.90** | -5.62 |
| 200 | **-13.79** | **-0.77** |

Two things to read here, and the second is the interesting one.

**The rule costs quality, and the cost grows with compute.** At 200 passes
it gives up 14 pLDDT against simply letting the decoder keep whatever it is
most confident about.

**Using the REAL contact map is worse than using a permuted one.** At 200
passes the shuffled control is essentially free, costing 0.77, while the
genuine contact map costs 13.79. A permuted matrix perturbs the survivor
set arbitrarily and the decoder's later steps repair the damage. A genuine
contact map perturbs it systematically, pushing apart exactly the positions
that are mutually constrained, and the decoder cannot repair that.

So the structural information is not merely useless here, it is pointed the
wrong way. Forbidding contacting residues from being decided together
removes the decoder's chance to settle them jointly. The opposite
prescription does not work either: forcing contacting residues together
costs 5 to 23 points. Both deviations from confidence ordering lose, which
is the same conclusion the absorbing-family grid reached by a different
route.
