# Harness fidelity, and a problem with the proposal's framing

Length 200, DPLM-150M, 48 sequences per cell, ESMFold pLDDT.

| decoder | forward passes | pLDDT | 95% CI |
|---|---|---|---|
| DPLM's own reparameterised decoder | 200 | **83.60** | 79.32 - 87.45 |
| absorbing, confidence order | 200 | 57.18 | 52.20 - 62.57 |
| absorbing, random order | 200 | 55.23 | 50.49 - 60.19 |
| absorbing, lowest entropy | 200 | 53.65 | 48.79 - 59.12 |

Published DPLM-150M at length 200 is 84.30. Our reimplementation of its
decoder returns 83.60, so the harness, the checkpoint, the folding model and
the scoring are all faithful.

Everything else in that table is the problem.

**The decoder the proposal describes is not the decoder DPLM ships.** The
proposal's premise is choosing which masked positions may be filled in the
same iteration, which presumes an absorbing process: a position, once
written, is final. DPLM does not work that way. Its published decoder
re-predicts every position at every step and returns the lowest-scoring
L*(1 - t/T) of them to the mask state, so nothing is final until the last
step. In our harness that difference is worth 26 pLDDT at equal forward
passes.

**Inside the absorbing family, the unmasking order barely matters.** At 200
forward passes for a 200-residue sequence the three orders span 3.5 points
with overlapping intervals. That is a very different picture from the
published decoding-order comparison, which reports random order at 40.0
against confidence order at 82.2 on the same base model. Those published
numbers are measured inside a decoder that remasks, where "order" means the
order in which positions stop being revised, not the order in which they are
irrevocably written.

Two consequences for Idea A:

1. Any gain it produces inside the absorbing family starts 26 points below
   the decoder DPLM already ships, so it cannot be presented as an
   improvement to DPLM without also closing that gap.
2. The rule has to be tested inside the reparameterised decoder, where the
   set of positions that survive a step is exactly the set written in
   parallel at that step. That is the faithful transfer, and it is run
   separately.

One caveat is still open: we rank commits by DPLM's Gumbel-perturbed score,
which is its re-masking criterion and may be too noisy to serve as a commit
ranking. A separate run repeats the absorbing arms with the clean
probability of the sampled token to settle whether the absorbing baseline is
weak for that reason or because irrevocable commitment is simply worse.
