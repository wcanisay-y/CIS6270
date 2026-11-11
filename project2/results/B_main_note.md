# Idea B: the decisive comparison

Three arms, 38M parameters each, same backbone, same SwissProt corpus, same
20,000-step budget at batch 256, same float64 sampling, same 100 forward
passes at generation. The only difference is the state space. Length 200,
48 sequences per arm.

| arm | pLDDT | ESM-2 pppl | amphipathic | diversity | novelty | repeat run |
|---|---|---|---|---|---|---|
| natural proteins | 90.12 | 3.67 | 0.150 | 0.929 | 0.476 | 2.9 |
| **flat, 20 letters** | **43.09** | **13.69** | **0.146** | 0.929 | 0.952 | 2.5 |
| two-level, 6 chemical families | 36.68 | 14.86 | 0.142 | 0.922 | 1.000 | 3.2 |
| two-level, 6 RANDOM groups | 35.41 | 16.37 | 0.139 | 0.928 | 1.000 | 2.9 |

95 percent intervals: flat 37.5-49.5, chemical 35.1-38.4, random 33.9-37.1.
No arm is degenerate; diversity is 0.92-0.93 and repeat runs are 2.5-3.2,
against 2.9 for natural proteins.

The flat baseline at 43.1 sits just above the published range for
from-scratch discrete diffusion at this scale on this corpus, which is 36 to
38, so the comparison is being made at a sane operating point.

Three things follow, and they point the same way.

**The two-level alphabet costs quality rather than adding it.** Both
hierarchical arms are below the flat baseline, by 6.4 and 7.7 pLDDT, and
both are worse on pseudo-perplexity. The extra channel is not free: at
matched forward passes it spends part of the budget writing families, and
the identity stage does not recover what that costs.

**The chemistry contributes almost nothing.** The whole case for this
project rests on the chemical partition beating an arbitrary one, because
the two-level mechanism itself is already published. The chemical families
beat a random partition of identical group sizes by 1.27 pLDDT, with
intervals that overlap across most of their range. On the training
objective the gap is visible and consistent (best validation 8.502 against
8.674, and the chemical families are easier to predict throughout
training), so the chemistry is real information. It simply does not survive
into generation quality.

**The headline test moves the wrong way.** The proposal's decisive test was
hydrophobic periodicity. The two-level arms score 0.142 and 0.139 against
0.146 for the flat arm and 0.150 for natural proteins, so the arm with an
explicit chemical channel produces slightly LESS hydrophobic patterning than
the arm without one.

One caveat worth stating plainly. These are 38M models trained for 75
minutes each on 409k sequences, and at that scale every arm is far from
natural quality. A mechanism that helps only at scale would not show here.
What the experiment does establish is that there is no effect large enough
to see at the scale where the project can actually run its ablations, and
that the control the idea depends on comes back negative.
