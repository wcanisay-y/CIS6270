# The commit ranking: a trap, and why DPLM's choice is right

DPLM scores every position with the maximum log-softmax of its logits after
Gumbel noise is added. We first suspected that reusing this as the ranking
for deciding what to COMMIT in an absorbing sampler was crippling the
baseline, because under it confidence ordering scored within a few points of
random ordering. Testing the alternative settled it, in the opposite
direction.

Length 200, 48 sequences per cell.

| commit ranking | T | pLDDT | ESM-2 pppl | diversity | mean longest run |
|---|---|---|---|---|---|
| clean probability, confidence order | 25 | 73.86 | 1.27 | 0.283 | **130.9** |
| clean probability, confidence order | 200 | 76.01 | 1.16 | 0.365 | **143.7** |
| clean probability, random order | 25 | 47.44 | 15.14 | 0.936 | 3.2 |
| Gumbel-perturbed, confidence order | 200 | 57.18 | 12.46 | 0.935 | 3.5 |
| natural proteins | - | 90.10 | 3.50 | 0.929 | 2.9 |

A representative sequence from the first row:

```
MLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL...
```

The 26-point pLDDT gain from the clean ranking is entirely mode collapse.
Ranking by raw confidence and committing irrevocably drives the sampler to
the single most predictable residue, 131 of 200 positions in an average
sequence are the same amino acid, pseudo-perplexity falls to 1.27 because
the sequence is trivially predictable, and ESMFold returns pLDDT 74 because
a polyleucine 200-mer is a confident single helix.

DPLM's appendix warns about exactly this, and the Gumbel perturbation is the
fix. Our original ranking was therefore the correct one, and the earlier
grid was not run against a crippled baseline.

The episode leaves two things worth keeping:

1. pLDDT on its own is not a usable criterion here. A sequence at 74 with a
   131-residue homopolymer run outscores every honest absorbing arm. Any
   claim in this project is reported with repetition, diversity and
   pseudo-perplexity beside it.
2. The narrow spread between confidence, random and entropy ordering in the
   absorbing family is a real property of that family and not an artefact
   of the score. Ordering matters much more inside DPLM's remasking
   decoder, which is where its published decoding-order numbers come from.
