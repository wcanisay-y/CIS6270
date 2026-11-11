# Conditioning, re-measured without the information leak

The headline +26.4 pLDDT for family-pattern conditioning was measured on
patterns drawn from the upstream SwissProt dev split, which is not
homology-clustered. 17 of those 48 sources shared more than half their
10-mers with a training sequence, the chemical arm gained +10.2 pLDDT on
exactly those, and the random arm gained nothing on them — the signature of
a chemically coherent pattern acting as a retrieval key for a memorised
fold. See the correction block in `B_conditioning_note.md`.

Re-run on `G_clean_patterns.json`: the 96 cleanest dev sequences of length
>= 200, maximum 10-mer containment **0.0262**, mean 0.0075, against a mean
of 0.37 for the original set. Both checkpoints see the same patterns, so the
chemical-minus-random contrast stays paired per source protein.

Generation and scoring: `run/clean_conditioning.sh`,
`results/G_cond_clean_scored.json`, analysis `analysis/clean_cond.py`.

## Result

| arm | n | pLDDT | pTM | foldability |
|---|---|---|---|---|
| natural proteins | 48 | 90.10 | 0.838 | 0.813 |
| **chemical pattern given** | 96 | **56.49** | 0.457 | **0.083** |
| chemical families, unconditional | 96 | 36.58 | 0.201 | 0.000 |
| **random-group pattern given** | 96 | **36.27** | 0.197 | 0.000 |
| random groups, unconditional | 96 | 35.30 | 0.188 | 0.000 |

| contrast | clean set (n=96) | original set (n=48) |
|---|---|---|
| **chemical - random, paired per source** | **+20.22 [+16.37, +24.14]** | +26.58 |
| chemical conditioned - its own unconditional | +19.91 | +26.44 |
| random conditioned - its own unconditional | +0.97 | +1.13 |

## Reading

**The effect survives, about a quarter smaller.** Contamination was worth
roughly 6 pLDDT of the original 26.4. The remaining +20.2 has a 95 %
interval of [+16.4, +24.1] on 96 paired sources whose maximum training
containment is 0.026, so it is not recall.

**The control still does nothing.** A random partition with the same group
sizes, carrying 490 bits against the chemical partition's 484 (measured on
this corpus), is worth +0.97. The gain is the chemistry, not the bit count
and not the act of constraining the output.

**pTM and foldability agree, and foldability is the sharpest form.** The
conditioned chemical arm is the only one of the four generated arms that
produces any foldable sequence at all: 0.083 against 0.000 for the
unconditional chemical arm, the random conditioned arm and the random
unconditional arm. A 38M from-scratch model given a real chemical blueprint
crosses a threshold that it never crosses otherwise.

**No retrieval on the clean set.** Identity of the generated sequence to its
own pattern source is 0.361, against a 0.305 floor for picking uniformly
inside each family and a 0.397 floor for always writing the family's most
common residue — i.e. the model lands *below* the trivial compliant
baseline. k-mer novelty against training is 0.9998.

**One number not to read.** The correlation between residual containment and
the gain is +0.209, but containment now spans only 0.0000 to 0.0262. That is
a degenerate predictor and the figure is noise; it neither supports nor
undermines anything.

## What is still not shown

The patterns are still read off real proteins, so this measures the capacity
of the chemical-class channel, not a design method. Nothing here shows that
a hand-written or consensus pattern works, and a hand-written pattern has no
guarantee of lying on the manifold of realisable patterns. The next
experiment is the degradation curve: corrupt a fraction f of pattern
positions to a random family and plot pLDDT against f. If the gain collapses
by f = 0.1 no hand-written pattern will ever work; if it degrades gently the
interface claim is live.

Also open: whether the conditioned sequences adopt the *source protein's*
fold or merely some confident fold. Each pattern has a known source, so a
TM-score comparison against it would settle this, and it matters because
"the pattern transfers a specific structure" and "the pattern makes
generation generically easier" are different claims.
