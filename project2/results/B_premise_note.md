# The periodicity premise, measured on an existing model

Idea B's decisive test, in the proposal's own words, is hydropathy
periodicity: "Real proteins show sharp power near 3.6 residues per cycle for
helices and 2 for strands. The current model has no mechanism for organising
patterning at that scale and should look flat by comparison."

That is a measurable claim, and it does not hold for DPLM.

Length 200, 48 sequences per cell. `amphW` is the fraction of 18-residue
windows whose least-squares spectrum puts more than 30 percent of its area
in the helical band (87.5-107.5 degrees per residue). `patterning` is that
number minus the same statistic on a shuffle of each sequence, which keeps
the amino-acid composition and destroys the arrangement. The difference is
the part of the periodicity that is patterning rather than composition.

| cell | pLDDT | ESM-2 pppl | amphW | patterning | mean repeat run |
|---|---|---|---|---|---|
| natural proteins | 90.10 | 3.67 | 0.078 | **0.017** | 2.9 |
| DPLM decoder, 500 passes | 87.44 | 4.04 | 0.111 | **0.044** | 4.7 |
| DPLM decoder, 200 passes | 83.60 | 4.77 | 0.101 | 0.036 | 5.8 |
| absorbing, confidence, 500 | 57.64 | 11.58 | 0.096 | 0.029 | 3.1 |
| absorbing, random, 500 | 51.53 | 14.20 | 0.088 | 0.025 | 2.8 |

DPLM's sequences carry about two and a half times the hydrophobic patterning
of natural proteins, not less. Even the much weaker absorbing arms, at
pLDDT 52-58, are above natural. The premise that a masked diffusion model
cannot organise hydrophobicity at the helical period is not supported.

Two readings, and they matter differently:

1. The metric is working and the model really does organise hydrophobicity.
   Then the mechanism Idea B proposes to add is aimed at a deficit that is
   not there, and the quality gap at this scale is elsewhere.
2. DPLM's excess patterning is an artefact of its repetitiveness. Its mean
   longest homopolymer run is 4.7-5.8 against 2.9 for natural, and a
   repetitive sequence produces spurious spectral peaks. On that reading the
   metric is partly measuring degeneracy, which makes it a poor headline
   test in either direction and means a two-level model scoring higher on it
   would not be evidence of success.

Either reading damages the proposal's test. The honest consequence is that
hydropathy periodicity should not be the decisive criterion, and that any
periodicity claim must be reported against both a composition-matched
shuffle and a repetition statistic.

This does not settle whether the two-level alphabet helps. It settles that
the specific diagnosis motivating it is wrong for a strong model. Whether
the premise holds for a small from-scratch model is tested separately, on
the matched 38M arms.
