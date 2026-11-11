# The conditioning effect replicates on CATH S35, a second classical dataset

Every conditioning number up to now came from SwissProt dev sequences, drawn
from the same corpus the model trains on. That tests the capacity of the
family channel but not whether the effect is a property of that split.

CATH S35 is the classical choice: one of the two canonical fold
classifications, non-redundant at 35 % sequence identity by construction, and
composed of structural DOMAINS rather than full-length UniProt records --
no signal peptides, no linkers, no multi-domain architecture. 3,876 of its
37,350 domains fall in this project's 200-254 window, mean length 224.3
against 225.8 for the SwissProt candidate pool. Length matched, protein
distribution not. UniRef50 was rejected despite being homology-clustered: it
is UniProt-derived like SwissProt and so would not test distribution shift.

Jobs 27447590 (purge), 27447591_[0-2] (retrain), 27447592 (the 2x2). Scripts
`tools/cath_patterns.py`, `run/{cath_split,retrain_cath,cond_cath}.sh`,
analysis `analysis/{clustered,cross_dataset}.py`.

## The CATH run is better controlled than the SwissProt one

| | SwissProt dev | CATH S35 |
|---|---|---|
| sources with a BLAST relative in training | 125/128 = **97.7 %** | 54/128 = **42.2 %** |
| direct relatives found | 19,645 | 2,127 |
| purge depth achieved | 1 (budget exceeded) | **2** (expansion ran) |
| sequences purged | 19,645 = 11.89 % | 6,640 = **4.02 %** |
| validation cost of the purge (flat arm) | 4.6328 -> 4.7536 | 4.6328 -> **4.6505** |

58 % of CATH sources have no detectable relative in training at all, most
likely because CATH domains are excised from parents longer than the 254
training cap, so there is often nothing to align against. The depth-2
single-linkage expansion ran here where it could not for SwissProt, and the
corpus perturbation is three times smaller, so the purged row is barely
handicapped by lost training data -- the main weakness of the SwissProt
comparison.

## Result

| dataset | original corpus | purged corpus | difference |
|---|---|---|---|
| SwissProt dev | +26.09 [+22.89, +29.28] | **+23.38 [+20.07, +26.68]** | +2.71 [+0.95, +4.55] |
| CATH S35 | +21.06 [+17.86, +24.24] | **+19.85 [+16.62, +23.05]** | +1.21 [-0.68, +3.06] |

chemical minus random, paired per source, 128 sources each.

Absolute cells:

| dataset | corpus | chem cond. | chem uncond. | rand cond. | rand uncond. |
|---|---|---|---|---|---|
| SwissProt | original | 62.48 | 36.67 | 36.39 | 35.67 |
| SwissProt | purged | 60.34 | 35.77 | 36.96 | 35.31 |
| CATH S35 | original | 54.93 | 36.67 | **33.87** | 35.67 |
| CATH S35 | purged | 53.47 | 35.44 | **33.62** | 35.49 |

## Reading

**The effect replicates.** +19.85 [+16.62, +23.05] on a dataset the model was
never pointed at, against +23.38 [+20.07, +26.68] on SwissProt. The intervals
overlap. (They are separate bootstraps on different pattern sets, so overlap
is an informal check and not a test of equality.) The conditioning result is
a property of protein chemistry, not of the SwissProt dev split.

**Memorisation is smaller here, as the homology difference predicts.**
+1.21 [-0.68, +3.06] against SwissProt's +2.71 [+0.95, +4.55], and this
interval contains zero. With 42 % of sources having a training relative
rather than 98 %, there is less to memorise, and the measurement says so.
Taken together the two datasets bound the memorisation contribution at
roughly 1-3 pLDDT of a ~20-26 pLDDT effect.

**CATH is about 3.5 pLDDT lower, and the shortfall is interpretable.** Less
homology to training (42 % vs 98 %) accounts for part of it directly. The
rest is plausibly that a CATH domain's chemical pattern specifies a single
compact fold, whereas a full-length SwissProt record's pattern carries a
whole architecture the model can satisfy more loosely. Not tested.

**THE RANDOM CONTROL GOES NEGATIVE, which is new.** On CATH patterns the
random-partition arm is -1.80 against its own unconditional baseline
(33.87 vs 35.67), where on SwissProt it was +0.72. This is the first time the
control has been actively harmful, and it corroborates the mechanism found in
`I_pattern_curve_note.md`: a chemically meaningless pattern costs something
because binding makes a family error unrecoverable, and it costs more on
compact structural domains, which have less slack than full-length records
with their linkers and disordered regions. Note this widens the
chemical-minus-random separation for the right reason -- the control getting
worse, not the chemical arm getting better.

**Foldability is lower on CATH and the qualitative claim survives in form but
weakens in size.** 0.047-0.055 for the conditioned chemical arm against
0.156-0.172 on SwissProt, with every random arm at 0.000. So the conditioned
chemical arm remains the only arm of four producing any foldable sequence on
either dataset, but a 200-residue slice of a CATH domain is harder to fold
confidently than a slice of a full-length protein. Do not quote the
SwissProt foldability as if it were general.

## What to quote

**+19.85 [+16.62, +23.05] on CATH S35 and +23.38 [+20.07, +26.68] on
SwissProt, both with the pattern sources' homologs deleted from training.**
The effect is dataset-independent within these intervals; memorisation
accounts for 1-3 pLDDT.

## Scope, declared

- The GENERATOR is SwissProt-trained in both experiments. This establishes
  that the pattern source can be independent of training, NOT that the effect
  survives a model trained on a different corpus. Training on CATH itself is
  not feasible here: 3,876 eligible domains is far too small for a 38M model
  from scratch.
- blastp single-linkage neighbourhoods, not MMseqs2 clustering (neither
  MMseqs2 nor CD-HIT is installed and running a downloaded MMseqs2 binary was
  not permitted). Fold-level homologs BLAST cannot detect remain in training
  in both experiments.
- Still one model pair, one length, one budget, one structure predictor.
  pLDDT, pTM, pAE and foldability all come from a single ESMFold pass, and
  the one quality metric not read off ESMFold -- ESM-2 pseudo-perplexity --
  does not favour the conditioned arm on either dataset (16.56 conditioned
  against 14.94 unconditional on CATH, i.e. slightly worse). That caveat from
  `G_clean_conditioning_note.md` is unchanged and is if anything sharper here.
