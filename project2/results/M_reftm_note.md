# Ref-TM: the conditioned sequences do adopt the source protein's fold

This is the measurement the conditioning result never had, and it is the
strongest result in the project.

Every number behind "+19.9 pLDDT on CATH S35" was ESMFold's confidence in its
own prediction. Confidence is not correctness: a model can be confident about
a structure that has nothing to do with the protein whose chemical pattern it
was handed. Asked plainly earlier today whether any of this was learning how
sequences fold, the honest answer was no -- we were measuring a predictor's
self-assessment and had never compared a generated structure with a real one.

The CATH arm makes the comparison possible, because its pattern sources are
structural DOMAINS with deposited coordinates. 127 of the 128 sources were
downloaded (`tools/fetch_cath_domains.py`, `data/cath_domains/`). Each
generated sequence is folded with ESMFold and structurally aligned to the
crystal structure of the domain whose pattern it was given, with the TM-score
normalised by the generated chain.

Job 27496556, `run/ref_tm.sh`, script `tools/ref_tm.py`, data
`results/M_reftm_K_{orig,purged}.json`, figure
`figures/fig9_independent_metrics` panel D.

## Result, CATH S35 with the sources' homologs purged from training

| arm | Ref-TM | TM >= 0.5 | Ref-RMSD |
|---|---|---|---|
| the source's own slice (ceiling) | 0.861 [0.835, 0.884] | 95.3 % | 1.7 Å |
| **chemical, conditioned** | **0.596 [0.558, 0.635]** | **64.6 %** | **3.8 Å** |
| random partition, conditioned | 0.299 [0.290, 0.309] | 0.8 % | 5.6 Å |
| chemical, unconditional | 0.288 [0.281, 0.294] | 0.0 % | 5.6 Å |
| random partition, unconditional | 0.289 [0.282, 0.295] | 0.0 % | 5.7 Å |

TM >= 0.5 is the conventional same-fold threshold. An unrelated pair of
domains from this very set scores 0.29, which is exactly where all three
control arms sit. The ceiling row is the real sequence folded by ESMFold and
aligned to its own crystal structure, so it is what this pipeline can reach at
all at this length -- 0.861, not 1.0, because a 200-residue slice of a domain
is not the whole domain.

Read against those two anchors, the chemical arm covers **54 % of the distance
from the unrelated-pair background to the ceiling** ((0.596 - 0.288) /
(0.861 - 0.288)); the random partition covers 2 %.

**Nearly two thirds of the conditioned sequences land in the same fold as the
protein whose chemical pattern they were given.** The mean is above the
same-fold threshold. The controls are at the unrelated-pair background, and
0.8 % of the random-partition arm crosses 0.5 against 64.6 % of the chemical
arm.

## Why this is not sequence identity in disguise

The obvious objection is that a conditioned sequence retains ~35 % identity to
its source, which is enough for a structure predictor to recognise the family
and return its known fold. The random-partition control refutes it:

| arm | NSR to source | Ref-TM |
|---|---|---|
| chemical, conditioned | 0.348 | **0.596** |
| random partition, conditioned | **0.393** | 0.299 |

**The random arm recovers MORE of the source's residues and gets the
background TM-score.** More identity, no fold. So what carries the structural
information is the chemical partition specifically, not how many native
residues happen to be reproduced -- and both arms sit below their own
modal-in-family identity floor, so neither is copying
(`L_extra_metrics_note.md` §2).

## What this does and does not establish

**Does.** A six-class chemical pattern read off a real domain is sufficient,
given a small masked-diffusion model that has never seen a BLAST relative of
that domain, to generate a sequence whose predicted structure matches the
domain's crystal structure at the same-fold threshold. The constraint is
chemical, not informational-in-general: the bit-matched random partition with
the same number of bits and higher residue recovery produces nothing.

**Does not.** The generated structure is still an ESMFold PREDICTION; only the
reference is experimental. And ESMFold's own training set was not purged -- it
has seen these CATH domains -- so this shows the generated sequence carries
enough signal for a predictor that knows the fold to recover it, which is
weaker than independent physical evidence. The random-partition control limits
how much of the result that can explain, since ESMFold knows the source for
those sequences too, and they have higher identity to it. An independent
predictor and, ultimately, an experiment would settle the remainder.

Note also that the conditioned arm's pLDDT on CATH is only 53.5. The
confidence is middling while the topology is right, which is itself worth
reporting: on this task pLDDT understates how much structural information the
conditioning carries, and the pLDDT-based foldability flag (0.047 on CATH)
badly understates it.

## The headline, stated carefully

Conditioning a masked-diffusion protein model on a real domain's six-class
chemical pattern yields sequences that are **not copies** (identity below the
modal-in-family floor, Max-ID to training 12 %) and that **adopt the source
domain's fold** (Ref-TM 0.596, 64.6 % above the same-fold threshold), where a
bit-matched random partition yields the unrelated-pair background (0.299,
0.8 %). That is a controllability result with a structural endpoint, and it is
a much stronger claim than the +19.9 pLDDT it grew out of.
