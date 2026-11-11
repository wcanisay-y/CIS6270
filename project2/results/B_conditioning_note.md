# The one result that works, and it is chemistry-specific

Idea B's third claim: hand the model a family pattern taken from a real
protein and let it fill identities. Compliance with the pattern is enforced
in code, so the question is whether the resulting sequences are any good,
and whether the chemistry of the partition matters.

Length 200, 48 patterns taken from held-out natural proteins, 100 forward
passes, same arms as everywhere else.

| arm | pLDDT | ESM-2 pppl | hydrophobic periodicity | compliance |
|---|---|---|---|---|
| natural proteins | 90.12 | 3.67 | 0.150 | - |
| chemical families, pattern given | **63.12** | 13.59 | **0.153** | 1.000 |
| chemical families, unconditional | 36.68 | 14.86 | 0.142 | - |
| random groups, pattern given | 36.54 | 15.80 | 0.150 | 1.000 |
| random groups, unconditional | 35.41 | 16.37 | 0.139 | - |

Giving the chemical family pattern is worth **+26.4 pLDDT**. Giving a
random-partition pattern derived from the same proteins is worth **+1.1**.

This is the only place in the study where the chemistry does real work, and
it does a lot of it. A pattern over six chemical families carries enough of
a protein's structure that a 38M model filling identities underneath it
reaches 63 pLDDT, against 37 when generating freely. The same experiment
with an arbitrary partition of identical group sizes does essentially
nothing, which is exactly the control the project needed and the one place
it comes back positive.

The hydrophobic periodicity of the conditioned sequences, 0.153, matches
natural proteins at 0.150, against 0.142 unconditional.

What this does and does not support:

- It supports the family channel as a CONDITIONING INTERFACE. Writing a
  family pattern by hand, from a known fold, a desired transmembrane
  segment or an amphipathic register, and having the model fill identities
  under it, demonstrably works and demonstrably depends on the chemistry.
- It does NOT support the two-level alphabet as a way to improve
  unconditional generation, which is what the proposal is mainly about.
  Unconditionally the chemical arm is 6.4 pLDDT below a plain flat model
  and only 1.3 above a random partition.

The honest reading is that the useful object here is the interface, not the
generative mechanism. The gain comes from a human supplying real structural
information, not from the model organising itself better.

---

## CORRECTION 2026-11-10: the pattern source set is homology-contaminated

The 48 patterns come from the upstream SwissProt dev split, which is not
homology-clustered. Checking each source against the full 165,231-sequence
training corpus by 10-mer containment: 6 of 48 exceed 0.90, 17 of 48 exceed
0.50, 29 of 48 exceed 0.10. No exact sequence matches.

The contamination interacts with the chemical channel specifically:

| arm | containment > 0.50 (n=17) | <= 0.50 (n=31) | difference |
|---|---|---|---|
| chemical pattern | 69.73 | 59.50 | +10.23 [+0.77, +19.18] |
| random pattern | 35.03 | 37.37 | -2.34 [-5.20, +0.38] |
| chemical - random | +34.70 | +22.12 | +12.57 [+2.95, +21.78] |

If contaminated sources were merely easier proteins the random arm would
also gain on them; it does not. A chemically coherent pattern appears to act
as a retrieval key for a memorised fold, and an incoherent one does not.

**What survives.** On the cleaner 31 sources the chemical pattern is still
worth +22.1 over the random pattern and +22.8 over the unconditional
baseline, about 85 % of the headline. The sequences are not copies: identity
to their own source is 0.372 against a 0.308 uniform-within-family floor and
a 0.392 modal-residue floor, with k-mer novelty 0.9997 against training. And
both arms carry the same contamination, so it cannot by itself produce a
26-point gap between them.

**Caveats.** 10-mer containment is not an alignment; the 0.50 cut is
arbitrary and the "clean" subset has median containment about 0.22. An
innocent alternative remains open: contaminated proteins belong to large
conserved families whose patterns may be more structurally determinative,
which is memorisation-free. Separating the two needs a retrain on a
clustered split.

**Consequence.** Do not quote +26.4 without this note. The headline must be
re-measured on a homology-clean pattern set before the pattern-degradation
experiment is worth running.
