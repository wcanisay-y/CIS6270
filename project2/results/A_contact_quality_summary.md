# Contact-prediction quality vs mask rate (stage 0)

L=200, 12 natural SwissProt sequences, contacts scored against ESMFold
CB-CB < 8 A at |i-j| >= 6 (density 1.23 %). "ALL-MASK NULL" is every residue
masked with cls/eos kept: by construction it carries zero sequence
information, so it is the floor any masked prediction must beat.

| input | AUROC ext | P@L ext | AUROC self | P@L self |
|---|---|---|---|---|
| unmasked           | 0.914 | 0.305 | 0.886 | 0.255 |
| 25 % masked        | 0.700 | 0.027 | 0.879 | 0.247 |
| 50 % masked        | 0.669 | 0.029 | 0.849 | 0.212 |
| 75 % masked        | 0.662 | 0.025 | 0.695 | 0.076 |
| 90 % masked        | 0.643 | 0.020 | 0.701 | 0.035 |
| ALL-MASK NULL      | 0.626 | 0.032 | 0.698 | 0.030 |

ext  = separate frozen ESM-2 650M forward pass + its contact head
       (token_dropout disabled), which is the mechanism the proposal specifies
self = the generator's own (DPLM-150M) attention maps read through the
       ESM-2 150M contact-head regression, at zero extra cost

Two findings:

1. The proposal's specified signal is unusable. The frozen ESM-2 650M head's
   precision@L falls from 0.305 to 0.027 at only 25 % masking, which is at or
   below the all-mask null (0.032). ESM-2 was trained at a 15 % mask rate; it
   has never seen the input distribution the selector would hand it.
2. The free alternative survives twice as far. DPLM's own attentions keep
   P@L 0.247 at 25 % and 0.212 at 50 % masking before collapsing by 75 %.
   DPLM was trained across 0-100 % masking, which is the likely reason.

Consequence for the method: contact-guided selection can only operate in the
part of the trajectory where at least about half the sequence is already
written. That is also where the parallel batches are smallest, so it is the
part of the trajectory with the least to gain.
