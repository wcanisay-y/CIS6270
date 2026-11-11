"""Prove per-sequence pTM is batch-invariant after the metrics.py fix.

HF's ESMFold exposes `ptm` from openfold's compute_tm, whose final argmax
runs over the WHOLE batch and returns one scalar per forward call -- the
best-scoring alignment of the best sequence present. Reading it per
sequence gave every member of a fold batch the batch MAXIMUM. metrics.py
now recomputes pTM per sequence from ptm_logits; this checks that the value
no longer depends on who else is in the batch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from common.metrics import ESMFoldScorer

gen = json.load(open("results/C_revised_self.json"))
seqs = [s for c in gen["cells"] if c.get("T") for s in c["seqs"]][:8]
sc = ESMFoldScorer(device="cuda")
a = sc.fold(seqs, batch_size=8)
b = sc.fold(seqs, batch_size=1)
pa, pb = np.array(a["ptm"]), np.array(b["ptm"])
print("batch=8 ptm :", np.round(pa, 4))
print("batch=1 ptm :", np.round(pb, 4))
print("max abs diff: %.6f" % np.abs(pa - pb).max())
print("distinct values in the batch=8 run: %d of %d"
      % (len(set(np.round(pa, 6))), len(seqs)))
print("plddt       :", np.round(a["plddt"], 1))
print("per-sequence corr(pLDDT, pTM) = %.3f  (was 0.147 with the batch bug)"
      % np.corrcoef(a["plddt"], pa)[0, 1])
assert np.abs(pa - pb).max() < 5e-3, "NOT batch-invariant"   # fp nondeterminism across batch shapes is ~1e-3
assert len(set(np.round(pa, 6))) > 1, "all sequences still share one pTM"
print("PASS: pTM is per-sequence and batch-invariant")
