#!/usr/bin/env bash
# Table 4.6: Modality 2. The invariant idea is "set guidance strength per element
# from a feasibility residual read off the endpoint estimate x_hat_1"; only the
# residual changes. For probability-simplex sequences, replace valency_residual()
# in guidance.py with the per-position entropy of the predicted simplex point.
# Everything else (cfg(), feasibility_gate(), the four gate modes) is reused.
echo "see README section 'Transfer to Modality 2'"
