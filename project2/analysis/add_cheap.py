"""Add newly-defined CPU metrics to an already-scored file, without refolding."""
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common import metrics as M            # noqa: E402
from common.run_score import boot_ci       # noqa: E402


def main(paths):
    for p in paths:
        if not os.path.exists(p):
            print(f"missing {p}")
            continue
        d = json.load(open(p))
        for c in d["cells"]:
            s = c["seqs"]
            ws = [M.amphipathic_window_stats(x) for x in s]
            for k in ("amph_window_frac", "amph_window_max"):
                v = [w[k] for w in ws]
                c[k] = float(np.nanmean(v))
                c[k + "_ci"] = boot_ci(v)
            rg = random.Random(0)
            sh = [''.join(rg.sample(x, len(x))) for x in s]
            vs = [M.amphipathic_window_stats(x)["amph_window_frac"]
                  for x in sh]
            c["amph_window_frac_shuffled"] = float(np.nanmean(vs))
            c["amph_patterning"] = (c["amph_window_frac"]
                                    - c["amph_window_frac_shuffled"])
            c["hmoment"] = float(np.nanmean(
                [M.hydrophobic_moment(x) for x in s]))
        json.dump(d, open(p, "w"))
        print(f"updated {os.path.basename(p)} ({len(d['cells'])} cells)")


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(HERE, "results",
                                       "A_fidelity_scored.json")])
