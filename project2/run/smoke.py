"""End-to-end feasibility gate on a GPU. Answers, in one job:
  1 does DPLM generate protein-like sequences through our two samplers
  2 does the ESM-2 contact head carry signal on a HEAVILY MASKED sequence,
    and does the cheap self-attention variant agree with the external one
  3 does ESMFold give us pLDDT, and what does it cost
  4 do the Idea-B arms train and sample
  5 what does the full grid cost, projected from measured per-unit times
"""
import collections, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from parallel.sample import load_dplm, generate_absorbing, generate_reparam
from parallel.contacts import SelfContacts, validate_transfer
from common import metrics as M
from common.data import load_swissprot

dev = "cuda"
rep = {}
print("=" * 70 + "\n1) DPLM GENERATION\n" + "=" * 70, flush=True)
tok, model = load_dplm(device=dev, half=True, need_attn=True)
prov = SelfContacts(150, device=dev)

t0 = time.time()
s_rep, d_rep = generate_reparam(tok, model, 100, 4, 100, seed=0, device=dev)
print("reparam T=100  %.1fs" % (time.time() - t0))
for s in s_rep[:2]:
    print("   ", s)

for sel, T in [("confidence", 25), ("contact", 25), ("contact_shuffled", 25),
               ("random", 25), ("punt", 25), ("confidence", 100)]:
    t0 = time.time()
    s, d = generate_absorbing(tok, model, 100, 4, T, selector=sel,
                              contact_provider=prov, contact_mode="self",
                              seed=0, device=dev)
    cp = float(np.mean(d["coupled_pair_frac"])) if d["coupled_pair_frac"] else None
    rep["gen_%s_T%d" % (sel, T)] = dict(sec=time.time() - t0,
                                        nfe=d["nfe_gen"], coupled=cp)
    print("%-17s T=%3d %5.1fs nfe=%d coupled_pairs=%s"
          % (sel, T, time.time() - t0, d["nfe_gen"], cp))
    print("   ", s[0])

print("\n" + "=" * 70 + "\n2) CONTACT HEAD UNDER MASKING\n" + "=" * 70,
      flush=True)
tr, va = load_swissprot(max_n=4000)
probe = [s[:120] for s in va[:6] if len(s) >= 120][:4]
del model, prov
torch.cuda.empty_cache()
rows = validate_transfer(probe, device=dev)
agg = collections.defaultdict(list)
for r in rows:
    agg[r["mask_frac"]].append(r)
print("%9s %19s %13s %9s %10s" % ("mask_frac", "spearman(self,ext)",
                                  "topL_overlap", "ext_mean", "self_mean"))
for k in sorted(agg):
    g = agg[k]
    print("%9.2f %19.3f %13.3f %9.4f %10.4f"
          % (k, np.mean([x["spearman"] for x in g]),
             np.mean([x["topL_overlap"] for x in g]),
             np.mean([x["ext_mean"] for x in g]),
             np.mean([x["self_mean"] for x in g])))
rep["contact_transfer"] = rows
torch.cuda.empty_cache()

print("\n" + "=" * 70 + "\n3) ESMFOLD\n" + "=" * 70, flush=True)
t0 = time.time()
fold = M.ESMFoldScorer(device=dev)
print("load %.0fs" % (time.time() - t0), flush=True)
test = s_rep[:4] + probe[:2]
t0 = time.time()
r = fold.fold(test, batch_size=3)
dt = time.time() - t0
print("folded %d seqs in %.1fs = %.2fs/seq" % (len(test), dt, dt / len(test)))
for s, p in zip(test, r["plddt"]):
    print("  pLDDT %6.2f  L=%d  %s" % (p, len(s), s[:30]))
rep["fold_sec_per_seq"] = dt / len(test)
rep["plddt_gen"] = [float(x) for x in r["plddt"][:4]]
rep["plddt_natural"] = [float(x) for x in r["plddt"][4:]]
del fold
torch.cuda.empty_cache()

print("\n" + "=" * 70 + "\n4) IDEA-B ARMS TRAIN AND SAMPLE\n" + "=" * 70,
      flush=True)
import torch.nn.functional as F
from hier.model import MDM, ARLM, count_params, loss_flat, loss_hier
from hier.sample import sample_flat, sample_hier, sample_ar
from hier.train import encode, batches
from common.alphabet import PARTITIONS, family_table
names, lut = family_table(PARTITIONS["dms6"])
lt = torch.as_tensor(lut, dtype=torch.long, device=dev)
X = encode(tr, 256)
for arm in ("flat", "hier", "ar"):
    m = (ARLM().to(dev) if arm == "ar" else
         MDM(n_fam=(6 if arm == "hier" else 0),
             fam_lut=(lut if arm == "hier" else None)).to(dev))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    it = batches(X, 32, np.random.default_rng(0), dev, 256)
    t0 = time.time()
    for i in range(30):
        x, pad = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if arm == "flat":
                l, _ = loss_flat(m, x, pad)
            elif arm == "hier":
                l, _ = loss_hier(m, x, lt, 6, pad)
            else:
                bos = torch.full((x.shape[0], 1), 20, dtype=torch.long,
                                 device=dev)
                lg = m(torch.cat([bos, x[:, :-1]], 1))
                ce = F.cross_entropy(lg.transpose(1, 2), x, reduction="none")
                l = (ce * ~pad).sum() / (~pad).sum()
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
    sec = (time.time() - t0) / 30
    m.eval()
    t0 = time.time()
    if arm == "flat":
        out = sample_flat(m, 4, 100, 25, device=dev)
    elif arm == "hier":
        out, fp = sample_hier(m, 4, 100, 25, device=dev)
    else:
        out = sample_ar(m, 4, 100, device=dev)
    print("%-5s params %5.2fM  loss %6.3f  %5.0f ms/step  sample %4.1fs"
          % (arm, count_params(m) / 1e6, float(l), sec * 1000,
             time.time() - t0))
    print("      " + out[0])
    rep["train_%s_ms_per_step" % arm] = sec * 1000
    del m, opt
    torch.cuda.empty_cache()

print("\n" + "=" * 70 + "\n5) COST PROJECTION\n" + "=" * 70, flush=True)
fs = rep["fold_sec_per_seq"]
grid_a = 7 * 6 * 48 + 6 * 48
print("Idea A: %d seqs to fold -> %.0f min ESMFold" % (grid_a, grid_a * fs / 60))
ms = rep["train_flat_ms_per_step"]
print("Idea B: 8 arms x 12000 steps x %.0f ms = %.1f GPU-h (%.1f h on 4 GPUs)"
      % (ms, 8 * 12000 * ms / 1000 / 3600, 8 * 12000 * ms / 1000 / 3600 / 4))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(rep, open(os.path.join(HERE, "results/smoke.json"), "w"),
          indent=2, default=str)
print("\nSMOKE OK", flush=True)
