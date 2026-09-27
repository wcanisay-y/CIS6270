"""Generate molecules and write them to a .pt file for metrics.py.

Flow matching       Euler integration of dx/dt = v_theta, t: 0 -> 1
Diffusion           DDPM ancestral chain, k: K -> 1

Guidance flags apply to the flow model:
    --w       classifier-free guidance strength
    --gate    none | global | shuffled | per_atom      (the innovation)
    --rho     additive valency-constraint gradient     (ablation control only)
"""
import argparse

import torch

from data.qm9 import H_SCALE, MAX_N
from diffusion import DiffusionModel
from flow_matching import FlowModel, PropertyNet, sample_prior, zero_com
from guidance import cfg, constraint_grad


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    cls = {"flow": FlowModel, "diff": DiffusionModel, "prop": PropertyNet}[ck["model"]]
    if ck["model"] == "prop":
        model = cls(a["nf"], max(a["layers"] - 2, 2))
    elif ck["model"] == "flow":
        model = cls(a["nf"], a["layers"], a["cond_drop"])
    else:
        model = cls(a["nf"], a["layers"], a["cond_drop"], a["steps"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval().requires_grad_(False), ck


def sample_mask(n_hist, b, device):
    n = torch.multinomial(n_hist.to(device), b, replacement=True)
    return (torch.arange(MAX_N, device=device)[None] < n[:, None]).float()


@torch.no_grad()
def sample_flow(model, mask, cond, steps, w, gate, beta, rho, one_sided=False):
    x, h = sample_prior(mask, mask.device)
    dt, nfe = 1.0 / steps, 0
    gate_log = []
    for i in range(steps):
        t = torch.full((mask.shape[0],), i * dt, device=mask.device)
        vx, vh, g = cfg(model, x, h, t, mask, cond, w, gate, beta, H_SCALE, one_sided)
        nfe += 2 if w != 0 else 1
        gate_log.append((g.squeeze(-1) * mask).sum() / mask.sum().clamp_min(1))
        if rho != 0:                      # additive-constraint control row
            gx, gh = constraint_grad(x, h, t, mask, one_sided, H_SCALE)
            vx, vh = vx + rho * gx, vh + rho * gh
        x = zero_com(x + dt * vx, mask)
        h = (h + dt * vh) * mask[..., None]
    return x, h, nfe, torch.stack(gate_log).mean().item()


@torch.no_grad()
def sample_diffusion(model, mask, cond, w):
    x, h = sample_prior(mask, mask.device)
    b, K, nfe = mask.shape[0], model.K, 0
    off, on = torch.zeros(b, device=mask.device), torch.ones(b, device=mask.device)
    for k in range(K, 0, -1):
        t = torch.full((b,), k / K, device=mask.device)
        ex, eh = model.field(x, h, t, mask, cond, off)
        nfe += 1
        if w != 0:
            cx, ch = model.field(x, h, t, mask, cond, on)
            ex, eh, nfe = ex + w * (cx - ex), eh + w * (ch - eh), nfe + 1
        sigma = (1 - model.abar[k]).sqrt()
        mx = (x - model.betas[k] * ex / sigma) / model.alphas[k].sqrt()
        mh = (h - model.betas[k] * eh / sigma) / model.alphas[k].sqrt()
        if k > 1:
            zx, zh = sample_prior(mask, mask.device)
            s = model.post_var[k].sqrt()
            x, h = mx + s * zx, mh + s * zh
        else:
            x, h = mx, mh
        x = zero_com(x, mask)
        h = h * mask[..., None]
    return x, h, nfe, 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--steps", type=int, default=200, help="Euler steps (flow only)")
    p.add_argument("--w", type=float, default=0.0)
    p.add_argument("--gate", default="none",
                   choices=["none", "global", "shuffled", "per_atom"])
    p.add_argument("--beta", type=float, default=1.0, help="gate sharpness")
    p.add_argument("--rho", type=float, default=0.0,
                   help="additive valency-constraint strength (ablation control)")
    p.add_argument("--one-sided", action="store_true",
                   help="penalise over-valence only (HLTF form); default is two-sided")
    p.add_argument("--target", type=float, default=None,
                   help="property value in original units; default = dataset mean")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load(a.ckpt, device)
    mu, sd = ck["prop_mean"].item(), ck["prop_std"].item()
    target = mu if a.target is None else a.target
    cond_val = (target - mu) / sd

    xs, hs, ms, nfe, gmean = [], [], [], 0, []
    done = 0
    while done < a.n:
        b = min(a.batch_size, a.n - done)
        mask = sample_mask(ck["n_hist"], b, device)
        cond = torch.full((b,), cond_val, device=device)
        if ck["model"] == "flow":
            x, h, f, g = sample_flow(model, mask, cond, a.steps, a.w, a.gate,
                                     a.beta, a.rho, a.one_sided)
        else:
            x, h, f, g = sample_diffusion(model, mask, cond, a.w)
        xs.append(x.cpu()); hs.append(h.cpu()); ms.append(mask.cpu())
        nfe, done = f, done + b
        gmean.append(g)
        print(f"  {done}/{a.n}")

    torch.save({"x": torch.cat(xs), "h": torch.cat(hs), "mask": torch.cat(ms),
                "target": target, "cond": cond_val, "nfe": nfe,
                "mean_gate": sum(gmean) / len(gmean),
                "config": vars(a), "family": ck["model"]}, a.out)
    print(f"saved {done} samples to {a.out}  (NFE {nfe}, mean gate "
          f"{sum(gmean)/len(gmean):.3f})")


if __name__ == "__main__":
    main()