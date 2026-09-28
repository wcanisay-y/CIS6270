"""Generate protein sequences and write them to a .pt file for metrics.py.

Flow matching: Euler integration of dz/dt = v_theta, t: 0 -> 1

Guidance flags:
    --w       classifier-free guidance strength
    --gate    none | global | shuffled | per_residue (the innovation)
    --beta    gate sharpness
"""
import argparse

import torch

from data.pfam import SEQ_LEN, HIDDEN_DIM, N_FAMILIES
from flow_matching import FlowModel, FamilyClassifier, sample_prior
from guidance import cfg


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    if ck["model"] == "flow":
        model = FlowModel(a["layers"], a["n_heads"], a["cond_drop"])
    elif ck["model"] == "classifier":
        model = FamilyClassifier()
    else:
        raise ValueError(ck["model"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval().requires_grad_(False), ck


def sample_family(family_hist, batch_size, device):
    """Sample family indices from training distribution."""
    return torch.multinomial(family_hist.to(device), batch_size, replacement=True)


@torch.no_grad()
def sample_flow(model, family, steps, w, gate_mode, beta, z_mean, z_std):
    """Sample from flow model using Euler integration.

    Args:
        model: FlowModel
        family: [B] target family indices
        steps: number of Euler steps
        w: guidance strength
        gate_mode: one of ("none", "global", "shuffled", "per_residue")
        beta: gate sharpness
        z_mean, z_std: standardization stats

    Returns:
        z: [B, L, D] final latent states
        nfe: number of function evaluations
        mean_gate: average gate value over trajectory
    """
    b = family.shape[0]
    device = family.device

    z = sample_prior(b, device)
    dt = 1.0 / steps
    nfe = 0
    gate_log = []

    for i in range(steps):
        t = torch.full((b,), i * dt, device=device)
        v, g = cfg(model, z, t, family, w, gate_mode, beta, z_mean, z_std)
        nfe += 2 if w != 0 else 1
        gate_log.append(g.mean().item())
        z = z + dt * v

    return z, nfe, sum(gate_log) / len(gate_log)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=100, help="Euler steps")
    p.add_argument("--w", type=float, default=0.0, help="guidance strength")
    p.add_argument("--gate", default="none",
                   choices=["none", "global", "shuffled", "per_residue"])
    p.add_argument("--beta", type=float, default=1.0, help="gate sharpness")
    p.add_argument("--target-family", type=int, default=None,
                   help="family index to generate (default: sample from training dist)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load(a.ckpt, device)

    z_mean = float(ck["z_mean"])
    z_std = float(ck["z_std"])
    family_hist = ck["family_hist"]
    family_names = ck["family_names"]

    zs, families, nfe_total, gmean = [], [], 0, []
    done = 0

    while done < a.n:
        b = min(a.batch_size, a.n - done)

        if a.target_family is not None:
            family = torch.full((b,), a.target_family, device=device, dtype=torch.long)
        else:
            family = sample_family(family_hist, b, device)

        z, nfe, g = sample_flow(
            model, family, a.steps, a.w, a.gate, a.beta, z_mean, z_std
        )

        zs.append(z.cpu())
        families.append(family.cpu())
        nfe_total = nfe
        gmean.append(g)
        done += b
        print(f"  {done}/{a.n}")

    # Save results
    torch.save({
        "z": torch.cat(zs),
        "family": torch.cat(families),
        "z_mean": z_mean,
        "z_std": z_std,
        "family_names": family_names,
        "nfe": nfe_total,
        "mean_gate": sum(gmean) / len(gmean),
        "config": vars(a),
    }, a.out)

    print(f"saved {done} samples to {a.out} (NFE {nfe_total}, mean gate "
          f"{sum(gmean)/len(gmean):.3f})")


if __name__ == "__main__":
    main()
