"""Baseline E(3)-equivariant diffusion model for 3D molecule generation (CIS 6270 Project 1).

Modality 1: 3D molecule coordinates from https://huggingface.co/datasets/structure-epflai/qm9
Reimplements the EDM recipe (Hoogeboom et al., ICML 2022, arXiv:2203.17003): an EGNN denoiser,
variance-preserving diffusion on the zero-center-of-mass subspace, atom types carried as scaled
one-hot vectors, epsilon-prediction with an L2 loss.

Preprocessing comes from modality1_data.py, which is src/data/qm9.py from the benchmarks branch
with bulk column reads and an optional uncharacterized-molecule filter.

Run (smoke test, CPU, a few minutes):
    python modality1_diffusion_model_baseline.py --smoke
Run (full, one GPU):
    python modality1_diffusion_model_baseline.py --epochs 300 --batch-size 64
Submit on PARCC:
    sbatch modality1_diffusion_model_baseline.sbatch
Install: pip install torch datasets numpy    # rdkit optional, enables validity/uniqueness

Outputs into --output: results.pt (weights, config, metrics), samples.xyz, metrics.json.

One caveat worth repeating from modality1_data.py: this dataset has a single "train" split and
includes the 3,054 molecules that failed QM9's geometry consistency check. The script makes its
own deterministic split, so it is internally sound, but the numbers are NOT directly comparable
to published EDM/GCDM results, which use the Cormorant split (100,000/17,748/13,083 after
removing those 3,054). Pass --uncharacterized with a local uncharacterized.txt to get closer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from modality1_data import ATOM_SYMBOLS as ATOMS, QM9_MAX_ATOMS as MAX_ATOMS, prepare_qm9

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM_X, NORM_H = 1.0, 4.0                   # EDM's normalize_factors for [coords, one-hot].
                                            # Dividing h by 4 puts atom types on a smaller scale
                                            # than coordinates, so the denoiser fixes rough
                                            # geometry before atom identity. EDM Table 10 shows
                                            # this is load-bearing: 82.0% vs 46.9% mol stability.

# Covalent bond-length tables in picometres, from EDM's qm9/bond_analyze.py, restricted to QM9's
# elements. Used only by the stability metric, never during training.
BONDS1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}
BONDS2 = {"C": {"C": 134, "N": 129, "O": 120}, "N": {"C": 129, "N": 125, "O": 121},
          "O": {"C": 120, "N": 121, "O": 121}}
BONDS3 = {"C": {"C": 120, "N": 116, "O": 113}, "N": {"C": 116, "N": 110}, "O": {"C": 113}}
MARGIN1, MARGIN2, MARGIN3 = 10, 5, 3
ALLOWED_BONDS = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}


# 1. Noise schedule: EDM's polynomial_2. alphas2[t] is alpha_bar_t^2, so alpha^2 + sigma^2 = 1.
def polynomial_schedule(timesteps, s=1e-5, power=2.0):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2
    # Clip each step's ratio so alpha never collapses too fast; EDM's clip_noise_schedule.
    alphas_step = alphas2[1:] / np.maximum(alphas2[:-1], 1e-12)
    alphas_step = np.clip(alphas_step, 0.001, 1.0)
    alphas2 = np.concatenate([np.ones(1), np.cumprod(alphas_step)])
    precision = 1 - 2 * s
    return torch.tensor(precision * alphas2 + s, dtype=torch.float32)


# 2. Zero-CoM subspace. EDM's likelihood lives on {x : sum_i x_i = 0}, which is what makes the
#    model translation invariant. Every coordinate tensor -- data, noise, and network output --
#    must be projected, counting only real atoms.
def remove_mean(x, mask):
    total = (x * mask).sum(1, keepdim=True)
    count = mask.sum(1, keepdim=True).clamp_min(1)
    return (x - total / count) * mask


def sample_zero_com_noise(shape, mask, device):
    return remove_mean(torch.randn(shape, device=device), mask)


# 3. EGNN denoiser. Messages see only squared interatomic distances, so the scalar features are
#    rotation invariant and the coordinate update is rotation equivariant.
class EGNNLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.node = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
        self.coord = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 1, bias=False))
        self.attention = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        nn.init.xavier_uniform_(self.coord[-1].weight, gain=0.001)  # Start near the identity.

    def forward(self, h, x, mask, edge_mask):
        difference = x[:, :, None] - x[:, None]                 # [B, N, N, 3]
        square = (difference ** 2).sum(-1, keepdim=True)        # [B, N, N, 1], rotation invariant
        pair = torch.cat([h[:, :, None].expand(-1, -1, h.shape[1], -1),
                          h[:, None].expand(-1, h.shape[1], -1, -1), square], dim=-1)
        message = self.edge(pair) * edge_mask
        message = message * self.attention(message)
        # Normalising the displacement keeps the coordinate update stable when atoms are close.
        # The 1e-8 inside the square root is load-bearing, not cosmetic: d(sqrt)/dx is infinite
        # at zero, and every diagonal pair (i == j) and every padded pair has distance exactly
        # zero. Without it the forward pass is finite but the backward pass is all NaN.
        weight = self.coord(message) * edge_mask
        shift = (difference / ((square + 1e-8).sqrt() + 1.0) * weight).sum(2)
        x = x + shift * mask
        h = h + self.node(torch.cat([h, message.sum(2)], dim=-1)) * mask
        return h, x


class EGNNDenoiser(nn.Module):
    """Predicts the noise added to coordinates and to scaled one-hot atom types."""

    def __init__(self, hidden=256, layers=9):
        super().__init__()
        self.embed = nn.Linear(len(ATOMS) + 1, hidden)          # atom-type features plus time
        self.layers = nn.ModuleList(EGNNLayer(hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, len(ATOMS))

    def forward(self, z_x, z_h, t, mask):
        edge_mask = mask[:, :, None] * mask[:, None]            # [B, N, N, 1]
        edge_mask = edge_mask * (1 - torch.eye(mask.shape[1], device=mask.device)[None, :, :, None])
        time = t.view(-1, 1, 1).expand(-1, mask.shape[1], 1)
        h = self.embed(torch.cat([z_h, time], dim=-1)) * mask
        x = z_x
        for layer in self.layers:
            h, x = layer(h, x, mask, edge_mask)
        # The network's coordinate output is a displacement from its input; the difference is the
        # equivariant part, and re-projecting keeps the prediction inside the zero-CoM subspace.
        return remove_mean(x - z_x, mask), self.head(h) * mask


class MLPDenoiser(nn.Module):
    """Non-equivariant control: flatten the padded molecule and regress the noise with an MLP.

    This is deliberately the simplest thing that fits the same interface, and it is NOT
    rotation equivariant. Rotating a molecule gives a different prediction, so the model has to
    learn 3D geometry separately for every orientation it sees. With rotation augmentation and
    enough data that is learnable in principle -- AlphaFold3 and ADiT both drop hard equivariance
    -- but at QM9 scale in a few CPU hours, expect molecule stability near zero. Its value here
    is as a control that isolates how much the EGNN's equivariance is actually contributing, not
    as a competitive baseline.

    Masking and the zero-CoM projection are still applied, so the diffusion process is identical
    to the EGNN arm and the two are directly comparable.
    """

    def __init__(self, hidden=256, layers=4, max_atoms=MAX_ATOMS):
        super().__init__()
        self.max_atoms = max_atoms
        width = max_atoms * (3 + len(ATOMS))
        blocks = [nn.Linear(width + 1, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.SiLU()]
        blocks += [nn.Linear(hidden, width)]
        self.net = nn.Sequential(*blocks)

    def forward(self, z_x, z_h, t, mask):
        flat = torch.cat([(z_x * mask).flatten(1), (z_h * mask).flatten(1), t[:, None]], dim=-1)
        out = self.net(flat)
        pred_x = out[:, : self.max_atoms * 3].view(-1, self.max_atoms, 3)
        pred_h = out[:, self.max_atoms * 3:].view(-1, self.max_atoms, len(ATOMS))
        return remove_mean(pred_x, mask), pred_h * mask


def build_denoiser(kind, hidden, layers, max_atoms=MAX_ATOMS):
    """Both arms take (z_x, z_h, t, mask) and return (eps_x, eps_h), so the diffusion process,
    the loss and the sampler are shared and the comparison is like for like."""
    if kind == "egnn":
        return EGNNDenoiser(hidden, layers)      # max_atoms is irrelevant: it is size-agnostic
    if kind == "mlp":
        return MLPDenoiser(hidden, layers, max_atoms)
    raise ValueError(f"unknown denoiser {kind!r}")


def random_rotation(batch, device):
    """Uniform SO(3) via QR, with the reflection removed so chirality is preserved."""
    q, r = torch.linalg.qr(torch.randn(batch, 3, 3, device=device))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    flip = torch.det(q) < 0
    q[flip, :, 0] = -q[flip, :, 0]
    return q


# 4. Diffusion. Forward: z_t = alpha_t * z_0 + sigma_t * eps. Training regresses eps.
class Diffusion:
    def __init__(self, timesteps=1000, device=DEVICE):
        self.T = timesteps
        alphas2 = polynomial_schedule(timesteps).to(device)
        self.alpha = alphas2.sqrt()
        self.sigma = (1 - alphas2).sqrt()

    def loss(self, model, x, h, mask):
        batch = len(x)
        t = torch.randint(0, self.T + 1, (batch,), device=x.device)
        alpha = self.alpha[t].view(-1, 1, 1)
        sigma = self.sigma[t].view(-1, 1, 1)
        eps_x = sample_zero_com_noise(x.shape, mask, x.device)
        eps_h = torch.randn_like(h) * mask
        z_x = alpha * (x / NORM_X) + sigma * eps_x
        z_h = alpha * (h / NORM_H) + sigma * eps_h
        pred_x, pred_h = model(z_x, z_h, t.float() / self.T, mask)
        error = ((pred_x - eps_x) ** 2).sum(-1) + ((pred_h - eps_h) ** 2).sum(-1)
        # Mean over real atoms only; padded slots contribute nothing.
        return (error * mask.squeeze(-1)).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, model, mask):
        shape = (len(mask), mask.shape[1], 3)
        z_x = sample_zero_com_noise(shape, mask, mask.device)
        z_h = torch.randn(len(mask), mask.shape[1], len(ATOMS), device=mask.device) * mask
        for step in range(self.T, 0, -1):
            s, t = step - 1, step
            alpha_ts = self.alpha[t] / self.alpha[s]
            sigma2_ts = self.sigma[t] ** 2 - alpha_ts ** 2 * self.sigma[s] ** 2
            pred_x, pred_h = model(z_x, z_h, torch.full((len(mask),), t / self.T,
                                                        device=mask.device), mask)
            mean_x = z_x / alpha_ts - (sigma2_ts / (alpha_ts * self.sigma[t])) * pred_x
            mean_h = z_h / alpha_ts - (sigma2_ts / (alpha_ts * self.sigma[t])) * pred_h
            noise_scale = (sigma2_ts.clamp_min(0).sqrt() * self.sigma[s] / self.sigma[t])
            z_x = mean_x + noise_scale * sample_zero_com_noise(shape, mask, mask.device)
            z_h = mean_h + noise_scale * torch.randn_like(z_h) * mask
            z_x = remove_mean(z_x, mask)
        # Final denoise at t=0, then undo the feature scaling.
        pred_x, pred_h = model(z_x, z_h, torch.zeros(len(mask), device=mask.device), mask)
        x = (z_x - self.sigma[0] * pred_x) / self.alpha[0] * NORM_X
        h = (z_h - self.sigma[0] * pred_h) / self.alpha[0] * NORM_H
        return remove_mean(x, mask), h.argmax(-1)


# 5. Stability metric: infer bond orders from interatomic distances, then check each atom's
#    valency against its element. This is EDM's definition, tables and margins included.
def bond_order(a, b, distance_angstrom):
    picometres = 100 * distance_angstrom
    if picometres >= BONDS1[a][b] + MARGIN1:
        return 0
    if a in BONDS2 and b in BONDS2[a] and picometres < BONDS2[a][b] + MARGIN2:
        if a in BONDS3 and b in BONDS3[a] and picometres < BONDS3[a][b] + MARGIN3:
            return 3
        return 2
    return 1


def check_stability(coords, types):
    symbols = [ATOMS[i] for i in types]
    valency = [0] * len(symbols)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            order = bond_order(symbols[i], symbols[j], distance)
            valency[i] += order
            valency[j] += order
    stable = [v == ALLOWED_BONDS[s] for v, s in zip(valency, symbols)]
    return all(stable), sum(stable), len(symbols)


def to_smiles(coords, types):
    """Build an RDKit molecule from inferred bonds. Returns None if RDKit is absent or the
    molecule fails sanitisation. Largest fragment only, as in EDM."""
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return None
    RDLogger.DisableLog("rdApp.*")
    order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    mol = Chem.RWMol()
    symbols = [ATOMS[i] for i in types]
    for symbol in symbols:
        mol.AddAtom(Chem.Atom(symbol))
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            order = bond_order(symbols[i], symbols[j],
                               float(np.linalg.norm(coords[i] - coords[j])))
            if order:
                mol.AddBond(i, j, order_map[order])
    try:
        Chem.SanitizeMol(mol)
        fragments = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest = max(fragments, key=lambda m: m.GetNumAtoms())
        return Chem.MolToSmiles(largest)
    except Exception:
        return None


def evaluate(molecules, reference_smiles=None):
    atoms_stable = atoms_total = molecules_stable = 0
    valid = []
    for coords, types in molecules:
        is_stable, n_stable, n_total = check_stability(coords, types)
        molecules_stable += int(is_stable)
        atoms_stable += n_stable
        atoms_total += n_total
        smiles = to_smiles(coords, types)
        if smiles is not None:
            valid.append(smiles)
    metrics = {
        "n_molecules": len(molecules),
        "atom_stability": atoms_stable / max(atoms_total, 1),
        "molecule_stability": molecules_stable / max(len(molecules), 1),
    }
    if valid or reference_smiles is not None:
        unique = set(valid)
        metrics["validity"] = len(valid) / max(len(molecules), 1)
        metrics["uniqueness"] = len(unique) / max(len(valid), 1)
        if reference_smiles:
            metrics["novelty"] = len(unique - reference_smiles) / max(len(unique), 1)
    else:
        metrics["note"] = "install rdkit for validity/uniqueness/novelty"
    return metrics


def write_xyz(path, molecules):
    lines = []
    for coords, types in molecules:
        lines.append(str(len(types)))
        lines.append("")
        for position, index in zip(coords, types):
            lines.append(f"{ATOMS[index]} {position[0]:.4f} {position[1]:.4f} {position[2]:.4f}")
    path.write_text("\n".join(lines) + "\n")


# 6. Training loop. AdamW with EDM's settings; the EMA weights are what gets evaluated.
#    Checkpoints every epoch so a Slurm walltime kill costs one epoch, not the whole run.
def train(model, diffusion, data, args):
    loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers)
    valid_loader = DataLoader(data["valid"], batch_size=args.batch_size, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-12)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []
    start_epoch = 0

    checkpoint_path = args.output / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema = {k: v.to(DEVICE) for k, v in state["ema"].items()}
        history = state["history"]
        start_epoch = state["epoch"]
        print(f"resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = count = 0.0
        for x, h, mask in loader:
            x, h, mask = x.to(DEVICE), h.to(DEVICE), mask.to(DEVICE)
            x = remove_mean(x, mask)
            if args.augment:
                # Only meaningful for a non-equivariant denoiser. The EGNN is equivariant by
                # construction, so rotating its input changes nothing it can learn from.
                x = x @ random_rotation(len(x), x.device).transpose(1, 2) * mask
            loss = diffusion.loss(model, x, h, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    if value.dtype.is_floating_point:
                        ema[key].mul_(args.ema).add_(value.detach(), alpha=1 - args.ema)
                    else:
                        ema[key].copy_(value)
            total += float(loss.detach())
            count += 1
        model.eval()
        with torch.no_grad():
            validation = [float(diffusion.loss(model, remove_mean(x.to(DEVICE), mask.to(DEVICE)),
                                               h.to(DEVICE), mask.to(DEVICE)))
                          for x, h, mask in valid_loader]
        record = {"epoch": epoch + 1, "train_loss": total / max(count, 1),
                  "valid_loss": float(np.mean(validation)) if validation else float("nan")}
        history.append(record)
        print(f"epoch {record['epoch']}: train {record['train_loss']:.4f} "
              f"valid {record['valid_loss']:.4f}", flush=True)
        # Write to a temporary file then rename: torch.save is not atomic, and a kill partway
        # through would otherwise leave an unloadable checkpoint.
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "ema": ema, "history": history, "epoch": epoch + 1}, temporary)
        temporary.replace(checkpoint_path)
    return ema, history


def generate(model, diffusion, size_distribution, count, batch_size=100):
    molecules = []
    for start in range(0, count, batch_size):
        n = min(batch_size, count - start)
        sizes = torch.multinomial(size_distribution, n, replacement=True)
        mask = (torch.arange(MAX_ATOMS)[None] < sizes[:, None]).float()[..., None].to(DEVICE)
        coords, types = diffusion.sample(model, mask)
        for i in range(n):
            keep = int(sizes[i])
            molecules.append((coords[i, :keep].cpu().numpy(), types[i, :keep].cpu().tolist()))
    return molecules


# 7. Entry point.
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--denoiser", choices=("egnn", "mlp"), default="egnn",
                        help="egnn is the real baseline; mlp is a non-equivariant control")
    parser.add_argument("--augment", action="store_true",
                        help="random SO(3) rotation per batch; only useful with --denoiser mlp")
    parser.add_argument("--canonical", action="store_true",
                        help="rotate every molecule into its inertial frame; the alternative to "
                             "--augment for a non-equivariant denoiser, and they are mutually "
                             "exclusive in spirit -- canonicalising then re-randomising undoes it")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="subsample the dataset")
    parser.add_argument("--uncharacterized", type=Path, default=None,
                        help="path to QM9's uncharacterized.txt; excludes the 3,054 bad geometries")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true",
                        help="continue from checkpoint.pt in --output if it exists")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny config that runs end to end in minutes")
    parser.add_argument("--output", type=Path,
                        default=ROOT.parent / "results" / "modality1_diffusion_model_baseline")
    args = parser.parse_args()
    if args.smoke:
        args.epochs, args.hidden, args.layers = 2, 64, 3
        args.denoiser = args.denoiser
        args.timesteps, args.samples, args.limit = 100, 50, 2000
        args.workers = 0
    if min(args.epochs, args.batch_size, args.timesteps, args.samples) < 1:
        parser.error("epochs, batch-size, timesteps and samples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if DEVICE.type == "cuda":
        # TF32 silently costs ~2e-3 relative accuracy in the coordinate MLPs, which is where the
        # zero-CoM invariant is maintained. Keep matmuls in true fp32.
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"device: {torch.cuda.get_device_name(0)}  arch_list: {torch.cuda.get_arch_list()}")
    else:
        torch.set_num_threads(4)
        print("device: cpu")

    if args.canonical and args.augment:
        parser.error("--canonical and --augment cancel out; pick one")
    data = prepare_qm9(max_atoms=MAX_ATOMS, n_molecules=args.limit, seed=args.seed,
                       uncharacterized_txt=args.uncharacterized, canonical=args.canonical)
    size_distribution = data["size_distribution"]
    print(f"train {len(data['train'])}  valid {len(data['valid'])}  test {len(data['test'])}")

    model = build_denoiser(args.denoiser, args.hidden, args.layers, MAX_ATOMS).to(DEVICE)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"denoiser: {args.denoiser}  parameters: {parameters:,}"
          f"  augment: {args.augment}  canonical: {args.canonical}")
    if args.denoiser == "mlp" and not args.augment:
        print("note: an MLP denoiser is not rotation equivariant; without --augment it sees "
              "each molecule in one fixed orientation only")
    diffusion = Diffusion(args.timesteps, DEVICE)

    ema, history = train(model, diffusion, data, args)
    model.load_state_dict(ema)              # Evaluate the EMA weights, as EDM does.
    model.eval()

    print(f"sampling {args.samples} molecules at {args.timesteps} steps...")
    molecules = generate(model, diffusion, size_distribution.to(DEVICE), args.samples)

    # Novelty reference: SMILES of the training split, built with the same bond inference so the
    # comparison is like for like. Capped at 20,000 molecules to keep this under a minute.
    reference = set()
    tensors = data["train_tensors"]
    for i in range(min(len(tensors["pos"]), 20000)):
        keep = int(tensors["mask"][i].sum())
        smiles = to_smiles(tensors["pos"][i, :keep].numpy(),
                           tensors["atom_type"][i, :keep].tolist())
        if smiles is not None:
            reference.add(smiles)
    print(f"novelty reference: {len(reference)} unique training SMILES")

    metrics = evaluate(molecules, reference)
    print(json.dumps(metrics, indent=2))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(
        {"metrics": metrics, "config": {k: str(v) for k, v in vars(args).items()},
         "parameters": parameters, "history": history}, indent=2))
    write_xyz(args.output / "samples.xyz", molecules)
    torch.save({"ema": ema, "config": vars(args), "metrics": metrics, "history": history,
                "size_distribution": size_distribution.cpu()}, args.output / "results.pt")
    print(f"saved weights, samples and metrics to {args.output}")


if __name__ == "__main__":
    main()
