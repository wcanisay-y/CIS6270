#!/usr/bin/env python3
"""Correctness checks for the protein denoiser. Run before any long job; takes seconds.

Every check is a property the training run depends on and that a silent bug would break.
"""
import torch

from sparse_egnn import (N_RESIDUE_TYPES, Diffusion, ProteinDenoiser, build_edges,
                         chirality_feature, remove_mean, sample_zero_com_noise)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def make_batch(lengths, n_nodes=128, seed=0):
    torch.manual_seed(seed)
    batch = len(lengths)
    lengths = torch.tensor(lengths, device=DEVICE)
    mask = (torch.arange(n_nodes, device=DEVICE)[None] < lengths[:, None]).float()[..., None]
    # A self-avoiding-ish walk, so the geometry resembles a chain rather than a gas.
    step = torch.randn(batch, n_nodes, 3, device=DEVICE)
    step = step / step.norm(dim=-1, keepdim=True) * 3.8
    x = remove_mean(torch.cumsum(step, dim=1), mask)
    h = torch.zeros(batch, n_nodes, N_RESIDUE_TYPES, device=DEVICE)
    h.scatter_(2, torch.randint(0, N_RESIDUE_TYPES, (batch, n_nodes, 1), device=DEVICE), 1.0)
    return x, h * mask, mask, lengths


def rotation(seed=0):
    torch.manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, device=DEVICE))
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


print("== geometry features ==")
x, h, mask, lengths = make_batch([128, 96, 41, 16])
chi = chirality_feature(x, mask)
chi_rot = chirality_feature(x @ rotation().T, mask)
mirror = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=DEVICE))
chi_mirror = chirality_feature(x @ mirror, mask)
check("chirality invariant under rotation", torch.allclose(chi, chi_rot, atol=1e-4),
      f"max diff {(chi - chi_rot).abs().max():.2e}")
check("chirality flips sign under reflection", torch.allclose(chi, -chi_mirror, atol=1e-4),
      f"max diff {(chi + chi_mirror).abs().max():.2e}")
check("chirality is non-trivial", chi.abs().max() > 0.05, f"max |chi| {chi.abs().max():.3f}")
check("chirality zero on padding and chain ends",
      float(chi[1, 96:].abs().max()) == 0.0 and float(chi[0, -2:].abs().max()) == 0.0)

print("\n== edge construction ==")
src, dst, is_bb = build_edges(x, mask, lengths, k=16)
n_nodes = x.shape[1]
node_valid = mask.reshape(-1).bool()
check("no edge touches a padded node",
      bool(node_valid[src].all() and node_valid[dst].all()))
check("no self loops", bool((src != dst).all()))
check("no duplicate edges", len(torch.unique(src * (src.max() + 1) + dst)) == src.numel(),
      f"{src.numel():,} edges")
check("no edge crosses two proteins", bool(((src // n_nodes) == (dst // n_nodes)).all()))
expected_bb = 2 * int((lengths - 1).sum())
check("every backbone bond present in both directions", int(is_bb.sum()) == expected_bb,
      f"{int(is_bb.sum())} vs {expected_bb}")
check("backbone edges are adjacent residues",
      bool(((src[is_bb.squeeze(-1) == 1] - dst[is_bb.squeeze(-1) == 1]).abs() == 1).all()))

print("\n== denoiser equivariance ==")
model = ProteinDenoiser(hidden=64, layers=3, k=16).to(DEVICE).eval()
t = torch.rand(len(lengths), device=DEVICE)
with torch.no_grad():
    eps_x, eps_h = model(x, h, t, mask, lengths)
    R = rotation(1)
    eps_x_rot, eps_h_rot = model(x @ R.T, h, t, mask, lengths)
    eps_x_mirror, _ = model(x @ mirror, h, t, mask, lengths)
error = (eps_x @ R.T - eps_x_rot).abs().max()
check("coordinate output is rotation equivariant", error < 1e-3, f"max diff {error:.2e}")
rotation_noise = float(error / eps_x.abs().max())   # the float32 noise floor, relative
check("scalar output is rotation invariant",
      torch.allclose(eps_h, eps_h_rot, atol=1e-3), f"max diff {(eps_h-eps_h_rot).abs().max():.2e}")
# Two things confound a naive reflection test. First, the coordinate head is deliberately
# initialised near the identity with gain 0.001, which suppresses the effect of EVERY input
# feature, chirality included. Second, index_add_ uses GPU atomics, so the summation order
# varies between runs and identical inputs do not give bit-identical outputs. So the test
# measures whether the feature is WIRED IN, using a copy whose coordinate head is scaled to
# normal strength, and compares against the same copy with chirality switched off.
def amplified(use_chirality):
    net = ProteinDenoiser(hidden=64, layers=3, k=16, use_chirality=use_chirality).to(DEVICE).eval()
    net.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        for layer in net.layers:                      # undo the near-identity initialisation
            layer.coord[-1].weight.mul_(1000.0)
    return net


with torch.no_grad():
    live, control = amplified(True), amplified(False)
    l_x, _ = live(x, h, t, mask, lengths)
    l_mirror, _ = live(x @ mirror, h, t, mask, lengths)
    c_x, _ = control(x, h, t, mask, lengths)
    c_mirror, _ = control(x @ mirror, h, t, mask, lengths)
mirror_gap = float((l_x @ mirror - l_mirror).abs().max() / l_x.abs().max())
control_gap = float((c_x @ mirror - c_mirror).abs().max() / c_x.abs().max())
check("reflection invariant once chirality is switched off", control_gap < 1e-3,
      f"control gap {control_gap:.2e} of output scale, atomics noise only")
check("chirality makes the model NOT reflection equivariant",
      mirror_gap > 20 * max(control_gap, 1e-9),
      f"reflection gap {mirror_gap:.2e} vs control {control_gap:.2e} "
      f"({mirror_gap/max(control_gap,1e-12):.0f}x)")

print("\n== masking and independence ==")
check("padded nodes predict exactly zero",
      float(eps_x[2, 41:].abs().max()) == 0.0 and float(eps_h[2, 41:].abs().max()) == 0.0)
com = (eps_x * mask).sum(1).abs().max()
check("coordinate output has zero centre of mass", com < 1e-4, f"max |CoM| {com:.2e}")
with torch.no_grad():
    alone_x, alone_h = model(x[:1], h[:1], t[:1], mask[:1], lengths[:1])
gap = (alone_x - eps_x[:1]).abs().max()
check("no cross-talk between proteins in a batch", gap < 1e-4, f"max diff {gap:.2e}")

print("\n== gradients and diffusion ==")
model.train()
pred_x, pred_h = model(x, h, t, mask, lengths)
(pred_x.square().sum() + pred_h.square().sum()).backward()
grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
check("all gradients finite", bool(torch.isfinite(grads).all()),
      f"{int((~torch.isfinite(grads)).sum())} non-finite of {grads.numel():,}")
# Coincident points are the case that makes the sqrt derivative blow up.
model.zero_grad()
x_degenerate = torch.zeros_like(x)
p_x, p_h = model(x_degenerate, h, t, mask, lengths)
(p_x.square().sum() + p_h.square().sum()).backward()
g2 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
check("gradients finite when all points coincide", bool(torch.isfinite(g2).all()))

diffusion = Diffusion(50, DEVICE)
z_x, z_h, e_x, e_h = diffusion.noise(x, h, mask, torch.full((len(lengths),), 25, device=DEVICE))
# Judged against the input's own floating-point residual: summing 128 coordinates of order
# 100 A cannot cancel to better than about 1e-3, so an absolute 1e-4 bound tests float32,
# not the code. The requirement is that adding noise does not make the residual worse.
input_residual = float((x * mask).sum(1).abs().max())
noise_residual = float((z_x * mask).sum(1).abs().max())
check("forward noise stays in the zero-CoM subspace",
      noise_residual <= max(1e-4, 20 * input_residual),
      f"noise {noise_residual:.2e} vs input {input_residual:.2e}")
check("forward noise respects the mask", float(z_x[2, 41:].abs().max()) == 0.0)
noise = sample_zero_com_noise((4, 128, 3), mask, DEVICE)
check("prior noise is zero-CoM", float((noise*mask).sum(1).abs().max()) < 1e-4)
with torch.no_grad():
    sx, sh, sm = diffusion.sample(model, lengths, 128, DEVICE)
check("sampling returns finite coordinates", bool(torch.isfinite(sx).all()))
sample_residual = float((sx * sm).sum(1).abs().max())
check("sampled coordinates are zero-CoM",
      sample_residual < 1e-2 * max(float(sx.abs().max()), 1e-6),
      f"residual {sample_residual:.2e} at coordinate scale {sx.abs().max():.2e}")
check("sampling respects the mask", float(sx[2, 41:].abs().max()) == 0.0)

# --------------------------------------------------------------------------------------------
# The decisive sampler test. A wrong reverse process still returns finite zero-CoM coordinates,
# so only an oracle reconstruction can tell a correct update from a plausible-looking one.
print("\n== sampler, end-to-end loop (weak check) ==")
x_true, h_true, mask_o, lengths_o = make_batch([128, 96, 64, 32], seed=3)
x_true = remove_mean(x_true / 10.0, mask_o)          # roughly the normalised training scale
h_true = h_true * 0.25
diffusion = Diffusion(200, DEVICE)


def oracle(z_x, z_h, t, mask, lengths, index):
    """The exact noise, given that the true sample is (x_true, h_true)."""
    a, s = diffusion.alpha[index], diffusion.sigma[index]
    if float(s) < 1e-8:
        return torch.zeros_like(z_x), torch.zeros_like(z_h)
    return ((z_x - a * x_true) / s) * mask, ((z_h - a * h_true) / s) * mask


with torch.no_grad():
    rec_x, rec_h, _ = diffusion.sample(None, lengths_o, 128, DEVICE, eps_fn=oracle)
scale = float(x_true.abs().max())
error = float((rec_x - x_true).abs().max())
# NOTE: this only shows the loop runs and converges. It is NOT evidence that the transition
# coefficients are right: an oracle that knows the true sample re-injects it at every step and
# converges even with an inverted ratio. Verified by deliberately injecting that bug. The
# authoritative test is the marginal-consistency block below.
check("oracle sampler reconstructs the coordinates", error < 0.05 * scale,
      f"max error {error:.4f} at coordinate scale {scale:.4f} ({error/scale:.2%})")
recovered_types = (rec_h / 0.25).argmax(-1)
true_types = (h_true / 0.25).argmax(-1)
agreement = float(((recovered_types == true_types).float() * mask_o.squeeze(-1)).sum()
                  / mask_o.sum())
check("oracle sampler reconstructs the residue types", agreement > 0.99,
      f"agreement {agreement:.4f}")
step_true = (x_true[0, 1:int(lengths_o[0])] - x_true[0, :int(lengths_o[0]) - 1]).norm(dim=-1).mean()
step_rec = (rec_x[0, 1:int(lengths_o[0])] - rec_x[0, :int(lengths_o[0]) - 1]).norm(dim=-1).mean()
check("oracle sampler preserves the backbone spacing",
      abs(float(step_rec - step_true)) < 0.02 * float(step_true),
      f"recovered {float(step_rec):.4f} vs true {float(step_true):.4f}")


print("\n== sampler, marginal consistency (the authoritative check) ==")
# The forward marginal q(z_t | x0) composed with one reverse transition must reproduce
# q(z_s | x0) exactly: mean alpha_s * x0, standard deviation sigma_s. This catches coefficient
# errors that the oracle reconstruction above sails straight through; three deliberately
# injected bugs (inverted alpha ratio, wrong posterior variance, missing sigma_s/sigma_t factor)
# were each rejected by this block with deviations of 0.65 to 2.26.
# The expected standard-deviation ratio is not exactly 1: projecting onto the zero-CoM subspace
# removes 3 of 3N degrees of freedom, so it is sqrt((N-1)/N).
n_nodes_m, draws = 40, 20000
mask_m = torch.ones(1, n_nodes_m, 1, device=DEVICE)
x0 = remove_mean(torch.randn(1, n_nodes_m, 3, device=DEVICE), mask_m)
h0 = torch.randn(1, n_nodes_m, N_RESIDUE_TYPES, device=DEVICE) * 0.25
dm = Diffusion(200, DEVICE)
expected = ((n_nodes_m - 1) / n_nodes_m) ** 0.5
worst_mean, worst_std = 0.0, 0.0
for t_index in (200, 100, 50, 10, 2):
    broadcast = mask_m.expand(draws, n_nodes_m, 1)
    eps = sample_zero_com_noise((draws, n_nodes_m, 3), broadcast, DEVICE)
    eps_h = torch.randn(draws, n_nodes_m, N_RESIDUE_TYPES, device=DEVICE)
    a_t, s_t = dm.alpha[t_index], dm.sigma[t_index]
    a_s, s_s = dm.alpha[t_index - 1], dm.sigma[t_index - 1]
    z_s, _ = dm.reverse_step(a_t * x0 + s_t * eps, a_t * h0 + s_t * eps_h,
                             eps, eps_h, broadcast, t_index)
    target = a_s * x0
    worst_mean = max(worst_mean, float((z_s.mean(0, keepdim=True) - target).abs().max()))
    worst_std = max(worst_std, abs(float((z_s - target).std() / s_s.clamp(min=1e-12)) - expected))
check("reverse step reproduces the forward marginal mean", worst_mean < 0.03,
      f"worst mean error {worst_mean:.4f}")
check("reverse step reproduces the forward marginal variance", worst_std < 0.01,
      f"worst deviation {worst_std:.4f} from the expected ratio {expected:.4f}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("failed:", FAIL)
    raise SystemExit(1)
