"""Correctness checks for the modality-1 baseline. No dataset download, no training.

Run before any long job, and again after any change to the denoiser or the diffusion process:
    python modality1_test_baseline.py

Checks 4-6 (rotation, translation, permutation) are the ones that matter most: break equivariance
and the model still trains to a plausible-looking loss while producing wrong samples.
"""
import torch, numpy as np
import modality1_diffusion_model_baseline as M

torch.manual_seed(0)
B, N, H = 3, 8, 32
model = M.EGNNDenoiser(hidden=H, layers=3)
model.eval()

sizes = torch.tensor([8, 5, 3])
mask = (torch.arange(N)[None] < sizes[:, None]).float()[..., None]
x = M.remove_mean(torch.randn(B, N, 3), mask)
h = torch.randn(B, N, 5) * mask
t = torch.rand(B)

with torch.no_grad():
    px, ph = model(x, h, t, mask)
print("1. shapes:", tuple(px.shape), tuple(ph.shape), "expected", (B,N,3), (B,N,5))
assert px.shape == (B,N,3) and ph.shape == (B,N,5)

# 2. padded atoms must produce exactly zero
pad = (1 - mask).bool().squeeze(-1)
print("2. padded coord max abs:", float(px[pad].abs().max()), " padded feat max abs:", float(ph[pad].abs().max()))
assert float(px[pad].abs().max()) == 0.0 and float(ph[pad].abs().max()) == 0.0

# 3. zero centre of mass of the coordinate prediction (over real atoms only)
com = (px * mask).sum(1).abs().max()
print("3. |CoM| of prediction:", float(com))
assert float(com) < 1e-4

# 4. rotation equivariance: f(Rx) == R f(x), and features invariant
q, _ = torch.linalg.qr(torch.randn(3,3))
if torch.det(q) < 0: q[:, 0] *= -1
with torch.no_grad():
    rx, rh = model(x @ q.T, h, t, mask)
coord_err = (rx - px @ q.T).abs().max()
feat_err  = (rh - ph).abs().max()
print("4. rotation equivariance err:", float(coord_err), " feature invariance err:", float(feat_err))
assert float(coord_err) < 1e-4 and float(feat_err) < 1e-4

# 5. translation invariance (inputs are re-centred, so a shift must not matter)
with torch.no_grad():
    tx, th = model(M.remove_mean(x + torch.tensor([1.,2.,3.]), mask), h, t, mask)
print("5. translation invariance err:", float((tx-px).abs().max()))
assert float((tx-px).abs().max()) < 1e-4

# 6. permutation equivariance of the first 3 real atoms in molecule 0
perm = torch.arange(N); perm[0], perm[2] = 2, 0
with torch.no_grad():
    ppx, pph = model(x[:, perm], h[:, perm], t, mask[:, perm])
print("6. permutation equivariance err:", float((ppx - px[:, perm]).abs().max()))
assert float((ppx - px[:, perm]).abs().max()) < 1e-4

# 7. schedule: alpha^2 + sigma^2 == 1, monotone
d = M.Diffusion(timesteps=50, device=torch.device("cpu"))
s = d.alpha**2 + d.sigma**2
print("7. alpha^2+sigma^2 range:", float(s.min()), float(s.max()), " alpha[0]:", float(d.alpha[0]), " alpha[-1]:", float(d.alpha[-1]))
assert abs(float(s.min())-1) < 1e-5 and abs(float(s.max())-1) < 1e-5
assert d.alpha[0] > 0.99 and d.alpha[-1] < 0.02
assert bool((d.alpha[1:] <= d.alpha[:-1] + 1e-6).all()), "alpha must be non-increasing"

# 8. loss is finite and backprops
loss = d.loss(model, x, h, mask)
loss.backward()
gnorm = sum(float(p.grad.norm())**2 for p in model.parameters() if p.grad is not None)**0.5
print("8. loss:", float(loss), " grad norm:", gnorm)
assert np.isfinite(float(loss)) and np.isfinite(gnorm) and gnorm > 0

# 9. sampling runs, returns zero-CoM coords and valid type indices
with torch.no_grad():
    sx, st = d.sample(model, mask)
print("9. sample coords finite:", bool(torch.isfinite(sx).all()), " |CoM|:", float((sx*mask).sum(1).abs().max()),
      " type range:", int(st.min()), int(st.max()))
assert torch.isfinite(sx).all() and float((sx*mask).sum(1).abs().max()) < 1e-3
assert 0 <= int(st.min()) and int(st.max()) < 5

# 10. stability metric on a real molecule: methane, C-H = 1.09 A tetrahedral
c = np.array([[0,0,0],[0.629,0.629,0.629],[-0.629,-0.629,0.629],[-0.629,0.629,-0.629],[0.629,-0.629,-0.629]])
types = [1,0,0,0,0]   # C,H,H,H,H
stable, n_stable, n_tot = M.check_stability(c, types)
print("10. methane stable:", stable, f"({n_stable}/{n_tot} atoms)")
assert stable and n_stable == 5

# 11. a deliberately broken molecule must NOT be stable
bad = np.array([[0,0,0],[5,0,0],[0,5,0],[0,0,5],[5,5,5]])
stable_bad, _, _ = M.check_stability(bad, types)
print("11. dissociated molecule stable:", stable_bad, "(expect False)")
assert not stable_bad

# 12. MLP control: same interface, same masking, same zero-CoM -- but NOT equivariant.
mlp = M.build_denoiser("mlp", hidden=64, layers=3, max_atoms=N)
mlp.eval()
with torch.no_grad():
    mx, mh = mlp(x, h, t, mask)
assert mx.shape == (B,N,3) and mh.shape == (B,N,5)
assert float(mx[pad].abs().max()) == 0.0, "MLP must still mask padded atoms"
assert float((mx*mask).sum(1).abs().max()) < 1e-4, "MLP output must still be zero-CoM"
with torch.no_grad():
    rmx, _ = mlp(x @ q.T, h, t, mask)
mlp_err = float((rmx - mx @ q.T).abs().max())
print(f"12. MLP masking + zero-CoM ok; rotation error {mlp_err:.3f} (expected LARGE: not equivariant)")
assert mlp_err > 1e-3, "MLP should NOT be equivariant -- if it is, something is wrong"

# 13. random_rotation is a proper rotation (det +1), so chirality is preserved
R = M.random_rotation(64, torch.device("cpu"))
dets = torch.linalg.det(R)
orth = (R @ R.transpose(1,2) - torch.eye(3)).abs().max()
print(f"13. random_rotation det range [{float(dets.min()):.4f}, {float(dets.max()):.4f}], orthogonality err {float(orth):.2e}")
assert float(dets.min()) > 0.999 and float(orth) < 1e-5

print("\nALL CHECKS PASSED")
