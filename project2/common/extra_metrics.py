"""Metrics for the conditioning result that are NOT ESMFold confidence.

Why this file exists. Every number carrying the conditioning claim so far --
pLDDT, pTM, pAE, foldability -- comes from a single ESMFold forward pass, and
per-sequence pTM correlates 0.909 with pLDDT, so that is one axis read four
ways. The only quality metric computed by a different model, ESM-2
pseudo-perplexity, barely moves (0.20 against its own unconditional arm while
pLDDT moves 19.91) and on CATH mildly favours the UNCONDITIONAL arm. A +20
pLDDT effect resting on one predictor's self-confidence is not established.

What each metric here is for:

    nsr            native sequence recovery -- does the generated sequence
                   reproduce the residues of the protein whose pattern it was
                   given? High NSR would mean the pattern is a retrieval key;
                   NSR at the compositional floor means it is a constraint.
    levenshtein    the same question without assuming positional alignment.
    frechet (FPD)  distance between the generated and natural distributions in
                   ESM-2 embedding space -- the protein analogue of FID, and
                   the only metric here that scores a SET rather than a
                   sequence, so it sees mode collapse that per-sequence means
                   cannot.
    mmd_rbf        the same comparison with a kernel two-sample statistic,
                   which unlike FPD assumes nothing Gaussian.

Max-ID (highest identity to any training sequence) is not here: it needs
blastp, so it lives in `tools/max_id.py`, which loads the BLAST+ module. The
structural metrics Ref-TM and Ref-RMSD need coordinates and live in
`tools/ref_tm.py`.

Nothing in this file reads ESMFold. That is the point of it.
"""
import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"


# ------------------------------------------------------- sequence recovery

def nsr(gen, src):
    """Native sequence recovery: positional identity against the source.

    The conditioning runs generate at a fixed length from `source[:length]`
    (hier/run_generate.py), so position i of the generated sequence and
    position i of the source are the same site by construction and no
    alignment is needed or wanted.
    """
    n = min(len(gen), len(src))
    if n == 0:
        return float("nan")
    return sum(1 for i in range(n) if gen[i] == src[i]) / n


def levenshtein(a, b):
    """Edit distance, two-row dynamic programme.

    Equal-length inputs make this Hamming-like, but it is the right statistic
    if a run ever generates at a different length, and it is cheap enough
    (200 x 200 per pair) not to special-case.
    """
    if len(a) < len(b):
        a, b = b, a
    prev = np.arange(len(b) + 1, dtype=np.int32)
    cur = np.empty_like(prev)
    for i, ca in enumerate(a, 1):
        cur[0] = i
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev, cur = cur, prev
    return int(prev[len(b)])


def composition_floor(srcs, gens):
    """The NSR a sequence of the generated composition reaches by chance.

    sum_a p_gen(a) * p_src(a), the expected agreement of two independent
    draws, which is the floor NSR has to beat before it means anything.
    """
    def freq(seqs):
        c = np.zeros(20)
        for s in seqs:
            for ch in s:
                k = AA20.find(ch)
                if k >= 0:
                    c[k] += 1
        return c / max(c.sum(), 1)
    return float(freq(gens) @ freq(srcs))


# ------------------------------------------------------- ESM-2 embeddings

class ESM2Embedder:
    """Mean-pooled ESM-2 650M representations.

    The same frozen backbone the pseudo-perplexity uses, so FPD and MMD are
    measured in a space this project already reports in. Pooling excludes the
    BOS/EOS/pad positions; including them makes every sequence look alike.
    """

    def __init__(self, name="facebook/esm2_t33_650M_UR50D", device="cuda",
                 batch=16):
        from transformers import AutoTokenizer, EsmModel
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = EsmModel.from_pretrained(name).eval().to(device)
        self.device, self.batch = device, batch

    @np.errstate(all="ignore")
    def __call__(self, seqs):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch):
                chunk = list(seqs[i:i + self.batch])
                enc = self.tok(chunk, return_tensors="pt", padding=True)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                h = self.model(**enc).last_hidden_state      # B, T, D
                m = enc["attention_mask"].clone()
                m[:, 0] = 0                                   # drop BOS
                idx = enc["attention_mask"].sum(1) - 1        # EOS position
                m[torch.arange(m.shape[0], device=m.device), idx] = 0
                w = m.unsqueeze(-1).to(h.dtype)
                out.append(((h * w).sum(1) / w.sum(1)).float().cpu().numpy())
        return np.concatenate(out, 0)


# --------------------------------------------- distributional distances

def frechet(X, Y, eps=1e-6):
    """Frechet distance between two Gaussians fitted to embedding sets.

    ||mu_X - mu_Y||^2 + tr(Sx + Sy - 2 (Sx Sy)^{1/2}), i.e. FID transplanted
    to protein embeddings, which the literature calls FPD. With 128 sequences
    and a 1280-dimensional embedding the covariance is rank-deficient, so the
    ridge below is load-bearing rather than cosmetic and the value is only
    comparable between arms measured the same way -- never against a
    published FPD.
    """
    from scipy import linalg
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    mx, my = X.mean(0), Y.mean(0)
    Sx = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    Sy = np.cov(Y, rowvar=False) + eps * np.eye(Y.shape[1])
    covmean, _ = linalg.sqrtm(Sx @ Sy, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mx - my) ** 2).sum() + np.trace(Sx + Sy - 2 * covmean))


def mmd_rbf(X, Y, gamma=None):
    """Unbiased MMD^2 with an RBF kernel, median-heuristic bandwidth.

    Makes no Gaussian assumption, which is why it is reported beside FPD
    rather than instead of it. Zero means the two sets are indistinguishable
    to the kernel; it is non-negative up to sampling noise, so a small
    negative value is noise and is reported as measured.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Z = np.vstack([X, Y])
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    if gamma is None:
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / med if med > 0 else 1.0
    K = np.exp(-gamma * d2)
    n, m = len(X), len(Y)
    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    return float(sxx + syy - 2 * Kxy.mean())


def boot_ci(x, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = np.array([rng.choice(x, len(x), True).mean() for _ in range(n_boot)])
    return float(x.mean()), *[float(v) for v in np.percentile(bs, [2.5, 97.5])]
