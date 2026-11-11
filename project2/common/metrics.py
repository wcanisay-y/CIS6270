"""Evaluation metrics for generated protein sequences.

Primary (field standard for de novo sequence generation):
    plddt            mean per-residue pLDDT from ESMFold
    pppl             ESM-2 650M masked pseudo-perplexity (exact, batched)
Idea-B specific:
    hydropathy_power normalised FFT power of the Kyte-Doolittle trace at the
                     alpha-helix period (3.6 res/cycle) and the beta period
    buried_hydrophobic  fraction of buried positions that are hydrophobic
Set-level:
    uniqueness, diversity (1 - mean pairwise identity), novelty (k-mer),
    aa composition JSD vs natural, max homopolymer run
"""
import numpy as np
import torch

from .alphabet import AA20, AA_IDX, KD_VEC, hydropathy_trace, FAMILIES_2, family_table

try:
    from transformers.models.esm.openfold_utils.loss import \
        compute_tm as _compute_tm
except Exception:                                     # pragma: no cover
    _compute_tm = None


def virtual_cb(n, ca, c):
    """Standard virtual CB from backbone N, CA, C (trRosetta/AF convention).

    Needed for glycine, which has no CB, so that a contact map is defined at
    every position rather than silently reading an unset atom slot.
    """
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


# ------------------------------------------------------------------ folding
class ESMFoldScorer:
    """ESMFold v1 wrapper. Returns mean pLDDT and coordinates."""

    def __init__(self, device="cuda", dtype=torch.float16, chunk_size=64):
        from transformers import AutoTokenizer, EsmForProteinFolding
        self.tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True)
        self.model = self.model.eval().to(device)
        # esm trunk in fp16 saves a lot of memory; the folding head stays fp32
        self.model.esm = self.model.esm.half()
        self.model.trunk.set_chunk_size(chunk_size)
        self.device = device

    @torch.no_grad()
    def fold(self, seqs, batch_size=4):
        """-> plddt, ptm, pae, foldability flag, per-residue plddt, coords.

        Foldability follows the composite criterion used by the protein
        masked-diffusion decoding-order papers: pLDDT > 80 and pTM > 0.7 and
        pAE < 10. Reporting it alongside mean pLDDT matters because pLDDT
        alone can be high for a confidently predicted single long helix.
        """
        out = {"plddt": [], "ptm": [], "pae": [], "plddt_res": [],
               "ca": [], "cb": [], "o_atom": []}
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            enc = self.tok(chunk, return_tensors="pt",
                           add_special_tokens=False, padding=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            o = self.model(**enc)
            mask = enc["attention_mask"].bool()
            pl = o["plddt"][..., 1]                      # CA atom
            # This port returns pLDDT in [0, 1]; the literature reports it
            # on [0, 100]. Rescale so our numbers are directly comparable
            # with published tables.
            if float(pl.max()) <= 1.5:
                pl = pl * 100.0
            ptm = o.get("ptm")
            ptm_logits = o.get("ptm_logits")
            pae = o.get("predicted_aligned_error")
            for b, sq in enumerate(chunk):
                n = len(sq)
                m = mask[b][:n]
                v = pl[b][:n][m].float().cpu().numpy()
                out["plddt_res"].append(v)
                out["plddt"].append(float(v.mean()))
                # HF exposes `ptm` from openfold's compute_tm, which ends
                # in `argmax` over the WHOLE batch and returns a single
                # scalar per forward call -- the best-scoring alignment of
                # the best sequence in the batch. Reading it per sequence
                # gives every member of a batch the batch MAXIMUM, which
                # silently made pTM a batch-level extreme-value statistic
                # (48 sequences -> 4 distinct values at fold_batch=12) and
                # corrupted the foldability flag that thresholds on it.
                # Recompute per sequence from ptm_logits instead.
                if ptm_logits is not None:
                    out["ptm"].append(float(_compute_tm(
                        ptm_logits[b][:n, :n].float(),
                        max_bin=31, no_bins=ptm_logits.shape[-1])))
                elif ptm is None:
                    out["ptm"].append(float("nan"))
                elif getattr(ptm, "ndim", 0) == 0:
                    out["ptm"].append(float(ptm))
                else:
                    out["ptm"].append(float(ptm.reshape(-1)[b]))
                if pae is None:
                    out["pae"].append(float("nan"))
                else:
                    out["pae"].append(float(
                        pae[b][:n, :n].float().mean()))
                pos = o["positions"][-1, b][:n].float().cpu().numpy()
                out["ca"].append(pos[:, 1])
                # `positions` is atom14, ordered N, CA, C, O, CB, ... so CB
                # is index 4 and index 3 is the BACKBONE OXYGEN. An earlier
                # version of this file took index 3 and called it CB, which
                # made every "CB-CB contact map" an O-O map. Glycine has no
                # atom14 CB, so a virtual one is built from N, CA, C.
                if pos.shape[1] > 4:
                    cb = pos[:, 4].copy()
                    isg = np.array([c == "G" for c in sq], dtype=bool)
                    if isg.any():
                        cb[isg] = virtual_cb(pos[isg, 0], pos[isg, 1],
                                             pos[isg, 2])
                else:
                    cb = pos[:, 1].copy()
                out["cb"].append(cb)
                out["o_atom"].append(pos[:, 3] if pos.shape[1] > 3
                                     else pos[:, 1])
        for k in ("plddt", "ptm", "pae"):
            out[k] = np.array(out[k])
        out["foldable"] = ((out["plddt"] > 80) & (out["ptm"] > 0.7)
                           & (out["pae"] < 10)).astype(float)
        return out


# ------------------------------------------------- ESM-2 pseudo-perplexity
class ESM2Scorer:
    """Exact masked pseudo-perplexity under a frozen ESM-2, plus contacts."""

    def __init__(self, name="facebook/esm2_t33_650M_UR50D", device="cuda",
                 half=True):
        from transformers import AutoTokenizer, EsmForMaskedLM
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = EsmForMaskedLM.from_pretrained(name).eval().to(device)
        if half:
            self.model = self.model.half()
        self.device = device
        self.mask_id = self.tok.mask_token_id

    @torch.no_grad()
    def pseudo_perplexity(self, seq, batch_size=32):
        """Exact: mask position i alone, read log p(x_i | x_-i), for every i."""
        ids = self.tok(seq, return_tensors="pt", add_special_tokens=True
                       )["input_ids"][0].to(self.device)
        L = ids.numel()
        pos = torch.arange(1, L - 1, device=self.device)      # skip cls/eos
        lps = []
        for i in range(0, pos.numel(), batch_size):
            p = pos[i:i + batch_size]
            batch = ids.unsqueeze(0).repeat(p.numel(), 1)
            batch[torch.arange(p.numel(), device=self.device), p] = self.mask_id
            logits = self.model(input_ids=batch).logits.float()
            lp = torch.log_softmax(
                logits[torch.arange(p.numel(), device=self.device), p], -1)
            lps.append(lp[torch.arange(p.numel(), device=self.device),
                          ids[p]].cpu())
        lp = torch.cat(lps)
        return float(torch.exp(-lp.mean()))

    @torch.no_grad()
    def contacts(self, token_ids, attention_mask):
        """Attention-based contact probabilities, (B, L, L) over real residues."""
        return self.model.predict_contacts(token_ids, attention_mask)


# ------------------------------------------------------- hydropathy spectrum
def hydropathy_power(seq, periods=(3.6, 2.2), detrend=True, window=True):
    """Normalised FFT power of the Kyte-Doolittle trace at given periods.

    Returns the power at each period divided by the mean power over all
    periods in [2, 10] residues/cycle, so the number is scale free and
    comparable across sequences of different length and composition.
    """
    x = hydropathy_trace(seq)
    n = x.size
    if n < 24:
        return {f"p{p}": float("nan") for p in periods} | {"spectrum": None}
    if detrend:
        x = x - x.mean()
    if window:
        x = x * np.hanning(n)
    f = np.fft.rfft(x)
    power = (np.abs(f) ** 2)[1:]                   # drop DC
    freqs = np.fft.rfftfreq(n, d=1.0)[1:]          # cycles per residue
    per = 1.0 / freqs                              # residues per cycle
    band = (per >= 2.0) & (per <= 10.0)
    denom = power[band].mean() if band.any() else np.nan
    res = {}
    for p in periods:
        j = int(np.argmin(np.abs(per - p)))
        res[f"p{p}"] = float(power[j] / denom) if denom and denom > 0 else np.nan
    res["spectrum"] = (per, power / denom if denom and denom > 0 else power)
    return res


def helix_band_power(seq, lo=3.2, hi=4.0):
    """Power integrated over the alpha-helical band, normalised as above.

    Less brittle than a single bin: real helices are not exactly 3.6.
    """
    x = hydropathy_trace(seq)
    n = x.size
    if n < 24:
        return float("nan")
    x = (x - x.mean()) * np.hanning(n)
    power = (np.abs(np.fft.rfft(x)) ** 2)[1:]
    per = 1.0 / np.fft.rfftfreq(n, d=1.0)[1:]
    band = (per >= 2.0) & (per <= 10.0)
    sel = (per >= lo) & (per <= hi)
    denom = power[band].mean()
    if not sel.any() or not denom > 0:
        return float("nan")
    return float(power[sel].mean() / denom)


# ------------------------------------------------------------ burial / set
def buried_hydrophobic_fraction(seq, cb, cutoff=10.0, k_buried=16):
    """Fraction of buried positions that are hydrophobic.

    Burial proxy = number of CB atoms within `cutoff` A; a position counts as
    buried when that neighbour count is >= k_buried. Avoids an SASA dependency.
    """
    _, hp = family_table(FAMILIES_2)              # 0 = hydrophobic
    d = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    nb = ((d < cutoff).sum(1) - 1)
    bur = nb >= k_buried
    if bur.sum() == 0:
        return float("nan")
    lab = np.array([hp[AA_IDX[c]] for c in seq])
    return float((lab[bur] == 0).mean())


def pairwise_identity(seqs):
    """Mean pairwise identity over equal-length sequences (Hamming)."""
    by_len = {}
    for s in seqs:
        by_len.setdefault(len(s), []).append(s)
    vals = []
    for L, group in by_len.items():
        if len(group) < 2:
            continue
        arr = np.array([[AA_IDX.get(c, 20) for c in s] for s in group])
        for i in range(len(group)):
            eq = (arr[i + 1:] == arr[i]).mean(1)
            vals.extend(eq.tolist())
    return float(np.mean(vals)) if vals else float("nan")


def kmer_novelty(seqs, ref_kmers, k=10):
    """Fraction of a sequence's k-mers NOT present in the reference set.

    A cheap stand-in for an MMseqs2 novelty search; reported as such.
    """
    out = []
    for s in seqs:
        km = {s[i:i + k] for i in range(len(s) - k + 1)}
        if not km:
            out.append(float("nan")); continue
        out.append(1.0 - len(km & ref_kmers) / len(km))
    return float(np.nanmean(out))


def aa_composition_jsd(seqs, ref_freq):
    cnt = np.zeros(20)
    for s in seqs:
        for c in s:
            if c in AA_IDX:
                cnt[AA_IDX[c]] += 1
    if cnt.sum() == 0:
        return float("nan")
    p = cnt / cnt.sum()
    q = np.asarray(ref_freq, dtype=float); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def max_repeat_run(seq):
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def shannon_entropy(seqs):
    cnt = np.zeros(20)
    for s in seqs:
        for c in s:
            if c in AA_IDX:
                cnt[AA_IDX[c]] += 1
    p = cnt / max(cnt.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ===================================================================
# Hydrophobic periodicity, done the way the periodicity literature does it.
#
# Eisenberg, Weiss & Terwilliger, PNAS 81:140-144 (1984) introduced the
# hydrophobic moment and reported ~3.6 residues/cycle for helices and ~2.3
# for strands. Cornette et al., J Mol Biol 195:659-685 (1987) is the
# reference for the spectral form: they recommend the LEAST-SQUARES power
# spectrum (fit a sinusoid of a given frequency and record the sum of
# squares it accounts for) rather than a naive FFT on short windows, and
# define the alpha-amphipathic index as the share of total spectral area
# under the ~97.5 degree peak. Angles are in degrees of rotation per
# residue: 100 deg = 3.6 residues/cycle for an alpha helix, 160-180 deg
# for a beta strand.
# ===================================================================
from .alphabet import trace as _trace

HELIX_ANGLE = 97.5          # Cornette's observed peak, ~3.69 residues/cycle
STRAND_ANGLE = 160.0        # EMBOSS hmoment beta default
ANGLES = np.arange(20.0, 180.1, 2.5)


def ls_spectrum(x, angles=ANGLES):
    """Least-squares power spectrum: variance share explained at each angle.

    For each angle d, fit x_n ~ a*cos(d n) + b*sin(d n) and report the sum
    of squares accounted for, divided by the total sum of squares. Bounded
    in [0, 1] and independent of the hydrophobicity scale's units.

    Solved in closed form and vectorised over angles. With a two-column
    design matrix the normal equations are a 2x2 system, so the whole
    spectrum is a handful of matrix products rather than one least-squares
    call per angle.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 12:
        return np.full(angles.size, np.nan)
    x = x - x.mean()
    sst = float((x ** 2).sum())
    if sst <= 0:
        return np.full(angles.size, np.nan)
    t = np.arange(n, dtype=np.float64)
    w = np.deg2rad(np.asarray(angles, dtype=np.float64))[:, None]   # (A, 1)
    C = np.cos(w * t)                                               # (A, n)
    S = np.sin(w * t)
    cc = (C * C).sum(1)
    ss = (S * S).sum(1)
    cs = (C * S).sum(1)
    a = C @ x
    b = S @ x
    det = cc * ss - cs * cs
    # SS explained = [a b] inv(G) [a b]'  with G = [[cc, cs], [cs, ss]].
    # At 180 degrees per residue the sine column is identically zero on
    # integer positions, so G is singular and the fit reduces to the cosine
    # column alone. Handle that rank-deficient case explicitly rather than
    # letting the division produce a spurious zero.
    tol = 1e-9 * np.maximum(cc * ss, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        full = (ss * a * a - 2.0 * cs * a * b + cc * b * b) / det
        cos_only = np.where(cc > 1e-12, a * a / np.maximum(cc, 1e-12), 0.0)
        sin_only = np.where(ss > 1e-12, b * b / np.maximum(ss, 1e-12), 0.0)
    rank1 = np.where(cc >= ss, cos_only, sin_only)
    ssm = np.where(det > tol, full, rank1)
    ssm = np.where(np.isfinite(ssm), ssm, 0.0)
    return np.clip(ssm / sst, 0.0, 1.0)


def windowed_spectrum(seq, scale="eisenberg", window=18, stride=3,
                      angles=ANGLES):
    """Mean least-squares spectrum over sliding windows.

    Short windows are what Cornette et al. use: periodicity is a local
    property of a helix, and a whole-protein transform smears it out.
    """
    x = _trace(seq, scale)
    if x.size < window:
        return np.full(angles.size, np.nan)
    specs = [ls_spectrum(x[i:i + window], angles)
             for i in range(0, x.size - window + 1, stride)]
    specs = [s for s in specs if np.isfinite(s).all()]
    if not specs:
        return np.full(angles.size, np.nan)
    return np.mean(specs, axis=0)


def amphipathic_index(seq, scale="eisenberg", window=18, stride=3,
                      center=HELIX_ANGLE, halfwidth=10.0, angles=ANGLES):
    """Share of total spectral area under the helical peak (Cornette).

    Returns nan for sequences shorter than the window.
    """
    s = windowed_spectrum(seq, scale, window, stride, angles)
    if not np.isfinite(s).any():
        return float("nan")
    band = (angles >= center - halfwidth) & (angles <= center + halfwidth)
    tot = s.sum()
    return float(s[band].sum() / tot) if tot > 0 else float("nan")


def periodicity_summary(seq, scale="eisenberg", window=18, stride=3):
    """Helical and strand-band spectral power plus the amphipathic index."""
    s = windowed_spectrum(seq, scale, window, stride)
    if not np.isfinite(s).any():
        return {"helix_power": np.nan, "strand_power": np.nan,
                "amphipathic_index": np.nan}
    hb = (ANGLES >= 87.5) & (ANGLES <= 107.5)
    sb = (ANGLES >= 150.0) & (ANGLES <= 180.0)
    tot = s.sum()
    return {"helix_power": float(s[hb].mean()),
            "strand_power": float(s[sb].mean()),
            "amphipathic_index": float(s[hb].sum() / tot) if tot > 0
            else np.nan}


def hydrophobic_moment(seq, scale="eisenberg", angle=100.0, window=18,
                       stride=1):
    """Eisenberg's mean hydrophobic moment, max over sliding windows.

    mu_H(d) = |sum_n h_n exp(i n d)| / window
    """
    x = _trace(seq, scale)
    if x.size < window:
        return float("nan")
    w = np.deg2rad(angle)
    best = -np.inf
    for i in range(0, x.size - window + 1, stride):
        seg = x[i:i + window]
        t = np.arange(seg.size)
        m = np.abs((seg * np.exp(1j * w * t)).sum()) / seg.size
        best = max(best, float(m))
    return best


def amphipathic_window_stats(seq, scale="eisenberg", window=18, stride=3,
                             lo=87.5, hi=107.5, thresh=0.30):
    """Per-window amphipathicity, which is how the literature reports it.

    The mean amphipathic index over all windows has a narrow dynamic range:
    a flat spectrum already gives the helical band's share of the angle grid
    (about 0.14 here), and natural proteins average only about 0.15 because
    most windows are not amphipathic helices. Cornette et al. instead report
    that roughly half of alpha-helices look amphipathic, which is a
    statement about the FRACTION of windows that cross a threshold, not
    about the average.

    Returns the fraction of windows whose helical-band share exceeds
    `thresh`, and the best window's share.
    """
    x = _trace(seq, scale)
    if x.size < window:
        return {"amph_window_frac": float("nan"),
                "amph_window_max": float("nan")}
    band = (ANGLES >= lo) & (ANGLES <= hi)
    shares = []
    for i in range(0, x.size - window + 1, stride):
        s = ls_spectrum(x[i:i + window], ANGLES)
        if not np.isfinite(s).any():
            continue
        tot = s.sum()
        if tot > 0:
            shares.append(float(s[band].sum() / tot))
    if not shares:
        return {"amph_window_frac": float("nan"),
                "amph_window_max": float("nan")}
    shares = np.asarray(shares)
    return {"amph_window_frac": float((shares > thresh).mean()),
            "amph_window_max": float(shares.max())}
