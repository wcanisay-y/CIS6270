"""Amino-acid alphabet, chemical families, and hydrophobicity scales.

Families (Idea B, as proposed):
    Aliphatic A V L I M | Aromatic F W Y | Polar S T N Q H
    Positive  K R       | Negative D E   | Special G P C
Every residue is in exactly one family, so the family label is a pure
function of the sequence -- nothing annotated, no structures needed.
"""
import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"          # canonical, alphabetical
AA_IDX = {a: i for i, a in enumerate(AA20)}

# ---------------------------------------------------------------- families
FAMILIES_6 = {                          # the proposal's partition
    "aliphatic": "AVLIM",
    "aromatic":  "FWY",
    "polar":     "STNQH",
    "positive":  "KR",
    "negative":  "DE",
    "special":   "GPC",
}
# granularity controls
FAMILIES_2 = {                          # hydrophobic / polar (HP model)
    "hydrophobic": "AVLIMFWYCG",
    "polar":       "STNQHKRDEP",
}
FAMILIES_10 = {                         # finer split of the same chemistry
    "small_aliphatic": "AV",
    "large_aliphatic": "LIM",
    "aromatic":        "FWY",
    "hydroxyl":        "ST",
    "amide":           "NQ",
    "histidine":       "H",
    "positive":        "KR",
    "negative":        "DE",
    "glycine_proline": "GP",
    "cysteine":        "C",
}
# Murphy et al. (2000) 8-group reduced alphabet, as a literature-standard
# comparison partition rather than a hand-made one.
FAMILIES_MURPHY8 = {
    "LVIMC": "LVIMC", "AG": "AG", "ST": "ST", "P": "P",
    "FYW": "FYW", "EDNQ": "EDNQ", "KR": "KR", "H": "H",
}


def family_table(fams):
    """-> (names, aa->family index array of length 20)."""
    names = list(fams.keys())
    lut = np.full(20, -1, dtype=np.int64)
    for fi, n in enumerate(names):
        for a in fams[n]:
            if a not in AA_IDX:
                raise ValueError(f"bad residue {a!r} in family {n}")
            if lut[AA_IDX[a]] != -1:
                raise ValueError(f"residue {a} in two families")
            lut[AA_IDX[a]] = fi
    if (lut == -1).any():
        missing = [AA20[i] for i in np.where(lut == -1)[0]]
        raise ValueError(f"residues unassigned: {missing}")
    return names, lut


def random_family_table(fams, seed):
    """Random partition with the SAME group sizes as `fams`.

    This is the decisive control for Idea B: if a random 6-way split helps as
    much as the chemical one, the gain is from the hierarchy, not the chemistry.
    """
    names, lut = family_table(fams)
    sizes = [int((lut == i).sum()) for i in range(len(names))]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(20)
    out = np.empty(20, dtype=np.int64)
    k = 0
    for fi, s in enumerate(sizes):
        out[perm[k:k + s]] = fi
        k += s
    return [f"rand{i}" for i in range(len(names))], out


# ------------------------------------------------------- hydrophobicity
# Kyte & Doolittle (1982) -- the standard scale for hydropathy plots.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}
# Eisenberg et al. (1984) consensus -- used for the hydrophobic moment.
EISENBERG = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
             "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
             "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
             "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}

KD_VEC = np.array([KD[a] for a in AA20], dtype=np.float64)
EIS_VEC = np.array([EISENBERG[a] for a in AA20], dtype=np.float64)


def hydropathy_trace(seq, scale="kd"):
    v = KD_VEC if scale == "kd" else EIS_VEC
    return np.array([v[AA_IDX[c]] for c in seq if c in AA_IDX], dtype=np.float64)


# ===================================================================
# Literature-standard partitions and scales.
#
# The proposal's 6 groups are, to within two residues, a published
# six-letter code: Dokholyan, Mirny & Shakhnovich, Physica A 314:600-606
# (2002) / arXiv:cond-mat/0007084, which uses the same six group names.
# It differs from the proposal in H (there: aromatic; proposal: polar) and
# C (there: aliphatic; proposal: special). Both variants are kept so the
# two deviations can be ablated rather than assumed.
# ===================================================================
FAMILIES_DMS6 = {                       # Dokholyan-Mirny-Shakhnovich
    "aliphatic": "AVLIMC",
    "aromatic":  "FWYH",
    "polar":     "STNQ",
    "positive":  "KR",
    "negative":  "DE",
    "special":   "GP",
}
FAMILIES_PROPOSAL6 = FAMILIES_6         # the proposal's own split, verbatim
FAMILIES_H_SINGLETON = {                # Murphy8/10 convention for H
    "aliphatic": "AVLIM",
    "aromatic":  "FWY",
    "polar":     "STNQ",
    "histidine": "H",
    "charged":   "KRDE",
    "special":   "GPC",
}
# Murphy, Wallqvist & Levy, Protein Engineering 13:149-152 (2000).
# Memberships taken from the Biopython Bio.Alphabet.Reduced tabulation;
# not verified against the original article in this session.
FAMILIES_MURPHY4 = {"LVIMC": "LVIMC", "AGSTP": "AGSTP",
                    "FYW": "FYW", "EDNQKRH": "EDNQKRH"}
FAMILIES_MURPHY10 = {"LVIM": "LVIM", "C": "C", "A": "A", "G": "G",
                     "ST": "ST", "P": "P", "FYW": "FYW",
                     "EDNQ": "EDNQ", "KR": "KR", "H": "H"}

PARTITIONS = {
    "proposal6": FAMILIES_PROPOSAL6,
    "dms6": FAMILIES_DMS6,
    "hsingle6": FAMILIES_H_SINGLETON,
    "hp2": FAMILIES_2,
    "murphy4": FAMILIES_MURPHY4,
    "murphy8": FAMILIES_MURPHY8,
    "murphy10": FAMILIES_MURPHY10,
    "fine10": FAMILIES_10,
}

# ----------------------------------------------------------- scales
# Cornette et al., J Mol Biol 195:659-685 (1987) derived an eigenvector-
# optimised scale specifically to display the ~3.6 residue/cycle period.
# Values below are the ExPASy ProtScale tabulation; NOT verified against
# the 1987 article in this session, so it is used only as a robustness
# check. Eisenberg consensus is the primary scale for periodicity.
CORNETTE = {"A": 0.2, "R": 1.4, "N": -0.5, "D": -1.2, "C": 4.1, "Q": -2.8,
            "E": -2.7, "G": 0.0, "H": 0.5, "I": 4.8, "L": 9.2, "K": -4.2,
            "M": 4.2, "F": 4.4, "P": -6.6, "S": -0.5, "T": -1.2, "W": 1.0,
            "Y": 3.2, "V": 4.7}
CORNETTE_VEC = np.array([CORNETTE[a] for a in AA20], dtype=np.float64)

SCALES = {"eisenberg": EIS_VEC, "kd": KD_VEC, "cornette": CORNETTE_VEC}


def trace(seq, scale="eisenberg"):
    v = SCALES[scale]
    return np.array([v[AA_IDX[c]] for c in seq if c in AA_IDX],
                    dtype=np.float64)
