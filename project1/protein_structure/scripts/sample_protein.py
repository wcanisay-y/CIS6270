#!/usr/bin/env python3
"""Generate Ca backbones from a trained checkpoint.

    python sample_protein.py --checkpoint <run>/checkpoint.pt --n 10000 --batch-size 1024
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from sparse_egnn import N_RESIDUE_TYPES, TYPE_SCALE, Diffusion, ProteinDenoiser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RES = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-pdb", type=int, default=0,
                    help="also write this many samples as Ca-only PDB files for inspection")
    args = ap.parse_args()
    out = args.out or args.checkpoint.parent / f"samples_{args.n}.npz"

    # weights_only=False is explicit: the checkpoint carries the run's args dict and length
    # histogram beside the tensors. Without it torch 2.4 warns about a default it will flip.
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    saved = state["args"]
    model = ProteinDenoiser(saved["hidden"], saved["layers"], saved["k"]).to(DEVICE)
    model.load_state_dict(state["ema"])          # the EMA weights are the ones to sample from
    model.eval()
    diffusion = Diffusion(saved["timesteps"], DEVICE)
    coord_std = state["coord_std"]
    histogram = state["length_histogram"].to(DEVICE)
    print(f"epoch {state['epoch']} | coord_std {coord_std:.3f} A | sampling {args.n:,} backbones")

    torch.manual_seed(args.seed)
    coords, types, lengths = [], [], []
    done = 0
    while done < args.n:
        batch = min(args.batch_size, args.n - done)
        # Chain length is drawn from the training length distribution, as the QM9 baseline draws
        # molecule size from its size histogram.
        length = torch.multinomial(histogram, batch, replacement=True).clamp(min=4, max=MAX_RES)
        x, h, mask = diffusion.sample(model, length, MAX_RES, DEVICE)
        coords.append((x * coord_std).cpu().numpy().astype(np.float32))
        types.append((h / TYPE_SCALE).argmax(-1).cpu().numpy().astype(np.int8))
        lengths.append(length.cpu().numpy().astype(np.int16))
        done += batch
        print(f"  {done:,}/{args.n:,}", flush=True)
    coords, types, lengths = (np.concatenate(coords), np.concatenate(types),
                              np.concatenate(lengths))
    np.savez_compressed(out, coords=coords, types=types, lengths=lengths)
    print(f"wrote {out}")
    if args.write_pdb:
        AA3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
               "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
               "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}
        one = list("ACDEFGHIKLMNPQRSTVWY")
        folder = out.parent / "pdb"
        folder.mkdir(exist_ok=True)
        for i in range(min(args.write_pdb, len(lengths))):
            n = int(lengths[i])
            lines = []
            for j in range(n):
                residue = AA3[one[int(types[i][j])]]
                x, y, z = coords[i][j]
                lines.append(f"ATOM  {j+1:5d}  CA  {residue} A{j+1:4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
            lines.append("TER")
            lines.append("END")
            (folder / f"sample_{i:04d}.pdb").write_text("\n".join(lines) + "\n")
        print(f"wrote {min(args.write_pdb, len(lengths))} Ca-only PDB files to {folder}")


if __name__ == "__main__":
    main()
