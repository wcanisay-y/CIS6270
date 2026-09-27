# Results: unconditional Ca-backbone diffusion on SwissProt-Pfam

One arm, run end to end on 2026-09-26. Everything below is reproduced by
`scripts/plot_results.py`, which rescores every structure rather than reading a stored table
and asserts the recomputed validity matches `metrics.json` before drawing.

Figures are in `../figures/`; `fig0_overview` is the single-page version of this document.

## The run

| | |
|---|---|
| model | sparse SE(3) EGNN denoiser, 4,164,637 parameters, 9 layers, hidden 256, k=16 spatial neighbours plus flagged backbone edges |
| extra features | chirality pseudoscalar per node, sinusoidal residue-index encoding |
| data | SwissProt-Pfam, single-Pfam-family chains of 16-128 residues, Ca traces from AlphaFold DB |
| splits | 37,476 train / 4,728 dev / 4,615 test structures, mean length ~91 |
| retention | 92% of 51,017 attempted accessions (~3% absent from AFDB, ~5% below mean pLDDT 70) |
| diffusion | 1,000 steps, polynomial schedule, zero-CoM projection, EMA 0.999 |
| training | 200 epochs, batch 128, lr 1e-4, 2.13 h on one A100-SXM4-40GB |
| sampling | 10,000 backbones, 1.02 h |
| coord_std | 10.895 A |

Wall clock end to end was 4.16 h against a 7 h plan: fetch 0.75 h, training 2.38 h,
sampling 1.02 h, scoring seconds.

## Training

Final train 0.1655, dev 0.1698, dev (EMA) 0.1604; best dev (EMA) 0.1596 at epoch 190.
Train and dev are equal, so there is no overfitting and more epochs are likely to help.
Speed rose from 47.6 s/epoch to 37.6 s/epoch (787 -> 997 proteins/s).

See `fig1_training`.

## Sample quality

10,000 generated backbones against the 4,615-structure AlphaFold test reference. Every metric
is computed identically for both rows, so the reference row is the ceiling, not a literature
number.

| metric | AlphaFold test | generated |
|---|---|---|
| validity | 0.9658 | 0.4839 |
| Ca spacing (A) | 3.8403 (sd 0.0301) | 3.7953 (sd 0.1258) |
| bonds in range | 0.9996 | 0.9975 |
| clash-free | 1.0000 | 0.5888 |
| clashes per structure | 0.0 | 0.7921 |
| radius of gyration (A) | 17.64 | 15.79 |
| right-handed windows | 0.6203 | 0.5628 |
| helical fraction | 0.3740 | 0.1323 |

Reading: local geometry is essentially solved, with the bond mean within 0.05 A and 99.75% of
bonds inside the accepted window. Validity is capped by CLASHES, not bonds, and the small
radius of gyration says samples are over-collapsed, which is the same defect seen from a
different angle. Handedness at 0.563 against a 0.500 random baseline and 0.620 in the data
means the chirality feature is learned and captures about half the available signal. Helical
content is the weakest axis.

A 256-sample preview predicted these closely (validity 0.473 vs 0.484), so small previews are
reliable here.

See `fig2_sample_quality` and `fig3_geometry_distributions`.

## Degradation with chain length

| length | n | validity | clash-free | clashes | radius (A) | reference radius (A) |
|---|---|---|---|---|---|---|
| 16-40 | 462 | 0.851 | 0.926 | 0.10 | 12.61 | 12.22 |
| 41-60 | 825 | 0.718 | 0.861 | 0.21 | 15.05 | 15.41 |
| 61-80 | 1,903 | 0.614 | 0.729 | 0.45 | 15.59 | 17.34 |
| 81-100 | 2,802 | 0.474 | 0.581 | 0.74 | 16.01 | 17.59 |
| 101-128 | 4,008 | 0.339 | 0.433 | 1.19 | 16.25 | 18.86 |

The reference is flat at 0.94-0.99 validity and zero clashes across the same bins. The radius
tracks the reference for short chains and undershoots badly for long ones.

See `fig4_length_stratified`.

## Two hypotheses tested and refuted

Do not repeat these.

1. **Receptive field too small.** WRONG. The k=16-plus-backbone graph has a diameter of only
   2.1 hops at 16-40 residues rising to 5.5 hops at 101-128, so 9 message-passing layers span
   every chain with room to spare. 0% of sampled structures exceed the layer count at any length.
2. **The model fits long chains worse.** LARGELY WRONG. Held-out denoising loss stratified by
   length, averaged over a fixed noise grid, runs 0.1434 at 16-40 to 0.1694 at 101-128, an 18%
   rise. That cannot explain validity falling 0.851 to 0.339, a 2.5x degradation.

The remaining explanation is **error accumulation in the reverse process**: the per-step
objective is fit almost uniformly across lengths, but 1,000 sequential steps compound small
errors, and longer chains have proportionally more coordinates to get wrong at once.

## What to try next

Attack sampling-time error accumulation first, since it targets the measured failure mode:
steric guidance or a clash penalty during sampling. Longer training is the cheap second lever,
because dev loss was still flat rather than rising at epoch 200.

## Files here

| Path | What it is |
|---|---|
| `ca_sparse_k16/history.json` | per-epoch train / dev / dev-EMA loss and throughput |
| `ca_sparse_k16/metrics.json` | the aggregate table above, as written by `evaluate_protein.py` |
| `ca_sparse_k16/pdb/` | 20 Ca-only PDB files for visual inspection |
| `ca_sparse_k16/checkpoint.pt` | model, EMA, optimiser, args, coord_std, length histogram (not in git) |
| `ca_sparse_k16/samples_10000.npz` | the scored sample set: coords, types, lengths (not in git) |
| `ca_sparse_k16/samples_256.npz` | the preview sample set (not in git) |

The checkpoint and the `.npz` archives are excluded from git by extension; they are on disk
here and also at `/mnt/isilon/tan_lab/pany3/protein_diffusion/runs/ca_sparse_k16`.

## Caveats

- A Ca trace and a residue sequence, not a full backbone: no bond angles, no dihedrals, no side chains.
- Generation, not structure prediction. It does not take a sequence and return a fold.
- The split is not homology-controlled, so any novelty claim is optimistic.
- 128 residues is a speed cap. It halves the data against a 200-residue cap and cuts families
  with 100 or more members from 241 to 117, which matters for a family-conditioned arm.
