# Modality 2: protein Ca backbones — sparse SE(3) diffusion on SwissProt-Pfam

Unconditional generation of Ca backbone traces, adapting the project-1 diffusion recipe from small
molecules to proteins. **Nothing here imports or modifies the QM9 code** in
`../molecular_coordinates/diffusion_baseline` or `../benchmarks`; the two trees share only the
recipe, so the results stay comparable.

## Dataset

[DanielHesslow/SwissProt-Pfam](https://huggingface.co/datasets/DanielHesslow/SwissProt-Pfam),
which is sequence-only, joined to **AlphaFold DB** structures through the UniProt accession each
row carries. The subset is single-Pfam-family entries of 16 to 128 residues.

| | Rows passing the filter | After exact deduplication |
|---|---:|---:|
| train | 56,771 | 40,824 |
| dev | ~7,200 | 5,157 |
| test | ~7,000 | 5,057 |

About 8% of accessions are then dropped: roughly 3% absent from AlphaFold DB, 5% below a mean
pLDDT of 70, and a handful where the model length disagrees with the dataset sequence.

The subset is dominated by ribosomal proteins, 17 of the 25 most common families, plus small
enzymes such as phosphoribosyl transferase, globin and dUTPase. AlphaFold models them well, at a
mean pLDDT of 85 to 97 per family.

**Two traps.** The integer `labels` column does **not** agree with the shipped `idx_mapping.json`;
zero of 20,000 checked rows matched, so family identity is read from `labels_str`. And the shipped
train/dev/test split is clean on exact sequences and accessions but is **not** homology-controlled,
so close relatives can straddle it. Reclustering at 30% identity is the right fix and is not yet
done; `mmseqs2` is not installed on this machine.

## Layout

Scripts, results and figures all live in this directory, so the arm is self-contained:

```
protein_structure/
  run.sh       the whole pipeline: tests, training, sampling, evaluation, figures
  scripts/     every script below
  results/     RESULTS.md and the runs; ca_sparse_k16/ is the run of record
  figures/     fig0_overview plus the five standalone figures, PNG and PDF, and figure_data.csv
  logs/        sbatch job output
```

| Script | What it does |
|---|---|
| `prepare_structures.py` | filter, deduplicate, fetch AlphaFold PDBs, parse Ca, pack tensors. Resumable in shards |
| `sparse_egnn.py` | the sparse SE(3) denoiser and the diffusion process |
| `train_protein.py` | training loop, EMA, per-epoch dev loss, resumable |
| `sample_protein.py` | generate backbones from a checkpoint using the EMA weights |
| `evaluate_protein.py` | protein metrics for samples and for the AlphaFold reference |
| `plot_results.py` | the overview and the five standalone figures, into `figures/` |
| `test_protein.py` | 30 correctness checks. Run before any long job |

The only thing kept outside the directory is the structure cache, which is bulky regenerable
input data: `/mnt/isilon/tan_lab/pany3/protein_diffusion/cache`, overridable with
`PROTEIN_DIFFUSION_ROOT`. Inside `results/`, the checkpoint and the `.npz` sample archives are
excluded from git by extension; everything else there is tracked.

## Results

Full write-up in [`results/RESULTS.md`](results/RESULTS.md); one-page version in
`figures/fig0_overview.png`. In short, from 10,000 samples against the 4,615-structure
AlphaFold test reference: bonds in range 0.998 vs 0.9996 and Ca spacing within 0.05 A, so local
geometry is solved; clash-free 0.589 vs 1.000, which is what caps validity at 0.484 vs 0.966;
right-handed windows 0.563 against a 0.500 random baseline and 0.620 in the data, so the
chirality feature is learned. Validity falls from 0.851 at 16-40 residues to 0.339 at 101-128.
Two explanations for that fall were tested and refuted; the remaining one is error accumulation
over the 1,000 reverse steps.

## What had to change from the QM9 baseline, and why

Three constants are genuinely just constants: five atom types become twenty residue types, the node
cap goes from 29 to 128, and the coordinate normaliser is refit, since the spread is about 10 A
against roughly 1.7 A for QM9. Four changes are architectural.

1. **Sparse edges.** The QM9 layer builds a dense `[B, N, N, 2*hidden+1]` pair tensor. Measured at
   128 nodes it exceeds a 40 GB A100 above batch 32 and runs about 15x slower than necessary. Real
   Ca graphs average 9 neighbours within 8 A and 16 within 10 A, so edges are k nearest spatial
   neighbours plus the two backbone bonds, carried on an edge flag.
2. **Chirality.** Messages built from squared distances alone are invariant to reflection, so such
   a model cannot distinguish a right-handed helix from a left-handed one, and every real protein
   helix is right-handed. A normalised scalar triple product over four consecutive Ca atoms is a
   pseudoscalar: unchanged by rotation, negated by reflection. Feeding it in as a node scalar makes
   the network SE(3)-equivariant instead of E(3). `test_protein.py` measures both halves of this.
3. **Chain order.** A protein is a sequence, not a set. Residue index enters through a sinusoidal
   encoding, and backbone bonds are always present as flagged edges.
4. **Evaluation.** Bond-order tables, valence stability and SMILES are meaningless for a Ca trace,
   so the scorer is new: backbone spacing, clash freedom, radius of gyration, handedness, helical
   fraction and residue composition, each computed identically for samples and for the reference.

Reused unchanged: the polynomial_2 schedule, the zero-centre-of-mass projection, epsilon
regression, residue one-hots scaled by 1/4, EMA and the training loop shape.

## Reference ceiling

Measured on 1,838 AlphaFold structures, so this is the best any sample set can score:

| Metric | AlphaFold reference |
|---|---|
| Validity | 0.970 |
| Consecutive Ca spacing | 3.841 A, sd 0.029 |
| Clash-free | 1.000 |
| Radius of gyration | 17.3 A |
| Right-handed windows | 0.630 |
| Helical fraction | 0.385 |

A model that has learned nothing about handedness will sit at 0.500 on the right-handed row. That
row is the single most informative check that the chirality feature did its job.

## Running it

`run.sh` is the whole pipeline in one place, and is the normal way to run this. Stages whose
outputs already exist are skipped and training always passes `--resume`, so it is idempotent:
re-running it against the finished run of record does nothing but redraw the figures.

```bash
bash run.sh                      # tests, training, sampling, evaluation, figures (~3.5 h)
sbatch run.sh                    # the same as a batch job; submit from THIS directory
SMOKE=1 bash run.sh              # 2 epochs on 2,000 proteins + 256 samples, a few minutes
EPOCHS=500 bash run.sh           # extend the existing run from epoch 200
STAGES=eval,figures bash run.sh  # rescore and redraw only
FORCE=1 bash run.sh              # redo stages whose outputs already exist
```

The arm is chosen with environment variables (`RUN`, `EPOCHS`, `BATCH`, `LR`, `HIDDEN`, `LAYERS`,
`K`, `TIMESTEPS`, `EMA`, `SEED`, `SAMPLES`, `SAMPLE_BATCH`, `WRITE_PDB`), so there is a single
place to look. Only `RUN=ca_sparse_k16` writes to the module's `figures/`; every other run keeps
its figures in `results/$RUN/figures`, so an experiment cannot overwrite the figures a report
cites. `GPU_UUID=GPU-xxxx` pins the card by identity, which matters on nodes whose `gres.conf`
lists devices they do not have.

The individual steps, if you want to drive them by hand:

```bash
cd scripts
python test_protein.py                       # 30 checks, seconds
python prepare_structures.py                 # ~45 min, resumable
python train_protein.py --smoke              # 2 epochs on 2,000 proteins
python train_protein.py --epochs 200         # the real run, into ../results/ca_sparse_k16
python sample_protein.py --checkpoint ../results/ca_sparse_k16/checkpoint.pt --n 10000
python evaluate_protein.py --samples ../results/ca_sparse_k16/samples_10000.npz \
                           --out ../results/ca_sparse_k16/metrics.json
python plot_results.py                       # the figures, about a minute
```

## Figures

`plot_results.py` reads `results/ca_sparse_k16` and writes `figures/` as PNG and PDF, plus
`figure_data.csv` with the plotted numbers. It rescores every structure with
`evaluate_protein.per_structure` and asserts the result matches `metrics.json`, so a figure
cannot drift from the reported table. Every panel is a function of `(ax, ctx)`, so the
overview and the standalone figures draw the same marks from the same numbers.

| Figure | What it shows |
|---|---|
| `fig0_overview` | the whole arm on one page: run header, headline tiles, and every panel below |
| `fig1_training` | train / dev / dev-EMA loss over 200 epochs, and the last 100 zoomed |
| `fig2_sample_quality` | the five headline rates against the AlphaFold ceiling |
| `fig3_geometry_distributions` | bond length, clashes, radius of gyration, helical fraction |
| `fig4_length_stratified` | validity, clashes and collapse against chain length |
| `fig5_backbones` | Ca traces, clash-free examples matched on length |

Measured on one A100-SXM4-40GB: 1,058 proteins/s at batch 128 using 9.9 GB, about 35 s per epoch,
so 200 epochs is close to 2 h. Sampling 10,000 backbones takes about 1.6 h, since each needs 1,000
sequential denoising passes.

## Caveats

- The model generates a Ca trace and a residue sequence, not full backbone or side-chain atoms.
  No bond angles, no dihedrals, no side chains.
- This is generation, not structure prediction. It does not take a sequence and return a fold.
- The split is not homology-controlled yet, so any novelty claim is optimistic.
- 128 residues is a cap chosen for speed. It halves the data against a 200-residue cap and cuts
  families with 100 or more members from 241 to 117, which matters for a family-conditioned arm.
