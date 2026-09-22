# QM9 diffusion benchmark: EDM vs GCDM

Reference results for project 1. Two published E(3)-equivariant diffusion models were run from
their authors' pretrained QM9 checkpoints, each generating 10,000 unconditional molecules, and the
samples were scored with one shared evaluator. Nothing here was retrained.

| Model | Repository | Checkpoint |
|---|---|---|
| EDM | github.com/ehoogeboom/e3_diffusion_for_molecules | `outputs/edm_qm9` (shipped with the repo) |
| GCDM | github.com/BioinfoMachineLearning/Bio-Diffusion | `QM9/Unconditional/model_1_epoch_979-EMA.ckpt` (Zenodo record 13375913) |

## Results (shared evaluator, 10,000 samples each, 95% bootstrap intervals)

| | Atom stability | Molecule stability | Validity | Uniqueness | Novelty |
|---|---|---|---|---|---|
| QM9 data | 0.994 (0.993-0.994) | 0.953 (0.949-0.957) | 0.975 (0.972-0.978) | 1.000 | n/a |
| EDM | 0.983 (0.982-0.984) | 0.810 (0.803-0.818) | 0.919 (0.914-0.925) | 0.985 (0.983-0.988) | 0.559 (0.548-0.569) |
| GCDM | 0.987 (0.987-0.988) | 0.864 (0.858-0.871) | 0.949 (0.945-0.953) | 0.985 (0.983-0.987) | 0.512 (0.503-0.522) |

Test negative log-likelihood from each model's own evaluator on the EDM test split:
EDM -111.39 (one pass), GCDM -172.33 (mean of five passes, -174.58 to -169.05).

Novelty here is measured against all 130,468 unique SMILES of the local QM9 parquet copy
(`structure-epflai/qm9` on Hugging Face). Each model's own evaluator measures it against its
training split only, giving 0.670 (EDM) and 0.628 (GCDM); see `figures/fig2_novelty_reference`.

## Files

- `unified_eval.py` shared scorer. Bonds are inferred from interatomic distances with EDM's
  bond-length tables; stability, validity (RDKit sanitization, largest fragment), uniqueness and
  novelty follow EDM's definitions. Imports EDM's code from a sibling `edm/` clone.
- `bootstrap_eval.py` per-molecule indicators plus 1,000 bootstrap resamples; writes
  `results/bootstrap.json` and `results/bootstrap_summary.csv`.
- `make_figures.py` draws `figures/` from `results/` (PNG and PDF; `figure_data.csv` is the table twin).
- `results/unified_all.json` point estimates for QM9 data, EDM and GCDM.

## Reproduction

The scripts were run from a working directory laid out as

    benchmarks/
      edm/        clone of e3_diffusion_for_molecules
      gcdm/       clone of Bio-Diffusion, with the Zenodo checkpoints unpacked
      results/    outputs
      *.py        the three scripts in this folder

in a conda environment (`qm9bench`, Python 3.9, torch 1.12.1) with each repo's requirements,
`pyarrow==14.0.2` and `matplotlib==3.6.2` (GCDM imports a module removed in matplotlib 3.7).
The QM9 download URLs in both repos had to be changed from
`springernature.figshare.com/ndownloader` to `ndownloader.figshare.com`.

Sampling and native evaluation:

    # EDM, from benchmarks/edm
    python eval_analyze.py --model_path outputs/edm_qm9 --n_samples 10000 \
        --batch_size_gen 100 --save_to_xyz True

    # GCDM, from benchmarks/gcdm
    PROJECT_ROOT=$PWD python src/mol_gen_eval.py datamodule=edm_qm9 model=qm9_mol_gen_ddpm \
        logger=csv trainer.accelerator=gpu "trainer.devices=[0]" \
        ckpt_path=checkpoints/QM9/Unconditional/model_1_epoch_979-EMA.ckpt \
        datamodule.dataloader_cfg.num_workers=1 model.diffusion_cfg.sample_during_training=false \
        num_samples=10000 sampling_batch_size=500 num_test_passes=5 save_molecules=True \
        output_dir=output/QM9/Unconditional/gcdm_model_1/

Shared scoring, bootstrap and figures, from `benchmarks/`:

    python unified_eval.py --samples EDM=edm/outputs/edm_qm9/eval/analyzed_molecules \
        GCDM=gcdm/output/QM9/Unconditional/gcdm_model_1 --out results/unified_all.json
    python bootstrap_eval.py
    python make_figures.py

Both sampling runs took about two hours each on one GPU. The generated xyz files, run logs and
the per-molecule tables were left out of the repository for size.
