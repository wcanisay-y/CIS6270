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
- `setup.sh`, `run.sh` one-command reproduction (see below); `environment-record.txt` exact package versions.

## Reproduction

Two scripts reproduce everything from scratch; both run from this directory and put all
downloads next to it (the `.gitignore` here keeps them out of the repository):

    bash setup.sh      # clones EDM and GCDM at the commits used, patches their QM9 download
                       # URLs, creates the conda env "qm9bench", downloads the QM9 parquet
                       # and the 2.4 GB GCDM checkpoint archive (resumable, md5-verified)
    bash run.sh        # samples 10,000 molecules per model with native evaluation
                       # (about 2 h for EDM and 4.5 h for GCDM on one GPU; PARALLEL=1 runs
                       # both at once), then shared scoring, bootstrap and figures

`run.sh` skips any stage whose output already exists, so it can be re-run after an
interruption. `N_SAMPLES=1000 bash run.sh` gives a quick smoke test. `unified_eval.py` finds
the parquet under `data/qm9/` or at `$QM9_PARQUET`.

What the scripts encode, for reference:

- environment: Python 3.9, PyTorch 1.12.1 with CUDA 11.6, PyG 2.2.0 and the scatter, cluster
  and sparse extensions from conda; GCDM's pip requirements from its `environment.yaml`;
  `pip==23.3` and `setuptools<70` because newer versions cannot build some of those pins;
  `pyarrow==14.0.2`; `matplotlib==3.6.2` because GCDM imports a module removed in 3.7.
  `environment-record.txt` lists the exact versions of the environment that produced the
  reported results.
- both repositories download QM9 from `springernature.figshare.com/ndownloader`, which no
  longer resolves; `setup.sh` rewrites those URLs to `ndownloader.figshare.com`.
- commits used: EDM `fce07d7`, GCDM `a328950`; GCDM checkpoints from Zenodo record 13375913.

The commands `run.sh` executes for sampling and native evaluation:

    # EDM, from edm/
    python eval_analyze.py --model_path outputs/edm_qm9 --n_samples 10000 \
        --batch_size_gen 100 --save_to_xyz True

    # GCDM, from gcdm/
    PROJECT_ROOT=$PWD python src/mol_gen_eval.py datamodule=edm_qm9 model=qm9_mol_gen_ddpm \
        logger=csv trainer.accelerator=gpu "trainer.devices=[0]" \
        ckpt_path=checkpoints/QM9/Unconditional/model_1_epoch_979-EMA.ckpt \
        datamodule.dataloader_cfg.num_workers=1 model.diffusion_cfg.sample_during_training=false \
        num_samples=10000 sampling_batch_size=500 num_test_passes=5 save_molecules=True \
        output_dir=output/QM9/Unconditional/gcdm_model_1/

followed by, from this directory:

    python unified_eval.py --samples EDM=edm/outputs/edm_qm9/eval/analyzed_molecules \
        GCDM=gcdm/output/QM9/Unconditional/gcdm_model_1 --out results/unified_all.json
    python bootstrap_eval.py
    python make_figures.py

The generated xyz files, run logs and the per-molecule tables were left out of the repository
for size.
