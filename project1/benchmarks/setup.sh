#!/usr/bin/env bash
# One-time setup for the QM9 diffusion benchmark (EDM vs GCDM).
# Clones the two model repositories at the commits used, patches their QM9 download URLs,
# creates the conda environment, and downloads the QM9 parquet and the GCDM checkpoints.
# Usage: bash setup.sh            (everything lands next to this script)
#        ENV_NAME=myenv bash setup.sh
#        SKIP_CKPT=1 bash setup.sh   (no GCDM checkpoint download: enough to re-score the committed samples)
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-qm9bench}"
EDM_COMMIT=fce07d7                     # github.com/ehoogeboom/e3_diffusion_for_molecules
GCDM_COMMIT=a328950                    # github.com/BioinfoMachineLearning/Bio-Diffusion
CKPT_URL=https://zenodo.org/record/13375913/files/GCDM_Checkpoints.tar.gz
CKPT_MD5=f4bd5c6001b990b72d6fedfb651e57ec
cd "$BENCH_DIR"
mkdir -p results data

echo "== 1/5 model repositories"
[ -d edm/.git ] || git clone -q https://github.com/ehoogeboom/e3_diffusion_for_molecules edm
git -C edm checkout -q "$EDM_COMMIT"
[ -d gcdm/.git ] || git clone -q https://github.com/BioinfoMachineLearning/Bio-Diffusion gcdm
git -C gcdm checkout -q "$GCDM_COMMIT"
# the figshare host both repos download QM9 from no longer resolves; the plain mirror does
grep -rl "springernature.figshare.com/ndownloader" edm/qm9 gcdm/src 2>/dev/null \
  | xargs -r sed -i 's#https://springernature.figshare.com/ndownloader#https://ndownloader.figshare.com#g'

echo "== 2/5 conda environment: $ENV_NAME"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" -c pyg -c pytorch -c nvidia -c conda-forge \
    python=3.9 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 pyg=2.2.0 \
    pytorch-scatter=2.1.0 pytorch-cluster=1.6.0 pytorch-sparse=0.6.16 \
    openbabel=3.1.1 numpy=1.23.1 "mkl<2024.1"
  # newer pip/setuptools cannot build some of GCDM's pinned packages
  conda install -y -n "$ENV_NAME" -c conda-forge "pip=23.3" "setuptools<70"
fi
run() { PYTHONNOUSERSITE=1 conda run --no-capture-output -n "$ENV_NAME" "$@"; }
# GCDM's pip dependencies, taken from its environment.yaml (torch itself comes from conda above)
awk '/^  - pip:/{f=1;next} f && /^      - /{sub(/^      - /,"");print} f && !/^      - /{f=0}' \
  gcdm/environment.yaml | grep -vE '^(argparse|torch|torchvision|torchaudio)==' > results/gcdm_pip_requirements.txt
run pip install -q -r results/gcdm_pip_requirements.txt
run pip install -q -e gcdm --no-deps
run pip install -q "pyarrow==14.0.2" "matplotlib==3.6.2" huggingface_hub   # GCDM imports a module removed in matplotlib 3.7
run python -c "import torch, torch_geometric, pytorch_lightning, rdkit, hydra; from openbabel import pybel; print('env ok, cuda', torch.cuda.is_available())"

echo "== 3/5 QM9 parquet (structure-epflai/qm9)"
ls data/qm9/data/*.parquet >/dev/null 2>&1 || \
  run python -c "from huggingface_hub import snapshot_download; snapshot_download('structure-epflai/qm9', repo_type='dataset', local_dir='data/qm9')"

echo "== 4/5 GCDM checkpoints (2.4 GB from Zenodo, resumable)"
if [ "${SKIP_CKPT:-0}" != "1" ] && [ ! -f gcdm/checkpoints/QM9/Unconditional/model_1_epoch_979-EMA.ckpt ]; then
  for attempt in $(seq 1 30); do
    wget -c -q --tries=5 --timeout=60 -O gcdm/GCDM_Checkpoints.tar.gz "$CKPT_URL" || true
    [ "$(md5sum gcdm/GCDM_Checkpoints.tar.gz | cut -d' ' -f1)" = "$CKPT_MD5" ] && break
    echo "   checksum not yet matching after attempt $attempt, resuming"; sleep 10
  done
  [ "$(md5sum gcdm/GCDM_Checkpoints.tar.gz | cut -d' ' -f1)" = "$CKPT_MD5" ] || { echo "checkpoint download failed"; exit 1; }
  tar -xzf gcdm/GCDM_Checkpoints.tar.gz -C gcdm && rm gcdm/GCDM_Checkpoints.tar.gz
fi

echo "== 5/5 done"
echo "EDM checkpoint ships with its repo (edm/outputs/edm_qm9). Raw QM9 for both repos downloads"
echo "on first run. Next: bash run.sh"
