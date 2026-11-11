#!/bin/bash
#SBATCH -J p2_venv
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/mkvenv.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/mkvenv.err
set -euo pipefail
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
export TMPDIR=/mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/.piptmp
mkdir -p "$TMPDIR"
/mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/.venv/bin/python -m venv --without-pip venv2 2>/dev/null || true
rm -rf venv2
python3 -m venv venv2
./venv2/bin/python -m pip install -q --upgrade pip
./venv2/bin/pip install -q "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
./venv2/bin/pip install -q "transformers==4.57.6" "numpy<2" "scipy" "pandas" \
    "matplotlib" "huggingface_hub" "safetensors" "accelerate" "pyarrow"
./venv2/bin/python - <<'PY'
import torch, transformers, numpy, scipy
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("transformers", transformers.__version__, "numpy", numpy.__version__)
PY
rm -rf "$TMPDIR"
echo VENV_DONE
