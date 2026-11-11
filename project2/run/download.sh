#!/bin/bash
#SBATCH -J p2_dl
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/download.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/download.err
set -euo pipefail
export HF_HOME=/home/pany3/.cache/huggingface
export HF_HUB_ENABLE_HXET=0
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities
./.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["facebook/esm2_t30_150M_UR50D", "facebook/esmfold_v1", "airkingbd/dplm_650m"]:
    print("downloading", r, flush=True)
    p = snapshot_download(r, allow_patterns=["*.json","*.bin","*.txt","*.model","*.safetensors"])
    print("OK", r, p, flush=True)
PY
du -sh /home/pany3/.cache/huggingface/hub/models--facebook--esmfold_v1 || true
echo DONE
