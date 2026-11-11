#!/bin/bash
#SBATCH -J p2_fix
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/fixdeps.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/fixdeps.err
set -euo pipefail
cd /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2
export TMPDIR=$PWD/.piptmp; mkdir -p "$TMPDIR"
./venv2/bin/pip install -q "pyarrow>=16" "pandas==2.2.3"
./venv2/bin/python -c "
import pandas as pd, pyarrow
print('pandas',pd.__version__,'pyarrow',pyarrow.__version__)
p='/home/pany3/.cache/huggingface/hub/datasets--DanielHesslow--SwissProt-Pfam/snapshots/0db4e0ea3153c4caa5c49e7387f65b78a8996148/data/train-00000-of-00001.parquet'
d=pd.read_parquet(p, columns=['seq']); print('rows', len(d))
"
rm -rf "$TMPDIR"; echo FIXDEPS_DONE
