#!/bin/bash
#SBATCH -J nettest
#SBATCH -p defq
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:05:00
#SBATCH -o /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/nettest.out
#SBATCH -e /mnt/isilon/tan_lab/pany3/CIS6270-both-modalities/project2/logs/nettest.err
hostname
echo "--- curl HF ---"
curl -sI --max-time 20 https://huggingface.co/api/models/airkingbd/dplm_150m | head -2
echo "curl_exit=$?"
echo "--- scratch writable? ---"
ls -d /scratch /lscratch /local 2>/dev/null
df -h /mnt/isilon/tan_lab/pany3 | tail -1
