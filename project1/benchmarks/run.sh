#!/usr/bin/env bash
# Runs the benchmark after setup.sh: sampling with native evaluation for EDM and GCDM, shared
# scoring, bootstrap intervals, figures. Stages whose outputs exist are skipped, so the script
# can be re-run after an interruption.
#
# Sampling is also skipped when a sample set is already available as results/samples_<MODEL>.tar.gz
# (the committed 10,000-molecule sets): the archive is unpacked into samples/<MODEL> and the
# scoring, bootstrap and figure stages run from it on the CPU in about ten minutes. The test NLL
# cannot be recomputed from samples and is then carried over from the committed results.
#
# Usage: bash run.sh              (EDM then GCDM on one GPU, about 2 h + 4.5 h)
#        PARALLEL=1 bash run.sh   (both samplers at once; together they used under 6 GB of GPU memory)
#        N_SAMPLES=1000 bash run.sh   (smoke test; ignores the archives)
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-qm9bench}"
N_SAMPLES="${N_SAMPLES:-10000}"
cd "$BENCH_DIR"; mkdir -p results
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
run() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }
declare -A REPO_SAMPLES=( [EDM]=edm/outputs/edm_qm9/eval/analyzed_molecules [GCDM]=gcdm/output/QM9/Unconditional/gcdm_model_1 )

samples_dir() {  # prints an existing directory holding $1's samples: the repo output, else the unpacked archive
  local repo="${REPO_SAMPLES[$1]}"
  if [ -d "$repo" ] && [ "$(ls "$repo" | wc -l)" -ge "$N_SAMPLES" ]; then echo "$repo"; return; fi
  if [ "$N_SAMPLES" = 10000 ] && [ ! -d "samples/$1" ] && [ -f "results/samples_$1.tar.gz" ]; then
    mkdir -p samples && tar -xzf "results/samples_$1.tar.gz" -C samples
  fi
  [ -d "samples/$1" ] && echo "samples/$1" || true
}
edm_done()  { grep -qs "Final test nll" results/edm_eval.log || { [ ! -s results/edm_eval.log ] && [ -n "$(samples_dir EDM)" ]; }; }
gcdm_done() { grep -qs "Test negative log-likelihood" results/gcdm_eval.log || { [ ! -s results/gcdm_eval.log ] && [ -n "$(samples_dir GCDM)" ]; }; }

sample_edm() {
  echo "== EDM: $N_SAMPLES samples + native metrics + test NLL (log: results/edm_eval.log)"
  ( cd edm && run python eval_analyze.py --model_path outputs/edm_qm9 --n_samples "$N_SAMPLES" \
      --batch_size_gen 100 --save_to_xyz True ) > results/edm_eval.log 2>&1
}
sample_gcdm() {
  echo "== GCDM: $N_SAMPLES samples + native metrics + test NLL over 5 passes (log: results/gcdm_eval.log)"
  ( cd gcdm && PROJECT_ROOT=$PWD run python src/mol_gen_eval.py datamodule=edm_qm9 model=qm9_mol_gen_ddpm \
      logger=csv trainer.accelerator=gpu "trainer.devices=[0]" \
      ckpt_path=checkpoints/QM9/Unconditional/model_1_epoch_979-EMA.ckpt \
      datamodule.dataloader_cfg.num_workers=1 model.diffusion_cfg.sample_during_training=false \
      num_samples="$N_SAMPLES" sampling_batch_size=500 num_test_passes=5 save_molecules=True \
      output_dir=output/QM9/Unconditional/gcdm_model_1/ ) > results/gcdm_eval.log 2>&1
}

if [ "${PARALLEL:-0}" = "1" ]; then
  edm_done  || sample_edm &
  gcdm_done || sample_gcdm
  wait
else
  edm_done  || sample_edm
  gcdm_done || sample_gcdm
fi
edm_done && gcdm_done || { echo "a sampling run did not finish; see the logs in results/"; exit 1; }
EDM_SAMPLES="$(samples_dir EDM)"; GCDM_SAMPLES="$(samples_dir GCDM)"
echo "== samples: EDM=$EDM_SAMPLES GCDM=$GCDM_SAMPLES"

if [ "$N_SAMPLES" = 10000 ]; then   # keep the full sample sets as small archives next to the results
  for m in EDM GCDM; do
    d="$(samples_dir $m)"
    [ -f "results/samples_$m.tar.gz" ] || tar -czf "results/samples_$m.tar.gz" -C "$(dirname "$d")" \
        --transform "s#^$(basename "$d")#$m#" "$(basename "$d")"
  done
fi

echo "== shared scoring (results/unified_all.json)"
run python unified_eval.py --samples "EDM=$EDM_SAMPLES" "GCDM=$GCDM_SAMPLES" --out results/unified_all.json \
  2>&1 | grep -v "Ignoring invalid" | tee results/unified_all.log
echo "== bootstrap intervals (results/bootstrap.json)"
run python bootstrap_eval.py 2>&1 | grep -v "Ignoring invalid" | tee results/bootstrap.log
echo "== figures (figures/)"
run python make_figures.py
echo "== done"
