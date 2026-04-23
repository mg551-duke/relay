#!/bin/bash
#SBATCH --job-name=gross144_memX_damp
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/gross144_memX_damp_%A_%a.out
#SBATCH --error=logs/gross144_memX_damp_%A_%a.err
#SBATCH --array=0-0

# Edit these paths for your cluster checkout.
REPO_ROOT="/hpc/home/mg551/storage/relay/relay"
VENV_PATH="/hpc/home/mg551/storage/relay/.venv/bin/activate"

source "${VENV_PATH}"
cd "${REPO_ROOT}"

OUTDIR="${REPO_ROOT}/results_gross_144_12_12_memory_X_disordered_damping_sinter"
CIRCUITS_DIR="${REPO_ROOT}/tests/testdata/bicycle_bivariate"
DECODER_CONFIG="${REPO_ROOT}/configs/decoder_specs_gross_144_12_12_memory_X_disordered_damping_sinter.json"

mkdir -p "${OUTDIR}" "${REPO_ROOT}/logs"

python "${REPO_ROOT}/scripts/run_gross_144_12_12_memory_X_disordered_damping_sinter.py" \
  --circuits-dir "${CIRCUITS_DIR}" \
  --decoder-config "${DECODER_CONFIG}" \
  --outdir "${OUTDIR}" \
  --filename-contains "circuit=bicycle_bivariate_144_12_12_memory_X" \
  --max-shots 10000 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --print-progress
