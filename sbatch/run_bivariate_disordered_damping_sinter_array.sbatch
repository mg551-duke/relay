#!/bin/bash
#SBATCH --job-name=bivariate_damp_sinter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/bivariate_damp_sinter_%A_%a.out
#SBATCH --error=logs/bivariate_damp_sinter_%A_%a.err
#SBATCH --exclusive

set -euo pipefail

# Edit these paths for your cluster checkout.
REPO_ROOT="/hpc/home/mg551/storage/relay/relay"
source /hpc/home/mg551/storage/relay/.venv/bin/activate
cd "${REPO_ROOT}"

OUTDIR_BASE="/hpc/home/mg551/storage/relay/relay/results_bivariate_disordered_damping_sinter"
CIRCUITS_DIR="${REPO_ROOT}/tests/testdata/bicycle_bivariate"
DECODER_CONFIG="${REPO_ROOT}/configs/decoder_specs_bivariate_disordered_damping.json"
SEED=${SLURM_ARRAY_TASK_ID}
OUTDIR="${OUTDIR_BASE}/seed_${SEED}"

mkdir -p "${OUTDIR}" logs

python scripts/run_bivariate_disordered_damping_sinter.py \
  --circuits-dir "${CIRCUITS_DIR}" \
  --outdir "${OUTDIR}" \
  --decoder-config "${DECODER_CONFIG}" \
  --save-name "bivariate_disordered_damping_seed_${SEED}" \
  --max-shots 1000 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --print-progress
