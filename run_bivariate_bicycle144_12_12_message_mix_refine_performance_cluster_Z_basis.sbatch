#!/bin/bash
#SBATCH --job-name=bike144_msgmix_perf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=common
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-${REPO_ROOT}/run_bivariate_bicycle144_12_12_message_mix_refine_performance_cluster_Z_basis.py}"
REFINEMENT_BASE="${REFINEMENT_BASE:-${REPO_ROOT}/examples/notebook_data/bicycle_bivariate_144_12_12_message_mix_grid_refine_Z_p003_training}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/examples/notebook_data/bicycle_bivariate_144_12_12_message_mix_grid_refine_relay_bp5_performance_cluster_Z_basis}"
P_VALUES="${P_VALUES:-0.001,0.002,0.003,0.004,0.005}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
PRINT_PROGRESS="${PRINT_PROGRESS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cd "${REPO_ROOT}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "SCRIPT_PATH=${SCRIPT_PATH}"
echo "REFINEMENT_BASE=${REFINEMENT_BASE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "P_VALUES=${P_VALUES}"
echo "MAX_BATCH_SIZE=${MAX_BATCH_SIZE}"
echo "PRINT_PROGRESS=${PRINT_PROGRESS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"

PROGRESS_ARGS=()
if [[ "${PRINT_PROGRESS}" != "0" ]]; then
  PROGRESS_ARGS+=(--print-progress)
fi

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --repo-root "${REPO_ROOT}" \
  --refinement-base "${REFINEMENT_BASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  --p-values "${P_VALUES}" \
  --max-batch-size "${MAX_BATCH_SIZE}" \
  "${PROGRESS_ARGS[@]}" \
  ${EXTRA_ARGS} \
  "$@"
