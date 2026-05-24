#!/bin/bash
#SBATCH --job-name=bike144_relay_bp5_perf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=common
#SBATCH --cpus-per-task=100
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_ROOT/run_bivariate_bicycle144_12_12_relay_bp5_performance_cluster_Z_basis.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/examples/notebook_data/bicycle_bivariate_144_12_12_relay_bp5_performance_cluster_Z_basis}"
STATIC_DISCRETE_ASSIGNMENT_BANK_BASE="${STATIC_DISCRETE_ASSIGNMENT_BANK_BASE:-$REPO_ROOT/examples/notebook_data/bicycle_bivariate_144_12_12_static_discrete_memory_Z_p003_training}"
STATIC_DISCRETE_ASSIGNMENT_BANK_TOP_K="${STATIC_DISCRETE_ASSIGNMENT_BANK_TOP_K:-10}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cd "$REPO_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "SCRIPT_PATH=$SCRIPT_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STATIC_DISCRETE_ASSIGNMENT_BANK_BASE=${STATIC_DISCRETE_ASSIGNMENT_BANK_BASE:-unset}"
echo "STATIC_DISCRETE_ASSIGNMENT_BANK_TOP_K=$STATIC_DISCRETE_ASSIGNMENT_BANK_TOP_K"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-unset}"

EXTRA_SCRIPT_ARGS=()
if [[ -n "${STATIC_DISCRETE_ASSIGNMENT_BANK_BASE}" ]]; then
  EXTRA_SCRIPT_ARGS+=(
    --static-discrete-assignment-bank-base "$STATIC_DISCRETE_ASSIGNMENT_BANK_BASE"
    --static-discrete-assignment-bank-top-k "$STATIC_DISCRETE_ASSIGNMENT_BANK_TOP_K"
  )
fi

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --repo-root "$REPO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --print-progress \
  "${EXTRA_SCRIPT_ARGS[@]}" \
  "$@"
