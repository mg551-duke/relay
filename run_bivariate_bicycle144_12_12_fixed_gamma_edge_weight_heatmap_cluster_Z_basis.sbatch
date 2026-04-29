#!/bin/bash
#SBATCH --job-name=bike144_edge_weight_heatmap
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_ROOT/run_bivariate_bicycle144_12_12_fixed_gamma_edge_weight_heatmap_cluster_Z_basis.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/examples/notebook_data/bicycle_bivariate_144_12_12_relay_bp1_r31_fixed_gamma_edge_weight_heatmap_cluster_Z_basis}"

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
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-unset}"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --repo-root "$REPO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --print-progress \
  "$@"
