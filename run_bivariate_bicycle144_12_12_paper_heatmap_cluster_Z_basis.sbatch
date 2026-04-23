#!/bin/bash
#SBATCH --job-name=bike144_paper_heatmap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=common
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_ROOT/run_bivariate_bicycle144_12_12_paper_heatmap_cluster_Z_basis.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/examples/notebook_data/bicycle_bivariate_144_12_12_paper_heatmap_cluster_Z_basis}"

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
