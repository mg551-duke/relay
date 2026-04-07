#!/bin/bash
#SBATCH --job-name=surfX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/surfX_%A_%a.out
#SBATCH --error=logs/surfX_%A_%a.err
#SBATCH --exclusive

set -euo pipefail

mkdir -p logs
source .venv/bin/activate

OUTDIR="/hpc/group/pfisterlab/mg551/storage/relay/relay/results_surface_x"
CIRCUITS_DIR="/hpc/group/pfisterlab/mg551/storage/relay/relay/tests/testdata/surface"
SEED=${SLURM_ARRAY_TASK_ID}

python run_surface_memoryX_folder_sinter.py \
  --circuits-dir "${CIRCUITS_DIR}" \
  --outdir "${OUTDIR}" \
  --seed "${SEED}" \
  --include-xyz \
  --include-pymatching \
  --max-errors 5
