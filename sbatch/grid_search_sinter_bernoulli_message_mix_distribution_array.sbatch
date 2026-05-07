#!/bin/bash
#SBATCH --job-name=bike144_msgmix_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=common
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-${REPO_ROOT}/scripts/grid_search_sinter_bernoulli_message_mix_distribution.py}"
CIRCUITS_DIR="${CIRCUITS_DIR:-${REPO_ROOT}/tests/testdata/bicycle_bivariate}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/examples/notebook_data/bicycle_bivariate_144_12_12_message_mix_grid_Z_p0035_r60}"
CIRCUIT_FILE="${CIRCUIT_FILE:-circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=0.0035,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim}"
FILENAME_CONTAINS="${FILENAME_CONTAINS:-circuit=bicycle_bivariate_144_12_12_memory_Z,error_rate=0.0035}"
BASIS_FILTER="${BASIS_FILTER:-Z}"
MAX_CIRCUITS="${MAX_CIRCUITS:-1}"
MAX_SHOTS="${MAX_SHOTS:-50000}"
SHOT_BATCH_SIZE="${SHOT_BATCH_SIZE:-256}"
TARGET_LOGICAL_ERRORS="${TARGET_LOGICAL_ERRORS:-10}"
DISTRIBUTION_REPEATS="${DISTRIBUTION_REPEATS:-1}"
SEED="${SEED:-0}"
GAMMA0="${GAMMA0:-none}"
PRE_ITER="${PRE_ITER:-80}"
NUM_SETS="${NUM_SETS:-60}"
SET_MAX_ITER="${SET_MAX_ITER:-60}"
STOP_NCONV="${STOP_NCONV:-5}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-10}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cd "${REPO_ROOT}"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
OUTPUT_DIR="${OUTPUT_BASE}/seed_${SHARD_INDEX}"
mkdir -p "${OUTPUT_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "SCRIPT_PATH=${SCRIPT_PATH}"
echo "CIRCUITS_DIR=${CIRCUITS_DIR}"
echo "CIRCUIT_FILE=${CIRCUIT_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FILENAME_CONTAINS=${FILENAME_CONTAINS}"
echo "BASIS_FILTER=${BASIS_FILTER}"
echo "MAX_SHOTS=${MAX_SHOTS}"
echo "SHOT_BATCH_SIZE=${SHOT_BATCH_SIZE}"
echo "TARGET_LOGICAL_ERRORS=${TARGET_LOGICAL_ERRORS}"
echo "DISTRIBUTION_REPEATS=${DISTRIBUTION_REPEATS}"
echo "SEED=${SEED}"
echo "GAMMA0=${GAMMA0}"
echo "PRE_ITER=${PRE_ITER}"
echo "NUM_SETS=${NUM_SETS}"
echo "SET_MAX_ITER=${SET_MAX_ITER}"
echo "STOP_NCONV=${STOP_NCONV}"
echo "SHARD_INDEX=${SHARD_INDEX}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --circuits-dir "${CIRCUITS_DIR}" \
  --circuit-file "${CIRCUIT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --filename-contains "${FILENAME_CONTAINS}" \
  --basis-filter "${BASIS_FILTER}" \
  --max-circuits "${MAX_CIRCUITS}" \
  --max-shots "${MAX_SHOTS}" \
  --shot-batch-size "${SHOT_BATCH_SIZE}" \
  --target-logical-errors "${TARGET_LOGICAL_ERRORS}" \
  --distribution-repeats "${DISTRIBUTION_REPEATS}" \
  --seed "${SEED}" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --gamma0 "${GAMMA0}" \
  --pre-iter "${PRE_ITER}" \
  --num-sets "${NUM_SETS}" \
  --set-max-iter "${SET_MAX_ITER}" \
  --stop-nconv "${STOP_NCONV}" \
  ${EXTRA_ARGS} \
  "$@"
