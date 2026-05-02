#!/bin/bash
#SBATCH --job-name=bike144_bernoulli_mem_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=common
#SBATCH --cpus-per-task=100
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-${REPO_ROOT}/scripts/train_sinter_bernoulli_memory_distribution.py}"
CIRCUITS_DIR="${CIRCUITS_DIR:-${REPO_ROOT}/tests/testdata/bicycle_bivariate}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/examples/notebook_data/bicycle_bivariate_144_12_12_bernoulli_memory_Z_p003_training}"
CIRCUIT_FILE="${CIRCUIT_FILE:-circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=0.003,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim}"
FILENAME_CONTAINS="${FILENAME_CONTAINS:-circuit=bicycle_bivariate_144_12_12_memory_Z,error_rate=0.003}"
BASIS_FILTER="${BASIS_FILTER:-Z}"
MAX_CIRCUITS="${MAX_CIRCUITS:-1}"
TRAIN_SHOTS="${TRAIN_SHOTS:-2000}"
VALIDATION_SHOTS="${VALIDATION_SHOTS:-2000}"
GENERATIONS="${GENERATIONS:-20}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-64}"
ELITE_COUNT="${ELITE_COUNT:-8}"
DISTRIBUTION_REPEATS="${DISTRIBUTION_REPEATS:-1}"
GAMMA0="${GAMMA0:-0.125}"
PRE_ITER="${PRE_ITER:-80}"
NUM_SETS="${NUM_SETS:-600}"
SET_MAX_ITER="${SET_MAX_ITER:-60}"
STOP_NCONV="${STOP_NCONV:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cd "${REPO_ROOT}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
OUTPUT_DIR="${OUTPUT_BASE}/seed_${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "SCRIPT_PATH=${SCRIPT_PATH}"
echo "CIRCUITS_DIR=${CIRCUITS_DIR}"
echo "CIRCUIT_FILE=${CIRCUIT_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FILENAME_CONTAINS=${FILENAME_CONTAINS}"
echo "BASIS_FILTER=${BASIS_FILTER}"
echo "MAX_CIRCUITS=${MAX_CIRCUITS}"
echo "GAMMA0=${GAMMA0}"
echo "PRE_ITER=${PRE_ITER}"
echo "NUM_SETS=${NUM_SETS}"
echo "SET_MAX_ITER=${SET_MAX_ITER}"
echo "STOP_NCONV=${STOP_NCONV}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --circuits-dir "${CIRCUITS_DIR}" \
  --circuit-file "${CIRCUIT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --filename-contains "${FILENAME_CONTAINS}" \
  --basis-filter "${BASIS_FILTER}" \
  --max-circuits "${MAX_CIRCUITS}" \
  --train-shots "${TRAIN_SHOTS}" \
  --validation-shots "${VALIDATION_SHOTS}" \
  --seed "${SEED}" \
  --generations "${GENERATIONS}" \
  --candidate-count "${CANDIDATE_COUNT}" \
  --elite-count "${ELITE_COUNT}" \
  --distribution-repeats "${DISTRIBUTION_REPEATS}" \
  --gamma0 "${GAMMA0}" \
  --pre-iter "${PRE_ITER}" \
  --num-sets "${NUM_SETS}" \
  --set-max-iter "${SET_MAX_ITER}" \
  --stop-nconv "${STOP_NCONV}" \
  ${EXTRA_ARGS} \
  "$@"
