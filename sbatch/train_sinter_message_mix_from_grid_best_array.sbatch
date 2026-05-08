#!/bin/bash
#SBATCH --job-name=bike144_msgmix_refine
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
WRAPPER_PATH="${WRAPPER_PATH:-${REPO_ROOT}/scripts/train_sinter_message_mix_from_grid_best.py}"
TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT_PATH:-${REPO_ROOT}/scripts/train_sinter_bernoulli_message_mix_distribution.py}"
CIRCUITS_DIR="${CIRCUITS_DIR:-${REPO_ROOT}/tests/testdata/bicycle_bivariate}"
GRID_OUTPUT_BASE="${GRID_OUTPUT_BASE:-${REPO_ROOT}/examples/notebook_data/bicycle_bivariate_144_12_12_message_mix_grid_Z_p0035_r60}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/examples/notebook_data/bicycle_bivariate_144_12_12_message_mix_grid_refine_Z_p003_training}"
CIRCUIT_FILE="${CIRCUIT_FILE:-circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=0.003,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim}"
FILENAME_CONTAINS="${FILENAME_CONTAINS:-circuit=bicycle_bivariate_144_12_12_memory_Z,error_rate=0.003}"
BASIS_FILTER="${BASIS_FILTER:-Z}"
MAX_CIRCUITS="${MAX_CIRCUITS:-1}"
TRAIN_SHOTS="${TRAIN_SHOTS:-150000}"
VALIDATION_SHOTS="${VALIDATION_SHOTS:-150000}"
GENERATIONS="${GENERATIONS:-12}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-32}"
ELITE_COUNT="${ELITE_COUNT:-6}"
DISTRIBUTION_REPEATS="${DISTRIBUTION_REPEATS:-1}"
TRAIN_MAX_LOGICAL_FAILURES="${TRAIN_MAX_LOGICAL_FAILURES:-50}"
VALIDATION_MAX_LOGICAL_FAILURES="${VALIDATION_MAX_LOGICAL_FAILURES:-0}"
GAMMA0="${GAMMA0:-none}"
PRE_ITER="${PRE_ITER:-80}"
NUM_SETS="${NUM_SETS:-600}"
SET_MAX_ITER="${SET_MAX_ITER:-60}"
STOP_NCONV="${STOP_NCONV:-5}"
RANK_COUNT="${RANK_COUNT:-5}"
RUNS_PER_CANDIDATE="${RUNS_PER_CANDIDATE:-2}"
BASE_SEED="${BASE_SEED:-0}"
WARM_START_RAW_STD="${WARM_START_RAW_STD:-0.7}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cd "${REPO_ROOT}"

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
RUN_SEED="$((BASE_SEED + TASK_INDEX))"
OUTPUT_DIR="${OUTPUT_BASE}/seed_${TASK_INDEX}"
mkdir -p "${OUTPUT_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "WRAPPER_PATH=${WRAPPER_PATH}"
echo "TRAIN_SCRIPT_PATH=${TRAIN_SCRIPT_PATH}"
echo "GRID_OUTPUT_BASE=${GRID_OUTPUT_BASE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CIRCUITS_DIR=${CIRCUITS_DIR}"
echo "CIRCUIT_FILE=${CIRCUIT_FILE}"
echo "FILENAME_CONTAINS=${FILENAME_CONTAINS}"
echo "BASIS_FILTER=${BASIS_FILTER}"
echo "TRAIN_SHOTS=${TRAIN_SHOTS}"
echo "VALIDATION_SHOTS=${VALIDATION_SHOTS}"
echo "GENERATIONS=${GENERATIONS}"
echo "CANDIDATE_COUNT=${CANDIDATE_COUNT}"
echo "ELITE_COUNT=${ELITE_COUNT}"
echo "TRAIN_MAX_LOGICAL_FAILURES=${TRAIN_MAX_LOGICAL_FAILURES}"
echo "VALIDATION_MAX_LOGICAL_FAILURES=${VALIDATION_MAX_LOGICAL_FAILURES}"
echo "GAMMA0=${GAMMA0}"
echo "PRE_ITER=${PRE_ITER}"
echo "NUM_SETS=${NUM_SETS}"
echo "SET_MAX_ITER=${SET_MAX_ITER}"
echo "STOP_NCONV=${STOP_NCONV}"
echo "TASK_INDEX=${TASK_INDEX}"
echo "RUN_SEED=${RUN_SEED}"
echo "RANK_COUNT=${RANK_COUNT}"
echo "RUNS_PER_CANDIDATE=${RUNS_PER_CANDIDATE}"
echo "WARM_START_RAW_STD=${WARM_START_RAW_STD}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"

"${PYTHON_BIN}" "${WRAPPER_PATH}" \
  --grid-output-base "${GRID_OUTPUT_BASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --task-index "${TASK_INDEX}" \
  --rank-count "${RANK_COUNT}" \
  --runs-per-candidate "${RUNS_PER_CANDIDATE}" \
  --train-script "${TRAIN_SCRIPT_PATH}" \
  --python-bin "${PYTHON_BIN}" \
  --warm-start-raw-std "${WARM_START_RAW_STD}" \
  --circuits-dir "${CIRCUITS_DIR}" \
  --circuit-file "${CIRCUIT_FILE}" \
  --filename-contains "${FILENAME_CONTAINS}" \
  --basis-filter "${BASIS_FILTER}" \
  --max-circuits "${MAX_CIRCUITS}" \
  --train-shots "${TRAIN_SHOTS}" \
  --validation-shots "${VALIDATION_SHOTS}" \
  --seed "${RUN_SEED}" \
  --generations "${GENERATIONS}" \
  --candidate-count "${CANDIDATE_COUNT}" \
  --elite-count "${ELITE_COUNT}" \
  --distribution-repeats "${DISTRIBUTION_REPEATS}" \
  --train-max-logical-failures "${TRAIN_MAX_LOGICAL_FAILURES}" \
  --validation-max-logical-failures "${VALIDATION_MAX_LOGICAL_FAILURES}" \
  --gamma0 "${GAMMA0}" \
  --pre-iter "${PRE_ITER}" \
  --num-sets "${NUM_SETS}" \
  --set-max-iter "${SET_MAX_ITER}" \
  --stop-nconv "${STOP_NCONV}" \
  ${EXTRA_ARGS} \
  "$@"
