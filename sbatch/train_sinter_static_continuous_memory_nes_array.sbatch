#!/bin/bash
#SBATCH --job-name=bike144_static_cont_nes_train
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
SCRIPT_PATH="${SCRIPT_PATH:-${REPO_ROOT}/scripts/train_sinter_static_memory_assignment.py}"
CIRCUITS_DIR="${CIRCUITS_DIR:-${REPO_ROOT}/tests/testdata/bicycle_bivariate}"
CIRCUIT_FILE="${CIRCUIT_FILE:-circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=0.003,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim}"
FILENAME_CONTAINS="${FILENAME_CONTAINS:-circuit=bicycle_bivariate_144_12_12_memory_Z,error_rate=0.003}"
BASIS_FILTER="${BASIS_FILTER:-Z}"
MAX_CIRCUITS="${MAX_CIRCUITS:-1}"
ASSIGNMENT_MODE="continuous"
CONTINUOUS_OPTIMIZER="${CONTINUOUS_OPTIMIZER:-nes}"
WARM_START_MODE="${WARM_START_MODE:-same-seed}"
WARM_START_BASE="${WARM_START_BASE:-${REPO_ROOT}/examples/notebook_data/bicycle_bivariate_144_12_12_static_continuous_memory_Z_p003_training}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/examples/notebook_data/bicycle_bivariate_144_12_12_static_continuous_memory_nes_Z_p003_training}"
TRAIN_SHOTS="${TRAIN_SHOTS:-150000}"
VALIDATION_SHOTS="${VALIDATION_SHOTS:-150000}"
GENERATIONS="${GENERATIONS:-12}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-32}"
ELITE_COUNT="${ELITE_COUNT:-6}"
DISTRIBUTION_REPEATS="${DISTRIBUTION_REPEATS:-1}"
INITIALIZATION_CANDIDATES="${INITIALIZATION_CANDIDATES:-32}"
TRAIN_MAX_LOGICAL_FAILURES="${TRAIN_MAX_LOGICAL_FAILURES:-50}"
VALIDATION_MAX_LOGICAL_FAILURES="${VALIDATION_MAX_LOGICAL_FAILURES:-0}"
NEGATIVE="${NEGATIVE:--0.0687506131581227}"
POSITIVE="${POSITIVE:-0.3557972204473916}"
P_POSITIVE="${P_POSITIVE:-0.5451025293701163}"
GAMMA0="${GAMMA0:-0.125}"
PRE_ITER="${PRE_ITER:-80}"
NUM_SETS="${NUM_SETS:-600}"
SET_MAX_ITER="${SET_MAX_ITER:-60}"
STOP_NCONV="${STOP_NCONV:-5}"
CONTINUOUS_NES_LEARNING_RATE="${CONTINUOUS_NES_LEARNING_RATE:-0.2}"
CONTINUOUS_NES_SIGMA_LEARNING_RATE="${CONTINUOUS_NES_SIGMA_LEARNING_RATE:-0.05}"
CONTINUOUS_NES_IMPACT_DECAY="${CONTINUOUS_NES_IMPACT_DECAY:-0.9}"
CONTINUOUS_SIGMA_MIN="${CONTINUOUS_SIGMA_MIN:-0.0001}"
CONTINUOUS_SIGMA_MAX="${CONTINUOUS_SIGMA_MAX:-0.25}"
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
echo "WARM_START_BASE=${WARM_START_BASE}"
echo "WARM_START_MODE=${WARM_START_MODE}"
echo "ASSIGNMENT_MODE=${ASSIGNMENT_MODE}"
echo "CONTINUOUS_OPTIMIZER=${CONTINUOUS_OPTIMIZER}"
echo "TRAIN_SHOTS=${TRAIN_SHOTS}"
echo "VALIDATION_SHOTS=${VALIDATION_SHOTS}"
echo "GENERATIONS=${GENERATIONS}"
echo "CANDIDATE_COUNT=${CANDIDATE_COUNT}"
echo "ELITE_COUNT=${ELITE_COUNT}"
echo "CONTINUOUS_NES_LEARNING_RATE=${CONTINUOUS_NES_LEARNING_RATE}"
echo "CONTINUOUS_NES_SIGMA_LEARNING_RATE=${CONTINUOUS_NES_SIGMA_LEARNING_RATE}"
echo "CONTINUOUS_NES_IMPACT_DECAY=${CONTINUOUS_NES_IMPACT_DECAY}"
echo "CONTINUOUS_SIGMA_MIN=${CONTINUOUS_SIGMA_MIN}"
echo "CONTINUOUS_SIGMA_MAX=${CONTINUOUS_SIGMA_MAX}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --circuits-dir "${CIRCUITS_DIR}" \
  --circuit-file "${CIRCUIT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --filename-contains "${FILENAME_CONTAINS}" \
  --basis-filter "${BASIS_FILTER}" \
  --max-circuits "${MAX_CIRCUITS}" \
  --assignment-mode "${ASSIGNMENT_MODE}" \
  --continuous-optimizer "${CONTINUOUS_OPTIMIZER}" \
  --warm-start-base "${WARM_START_BASE}" \
  --warm-start-mode "${WARM_START_MODE}" \
  --continuous-nes-learning-rate "${CONTINUOUS_NES_LEARNING_RATE}" \
  --continuous-nes-sigma-learning-rate "${CONTINUOUS_NES_SIGMA_LEARNING_RATE}" \
  --continuous-nes-impact-decay "${CONTINUOUS_NES_IMPACT_DECAY}" \
  --continuous-sigma-min "${CONTINUOUS_SIGMA_MIN}" \
  --continuous-sigma-max "${CONTINUOUS_SIGMA_MAX}" \
  --train-shots "${TRAIN_SHOTS}" \
  --validation-shots "${VALIDATION_SHOTS}" \
  --seed "${SEED}" \
  --generations "${GENERATIONS}" \
  --candidate-count "${CANDIDATE_COUNT}" \
  --elite-count "${ELITE_COUNT}" \
  --distribution-repeats "${DISTRIBUTION_REPEATS}" \
  --initialization-candidates "${INITIALIZATION_CANDIDATES}" \
  --train-max-logical-failures "${TRAIN_MAX_LOGICAL_FAILURES}" \
  --validation-max-logical-failures "${VALIDATION_MAX_LOGICAL_FAILURES}" \
  --negative "${NEGATIVE}" \
  --positive "${POSITIVE}" \
  --p-positive "${P_POSITIVE}" \
  --gamma0 "${GAMMA0}" \
  --pre-iter "${PRE_ITER}" \
  --num-sets "${NUM_SETS}" \
  --set-max-iter "${SET_MAX_ITER}" \
  --stop-nconv "${STOP_NCONV}" \
  ${EXTRA_ARGS} \
  "$@"
