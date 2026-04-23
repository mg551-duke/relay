#!/bin/bash
#SBATCH --job-name=cc_assign_bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/cc_assign_bank_%A_%a.out
#SBATCH --error=logs/cc_assign_bank_%A_%a.err

set -euo pipefail

# Edit these defaults for your cluster checkout before submitting.
REPO_ROOT="${REPO_ROOT:-/hpc/home/USER/storage/relay}"
MANIFEST_CSV="${MANIFEST_CSV:-${REPO_ROOT}/manifests/code_capacity_static_assignment_manifest.csv}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

ROW_JSON="$("${PYTHON_BIN}" - "${MANIFEST_CSV}" "${TASK_INDEX}" <<'PY'
import csv
import json
import sys

manifest_path = sys.argv[1]
task_index = int(sys.argv[2])

with open(manifest_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if task_index < 0 or task_index >= len(rows):
    raise SystemExit(
        f"Array index {task_index} is out of range for manifest with {len(rows)} rows: {manifest_path}"
    )

print(json.dumps(rows[task_index]))
PY
)"

eval "$("${PYTHON_BIN}" - "${ROW_JSON}" <<'PY'
import json
import shlex
import sys

row = json.loads(sys.argv[1])
for key, value in row.items():
    print(f"{key.upper()}={shlex.quote(str(value))}")
PY
)"

"${PYTHON_BIN}" scripts/run_code_capacity_static_assignment_campaign.py run \
  --code-label "${CODE_LABEL}" \
  --code-path "${CODE_PATH}" \
  --p-value "${P_VALUE}" \
  --family "${FAMILY}" \
  --bank-size "${BANK_SIZE}" \
  --restart-index "${RESTART_INDEX}" \
  --output-dir "${OUTPUT_DIR}" \
  ${EXTRA_ARGS}
