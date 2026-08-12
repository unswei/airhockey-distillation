#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ID CODE_COMMIT" >&2
  exit 2
fi

readonly RUN_ID="$1"
readonly CODE_COMMIT="$2"
readonly CODE_ROOT="/home/oliver/Code/airhockey-memory-distillation"
readonly EXPERIMENT_ROOT="/home/oliver/experiments/airhockey-memory-distillation"
readonly RUN_ROOT="${EXPERIMENT_ROOT}/${RUN_ID}"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly CONFIG="configs/teacher/dreamerv3_v3.yaml"

cd "${CODE_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test -z "$(git status --porcelain)"
mkdir -p "${RUN_ROOT}"

docker run --rm --gpus all --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${RUN_ROOT}:/run-output:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/smoke_test_teacher.py \
    --config "${CONFIG}" \
    --profile full \
    --output /run-output/training \
    --code-commit "${CODE_COMMIT}" \
    --resume \
  2>&1 | tee -a "${RUN_ROOT}/training_console.log"

docker run --rm --gpus all --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${RUN_ROOT}:/run-output:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/evaluate_teacher_checkpoints.py \
    --config "${CONFIG}" \
    --profile full \
    --checkpoints /run-output/training/dreamer/ckpt \
    --output /run-output/validation \
    --code-commit "${CODE_COMMIT}" \
  2>&1 | tee -a "${RUN_ROOT}/validation_console.log"
