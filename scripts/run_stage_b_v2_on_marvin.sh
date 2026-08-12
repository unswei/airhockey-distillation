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
readonly CONFIG="configs/student/feed_forward_stage_b_v2.yaml"
readonly TEACHER_CONFIG="configs/teacher/dreamerv3_v3.yaml"
readonly TEACHER_RUN="teacher-full-v3-2026-08-12-v1"
readonly TEACHER_CHECKPOINT="20260812T111017F640089"
readonly FROZEN_TEACHER="frozen-teachers/dreamerv3-teacher-v3-step-700000"
readonly DATASET="teacher-datasets/teacher-v3-stage-b-v2-2026-08-12-v1"
readonly STUDENT="students/feed-forward-stage-b-v2-2026-08-12-v1"

cd "${CODE_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test -z "$(git status --porcelain)"
mkdir -p "${RUN_ROOT}"

if [[ ! -f "${EXPERIMENT_ROOT}/${FROZEN_TEACHER}/manifest.json" ]]; then
  docker run --rm --gpus all --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/freeze_teacher_checkpoint.py \
      --checkpoint "/experiments/${TEACHER_RUN}/training/dreamer/ckpt/${TEACHER_CHECKPOINT}" \
      --selection "/experiments/${TEACHER_RUN}/validation/selection.json" \
      --config "${TEACHER_CONFIG}" \
      --output "/experiments/${FROZEN_TEACHER}" \
      --teacher-id dreamerv3_teacher_v3 \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/freeze_console.log"
fi

if [[ ! -f "${EXPERIMENT_ROOT}/${DATASET}/manifest.json" ]]; then
  docker run --rm --gpus all --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/collect_teacher_dataset.py \
      --config "${CONFIG}" \
      --teacher-config "${TEACHER_CONFIG}" \
      --checkpoint "/experiments/${FROZEN_TEACHER}" \
      --output "/experiments/${DATASET}" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/dataset_console.log"
fi

if [[ ! -f "${EXPERIMENT_ROOT}/${STUDENT}/result.json" ]]; then
  test ! -e "${EXPERIMENT_ROOT}/${STUDENT}"
  docker run --rm --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/train_feed_forward.py \
      --config "${CONFIG}" \
      --dataset "/experiments/${DATASET}" \
      --output "/experiments/${STUDENT}" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/training_console.log"
fi

if [[ ! -f "${RUN_ROOT}/feed_forward.json" ]]; then
  docker run --rm --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/evaluate_feed_forward.py \
      --config "${CONFIG}" \
      --checkpoint "/experiments/${STUDENT}/checkpoint.npz" \
      --output "/experiments/${RUN_ID}/feed_forward.json" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/evaluation_console.log"
fi

if [[ ! -f "${RUN_ROOT}/memory_gate.json" ]]; then
  docker run --rm --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/run_stage_b_memory_gate.py \
      --config "${CONFIG}" \
      --teacher-result "/experiments/${TEACHER_RUN}/validation/checkpoint-000700000.json" \
      --feed-forward-result "/experiments/${RUN_ID}/feed_forward.json" \
      --output "/experiments/${RUN_ID}/memory_gate.json" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/memory_gate_console.log"
fi

sha256sum \
  "${EXPERIMENT_ROOT}/${FROZEN_TEACHER}/manifest.json" \
  "${EXPERIMENT_ROOT}/${FROZEN_TEACHER}/agent.pkl" \
  "${EXPERIMENT_ROOT}/${DATASET}/manifest.json" \
  "${EXPERIMENT_ROOT}/${STUDENT}/checkpoint.npz" \
  "${RUN_ROOT}/feed_forward.json" \
  "${RUN_ROOT}/memory_gate.json" \
  | tee "${RUN_ROOT}/sha256sums.txt"
