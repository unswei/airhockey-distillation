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
readonly ORIGINAL_DATASET="teacher-v3-structured-n64-k2-full-deterministic-2026-08-12-v1"
readonly SHADOW_DATASET="teacher-v3-structured-n64-k2-shadow-round1-deterministic-2026-08-13-v1"
readonly AGGREGATE_DATASET="teacher-v3-structured-n64-k2-teacher-plus-shadow-round1-2026-08-13-v1"
readonly STUDENT_RUN="structured-n64-k2-shadow-round1-seed-14303-2026-08-13-v1"
readonly BEHAVIOUR_STUDENT="structured-n64-k2-full-all-steps-seed-14303-2026-08-13-v2"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly CONFIG="configs/student/structured_n64_k2_shadow_round1_v1.yaml"

cd "${CODE_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test -z "$(git status --porcelain)"
test ! -e "${RUN_ROOT}"
test ! -e "${EXPERIMENT_ROOT}/teacher-datasets/${SHADOW_DATASET}"
test ! -e "${EXPERIMENT_ROOT}/teacher-datasets/${AGGREGATE_DATASET}"
test ! -e "${EXPERIMENT_ROOT}/students/${STUDENT_RUN}"
mkdir -p "${RUN_ROOT}"

docker run --rm --gpus all --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/collect_student_shadow_dataset.py \
    --config "${CONFIG}" \
    --teacher-config configs/teacher/dreamerv3_v3.yaml \
    --teacher-checkpoint /experiments/frozen-teachers/dreamerv3-teacher-v3-step-700000 \
    --student-checkpoint "/experiments/students/${BEHAVIOUR_STUDENT}/checkpoint.npz" \
    --output "/experiments/teacher-datasets/${SHADOW_DATASET}" \
    --code-commit "${CODE_COMMIT}" \
  2>&1 | tee "${RUN_ROOT}/collection.log"

docker run --rm --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/aggregate_teacher_shadow_datasets.py \
    --config "${CONFIG}" \
    --original-dataset "/experiments/teacher-datasets/${ORIGINAL_DATASET}" \
    --shadow-dataset "/experiments/teacher-datasets/${SHADOW_DATASET}" \
    --output "/experiments/teacher-datasets/${AGGREGATE_DATASET}" \
    --code-commit "${CODE_COMMIT}" \
  2>&1 | tee "${RUN_ROOT}/aggregation.log"

docker run --rm --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/train_structured_student.py \
    --config "${CONFIG}" \
    --dataset "/experiments/teacher-datasets/${AGGREGATE_DATASET}" \
    --output "/experiments/students/${STUDENT_RUN}" \
    --code-commit "${CODE_COMMIT}" \
  2>&1 | tee "${RUN_ROOT}/training.log"

docker run --rm --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/evaluate_structured_student.py \
    --config "${CONFIG}" \
    --checkpoint "/experiments/students/${STUDENT_RUN}/checkpoint.npz" \
    --output "/experiments/students/${STUDENT_RUN}/no_blackout_validation.json" \
    --code-commit "${CODE_COMMIT}" \
  2>&1 | tee "${RUN_ROOT}/evaluation.log"

docker run --rm --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/gate_structured_no_blackout.py \
    --config "${CONFIG}" \
    --training-result "/experiments/students/${STUDENT_RUN}/result.json" \
    --evaluation-result "/experiments/students/${STUDENT_RUN}/no_blackout_validation.json" \
    --output "/experiments/${RUN_ID}/no_blackout_gate.json" \
    --code-commit "${CODE_COMMIT}" \
  2>&1 | tee "${RUN_ROOT}/gate.log"

sha256sum \
  "${EXPERIMENT_ROOT}/teacher-datasets/${SHADOW_DATASET}/manifest.json" \
  "${EXPERIMENT_ROOT}/teacher-datasets/${AGGREGATE_DATASET}/manifest.json" \
  "${EXPERIMENT_ROOT}/students/${STUDENT_RUN}/checkpoint.npz" \
  "${EXPERIMENT_ROOT}/students/${STUDENT_RUN}/result.json" \
  "${EXPERIMENT_ROOT}/students/${STUDENT_RUN}/no_blackout_validation.json" \
  "${RUN_ROOT}/no_blackout_gate.json" \
  > "${RUN_ROOT}/sha256sums.txt"

cat "${RUN_ROOT}/no_blackout_gate.json"
