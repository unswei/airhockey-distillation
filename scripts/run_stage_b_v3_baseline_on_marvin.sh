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
readonly CONFIG="configs/student/feed_forward_stage_b_v3.yaml"
readonly TEACHER_CONFIG="configs/teacher/dreamerv3_v3.yaml"
readonly TEACHER_RUN="teacher-full-v3-2026-08-12-v1"
readonly FROZEN_TEACHER="frozen-teachers/dreamerv3-teacher-v3-step-700000"
readonly TEACHER_QUALIFICATION="${EXPERIMENT_ROOT}/${TEACHER_RUN}/validation/checkpoint-000700000.json"
readonly QUALIFICATION="${RUN_ROOT}/baseline_qualification.json"
readonly SEEDS=(14303 14304 14305)

cd "${CODE_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test -z "$(git status --porcelain)"
mkdir -p "${RUN_ROOT}"

for seed in "${SEEDS[@]}"; do
  student="students/feed-forward-ppo-stage-b-v3-seed-${seed}"
  student_root="${EXPERIMENT_ROOT}/${student}"
  qualification_result="${RUN_ROOT}/qualification-seed-${seed}.json"

  if [[ ! -f "${student_root}/result.json" ]]; then
    docker run --rm --ipc host \
      --env PYTHONPATH=/work/src:/src/2025-challenge \
      --volume "${CODE_ROOT}:/work:ro" \
      --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
      --workdir /work \
      "${IMAGE}" \
      python3 scripts/train_feed_forward_ppo.py \
        --config "${CONFIG}" \
        --profile full \
        --seed "${seed}" \
        --output "/experiments/${student}" \
        --code-commit "${CODE_COMMIT}" \
        --resume \
      2>&1 | tee -a "${RUN_ROOT}/training-seed-${seed}.log"
  fi

  if [[ ! -f "${qualification_result}" ]]; then
    docker run --rm --ipc host \
      --env PYTHONPATH=/work/src:/src/2025-challenge \
      --volume "${CODE_ROOT}:/work:ro" \
      --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
      --workdir /work \
      "${IMAGE}" \
      python3 scripts/evaluate_feed_forward_ppo.py \
        --config "${CONFIG}" \
        --evaluation qualification_evaluation \
        --checkpoint "/experiments/${student}/model.zip" \
        --training-result "/experiments/${student}/result.json" \
        --output "/experiments/${RUN_ID}/qualification-seed-${seed}.json" \
        --code-commit "${CODE_COMMIT}" \
      2>&1 | tee -a "${RUN_ROOT}/qualification-seed-${seed}.log"
  fi
done

if [[ ! -f "${QUALIFICATION}" ]]; then
  docker run --rm --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/qualify_feed_forward_baseline.py \
      --config "${CONFIG}" \
      --teacher-result "/experiments/${TEACHER_RUN}/validation/checkpoint-000700000.json" \
      --baseline-result "/experiments/${RUN_ID}/qualification-seed-14303.json" \
      --baseline-result "/experiments/${RUN_ID}/qualification-seed-14304.json" \
      --baseline-result "/experiments/${RUN_ID}/qualification-seed-14305.json" \
      --output "/experiments/${RUN_ID}/baseline_qualification.json" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/baseline_qualification.log"
fi

if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${QUALIFICATION}")" != "GO" ]]; then
  sha256sum "${RUN_ROOT}"/qualification-seed-*.json "${QUALIFICATION}" \
    | tee "${RUN_ROOT}/sha256sums.txt"
  echo "visible-observation baseline did not qualify; confirmation split remains unopened" >&2
  exit 3
fi

readonly SELECTED_SEED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_training_seed"])' "${QUALIFICATION}")"
readonly SELECTED_STUDENT="students/feed-forward-ppo-stage-b-v3-seed-${SELECTED_SEED}"
readonly TEACHER_CONFIRMATION="${RUN_ROOT}/teacher_confirmation.json"
readonly BASELINE_CONFIRMATION="${RUN_ROOT}/feed_forward_confirmation.json"

if [[ ! -f "${TEACHER_CONFIRMATION}" ]]; then
  docker run --rm --gpus all --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/evaluate_teacher_checkpoint.py \
      --config "${TEACHER_CONFIG}" \
      --profile full \
      --evaluation-config "${CONFIG}" \
      --evaluation-section confirmation_evaluation \
      --checkpoint "/experiments/${FROZEN_TEACHER}" \
      --output "/experiments/${RUN_ID}/teacher_confirmation.json" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/teacher_confirmation.log"
fi

if [[ ! -f "${BASELINE_CONFIRMATION}" ]]; then
  docker run --rm --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/evaluate_feed_forward_ppo.py \
      --config "${CONFIG}" \
      --evaluation confirmation_evaluation \
      --checkpoint "/experiments/${SELECTED_STUDENT}/model.zip" \
      --training-result "/experiments/${SELECTED_STUDENT}/result.json" \
      --output "/experiments/${RUN_ID}/feed_forward_confirmation.json" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/feed_forward_confirmation.log"
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
      --teacher-result "/experiments/${RUN_ID}/teacher_confirmation.json" \
      --feed-forward-result "/experiments/${RUN_ID}/feed_forward_confirmation.json" \
      --output "/experiments/${RUN_ID}/memory_gate.json" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/memory_gate.log"
fi

sha256sum \
  "${QUALIFICATION}" \
  "${EXPERIMENT_ROOT}/${SELECTED_STUDENT}/model.zip" \
  "${TEACHER_CONFIRMATION}" \
  "${BASELINE_CONFIRMATION}" \
  "${RUN_ROOT}/memory_gate.json" \
  | tee "${RUN_ROOT}/sha256sums.txt"
