#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_ID STUDENT_RUN CODE_COMMIT CONFIG" >&2
  exit 2
fi

readonly RUN_ID="$1"
readonly STUDENT_RUN="$2"
readonly CODE_COMMIT="$3"
readonly CONFIG="$4"
readonly CODE_ROOT="/home/oliver/Code/airhockey-memory-distillation"
readonly EXPERIMENT_ROOT="/home/oliver/experiments/airhockey-memory-distillation"
readonly RUN_ROOT="${EXPERIMENT_ROOT}/${RUN_ID}"
readonly DATASET_RUN="teacher-v3-structured-n64-k2-full-deterministic-2026-08-12-v1"
readonly DATASET_ROOT="${EXPERIMENT_ROOT}/teacher-datasets/${DATASET_RUN}"
readonly STUDENT_ROOT="${EXPERIMENT_ROOT}/students/${STUDENT_RUN}"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"

cd "${CODE_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test -z "$(git status --porcelain)"
test ! -e "${RUN_ROOT}"
test ! -e "${STUDENT_ROOT}"
mkdir -p "${RUN_ROOT}"

python3 - "${DATASET_ROOT}/manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
manifest = json.loads(path.read_text())
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert manifest["status"] == "completed"
assert manifest["dataset_id"] == "teacher_v3_structured_n64_k2_full_deterministic_v1"
assert manifest["teacher_action_semantics"] == "deterministic_actor_mean_after_public_adapter_clip"
assert manifest["episode_count"] == 20000
assert digest == "923497e3622209c7060196ff8322f4d333626138048ad94577034c67f01c60cf"
PY

docker run --rm --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/train_structured_student.py \
    --config "${CONFIG}" \
    --dataset "/experiments/teacher-datasets/${DATASET_RUN}" \
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
  "${STUDENT_ROOT}/config.yaml" \
  "${STUDENT_ROOT}/metrics.jsonl" \
  "${STUDENT_ROOT}/checkpoint.npz" \
  "${STUDENT_ROOT}/result.json" \
  "${STUDENT_ROOT}/no_blackout_validation.json" \
  "${RUN_ROOT}/training.log" \
  "${RUN_ROOT}/evaluation.log" \
  "${RUN_ROOT}/gate.log" \
  "${RUN_ROOT}/no_blackout_gate.json" \
  > "${RUN_ROOT}/sha256sums.txt"

cat "${RUN_ROOT}/no_blackout_gate.json"
