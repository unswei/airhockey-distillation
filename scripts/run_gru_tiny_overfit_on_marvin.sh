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
readonly DATASET="teacher-v3-structured-n64-k2-tiny-deterministic-2026-08-12-v1"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly CONFIG="configs/student/gru_n64_tiny_overfit_v1.yaml"

cd "${CODE_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test -z "$(git status --porcelain)"
test ! -e "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

docker run --rm --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/train_gru_tiny_overfit.py \
    --config "${CONFIG}" \
    --dataset "/experiments/teacher-datasets/${DATASET}" \
    --output "/experiments/${RUN_ID}/student" \
    --code-commit "${CODE_COMMIT}" \
  2>&1 | tee "${RUN_ROOT}/training.log"

sha256sum \
  "${RUN_ROOT}/student/checkpoint.npz" \
  "${RUN_ROOT}/student/metrics.jsonl" \
  "${RUN_ROOT}/student/result.json" \
  > "${RUN_ROOT}/sha256sums.txt"

cat "${RUN_ROOT}/student/result.json"
