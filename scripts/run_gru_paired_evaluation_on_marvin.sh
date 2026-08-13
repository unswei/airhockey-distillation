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
readonly STUDENT_RUN="gru-n64-full-pilot-seed-14303-2026-08-13-v1"
readonly GATE_RUN="phase3-gru-n64-full-pilot-no-blackout-2026-08-13-v1"
readonly GATE_SHA256="284d0051f52f71ca73a3346fb245f5674036175e23279a5bc48bd47a954cdd31"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly CONFIG="configs/student/gru_n64_full_pilot_paired_v1.yaml"
readonly GATE="${EXPERIMENT_ROOT}/${GATE_RUN}/no_blackout_gate.json"

cd "${CODE_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test -z "$(git status --porcelain)"
test ! -e "${RUN_ROOT}"
echo "${GATE_SHA256}  ${GATE}" | sha256sum --check --status
python3 -c 'import json, sys; assert json.load(open(sys.argv[1]))["decision"] == "GO"' "${GATE}"
mkdir -p "${RUN_ROOT}"

docker run --rm --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --volume "${CODE_ROOT}:/work:ro" \
  --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
  --workdir /work \
  "${IMAGE}" \
  python3 scripts/evaluate_gru_student.py \
    --config "${CONFIG}" \
    --checkpoint "/experiments/students/${STUDENT_RUN}/checkpoint.npz" \
    --output "/experiments/${RUN_ID}/paired_validation.json" \
    --code-commit "${CODE_COMMIT}" \
  2>&1 | tee "${RUN_ROOT}/evaluation.log"

sha256sum "${RUN_ROOT}/paired_validation.json" > "${RUN_ROOT}/sha256sums.txt"
