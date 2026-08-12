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
readonly DATASET_RUN="teacher-v3-structured-n64-k2-full-deterministic-2026-08-12-v1"
readonly DATASET_ROOT="${EXPERIMENT_ROOT}/teacher-datasets/${DATASET_RUN}"
readonly DATASET_CODE_COMMIT="cdbde7d81aa6f5624cfbdc709ea368515634d39c"
readonly STUDENT_RUN="structured-n64-k2-full-seed-14303-2026-08-13-v2"
readonly STUDENT_ROOT="${EXPERIMENT_ROOT}/students/${STUDENT_RUN}"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly CONFIG="configs/student/structured_n64_k2_full_seed_14303_v2.yaml"

cd "${CODE_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test -z "$(git status --porcelain)"
mkdir -p "${RUN_ROOT}"

test -f "${DATASET_ROOT}/manifest.json"
python3 - "${DATASET_ROOT}/manifest.json" "${DATASET_CODE_COMMIT}" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["status"] == "completed"
assert manifest["code_commit"] == sys.argv[2]
assert manifest["dataset_id"] == (
    "teacher_v3_structured_n64_k2_full_deterministic_v1"
)
assert manifest["teacher_action_semantics"] == (
    "deterministic_actor_mean_after_public_adapter_clip"
)
assert manifest["deterministic_inference"] is True
assert manifest["episode_count"] == 20000
PY
printf 'reused frozen dataset %s collected at commit %s\n' \
  "${DATASET_RUN}" "${DATASET_CODE_COMMIT}" > "${RUN_ROOT}/dataset_binding.log"

if [[ ! -f "${STUDENT_ROOT}/result.json" ]]; then
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
    2>&1 | tee -a "${RUN_ROOT}/training.log"
else
  python3 - "${STUDENT_ROOT}/result.json" "${CODE_COMMIT}" <<'PY'
import json
import pathlib
import sys

result = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert result["status"] == "completed"
assert result["code_commit"] == sys.argv[2]
assert result["teacher_action_semantics"] == (
    "deterministic_actor_mean_after_public_adapter_clip"
)
assert result["training_seed"] == 14303
PY
fi

if [[ ! -f "${STUDENT_ROOT}/closed_loop_validation.json" ]]; then
  docker run --rm --ipc host \
    --env PYTHONPATH=/work/src:/src/2025-challenge \
    --volume "${CODE_ROOT}:/work:ro" \
    --volume "${EXPERIMENT_ROOT}:/experiments:rw" \
    --workdir /work \
    "${IMAGE}" \
    python3 scripts/evaluate_structured_student.py \
      --config "${CONFIG}" \
      --checkpoint "/experiments/students/${STUDENT_RUN}/checkpoint.npz" \
      --output "/experiments/students/${STUDENT_RUN}/closed_loop_validation.json" \
      --code-commit "${CODE_COMMIT}" \
    2>&1 | tee -a "${RUN_ROOT}/evaluation.log"
fi

python3 - "${RUN_ROOT}" "${DATASET_ROOT}" "${STUDENT_ROOT}" "${CODE_COMMIT}" <<'PY'
import hashlib
import json
import pathlib
import platform
import sys
from datetime import UTC, datetime

run_root, dataset_root, student_root = map(pathlib.Path, sys.argv[1:4])
paths = [
    dataset_root / "config.yaml",
    dataset_root / "manifest.json",
    *sorted((dataset_root / "shards").glob("*.npz")),
    student_root / "config.yaml",
    student_root / "metrics.jsonl",
    student_root / "checkpoint.npz",
    student_root / "result.json",
    student_root / "closed_loop_validation.json",
    run_root / "dataset_binding.log",
    run_root / "training.log",
    run_root / "evaluation.log",
]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "status": "completed",
    "created_at": datetime.now(UTC).isoformat(),
    "code_commit": sys.argv[4],
    "hostname": platform.node(),
    "files": [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ],
}
(run_root / "run_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
