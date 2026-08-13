#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_HOST="marvin"
readonly EXPECTED_COMMIT="${PRINCIPAL_SWEEP_COMMIT:-}"
readonly REPO_HOST="/home/oliver/Code/airhockey-memory-distillation"
readonly EXPERIMENT_HOST="/home/oliver/experiments/airhockey-memory-distillation"
readonly RUN_ID="${PRINCIPAL_SWEEP_RUN_ID:-principal-sweep-v1-2026-08-13-v1}"
readonly RUN_HOST="${EXPERIMENT_HOST}/${RUN_ID}"
readonly RUN="/experiments/${RUN_ID}"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly IMAGE_DIGEST="sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f"
readonly PROTOCOL="configs/experiments/principal_sweep_execution_v1.yaml"
readonly BASE="/experiments/teacher-datasets/teacher-v3-structured-n64-k2-full-deterministic-2026-08-12-v1"
readonly TEACHER="/experiments/frozen-teachers/dreamerv3-teacher-v3-step-700000"
readonly TEACHER_CONFIG="configs/teacher/dreamerv3_v3.yaml"
readonly FAMILIES=(feed_forward finite_stack_10 structured_k0 structured_k1 structured_k2 structured_k4 gru_n64)
readonly SEEDS=(14303 14304 14305 14306 14307)

if [[ "$(hostname -s)" != "${EXPECTED_HOST}" ]]; then
  echo "principal sweep must run on ${EXPECTED_HOST}" >&2
  exit 2
fi
if [[ ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "PRINCIPAL_SWEEP_COMMIT must be the frozen 40-character execution commit" >&2
  exit 2
fi
cd "${REPO_HOST}"
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]] || [[ -n "$(git status --porcelain)" ]]; then
  echo "Marvin checkout is not clean at the frozen execution commit" >&2
  exit 2
fi
if [[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" != "${IMAGE_DIGEST}" ]]; then
  echo "pinned container digest mismatch" >&2
  exit 2
fi

mkdir -p "${RUN_HOST}/orchestrator/logs"

write_status() {
  local stage="$1"
  local detail="$2"
  python3 - "${RUN_HOST}/orchestrator/status.json" "${stage}" "${detail}" "${EXPECTED_COMMIT}" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "status": "running",
    "stage": sys.argv[2],
    "detail": sys.argv[3],
    "code_commit": sys.argv[4],
    "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "pid": os.getppid(),
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

container() {
  docker run --rm --ipc host \
    -e PYTHONPATH=/workspace:/workspace/src:/src/2025-challenge \
    -v "${REPO_HOST}:/workspace:ro" \
    -v "${EXPERIMENT_HOST}:/experiments" \
    -w /workspace "${IMAGE}" "$@"
}

gpu_container() {
  docker run --rm --gpus all --ipc host \
    -e PYTHONPATH=/workspace:/workspace/src:/src/2025-challenge \
    -v "${REPO_HOST}:/workspace:ro" \
    -v "${EXPERIMENT_HOST}:/experiments" \
    -w /workspace "${IMAGE}" "$@"
}

latency_container() {
  docker run --rm --ipc host --hostname marvin --cpuset-cpus 0 \
    -e PYTHONPATH=/workspace:/workspace/src:/src/2025-challenge \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "${REPO_HOST}:/workspace:ro" \
    -v "${EXPERIMENT_HOST}:/experiments" \
    -w /workspace "${IMAGE}" "$@"
}

freeze_path() {
  local relative="$1"
  docker run --rm -v "${EXPERIMENT_HOST}:/experiments" "${IMAGE}" \
    chmod -R a-w "/experiments/${relative}"
}

prepare_training_output() {
  local host_path="$1"
  if [[ -d "${host_path}" && ! -f "${host_path}/result.json" ]]; then
    local preserved="${host_path}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${host_path}" "${preserved}"
  fi
}

write_status "preflight" "rechecking frozen inputs and release controls"
container python - "${EXPECTED_COMMIT}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
from airhockey_distill.principal_release import evaluate_test_release
from airhockey_distill.principal_sweep import load_principal_protocol, sha256_file

commit = sys.argv[1]
protocol_path = pathlib.Path("configs/experiments/principal_sweep_execution_v1.yaml")
protocol = load_principal_protocol(protocol_path)
base = pathlib.Path("/experiments/teacher-datasets/teacher-v3-structured-n64-k2-full-deterministic-2026-08-12-v1")
teacher = pathlib.Path("/experiments/frozen-teachers/dreamerv3-teacher-v3-step-700000")
assert sha256_file(base / "manifest.json") == protocol["base_teacher_dataset"]["manifest_sha256"]
manifest = json.loads((base / "manifest.json").read_text())
for entry in manifest["shards"]:
    assert sha256_file(base / entry["file"]) == entry["sha256"]
assert sha256_file(teacher / "agent.pkl") == protocol["teacher"]["actor_sha256"]
closed = evaluate_test_release(
    protocol_path,
    "configs/experiments/principal_test_release_evidence_template_v1.json",
)
assert closed["decision"] == "NO_GO"
assert closed["principal_test_schedule_sha256"] is None
assert commit == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("frozen preflight passed; principal test remains closed")
PY

for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    output_host="${RUN_HOST}/collectors/${family}/${seed}"
    output="${RUN}/collectors/${family}/${seed}"
    if [[ -f "${output_host}/result.json" && -f "${output_host}/checkpoint.npz" ]]; then
      continue
    fi
    write_status "collector_training" "${family} seed ${seed}"
    prepare_training_output "${output_host}"
    container python scripts/train_principal_student.py \
      --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
      --stage collector --dataset "${BASE}" --output "${output}" \
      --code-commit "${EXPECTED_COMMIT}" \
      >"${RUN_HOST}/orchestrator/logs/collector-${family}-${seed}.log" 2>&1
    freeze_path "${RUN_ID}/collectors/${family}/${seed}"
  done
done

for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    output_host="${RUN_HOST}/shadow/${family}/${seed}"
    output="${RUN}/shadow/${family}/${seed}"
    if [[ -f "${output_host}/manifest.json" ]]; then
      continue
    fi
    write_status "shadow_collection" "${family} seed ${seed}"
    gpu_container python scripts/collect_principal_shadow_dataset.py \
      --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
      --teacher-config "${TEACHER_CONFIG}" --teacher-checkpoint "${TEACHER}" \
      --student-checkpoint "${RUN}/collectors/${family}/${seed}/checkpoint.npz" \
      --output "${output}" --code-commit "${EXPECTED_COMMIT}" \
      >"${RUN_HOST}/orchestrator/logs/shadow-${family}-${seed}.log" 2>&1
    freeze_path "${RUN_ID}/shadow/${family}/${seed}"
  done
  audit_host="${RUN_HOST}/audits/${family}.json"
  if [[ ! -f "${audit_host}" ]]; then
    write_status "shadow_audit" "${family}"
    audit_arguments=()
    for seed in "${SEEDS[@]}"; do
      audit_arguments+=(--shadow-partition "${RUN}/shadow/${family}/${seed}")
    done
    container python scripts/audit_principal_shadow_budget.py \
      --protocol "${PROTOCOL}" --family "${family}" \
      "${audit_arguments[@]}" --code-commit "${EXPECTED_COMMIT}" \
      --output "${RUN}/audits/${family}.json" \
      >"${RUN_HOST}/orchestrator/logs/audit-${family}.log" 2>&1
    freeze_path "${RUN_ID}/audits/${family}.json"
  fi
  aggregate_host="${RUN_HOST}/datasets/${family}"
  if [[ ! -f "${aggregate_host}/manifest.json" ]]; then
    write_status "family_aggregation" "${family}"
    aggregate_arguments=()
    for seed in "${SEEDS[@]}"; do
      aggregate_arguments+=(--shadow-partition "${RUN}/shadow/${family}/${seed}")
    done
    container python scripts/aggregate_principal_shadow_dataset.py \
      --protocol "${PROTOCOL}" --family "${family}" --base-dataset "${BASE}" \
      "${aggregate_arguments[@]}" --output "${RUN}/datasets/${family}" \
      --code-commit "${EXPECTED_COMMIT}" \
      >"${RUN_HOST}/orchestrator/logs/aggregate-${family}.log" 2>&1
    freeze_path "${RUN_ID}/datasets/${family}"
  fi
done

for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    output_host="${RUN_HOST}/final/${family}/${seed}"
    output="${RUN}/final/${family}/${seed}"
    if [[ -f "${output_host}/result.json" && -f "${output_host}/checkpoint.npz" ]]; then
      continue
    fi
    write_status "final_training" "${family} seed ${seed}"
    prepare_training_output "${output_host}"
    container python scripts/train_principal_student.py \
      --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
      --stage final --dataset "${RUN}/datasets/${family}" --output "${output}" \
      --code-commit "${EXPECTED_COMMIT}" \
      >"${RUN_HOST}/orchestrator/logs/final-${family}-${seed}.log" 2>&1
    freeze_path "${RUN_ID}/final/${family}/${seed}"
  done
done

for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    checkpoint="${RUN}/final/${family}/${seed}/checkpoint.npz"
    if [[ ! -f "${RUN_HOST}/validation/${family}-${seed}.json" ]]; then
      write_status "paired_validation" "${family} seed ${seed}"
      container python scripts/evaluate_principal_student.py \
        --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
        --checkpoint "${checkpoint}" --split validation \
        --output "${RUN}/validation/${family}-${seed}.json" \
        --code-commit "${EXPECTED_COMMIT}" \
        >"${RUN_HOST}/orchestrator/logs/validation-${family}-${seed}.log" 2>&1
      freeze_path "${RUN_ID}/validation/${family}-${seed}.json"
    fi
    if [[ ! -f "${RUN_HOST}/accounting/${family}-${seed}.json" ]]; then
      write_status "policy_accounting" "${family} seed ${seed}"
      container python scripts/account_principal_policy.py \
        --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
        --checkpoint "${checkpoint}" \
        --output "${RUN}/accounting/${family}-${seed}.json" \
        --code-commit "${EXPECTED_COMMIT}" \
        >"${RUN_HOST}/orchestrator/logs/accounting-${family}-${seed}.log" 2>&1
      freeze_path "${RUN_ID}/accounting/${family}-${seed}.json"
    fi
    if [[ ! -f "${RUN_HOST}/latency/${family}-${seed}.json" ]]; then
      write_status "cpu_latency" "${family} seed ${seed}"
      latency_container python scripts/benchmark_principal_cpu_latency.py \
        --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
        --checkpoint "${checkpoint}" --container-digest "${IMAGE_DIGEST}" \
        --output "${RUN}/latency/${family}-${seed}.json" \
        --code-commit "${EXPECTED_COMMIT}" \
        >"${RUN_HOST}/orchestrator/logs/latency-${family}-${seed}.log" 2>&1
      freeze_path "${RUN_ID}/latency/${family}-${seed}.json"
    fi
  done
done

if [[ ! -f "${RUN_HOST}/release/evidence.json" ]]; then
  write_status "evidence_freeze" "freezing every pre-test artefact"
  evidence_arguments=()
  for family in "${FAMILIES[@]}"; do
    evidence_arguments+=(--family-dataset-manifest "${RUN}/datasets/${family}/manifest.json")
    evidence_arguments+=(--shadow-budget-audit "${RUN}/audits/${family}.json")
    for seed in "${SEEDS[@]}"; do
      evidence_arguments+=(--collector-checkpoint "${RUN}/collectors/${family}/${seed}/checkpoint.npz")
      evidence_arguments+=(--final-checkpoint "${RUN}/final/${family}/${seed}/checkpoint.npz")
      evidence_arguments+=(--validation-result "${RUN}/validation/${family}-${seed}.json")
      evidence_arguments+=(--accounting-result "${RUN}/accounting/${family}-${seed}.json")
      evidence_arguments+=(--cpu-latency-result "${RUN}/latency/${family}-${seed}.json")
    done
  done
  container python scripts/freeze_principal_release_evidence.py \
    --protocol "${PROTOCOL}" --analysis-plan docs/principal_sweep_v1.md \
    --aggregation-code-commit "${EXPECTED_COMMIT}" \
    --declare-principal-test-uninspected "${evidence_arguments[@]}" \
    --output "${RUN}/release/evidence.json" \
    >"${RUN_HOST}/orchestrator/logs/freeze-evidence.log" 2>&1
  freeze_path "${RUN_ID}/release/evidence.json"
fi

if [[ ! -f "${RUN_HOST}/release/gate.json" ]]; then
  write_status "release_gate" "reopening and validating all frozen pre-test evidence"
  container python scripts/gate_principal_test_release.py \
    --protocol "${PROTOCOL}" --evidence-manifest "${RUN}/release/evidence.json" \
    --output "${RUN}/release/gate.json" \
    >"${RUN_HOST}/orchestrator/logs/release-gate.log" 2>&1
  freeze_path "${RUN_ID}/release/gate.json"
fi
if [[ "$(jq -r .decision "${RUN_HOST}/release/gate.json")" != "GO" ]]; then
  echo "principal test remains closed because the release gate did not pass" >&2
  exit 2
fi

write_status "principal_test" "single released opening for all students and teacher"
gpu_container python scripts/evaluate_principal_test_once.py \
  --protocol "${PROTOCOL}" --release-report "${RUN}/release/gate.json" \
  --final-root "${RUN}/final" --teacher-config "${TEACHER_CONFIG}" \
  --teacher-checkpoint "${TEACHER}" --output "${RUN}/test" \
  --code-commit "${EXPECTED_COMMIT}" \
  >"${RUN_HOST}/orchestrator/logs/principal-test.log" 2>&1
freeze_path "${RUN_ID}/test"

if [[ ! -f "${RUN_HOST}/analysis/manifest.json" ]]; then
  write_status "analysis" "statistics, figure and efficiency table"
  container python scripts/analyse_principal_sweep.py \
    --protocol "${PROTOCOL}" --test-root "${RUN}/test" \
    --accounting-root "${RUN}/accounting" --latency-root "${RUN}/latency" \
    --output "${RUN}/analysis" --code-commit "${EXPECTED_COMMIT}" \
    >"${RUN_HOST}/orchestrator/logs/analysis.log" 2>&1
  freeze_path "${RUN_ID}/analysis"
fi

python3 - "${RUN_HOST}/orchestrator/status.json" "${EXPECTED_COMMIT}" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "status": "completed",
    "stage": "completed",
    "detail": "principal sweep, released test and final analysis completed",
    "code_commit": sys.argv[2],
    "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "pid": os.getppid(),
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
