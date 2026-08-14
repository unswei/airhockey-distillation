#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_HOST="marvin"
readonly MEASUREMENT_CODE_COMMIT="099602c8b12fe804eb8a4da684b7afd2a667572a"
readonly FINAL_TRAINING_ORCHESTRATION_COMMIT="f3e2b8cbf123d2cb94486dfd4896adf8f893a115"
readonly ORCHESTRATION_COMMIT="${V3_MEASUREMENT_ORCHESTRATION_COMMIT:-}"
readonly ORCHESTRATION_REPO="/home/oliver/Code/airhockey-memory-distillation"
readonly MEASUREMENT_REPO="/home/oliver/Code/airhockey-memory-distillation-v3-099602c"
readonly EXPERIMENT_HOST="/home/oliver/experiments/airhockey-memory-distillation"
readonly SOURCE_RUN_ID="principal-sweep-v1-2026-08-13-v2"
readonly SOURCE_RUN_HOST="${EXPERIMENT_HOST}/${SOURCE_RUN_ID}"
readonly RUN_ID="${PRINCIPAL_V3_RUN_ID:-principal-sweep-v1-2026-08-14-v3}"
readonly RUN_HOST="${EXPERIMENT_HOST}/${RUN_ID}"
readonly RUN="/experiments/${RUN_ID}"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly IMAGE_DIGEST="sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f"
readonly PROTOCOL="configs/experiments/principal_sweep_execution_v1.yaml"
readonly PROTOCOL_SHA256="1f8ececf4492f68517938ccd2fd1e05bb459e0fdd7275851536f71d782bed448"
readonly SOURCE_SEAL_SHA256="94ea1e57d7959f1ef368a681945d2ff29e5ed60769a1513e6af212f1c4933be3"
readonly VALIDATION_WORKERS=24
readonly FAMILIES=(feed_forward finite_stack_10 structured_k0 structured_k1 structured_k2 structured_k4 gru_n64)
readonly SEEDS=(14303 14304 14305 14306 14307)
readonly CONTROL_HOST="${RUN_HOST}/orchestrator/measurements"

if [[ "$(hostname -s)" != "${EXPECTED_HOST}" ]]; then
  echo "V3 measurements must run on ${EXPECTED_HOST}" >&2
  exit 2
fi
if [[ ! "${ORCHESTRATION_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "V3_MEASUREMENT_ORCHESTRATION_COMMIT must be a frozen Git commit" >&2
  exit 2
fi
if [[ "$(git -C "${ORCHESTRATION_REPO}" rev-parse HEAD)" != "${ORCHESTRATION_COMMIT}" ]] ||
  [[ -n "$(git -C "${ORCHESTRATION_REPO}" status --porcelain)" ]]; then
  echo "measurement orchestration checkout is not clean at its declared commit" >&2
  exit 2
fi
if [[ "$(git -C "${MEASUREMENT_REPO}" rev-parse HEAD)" != "${MEASUREMENT_CODE_COMMIT}" ]] ||
  [[ -n "$(git -C "${MEASUREMENT_REPO}" status --porcelain)" ]]; then
  echo "measurement worktree is not clean at the correction commit" >&2
  exit 2
fi
if [[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" != "${IMAGE_DIGEST}" ]]; then
  echo "pinned container digest mismatch" >&2
  exit 2
fi

mkdir -p "${CONTROL_HOST}/logs" "${CONTROL_HOST}/jobs" \
  "${CONTROL_HOST}/latency_jobs" "${CONTROL_HOST}/schedules"
exec 9>"${CONTROL_HOST}/runner.lock"
if ! flock -n 9; then
  echo "another V3 measurement runner holds the lock" >&2
  exit 2
fi
printf '%s\n' "$$" >"${CONTROL_HOST}/runner.pid"

write_status() {
  local status="$1"
  local stage="$2"
  local detail="$3"
  python3 - "${CONTROL_HOST}/status.json" "${status}" "${stage}" "${detail}" \
    "${MEASUREMENT_CODE_COMMIT}" "${ORCHESTRATION_COMMIT}" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "detail": sys.argv[4],
    "measurement_code_commit": sys.argv[5],
    "orchestration_code_commit": sys.argv[6],
    "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "pid": os.getppid(),
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

write_job_status() {
  local directory="$1"
  local phase="$2"
  local family="$3"
  local seed="$4"
  local status="$5"
  local detail="$6"
  local exit_code="$7"
  python3 - "${directory}/${family}-${seed}.json" "${phase}" "${family}" \
    "${seed}" "${status}" "${detail}" "${exit_code}" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "phase": sys.argv[2],
    "family_id": sys.argv[3],
    "training_seed": int(sys.argv[4]),
    "status": sys.argv[5],
    "detail": sys.argv[6],
    "exit_code": int(sys.argv[7]),
    "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "worker_pid": os.getppid(),
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

freeze_v3_path() {
  local relative="$1"
  docker run --rm -v "${RUN_HOST}:/run-output" "${IMAGE}" \
    chmod -R a-w "/run-output/${relative}"
}

freeze_if_present() {
  local host_path="$1"
  local relative="$2"
  [[ ! -e "${host_path}" ]] || freeze_v3_path "${relative}"
}

measurement_container() {
  local cpu="$1"
  shift
  docker run --rm --ipc host --hostname marvin --cpuset-cpus "${cpu}" \
    -e PYTHONPATH=/workspace:/workspace/src:/src/2025-challenge \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "${MEASUREMENT_REPO}:/workspace:ro" \
    -v "${EXPERIMENT_HOST}:/experiments:ro" \
    -v "${RUN_HOST}:${RUN}" \
    -w /workspace "${IMAGE}" "$@"
}

control_container() {
  docker run --rm --ipc host --cpuset-cpus 1 \
    -e PYTHONPATH=/control:/control/src:/src/2025-challenge \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "${ORCHESTRATION_REPO}:/control:ro" \
    -v "${EXPERIMENT_HOST}:/experiments:ro" \
    -v "${RUN_HOST}:${RUN}" \
    -w /control "${IMAGE}" "$@"
}

latency_container() {
  docker run --rm --ipc host --hostname marvin --cpuset-cpus 0 \
    -e PYTHONPATH=/workspace:/workspace/src:/src/2025-challenge \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "${MEASUREMENT_REPO}:/workspace:ro" \
    -v "${EXPERIMENT_HOST}:/experiments:ro" \
    -v "${RUN_HOST}:${RUN}" \
    -w /workspace "${IMAGE}" "$@"
}

validation_job_complete() {
  local family="$1"
  local seed="$2"
  local status="${CONTROL_HOST}/jobs/${family}-${seed}.json"
  local validation="${RUN_HOST}/validation/${family}-${seed}.json"
  local accounting="${RUN_HOST}/accounting/${family}-${seed}.json"
  [[ -f "${status}" && -f "${validation}" && -f "${accounting}" ]] || return 1
  [[ "$(jq -r .status "${status}")" == "completed" ]] || return 1
  [[ -z "$(find "${validation}" "${accounting}" -writable -print -quit)" ]]
}

run_validation_job() {
  local cpu="$1"
  local family="$2"
  local seed="$3"
  local checkpoint_host="${RUN_HOST}/final/${family}/${seed}/checkpoint.npz"
  local checkpoint="${RUN}/final/${family}/${seed}/checkpoint.npz"
  local validation_host="${RUN_HOST}/validation/${family}-${seed}.json"
  local validation="${RUN}/validation/${family}-${seed}.json"
  local accounting_host="${RUN_HOST}/accounting/${family}-${seed}.json"
  local accounting="${RUN}/accounting/${family}-${seed}.json"
  local status="${CONTROL_HOST}/jobs/${family}-${seed}.json"
  local log="${CONTROL_HOST}/logs/validation-${family}-${seed}.log"
  local checkpoint_hash
  checkpoint_hash="$(sha256sum "${checkpoint_host}" | cut -d' ' -f1)"

  if validation_job_complete "${family}" "${seed}"; then
    return 0
  fi
  if [[ -e "${validation_host}" || -e "${accounting_host}" || -e "${status}" ]]; then
    echo "refusing to overwrite incomplete validation evidence for ${family}/${seed}" >&2
    return 1
  fi
  write_job_status "${CONTROL_HOST}/jobs" "paired_validation_and_accounting" \
    "${family}" "${seed}" "running" "logical CPU ${cpu}" 0

  set +e
  measurement_container "${cpu}" python scripts/evaluate_principal_student.py \
    --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
    --checkpoint "${checkpoint}" --split validation --output "${validation}" \
    --code-commit "${MEASUREMENT_CODE_COMMIT}" >"${log}" 2>&1
  local validation_exit=$?
  if [[ ${validation_exit} -eq 0 ]]; then
    measurement_container "${cpu}" python scripts/account_principal_policy.py \
      --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
      --checkpoint "${checkpoint}" --output "${accounting}" \
      --code-commit "${MEASUREMENT_CODE_COMMIT}" >>"${log}" 2>&1
  fi
  local measurement_exit=$?
  set -e
  if [[ ${validation_exit} -ne 0 || ${measurement_exit} -ne 0 ]] ||
    [[ "$(jq -r .status "${validation_host}" 2>/dev/null)" != "completed" ]] ||
    [[ "$(jq -r .family_id "${validation_host}" 2>/dev/null)" != "${family}" ]] ||
    [[ "$(jq -r .training_seed "${validation_host}" 2>/dev/null)" != "${seed}" ]] ||
    [[ "$(jq -r .checkpoint_sha256 "${validation_host}" 2>/dev/null)" != "${checkpoint_hash}" ]] ||
    [[ "$(jq -r .code_commit "${validation_host}" 2>/dev/null)" != "${MEASUREMENT_CODE_COMMIT}" ]] ||
    [[ "$(jq -r .evaluation_split "${validation_host}" 2>/dev/null)" != "validation" ]] ||
    [[ "$(jq -r '.episodes | length' "${validation_host}" 2>/dev/null)" != "1125" ]] ||
    [[ "$(jq -r .decision "${accounting_host}" 2>/dev/null)" != "GO" ]] ||
    [[ "$(jq -r .checkpoint_sha256 "${accounting_host}" 2>/dev/null)" != "${checkpoint_hash}" ]]; then
    freeze_if_present "${validation_host}" "validation/${family}-${seed}.json"
    freeze_if_present "${accounting_host}" "accounting/${family}-${seed}.json"
    chmod a-w "${log}"
    write_job_status "${CONTROL_HOST}/jobs" "paired_validation_and_accounting" \
      "${family}" "${seed}" "failed" "fail-closed measurement check failed" 2
    chmod a-w "${status}"
    return 1
  fi

  freeze_v3_path "validation/${family}-${seed}.json"
  freeze_v3_path "accounting/${family}-${seed}.json"
  chmod a-w "${log}"
  write_job_status "${CONTROL_HOST}/jobs" "paired_validation_and_accounting" \
    "${family}" "${seed}" "completed" \
    "1,125 paired episodes and checkpoint accounting frozen" 0
  chmod a-w "${status}"
}

latency_job_complete() {
  local family="$1"
  local seed="$2"
  local status="${CONTROL_HOST}/latency_jobs/${family}-${seed}.json"
  local result="${RUN_HOST}/latency/${family}-${seed}.json"
  [[ -f "${status}" && -f "${result}" ]] || return 1
  [[ "$(jq -r .status "${status}")" == "completed" ]] || return 1
  [[ ! -w "${result}" ]]
}

run_latency_job() {
  local family="$1"
  local seed="$2"
  local checkpoint_host="${RUN_HOST}/final/${family}/${seed}/checkpoint.npz"
  local checkpoint="${RUN}/final/${family}/${seed}/checkpoint.npz"
  local output_host="${RUN_HOST}/latency/${family}-${seed}.json"
  local output="${RUN}/latency/${family}-${seed}.json"
  local status="${CONTROL_HOST}/latency_jobs/${family}-${seed}.json"
  local log="${CONTROL_HOST}/logs/latency-${family}-${seed}.log"
  local checkpoint_hash
  checkpoint_hash="$(sha256sum "${checkpoint_host}" | cut -d' ' -f1)"

  if latency_job_complete "${family}" "${seed}"; then
    return 0
  fi
  if [[ -e "${output_host}" || -e "${status}" ]]; then
    echo "refusing to overwrite incomplete latency evidence for ${family}/${seed}" >&2
    return 1
  fi
  write_job_status "${CONTROL_HOST}/latency_jobs" "cpu_latency" \
    "${family}" "${seed}" "running" "isolated logical CPU 0" 0
  set +e
  latency_container python scripts/benchmark_principal_cpu_latency.py \
    --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
    --checkpoint "${checkpoint}" --container-digest "${IMAGE_DIGEST}" \
    --output "${output}" --code-commit "${MEASUREMENT_CODE_COMMIT}" \
    >"${log}" 2>&1
  local latency_exit=$?
  set -e
  if [[ ${latency_exit} -ne 0 ]] ||
    [[ "$(jq -r .decision "${output_host}" 2>/dev/null)" != "GO" ]] ||
    [[ "$(jq -r .checkpoint_sha256 "${output_host}" 2>/dev/null)" != "${checkpoint_hash}" ]] ||
    [[ "$(jq -r .code_commit "${output_host}" 2>/dev/null)" != "${MEASUREMENT_CODE_COMMIT}" ]]; then
    freeze_if_present "${output_host}" "latency/${family}-${seed}.json"
    chmod a-w "${log}"
    write_job_status "${CONTROL_HOST}/latency_jobs" "cpu_latency" \
      "${family}" "${seed}" "failed" "fail-closed latency check failed" 2
    chmod a-w "${status}"
    return 1
  fi
  freeze_v3_path "latency/${family}-${seed}.json"
  chmod a-w "${log}"
  write_job_status "${CONTROL_HOST}/latency_jobs" "cpu_latency" \
    "${family}" "${seed}" "completed" "predeclared benchmark frozen" 0
  chmod a-w "${status}"
}

write_status "running" "preflight" "checking V3 finals, V2 seal and closed test gate"
if [[ "$(sha256sum "${SOURCE_RUN_HOST}/orchestrator/seal_manifest.sha256" | cut -d' ' -f1)" != \
  "${SOURCE_SEAL_SHA256}" ]] ||
  [[ -n "$(find "${SOURCE_RUN_HOST}" -writable -print -quit)" ]] ||
  find "${SOURCE_RUN_HOST}" -maxdepth 1 -name 'principal_test*' -print -quit | grep -q . ||
  [[ -e "${RUN_HOST}/test" ]] ||
  find "${RUN_HOST}" -maxdepth 1 -name 'principal_test*' -print -quit | grep -q .; then
  write_status "failed" "preflight" "sealed source or unopened-test invariant failed"
  exit 2
fi
if [[ "$(sha256sum "${MEASUREMENT_REPO}/${PROTOCOL}" | cut -d' ' -f1)" != \
  "${PROTOCOL_SHA256}" ]]; then
  write_status "failed" "preflight" "protocol hash mismatch"
  exit 2
fi
if [[ ! -f "${CONTROL_HOST}/logs/v2-preflight-seal-check.log" ]]; then
  (cd "${SOURCE_RUN_HOST}" && sha256sum -c orchestrator/seal_manifest.sha256) \
    >"${CONTROL_HOST}/logs/v2-preflight-seal-check.log"
  chmod a-w "${CONTROL_HOST}/logs/v2-preflight-seal-check.log"
fi
python3 - "${RUN_HOST}" "${MEASUREMENT_CODE_COMMIT}" \
  "${FINAL_TRAINING_ORCHESTRATION_COMMIT}" <<'PY'
import hashlib, json, pathlib, stat, sys
root = pathlib.Path(sys.argv[1])
manifest_path = root / "orchestrator" / "final_training_manifest.json"
manifest = json.loads(manifest_path.read_text())
if manifest.get("status") != "completed" or manifest.get("principal_test_opened") is not False:
    raise RuntimeError("V3 final-training manifest is incomplete")
if manifest.get("training_code_commit") != sys.argv[2] or manifest.get("orchestration_code_commit") != sys.argv[3]:
    raise RuntimeError("V3 final-training commit provenance differs")
models = manifest.get("models", [])
if len(models) != 35:
    raise RuntimeError("V3 does not contain exactly 35 final models")
for record in models:
    final = root / "final" / record["family_id"] / str(record["training_seed"])
    checkpoint = final / "checkpoint.npz"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint changed: {record['family_id']}/{record['training_seed']}")
    if any(stat.S_IMODE(path.stat().st_mode) & 0o222 for path in final.rglob("*") if path.is_file()):
        raise RuntimeError(f"writable final model: {record['family_id']}/{record['training_seed']}")
PY

if [[ ! -f "${CONTROL_HOST}/schedules/validation.tsv" ]]; then
  index=0
  for family in "${FAMILIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      lane=$((index % VALIDATION_WORKERS))
      printf '%s\t%s\t%s\n' "${lane}" "${family}" "${seed}" \
        >>"${CONTROL_HOST}/schedules/validation.tsv"
      index=$((index + 1))
    done
  done
  chmod -R a-w "${CONTROL_HOST}/schedules"
fi

run_validation_lane() {
  local lane="$1"
  while IFS=$'\t' read -r scheduled_lane family seed; do
    [[ "${scheduled_lane}" != "${lane}" ]] || \
      run_validation_job "${lane}" "${family}" "${seed}" || return 1
  done <"${CONTROL_HOST}/schedules/validation.tsv"
}

write_status "running" "paired_validation" \
  "35 models on the identical 1,125-episode schedule across 24 one-core workers"
worker_pids=()
for ((lane = 0; lane < VALIDATION_WORKERS; lane++)); do
  run_validation_lane "${lane}" &
  worker_pids+=("$!")
done
validation_failures=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "${worker_pid}"; then
    validation_failures=$((validation_failures + 1))
  fi
done
if [[ ${validation_failures} -ne 0 ]]; then
  write_status "failed" "paired_validation" \
    "${validation_failures} validation worker lanes failed; no automatic retry"
  exit 1
fi

write_status "running" "cpu_latency" \
  "35 serial predeclared benchmarks on isolated logical CPU 0"
if [[ -n "$(docker ps -q)" ]]; then
  write_status "failed" "cpu_latency" "another container prevents isolated benchmarking"
  exit 2
fi
for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if ! run_latency_job "${family}" "${seed}"; then
      write_status "failed" "cpu_latency" \
        "latency measurement failed for ${family}/${seed}; no automatic retry"
      exit 1
    fi
  done
done

write_status "running" "measurement_freeze" \
  "independently checking and hashing all pre-test measurements"
control_container python scripts/verify_principal_v3_measurements.py \
  --protocol "${PROTOCOL}" --run-root "${RUN}" \
  --final-training-manifest "${RUN}/orchestrator/final_training_manifest.json" \
  --measurement-code-commit "${MEASUREMENT_CODE_COMMIT}" \
  --orchestration-code-commit "${ORCHESTRATION_COMMIT}" \
  --source-v2-seal-sha256 "${SOURCE_SEAL_SHA256}" \
  --output "${RUN}/orchestrator/measurements/manifest.json" \
  >"${CONTROL_HOST}/logs/measurement-freeze.log" 2>&1
freeze_v3_path "orchestrator/measurements/manifest.json"
chmod a-w "${CONTROL_HOST}/logs/measurement-freeze.log"

if [[ "$(jq -r .decision "${CONTROL_HOST}/manifest.json")" != "GO" ]] ||
  [[ "$(sha256sum "${SOURCE_RUN_HOST}/orchestrator/seal_manifest.sha256" | cut -d' ' -f1)" != \
    "${SOURCE_SEAL_SHA256}" ]] ||
  [[ -n "$(find "${SOURCE_RUN_HOST}" -writable -print -quit)" ]] ||
  [[ -e "${RUN_HOST}/test" ]] ||
  find "${RUN_HOST}" -maxdepth 1 -name 'principal_test*' -print -quit | grep -q .; then
  write_status "failed" "postflight" "measurement or unopened-test invariant failed"
  exit 2
fi
(cd "${SOURCE_RUN_HOST}" && sha256sum -c orchestrator/seal_manifest.sha256) \
  >"${CONTROL_HOST}/logs/v2-postflight-seal-check.log"
chmod a-w "${CONTROL_HOST}/logs/v2-postflight-seal-check.log"
write_status "completed" "measurements_complete" \
  "35 paired validations, accounting records and isolated latency benchmarks verified and frozen"
