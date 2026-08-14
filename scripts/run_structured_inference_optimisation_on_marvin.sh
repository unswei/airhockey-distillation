#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_HOST="marvin"
readonly OPTIMISATION_CODE_COMMIT="6f54e18219e3b478d1b6e0c2bddf62375925cf7c"
readonly ORCHESTRATION_COMMIT="${V4_ORCHESTRATION_COMMIT:-}"
readonly ORCHESTRATION_REPO="/home/oliver/Code/airhockey-memory-distillation"
readonly OPTIMISATION_REPO="/home/oliver/Code/airhockey-memory-distillation-v4-6f54e18"
readonly EXPERIMENT_HOST="/home/oliver/experiments/airhockey-memory-distillation"
readonly SOURCE_V2_ID="principal-sweep-v1-2026-08-13-v2"
readonly SOURCE_V2_HOST="${EXPERIMENT_HOST}/${SOURCE_V2_ID}"
readonly SOURCE_V3_ID="principal-sweep-v1-2026-08-14-v3"
readonly SOURCE_V3_HOST="${EXPERIMENT_HOST}/${SOURCE_V3_ID}"
readonly RUN_ID="${PRINCIPAL_V4_RUN_ID:-principal-sweep-v1-2026-08-15-v4-attempt2}"
readonly RUN_HOST="${EXPERIMENT_HOST}/${RUN_ID}"
readonly RUN="/experiments/${RUN_ID}"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly IMAGE_DIGEST="sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f"
readonly PROTOCOL="configs/experiments/principal_sweep_execution_v1.yaml"
readonly PROTOCOL_SHA256="1f8ececf4492f68517938ccd2fd1e05bb459e0fdd7275851536f71d782bed448"
readonly SOURCE_V2_SEAL_SHA256="94ea1e57d7959f1ef368a681945d2ff29e5ed60769a1513e6af212f1c4933be3"
readonly SOURCE_V3_FINAL_MANIFEST_SHA256="74722c6add5234ed995d084ce21b24422880b16adaf3c4eb4bd937f9d08a4ada"
readonly SOURCE_V3_MEASUREMENT_MANIFEST_SHA256="d4e30d7864726099232f67692a689509b1e8bdf2bcdfb46c1fa38b1ee7da970d"
readonly STRUCTURED_FAMILIES=(structured_k0 structured_k1 structured_k2 structured_k4)
readonly FAMILIES=(feed_forward finite_stack_10 structured_k0 structured_k1 structured_k2 structured_k4 gru_n64)
readonly SEEDS=(14303 14304 14305 14306 14307)
readonly CONTROL_HOST="${RUN_HOST}/orchestrator"

if [[ "$(hostname -s)" != "${EXPECTED_HOST}" ]]; then
  echo "structured optimisation evidence must run on ${EXPECTED_HOST}" >&2
  exit 2
fi
if [[ ! "${ORCHESTRATION_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "V4_ORCHESTRATION_COMMIT must be a frozen Git commit" >&2
  exit 2
fi
if [[ -e "${RUN_HOST}" ]]; then
  echo "refusing to overwrite existing V4 evidence at ${RUN_HOST}" >&2
  exit 2
fi
if [[ "$(git -C "${ORCHESTRATION_REPO}" rev-parse HEAD)" != "${ORCHESTRATION_COMMIT}" ]] ||
  [[ -n "$(git -C "${ORCHESTRATION_REPO}" status --porcelain)" ]]; then
  echo "orchestration checkout is not clean at its declared commit" >&2
  exit 2
fi
if [[ "$(git -C "${OPTIMISATION_REPO}" rev-parse HEAD)" != "${OPTIMISATION_CODE_COMMIT}" ]] ||
  [[ -n "$(git -C "${OPTIMISATION_REPO}" status --porcelain)" ]]; then
  echo "optimisation worktree is not clean at the frozen implementation commit" >&2
  exit 2
fi
if [[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" != "${IMAGE_DIGEST}" ]]; then
  echo "pinned container digest mismatch" >&2
  exit 2
fi

mkdir -p "${CONTROL_HOST}/logs" "${CONTROL_HOST}/jobs" \
  "${CONTROL_HOST}/latency_jobs" "${RUN_HOST}/native" \
  "${RUN_HOST}/verification" "${RUN_HOST}/validation" "${RUN_HOST}/latency"
exec 9>"${CONTROL_HOST}/runner.lock"
if ! flock -n 9; then
  echo "another V4 runner holds the lock" >&2
  exit 2
fi
printf '%s\n' "$$" >"${CONTROL_HOST}/runner.pid"

write_status() {
  local status="$1"
  local stage="$2"
  local detail="$3"
  python3 - "${CONTROL_HOST}/status.json" "${status}" "${stage}" "${detail}" \
    "${OPTIMISATION_CODE_COMMIT}" "${ORCHESTRATION_COMMIT}" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "detail": sys.argv[4],
    "optimisation_code_commit": sys.argv[5],
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

freeze_path() {
  local relative="$1"
  docker run --rm -v "${RUN_HOST}:/run-output" "${IMAGE}" \
    chmod -R a-w "/run-output/${relative}"
}

completed=0
record_failed_exit() {
  local exit_code=$?
  if [[ ${exit_code} -ne 0 && ${completed} -eq 0 && -w "${CONTROL_HOST}" ]]; then
    write_status "failed" "runner_exit" \
      "runner exited ${exit_code}; preserve this attempt and do not auto-retry"
  fi
}
trap record_failed_exit EXIT

measurement_container() {
  local cpu="$1"
  shift
  docker run --rm --ipc host --hostname marvin --cpuset-cpus "${cpu}" \
    -e AIRHOCKEY_STRUCTURED_PAIRWISE_BACKEND=native \
    -e PYTHONPATH=/native:/workspace:/workspace/src:/src/2025-challenge \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "${OPTIMISATION_REPO}:/workspace:ro" \
    -v "${RUN_HOST}/native:/native:ro" \
    -v "${EXPERIMENT_HOST}:/experiments:ro" \
    -v "${RUN_HOST}:${RUN}:rw" \
    -w /workspace "${IMAGE}" "$@"
}

control_container() {
  docker run --rm --ipc host --cpuset-cpus 1 \
    -e PYTHONPATH=/control:/control/src:/src/2025-challenge \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "${ORCHESTRATION_REPO}:/control:ro" \
    -v "${EXPERIMENT_HOST}:/experiments:ro" \
    -v "${RUN_HOST}:${RUN}:rw" \
    -w /control "${IMAGE}" "$@"
}

closed_test_gate() {
  local root="$1"
  [[ ! -e "${root}/test" ]] &&
    ! find "${root}" -maxdepth 1 -name 'principal_test*' -print -quit | grep -q .
}

write_status "running" "preflight" \
  "checking immutable V2/V3 inputs, pinned code and unopened principal_test"
if [[ "$(sha256sum "${SOURCE_V2_HOST}/orchestrator/seal_manifest.sha256" | cut -d' ' -f1)" != \
  "${SOURCE_V2_SEAL_SHA256}" ]] ||
  [[ -n "$(find "${SOURCE_V2_HOST}" -writable -print -quit)" ]] ||
  [[ "$(sha256sum "${SOURCE_V3_HOST}/orchestrator/final_training_manifest.json" | cut -d' ' -f1)" != \
    "${SOURCE_V3_FINAL_MANIFEST_SHA256}" ]] ||
  [[ "$(sha256sum "${SOURCE_V3_HOST}/orchestrator/measurements/manifest.json" | cut -d' ' -f1)" != \
    "${SOURCE_V3_MEASUREMENT_MANIFEST_SHA256}" ]] ||
  ! closed_test_gate "${SOURCE_V2_HOST}" ||
  ! closed_test_gate "${SOURCE_V3_HOST}" ||
  ! closed_test_gate "${RUN_HOST}" ||
  [[ "$(sha256sum "${OPTIMISATION_REPO}/${PROTOCOL}" | cut -d' ' -f1)" != \
    "${PROTOCOL_SHA256}" ]]; then
  write_status "failed" "preflight" "an immutable-input or closed-test invariant failed"
  exit 2
fi
(cd "${SOURCE_V2_HOST}" && sha256sum -c orchestrator/seal_manifest.sha256) \
  >"${CONTROL_HOST}/logs/v2-preflight-seal-check.log"
chmod a-w "${CONTROL_HOST}/logs/v2-preflight-seal-check.log"

write_status "running" "native_build" \
  "building the exact pairwise kernel with recorded compiler provenance"
python3 "${OPTIMISATION_REPO}/scripts/build_structured_native_kernel.py" \
  --output-directory "${RUN_HOST}/native" \
  --manifest "${RUN_HOST}/native/build.json" \
  --code-commit "${OPTIMISATION_CODE_COMMIT}" \
  >"${CONTROL_HOST}/logs/native-build.log" 2>&1
freeze_path "native"
chmod a-w "${CONTROL_HOST}/logs/native-build.log"

write_status "running" "bit_exact_verification" \
  "checking the native kernel and all 20 frozen structured checkpoints"
verification_arguments=(
  python scripts/verify_structured_native_kernel.py
  --kernel-manifest "${RUN}/native/build.json"
  --output "${RUN}/verification/native-kernel.json"
  --code-commit "${OPTIMISATION_CODE_COMMIT}"
  --linear-trials 512
  --policy-steps 2000
)
for family in "${STRUCTURED_FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    verification_arguments+=(
      --checkpoint "${family}" "${seed}"
      "/experiments/${SOURCE_V3_ID}/final/${family}/${seed}/checkpoint.npz"
    )
  done
done
measurement_container 0 "${verification_arguments[@]}" \
  >"${CONTROL_HOST}/logs/native-verification.log" 2>&1
freeze_path "verification/native-kernel.json"
chmod a-w "${CONTROL_HOST}/logs/native-verification.log"

run_validation_job() {
  local cpu="$1"
  local family="$2"
  local seed="$3"
  local output_host="${RUN_HOST}/validation/${family}-${seed}.json"
  local output="${RUN}/validation/${family}-${seed}.json"
  local status="${CONTROL_HOST}/jobs/${family}-${seed}.json"
  local log="${CONTROL_HOST}/logs/validation-${family}-${seed}.log"
  write_job_status "${CONTROL_HOST}/jobs" "structured_paired_validation" \
    "${family}" "${seed}" "running" "logical CPU ${cpu}" 0
  set +e
  measurement_container "${cpu}" python scripts/evaluate_principal_student.py \
    --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
    --checkpoint "/experiments/${SOURCE_V3_ID}/final/${family}/${seed}/checkpoint.npz" \
    --split validation --output "${output}" \
    --code-commit "${OPTIMISATION_CODE_COMMIT}" >"${log}" 2>&1
  local exit_code=$?
  set -e
  if [[ ${exit_code} -ne 0 ]] || [[ ! -f "${output_host}" ]]; then
    [[ ! -e "${output_host}" ]] || freeze_path "validation/${family}-${seed}.json"
    chmod a-w "${log}"
    write_job_status "${CONTROL_HOST}/jobs" "structured_paired_validation" \
      "${family}" "${seed}" "failed" "no automatic retry" "${exit_code}"
    chmod a-w "${status}"
    return 1
  fi
  freeze_path "validation/${family}-${seed}.json"
  chmod a-w "${log}"
  write_job_status "${CONTROL_HOST}/jobs" "structured_paired_validation" \
    "${family}" "${seed}" "completed" "native result frozen" 0
  chmod a-w "${status}"
}

write_status "running" "structured_paired_validation" \
  "20 structured models on the identical frozen validation schedule"
worker_pids=()
index=0
for family in "${STRUCTURED_FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_validation_job "${index}" "${family}" "${seed}" &
    worker_pids+=("$!")
    index=$((index + 1))
  done
done
validation_failures=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "${worker_pid}"; then
    validation_failures=$((validation_failures + 1))
  fi
done
if [[ ${validation_failures} -ne 0 ]]; then
  write_status "failed" "structured_paired_validation" \
    "${validation_failures} workers failed; no automatic retry"
  exit 1
fi

run_latency_job() {
  local family="$1"
  local seed="$2"
  local output_host="${RUN_HOST}/latency/${family}-${seed}.json"
  local output="${RUN}/latency/${family}-${seed}.json"
  local status="${CONTROL_HOST}/latency_jobs/${family}-${seed}.json"
  local log="${CONTROL_HOST}/logs/latency-${family}-${seed}.log"
  write_job_status "${CONTROL_HOST}/latency_jobs" "isolated_cpu_latency" \
    "${family}" "${seed}" "running" "logical CPU 0" 0
  set +e
  measurement_container 0 python scripts/benchmark_principal_cpu_latency.py \
    --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
    --checkpoint "/experiments/${SOURCE_V3_ID}/final/${family}/${seed}/checkpoint.npz" \
    --output "${output}" --code-commit "${OPTIMISATION_CODE_COMMIT}" \
    --container-digest "${IMAGE_DIGEST}" >"${log}" 2>&1
  local exit_code=$?
  set -e
  if [[ ${exit_code} -ne 0 ]] || [[ ! -f "${output_host}" ]]; then
    [[ ! -e "${output_host}" ]] || freeze_path "latency/${family}-${seed}.json"
    chmod a-w "${log}"
    write_job_status "${CONTROL_HOST}/latency_jobs" "isolated_cpu_latency" \
      "${family}" "${seed}" "failed" "no automatic retry" "${exit_code}"
    chmod a-w "${status}"
    return 1
  fi
  freeze_path "latency/${family}-${seed}.json"
  chmod a-w "${log}"
  write_job_status "${CONTROL_HOST}/latency_jobs" "isolated_cpu_latency" \
    "${family}" "${seed}" "completed" "isolated result frozen" 0
  chmod a-w "${status}"
}

write_status "running" "cpu_latency" \
  "35 serial benchmarks in one isolated logical-CPU-0 session"
if [[ -n "$(docker ps -q)" ]]; then
  write_status "failed" "cpu_latency" "another container prevents isolation"
  exit 2
fi
for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if ! run_latency_job "${family}" "${seed}"; then
      write_status "failed" "cpu_latency" \
        "latency failed for ${family}/${seed}; no automatic retry"
      exit 1
    fi
  done
done

write_status "running" "evidence_freeze" \
  "checking unchanged outcomes, reused hashes and new isolated latency"
control_container python scripts/verify_structured_optimisation_evidence.py \
  --protocol "${PROTOCOL}" --run-root "${RUN}" \
  --source-v3-root "/experiments/${SOURCE_V3_ID}" \
  --optimisation-code-commit "${OPTIMISATION_CODE_COMMIT}" \
  --orchestration-code-commit "${ORCHESTRATION_COMMIT}" \
  --source-v2-seal-sha256 "${SOURCE_V2_SEAL_SHA256}" \
  --source-v3-final-manifest-sha256 "${SOURCE_V3_FINAL_MANIFEST_SHA256}" \
  --source-v3-measurement-manifest-sha256 "${SOURCE_V3_MEASUREMENT_MANIFEST_SHA256}" \
  --output "${RUN}/manifest.json" \
  >"${CONTROL_HOST}/logs/evidence-freeze.log" 2>&1
freeze_path "manifest.json"
chmod a-w "${CONTROL_HOST}/logs/evidence-freeze.log"

if [[ "$(jq -r .decision "${RUN_HOST}/manifest.json")" != "GO" ]] ||
  [[ "$(sha256sum "${SOURCE_V2_HOST}/orchestrator/seal_manifest.sha256" | cut -d' ' -f1)" != \
    "${SOURCE_V2_SEAL_SHA256}" ]] ||
  [[ "$(sha256sum "${SOURCE_V3_HOST}/orchestrator/measurements/manifest.json" | cut -d' ' -f1)" != \
    "${SOURCE_V3_MEASUREMENT_MANIFEST_SHA256}" ]] ||
  ! closed_test_gate "${SOURCE_V2_HOST}" ||
  ! closed_test_gate "${SOURCE_V3_HOST}" ||
  ! closed_test_gate "${RUN_HOST}"; then
  write_status "failed" "postflight" "evidence or closed-test invariant failed"
  exit 2
fi
(cd "${SOURCE_V2_HOST}" && sha256sum -c orchestrator/seal_manifest.sha256) \
  >"${CONTROL_HOST}/logs/v2-postflight-seal-check.log"
chmod a-w "${CONTROL_HOST}/logs/v2-postflight-seal-check.log"
write_status "completed" "optimisation_complete" \
  "bit-exact native inference, identical validation and 35 latency results frozen"
(cd "${RUN_HOST}" && find . -type f ! -path './orchestrator/seal_manifest.sha256' \
  -print0 | sort -z | xargs -0 sha256sum) \
  >"${CONTROL_HOST}/seal_manifest.sha256"
completed=1
trap - EXIT
freeze_path "."
