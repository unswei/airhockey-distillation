#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_HOST="marvin"
readonly TRAINING_COMMIT="099602c8b12fe804eb8a4da684b7afd2a667572a"
readonly ORCHESTRATION_COMMIT="${V3_ORCHESTRATION_COMMIT:-}"
readonly ORCHESTRATION_REPO="/home/oliver/Code/airhockey-memory-distillation"
readonly TRAINING_REPO="/home/oliver/Code/airhockey-memory-distillation-v3-099602c"
readonly EXPERIMENT_HOST="/home/oliver/experiments/airhockey-memory-distillation"
readonly SOURCE_RUN_ID="principal-sweep-v1-2026-08-13-v2"
readonly SOURCE_RUN_HOST="${EXPERIMENT_HOST}/${SOURCE_RUN_ID}"
readonly RUN_ID="${PRINCIPAL_V3_RUN_ID:-principal-sweep-v1-2026-08-14-v3}"
readonly RUN_HOST="${EXPERIMENT_HOST}/${RUN_ID}"
readonly RUN="/experiments/${RUN_ID}"
readonly SOURCE_RUN="/experiments/${SOURCE_RUN_ID}"
readonly IMAGE="marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt"
readonly IMAGE_DIGEST="sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f"
readonly PROTOCOL="configs/experiments/principal_sweep_execution_v1.yaml"
readonly PROTOCOL_SHA256="1f8ececf4492f68517938ccd2fd1e05bb459e0fdd7275851536f71d782bed448"
readonly SOURCE_COMMIT="f781ee1cbd85fce6e966bf65b43d017236946ca0"
readonly SOURCE_SEAL_SHA256="94ea1e57d7959f1ef368a681945d2ff29e5ed60769a1513e6af212f1c4933be3"
readonly SOURCE_FAILED_SHA256="e369597eaa640692c6db2413e82266d10e731af6be919c24065494acbded57a1"
readonly WORKER_CPUSETS=("0-23")
readonly FAMILIES=(feed_forward finite_stack_10 structured_k0 structured_k1 structured_k2 structured_k4 gru_n64)
readonly SEEDS=(14303 14304 14305 14306 14307)

if [[ "$(hostname -s)" != "${EXPECTED_HOST}" ]]; then
  echo "V3 final training must run on ${EXPECTED_HOST}" >&2
  exit 2
fi
if [[ ! "${ORCHESTRATION_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "V3_ORCHESTRATION_COMMIT must be the frozen control-plane commit" >&2
  exit 2
fi
if [[ "$(git -C "${ORCHESTRATION_REPO}" rev-parse HEAD)" != "${ORCHESTRATION_COMMIT}" ]] ||
  [[ -n "$(git -C "${ORCHESTRATION_REPO}" status --porcelain)" ]]; then
  echo "orchestration checkout is not clean at its declared commit" >&2
  exit 2
fi
if [[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" != "${IMAGE_DIGEST}" ]]; then
  echo "pinned container digest mismatch" >&2
  exit 2
fi

mkdir -p "${RUN_HOST}/orchestrator/logs" \
  "${RUN_HOST}/orchestrator/jobs" \
  "${RUN_HOST}/orchestrator/verifications"
exec 9>"${RUN_HOST}/orchestrator/runner.lock"
if ! flock -n 9; then
  echo "another V3 runner holds the orchestration lock" >&2
  exit 2
fi

write_status() {
  local status="$1"
  local stage="$2"
  local detail="$3"
  python3 - "${RUN_HOST}/orchestrator/status.json" "${status}" "${stage}" \
    "${detail}" "${TRAINING_COMMIT}" "${ORCHESTRATION_COMMIT}" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "detail": sys.argv[4],
    "training_code_commit": sys.argv[5],
    "orchestration_code_commit": sys.argv[6],
    "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "pid": os.getppid(),
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

write_job_status() {
  local family="$1"
  local seed="$2"
  local status="$3"
  local detail="$4"
  local exit_code="$5"
  python3 - "${RUN_HOST}/orchestrator/jobs/${family}-${seed}.json" \
    "${family}" "${seed}" "${status}" "${detail}" "${exit_code}" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "family_id": sys.argv[2],
    "training_seed": int(sys.argv[3]),
    "status": sys.argv[4],
    "detail": sys.argv[5],
    "exit_code": int(sys.argv[6]),
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

training_container() {
  local cpuset="$1"
  shift
  docker run --rm --ipc host --cpuset-cpus "${cpuset}" \
    --memory 9g --memory-swap 10g \
    -e PYTHONPATH=/workspace:/workspace/src:/src/2025-challenge \
    -e OMP_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
    -e MKL_NUM_THREADS=16 -e NUMEXPR_NUM_THREADS=16 \
    -v "${TRAINING_REPO}:/workspace:ro" \
    -v "${EXPERIMENT_HOST}:/experiments:ro" \
    -v "${RUN_HOST}:${RUN}" \
    -w /workspace "${IMAGE}" "$@"
}

control_container() {
  docker run --rm --ipc host \
    -e PYTHONPATH=/control:/control/src:/src/2025-challenge \
    -v "${ORCHESTRATION_REPO}:/control:ro" \
    -v "${EXPERIMENT_HOST}:/experiments:ro" \
    -v "${RUN_HOST}:${RUN}" \
    -w /control "${IMAGE}" "$@"
}

job_is_complete() {
  local family="$1"
  local seed="$2"
  local verification="${RUN_HOST}/orchestrator/verifications/${family}-${seed}.json"
  local checkpoint="${RUN_HOST}/final/${family}/${seed}/checkpoint.npz"
  [[ -f "${verification}" && -f "${checkpoint}" ]] || return 1
  [[ "$(jq -r .decision "${verification}")" == "GO" ]] || return 1
  [[ "$(sha256sum "${checkpoint}" | cut -d' ' -f1)" == \
    "$(jq -r .checkpoint_sha256 "${verification}")" ]] || return 1
  [[ -z "$(find "${RUN_HOST}/final/${family}/${seed}" -type f -writable -print -quit)" ]]
}

run_job() {
  local lane="$1"
  local family="$2"
  local seed="$3"
  local cpuset="${WORKER_CPUSETS[${lane}]}"
  local output_host="${RUN_HOST}/final/${family}/${seed}"
  local output="${RUN}/final/${family}/${seed}"
  local verification_host="${RUN_HOST}/orchestrator/verifications/${family}-${seed}.json"
  local verification="${RUN}/orchestrator/verifications/${family}-${seed}.json"
  local status_path="${RUN_HOST}/orchestrator/jobs/${family}-${seed}.json"
  local log="${RUN_HOST}/orchestrator/logs/final-${family}-${seed}.log"

  if job_is_complete "${family}" "${seed}"; then
    return 0
  fi
  if [[ -f "${status_path}" && "$(jq -r .status "${status_path}")" == "failed" ]]; then
    echo "refusing automatic retry of failed job ${family}/${seed}" >&2
    return 1
  fi
  if [[ -e "${output_host}" || -e "${verification_host}" ]]; then
    echo "refusing to overwrite incomplete V3 artefacts for ${family}/${seed}" >&2
    write_job_status "${family}" "${seed}" "failed" \
      "incomplete artefacts require explicit diagnosis" 2
    return 1
  fi

  write_job_status "${family}" "${seed}" "running" \
    "lane ${lane}, CPUs ${cpuset}" 0
  set +e
  training_container "${cpuset}" python scripts/train_principal_student.py \
    --protocol "${PROTOCOL}" --family "${family}" --seed "${seed}" \
    --stage final --dataset "${SOURCE_RUN}/datasets/${family}" \
    --output "${output}" --code-commit "${TRAINING_COMMIT}" \
    >"${log}" 2>&1
  local training_exit=$?
  set -e
  if [[ ${training_exit} -ne 0 ]]; then
    [[ ! -e "${output_host}" ]] || freeze_v3_path "final/${family}/${seed}"
    chmod a-w "${log}"
    write_job_status "${family}" "${seed}" "failed" \
      "training command failed; no automatic retry" "${training_exit}"
    chmod a-w "${status_path}"
    return 1
  fi

  local source_arguments=()
  if [[ -f "${SOURCE_RUN_HOST}/final/${family}/${seed}/checkpoint.npz" ]]; then
    source_arguments=(--source-checkpoint \
      "${SOURCE_RUN}/final/${family}/${seed}/checkpoint.npz")
  fi
  set +e
  control_container python scripts/verify_principal_v3_final.py \
    --family "${family}" --seed "${seed}" \
    --final-directory "${output}" \
    --dataset-manifest "${SOURCE_RUN}/datasets/${family}/manifest.json" \
    --expected-code-commit "${TRAINING_COMMIT}" \
    --expected-protocol-sha256 "${PROTOCOL_SHA256}" \
    "${source_arguments[@]}" --output "${verification}" \
    >>"${log}" 2>&1
  local verification_exit=$?
  set -e
  freeze_v3_path "final/${family}/${seed}"
  [[ ! -f "${verification_host}" ]] || \
    freeze_v3_path "orchestrator/verifications/${family}-${seed}.json"
  chmod a-w "${log}"
  if [[ ${verification_exit} -ne 0 ]]; then
    write_job_status "${family}" "${seed}" "failed" \
      "fail-closed final checkpoint verification failed" "${verification_exit}"
    chmod a-w "${status_path}"
    return 1
  fi
  write_job_status "${family}" "${seed}" "completed" \
    "trained, independently verified and frozen" 0
  chmod a-w "${status_path}"
}

write_status "running" "preflight" "verifying sealed V2 reuse inputs"
if [[ ! -d "${TRAINING_REPO}" ]]; then
  git -C "${ORCHESTRATION_REPO}" worktree add --detach \
    "${TRAINING_REPO}" "${TRAINING_COMMIT}"
fi
if [[ "$(git -C "${TRAINING_REPO}" rev-parse HEAD)" != "${TRAINING_COMMIT}" ]] ||
  [[ -n "$(git -C "${TRAINING_REPO}" status --porcelain)" ]]; then
  echo "training worktree is not clean at the correction commit" >&2
  exit 2
fi
if [[ "$(sha256sum "${SOURCE_RUN_HOST}/orchestrator/seal_manifest.sha256" | cut -d' ' -f1)" != \
  "${SOURCE_SEAL_SHA256}" ]]; then
  echo "V2 seal manifest hash mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${SOURCE_RUN_HOST}/orchestrator/SEALED_FAILED.json" | cut -d' ' -f1)" != \
  "${SOURCE_FAILED_SHA256}" ]]; then
  echo "V2 sealed-failure record hash mismatch" >&2
  exit 2
fi
if [[ -n "$(find "${SOURCE_RUN_HOST}" -type f -writable -print -quit)" ]] ||
  find "${SOURCE_RUN_HOST}" -maxdepth 1 -name 'principal_test*' -print -quit | grep -q .; then
  echo "V2 is writable or principal_test exists" >&2
  exit 2
fi
if [[ ! -f "${RUN_HOST}/orchestrator/logs/v2-full-seal-check.log" ]]; then
  (cd "${SOURCE_RUN_HOST}" && \
    sha256sum -c orchestrator/seal_manifest.sha256 \
      >"${RUN_HOST}/orchestrator/logs/v2-full-seal-check.log")
  chmod a-w "${RUN_HOST}/orchestrator/logs/v2-full-seal-check.log"
fi

if [[ ! -e "${RUN_HOST}/reuse" ]]; then
  python3 "${ORCHESTRATION_REPO}/scripts/prepare_principal_v3_reuse.py" \
    --source-attempt "${SOURCE_RUN_HOST}" \
    --experiment-host-root "${EXPERIMENT_HOST}" \
    --output-directory "${RUN_HOST}/reuse" \
    --expected-source-seal-sha256 "${SOURCE_SEAL_SHA256}" \
    --expected-source-code-commit "${SOURCE_COMMIT}" \
    --training-code-commit "${TRAINING_COMMIT}" \
    --orchestration-code-commit "${ORCHESTRATION_COMMIT}" \
    --container-image-digest "${IMAGE_DIGEST}" \
    >"${RUN_HOST}/orchestrator/logs/reuse-preflight.log" 2>&1
  chmod -R a-w "${RUN_HOST}/reuse" \
    "${RUN_HOST}/orchestrator/logs/reuse-preflight.log"
fi
if [[ "$(jq -r .selected_manifest.sha256 "${RUN_HOST}/reuse/manifest.json")" != \
  "c3a408686bd7bac4d90f58f0f68fbdb56624bf7953f8ebdea7700fe7353813ca" ]]; then
  echo "V3 reuse manifest differs" >&2
  exit 2
fi

if [[ ! -f "${RUN_HOST}/orchestrator/resolved_run.json" ]]; then
  python3 - "${RUN_HOST}/orchestrator/resolved_run.json" <<PY
import datetime, json, pathlib
path = pathlib.Path("${RUN_HOST}/orchestrator/resolved_run.json")
path.write_text(json.dumps({
    "schema_version": 1,
    "status": "final_training_only",
    "run_id": "${RUN_ID}",
    "source_run_id": "${SOURCE_RUN_ID}",
    "source_aggregation_code_commit": "${SOURCE_COMMIT}",
    "training_code_commit": "${TRAINING_COMMIT}",
    "orchestration_code_commit": "${ORCHESTRATION_COMMIT}",
    "protocol_sha256": "${PROTOCOL_SHA256}",
    "container_image_digest": "${IMAGE_DIGEST}",
    "parallel_worker_lanes": 1,
    "worker_cpu_sets": ["0-23"],
    "parallelism_decision": "one trainer is maximum useful under the frozen 16-thread runtime; concurrent-trainer benchmarks reduced throughput",
    "gpu_used": False,
    "principal_test_opened": False,
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
}, indent=2, sort_keys=True) + "\n")
PY
  chmod a-w "${RUN_HOST}/orchestrator/resolved_run.json"
fi

write_status "running" "canary_final_training" \
  "three overlapping checkpoints before the remaining campaign"
if ! run_job 0 structured_k0 14306 ||
  ! run_job 0 feed_forward 14303 ||
  ! run_job 0 finite_stack_10 14303; then
  write_status "failed" "canary_final_training" \
    "at least one canary failed; remaining models were not dispatched"
  exit 1
fi

if [[ ! -f "${RUN_HOST}/orchestrator/schedule.json" ]]; then
  python3 - "${SOURCE_RUN_HOST}" "${RUN_HOST}/orchestrator" <<'PY'
import json, pathlib, sys
source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
families = (
    "feed_forward", "finite_stack_10", "structured_k0", "structured_k1",
    "structured_k2", "structured_k4", "gru_n64",
)
seeds = (14303, 14304, 14305, 14306, 14307)
canaries = {("feed_forward", 14303), ("finite_stack_10", 14303), ("structured_k0", 14306)}
jobs = []
for family in families:
    for seed in seeds:
        if (family, seed) in canaries:
            continue
        result = json.loads(
            (source / "collectors" / family / str(seed) / "result.json").read_text()
        )
        jobs.append({
            "family_id": family,
            "training_seed": seed,
            "estimated_seconds": float(result["duration_seconds"]) * 2.0,
        })
jobs.sort(key=lambda value: (-value["estimated_seconds"], value["family_id"], value["training_seed"]))
lanes = [{"lane": 0, "estimated_seconds": 0.0, "jobs": []}]
for job in jobs:
    lane = min(lanes, key=lambda value: (value["estimated_seconds"], value["lane"]))
    lane["jobs"].append(job)
    lane["estimated_seconds"] += job["estimated_seconds"]
schedule = {"schema_version": 1, "strategy": "longest_processing_time_first_from_frozen_collector_durations", "lanes": lanes}
(output / "schedule.json").write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
for lane in lanes:
    text = "".join(f"{job['family_id']}\t{job['training_seed']}\n" for job in lane["jobs"])
    (output / f"worker-{lane['lane']}.tsv").write_text(text)
PY
  chmod a-w "${RUN_HOST}/orchestrator/schedule.json" \
    "${RUN_HOST}/orchestrator/worker-0.tsv"
fi

run_worker() {
  local lane="$1"
  local family
  local seed
  while IFS=$'\t' read -r family seed; do
    run_job "${lane}" "${family}" "${seed}" || return 1
  done <"${RUN_HOST}/orchestrator/worker-${lane}.tsv"
}

write_status "running" "remaining_final_training" \
  "32 remaining models on the maximum-throughput frozen-runtime lane"
if ! run_worker 0; then
  write_status "failed" "remaining_final_training" \
    "at least one lane failed; inspect immutable per-job evidence"
  exit 1
fi

write_status "running" "final_training_freeze" \
  "verifying and hashing all 35 frozen final models"
python3 - "${RUN_HOST}" "${TRAINING_COMMIT}" "${ORCHESTRATION_COMMIT}" <<'PY'
import datetime, hashlib, json, pathlib, stat, sys
root = pathlib.Path(sys.argv[1])
families = ("feed_forward", "finite_stack_10", "structured_k0", "structured_k1", "structured_k2", "structured_k4", "gru_n64")
seeds = (14303, 14304, 14305, 14306, 14307)
records = []
source_exact = 0
for family in families:
    for seed in seeds:
        final = root / "final" / family / str(seed)
        verification_path = root / "orchestrator" / "verifications" / f"{family}-{seed}.json"
        verification = json.loads(verification_path.read_text())
        if verification["decision"] != "GO":
            raise RuntimeError(f"non-GO verification: {family}/{seed}")
        if any(stat.S_IMODE(path.stat().st_mode) & 0o222 for path in final.rglob("*") if path.is_file()):
            raise RuntimeError(f"writable final artefact: {family}/{seed}")
        equivalence = verification.get("source_equivalence")
        if equivalence is not None:
            if equivalence["parameter_arrays_exact"] is not True:
                raise RuntimeError(f"source parameter mismatch: {family}/{seed}")
            source_exact += 1
        records.append({
            "family_id": family,
            "training_seed": seed,
            "checkpoint_sha256": verification["checkpoint_sha256"],
            "result_sha256": hashlib.sha256((final / "result.json").read_bytes()).hexdigest(),
            "verification_sha256": hashlib.sha256(verification_path.read_bytes()).hexdigest(),
            "source_parameter_arrays_exact": None if equivalence is None else True,
        })
if len(records) != 35 or source_exact != 14:
    raise RuntimeError("final checkpoint or source-equivalence count differs")
manifest = {
    "schema_version": 1,
    "status": "completed",
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "training_code_commit": sys.argv[2],
    "orchestration_code_commit": sys.argv[3],
    "final_models": 35,
    "source_overlapping_models_with_exact_parameter_arrays": source_exact,
    "principal_test_opened": False,
    "models": records,
}
path = root / "orchestrator" / "final_training_manifest.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
chmod a-w "${RUN_HOST}/orchestrator/final_training_manifest.json"

if [[ "$(sha256sum "${SOURCE_RUN_HOST}/orchestrator/seal_manifest.sha256" | cut -d' ' -f1)" != \
  "${SOURCE_SEAL_SHA256}" ]] ||
  [[ -n "$(find "${SOURCE_RUN_HOST}" -type f -writable -print -quit)" ]] ||
  find "${SOURCE_RUN_HOST}" -maxdepth 1 -name 'principal_test*' -print -quit | grep -q .; then
  write_status "failed" "postflight" "V2 seal or principal_test state changed"
  exit 2
fi
write_status "completed" "final_training_complete" \
  "35 final models independently verified and frozen; principal_test remains closed"
