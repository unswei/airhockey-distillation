#!/usr/bin/env bash
# Run inside the pinned container with /workspace read-only, /experiments
# read-only, and a fresh writable /output. CPU ids 0--19 must be available.
set -euo pipefail
readonly IMAGE_DIGEST="sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f"
readonly BASE_COMMIT="57f6118633e66ddabbef6061e1236db23e93231c"
readonly PRINCIPAL_TEST="/experiments/principal-test-v1-2026-08-15-v1/test"
readonly NATIVE="/experiments/principal-sweep-v1-2026-08-15-v4-attempt2/native"
export PYTHONPATH="/workspace:/workspace/src:/src/2025-challenge:${NATIVE}"
export PYTHONDONTWRITEBYTECODE=1
export AIRHOCKEY_STRUCTURED_PAIRWISE_BACKEND=native
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p /output/logs
python -m scripts.evaluate_observation_noise --prepare --output /output \
  --principal-test "${PRINCIPAL_TEST}" --base-code-commit "${BASE_COMMIT}" \
  --container-digest "${IMAGE_DIGEST}"

# All zero-noise regression checks finish before opening the new shot grid.
for family in structured_k0 structured_k4 gru_n64 teacher; do
  taskset -c 0-3 python -u -m scripts.evaluate_observation_noise --output /output \
    --family "${family}" --seed 14303 --smoke >"/output/logs/smoke-${family}.log" 2>&1
done

pids=()
taskset -c 0-3 python -u -m scripts.evaluate_observation_noise --output /output \
  --family teacher > /output/logs/teacher.log 2>&1 &
pids+=("$!")
cpu=4
for family in structured_k0 structured_k4 gru_n64; do
  for seed in 14303 14304 14305 14306 14307; do
    taskset -c "${cpu}" python -u -m scripts.evaluate_observation_noise --output /output \
      --family "${family}" --seed "${seed}" >"/output/logs/${family}-${seed}.log" 2>&1 &
    pids+=("$!")
    cpu=$((cpu + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" != 0 ]]; then
  echo "One or more evaluations failed; preserve all logs and partial evidence." >&2
  exit 1
fi
echo "All 16 frozen-policy evaluations completed."
