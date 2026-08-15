# Principal sweep evidence and test release

The principal test stays closed by default. These controls turn the v1
predeclaration into executable checks without changing
`principal_sweep_v1.yaml` or its hash.

## Shadow-budget audit

After all five 4,000-episode partitions for one family are complete, run:

```bash
python scripts/audit_principal_shadow_budget.py \
  --protocol configs/experiments/principal_sweep_execution_v1.yaml \
  --family structured_k2 \
  --shadow-partition /experiments/shadow/structured_k2/14303 \
  --shadow-partition /experiments/shadow/structured_k2/14304 \
  --shadow-partition /experiments/shadow/structured_k2/14305 \
  --shadow-partition /experiments/shadow/structured_k2/14306 \
  --shadow-partition /experiments/shadow/structured_k2/14307 \
  --code-commit "$(git rev-parse HEAD)" \
  --output /experiments/audits/structured_k2.json
```

The auditor reopens every shard. It verifies hashes, all 20,000 realised
episode assignments, deterministic teacher-mean semantics, finite public
arrays, previous behaviour-command history, the frozen teacher, and the five
collector seeds. Episodes are the fixed budget. It reports the realised
teacher-query count but does not top up families with shorter episodes. A
failure writes `NO_GO` and exits non-zero.

## Parameter and state accounting

For each of the 35 final checkpoints, run:

```bash
python scripts/account_principal_policy.py \
  --protocol configs/experiments/principal_sweep_execution_v1.yaml \
  --family structured_k2 --seed 14303 \
  --checkpoint /experiments/final/structured_k2/14303/checkpoint.npz \
  --code-commit "$(git rev-parse HEAD)" \
  --output /experiments/accounting/structured_k2-14303.json
```

Counts come from the actual exported float32 tensors. Core parameters are
recurrent-update tensors only and are zero for feed-forward and finite-stack.
Recurrent-memory bytes include only the 64-value recurrent state. Complete
carry bytes additionally include the prior two-value requested action for the
recurrent families, or the 30-value puck-history buffer for finite-stack.
Multiply-adds count dense weight elements and the structured diagonal
recurrence, excluding biases, nonlinearities and buffer copies.

## CPU latency on Marvin

Run the predeclared benchmark inside the pinned container on Marvin. The
thread variables must exist before Python starts, the container hostname must
be `marvin`, and logical CPU 0 must be available to the container:

```bash
docker run --rm --ipc host --hostname marvin --cpuset-cpus 0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
  -v "$PWD:/workspace:ro" -v /home/oliver/experiments:/experiments \
  -w /workspace \
  marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt \
  python scripts/benchmark_principal_cpu_latency.py \
    --protocol configs/experiments/principal_sweep_execution_v1.yaml \
    --family structured_k2 --seed 14303 \
    --checkpoint /experiments/final/structured_k2/14303/checkpoint.npz \
    --container-digest sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f \
    --code-commit "$(git rev-parse HEAD)" \
    --output /experiments/latency/structured_k2-14303.json
```

The benchmark refuses another host, CPU, container digest, affinity or thread
environment. It records the CPU governor and observed frequency, runs 10,000
warm-up calls and ten repetitions of 10,000 complete batch-one NumPy
`policy.act` calls, then reports the median and p95 of the ten repetition
means. Loading and simulation are outside the timed region.

## Freeze and release gate

Keep all raw validation rows and release evidence outside Git. The schema-v2
JSON template documents the mixed provenance, while the freezer builds paths
and hashes from the actual files. V2 supplies the collectors, family datasets,
shadow audits and truthful aggregation commit. V3 supplies all 35 final
checkpoints, the 15 unchanged nonstructured validation files and all 35
accounting files. V4 supplies the 20 bit-exact structured validation reruns and
all 35 new isolated latency files.

The manifest also includes the frozen V3 final-training, V3 measurement and V4
optimisation manifests. The independent gate cross-checks every selected file
against those three sources, including the V4 native binary and bit-exact
verification. A plausible file with the right family and seed is insufficient
if it is not hash-bound by the correct source manifest.

```bash
python scripts/freeze_principal_release_evidence.py \
  --protocol configs/experiments/principal_sweep_execution_v1.yaml \
  --analysis-plan docs/principal_sweep_v1.md \
  --aggregation-code-commit V2_AGGREGATION_COMMIT \
  --v3-final-training-manifest /experiments/v3/orchestrator/final_training_manifest.json \
  --v3-measurement-manifest /experiments/v3/orchestrator/measurements/manifest.json \
  --v4-optimisation-manifest /experiments/v4/manifest.json \
  --declare-principal-test-uninspected \
  ...all repeated evidence arguments... \
  --output /experiments/release/evidence.json

python scripts/gate_principal_test_release.py \
  --protocol configs/experiments/principal_sweep_execution_v1.yaml \
  --evidence-manifest /experiments/release/evidence.json \
  --output /experiments/release/gate.json
```

The gate exits 2 and writes `NO_GO` if anything is missing, duplicated,
changed or inconsistent. It reopens checkpoints, datasets, shadow shards and
validation episode rows; binds accounting and latency to each exact final
checkpoint; verifies the V3 training commit separately from the V4
implementation commit; and requires native structured validation and latency
where declared. The declaration that the test was never inspected is
necessarily a frozen operator declaration; repository files cannot
independently prove it.

Only a `GO` report can be supplied to the evaluator:

```bash
python scripts/evaluate_principal_student.py \
  --protocol configs/experiments/principal_sweep_execution_v1.yaml \
  --split test --release-report /experiments/release/gate.json \
  --family structured_k2 --seed 14303 \
  --checkpoint /experiments/final/structured_k2/14303/checkpoint.npz \
  --code-commit "$(git rev-parse HEAD)" \
  --output /experiments/test/structured_k2-14303.json
```

Before constructing the 1,350-episode test schedule, evaluation re-hashes the
evidence manifest and re-runs the whole gate. Editing a report to say `GO`
does not open the test. Once released, the protocol forbids retraining,
reselection and hyperparameter changes.
