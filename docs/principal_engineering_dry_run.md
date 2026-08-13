# Principal-family engineering dry run

This is a small, explicitly non-principal check that all seven predeclared
student implementations can traverse the same executable pipeline before the
principal sweep starts. Its checkpoints, rollouts and latency values are not
eligible for the principal release manifest.

The driver checks:

- overfitting one no-blackout episode from the frozen deterministic
  teacher-mean tiny dataset;
- exact framework-neutral checkpoint reload;
- NumPy/PyTorch action and carry agreement;
- identical small shadow-schedule projections for all seven families;
- four real MuJoCo validation episodes per family, using two validation shots
  at blackout lengths zero and five; and
- a reduced batch-one NumPy latency measurement on Marvin.

The overfit gate requires action MSE at most `0.01` and at least 98% reduction
from initial loss. Both conditions matter. This is a family-neutral engineering
criterion, not the principal training objective or a claim that the tiny
models have useful closed-loop performance. The no-blackout target prevents
the memoryless family from being asked to reproduce a history-dependent
blackout target during this plumbing check.

The latency check validates the predeclared Marvin host, CPU, container,
logical-CPU affinity and single-thread environment. It deliberately uses only
100 warm-up calls and three repetitions of 200 timed calls, so these values
must not be substituted for the principal benchmark's 10,000 warm-up calls
and ten repetitions of 10,000 calls.

Run it on Marvin with the code checkout mounted read-only:

```bash
docker run --rm --ipc host --hostname marvin --cpuset-cpus 0 \
  -e PYTHONPATH=/workspace:/workspace/src:/src/2025-challenge \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
  -v /home/oliver/Code/airhockey-memory-distillation:/workspace:ro \
  -v /home/oliver/experiments/airhockey-memory-distillation:/experiments \
  -w /workspace \
  marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt \
  python scripts/run_principal_engineering_dry_run.py \
    --protocol configs/experiments/principal_sweep_execution_v1.yaml \
    --tiny-dataset /experiments/teacher-datasets/teacher-v3-structured-n64-k2-tiny-deterministic-2026-08-12-v1 \
    --output /experiments/PRINCIPAL_DRY_RUN_ID \
    --code-commit FULL_40_CHARACTER_COMMIT \
    --epochs 3000 --overfit-mse 0.01 \
    --minimum-loss-reduction 0.98 \
    --validation-shots 2 --validation-blackouts 2 \
    --latency-warmup 100 --latency-calls 200 \
    --latency-repetitions 3 \
    --container-digest sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f
```

The result file and every checkpoint remain outside Git. The top-level result
sets both `principal_evidence_eligible` and `release_manifest_eligible` to
false, and every checkpoint is labelled `non_principal_dry_run`.
