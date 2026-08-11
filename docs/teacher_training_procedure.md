# Corrected teacher training procedure

Validated: 2026-08-12

Decision: **the learning procedure is fixed; schedule the full teacher run**

## Fault in the first diagnostic

The first 20,000-step diagnostic used a replay ratio of 8. The pinned upstream
air-hockey procedure uses 80, and the retained published model configuration
uses 128. Dreamer also applies a 1,000-update optimiser warm-up. The first
diagnostic logged only about 610 optimiser updates, so it ended before the
optimiser completed warm-up.

The runner also coupled `run.envs` to the logging interval by mistake, used a
hard-coded one-second logging interval, retained only the latest wall-clock
checkpoint, and did not force a checkpoint at the completed step. These did
not explain the weak policy by themselves, but they made a long run harder to
resume and impossible to select reliably by validation performance.

## Correction

The versioned `learning_check` and `full` profiles now use:

- replay ratio 80;
- the upstream air-hockey replay mixture: 0.2 uniform, 0.6 prioritised and
  0.2 recency sampling;
- independent environment, logging, reporting and checkpoint settings;
- one serial environment, avoiding identically seeded parallel samplers;
- retained checkpoints at exact environment-step intervals;
- an explicit checkpoint at the completed step;
- exact-state resume of the agent, replay and environment-step counter;
- deterministic validation of every retained checkpoint;
- selection by validation save rate averaged across blackout durations, with
  mean return and then earlier step as tie-breaks.

The final test split remains untouched. The implementation is a narrow runtime
adaptation around the pinned Dreamer training loop; Dreamer itself is not
rewritten.

## Corrected learning check

The corrected check held the model size, task, public observation and 20,000
environment-step budget fixed. It changed the optimisation and replay
procedure described above. Training completed in 146.18 seconds and wrote an
exact step-20,000 checkpoint. The measured replay ratio reached 81.24. Reward
loss fell from 5.48 to 2.45, compared with 5.54 to 4.19 under the original
procedure.

The identically seeded untrained and trained policies were evaluated on the
same 30 validation shots at blackout lengths 0, 10 and 20.

| Policy | Saves | Concessions | Mean return |
| --- | ---: | ---: | ---: |
| Untrained | 26/90 (28.9%) | 58/90 (64.4%) | -0.282 |
| Step 20,000 | 30/90 (33.3%) | 50/90 (55.6%) | -0.131 |

Across the paired cases, training gained 21 saves and lost 17. Episode return
improved in 28 cases, was unchanged in 43 and worsened in 19. This is a small
positive learning check, not evidence that the teacher is ready. It is enough
to distinguish the corrected procedure from the original run, which reduced
held-out save rate and mean return.

An additional 1,000-step smoke run preserved exact checkpoints at steps 0,
500 and 1,000, wrote training metrics, and completed successfully. This
validates the checkpoint schedule before the long run.

## Full run

The `full` profile requests 1,000,000 environment steps with seed 7201 and the
`size1m` model. It retains checkpoints every 20,000 steps and at completion.
After training, every retained checkpoint is evaluated on 100 fixed validation
shots at blackout lengths 0, 5, 10, 15 and 20: 500 paired episodes per
checkpoint. The selected checkpoint must still satisfy the teacher-readiness
criteria before student training begins.

The scheduled Marvin run is:

```text
systemd unit: airhockey-teacher-full-v1-v2.service
run id: teacher-full-v1-2026-08-12-v2
code commit: f1fb9606173bc7312ecbb4c1d91d5b996ed05c5e
artefact path: /home/oliver/experiments/airhockey-memory-distillation/teacher-full-v1-2026-08-12-v2
```

The unit uses `scripts/run_full_teacher_on_marvin.sh`, restarts on failure and
resumes from the latest complete checkpoint. Successful training is followed
automatically by validation checkpoint selection. The initial v1 systemd
launch failed before starting Docker because Marvin's long-lived user service
manager had not inherited the user's Docker group. Its run directory is
preserved. The v2 unit enters the `docker` group explicitly and reached the
Dreamer training loop with no restart.

## Evidence

Corrected learning check:

```text
/home/oliver/experiments/airhockey-memory-distillation/teacher-learning-check-corrected-2026-08-12-v1
```

| File | SHA-256 |
| --- | --- |
| `untrained_evaluation.json` | `593e3a1b6498fcef7193c86a1b550ddb4daf4d1efa6de4565f7cf060c0514cc9` |
| `trained_evaluation.json` | `21a383eece5d8ac7430543c149d37e405909cbfc7adb47f0837c99ac2cde732f` |
| `training/learning_check_result.json` | `53e6268ad99a9f56aaa6557d9d2bfaac90541c6df8a50fa72d6f5ebe804a95fe` |
| `training/dreamer/metrics.jsonl` | `241d076538b4209e914227535a857e31c590f885a9b94b86f0d307598b715c8b` |

Exact-step checkpoint smoke:

```text
/home/oliver/experiments/airhockey-memory-distillation/teacher-checkpoint-procedure-smoke-2026-08-12-v2
```

| File | SHA-256 |
| --- | --- |
| `training/smoke_result.json` | `6138be8aded79131ec7365e7c4588444c87c8169fae3d3572dbb5767cc4f0191` |
| `training/dreamer/metrics.jsonl` | `6d239317848c9898d15368e954ffb93dc6dc8ff966d91b6da2969b35032e5720` |
