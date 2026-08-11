# DreamerV3 teacher smoke test

Validated: 2026-08-11

Decision: **PASS for runtime compatibility**

This smoke test connects the project task to the pinned DreamerV3 fork. It is
deliberately too short, and uses a zero reward, so it is not evidence of policy
quality or learning progress.

## Scope

The final run used:

- code commit `55db96a9386aca5f780ca3e8192b362db587f442`;
- the pinned Blackwell image at
  `sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f`;
- DreamerV3 commit `4049794d4135e41c691f18da38a9af7541b01553`;
- the `size1m` preset, containing 639,446 parameters with this interface;
- one serial MuJoCo environment on JAX 0.5.3 `cuda:0`;
- the 19-dimensional public observation and two-dimensional action;
- `direct_launch_v2` train shots;
- one blackout per episode, with length sampled uniformly from 0 to 20 steps;
- 1,000 environment steps, batch size 4, sequence length 16 and training ratio
  8.

The runner rejects a nominally completed process unless at least one
`train/*` metric is written.

## Result

The run completed in 35.25 seconds. Dreamer compiled its policy, training and
report functions on the RTX 5090, collected replay, performed 108 optimiser
updates and saved a checkpoint at step 1,000. Six metric records contained
training values, from step 210 through step 1,000. No numeric metric was
non-finite.

Selected final metrics are compatibility diagnostics only:

| Metric | Value |
| --- | ---: |
| World-model dynamics loss | 5.9384 |
| Observation reconstruction loss | 7.1051 |
| Reward loss | 5.5239 |
| Policy loss | -0.0000928 |
| Optimiser gradient norm | 196.62 |
| Optimiser update RMS | 0.00000474 |
| Replay ratio | 8.5 |
| Policy throughput | 114.6 steps/s |
| Training throughput | 917.2 samples/s |

The upstream defence reward is exactly zero. Episode scores and the reward
model therefore carry no task-performance meaning in this run. A real teacher
run must wait for a versioned reward that distinguishes concessions, contact
and conclusive saves.

## Corrections exposed by the smoke sequence

Four immutable attempts preceded the passing run:

1. v1 failed before construction because the pinned flag parser requires
   capitalised boolean strings.
2. v2 exposed a missing `elements` import in the pinned Gymnasium adapter. The
   project runner applies a local runtime shim without changing upstream.
3. v3 reached CUDA compilation and checkpoint initialisation, then detected a
   public observation at -1.0334 outside the declared `[-1,1]` space.
4. V4 completed 160 environment steps but wrote no training metrics, so it did
   not satisfy the strengthened gradient-evidence criterion.

The observation adapter now clips normalisation overshoot consistently to its
declared range, with pure and pinned-MuJoCo regression tests. The complete
Marvin suite passes 36 tests.

## Evidence

The passing artefact is outside Git at:

```text
/home/oliver/experiments/airhockey-memory-distillation/teacher-smoke-2026-08-11-v5
```

| File | SHA-256 |
| --- | --- |
| `smoke_result.json` | `8ba5f7dd47ceafecf5435061fd60b5ee04a47f385ab49af7ea0f402014a5fd41` |
| `dreamer/metrics.jsonl` | `2889654d7c6e926ec01a9c9113061746ec68fd4e132e5ae64d71a76e7b1a2a45` |
| `dreamer/config.yaml` | `0366d292712d8c41e74463b62b5108eae6b3b1fdb4e2f43443f7b7f266146a51` |
| Final `agent.pkl` | `50a7a84f7241659c8ca4dc7cf5574f290fe6c0807dfe6231382fc3d4050b3a65` |
| Console log | `622faee013ef1ed1279f540bfe262b5eabfda32f9486b33ae872f16b9a191280` |

The passing directory occupies 8.7 MB. It is reproducible evidence, not a
teacher checkpoint to retain for evaluation.
