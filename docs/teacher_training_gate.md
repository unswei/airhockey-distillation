# Teacher-training gate

Applied: 2026-08-11

Decision: **NO_GO**

Teacher training must not begin until every check in the machine-readable gate
returns `PASS`. A failed check and a check with insufficient evidence both
block training.

## Criteria

The qualitative requirements in the project specification are operationalised
as follows:

| Check | Pass condition |
| --- | --- |
| Paired coverage | Every controller is evaluated on the same non-empty set of distinct shot IDs |
| Inactive defender | Concession rate at least 0.80 over at least 200 distinct shots |
| Fixed-centre defender | Concession rate at least 0.80 over at least 200 distinct shots |
| Privileged controller | Explicit clear, arrest or return on at least 0.80 of at least 200 distinct shots |
| Replay | Repetitions produce the exact same public trajectory hash |
| Observation | No puck velocity or opponent feature; blackout masking remains exact |
| Marvin reliability | At least 20 completed smoke episodes and zero simulator faults |

The 0.80 rate and 200-shot minimum are conservative pre-training thresholds,
not paper evaluation results. Repeating one deterministic shot does not
increase the number of distinct shots or the evidence for a rate.

Defender joint velocity remains in the proposed observation intentionally.
The isolation check concerns puck velocity and opponent information.

## Current evidence

| Check | Observation | Status |
| --- | --- | --- |
| Paired coverage | All three controls used `near_post_right_v1` | `PASS` |
| Inactive concession | 1/1 | `INSUFFICIENT_EVIDENCE` |
| Fixed-centre concession | 1/1 | `INSUFFICIENT_EVIDENCE` |
| Privileged save | 1/1, classified as returned | `INSUFFICIENT_EVIDENCE` |
| Exact replay | All 20 stability trajectories matched | `PASS` |
| Public observation isolation | Puck velocity invariant; no opponent component | `PASS` |
| Marvin simulator reliability | 20/20 episodes, zero faults | `PASS` |

The full test suite also passed in the pinned Marvin image: 19 tests in 1.68
seconds. The headless run emits a GLFW warning because `DISPLAY` is unset; no
rendering is requested and this did not affect the simulation.

The current point estimates look favourable, but they describe one fixed
direct launch. They do not establish that inactive and fixed defenders concede
often enough across a useful distribution, or that the privileged controller
saves most shots. Teacher training therefore remains blocked.

## Command and evidence

Run in the audited container on Marvin:

```bash
python3 scripts/run_teacher_gate.py \
  --output /run-output/teacher_gate.json
```

The runner exits with status 2 for `NO_GO`, including when evidence is
insufficient. The final report for this gate application is stored outside Git
at:

```text
/home/oliver/experiments/airhockey-memory-distillation/teacher-gate-2026-08-11-v2/teacher_gate.json
```

SHA-256:
`2c50a70f83e4dcfb9ad797a4f93dd776561cc62ea6e0bbd6794facedf8a0bd31`.

## Work required for GO

1. Implement and version the incoming-shot distribution, initially using
   direct launch if necessary and retaining the fixed shot as a regression
   case.
2. Cover goal-centre and near-post targets, at least three lateral launch
   regions, several angles and calibrated arrival times.
3. Add contact-aware terminal outcomes so saves and bare timeouts are separated
   reliably.
4. Evaluate all three controls on the same 200 distinct shots.
5. Re-run this gate. Begin the Dreamer smoke run only if the report says `GO`.
