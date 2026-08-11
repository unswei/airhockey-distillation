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
| Privileged controller | Explicit arrest, return or safe deflection on at least 0.80 of at least 200 distinct shots |
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
| Paired coverage | All controls used the same 216 `direct_launch_v1` shots | `PASS` |
| Inactive concession | 90/216, 41.7% | `FAIL` |
| Fixed-centre concession | 87/216, 40.3% | `FAIL` |
| Privileged save | 216/216, 100% | `PASS` |
| Exact replay | All 20 stability trajectories matched | `PASS` |
| Public observation isolation | Puck velocity invariant; no opponent component | `PASS` |
| Marvin simulator reliability | 20/20 episodes, zero faults | `PASS` |

The full contact-aware test suite also passed in the pinned Marvin image: 32
tests. The headless run emits a GLFW warning because `DISPLAY` is unset; no
rendering is requested and this did not affect the simulation.

The privileged controller produced 205 returns and 11 arrests, with no
timeouts or concessions. It already exceeds the required save rate. The
remaining failure is task difficulty: inactive and fixed-centre mallets return
many trajectories because v1 weights goal-centre and near-post targets
equally. Do not weaken the gate; preserve v1 and recalibrate a versioned v2
distribution away from the neutral mallet while retaining centre coverage.

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
/home/oliver/experiments/airhockey-memory-distillation/teacher-gate-2026-08-11-v3/teacher_gate.json
```

SHA-256:
`a56f3499cee1ffa484fe985e55e6c55737b596b1f1bc7c1ea49437d67af5b5ab`.

## Work required for GO

1. Create `direct_launch_v2` with fewer centre-target shots and outward-shifted
   near-post ranges, without modifying the preserved v1 manifest.
2. Re-run all controls on the paired v2 calibration manifest.
3. Begin the Dreamer smoke run only if the report says `GO`.
