# Direct-launch distribution v2

Validated: 2026-08-11

`direct_launch_v2` is the first direct-launch distribution to pass the
pre-training task-validity gate. It preserves all three lateral launch regions,
goal-centre and both near-post targets, but does not treat physically different
launch/target pairings as equally probable.

## Correction from v1

The balanced v1 distribution directed one third of shots through the neutral
mallet and gave equal weight to cross-table approaches that also intersected
it. Its inactive and fixed-centre concession rates were therefore only 41.7%
and 40.3%.

V2 makes three linked corrections:

- goal-centre shots remain present but are a minority;
- left and right launches usually target the matching near post, while every
  cross-table pairing remains represented;
- near-post target centres use the physically open corridor at
  `|y|=0.088--0.095` m, and the centre launch band is narrowed to
  `[-0.02, 0.02]` m.

The other lateral launch bands, launch x interval and calibrated timing range
are unchanged from v1.

## Calibration manifest

The calibration split has 216 distinct shots and 72 launches from each lateral
region. Its pair counts are:

| Launch | Near-post left | Goal centre | Near-post right | Total |
| --- | ---: | ---: | ---: | ---: |
| Left | 66 | 4 | 2 | 72 |
| Centre | 34 | 4 | 34 | 72 |
| Right | 2 | 4 | 66 | 72 |
| Total | 102 | 12 | 102 | 216 |

The train split has 900 shots, with pair rows `285/10/5`, `145/10/145`
and `5/10/285`. Validation and test each have 225 independently seeded shots,
with rows `68/5/2`, `35/5/35` and `2/5/68`. The test split remains reserved for
final evaluation.

The frozen calibration manifest SHA-256 is
`6e2b61f71135c23c2c1d459c8d90c65e8a153cc645986103d95c67b2e0c2733c`.
Nominal speeds span 1.53--3.54 m/s and approach angles span -15.4--15.2
degrees.

## Marvin validation

All 216 shots were simulated in the pinned Marvin image:

- 216/216 reached the defender approach plane;
- realised approach times spanned 0.42--0.94 seconds;
- all arrivals lay inside the requested 0.4--1.0 second range;
- no non-finite state or simulator fault occurred.

The full calibration report is outside Git at:

```text
/home/oliver/experiments/airhockey-memory-distillation/direct-launch-v2-calibration-2026-08-11-v2/calibration.json
```

Report SHA-256:
`e48bf07c08b565211bfcd5da3cb46f5f6aec8e89ec036bf99a40b69532a8d0db`.

The paired gate returned `GO`: inactive and fixed-centre each conceded 200/216
shots (92.6%), while the privileged controller saved 216/216. Exact replay,
observation isolation and the 20-episode reliability check also passed. See
[`teacher_training_gate.md`](teacher_training_gate.md) for the complete gate
record.

## Reproduction

The validated v2 configuration is now the default for both commands:

```bash
python3 scripts/validate_direct_launch_distribution.py \
  --split calibration \
  --simulate \
  --output /run-output/calibration.json

python3 scripts/run_teacher_gate.py \
  --output /run-output/teacher_gate.json
```

V1 remains immutable historical evidence. Direct launch remains a development
and regression mode; scripted physical strikes will need their own calibrated,
versioned distribution.
