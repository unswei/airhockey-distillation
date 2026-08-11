# Direct-launch distribution v1

Validated: 2026-08-11

`direct_launch_v1` is a deterministic, versioned distribution for developing
and validating the set-piece task before scripted opponent strikes are added.
It initialises puck position and velocity at reset; it does not simulate an
opponent strike.

## Construction

The generator is stratified across three lateral launch regions and three goal
target regions:

| Kind | Region | Table-frame y range (m) |
| --- | --- | ---: |
| Launch | Left | `[-0.36, -0.16]` |
| Launch | Centre | `[-0.10, 0.10]` |
| Launch | Right | `[0.16, 0.36]` |
| Target | Near post, left | `[-0.11, -0.08]` |
| Target | Goal centre | `[-0.025, 0.025]` |
| Target | Near post, right | `[0.08, 0.11]` |

Launch x is sampled in `[0.45, 0.65]` m. Each launch/target pairing uses a
four-dimensional Latin hypercube over launch x, launch y, target y and
ballistic approach time. Velocity is solved so that the puck reaches the
defender approach plane at `x=-0.65` m after the sampled time and its
straight-line continuation reaches the selected point on the goal plane at
`x=-0.974` m.

The ballistic interval is `[0.36, 0.75]` seconds. It was calibrated against
MuJoCo friction to produce a realised approach interval of approximately
`[0.4, 1.0]` seconds.

## Splits

| Split | Seed | Shots per region pair | Total | Use |
| --- | ---: | ---: | ---: | --- |
| Calibration | 1101 | 24 | 216 | Task tuning and pre-training gate |
| Train | 2101 | 100 | 900 | Teacher training |
| Validation | 3101 | 25 | 225 | Model selection |
| Test | 4101 | 25 | 225 | Final evaluation only |

The seeds and continuous samples differ between splits. The test split must
not be used to tune the task, controller or reward.

The calibration manifest contains 216 distinct shot IDs, exactly 24 from each
of the nine region pairings. Its canonical manifest SHA-256 is
`82986091f72a3e51cde803ed73527e060bb0d0a89fda6d08e82eb810abf8724e`.
Nominal speeds span 1.55--3.42 m/s and approach angles span -16.7--17.3
degrees relative to the incoming table x-axis.

## Marvin validation

All 216 calibration shots were simulated in the pinned Marvin image with the
inactive controller:

- 216/216 reached the defender approach plane;
- realised approach times spanned 0.40--0.94 seconds;
- 216/216 lay inside the requested 0.4--1.0 second interval;
- no non-finite state or simulator fault occurred.

The full report is outside Git at:

```text
/home/oliver/experiments/airhockey-memory-distillation/direct-launch-v1-calibration-2026-08-11-v3/calibration.json
```

Report SHA-256:
`48f08727395073a1e73a7ded4954e1b5051580f00b9e56f3568da13ac4525065`.

Reproduce the manifest and MuJoCo timing sweep with:

```bash
python3 scripts/validate_direct_launch_distribution.py \
  --config configs/env/direct_launch_v1.yaml \
  --split calibration \
  --simulate \
  --output /run-output/calibration.json
```

## Limits

This validates deterministic coverage, geometry, timing and simulator
stability. The subsequent contact-aware gate showed that the privileged
controller saves 100% of v1, but inactive and fixed-centre concession rates are
only 41.7% and 40.3%. The balanced target mixture therefore sends too many
shots through the neutral mallet. V1 is preserved as evidence and is no longer
the default. The corrected
[`direct_launch_v2`](direct_launch_distribution_v2.md) retains every required
region and passes the gate. Direct launch also remains a development mode;
physical realism must be checked again when scripted strikes are introduced.
