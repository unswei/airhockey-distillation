# Minimal direct-launch task slice

Implemented: 2026-08-11

## Scope

`DefendShotTrackingLoss` currently contains one deterministic incoming shot,
one deterministic blackout and one defending robot. It uses the pinned
upstream single-robot `defend` environment and normal MuJoCo dynamics. The
launch overrides the upstream random reset before simulation begins; it does
not teleport the puck during an episode.

Scripted opponent strikes and a shot distribution are deliberately deferred.
This slice exists to test the observation boundary, action adapter, replay and
basic task validity before adding those sources of variation.

## Fixed episode

| Quantity | Value |
| --- | --- |
| Shot ID | `near_post_right_v1` |
| Initial table position | `(0.55, 0.11)` m |
| Initial table velocity | `(-1.6, 0.0)` m/s |
| Control rate | 50 Hz |
| Timeout | 125 steps / 2.5 s |
| Blackout observations | steps 5--14 inclusive |

Observation step zero is returned by `reset`. The blackout is the half-open
interval `[5, 15)`, giving five visible observations before tracking is lost.
The same shot and blackout are restored by every reset.

## Policy boundary

Every learned policy receives 19 values in this order:

| Component | Dimension |
| --- | ---: |
| Joint position | 7 |
| Joint velocity | 7 |
| Planar mallet position | 2 |
| Visible planar puck position | 2 |
| `puck_visible` | 1 |

Puck position is set to zero during blackout. Puck velocity is removed rather
than masked and cannot affect the public observation. Simulator state is not
returned in public `info`; evaluation code must cross the explicit
`privileged_state()` boundary.

Policies produce a two-dimensional normalised mallet target. The action
adapter appends fixed normalised zeros for stiffness and damping. Under the
upstream ranges, these are physical stiffness `(13.5, 13.5)` and damping
`(0.25, 0.25)`. The existing clipping, action filtering, inverse kinematics
and joint controller remain upstream.

## Controls

- `inactive` commands the currently observed mallet position.
- `fixed_centre` commands the neutral physical target `(0.65, 0.0)` in the
  robot frame.
- `privileged_intercept` projects the true puck trajectory to a fixed
  defensive x-coordinate. It is a task-validity control, not a policy
  baseline, because it reads simulator puck position and velocity.

On the audited Marvin image, the current fixed shot produced:

| Control | Steps | Outcome |
| --- | ---: | --- |
| Inactive | 75 | Goal conceded |
| Fixed centre | 76 | Goal conceded |
| Privileged intercept | 125 | Puck returned; no concession |

The privileged run reaches the episode timeout with the puck travelling back
towards the opponent. These three trajectories only establish that the fixed
shot is non-trivial and controllable. They do not establish performance over
a distribution.

## Running it on Marvin

```bash
run_dir=/home/oliver/experiments/airhockey-memory-distillation/minimal-task-slice
mkdir -p "${run_dir}"

docker run --rm --gpus all --ipc host \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --volume /home/oliver/Code/airhockey-memory-distillation:/work:ro \
  --volume "${run_dir}:/run-output:rw" \
  --workdir /work \
  marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt \
  python3 scripts/run_minimal_task.py \
  --output /run-output/baselines.json
```

The script runs all three controls and replays the fixed-centre trajectory a
second time. It fails if the two public trajectory hashes differ.

The first frozen validation report is stored outside Git at
`/home/oliver/experiments/airhockey-memory-distillation/minimal-task-slice-2026-08-11-v2/baselines.json`
with SHA-256
`fac360cc148995d2f84aa6eba554edc6d4228974c7abaa1405f8cdc1e07a266e`.

## Tests and next limit

Pure tests cover the exact blackout interval, position masking, removal of
puck velocity, public-info leakage, action expansion and deterministic replay.
The Marvin integration test resets the real simulator at the requested puck
state and confirms an identical 20-step public trajectory after replay.

The next task step is a scripted physical strike followed by calibration of a
versioned train/validation/test shot distribution. The fixed shot should
remain as a fast regression test.
