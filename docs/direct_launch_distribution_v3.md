# Observation-aliased direct-launch distribution v3

Status: task, alias and teacher-readiness gates passed on Marvin; Stage B v2
returned `NO_GO` because the feed-forward baseline was not credible without
blackout.

`direct_launch_v3` corrects the ambiguity weakness found by the first Stage B
experiment. The v2 feed-forward policy saved 84% without blackout and 62% at a
20-step blackout. Memory helped, but the feed-forward policy could still infer
much of the defence from deterministic position and robot-state correlations.

## Task correction

V3 adds paired rendezvous families. Two shots in each family:

- start from the centre launch region;
- have different lateral velocities;
- nominally reach the same puck position after five visible policy steps;
- then travel to opposite near posts.

The blackout begins at that rendezvous. The target separation is at least
0.176 m. A stateful policy can infer motion from the visible prefix, while the
instantaneous hidden observation does not contain puck position or velocity.

Policy actions are locked for the five-step visible prefix. The environment
commands the current mallet position during that interval and releases control
at the first hidden observation. This prevents a feed-forward policy from using
its earlier commands and resulting robot motion as an external memory channel.
The public observation and two-dimensional action interfaces are unchanged.

## Versioned splits

The calibration split contains 216 shots:

- 90 alias families, or 180 paired near-post shots;
- 14 left-to-left and 14 right-to-right support shots;
- four left and four right launch support shots aimed at goal centre.

The train split contains 900 shots. Validation and test contain 225 shots each,
with 180 alias shots in 90 families. The test seed remains unused. The frozen
calibration manifest SHA-256 before simulation is
`359b093d97d213d52764ad553b9c81478296d8600a9edd7379edc005bcd88c9a`.

## Predeclared gates

The alias audit must pass all of the following on the calibration split:

- at least 90 complete alias families;
- at most 0.01 m separation at the last visible puck position;
- at most `1e-6` maximum difference in the paired first-hidden public
  observations under the locked prefix;
- at least 0.25 normalised action distance between privileged intercepts.

The ordinary task-validity gate remains unchanged: at least 200 distinct shots,
at least 80% concessions for inactive and fixed-centre defenders, at least 80%
saves for the privileged controller, exact replay, clean observations and no
Marvin simulator faults.

Before collecting another student dataset, the teacher must save at least 85%
without blackout, 80% overall and 70% at 20-step blackout over all 225
validation shots at each of five blackout lengths. The existing frozen teacher
is tested first. Failure means retraining on v3; thresholds will not be changed.

Stage B v2 retains the original feed-forward memory gate thresholds but uses
all 225 validation shots. It also includes a causal ablation that resets the
teacher recurrent state exactly when blackout begins. At 20 steps, that reset
must reduce paired save rate by at least 10 points with a positive bootstrap
95% lower bound. The zero-blackout control must remain identical.

Configurations are frozen in `configs/env/direct_launch_v3.yaml`,
`configs/teacher/dreamerv3_v3.yaml` and
`configs/student/feed_forward_stage_b_v2.yaml` before Marvin results are read.

## Marvin validation

The pinned container validated all 216 calibration shots:

- 216/216 reached the defender approach plane;
- realised approach times span 0.40--0.94 seconds;
- no simulator or non-finite-state fault occurred;
- calibration report SHA-256 is
  `ac699b2be829d6de491ccec82b0dfc2b4b08415238c8f12f97c49549b391511e`.

All alias checks passed. The maximum last-visible paired separation is 8.10 mm
and first-hidden public observations are bit-identical. Actual first-hidden puck
positions remain within 0.94 mm, while privileged action distance is at least
0.315. Alias report SHA-256 is
`48fc4ad19b20403996b00959cc7dd39ef97fa237e7176b4bb6c07c2c17cbb468`.

The ordinary task-validity gate also returned `GO`. Inactive and fixed-centre
defenders conceded 208/216 and 209/216 shots. The privileged controller saved
214/216. Replay, observation isolation and 20 fault-free reliability episodes
passed. Gate report SHA-256 is
`34701715810b3b9164f778d98a15f73b7aa64aac38afbc2c5ee73dc987237587`.

The frozen v2 teacher did not pass v3 readiness. Its save rates were 75.1%
without blackout, 58.0% overall and 45.3% at 20 steps, against thresholds of
85%, 80% and 70%. This is expected distribution shift, so a new v3 teacher is
being trained rather than weakening the thresholds. The probe and readiness
reports have SHA-256 hashes
`1ddfa8f84e855e8f28a54b887d922899be6973788a742ae8becee47a53192476`
and `ac7f88149da445da8e647fd48d0ca09187aaa5b6c07ca41cba7bc9a9b4f0b346`.

A fresh 1,000-step v3 training smoke passed with exact checkpoints at 0, 500
and 1,000 and non-zero returns spanning -1.0--1.2. Full run
`teacher-full-v3-2026-08-12-v1` completed from commit `66fbc4c`. Validation
selected step 700,000, which saved 98.9% over 1,125 cases and 98.2% at the
20-step blackout.

The paired Stage B v2 comparison did not yet establish that blackout memory
causes the teacher advantage. The feed-forward baseline saved 60.4% at 20
steps, compared with 98.2% for the teacher, but it saved only 56.0% without
blackout. The baseline therefore failed the predeclared visible-performance
and comparability checks. Full results are in
`docs/stage_b_memory_validation_v2.md`.
