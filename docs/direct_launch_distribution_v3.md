# Observation-aliased direct-launch distribution v3

Status: predeclared, awaiting Marvin validation.

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
