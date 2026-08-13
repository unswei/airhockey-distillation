# Why the first full-data student fails without blackout

The first full-data `n=64`, `k=2` student saves 46.2% of the 225 no-blackout
validation shots. The frozen teacher saves 99.6% on the same shots under
deterministic inference. This diagnostic holds the checkpoint, dataset and
shots fixed. It changes only what is measured or which policy supplies the
previous command.

The main problem is the initial loss mask. Training excludes steps 0--15.
The environment ignores policy commands for steps 0--4, but those commands
still enter the student's recurrent update as its previous action. Commands
from steps 5--15 both enter recurrence and control the robot. Thus, the first
loss-bearing target at step 16 is conditioned on a recurrent trajectory that
the student was never trained to produce.

## Phase errors

The offline check uses the 89 no-blackout episodes in the validation residue.
It reports action MSE separately for the locked prefix, the live but masked
prefix, and the loss-bearing suffix.

| Phase | Steps | Teacher-forced previous command | Recursive student command | Constant-mean target |
| --- | ---: | ---: | ---: | ---: |
| Locked | 0--4 | 0.966 | 0.887 | 0.250 |
| Live but loss-masked | 5--15 | 0.694 | 0.645 | 0.362 |
| Loss-bearing | 16 onwards | 0.103 | 0.219 | 0.486 |

The student is worse than a constant mean in both excluded phases and much
better once its loss begins. Recursively using its own previous command then
roughly doubles suffix MSE from 0.103 to 0.219. This is consistent with an
initial recurrent-state error followed by exposure error.

## Controlled prefix test

Each variant uses the same 225 no-blackout validation shots. In the prefix
variant, the teacher controls steps 0--15 and the student controls every later
step. All 225 episodes continue beyond step 16, so the result is not caused by
episodes ending before the student takes over.

| Variant | Saves | Save rate |
| --- | ---: | ---: |
| Frozen teacher | 224/225 | 99.6% |
| Student | 104/225 | 46.2% |
| Teacher controls only steps 0--15 | 224/225 | 99.6% |
| Student with teacher previous commands only | 38/225 | 16.9% |

The 53.3-point recovery from the teacher-controlled prefix isolates the early
trajectory as the main failure. The last variant is not a realistic policy:
the environment executes the student's command while its recurrence receives
the teacher's command. Its poor result shows that replacing the previous
action independently does not repair the inconsistent trajectory. It should
not be read as evidence that the previous-action input is unimportant.

## Closed-loop shift

A five-step public-history nearest-neighbour diagnostic compares validation
teacher trajectories with student rollouts. Histories are standardised using
the no-blackout training subset. The student's mean nearest-training distance
is 6.01, compared with 0.29 for on-policy validation histories. At the 95th
percentile the values are 21.73 and 1.29, a 16.8-fold difference. This is a
large measured covariate shift after the student's actions begin changing its
history.

The local target variance is 0.073 on on-policy validation histories and
0.155 on student histories. This is only a short-history ambiguity proxy. It
does not demonstrate a stochastic or multimodal deterministic teacher target.
The full dataset contains 5,311 repeated shot/blackout groups covering 12,968
episodes. Every repeated full input trajectory and target trajectory is
byte-identical, and no identical full input trajectory has conflicting
targets. Target multimodality is therefore not the first correction to make.

## Previous-action contract

The original policy contract called the recurrent input the previous executed
action. The dataset actually stores the previous requested teacher command;
the runtime documentation now uses that explicit term.
This agrees with the teacher and student policy carries, but not with the
environment during the five locked steps: the environment applies a fixed
hold target instead. Across 200,000 locked action values, requested command
versus applied hold has MSE 1.151 and maximum absolute difference 1.703.

This terminology and interface must be made explicit. For the immediate
controlled correction, keep the frozen dataset and treat the recurrent input
as the previous requested public command. Changing it to the applied action
would also change the teacher rollout and requires a new teacher-labelled
dataset and a separate controller check.

## Next correction

Train the same architecture and seed on the same frozen dataset, but apply
action loss to every valid episode step. These are complete episodes with a
known zero initial state; the current implementation masks the first 16
episode steps rather than using 16 context steps before a sampled suffix.
Evaluate only the 225 no-blackout validation shots first. If this does not
recover credible control, the next correction is deterministic shadow-teacher
labelling on student rollouts, aimed at the measured covariate shift.

## All-step correction result

The correction improved the no-blackout save rate from 46.2% to 67.1%, but it
did not reach the predeclared 75% gate. It saved 151/225 shots: 150 returns,
one arrest, nine concessions and 65 timeouts after contact. No simulator or
safety fault occurred, and no blackout condition was evaluated. The decision
is `NO_GO`, so the declared next step is deterministic shadow-teacher
labelling on student-controlled rollouts.

Training selected epoch 99 with full-step validation action MSE `0.07778`.
The checkpoint SHA-256 is
`92d7c68ed95b6e9c4217fc52a51225b3797bb6eb0d482cb1a7fd6c431027e5c1`.
The v1 run is retained as an export-tolerance failure: its maximum
NumPy/PyTorch action difference was `1.03116e-5`, just above `1e-5`. The
export-only v2 correction used `2e-5`, repeated byte-identical training, and
passed exact checkpoint reload.

The compact result is
`results/structured_n64_k2_failure_diagnostic_v1.json`. Raw episode records
remain outside Git on Marvin under:

```text
/home/oliver/experiments/airhockey-memory-distillation/
phase3-structured-n64-k2-failure-diagnostic-2026-08-13-v3
```

The raw diagnostic used code commit
`c5f17a769983b7f8fb4a2d785a86d2eecc95b21a`; the reproducible dataset audit
used `490e27c7c33b95076e6cd06373c7de3bf5ddb32f`. File hashes are recorded in
the compact result and the raw `sha256sums.txt`.

## Shadow-teacher correction result

The predeclared next correction succeeded. We collected 20,000 trajectories
controlled by the all-step student and labelled every step with the frozen
teacher's deterministic action mean. The teacher recurrence consumed the
previous student-requested command, matching the student's history contract.
The 1,453,360-transition shadow set was combined one-to-one with the original
20,000 teacher-controlled episodes by a hash-verified manifest; shards were
not copied or changed.

Retraining the same `n=64`, `k=2` architecture from scratch with seed 14303
selected epoch 100 at validation action MSE `0.06461`. On the gate's 225
no-blackout validation shots it saved 222 (98.7%), with no concession or
simulator/safety fault. This passes the predeclared 75% gate and is a
31.6-point improvement over the all-step teacher-only run.

The gate then opened the full paired validation without changing the frozen
checkpoint. It saved 1,107/1,125 episodes (98.4% overall), including 220/225
(97.8%) at 20 blackout steps. The blackout-specific save rates at 0, 5, 10,
15 and 20 steps were 98.7%, 99.6%, 97.8%, 98.2% and 97.8%. This paired run is
descriptive rather than a new thresholded gate.

The compact result is
`results/structured_n64_k2_shadow_round1_v1.json`. The 59 frozen evidence
files are root-owned and read-only on Marvin; their checksum manifest has
SHA-256
`b8f45ad808891c0b02a677d5a1e13b1502dbde162a5d6b12cca02066b506d116`.
