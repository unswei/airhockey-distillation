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
