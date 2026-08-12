# Stage B v3 visible-observation baseline correction

Status: this procedure was predeclared before qualification or test results
were read. It is now complete; the preserved outcome is documented in
[`stage_b_memory_validation_v3.md`](stage_b_memory_validation_v3.md).

The protocol below is retained as the preregistered correction.

Stage B v2 did not isolate a memory effect because its feed-forward policy
saved only 56.0% of no-blackout shots. That policy was trained by regressing
the recurrent teacher's actions. At observation-aliased states, the same
instantaneous public observation can correspond to different teacher actions.
More regression epochs cannot remove this target conflict, and offline
imitation also leaves closed-loop covariate shift uncorrected.

The v3 correction instead trains a memoryless policy directly on the defence
reward with PPO. This asks for the strongest action that can be chosen from the
current public observation, rather than the best average of incompatible
teacher actions. The baseline still receives exactly 19 public values and
emits the same two-dimensional action. It has no previous action, observation
stack, recurrent state, puck velocity or privileged state.

## Fixed training procedure

The actor and value networks each have two 256-unit tanh hidden layers. PPO
uses one million environment steps, eight serial vector environments, 256
steps per rollout and 10 optimisation epochs. The remaining optimiser settings
are frozen in `configs/student/feed_forward_stage_b_v3.yaml`.

Training samples only blackout lengths 0, 5, 10, 15 and 20. Half of the
episodes have no blackout; each other length has probability 0.125. This
deliberately favours the visible condition and makes the later memory test more
conservative. The policy receives no indication of the sampled length.

Three fixed seeds, 14303--14305, are trained. Candidate selection uses only the
225-shot validation split with no blackout. It selects highest save rate, then
highest mean score, then the lower seed.

## Qualification before confirmation

The selected policy qualifies only if:

- its no-blackout save rate is at least 75%;
- the paired teacher advantage without blackout is at most 10 percentage
  points;
- no simulator fault occurs.

The second requirement is stricter for the current 99.1% teacher and therefore
requires the baseline to save at least 89.1% on the paired validation shots.

The `direct_launch_v3` test split remains unopened unless all qualification
checks pass. If qualification returns `GO`, the selected seed and frozen
teacher are evaluated once on the 225 test shots at blackout lengths 0, 5, 10,
15 and 20. The existing memory thresholds are unchanged: credible and
comparable no-blackout performance, at least a 15-point teacher advantage at
20 steps, at least 10 points of advantage growth, and a positive bootstrap 95%
lower bound at 20 steps.

`scripts/run_stage_b_v3_baseline_on_marvin.sh` enforces this sequence. A failed
qualification stops before either policy is evaluated on the test split.
