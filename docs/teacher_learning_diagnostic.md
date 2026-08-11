# Short teacher learning diagnostic

Validated: 2026-08-11

Decision: **no held-out learning signal; do not start the main teacher run**

## Protocol

The diagnostic used code commit
`c07d270dbe7acb30c22fb6b7dfa6ce57c88a2a48` and the audited Marvin image.
One `size1m` DreamerV3 policy was trained for 20,000 requested environment
steps on the `direct_launch_v2` train split with `defend_shot_v1`. Blackout
length was sampled uniformly from 0 to 20 steps and began at public
observation step 5.

The run completed 19,992 logged environment steps in 138.76 seconds. It
recorded 383 completed episodes, 96 training-metric records and a final
logged optimiser-update mean of 609.5. The retained checkpoint was from step
18,820.

Evaluation compared the checkpoint with its identically seeded untrained
initialisation. Both policies received the same 30 evenly selected validation
shots at fixed blackout lengths 0, 10 and 20, giving 90 paired cases per
policy. Recurrent state was reset at every episode boundary and evaluation
used deterministic actions. The final test split was not used.

## Held-out result

| Blackout steps | Untrained saves | Trained saves | Change |
| ---: | ---: | ---: | ---: |
| 0 | 8/30 (26.7%) | 6/30 (20.0%) | -6.7 points |
| 10 | 11/30 (36.7%) | 4/30 (13.3%) | -23.3 points |
| 20 | 5/30 (16.7%) | 10/30 (33.3%) | +16.7 points |
| **All** | **24/90 (26.7%)** | **20/90 (22.2%)** | **-4.4 points** |

Mean episode return changed from -0.329 to -0.362. The trained policy
conceded 59 shots, compared with 60 for the untrained policy, but converted
more cases into unresolved post-contact timeouts rather than genuine saves.
A timeout remains separate from a save.

The paired save transitions were 4 saved by both policies, 20 lost only after
training, 16 gained only after training and 50 saved by neither. An exact
two-sided McNemar test gives `p = 0.618`; this small run provides no evidence
that training improved save probability. Paired episode return improved in
20 cases, was unchanged in 47 and worsened in 23.

## What did learn

The optimisation path itself was live and numerically stable. No numeric
metric was non-finite. From the first to last training record, dynamics loss
fell from 6.11 to 1.02, observation reconstruction loss from 11.40 to 0.72,
and reward-model loss from 5.54 to 4.19. The mean online episode return moved
from -0.342 over the first 100 episodes to -0.244 over the last 100.

That is evidence that replay, the world model and the reward channel are
working. It is not evidence of a useful control policy. The reward model was
still improving at the end, the value loss rose from 1.33 to 6.69, and the
held-out policy result did not improve. The most conservative interpretation
is that 20,000 steps are insufficient for this sparse, delayed control task;
one seed cannot distinguish that from unstable or poorly generalising policy
learning.

## Decision and next correction

Do not spend the main teacher budget yet, and do not change the reward from
this result alone. The next bounded experiment should retain checkpoints and
run seed-matched validation at fixed intervals during a medium pilot. A useful
next protocol is three seeds, at most 100,000 steps each, evaluated every
20,000 steps on the same validation design. This will show whether learning is
merely late, peaks and collapses, or remains absent. If all three seeds remain
at the untrained level, the next diagnostic should isolate the no-blackout
task before introducing any additional reward shaping.

## Evidence

The immutable run is outside Git at:

```text
/home/oliver/experiments/airhockey-memory-distillation/teacher-learning-diagnostic-2026-08-11-v2
```

The directory occupies 176 MB.

| File | SHA-256 |
| --- | --- |
| `untrained_evaluation.json` | `905a77aac8a7e67a79a27b50f03f8ac86d54de755fde915c90e66a5038254087` |
| `trained_evaluation.json` | `676d2071aa84654d70315d19586fe8a6ab037de5d68d20a4ae8f8e220e379ed` |
| `training/diagnostic_result.json` | `6963f43fb4c7cca727a98434e43491a8c280624fac3058e1fa800c5dfceee91a` |
| `training/dreamer/metrics.jsonl` | `0c720b113a1a1a579c0a66dc2461b51013ffad0015ccb280a6f73e59e1a79b7d` |
| Retained `agent.pkl` | `a9091d5f1393fc768b20452eb6d863b7c93d60c446e5de504bc29713c448dfdb` |
| Training console log | `487bfdd4a1f1cf6851424b49fc2d605133f006835a148ff7f4aa6035dec7c4b1` |
