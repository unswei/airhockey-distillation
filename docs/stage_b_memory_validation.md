# Stage B memory validation

Stage B compared the validation-selected 720,000-step DreamerV3 teacher with a
small observation-only feed-forward policy on exact paired blackout episodes.
The result is `NO_GO` for the strong claim that the current task has been shown
to require memory. The effect is in the expected direction and statistically
positive at 20 blackout steps, but it does not meet the predeclared effect-size
criteria.

## Frozen teacher

The selected checkpoint was copied into a read-only inference bundle on Marvin:

```text
/home/oliver/experiments/airhockey-memory-distillation/
  frozen-teachers/dreamerv3-teacher-v1-step-720000
```

The `agent.pkl` payload is 7,955,303 bytes and has SHA-256
`aafa486922860eb6a4a1d046667b0f1422170f4b242fa7eaa5c5cbf93782c94e`.
The frozen manifest has SHA-256
`7d74c398749c2d4cd65dafba8e4250765381489185f9b9008307bc26091de054`.
It records the selected step, validation summary, training and freezing commits,
teacher configuration hash and source-selection hash.

## Feed-forward baseline

The teacher generated 20,000 deterministic training episodes with uniformly
sampled 0--20-step blackouts. The completed 40-shard dataset contains 870,634
transitions. Its manifest has SHA-256
`788ca4939be22c3dd72edb9bb7b6e440921394ecda2f750385325fa166bec5a2`.

The first collection attempt revealed that Dreamer can emit raw actions outside
the public `[-1, 1]` range. The environment had always clipped those values
before execution. The invalid preliminary shards were quarantined, and schema 2
stores both the raw diagnostic value and the clipped action actually executed.
The trainer accepts only finite schema-2 executed actions within the public
range.

The baseline has no recurrent state, observation stack or previous-action
input. Its network is `19 -> 64 -> 32 -> 64 -> 2`, with SiLU hidden activations
and a `tanh` output. It has 5,602 parameters. Training used 698,540 transitions,
selected epoch 99 on 87,167 validation transitions, and did not use the 84,927
test transitions for selection. Validation action MSE was 0.2828 overall,
0.2665 while the puck was visible and 0.3406 while it was hidden.

The exported checkpoint has SHA-256
`2a473a917c89859f1f442d10b8a87ece98a2214d16d694ee879ef34b7fb9f5a4`.
Framework-neutral NumPy inference matched PyTorch within `1.70e-6` absolute
error on the export check.

## Paired result

Each policy ran the same 100 validation shots at each fixed blackout length.
A save is a return, arrest or safe deflection; a timeout after contact is not
counted as a save.

| Blackout steps | Teacher saves | Feed-forward saves | Paired teacher advantage | Paired bootstrap 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 88/100 | 84/100 | 4 points | [-5, 13] |
| 5 | 89/100 | 80/100 | 9 points | [0, 18] |
| 10 | 82/100 | 75/100 | 7 points | [-4, 18] |
| 15 | 82/100 | 67/100 | 15 points | [5, 25] |
| 20 | 74/100 | 62/100 | 12 points | [1, 23] |

The feed-forward checkpoint is a credible visible-observation baseline: its
84% no-blackout save rate exceeds the required 75%, and it is only four points
behind the teacher there. The 20-step difference is positive under the paired
bootstrap. However, the predeclared gate required a teacher advantage of at
least 15 points at 20 steps and growth of at least 10 points from zero to 20
steps. The observed values were 12 and 8 points. Both checks failed, so changing
the thresholds after seeing the results would be invalid.

The feed-forward result and gate reports have SHA-256 hashes
`6ab18b60e2399d71b32f68f8da8c3377612795396032af9172b41cebed33f21e`
and `d180c770dd5776d2d731ed39d1a3b147c56397049708f9684651c1bdb542778e`.
Raw artefacts remain outside Git under
`/home/oliver/experiments/airhockey-memory-distillation` on Marvin.

## Interpretation and next correction

The current result is evidence that memory helps, especially at 15--20 steps,
but not yet that the task distribution requires it. A feed-forward policy can
still recover much of the defence from the instantaneous puck position and
robot state. The current deterministic shot families may therefore make hidden
motion too predictable from a single observation.

The next correction should be a separately versioned, predeclared Stage B
experiment rather than a reinterpretation of this one. It should introduce
observation-aliased shot families: different approach velocities and targets
that pass through closely matched visible puck positions but demand different
actions during blackout. Teacher readiness must be re-established on that
distribution. Evaluation should use all available validation shots and include
a causal Dreamer-state ablation at blackout onset alongside the feed-forward
baseline. This would distinguish a genuine information requirement from model
capacity or behaviour-cloning error.
