# Stage B v2 memory validation

Stage B v2 compared the validation-selected 700,000-step DreamerV3 teacher
with the predeclared observation-only feed-forward policy on the same 225 v3
validation shots at five blackout lengths. The result is `NO_GO` for the
memory-required claim.

## Frozen teacher

Validation selected step 700,000 from the one-million-step v3 run. The frozen
bundle is read-only on Marvin at
`frozen-teachers/dreamerv3-teacher-v3-step-700000`. The inference payload
`agent.pkl` has SHA-256
`6a672d5b6d7c2b9ca2335f1a85b69280ca7db58deb5e2f56d2ba033ebf636254`.
Its manifest has SHA-256
`b269950b0f10b32c7b411ece890b349d5c1280cc95f6d4864818db4c3ac72c15`.

The selected teacher passed the frozen readiness checks. Across 1,125
validation episodes, it saved 98.9% and conceded no shots. Its no-blackout and
20-step save rates were 99.1% and 98.2%.

## Feed-forward baseline

The fixed dataset contains 20,000 complete teacher episodes and 732,168
transitions. Training used only the 19-dimensional public observation and the
teacher action executed through the public adapter. The dataset retains
privileged puck state only for diagnostics; it is not loaded as a student
input.

The 5,602-parameter MLP selected epoch 99 by validation action MSE. Its
validation MSE was 0.368 overall, 0.360 on visible frames and 0.391 on hidden
frames. The framework-neutral checkpoint reproduces the trained PyTorch
network within `2.12e-6` maximum absolute action error.

## Paired closed-loop result

| Blackout steps | Teacher save rate | Feed-forward save rate | Paired teacher advantage |
| ---: | ---: | ---: | ---: |
| 0 | 99.1% | 56.0% | 43.1 points |
| 5 | 99.6% | 54.2% | 45.3 points |
| 10 | 98.7% | 58.7% | 40.0 points |
| 15 | 99.1% | 63.6% | 35.6 points |
| 20 | 98.2% | 60.4% | 37.8 points |

At 20 steps, the paired teacher advantage is 37.8 percentage points with a
bootstrap 95% interval of [31.6, 44.4]. This establishes a performance gap,
but it does not isolate memory as the cause. The feed-forward policy saved only
56.0% with no blackout, below the predeclared 75% credibility threshold, and
the teacher already led it by 43.1 points in that condition. The advantage
therefore shrank by 5.3 points rather than growing by the required 10 points as
blackout length increased.

Three checks block progression: credible no-blackout feed-forward performance,
comparable no-blackout performance, and advantage growth with blackout. The
causal Dreamer-state ablation was not used to override this failure. It remains
a separate confirmatory check after a credible visible-observation baseline is
available.

The machine-readable result and artefact hashes are in
`results/stage_b_memory_validation_v2.json`. Large checkpoints, datasets and
raw episode results remain outside Git under Marvin's experiment root.
