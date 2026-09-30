# Post hoc principal test subgroup analysis

This analysis separates observation-alias shots from support shots and checks
left/right target performance in the existing principal test. It supports the
memory result, but also exposes a ceiling: every structured recurrent model
saves every alias shot at all six tested blackout durations. Differences among
those models occur entirely on the smaller support subset. This is additional
description of one task, not evidence from new tasks or new evaluations.

No policies were run, checkpoints selected or principal results changed.
The paper reports a short post-hoc breakdown and qualifies its recurrent
complexity, extrapolation and finite-history interpretations accordingly.

## Data and scope

All 48,600 frozen episode rows are retained: seven student families, five
matched training seeds, one frozen teacher, and six blackout durations from
0 to 500 ms. At each duration, each policy encounters the same 225 shots:

| Subgroup | Unique shots | Resampling units |
| --- | ---: | ---: |
| Observation alias | 180 | 90 paired families |
| Support | 45 | 45 individual shots |
| Left target, all shots | 110 | 90 alias families and 20 support shots |
| Right target, all shots | 110 | 90 alias families and 20 support shots |
| Centre target | 5 | 5 support shots |
| Left target, alias only | 90 | The same 90 alias families |
| Right target, alias only | 90 | The same 90 alias families |

Student rates average the five seeds. Thus each alias percentage uses 900
episode rows per family and duration, but only 180 unique shots, not 900
independent shots. The teacher has no training-seed replication.

## Save rates

Values are percentages. The full result file includes all six durations,
per-seed rates, subgroup intervals and paired contrasts; this table focuses
on the principal 400 ms comparison and existing 500 ms extrapolation.

| Policy | Alias 400 ms | Support 400 ms | Alias 500 ms | Support 500 ms |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 100.0 | 91.1 | 100.0 | 82.2 |
| Feed-forward | 50.9 | 18.2 | 49.9 | 10.7 |
| Ten-step stack | 94.6 | 51.6 | 96.3 | 42.7 |
| Structured k=0 | 100.0 | 91.6 | 100.0 | 71.1 |
| Structured k=1 | 100.0 | 89.3 | 100.0 | 79.6 |
| Structured k=2 | 100.0 | 93.8 | 100.0 | 77.8 |
| Structured k=4 | 100.0 | 90.7 | 100.0 | 70.2 |
| GRU-64 | 99.3 | 91.6 | 99.2 | 73.8 |

At zero blackout, the teacher, all structured models and GRU-64 save every
alias shot; the feed-forward model saves 98.3%. Its decline to 50.9% at
400 ms therefore occurs on shots it can usually defend while tracking is
available. This reinforces the memory interpretation on the deliberately
aliased cases, rather than attributing the result solely to support shots.

At 400 ms, k=0 and GRU-64 have identical mean support save rates. At 500 ms,
k=0 falls to 71.1% on support shots, compared with 73.8% for GRU-64 and 82.2%
for the teacher. The corresponding k=0 differences are -2.7 percentage
points with a pointwise 95% interval of [-15.6, 10.7] against GRU-64, and
-11.1 points [-23.6, 0.9] against the teacher. These broad intervals do not
establish equivalence or rule out a practically important deficit.

Ranks 1 and 2 have higher observed support rates than k=0 at 500 ms:
differences are +8.4 points [0.0, 18.7] and +6.7 points [-3.1, 16.9]. Rank 4
does not improve the observed rate. This is not a monotonic rank benefit,
but the subgroup is too small to conclude that recurrent complexity never
helps. In particular, an interval touching zero is not proof of no effect.

## Paired comparisons at 400 ms

Differences are percentage points, with exploratory pointwise 95% intervals.
They are not adjusted for multiple comparisons.

| Contrast | Alias shots | Support shots |
| --- | ---: | ---: |
| k=0 minus feed-forward | +49.1 [34.3, 62.0] | +73.3 [62.7, 82.7] |
| k=0 minus ten-step stack | +5.4 [0.2, 13.2] | +40.0 [25.3, 54.2] |
| k=0 minus GRU-64 | +0.7 [0.0, 2.4] | 0.0 [-6.2, 7.1] |

The fixed stack retains high observed alias performance even beyond its
approximately 200 ms observation window. Its larger deficit is on support
shots. These results do not establish whether current robot state, earlier
actions or other correlations explain the stack's performance after visible
samples leave its window.

## Left and right targets

The alias-only comparison holds the paired shot family fixed and avoids
mixing different support-shot geometries. Values are save percentages.

| Policy | Alias left 400 ms | Alias right 400 ms | Alias left 500 ms | Alias right 500 ms |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 100.0 | 100.0 | 100.0 | 100.0 |
| Feed-forward | 55.6 | 46.2 | 54.7 | 45.1 |
| Ten-step stack | 89.1 | 100.0 | 92.7 | 100.0 |
| Structured k=0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Structured k=1 | 100.0 | 100.0 | 100.0 | 100.0 |
| Structured k=2 | 100.0 | 100.0 | 100.0 | 100.0 |
| Structured k=4 | 100.0 | 100.0 | 100.0 | 100.0 |
| GRU-64 | 98.7 | 100.0 | 98.4 | 100.0 |

There is no observed one-sided alias failure hidden by k=0's average.
The stack's alias failures are confined to left targets: its left-minus-right
difference at 400 ms is -10.9 points [-26.4, -0.4]. This is an exploratory
asymmetry, not a multiplicity-controlled discovery or an established cause.

Across all left/right targets, including support shots, k=0 saves 98.4%/99.5%
at 400 ms and 94.0%/95.1% at 500 ms. The full results also retain the
five-shot centre-target subgroup; it is too small for a robustness claim.

## Uncertainty and validation

The analysis uses 10,000 paired hierarchical bootstrap replicates, seed
9302026. The five training-seed indices are resampled jointly across student
families. Evaluation units are resampled within four fixed strata: 90 alias
families, 20 left-target support shots, 20 right-target support shots and five
centre-target support shots. Both members of each sampled alias family stay
together. The same draws are used across models, subgroups and horizons.
The teacher uses only the shared shot-unit draws, without artificial
training-seed replication. Rates remain shot-weighted, not equally weighted
averages of subgroup rates.

These intervals condition on the observed subgroup composition. The added
stratification differs from the original principal bootstrap; it does not
replace its published intervals. No adjustment for the many subgroup
comparisons is made, so interpretation is descriptive and exploratory.

Failure-free subgroups yield degenerate empirical bootstrap intervals of
[100, 100]. That reflects the observed sample, not certainty about unseen
shots, new training seeds or deployment performance.

Input checks reject duplicate/missing shot-duration rows, incomplete alias
pairs, metadata mismatches and inconsistent schedule/protocol hashes. All
216 policy-seed/horizon rates reproduce the frozen statistics, and weighted
alias/support rates reconstruct every overall rate. Source file hashes and
the analysis-script hash are recorded in the result file.

## Reproduction

From the code repository, with the existing frozen test directory available:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.analyse_principal_subgroups \
  --test-root /path/to/frozen/test \
  --output /path/to/new/subgroups.json

PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_principal_subgroups.py tests/test_principal_analysis.py
```

The output must be a new file. The script does not overwrite existing
results. The source principal statistics are hash-checked before analysis.

## Implication for the paper

This breakdown strengthens the account of where memory helps and confirms
that k=0's alias performance is not an average over a failed direction.
It does not reduce the single-task breadth limitation. It also makes the
ceiling limitation more concrete: the architecture comparison has no
discriminating failures among structured models on 80% of shots, while the
remaining 20% show substantially lower performance at the longest blackout.
The paper addition includes that qualification alongside the positive memory
result and identifies the additional subgroup intervals as exploratory.

Full numerical output: [post hoc subgroup results](../results/principal_subgroups_posthoc_v1.json).
Analysis source: [subgroup analysis script](../scripts/analyse_principal_subgroups.py).
