# Frozen-policy observation-noise extension

Completed 30 September 2026. This supplementary experiment does not replace
the principal test. No policies were trained, selected or changed.

## Question and design

Do the existing recurrent policies retain their save rates when the visible
puck-position measurements are noisy?

- Policies: the frozen teacher and all five existing seeds (14303–14307)
  of structured `k=0`, structured `k=4` and GRU-64. Checkpoint hashes were
  checked against the original principal-test records before evaluation.
- Noise: zero-mean Gaussian, independent in the two coordinates and across
  observations; per-coordinate standard deviations 0, 1 and 5 mm. These
  are synthetic levels, not measured sensor errors.
- Shots: 225 fresh shots from the unchanged direct-launch distribution,
  generated with seed 930306: 90 alias pairs and 45 support shots.
- Blackouts: 0 and 20 control steps (0 and 400 ms), with the original
  five-step visible, action-locked prefix and 125-step timeout.
- Pairing: one standard-normal trace per shot unit, scaled across levels
  and shared across policies, training seeds and blackout durations.
  Alias-pair members share a trace. Noise seed: 19302026, using PCG64 keyed
  by a SHA-256 digest of the unit identifier. Traces do not depend on rollout
  order, termination time or policy actions.
- Injection: visible puck coordinates only, before public clipping;
  conversion to normalised coordinates uses the upstream puck ranges,
  1.8847 m and 0.9747 m. Visibility and the hidden-position mask are unchanged.
  Ground-truth dynamics and outcome classification are unchanged.
- Size: 1,350 episodes per policy, 16 policies, 21,600 episodes in total.

Conditions, shot/noise seeds, checkpoint hashes, inference-source hashes and
the analysis plan were frozen before the new shots were evaluated. This was
an extension designed after viewing the principal results, not part of the
original predeclared principal study. The opening SHA-256 is
`fdab476e8f9efd7192f51f6310c04c52a6cf10abb6ba95ef2b1f734167d07124`.

## Results

Save rates in percent; students are means over all five seeds.

| Blackout | Noise SD | k=0 | k=4 | GRU-64 | Teacher |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 ms | 0 mm | 98.6 | 98.5 | 99.4 | 99.1 |
| 0 ms | 1 mm | 98.3 | 98.6 | 99.2 | 99.1 |
| 0 ms | 5 mm | 98.4 | 98.9 | 97.9 | 98.7 |
| 400 ms | 0 mm | 98.1 | 97.6 | 96.8 | 97.3 |
| 400 ms | 1 mm | 98.0 | 97.5 | 96.8 | 96.9 |
| 400 ms | 5 mm | 98.5 | 94.5 | 92.9 | 96.0 |

At 5 mm and 400 ms, the paired `k=0` minus GRU-64 difference is 5.6
percentage points, with a pointwise 95% interval of [2.8, 9.0]. Relative to
clean measurements on these same shots, `k=0` changes by +0.4 points
[0.0, 1.1], GRU-64 by −3.9 [−6.9, −1.3], `k=4` by −3.1 [−7.6, 0.0],
and the teacher by −1.3 [−3.1, 0.0].

The alias/support breakdown matters. At 5 mm and 400 ms, alias save rates
are 100.0%, 96.1%, 94.1% and 98.3% for `k=0`, `k=4`, GRU-64 and the
teacher; support rates are 92.4%, 88.0%, 88.0% and 86.7%. The support-shot
`k=0` minus GRU-64 difference is 4.4 points [−1.3, 11.6], leaving it
unresolved. Without blackout at 5 mm, `k=4` saves more support shots than
`k=0`: the `k=0` minus `k=4` difference is −4.0 points [−8.4, −0.9].
Thus the extension does not show uniform dominance across conditions.

At 5 mm/400 ms the five individual `k=0` rates are 99.1, 98.2, 99.1,
98.7 and 97.3%; GRU-64 rates are 94.2, 88.4, 95.6, 94.2 and 92.0%.
All per-seed outcomes and all prespecified aggregate/subgroup comparisons,
including null and unfavourable ones, are retained in the result artefacts.

## Uncertainty and scope

The 10,000-replicate paired bootstrap (seed 29302026) resamples matched
training seeds and shot units, keeping each alias pair intact. Units are
stratified as 90 alias families, 20 left-target support shots, 20 right-target
support shots and five centre-target support shots. Shared draws are used
for every family, noise level and blackout condition; the teacher is not
artificially replicated across training seeds. Intervals are pointwise
percentile intervals without multiplicity adjustment. The fixed observed
stratum composition differs from the principal bootstrap.

One noise trace per unit gives a paired finite-sample test, not a sweep over
independent noise realisations for each shot. A 100% empirical alias rate
does not guarantee perfect performance on unseen shots or noise traces.
The experiment changes measurement noise, not puck dynamics, contact
physics or the task. It does not establish a mechanism for the difference,
general noise robustness, hardware robustness or universal sufficiency of
linear recurrence. The small positive `k=0` change is not a claim that noise
improves the policy in general.

## Reproduction and checks

- Protocol: `configs/experiments/observation_noise_v1.yaml` and
  `configs/env/direct_launch_noise_v1.yaml`.
- Runner: `scripts/evaluate_observation_noise.py`; container orchestration:
  `scripts/run_observation_noise.sh`.
- Runtime: original pinned container digest
  `sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f`;
  Intel Core Ultra 9 285 CPU and RTX 5090 for teacher inference. Existing
  experiment directories are mounted read-only. No new latency claim is made.
- Four engineering smoke checks use already-opened validation shots. Each
  policy family agrees exactly with the original evaluator at zero noise
  on shot identity, horizon, outcome, score and episode length.
- Public-sized evidence: `results/observation_noise_v1.json` and
  `results/observation_noise_v1_episodes.json.gz`. The compressed archive
  contains all 21,600 episode rows and the exact opening/result JSON texts,
  preserving their hashes. Full original checkpoints are not in this archive.
- The runtime source snapshot, logs, smoke checks and raw result files have
  also been copied into the local project's `.artifacts/` directory. This
  is a second copy of the noise-run evidence, not a backup of all principal
  checkpoints and training datasets.

Recompute the result from the compact archive without running policies:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.analyse_observation_noise \
  --archive results/observation_noise_v1_episodes.json.gz \
  --output /path/to/new-noise-analysis.json
```

The analyser refuses to overwrite existing outputs and checks the complete
paired grid, provenance, checkpoint identity and outcome validity. Unit
tests reproduce the saved analysis from the archive and check zero-noise
identity, units, clipping, visibility, trace pairing and invalid inputs.
`scripts/audit_observation_noise_paper.py --paper /path/to/main.tex` checks
the new table and numerical prose against the analysis.

These materials are included under the existing `acra-2026-submission` tag.
The tag has been updated to include the noise extension; the earlier principal
artefacts are unchanged.
