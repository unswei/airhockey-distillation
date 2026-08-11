# Project status

Last updated: 2026-08-11

## Current phase

Phase 0 — source audit and reproduction complete. Phase 1 — minimal
direct-launch set-piece slice complete. Phase 2 — teacher integration started.
The v2 task gate is `GO`, and the bounded DreamerV3 CUDA smoke test passes.
The first 20,000-step learning diagnostic did not improve held-out save rate.
The corrected upstream-like replay and optimisation procedure does improve
held-out concession rate and return at the same budget. Full teacher training
is scheduled on Marvin with exact periodic checkpoints and validation
selection.

## Completed

- Split the work into independent code and paper repositories.
- Initialised both repositories on the `main` branch without remotes.
- Created the initial code, configuration, test and provenance layout.
- Recorded `marvin` as the canonical Linux execution host.
- Kept large experiment products outside Git history.
- Audited Marvin's hardware, storage, Docker and GPU runtime.
- Pinned the current challenge, DRL and Dreamer source revisions and licences.
- Reconstructed the current Blackwell-compatible image from clean source
  archives and recorded its immutable digest and package set.
- Confirmed JAX 0.5.3 sees the RTX 5090 from the rebuilt image.
- Measured the current 20-observation, 6-action policy interface and the raw
  challenge interface with a machine-readable one-step probe.
- Reproduced the supplied 2023 two-agent self-play checkpoint as a rendered
  H.264 video from its immutable container image.
- Confirmed that the historical 40-observation checkpoint is not a valid
  teacher for the proposed 19-observation task.
- Implemented one deterministic direct-launched shot in the pinned single
  defender environment.
- Implemented the 19-dimensional public observation, deterministic blackout
  and two-dimensional action adapter with fixed mid-range impedance.
- Added inactive, fixed-centre and explicitly privileged intercept controls.
- Verified exact replay of the same public trajectory in MuJoCo.
- Added masking, visibility timing, privileged-state leakage and replay tests.
- Added a fail-closed teacher-training gate with explicit evidence thresholds.
- Verified 20 exact, fault-free fixed-shot repetitions in the pinned Marvin
  simulator.
- Applied the gate: replay, observation isolation and Marvin reliability pass;
  the three control-rate checks have insufficient evidence at one distinct
  shot.
- Added deterministic, independently seeded calibration, train, validation and
  test manifests for `direct_launch_v1`.
- Validated all 216 calibration shots on Marvin: balanced centre/near-post
  coverage, three lateral launch regions, 0.40--0.94 s realised approach
  times, and zero simulator faults.
- Added contact-aware concession, return, arrest, safe-deflection and timeout
  outcomes using MuJoCo collision events at every simulation substep.
- Evaluated all controls on the paired 216-shot v1 manifest. The privileged
  controller saved 216/216 shots; no controller calibration was required.
- Preserved v1 and added pair-weighted `direct_launch_v2`, retaining all nine
  launch/target pairings while concentrating the task on physically useful
  same-side and centre-to-near-post approaches.
- Validated all 216 v2 calibration shots on Marvin: 0.42--0.94 s realised
  approach times and zero faults.
- Re-ran the paired gate on v2. Inactive and fixed-centre each conceded
  200/216; the privileged controller saved 216/216; every check passed.
- Added a deterministic v2 training-shot sampler with uniformly sampled
  0--20-step blackouts and no privileged observation path.
- Connected the 19-observation, two-action Gymnasium task to the pinned
  DreamerV3 fork, including a local shim for its missing `elements` import.
- Clipped public normalisation overshoot to the declared `[-1,1]` range and
  added pinned-MuJoCo space-contract coverage.
- Completed a 1,000-step CUDA smoke run with 108 optimiser updates, finite
  metrics and a step-1,000 checkpoint.
- Added the versioned `defend_shot_v1` reward: -1 for concession, +1 for a
  conclusive contact-aware save and +0.2 once for first valid contact.
- Completed a reward-bearing Dreamer smoke run with 21/21 non-zero episode
  returns spanning -1.0 to +1.2, 108 optimiser updates and finite metrics.
- Completed a 20,000-step `size1m` learning diagnostic with finite metrics,
  live world-model and reward learning, and 90 paired held-out cases per
  policy.
- Measured no held-out policy improvement: save rate changed from 24/90
  untrained to 20/90 trained, while mean return changed from -0.329 to -0.362.
- Identified that the diagnostic's replay ratio of 8 produced only about 610
  optimiser updates and ended before Dreamer's 1,000-update warm-up, compared
  with replay ratios 80--128 in the pinned air-hockey procedures.
- Restored replay ratio 80 and the upstream 0.2 uniform, 0.6 prioritised, 0.2
  recency mixture; separated environment and logging configuration; added
  exact-state resume, exact step checkpoints and validation selection.
- Repeated the 20,000-step check. Concessions fell from 58/90 untrained to
  50/90 trained, save rate rose from 28.9% to 33.3%, and mean return improved
  from -0.282 to -0.131 on paired validation cases.
- Verified exact retained checkpoints at steps 0, 500 and 1,000 in a passing
  reward-bearing smoke run.

## Next actions

- Monitor `teacher-full-v1-2026-08-12-v2`; after its automatic validation
  pass, apply the teacher-readiness criteria before beginning student work.
- Add a scripted physical strike while retaining direct launch as a regression
  mode.
- Select a backed-up artefact destination before large checkpoints or datasets.

## Commands and results

| Date | Command or action | Result |
| --- | --- | --- |
| 2026-08-10 | Created repository scaffold from the umbrella project brief | Complete |
| 2026-08-10 | `git init -b main` in the resolved `Code` and `Paper` targets | Two independent worktrees; umbrella remains outside Git |
| 2026-08-10 | Selected `marvin` as the code and experiment runtime | Recorded; connection and runtime audit pending |
| 2026-08-10 | Parsed all draft YAML and TOML; compiled the Python package | Passed |
| 2026-08-11 | Audited `marvin`, upstream clones, licences and source history | Source and host provenance recorded in `docs/upstream_audit.md` |
| 2026-08-11 | Built the pinned 2025 source stack with `scripts/build_upstream_images.sh` | Final image `sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f` |
| 2026-08-11 | Ran the interface probe on the rebuilt image | 20 policy observations, 6 policy actions, 50 Hz; identical to the pre-existing image report |
| 2026-08-11 | Ran JAX GPU discovery in the rebuilt image | JAX 0.5.3 found one RTX 5090 CUDA device |
| 2026-08-11 | Rendered 300 upstream self-play steps with `scripts/reproduce_upstream_demo.sh` | 13.07 s, 1920x1080 H.264 video; SHA-256 `815c8c65e4ce9274e46cdbabdf224d71b4a99ce83e2d68c1309eaf4a63c3acab` |
| 2026-08-11 | Ran minimal task contract tests locally | 14 passed; MuJoCo integration skipped outside Marvin |
| 2026-08-11 | Ran the complete minimal task suite in the audited Marvin image | 15 passed |
| 2026-08-11 | Ran inactive, fixed-centre and privileged controls on the same direct-launched shot | Inactive and fixed-centre conceded; privileged controller returned the puck; replay matched exactly |
| 2026-08-11 | Ran the expanded suite in the pinned Marvin image | 19 passed in 1.68 s |
| 2026-08-11 | Ran `scripts/run_teacher_gate.py` on Marvin | `NO_GO`; 20/20 reliable exact replays and clean observation contract, but only 1/200 required distinct shots for the three rate checks |
| 2026-08-11 | Generated `direct_launch_v1` calibration manifest | 216 distinct shots; 24 per launch/target pairing; manifest SHA-256 `82986091f72a3e51cde803ed73527e060bb0d0a89fda6d08e82eb810abf8724e` |
| 2026-08-11 | Simulated all calibration shots in the pinned Marvin image | 216/216 reached the approach plane in 0.40--0.94 s; zero faults |
| 2026-08-11 | Ran the contact-aware suite in the pinned Marvin image | 32 passed |
| 2026-08-11 | Ran the paired gate on all 216 v1 shots | `NO_GO`; inactive conceded 41.7%, fixed centre 40.3%, privileged saved 100%; all technical checks passed |
| 2026-08-11 | Generated and simulated `direct_launch_v2` | 216 distinct shots; all reached the approach plane in 0.42--0.94 s; manifest SHA-256 `6e2b61f71135c23c2c1d459c8d90c65e8a153cc645986103d95c67b2e0c2733c` |
| 2026-08-11 | Ran the paired gate on all 216 v2 shots | `GO`; inactive and fixed centre each conceded 92.6%, privileged saved 100%, and all technical checks passed |
| 2026-08-11 | Ran the 1,000-step DreamerV3 smoke test on Marvin | `PASS`; JAX 0.5.3 used `cuda:0`, 108 optimiser updates completed, all metrics were finite, and a step-1,000 checkpoint was written |
| 2026-08-11 | Ran the reward-bearing DreamerV3 smoke test | `PASS`; 21/21 logged episode returns were non-zero (-1.0 to +1.2), 108 optimiser updates completed, and all metrics were finite |
| 2026-08-11 | Ran the short reward-bearing learning diagnostic | `HOLD`; training completed 19,992 logged steps with finite metrics, but held-out save rate changed from 26.7% untrained to 22.2% trained across 90 paired cases |
| 2026-08-12 | Ran the corrected 20,000-step learning check | `PASS`; replay ratio reached 81.2, an exact final checkpoint was written, concessions changed from 58/90 untrained to 50/90 trained, and mean return improved from -0.282 to -0.131 |
| 2026-08-12 | Ran the exact-step checkpoint smoke | `PASS`; checkpoints at steps 0, 500 and 1,000 were retained and the explicit final checkpoint matched the requested step |
| 2026-08-12 | Scheduled `teacher-full-v1-2026-08-12-v2` on Marvin | Active systemd unit pinned to `f1fb960`: 1,000,000 steps, checkpoints every 20,000 steps, then 500 validation episodes per retained checkpoint; v1 failed before Docker because the user service lacked the Docker group |

## Blockers

- Marvin's experiment directory is not yet backed up to a durable artefact
  store.
- The full run is not itself evidence of teacher readiness. Its selected
  checkpoint must pass the no-blackout, blackout, reset, latency and 500-
  episode stability criteria before distillation begins.
- The Blackwell image deliberately overrides Dreamer's declared JAX 0.4.33 and
  CUDA NVCC 12.2 bounds with JAX 0.5.3 and CUDA NVCC 12.9.86. Device discovery
  and a 108-update training smoke test pass; `pip check` records the two
  declared-version conflicts.
