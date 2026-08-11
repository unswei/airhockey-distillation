# Project status

Last updated: 2026-08-11

## Current phase

Phase 0 — source audit and reproduction complete. Phase 1 — minimal
direct-launch set-piece slice complete. The teacher-training gate is `NO_GO`
because `direct_launch_v1` is too easy for inactive and fixed-centre defenders.

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

## Next actions

- Add a scripted physical strike while retaining direct launch as a regression
  mode.
- Preserve v1 and create `direct_launch_v2` with less goal-centre weight and
  outward-shifted near-post targets so neutral defenders concede often enough.
- Re-run the paired gate on v2.
- Run a short Dreamer training smoke test only after the gate returns `GO`.
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

## Blockers

- Marvin's experiment directory is not yet backed up to a durable artefact
  store.
- Teacher training is blocked because v1 inactive and fixed-centre concession
  rates are below 80%; the privileged controller already passes at 100%.
- The Blackwell image deliberately overrides Dreamer's declared JAX 0.4.33 and
  CUDA NVCC 12.2 bounds with JAX 0.5.3 and CUDA NVCC 12.9.86. Device discovery
  and interface smoke tests pass; `pip check` records the two conflicts.
