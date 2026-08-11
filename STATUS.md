# Project status

Last updated: 2026-08-11

## Current phase

Phase 0 — source audit and reproduction complete. Phase 1 — minimal
direct-launch set-piece slice complete. The teacher-training gate is `NO_GO`
until baseline rates are established over a calibrated shot distribution.

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

## Next actions

- Add a scripted physical strike while retaining direct launch as a regression
  mode.
- Calibrate and version non-trivial train, validation and test shot ranges.
- Add contact-aware outcome classification before task-scale evaluation.
- Evaluate inactive, fixed-centre and privileged controls on the same 200
  distinct shots and re-run the gate.
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

## Blockers

- Marvin's experiment directory is not yet backed up to a durable artefact
  store.
- Teacher training is blocked by insufficient distribution-level evidence:
  only one distinct direct-launched shot exists, versus 200 required by the
  gate.
- The Blackwell image deliberately overrides Dreamer's declared JAX 0.4.33 and
  CUDA NVCC 12.2 bounds with JAX 0.5.3 and CUDA NVCC 12.9.86. Device discovery
  and interface smoke tests pass; `pip check` records the two conflicts.
