# Project status

Last updated: 2026-08-13

## Current phase

Phase 0 — source audit and reproduction complete. Phase 1 — minimal
direct-launch set-piece slice complete. Phase 2 — v3 teacher complete and
frozen. Stage B v3 returned a predeclared `GO`: a credible memoryless policy
is close to the teacher without blackout and falls far behind under a 20-step
blackout. The task is now demonstrably memory-dependent under the fixed gate.
The causal teacher-state ablation also returned `GO`: erasing state at
blackout onset causes a 45.8-point drop at 20 steps under deterministic
inference. Phase 3 — distillation infrastructure — is authorised.
The first `n=64`, `k=2` structured recurrent student is implemented and its
corrected tiny-dataset overfit gate returns `GO`. Its 20,000-episode
deterministic-mean collection, single-seed training and paired validation run
are complete. The initial vertical slice was behaviourally weak and the
all-valid-step correction remained below its no-blackout gate. The
predeclared deterministic shadow-teacher correction then succeeded: the same
architecture and seed now save 98.7% of no-blackout validation shots and
98.4% across the subsequently opened paired blackout evaluation. The first
full-data structured-student vertical slice is successful.

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
- Completed the one-million-step teacher run with 75,977 optimiser updates and
  no non-finite metrics or restarts.
- Selected the 720,000-step checkpoint on 500 validation episodes. It saved
  415/500 overall: 88%, 89%, 82%, 82% and 74% at blackout lengths 0, 5, 10,
  15 and 20.
- Froze the selected teacher as a read-only inference bundle. Its `agent.pkl`
  SHA-256 is `aafa486922860eb6a4a1d046667b0f1422170f4b242fa7eaa5c5cbf93782c94e`.
- Collected a schema-2 teacher dataset with 20,000 episodes and 870,634
  transitions. It records the clipped action actually executed and retains the
  raw Dreamer output only as diagnostic metadata.
- Trained a 5,602-parameter observation-only feed-forward policy on public
  observations and executed teacher actions, selecting epoch 99 by validation
  action MSE.
- Evaluated teacher and feed-forward policies on 500 exact paired validation
  cases. Feed-forward save rate fell from 84% at no blackout to 62% at 20
  steps; teacher save rate fell from 88% to 74%.
- Applied the predeclared Stage B gate. The 20-step paired teacher advantage
  was 12 points with bootstrap 95% interval [1, 23], but the required advantage
  was 15 points. Advantage growth from zero to 20 steps was 8 points rather
  than the required 10. Decision: `NO_GO` for the memory-required claim.
- Added a locked five-step visible prefix that prevents policy actions from
  encoding shot identity into robot motion before blackout.
- Added `direct_launch_v3` with 90 paired alias families in calibration and
  validation. Paired shots have closely matched last-visible positions but
  travel to opposite near posts.
- Validated all 216 v3 calibration shots on Marvin. All reached the approach
  plane in 0.40--0.94 seconds with zero faults.
- Passed the alias gate: hidden public observations are bit-identical, maximum
  last-visible separation is 8.10 mm and minimum privileged-action distance is
  0.315.
- Passed the v3 task gate: inactive and fixed-centre defenders conceded 96.3%
  and 96.8%; the privileged controller saved 99.1%.
- Rejected reuse of the frozen v2 teacher. On v3 it saved 75.1% without
  blackout, 58.0% overall and 45.3% at 20 steps, failing all three readiness
  rate thresholds.
- Completed a fresh v3 Dreamer smoke run with non-zero rewards and exact
  checkpoints at 0, 500 and 1,000. Scheduled the full v3 teacher run from
  commit `66fbc4c`.
- Completed the one-million-step v3 teacher run and evaluated all 51 retained
  checkpoints over 1,125 validation cases each.
- Selected and froze the 700,000-step checkpoint. It saved 98.9% overall and
  98.2% at 20-step blackout; its read-only `agent.pkl` SHA-256 is
  `6a672d5b6d7c2b9ca2335f1a85b69280ca7db58deb5e2f56d2ba033ebf636254`.
- Collected 20,000 v3 teacher episodes containing 732,168 transitions and
  trained the predeclared 5,602-parameter feed-forward baseline.
- Compared the teacher and feed-forward policy over 1,125 paired cases. The
  teacher led by 37.8 points at 20-step blackout with bootstrap 95% interval
  [31.6, 44.4], but the feed-forward policy saved only 56.0% without blackout.
- Applied the Stage B v2 gate: `NO_GO`. The weak no-blackout baseline and
  shrinking advantage with blackout prevent attributing the gap to memory.
- Predeclared a stronger Stage B v3 baseline: a strictly feed-forward PPO
  policy trained directly on task reward for three fixed seeds, with 50%
  no-blackout sampling and no change to the public policy interface.
- Added a validation-only qualification gate. A baseline must be credible and
  within 10 points of the teacher without blackout before the untouched test
  split and confirmatory memory gate can run.
- Trained all three predeclared one-million-step PPO seeds. Validation-only
  qualification selected seed 14304 at 92.0% no-blackout saves; the teacher
  advantage was 7.1 points and no simulator fault occurred.
- Opened the test split only after qualification passed, then evaluated teacher
  and selected baseline on 1,125 paired cases. At 20 steps, the teacher saved
  96.9% and the baseline 33.3%; the 63.6-point paired advantage had bootstrap
  95% interval [56.9, 70.2].
- Applied the Stage B v3 gate: `GO`. All five checks passed, including
  90.7% visible baseline performance, an 8.9-point no-blackout gap and
  54.7 points of advantage growth.
- Preserved the raw episode records outside Git and added the byte-identical
  compact gate result plus verified artefact hashes to the repository.
- Added deterministic Dreamer evaluation using the RSSM categorical mode and
  actor mean, with sampled inference retained as the default for older runs.
- Repeated the normal-state teacher evaluation in independent processes. All
  1,125 episode records matched exactly.
- Applied the predeclared causal recurrent-state ablation. At 20 blackout
  steps, resetting state reduced saves from 216/225 to 113/225: a 45.8-point
  paired drop with bootstrap 95% interval [38.7, 52.9]. All 225 no-blackout
  episode records were identical and the decision was `GO`.
- Froze the canonical raw run as root-owned, read-only evidence and recorded
  its code, configuration, checkpoint, result and checksum-manifest hashes.
- Copied the frozen teacher, canonical Stage B runs, teacher selection report,
  three final PPO seeds and a complete Git bundle to iCloud Drive. All 57
  files passed source and whole-backup checks; the provider reported caught up.
- Implemented the principal `n=64`, `k=2` structured recurrent policy as
  matched NumPy and PyTorch runtimes. It has 12,328 parameters, including
  2,630 recurrent-core parameters, and a 256-byte float32 state.
- Added explicit episode carry, previous-action recurrence, stable diagonal
  time-constant initialisation, sequence unrolling and framework-neutral NPZ
  checkpoint export.
- Confirmed by automatic differentiation that the state-dependent departure
  from the fixed diagonal recurrent Jacobian has rank at most two. Checkpoint
  reload exactly reproduces action and state sequences.
- Collected 16 new complete episodes containing 541 transitions from
  deterministic RSSM predictions and actor means. The new manifest labels
  these targets explicitly; the older sampled-action dataset was not reused.
- Preserved the original 16-episode overfit `NO_GO`: MSE fell 99.28% but
  stopped at `0.00648`, showing that the gate had become a capacity test.
- Applied the versioned one-episode correction. MSE reached `7.4562e-5`,
  NumPy/PyTorch error was `2.2054e-6`, reload was exact and all checks passed.
- Repeated corrected training from scratch and obtained byte-identical metrics
  and checkpoint hash `ef76db026a4093238d0734bdf44be8b242d998777da7ebe065147e3768e3c0fa`.
  Froze and hashed the dataset, v1, v2, repeat and logs. The full pinned suite
  passes 99 tests after adding the corrected full-data path.
- Predeclared the first full-data vertical slice at 20,000 newly collected
  deterministic-mean episodes, split 16,000/2,000/2,000 by episode seed.
  Added 64-step truncated training with state carry across chunks,
  validation-loss selection, exact export checks and paired closed-loop
  validation. Sampled-action manifests are rejected.
- Collected all 20,000 episodes and 713,257 deterministic-mean transitions.
  The first 100-epoch training attempt stopped during checkpoint export on a
  metadata-name conflict. Preserved that run and predeclared a v2
  metadata-only code correction with unchanged training settings.
- Completed corrected seed 14303 at epoch 100 with validation MSE `0.10859`,
  exact reload and checkpoint hash `8fbe2171fc4d7272dda0cbdd82adbf8c4bac506aaac21aada62fe9e37a99271f`.
  The paired 1,125-episode validation save rate is only 45.3%, including 46.2%
  with no blackout and 40.9% at 20 steps. The checkpoint is not promoted.
- Diagnosed the weak no-blackout control on 225 fixed validation shots. The
  ordinary student saved 104/225; teacher control for only steps 0--15 followed
  by student control saved 224/225, equal to the frozen teacher. Every episode
  continued beyond the hand-off.
- Measured action MSE of `0.966` on locked steps 0--4 and `0.694` on live but
  loss-masked steps 5--15, compared with `0.103` on the loss-bearing suffix
  under teacher-forced previous commands.
- Measured closed-loop covariate shift: the student's 95th-percentile nearest
  training-history distance is 21.73 versus 1.29 on on-policy validation
  histories, a 16.8-fold difference.
- Audited all 20,000 deterministic episodes. All 5,311 repeated shot/blackout
  groups have byte-identical input and target trajectories, with no conflicting
  target for an identical full input trajectory. This does not support target
  multimodality as the primary failure.
- Found that the recurrent action input is the previous requested command,
  not the applied hold action during the five locked steps. The locked-command
  discrepancy has MSE `1.151`; the contract needs explicit terminology.
- Retrained seed 14303 on the same frozen dataset with loss on every valid
  step. The no-blackout save rate improved from 46.2% to 67.1% but missed the
  predeclared 75% gate, so the decision is `NO_GO`. No blackout condition was
  opened and no simulator or safety fault occurred.
- Collected 20,000 student-controlled trajectories with 1,453,360 exact
  deterministic teacher-mean labels and combined them one-to-one with the
  original frozen teacher-controlled set.
- Retrained the same `n=64`, `k=2` architecture and seed from scratch on the
  40,000-episode aggregate. It selected epoch 100 at validation MSE `0.06461`.
- Passed the no-blackout-only gate with 222/225 saves (98.7%), zero
  concessions and zero simulator/safety faults.
- Opened the full paired blackout evaluation only after that `GO`. The frozen
  checkpoint saved 1,107/1,125 episodes overall and 220/225 at 20 steps.
- Froze 59 raw evidence files as root-owned, read-only artefacts on Marvin and
  recorded a complete checksum manifest.

## Next actions

- Implement and evaluate the remaining predeclared matched recurrent student
  family now that the structured vertical slice passes.
- Retain direct launch as the core controlled task; a scripted physical strike
  remains a later secondary extension.

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
| 2026-08-12 | Completed and validation-selected the full teacher run | Selected step 720,000; 83.0% save rate over 500 cases and 74.0% at 20-step blackout |
| 2026-08-12 | Froze teacher and collected Stage B demonstrations | Read-only teacher hash recorded; 20,000 episodes and 870,634 schema-2 transitions |
| 2026-08-12 | Trained and evaluated the observation-only feed-forward policy | 5,602 parameters; 84% saves at blackout 0 and 62% at blackout 20 |
| 2026-08-12 | Applied the paired Stage B memory gate | `NO_GO`; positive 12-point long-blackout advantage, but two predeclared effect-size checks failed |
| 2026-08-12 | Validated `direct_launch_v3` physical and alias properties | `GO`; 216/216 arrivals, 90/90 alias families pass, zero faults |
| 2026-08-12 | Applied the v3 task-validity gate | `GO`; inactive/fixed concede above 96%, privileged saves 99.1% |
| 2026-08-12 | Probed frozen 720k teacher on v3 | `NO_GO`; fails no-blackout, overall and long-blackout readiness thresholds |
| 2026-08-12 | Ran fresh v3 training smoke and scheduled full run | Smoke passed; `teacher-full-v3-2026-08-12-v1` active from `66fbc4c` |
| 2026-08-12 | Completed and validation-selected the full v3 teacher run | Selected step 700,000; 98.9% save rate over 1,125 cases and 98.2% at 20-step blackout |
| 2026-08-12 | Froze the v3 teacher and collected new demonstrations | Read-only agent hash recorded; 20,000 episodes and 732,168 transitions |
| 2026-08-12 | Trained and evaluated the Stage B v2 feed-forward policy | 5,602 parameters; 56.0% saves at blackout 0 and 60.4% at blackout 20 |
| 2026-08-12 | Applied the paired Stage B v2 memory gate | `NO_GO`; teacher gap is large, but the feed-forward policy fails the no-blackout credibility and comparability checks |
| 2026-08-12 | Predeclared the Stage B v3 visible-baseline correction | Direct-reward memoryless PPO; three fixed seeds; validation qualification before test confirmation |
| 2026-08-12 | Trained and qualified all three Stage B v3 PPO seeds | Seed 14304 selected at 92.0% no-blackout validation saves; qualification `GO` |
| 2026-08-12 | Applied the held-out Stage B v3 memory gate | `GO`; 8.9-point no-blackout gap, 63.6-point 20-step gap and all five checks pass |
| 2026-08-12 | Closed and froze Stage B v3 evidence | Compact canonical result in Git; 1,125 episode rows remain under the hashed Marvin run |
| 2026-08-12 | Committed deterministic Dreamer inference and repeated the full normal evaluation | 1,125/1,125 episode rows reproduce exactly |
| 2026-08-12 | Applied and froze the deterministic causal state ablation | `GO`; 45.8-point drop at 20 steps, 95% CI [38.7, 52.9], and identical no-blackout records |
| 2026-08-12 | Backed up the frozen teacher and canonical Stage B evidence | 57 files and 19,245,888 bytes copied to iCloud Drive; source hashes, complete Git bundle and manifest verified |
| 2026-08-12 | Implemented the structured `n=64`, `k=2` student | Matched NumPy/PyTorch runtimes, exact checkpoint replay, rank-2 Jacobian test and 89 passing tests |
| 2026-08-12 | Ran the original 16-episode structured-student overfit gate | `NO_GO`; 99.28% loss reduction but selected MSE `0.00648` did not reach `1e-4` |
| 2026-08-12 | Applied and repeated the one-episode overfit correction | `GO`; MSE `7.4562e-5`, exact reload and identical repeated checkpoint hash |
| 2026-08-12 | Predeclared the first full-data structured-student seed | 20,000 deterministic-mean episodes; 80/10/10 split; seed 14303; 98 passing tests |
| 2026-08-13 | Collected the full deterministic-mean dataset | 20,000 episodes, 713,257 transitions and 40 hashed shards |
| 2026-08-13 | Ran full-data structured seed 14303 and paired validation | Training/export complete; validation MSE `0.10859`; weak 45.3% closed-loop save rate |
| 2026-08-13 | Diagnosed the structured student's no-blackout failure | The unsupervised 16-step prefix is primary; teacher prefix recovers 224/225 saves; large rollout shift measured; exact repeats have no conflicting targets |
| 2026-08-13 | Applied the all-valid-steps no-blackout gate | `NO_GO`; saves improve from 46.2% to 67.1% but remain below the predeclared 75% threshold |
| 2026-08-13 | Collected deterministic shadow-teacher round one | 20,000 student-controlled episodes, 1,453,360 transitions and 40 hashed shards |
| 2026-08-13 | Applied the shadow-round no-blackout gate | `GO`; 222/225 saves (98.7%), no concessions and no faults |
| 2026-08-13 | Evaluated the frozen shadow-round student on paired blackouts | 1,107/1,125 saves overall and 220/225 at 20 steps; raw evidence frozen and hashed |

## Blockers

- The Blackwell image deliberately overrides Dreamer's declared JAX 0.4.33 and
  CUDA NVCC 12.2 bounds with JAX 0.5.3 and CUDA NVCC 12.9.86. Device discovery
  and a 108-update training smoke test pass; `pip check` records the two
  declared-version conflicts.
