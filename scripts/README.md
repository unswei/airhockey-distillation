# Scripts

This directory contains small, non-interactive entry points for the Marvin
workflow.

## Upstream audit

- `build_upstream_images.sh` verifies the pinned upstream clones, applies the
  recorded compatibility patch and builds the 2025 Blackwell image.
- `audit_upstream_interface.py` prints the current challenge and DRL policy
  interfaces as JSON and performs a one-step MuJoCo smoke test.
- `reproduce_upstream_demo.sh [output_dir]` renders the supplied 2023
  self-play checkpoint through Xvfb and records an H.264 video. Set the rollout
  length with `STEPS`; the default is 500.
- `run_minimal_task.py [--output PATH]` runs the inactive, fixed-centre and
  privileged controls on the deterministic direct-launch slice and verifies a
  repeated public trajectory.
- `run_teacher_gate.py [--output PATH]` runs the fail-closed pre-training gate.
  It uses the validated `direct_launch_v2` calibration split by default and
  exits with status 2 when any criterion fails or lacks enough evidence.
- `validate_direct_launch_distribution.py` materialises a versioned shot
  manifest and optionally measures all approach times in MuJoCo with
  `--simulate`.
- `smoke_test_teacher.py --profile PROFILE --output PATH --code-commit SHA`
  runs a versioned DreamerV3 training profile. It supports exact-state resume,
  retains the requested number of checkpoints and writes an explicit final
  checkpoint. The `smoke` profile remains a compatibility test rather than a
  policy-quality experiment.
- `evaluate_teacher_checkpoint.py` evaluates an untrained or checkpointed
  Dreamer policy on identical held-out validation shots at fixed blackout
  lengths and records contact-aware outcomes and batch-one latency.
- `evaluate_teacher_checkpoints.py` evaluates every retained checkpoint and
  selects by validation save rate averaged across the configured blackout
  durations. Mean return and then earlier step are deterministic tie-breaks.
- `run_full_teacher_on_marvin.sh RUN_ID CODE_COMMIT` runs or resumes the
  versioned full profile in the audited container, then performs checkpoint
  validation selection. It refuses a dirty or mismatched Marvin checkout.
- `freeze_teacher_checkpoint.py` copies only the inference checkpoint payload,
  binds it to the validation selection and teacher configuration, records
  SHA-256 hashes and makes the frozen directory read-only.
- `collect_teacher_dataset.py` collects deterministic teacher trajectories into
  restart-safe 500-episode shards. Public observations and teacher actions are
  the training interface; previous actions and explicit evaluation-only state
  are retained for later diagnostics but are not student inputs.
- `train_feed_forward.py` trains and validation-selects the Stage B
  observation-only policy, then exports a framework-neutral NumPy checkpoint.
- `evaluate_feed_forward.py` evaluates that checkpoint on the exact validation
  shots and blackout lengths used for the selected teacher.
- `run_stage_b_memory_gate.py` pairs outcomes by shot and blackout, bootstraps
  the teacher advantage and applies the predeclared memory criteria in
  `configs/student/feed_forward_stage_b.yaml`.

The build and rendering scripts are intended to run on Marvin. They fail on a
missing prerequisite and the video script refuses to overwrite its primary
artefacts. Exact commands and audited outputs are in
[`docs/upstream_audit.md`](../docs/upstream_audit.md).

## Planned experiment entry points

- `train_core_students.sh`
- `evaluate_core.sh`
- `make_figures.sh`
- `render_demo.sh`
- `reproduce_core.sh`

These scripts will be added only as their underlying commands become
reproducible.
