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
- `smoke_test_teacher.py --output PATH --code-commit SHA` runs the bounded
  DreamerV3 compatibility test using v2 train shots and uniformly sampled
  0--20-step blackouts. It fails unless Dreamer writes training metrics and a
  non-zero episode return. It is not a policy-quality experiment.
- `evaluate_teacher_checkpoint.py` evaluates an untrained or checkpointed
  Dreamer policy on identical held-out validation shots at fixed blackout
  lengths and records contact-aware outcomes and batch-one latency.

The build and rendering scripts are intended to run on Marvin. They fail on a
missing prerequisite and the video script refuses to overwrite its primary
artefacts. Exact commands and audited outputs are in
[`docs/upstream_audit.md`](../docs/upstream_audit.md).

## Planned experiment entry points

- `train_teacher.sh`
- `collect_dataset.sh`
- `train_core_students.sh`
- `evaluate_core.sh`
- `make_figures.sh`
- `render_demo.sh`
- `reproduce_core.sh`

These scripts will be added only as their underlying commands become
reproducible.
