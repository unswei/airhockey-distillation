# Air hockey on Marvin

This note describes how the upstream robot air-hockey stack was installed on
Marvin and how to use it as the base for later project experiments. The full
source and interface audit is in
[`upstream_audit.md`](upstream_audit.md).

## A. How it was set up

Marvin is an Ubuntu 24.04 machine with an NVIDIA RTX 5090, driver 595.84 and
Docker 29.7.1 with the NVIDIA runtime. Experiments run directly through SSH
and Docker; there is no job scheduler.

The files are separated into three locations:

| Purpose | Marvin path |
| --- | --- |
| This code repository | `/home/oliver/Code/airhockey-memory-distillation` |
| Clean upstream clones | `/home/oliver/Code/upstream/airhockey-memory-distillation` |
| Runs, checkpoints and videos | `/home/oliver/experiments/airhockey-memory-distillation` |

The upstream clones are pinned to:

- `AndrejOrsula/drl_air_hockey` at
  `a41081c4c3860877d7553c7eb9e84c1b7f5f7da0`;
- `AirHockeyChallenge/air_hockey_challenge` at
  `34729081d4327141dcf560c0fb8274cd4882e82c`;
- Andrej Orsula's DreamerV3 fork at
  `4049794d4135e41c691f18da38a9af7541b01553`.

The first two repositories can be prepared from a clean Marvin account with:

```bash
upstream_root=/home/oliver/Code/upstream/airhockey-memory-distillation
mkdir -p "${upstream_root}"

git clone https://github.com/AndrejOrsula/drl_air_hockey.git \
  "${upstream_root}/drl_air_hockey"
git -C "${upstream_root}/drl_air_hockey" checkout --detach \
  a41081c4c3860877d7553c7eb9e84c1b7f5f7da0

git clone https://github.com/AirHockeyChallenge/air_hockey_challenge.git \
  "${upstream_root}/air_hockey_challenge"
git -C "${upstream_root}/air_hockey_challenge" checkout --detach \
  34729081d4327141dcf560c0fb8274cd4882e82c
```

The project build script checks those commits and rejects dirty upstream
trees. It creates temporary source archives, applies the small compatibility
[`patch`](../patches/drl_air_hockey_marvin.patch), and builds the challenge,
DreamerV3 and DRL code into a Blackwell-compatible image:

```bash
cd /home/oliver/Code/airhockey-memory-distillation
./scripts/build_upstream_images.sh
```

The resulting experiment image is:

```text
marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt
sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f
```

It contains Python 3.12, MuJoCo 3.3.1 and JAX 0.5.3. The audit confirmed that
JAX sees the RTX 5090 and that a MuJoCo reset and step work at the upstream
50 Hz control rate. The current DRL wrapper exposes 20 observations and 6
actions; [`audit_upstream_interface.py`](../scripts/audit_upstream_interface.py)
records their exact layout.

## B. Using it for future experiments

Treat the upstream clones as read-only inputs. New task wrappers, policy
adapters, training code and configurations belong in this repository. Raw run
outputs belong under the experiment path, not in either Git checkout.

A future experiment can use the audited image while mounting the current code
and a fresh output directory:

```bash
code_root=/home/oliver/Code/airhockey-memory-distillation
run_root=/home/oliver/experiments/airhockey-memory-distillation
run_id=smoke-$(date +%Y%m%d-%H%M%S)
mkdir -p "${run_root}/${run_id}"

docker run --rm \
  --gpus all \
  --ipc host \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env PYTHONPATH=/work/src:/src/2025-challenge \
  --volume "${code_root}:/work:ro" \
  --volume "${run_root}/${run_id}:/run-output:rw" \
  --workdir /work \
  marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt \
  bash -lc \
  'python3 scripts/audit_upstream_interface.py > /run-output/interface.json'
```

Replace the final command with the relevant versioned experiment script or
Python module. Keep the repository mount read-only and write checkpoints,
logs, metrics and videos through `/run-output`. Every real run should record:

- the Code repository commit and whether its worktree was dirty;
- the resolved YAML configuration and random seeds;
- the upstream commits and container image digest;
- package and hardware information;
- checkpoint hashes, stdout/stderr and machine-readable metrics.

For the tracking-loss project, the next environment adapter should remove the
two upstream puck-velocity inputs, add the visibility mask, and expose only the
two planar target actions. It should supply fixed, validated stiffness and
damping values to the remaining four upstream action fields. Teacher and
student policies must use this same 19-observation, 2-action interface.

The historical pre-trained policy is useful for checking rendering, but it
uses a different 40-dimensional observation and must not be used as the
project teacher. The rebuilt image also overrides Dreamer's declared JAX and
CUDA package bounds for RTX 5090 support. Interface and GPU probes pass, but a
short Dreamer training run should be completed before starting a long teacher
experiment.
