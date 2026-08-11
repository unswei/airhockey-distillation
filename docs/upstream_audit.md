# Upstream audit and reproduction

Audit date: 2026-08-11

## Outcome

The upstream demonstration runs on Marvin from an immutable 2023 container
image, and the current 2025 source stack builds and exposes a working MuJoCo
interface on the RTX 5090. The demonstration and current source are not the
same policy interface. The historical checkpoint is therefore useful for
checking installation, rendering and evaluation, but it is not a teacher for
the tracking-loss experiment.

The current source stack will be the implementation base. Its policy
interface is 20 observations and 6 actions. The proposed public interface can
be obtained by removing the two puck-velocity inputs, adding one visibility
input, and fixing the four impedance actions in an adapter. This gives the
specified 19 observations and 2 learned actions without changing the
low-level controller.

## Source pins

| Component | Branch inspected | Commit | Commit date | Licence |
| --- | --- | --- | --- | --- |
| [`drl_air_hockey`](https://github.com/AndrejOrsula/drl_air_hockey) | `main` | `a41081c4c3860877d7553c7eb9e84c1b7f5f7da0` | 2025-10-22 | MIT, Andrej Orsula |
| [`air_hockey_challenge`](https://github.com/AirHockeyChallenge/air_hockey_challenge) | `qualifying-2025` | `34729081d4327141dcf560c0fb8274cd4882e82c` | 2025-09-18 | MIT, Atalay Donat, Puze Liu, Jonas Guenster and Davide Tateo |
| [Andrej Orsula's DreamerV3 fork](https://github.com/AndrejOrsula/dreamerv3) | `main` | `4049794d4135e41c691f18da38a9af7541b01553` | 2025-07-16 | MIT, Danijar Hafner |

The challenge link in the project brief and current `drl_air_hockey` README,
`AndrejOrsula/air_hockey_challenge`, no longer permits an anonymous clone. The
public organisation repository above is the canonical repository found for
the current challenge code. Its default branch is `qualifying-2025`.

All three licences permit use and modification provided their copyright and
permission notices are retained in copies or substantial portions. The
project should preserve these notices if upstream source is redistributed.
No upstream source is copied into this Git repository; the build script
creates temporary archives from separate pinned clones.

## Source drift

The old report and pre-built image do not describe the current `main` branch:

- The 2023 policy consumes a 40-dimensional vector. It includes a ten-position
  puck history and other features from the old agent implementation. Its
  learned action is a two-dimensional Cartesian mallet target.
- The current policy consumes 20 values and produces 6 values. It does not use
  a puck-position stack, but it receives the current puck velocity.
- The current source checkout contains configuration files but no trained
  checkpoint. A new teacher must be trained for the project task.
- The pre-built image does not embed an authoritative `drl_air_hockey` source
  revision. Three key files in it match commit
  `94fbc1a85b62587cf80a9457147cdfb349afb71c`, dated 2023-12-14, but the image
  digest remains the reproducible identifier.

The current command-line interface only selects `tournament`, although the
challenge wrapper contains `hit`, `defend` and `prepare` environments. The
current DRL environment factory implements MuJoCo; the exposed `mjx` option is
not implemented there. The set-piece task should use the MuJoCo environment
directly rather than depending on these incomplete command-line choices.

## Current interface

The machine-readable probe is
[`scripts/audit_upstream_interface.py`](../scripts/audit_upstream_interface.py).
It constructs the pinned 2025 tournament environment, resets the raw and
policy wrappers, and performs one zero-action step.

### Challenge-level interface

| Quantity | Observed value |
| --- | --- |
| Environment | `IiwaPositionTournament` |
| Control period | 0.02 s (50 Hz) |
| Agents | 2 |
| Table length / width / goal width | 1.948 m / 1.038 m / 0.25 m |
| Per-agent raw observation | 23 values |
| Combined reset observation | 46 values |
| Combined low-level joint action | 14 values |
| Per-agent interpolated action | `2 x 7`: joint position and velocity |

The raw per-agent observation indices are:

| Component | Indices |
| --- | --- |
| Puck position | 0--2 |
| Puck velocity | 3--5 |
| Joint position | 6--12 |
| Joint velocity | 13--19 |
| Opponent end-effector position | 20--22 |

### Current DRL policy interface

| Observation component | Dimension |
| --- | ---: |
| Joint position | 7 |
| Joint velocity | 7 |
| Planar end-effector position | 2 |
| Planar puck position | 2 |
| Planar puck velocity | 2 |
| **Total** | **20** |

| Action component | Dimension |
| --- | ---: |
| Planar mallet target | 2 |
| Planar stiffness | 2 |
| Planar damping | 2 |
| **Total** | **6** |

The measured end-effector workspace is
`x=[0.59665, 1.31]`, `y=[-0.46585, 0.46585]`. The puck workspace is
`x=[0.56765, 2.45235]`, `y=[-0.48735, 0.48735]`. The upstream controller maps
stiffness from `[2, 25]` and damping from `[0, 0.5]`.

The one-step probe returned reward 0, did not terminate, and returned the
constraint, fault and score fields. The report from the clean rebuilt image
is identical to the report from the pre-existing Marvin image.

### Consequence for this project

The 19-dimensional observation in the project brief is consistent with the
current source:

```text
current 20 - puck_velocity_xy[2] + puck_visible[1] = project 19
```

The action adapter should expose only the two target coordinates and supply
fixed, validated stiffness and damping values. Those values must be held fixed
for the teacher and every student. The adapter must also retain the current
workspace normalisation, clipping, inverse kinematics and safety constraints.

## Marvin runtime

| Component | Audited value |
| --- | --- |
| Host OS | Ubuntu 24.04, kernel `6.17.0-1030-oem`, x86-64 |
| CPU / memory | Intel Core Ultra 9 285, 24 cores, 30 GiB RAM |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB |
| NVIDIA driver | 595.84 |
| Docker | 29.7.1, `overlayfs`, NVIDIA runtime available |
| Host Python | 3.12.3 |
| Container Python | 3.12 |
| MuJoCo | 3.3.1 |
| JAX / jaxlib | 0.5.3 / 0.5.3 |
| CUDA base | `nvidia/cuda:12.9.1-base-ubuntu24.04` |
| CUDA base digest | `sha256:29e5e3425e2e0f5a4e97c9fb4695ba4887cd78210a43cf94c3bcafc6ab01c5e6` |
| Rebuilt challenge image | `sha256:032170f092ff3cbcefbcfaca728507daa8d3a8cc64b901c5a098095e8d9f41e8` |
| Rebuilt final image | `sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f` |

The final image is tagged
`marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt`. A GPU smoke test
reported `JAX 0.5.3` and `[CudaDevice(id=0)]`.

The upstream requirements are not fully pinned. The package freeze from the
clean rebuild differs from the pre-existing Marvin image only in the omitted
`pip` record and `typing-inspection` 0.4.3 rather than 0.4.2. The immutable
image digest and saved package freeze are authoritative for this audit.

There is one known dependency inconsistency. The Dreamer package declares
JAX 0.4.33 and `nvidia-cuda-nvcc-cu12<=12.2`; the Blackwell-compatible image
overrides these with JAX 0.5.3 and CUDA NVCC 12.9.86. `pip check` reports these
two conflicts. JAX device discovery and the MuJoCo interface probe pass, but a
short Dreamer training smoke test is required before a long teacher run.

## Marvin paths

| Purpose | Path |
| --- | --- |
| Project checkout mirror | `/home/oliver/Code/airhockey-memory-distillation` |
| Separate upstream clones | `/home/oliver/Code/upstream/airhockey-memory-distillation` |
| Run and video artefacts | `/home/oliver/experiments/airhockey-memory-distillation` |
| Audit provenance | `/home/oliver/experiments/airhockey-memory-distillation/provenance/audit-2026-08-11` |
| Rebuild provenance | `/home/oliver/experiments/airhockey-memory-distillation/provenance/rebuilt-images-2026-08-11` |

Marvin has no Slurm or PBS command installed. Runs currently execute directly
through SSH and Docker. The experiment path has about 2.6 TiB free on its
`/home` filesystem, but it is not yet a durable or backed-up artefact store.

## Reconstructed current image

Run the build on Marvin after placing clean upstream clones at the paths above:

```bash
cd /home/oliver/Code/airhockey-memory-distillation
./scripts/build_upstream_images.sh
```

The build script verifies each source commit and refuses dirty upstream
checkouts. It then builds from temporary `git archive` trees. The project patch
contains only three compatibility fixes:

1. close the environment only when the agent workflow receives an environment
   instance rather than a factory;
2. preserve the original challenge `step` and `reset` methods when multiple
   training environments are constructed in one process;
3. correct the one-item `argparse` choice tuple for the spacer agent.

The patch is recorded in
[`patches/drl_air_hockey_marvin.patch`](../patches/drl_air_hockey_marvin.patch).
The source commits and Dreamer revision are also embedded as OCI labels. The
build disables mutable BuildKit provenance attestations so repeated cached
builds retain the same runtime image manifest digest.

Re-run the interface probe with:

```bash
docker run --rm \
  -v /home/oliver/Code/airhockey-memory-distillation:/work:ro \
  marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt \
  python3 /work/scripts/audit_upstream_interface.py
```

## Historical demonstration

The supplied pre-trained self-play demonstration was reproduced from:

```text
andrejorsula/drl_air_hockey@sha256:16356368969865589a47d815077c0694b4afe2db042a21359dd857893beece65
```

The script runs inference on CPU, renders through an Xvfb display and records
the display with FFmpeg:

```bash
STEPS=300 ./scripts/reproduce_upstream_demo.sh \
  /home/oliver/experiments/airhockey-memory-distillation/upstream-demo-2023-rendered-v2
```

The two balanced agents loaded the supplied checkpoint successfully. Each
reported 7,530,280 world-model variables, 1,052,676 actor variables and
1,181,439 critic variables. The final video is H.264, 1920 by 1080, 30 frames
per second and 13.07 seconds long.

| Artefact | SHA-256 |
| --- | --- |
| Trimmed video `self_play.mp4` | `815c8c65e4ce9274e46cdbabdf224d71b4a99ce83e2d68c1309eaf4a63c3acab` |
| Raw display capture `self_play_raw.mp4` | `d4625d2d17d4a66764ec778bfdfcc1343aa2b069640549bad190969d3dc25424` |

The script removes the black interval while JAX traces and the checkpoints
load. It refuses to overwrite existing video files. The video and logs remain
outside Git in the Marvin experiment directory.

## Decisions and remaining checks

- Use the pinned 2025 MuJoCo source as the base for the set-piece task.
- Keep the 2023 image only as a reproduction reference; do not use its
  checkpoint as the teacher.
- Implement the 19-observation, 2-action public adapter before teacher work.
- Use fixed impedance values for all policies and validate them with the
  privileged and inactive controls.
- Resolve or explicitly validate the Dreamer JAX declaration mismatch with a
  short training run before committing substantial GPU time.
- Choose a durable backup location before generating teacher checkpoints or
  large trajectory datasets.
