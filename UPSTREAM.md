# Upstream provenance

This file records the exact upstream sources used by the project. The detailed
inspection and reproduction results are in
[`docs/upstream_audit.md`](docs/upstream_audit.md).

| Component | Repository | Commit | Licence/obligations | Status |
| --- | --- | --- | --- | --- |
| `drl_air_hockey` | <https://github.com/AndrejOrsula/drl_air_hockey> | `a41081c4c3860877d7553c7eb9e84c1b7f5f7da0` | MIT; retain Andrej Orsula's copyright and permission notice | Pinned from `main` |
| `air_hockey_challenge` | <https://github.com/AirHockeyChallenge/air_hockey_challenge> | `34729081d4327141dcf560c0fb8274cd4882e82c` | MIT; retain the 2025 authors' copyright and permission notice | Pinned from `qualifying-2025` |
| DreamerV3 fork used upstream | <https://github.com/AndrejOrsula/dreamerv3> | `4049794d4135e41c691f18da38a9af7541b01553` | MIT; retain Danijar Hafner's copyright and permission notice | Matches the current upstream Dockerfile |

The old `AndrejOrsula/air_hockey_challenge` link is no longer anonymously
cloneable. The public organisation repository above supplies the inspected
2025 challenge code.

## Runtime provenance

Canonical execution host: `marvin`

- Linux distribution/kernel: Ubuntu 24.04, `6.17.0-1030-oem`, x86-64
- CPU: Intel Core Ultra 9 285, 24 cores; 30 GiB RAM
- GPU: NVIDIA GeForce RTX 5090, 32607 MiB
- NVIDIA driver/CUDA: driver 595.84; CUDA 12.9.1 base
- Python: 3.12
- MuJoCo: 3.3.1
- JAX/jaxlib: 0.5.3/0.5.3
- Container base image: `nvidia/cuda:12.9.1-base-ubuntu24.04`
- Container base digest: `sha256:29e5e3425e2e0f5a4e97c9fb4695ba4887cd78210a43cf94c3bcafc6ab01c5e6`
- Rebuilt final image: `sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f`
- Package freeze: stored with the rebuild provenance on Marvin

The historical demonstration uses the immutable image
`andrejorsula/drl_air_hockey@sha256:16356368969865589a47d815077c0694b4afe2db042a21359dd857893beece65`.
Its 40-dimensional policy interface is not the project teacher interface.
