# Marvin execution environment

`marvin` is the canonical Linux host for running this repository and all
experiments. Local workstation results are development diagnostics only unless
explicitly labelled otherwise.

## Audited host

- SSH: `ssh oliver@marvin`
- OS: Ubuntu 24.04, kernel `6.17.0-1030-oem`, x86-64
- CPU: Intel Core Ultra 9 285, 24 cores, one thread per core
- Memory: 30 GiB RAM and 8 GiB swap
- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 595.84
- Docker: 29.7.1 with the NVIDIA runtime and `overlayfs` storage
- Free space under `/home`: about 2.6 TiB at audit time
- Scheduler: none; `sbatch` and `qsub` are not installed

## Paths

| Purpose | Path |
| --- | --- |
| Code repository mirror | `/home/oliver/Code/airhockey-memory-distillation` |
| Pinned upstream clones | `/home/oliver/Code/upstream/airhockey-memory-distillation` |
| Runs, checkpoints and videos | `/home/oliver/experiments/airhockey-memory-distillation` |

The experiment path is local storage, not a confirmed durable artefact store.
Choose and test a backup destination before producing large datasets or main
teacher checkpoints.

## Current container

The audited tag is
`marvin/drl-air-hockey:2025-a41081c4c386-blackwell-rebuilt`, with immutable
local digest
`sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f`.
It contains Python 3.12, MuJoCo 3.3.1 and JAX 0.5.3. A GPU probe sees one
`CudaDevice`.

Build it from the pinned upstream clones with:

```bash
cd /home/oliver/Code/airhockey-memory-distillation
./scripts/build_upstream_images.sh
```

See [`upstream_audit.md`](upstream_audit.md) for source pins, the package
compatibility caveat and the exact image provenance.

Do not commit credentials, SSH configuration, host tokens or machine-specific
secret files. Paths that differ by machine should be supplied through an
ignored environment file or command-line configuration, while the resolved
non-secret configuration is captured with every run.

## Required provenance for every run

Each run directory must contain the resolved configuration, repository commit
and dirty-state flag, upstream commits, seeds, package versions, hardware
description, checkpoint hash, metrics and stdout/stderr log.

## Source flow

Until the GitHub repository exists, no automatic deployment is configured.
The local repository is mirrored to the path above for this audit. Once the
GitHub repository exists, Marvin should use a normal clone and pull reviewed
commits. Raw experiment products must remain outside the checkout and outside
Git history.
