# Marvin execution environment

`marvin` is the canonical Linux host for running this repository and all
experiments. Local workstation results are development diagnostics only unless
explicitly labelled otherwise.

## Setup still to audit

- SSH host/connection method
- Repository checkout path
- Scratch path for datasets, checkpoints and run directories
- Durable artefact-store path and backup policy
- Scheduler or interactive execution convention
- CPU, GPU, memory, driver and CUDA details
- Container runtime and exact image digest
- Python/package environment and lock-file strategy

Do not commit credentials, SSH configuration, host tokens or machine-specific
secret files. Paths that differ by machine should be supplied through an
ignored environment file or command-line configuration, while the resolved
non-secret configuration is captured with every run.

## Required provenance for every run

Each run directory must contain the resolved configuration, repository commit
and dirty-state flag, upstream commits, seeds, package versions, hardware
description, checkpoint hash, metrics and stdout/stderr log.

## Planned source flow

Until the GitHub repository exists, no automatic deployment is configured.
Once it exists, Marvin should use a normal clone of this code repository. Raw
experiment products must remain outside the checkout and outside Git history.
