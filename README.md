# Air-hockey memory distillation

This repository contains the software and experiment definitions for
**Distilling Recurrent World-Model Policies for Robot Air Hockey under
Tracking Loss**.

The project studies whether a DreamerV3 defence policy can be distilled into
a compact recurrent policy whose memory has diagonal linear dynamics and only
`k = 0, 1, 2, or 4` nonlinear innovation channels. The core task is a
controlled incoming shot with a single temporary loss of puck tracking.

## Repository boundary

This repository contains code, configuration, tests, provenance records,
result schemas and figure/table generation. The LaTeX paper is maintained in a
separate Git repository. The parent project directory is only an unversioned
umbrella containing links to the two repositories and the private project
brief.

Large datasets, checkpoints, raw run directories and videos must not be added
to Git. Store them in an appropriate artefact location and version their
hashes and retrieval instructions here.

## Execution environment

Development files may be edited locally, but code and experiments are run on
the `marvin` Linux box. Marvin's exact checkout path, accelerator details,
container digest and package versions will be recorded during the upstream
audit; see [`docs/marvin.md`](docs/marvin.md).

No GitHub remote or deployment path is configured yet.

## Planned layout

```text
configs/                 Versioned environment and experiment definitions
src/airhockey_distill/   Environment, teacher, student and evaluation code
scripts/                 Reproduction and Marvin execution entry points
tests/                   Unit and integration tests
docs/                    Upstream and execution-environment audits
results/                 Result schema and retrieval metadata only
artifacts/                Generated paper figures/tables (ignored by Git)
STATUS.md                 Commands, results, decisions and blockers
UPSTREAM.md               Dependency commits and licence obligations
```

The first implementation milestone is a minimal vertical slice: one shot, one
blackout, teacher inference, one `k = 2` student and one evaluation episode.

## Scientific guardrails

- Every policy receives the same non-privileged public observation.
- No puck velocity, observation stack or hidden simulator state enters the
  principal teacher or recurrent students.
- Recurrent state and previous action reset only at episode boundaries.
- Final test seeds are never used for training or architecture selection.
- Raw data and results are immutable; plots and tables are derived by script.
- Experimental values and citations remain explicit TODOs until verified.

See [`STATUS.md`](STATUS.md) for the current state of the project.
