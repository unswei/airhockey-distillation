# Project status

Last updated: 2026-08-10

## Current phase

Phase 0 — source audit and reproduction.

## Completed

- Split the work into independent code and paper repositories.
- Initialised both repositories on the `main` branch without remotes.
- Created the initial code, configuration, test and provenance layout.
- Recorded `marvin` as the canonical Linux execution host.
- Kept large experiment products outside Git history.

## Next actions

- Inspect the upstream READMEs, licences, Dockerfiles, training scripts and
  agent interfaces.
- Pin the exact `drl_air_hockey`, `air_hockey_challenge` and DreamerV3 commits.
- Reproduce the upstream pretrained demonstration on Marvin.
- Record the actual observation/action interfaces and software environment in
  `docs/upstream_audit.md`.

## Commands and results

| Date | Command or action | Result |
| --- | --- | --- |
| 2026-08-10 | Created repository scaffold from the umbrella project brief | Complete |
| 2026-08-10 | `git init -b main` in the resolved `Code` and `Paper` targets | Two independent worktrees; umbrella remains outside Git |
| 2026-08-10 | Selected `marvin` as the code and experiment runtime | Recorded; connection and runtime audit pending |
| 2026-08-10 | Parsed all draft YAML and TOML; compiled the Python package | Passed |

## Blockers

- Marvin checkout, scratch and artefact-store paths have not yet been chosen.
- Upstream commits, licences and the container/runtime stack have not yet been
  audited.
