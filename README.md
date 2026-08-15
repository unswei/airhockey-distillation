# Air-hockey memory distillation

How much memory does a robot need when it temporarily loses sight of a moving
object?

This project studies that question in simulated robot air hockey. A defending
robot observes an incoming puck, loses puck tracking for up to 400 ms, and must
keep moving before vision returns. A policy that only sees the current frame
cannot know which way an unseen puck is travelling, so successful defence
requires information from earlier observations.

<p align="center">
  <a href="docs/assets/upstream-self-play-2023.mp4?raw=1">
    <img
      src="docs/assets/upstream-self-play-2023-preview.gif"
      alt="Two KUKA iiwa robots playing simulated air hockey"
      width="900"
    >
  </a>
</p>

<p align="center">
  <sub>Simulated robot air hockey. Open the preview to view the full video.</sub>
</p>

## What we are testing

We first train a strong recurrent DreamerV3 policy to defend the goal during
temporary tracking loss. We then distil its behaviour into smaller policies
and compare several ways of retaining history:

- no memory;
- a fixed window of recent observations;
- a conventional GRU;
- structured recurrent memory with mostly linear dynamics and a small
  nonlinear correction.

The structured students vary the number of nonlinear recurrent channels
(`k = 0, 1, 2, 4`). This lets us test whether the task needs complex recurrent
dynamics, or whether simple state propagation with a small learned correction
is enough.

Policies are compared on closed-loop save rate as tracking loss becomes
longer, as well as parameter count, recurrent-state size, action agreement and
single-step inference time.

## Current status

The seven-family, five-seed principal validation sweep is complete. At the
longest 20-step blackout, structured `k=0,1,2,4` and GRU-64 all save about
97% of shots, while the feed-forward and finite-stack policies save 42.2% and
85.3%. The structured policies use roughly 12,000 parameters, compared with
28,898 for GRU-64.

The canonical structured batch-one implementation has also been optimised
without changing its outputs. Its five-seed median Marvin latency is 23.03
microseconds for `k=0` and 28.20--28.31 microseconds for `k=1,2,4`, versus
40.06 microseconds for GRU-64. All 20 structured validation reruns reproduced
the earlier episode rows exactly.

These are validation results. The release gate has not yet been applied and
the principal test split remains unopened.

## Experimental task

Each episode contains one incoming shot and one defending KUKA iiwa robot. The
policy receives robot proprioception, puck position when visible, and a
visibility flag. During a blackout, puck position is removed. The policy never
receives puck velocity or privileged simulator state.

Evaluation uses matched shots and blackout schedules so that every policy
faces the same situations. Special paired shots create the same observation at
the start of a blackout while requiring different defensive movements; these
pairs make the need for observation history directly testable.

## Repository contents

```text
configs/                 Environment, model and experiment definitions
src/airhockey_distill/   Task, policy and evaluation implementations
scripts/                 Training, evaluation and reproduction entry points
tests/                   Unit and simulator integration tests
docs/                    Method, provenance and reproducibility notes
results/                 Compact, versioned result summaries
```

Large datasets, checkpoints and raw evaluation runs are stored outside Git.
Versioned configurations, hashes and compact result summaries keep those
artefacts traceable.

## Reproducibility

The project pins the upstream source revisions and records the software
environment used for experiments. See [UPSTREAM.md](UPSTREAM.md) for dependency
provenance and [scripts/README.md](scripts/README.md) for the available command
line entry points.

The main implementation builds on:

- [Learning to Play Air Hockey with Model-Based Deep Reinforcement Learning](https://github.com/AndrejOrsula/drl_air_hockey)
- [Robot Air Hockey Challenge](https://github.com/AndrejOrsula/air_hockey_challenge)
- [DreamerV3](https://github.com/danijar/dreamerv3)

## Scope

This is research software for a controlled simulation study. It evaluates
policy behaviour under temporary perception loss; it does not claim
sim-to-real performance, safety guarantees or general-purpose robot memory.
