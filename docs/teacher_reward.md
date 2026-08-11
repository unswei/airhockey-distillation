# Teacher reward v1

Validated: 2026-08-11

`defend_shot_v1` replaces the upstream defence environment's identically zero
reward. It uses contact and terminal outcomes available inside the environment;
none of these evaluation-only signals are added to the policy observation or
public step information.

## Definition

| Event or outcome | Reward |
| --- | ---: |
| First puck--mallet contact in an episode | +0.2 once |
| `returned` | +1.0 |
| `arrested` | +1.0 |
| `safe_deflection` | +1.0 |
| `goal_conceded` | -1.0 |
| Miss without contact | 0.0 |
| Bare or post-contact timeout | 0.0 |
| Other unresolved upstream termination | 0.0 |

Rewards on the same control step are additive. A first-contact return is
therefore +1.2, while a first-contact concession is -0.8. A later return is
+1.0 because the contact bonus was already issued. Misses and timeouts are not
silently treated as saves.

The reward intentionally contains no hidden-position shaping, predicted
trajectory, distance-to-intercept or future information. Safety and
action-rate penalties remain zero until their underlying signals and scales are
defined. Reward engineering should stay modest unless a learning diagnostic
identifies a specific failure.

The frozen configuration is
[`configs/reward/defend_shot_v1.yaml`](../configs/reward/defend_shot_v1.yaml).

## Validation

Pure tests cover every terminal category, first-contact-once behaviour,
same-step addition and reset between episodes. A pinned-MuJoCo integration test
confirms that the privileged return receives a total episode reward of +1.2.
The complete Marvin suite passes 47 tests.

The 1,000-step DreamerV3 reward smoke test completed 108 optimiser updates and
logged 21 non-zero episode scores:

| Score | Episodes | Interpretation |
| ---: | ---: | --- |
| -1.0 | 16 | Concession without contact |
| +0.2 | 2 | Contact followed by no conclusive terminal save |
| +1.2 | 3 | First contact and conclusive save |

All numeric metrics were finite, and the latest checkpoint recorded step
1,000. This validates reward delivery and training compatibility, not policy
quality.

The evidence is outside Git at:

```text
/home/oliver/experiments/airhockey-memory-distillation/teacher-smoke-2026-08-11-v6
```

| File | SHA-256 |
| --- | --- |
| `smoke_result.json` | `22f7fe32738ff41016413519ff47de6219f4ac347f6c9bc12450b6a05a38f5df` |
| `dreamer/metrics.jsonl` | `51d03b235ab3bc89fe06ac7d5175bc9527954d8cf6bdb41c45b93384c8e00d54` |
| `dreamer/scores.jsonl` | `9ffdb6e91d80446f5121ab118231add54f478946e9aa698e406bc3910b2e75e3` |
| `dreamer/config.yaml` | `c5b3dbeb069f61b558d974c005abb9853be4c82fafb2d4155fbccf0d8ada0c5a` |
| Final `agent.pkl` | `2b2bc9d8bd76f2de4cd4fb1e1416fdea578b7a8600be50b316527a300a1ff52a` |
| Console log | `fab2c5f6bc2788396fc7ad21a7bd898824719960e5d6f25e5777946d72a6ae21` |
