# Contact-aware outcomes

Implemented: 2026-08-11

The task now records actual MuJoCo contact between the upstream `puck` and
`robot_1/ee` collision groups. Contact is checked after every 1 ms simulation
substep and accumulated over the 50 Hz control step. It is evaluation-only
state: the policy observation and public step information do not reveal
contact, contact time or remaining episode duration.

## Terminal outcomes

| Outcome | Condition | Counts as privileged save |
| --- | --- | --- |
| `goal_conceded` | Puck crosses the defending goal plane inside the opening | No |
| `returned` | Defender contact occurred, then puck crosses table x=-0.55 m with vx at least 0.15 m/s towards the opponent | Yes |
| `arrested` | Defender contact occurred and puck speed is at most 0.1 m/s for three control steps, or the upstream stop condition confirms it | Yes |
| `safe_deflection` | Defender contact occurred and puck exits the defending end outside the goal, or exits a side boundary | Yes |
| `missed_goal_without_contact` | Puck exits outside the goal without defender contact | No |
| `timeout_after_contact` | Contact occurred but no conclusive save or concession followed | No |
| `timeout_without_contact` | Episode timed out untouched | No |
| `upstream_terminal_after_contact` | Upstream stopped for another unresolved post-contact condition | No |
| `upstream_terminal_without_contact` | Upstream stopped for another unresolved untouched condition | No |

Concession takes precedence over a contact in the same control step, so a late
touch cannot convert a goal into a save. Returns terminate before the wrapper
timeout rather than being inferred from the final timeout state.

## Validation

Pure tests cover every outcome above, stable arrest confirmation, contact
precedence, early return termination and timeout separation. A pinned-MuJoCo
integration test confirms that the original privileged control produces a real
puck--mallet contact and terminates as `returned` before 2.5 seconds. The full
Marvin suite passes 32 tests.

On the 216-shot `direct_launch_v1` calibration manifest, the privileged
controller saved every shot:

- 205 returned;
- 11 arrested;
- 0 bare or post-contact timeouts;
- 0 concessions.

Its measured save rate is therefore 100%, above the 80% gate. No privileged
controller change or removal of unsaveable shots is needed.
