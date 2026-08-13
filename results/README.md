# Results

Git tracks result schemas, manifests, hashes and retrieval instructions here;
large or generated result files are ignored.

The compact canonical gate results are
[`stage_b_memory_validation_v3.json`](stage_b_memory_validation_v3.json) and
[`causal_memory_ablation.json`](causal_memory_ablation.json). They contain
gate-level summaries only. Their episode records remain outside Git at the
hashed Marvin runs documented in
[`docs/stage_b_memory_validation_v3.md`](../docs/stage_b_memory_validation_v3.md)
and
[`docs/causal_memory_ablation.md`](../docs/causal_memory_ablation.md).

The structured-student plumbing gate is preserved as both its original
[`v1 NO_GO`](structured_n64_k2_tiny_overfit_v1.json) and corrected
[`v2 GO`](structured_n64_k2_tiny_overfit_v2.json). These are compact training
summaries without the dataset or checkpoint. Their frozen raw paths and hashes
are documented in
[`docs/structured_student_n64_k2.md`](../docs/structured_student_n64_k2.md).

The first [`full-data seed`](structured_n64_k2_full_seed_14303_v2.json)
records its deterministic-mean dataset binding, offline losses and compact
closed-loop summary. The run completed technically, but its 45.3% validation
save rate is weak and the checkpoint is not promoted as a successful policy.
The follow-up
[`failure diagnostic`](structured_n64_k2_failure_diagnostic_v1.json) records
the phase errors, controlled-prefix comparison, history-distribution shift and
dataset action-semantics audit. Its raw episode records remain outside Git.
The subsequent [`all-step correction`](structured_n64_k2_all_steps_v2.json)
improves no-blackout saves to 67.1% but returns `NO_GO` at the predeclared 75%
gate. Only the no-blackout validation condition was opened.

Canonical episode rows will include:

```text
policy,teacher_id,student_seed,eval_seed,shot_id,blackout_steps,
outcome,return,goal_conceded,puck_contact,clear,timeout,safety_fault,
action_mse,episode_steps,policy_latency_ms
```

Raw result files are immutable. Corrections produce a new version rather than
overwriting an existing result.
