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
The first deterministic
[`shadow-teacher round`](structured_n64_k2_shadow_round1_v1.json) then returns
`GO`: the same architecture and seed save 222/225 no-blackout shots. The
subsequently opened paired evaluation records 1,107/1,125 saves across the
five blackout lengths. Its 1,125 episode rows remain outside Git in the
frozen Marvin run.

The matched GRU-64
[`tiny-overfit gate`](gru_n64_tiny_overfit_v1.json) records deterministic-mean
training, PyTorch/NumPy parity and exact framework-neutral checkpoint reload.
It returns `GO`, but is an engineering result rather than a full-data policy.
The subsequent [`full-data GRU pilot`](gru_n64_full_pilot_seed_14303_v1.json)
records one seed trained on the current 40,000-episode aggregate. It is an
offline engineering result only because the shadow half of that aggregate was
collected under the structured `k=2` policy.
Its staged [`behavioural evaluation`](gru_n64_full_pilot_evaluation_v1.json)
passes the 225-shot no-blackout gate at 99.1%, then records 98.4% saves across
1,125 paired blackout episodes. This is closed-loop evidence for the frozen
checkpoint, but the dataset provenance still prevents a fair recurrent-family
comparison. The episode rows remain outside Git in the frozen Marvin runs.

Canonical episode rows will include:

```text
policy,teacher_id,student_seed,eval_seed,shot_id,blackout_steps,
outcome,return,goal_conceded,puck_contact,clear,timeout,safety_fault,
action_mse,episode_steps,policy_latency_ms
```

Raw result files are immutable. Corrections produce a new version rather than
overwriting an existing result.

The compact
[`seven-family engineering dry-run result`](principal_engineering_dry_run_v1.json)
hash-binds the canonical Marvin run and its seven checkpoints. It is a
non-principal plumbing `GO`: its tiny-data losses, four-rollout validation
samples and reduced latency measurements are expressly ineligible for the
principal release manifest.

The compact
[`structured inference optimisation result`](principal_structured_optimisation_v4.json)
records the bit-exact V4 native kernel, identical structured validation rows,
all 35 new isolated latency measurements and the hashes of the frozen external
evidence. This is pre-release validation evidence; it does not itself open the
principal test split.

The compact frozen
[`principal-test statistics`](principal_sweep_v1_statistics.json) contain the
five-seed save-rate curves, paired hierarchical bootstrap intervals and
predeclared 400-ms contrasts used for the principal paper figure. The tracked
file is an exact copy of the validated Marvin analysis product and has SHA-256
`55ad1f50399e54487728ade10e6605a993064369a80ec48c0674704f2f378902`.
Together with the compact V4 structured-inference result, it also supplies the
five-seed performance and isolated latency measurements used in the
performance--cost frontier figure.

The frozen analysis also emitted the complete seven-family
[`efficiency table`](principal_sweep_v1_efficiency_table.csv) used for the paper's
comparison table. It defines overall performance over the five predeclared
0--400 ms conditions and has SHA-256
`a6e8edc0ebb541c3922b9b589926527519045f370c46c370c78fc749392e9311`.
