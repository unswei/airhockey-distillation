# Results

Git tracks result schemas, manifests, hashes and retrieval instructions here;
large or generated result files are ignored.

Canonical episode rows will include:

```text
policy,teacher_id,student_seed,eval_seed,shot_id,blackout_steps,
outcome,return,goal_conceded,puck_contact,clear,timeout,safety_fault,
action_mse,episode_steps,policy_latency_ms
```

Raw result files are immutable. Corrections produce a new version rather than
overwriting an existing result.
