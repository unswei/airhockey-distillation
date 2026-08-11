# Test plan

Tests must cover observation masking and leakage, visibility intervals, shared
policy observations, recurrent reset semantics, the `k = 0` branch, numerical
Jacobian rank, parameter accounting, checkpoint round trips, paired shot
replay, row-order-invariant aggregation and a clean 20-episode CPU smoke run.

The minimal task slice currently tests masking, visibility timing, removal of
privileged puck velocity, the public-info boundary, two-to-six dimensional
action expansion, deterministic replay and fail-closed teacher-gate semantics.
The direct-launch distribution tests freeze its count, stratification,
coverage, timing construction and deterministic split generation. The
contact-aware outcome tests cover concession precedence, return, arrest, safe
deflection, non-contact exits and both timeout categories. Reward tests cover
every terminal mapping, first-contact-once semantics and episode reset. The
`integration` tests additionally require the pinned upstream MuJoCo environment
and run in the audited Marvin container, including the real-contact +1.2
return. Later phases must add the remaining recurrent, checkpoint, aggregation
and multi-episode tests listed above.
