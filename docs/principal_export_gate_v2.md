# Principal export gate v2

The second principal-sweep attempt stopped before release on
`structured_k0`, seed 14306. The action difference was
`4.559755325317383e-6`, below the declared `2e-5` action limit. The recurrent
carry difference was `2.288818359375e-5`, just above the same absolute limit.

The worst carry comparison was `-30.663333892822266` in PyTorch and
`-30.66335678100586` in NumPy. Its relative error was
`7.464343525498407e-7`. The latent carry is not bounded to the action range, so
a single absolute limit did not represent float32 agreement consistently at
different carry magnitudes.

The action gate remains an absolute `2e-5`. The carry gate is now applied to
every compared value as

```text
abs(torch - numpy) <= 2e-5 + 1e-6 * max(abs(torch), abs(numpy))
```

Exact checkpoint reload is still required. Results retain the maximum absolute
carry error and also record the maximum fraction of the allowed scale-aware
tolerance. A fraction above one fails closed.

The verifier also performs a same-state, one-step comparison. At each timestep,
PyTorch receives the exact previous carry produced by NumPy, so recurrent drift
cannot hide a local implementation disagreement. The resulting action and carry
must each agree within an absolute `2e-5`. This check is independent of the
scale-aware free-running carry check.

Every export result records the worst action case, maximum-absolute-error carry
case, maximum-tolerance-fraction carry case and worst same-state one-step case.
Each record includes the absolute error, state magnitude, validation episode
identity, timestep, dimension and compared values. Both carry records also
include the allowed error and tolerance fraction.

The execution protocol file and its hash are unchanged. This correction changes
only post-training cross-framework export verification. It does not change the
models, training data, loss, seeds, checkpoint selection, NumPy deployment
arithmetic, validation schedule or unopened principal test.

## Sealed regression and corruption validation

Commit `099602c8b12fe804eb8a4da684b7afd2a667572a` re-evaluated the 13
historically successful V2 checkpoints and the stopped `structured_k0`, seed
14306 checkpoint on the same 32 frozen validation episodes. All 14 passed. Seed
14306 retained a maximum absolute carry error of `2.288818359375e-5`, while its
maximum scale-aware tolerance fraction was `0.5576080083847046`; its maximum
same-state one-step carry error was `3.814697265625e-6` and checkpoint reload
remained exact.

Ten deliberate corruption cases were then checked. Parameter changes were
rejected by the action, carry and same-state predicates; a changed on-disk
checkpoint was rejected by the exact-reload predicate; and truncated, missing,
wrong-shape, non-finite and invalid-metadata checkpoints were rejected by their
loaders. All ten failed closed. The action and carry cases each exercised their
intended predicate independently. The same-state recurrent corruption also
violated the free-running carry predicate, so it is supporting evidence for the
one-step check rather than an isolation claim.

The canonical result is
`results/principal_export_gate_v2_validation.json`, SHA-256
`99f52a742875d24590f0de2c1f8fd9ccdb1ad7b3ee643a2d89ca4a6d2ec58d6c`.
The complete evidence, including the nine corrupted checkpoint copies, is
sealed on Marvin at
`gate-validations/principal-export-gate-v2-validation-2026-08-14-v1`; its seal
manifest SHA-256 is
`25978bf2bf8ab02c36fec9990af309620dabb5bbb3918180a89f78f79a445b1c`.
The V2 seal remained unchanged and `principal_test` was not constructed or
evaluated.
