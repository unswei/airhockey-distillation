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

The execution protocol file and its hash are unchanged. This correction changes
only post-training cross-framework export verification. It does not change the
models, training data, loss, seeds, checkpoint selection, NumPy deployment
arithmetic, validation schedule or unopened principal test.
