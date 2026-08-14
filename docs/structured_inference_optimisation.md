# Structured inference optimisation

The canonical structured policies deliberately use a fixed float32 pairwise
reduction tree. This makes the NumPy export agree with the PyTorch model, but
constructing and reducing a temporary NumPy tensor for every linear map is
slow for batch-one control.

The optional `native_pairwise_float32_v1` backend implements only those
pairwise linear reductions in a small CPython extension. Activations, state
updates and the policy interface remain in NumPy. The C loop uses the same
float32 products, the same balanced addition tree and the same final bias
addition as the canonical implementation. Floating-point contraction and fast
math are disabled when it is compiled.

This is an implementation-only change. It does not alter checkpoints,
parameters, actions, recurrent state, evaluation schedules, outcome rules or
release thresholds. The NumPy implementation remains the default. Evidence
runs must explicitly set
`AIRHOCKEY_STRUCTURED_PAIRWISE_BACKEND=native`; this setting fails closed if
the compiled extension is missing.

Build the extension on Marvin with:

```bash
python scripts/build_structured_native_kernel.py \
  --output-directory /path/to/frozen/native \
  --manifest /path/to/frozen/native-build.json \
  --code-commit FULL_GIT_COMMIT
```

The build manifest records hashes of the source, compiler and output binary,
as well as the exact flags and Python ABI. Run
`scripts/verify_structured_native_kernel.py` against every frozen structured
checkpoint before using the backend for validation or latency measurement.
That verifier requires bitwise agreement with the canonical NumPy calculation
for direct linear-map trials and closed-loop policy steps.
