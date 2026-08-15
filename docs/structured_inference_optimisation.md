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

## Frozen V4 evidence

The implementation commit is
`6f54e18219e3b478d1b6e0c2bddf62375925cf7c`. The successful Marvin evidence
is frozen outside Git under
`principal-sweep-v1-2026-08-15-v4-attempt2`. Its compact manifest SHA-256 is
`18b3c8d5034290b087c79a427bf1daf87654d1c57b31980d0ddd5b1222cb714a`.

The verifier covered 237,056 direct scalar results and 2,000 closed-loop steps
for each of 20 structured checkpoints with zero bit disagreement. All 20
paired validation reruns reproduced the V3 episode rows exactly. The new
isolated latency session measured all 35 models. Relative to V3, structured
latency fell by 62.5% for `k=0` and 64.1--65.0% for `k=1,2,4`.

The first V4 orchestration attempt stopped before validation because a
container-owned result could not be frozen by the host user. It is separately
sealed as failed engineering evidence and was never resumed. Attempt 2 uses a
container-scoped freeze operation. Neither attempt constructed or evaluated
`principal_test`.
