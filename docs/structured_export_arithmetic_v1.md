# Structured export arithmetic correction

The first principal-sweep attempt stopped during `structured_k0`, seed 14305.
Its selected checkpoint differed between PyTorch and NumPy by
`2.09808349609375e-5` in the recurrent carry. The declared limit was `2e-5`.
The release gate was not created and `principal_test` remained unopened.

The discrepancy came from accumulating small float32 reduction differences
over a recurrent episode. NumPy and PyTorch used different CPU matrix kernels,
so their mathematically equivalent reductions did not always round in the same
order. This was an export defect, not a training failure.

Principal structured checkpoints now declare `canonical_float32_v1` inference.
It uses an explicit balanced float32 reduction tree and explicit float32 SiLU
and tanh operations in both NumPy and PyTorch. Training and model selection
remain on the existing fast PyTorch float32 path. The canonical path is enabled
before the selected metrics and export checks are computed, and the checkpoint
records `training_arithmetic: pytorch_float32` separately.

On the checkpoint that stopped the first attempt, the corrected path gives a
maximum action error of `4.172325134277344e-6` and a maximum carry error of
`1.621246337890625e-5` over the 32 declared export-verification episodes. The
checkpoint reload remains exact. On Marvin, exported batch-one inference took
about 61 microseconds per call in a short pinned single-thread diagnostic,
against the 20 millisecond control period. The principal sweep still performs
the complete predeclared latency measurement for every final checkpoint.

Existing non-principal structured checkpoints remain on the legacy arithmetic
unless their metadata explicitly requests the canonical path. This preserves
the frozen k=2 checkpoint behaviour and hashes.
