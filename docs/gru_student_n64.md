# Matched GRU-64 student

The conventional recurrent baseline now has matched NumPy and PyTorch
implementations. It uses the same public 19-dimensional observation, 64--32
`SiLU` encoder, 64-value memory, previous requested command and 96--64--2
action head as the structured student. It emits the deterministic `tanh`
action mean and resets its carry only at episode boundaries.

The recurrence is a standard GRU cell with PyTorch's reset, update and new
gate ordering. A unit test copies its parameters into `torch.nn.GRUCell` and
checks the state update directly. Other tests cover single and batched NumPy
steps, previous-action dependence, closed-loop carry, sequence gradients,
PyTorch/NumPy agreement and exact framework-neutral NPZ reload.

The architecture is matched by encoder, head, state dimension and input
contract, not by parameter count. GRU-64 has 28,898 trainable parameters and
19,200 recurrent-core parameters. The structured `n=64`, `k=2` student has
12,328 and 2,630 respectively. Both carry 256 bytes of float32 recurrent
state.

## Tiny deterministic overfit gate

The engineering gate uses the first complete episode from the existing
16-episode deterministic-mean dataset. It does not reuse sampled-action data.
All 32 valid steps contribute action loss, giving 64 supervised action values.
The seed is 13303 and the fixed optimisation settings are recorded in
`configs/student/gru_n64_tiny_overfit_v1.yaml`.

Training reduced action MSE from `0.48862` to `7.7257e-5` at epoch 1150, a
99.984% reduction. Maximum PyTorch/NumPy error was `5.9605e-7` for actions and
`2.9802e-7` for states. Reloading the NPZ checkpoint reproduced both NumPy
sequences exactly. Every predeclared check passed, so the decision is `GO`.

The first invocation at commit `d1b0e3c` failed before dataset loading because
the direct script could not import a shared training helper. Its log is
preserved. Commit `ef6b93b` corrected only that execution path and produced
the successful v2 run. Both run directories are root-owned and read-only on
Marvin. Their seven-file checksum manifest has SHA-256
`a0b0d036135f1b09938f848408c76e1b30541705ccfe013a7c505d99f1c5ef61`.

The compact result is `results/gru_n64_tiny_overfit_v1.json`. This establishes
the GRU training and export path only; the checkpoint is not a full-data
controller and must not be used as a behavioural baseline.
