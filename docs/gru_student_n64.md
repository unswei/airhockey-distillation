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

## Full-data engineering pilot

One GRU-64 seed was trained on the frozen 40,000-episode aggregate used by the
successful structured vertical slice. Seed 14303, the episode split, all-step
loss, optimiser, learning rate, batch size, validation selection and 100-epoch
budget match the structured run. The aggregate manifest SHA-256 is
`e67438a24da12cfc1aed92f6df64253f82ce000cb52c3085d28468007ebf0d0d`.

Epoch 97 was selected at validation action MSE `0.04521`. Train and internal
test MSE were `0.04402` and `0.04551`; internal-test hidden-step MSE was
`0.00778`. Maximum PyTorch/NumPy error was `3.1590e-6` for actions and
`1.0729e-6` for states, and checkpoint reload was exact. Training took 1,366.7
seconds on Marvin.

This is an engineering pilot, not a fair GRU-versus-structured comparison.
Half of the aggregate consists of deterministic labels on trajectories
controlled by the structured `k=2` policy. The result establishes that the
GRU full-data path works and gives an offline imitation measurement. It does
not establish relative quality between recurrent families.

The checkpoint and raw logs are root-owned and read-only on Marvin. Their
six-file checksum manifest has SHA-256
`33d9745099c796dbf5c8faf0bbaf9b31c752b81aeea8491f5f3b073f910e8ee8`.
The compact result is `results/gru_n64_full_pilot_seed_14303_v1.json`.

## Staged behavioural engineering evaluation

The frozen checkpoint was first gated on the same 225 no-blackout validation
shots used for the structured student. It saved 223/225 shots (99.1%), with
two contact timeouts kept separate from genuine saves, no concessions and no
simulator or safety faults. This exceeded the predeclared 75% threshold and
returned `GO`.

Only after that result was hashed did a separate configuration open the five
paired blackout lengths. Across 1,125 episodes, the unchanged checkpoint
saved 1,107 (98.4%) and conceded three. Save rates at blackout lengths 0, 5,
10, 15 and 20 were 99.1%, 100.0%, 98.2%, 97.3% and 97.3%. The 225 no-blackout
episode records reproduced the gate evaluation exactly.

This demonstrates strong closed-loop behaviour for this frozen engineering
checkpoint. It is still not a fair GRU-versus-structured comparison because
the shared aggregate's shadow half was collected under the structured
student. A principal comparison needs data whose collection policy does not
favour either recurrent family.

The compact evaluation result is
`results/gru_n64_full_pilot_evaluation_v1.json`. The two raw evaluation
directories are root-owned and read-only on Marvin. Their eight-file checksum
manifest has SHA-256
`e1239cb77156a730be8d12b43140550dbc9c5e1f049ca9e8f854b74e2a84bcf8`;
the 1,125 episode rows remain outside Git.
