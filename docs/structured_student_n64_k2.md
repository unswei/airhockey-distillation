# Structured recurrent student: n=64, k=2

Status: **implemented and tested; not yet trained**.

This is the first Phase 3 vertical-slice student. It implements the principal
structured recurrence from the project brief with a 64-dimensional memory and
a two-dimensional nonlinear innovation. The versioned configuration is
`configs/student/structured_n64_k2.yaml`.

## Policy definition

The policy encodes the 19-value public observation with a 19--64--32 MLP and
SiLU activations. Its recurrent update is:

```text
z_t = tanh(alpha) * z_(t-1) + B_x x_t + B_a a_(t-1) + b_z
      + U tanh(V z_(t-1) + W_x x_t + W_a a_(t-1) + b_r)
```

Here `z` has dimension 64, `U` has shape 64 by 2 and `V` has shape 2 by 64.
The linear state dynamics are diagonal. Their entries are `tanh(alpha)`, so
their magnitude remains below one. Initial positive entries span time
constants from 40 ms to 2 s at the 20 ms control period.

The action head receives the concatenated 64-value updated state and 32-value
encoded observation. A 96--64--2 SiLU MLP emits a tanh-bounded deterministic
action mean. The previous executed two-dimensional action enters both the
linear and nonlinear recurrent branches. Memory and previous action are reset
only by explicitly constructing the initial carry at an episode boundary.

## Implementations

`src/airhockey_distill/students/structured.py` is the framework-neutral NumPy
evaluation policy. It provides single-step and batched inference,
teacher-forced contiguous sequences, explicit closed-loop carry, analytical
recurrent Jacobians and self-describing NPZ checkpoints.

`src/airhockey_distill/students/structured_torch.py` is the matching trainable
PyTorch module. It supports batched sequence unrolling for truncated
backpropagation through time and exports its parameters directly to the NumPy
checkpoint format. PyTorch remains an optional `student` dependency; importing
the ordinary student package does not require it.

## Fixed accounting

| Quantity | Value |
| --- | ---: |
| Public observation | 19 values |
| Encoded observation | 32 values |
| Recurrent state | 64 values |
| Nonlinear innovation rank | 2 |
| Public action | 2 values |
| Total trainable parameters | 12,328 |
| Recurrent-core parameters | 2,630 |
| Float32 recurrent state | 256 bytes |

The state-dependent recurrent Jacobian is
`diag(tanh(alpha)) + U D V`. Automatic differentiation tests compare this
against the analytical Jacobian and confirm that the numerical rank of the
departure from the fixed diagonal dynamics is at most two.

## Verified contracts

The focused tests cover:

- fixed `n=64`, `k=2` shapes and parameter counts;
- declared 40 ms to 2 s diagonal initialisation;
- explicit episode-boundary resets and continued carry between steps;
- previous-action input to both recurrent branches;
- batched and scalar inference agreement;
- autodiff and analytical Jacobian agreement with rank at most two;
- exact action and state sequences after NPZ checkpoint reload;
- PyTorch and NumPy runtime agreement;
- finite, non-zero gradients through the recurrent core and action head.

The full pinned Marvin container suite passes 89 tests. This does not establish
student quality. The next gate is to overfit a tiny set of complete teacher
episodes using the configured 64-step sequences, 16-step burn-in and 48-step
loss suffix. The existing Stage B v2 dataset stores executed sampled teacher
actions, not deterministic teacher means. The overfit diagnostic therefore
needs a small newly collected deterministic-mean target set; the old target
semantics must not be relabelled. Only after that passes should the full
dataset be collected or trained.

The diagnostic is predeclared in
`configs/student/structured_n64_k2_tiny_overfit.yaml`: 16 complete episodes,
128 padded steps, a 16-step burn-in and full-batch AdamW. It passes only if
training action MSE reaches `1e-4`, loss falls by at least 99%, NumPy export
error is at most `5e-6`, and checkpoint reload is exact.
