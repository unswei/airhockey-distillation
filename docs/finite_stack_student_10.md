# Ten-step finite-stack student

Status: **implemented; untrained**.

This is the predeclared finite-memory baseline for the principal student
sweep. It exposes exactly ten public puck-position/visibility triples to a
feed-forward policy. It provides more history than the observation-only
baseline without a learned recurrent state.

## Input and carry

At each control step, the policy shifts its 30-value carry by one triple and
appends the current public `(puck_x, puck_y, visible)` values. The history is
ordered from oldest to newest, so the current triple occupies the final three
positions. The network input concatenates:

- the current 16 proprioceptive values: seven joint positions, seven joint
  velocities and two defender-mallet coordinates;
- ten three-value puck triples, totalling 30 values.

This gives a 46-value input. There is no puck velocity or opponent state, and
previous action is not an input. During a blackout, the public observation
adapter supplies `(0, 0, 0)`, which is appended normally while older history
remains intact. Episode start is padded with ten `(0, 0, 0)` triples. Only a
true episode boundary resets the carry.

## Network and accounting

The policy is a `46--64--32--64--2` MLP. All hidden layers use SiLU and the
two action means use `tanh`. It has 7,330 trainable parameters and no learned
recurrent core. Its complete persistent carry is 30 float32 values, or 120
bytes.

The canonical architecture configuration is
`configs/student/finite_stack_10.yaml`.

## Implementations and export

`src/airhockey_distill/students/finite_stack.py` provides NumPy batch-one and
batched inference, explicit carry advancement, contiguous-sequence evaluation
and self-describing NPZ checkpoint load/save.

`src/airhockey_distill/students/finite_stack_torch.py` provides matched
single-step and sequence PyTorch execution for training. It exports float32
NumPy parameters directly into the framework-neutral policy and checkpoint
format. Importing the ordinary student package does not require PyTorch.

Tests verify stack order and capacity, zero padding, blackout behaviour,
episode-boundary reset semantics, batched execution, architecture accounting,
NumPy/PyTorch agreement and exact checkpoint reload. These are implementation
checks only; no finite-stack training result exists yet.
