# Structured recurrent student: n=64, k=2

Status: **implemented; full-data seed diagnosed; correction required**.

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
action mean. The previous requested two-dimensional command enters both the
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

The full pinned Marvin container suite passes 93 tests. This does not establish
student quality. The tiny-overfit gate used newly collected deterministic-mean
targets with 128 padded steps and a 16-step burn-in. The existing Stage B v2
dataset stores executed sampled teacher actions, not deterministic teacher
means, and was neither reused nor relabelled. With this gate passed, the next
step is to collect the full deterministic-mean dataset and train one full-data
seed.

The diagnostic is predeclared in
`configs/student/structured_n64_k2_tiny_overfit.yaml`: 16 complete episodes,
128 padded steps, a 16-step burn-in and full-batch AdamW. It passes only if
training action MSE reaches `1e-4`, loss falls by at least 99%, NumPy export
error is at most `5e-6`, and checkpoint reload is exact.

That v1 procedure is preserved after returning `NO_GO`: its selected MSE was
`0.00648` across 16 diverse episodes. A diagnostic on one complete episode
reached `3.93e-5`, showing that the training path can overfit but that v1 had
become a capacity test. The versioned v2 correction uses the first complete
episode, learning rate `1e-3`, at most 3,000 epochs and a `1e-5` cross-runtime
tolerance. It retains the `1e-4` MSE, 99% reduction and exact-reload checks.

## Tiny deterministic-mean overfit result

The new dataset contains 16 complete deterministic-inference episodes and 541
transitions. All episode lengths are 31--36 steps and all first previous
actions are zero. Its manifest records
`deterministic_actor_mean_after_public_adapter_clip`; the older sampled-action
dataset remains unchanged.

The original v1 procedure returned `NO_GO` after 2,000 epochs. Across all 16
episodes, MSE fell from `0.89558` to `0.00648`, a 99.28% reduction. Checkpoint
reload was exact, but the near-zero loss and cross-runtime tolerance checks
failed. This result remains in Git and in the frozen raw evidence.

The v2 correction returned `GO` on the first complete 32-step episode:

| Check | Observed | Required |
| --- | ---: | ---: |
| Training action MSE | `7.4562e-5` | at most `1e-4` |
| Fractional loss reduction | 99.988% | at least 99% |
| NumPy/PyTorch maximum error | `2.2054e-6` | at most `1e-5` |
| Exact checkpoint reload | true | true |

The selected checkpoint is from epoch 1,550. An independent rerun produced
byte-identical metrics and the same checkpoint SHA-256:
`ef76db026a4093238d0734bdf44be8b242d998777da7ebe065147e3768e3c0fa`.
This passes the training-path gate; it is not a policy-quality result and this
one-episode checkpoint must not be used as the principal student.

The compact v1 and v2 results are
`results/structured_n64_k2_tiny_overfit_v1.json` and
`results/structured_n64_k2_tiny_overfit_v2.json`. Their SHA-256 hashes are
`b249df1937036c2297e5c76d2ca6b0580f42cd9a149ae17c8cd5972107590f79`
and
`0a2f57df3a54d40af6bf024371753646c315c11db4342db84278f25ca691afb5`.

Raw evidence remains outside Git under Marvin's experiment root:

```text
teacher-datasets/teacher-v3-structured-n64-k2-tiny-deterministic-2026-08-12-v1
students/structured-n64-k2-tiny-overfit-2026-08-12-v1
students/structured-n64-k2-tiny-overfit-2026-08-12-v2
students/structured-n64-k2-tiny-overfit-2026-08-12-v2-repeat
structured-n64-k2-tiny-overfit-2026-08-12-v1
```

All five directories are root-owned and read-only. Their combined checksum
manifest has SHA-256
`6fa46ffd875b8f2848ae0eb874a261e5f79086a033b4125942dad7d65091e188`
and passes after freezing. Other key bindings are:

| Artefact | SHA-256 |
| --- | --- |
| Deterministic dataset manifest | `a55c31c9581a09dcb36571c89ac448ba9bed14b0aa969fd0826e2a53f99dcd1a` |
| Deterministic dataset shard | `78b2f46fb13f96c8ef5ff7c8f33e2562b7b40ffde36c5580d42179340688ea27` |
| v1 failed checkpoint | `29c65ceac47a1168b5520d8f730061f7289e4aa5088d858f4f3d17d80a7d0cae` |
| v2 and repeated-v2 checkpoint | `ef76db026a4093238d0734bdf44be8b242d998777da7ebe065147e3768e3c0fa` |
| Frozen teacher actor | `6a672d5b6d7c2b9ca2335f1a85b69280ca7db58deb5e2f56d2ba033ebf636254` |

Dataset collection used commit
`ab86873105bd8cf013f7215d21b23175f6aed994`. The corrected v2 training and
repeat used commit `4afa8e38cea33eb8a9606110724bb491210efbbc`.

## First full-data seed

The first full-data run was predeclared in
`configs/student/structured_n64_k2_full_seed_14303.yaml`. It collected 20,000
new deterministic-mean episodes and split them by episode seed into 16,000
training, 2,000 validation and 2,000 internal-test episodes. The sampled Stage
B datasets are incompatible inputs and the trainer rejects them.

Training seed 14303 uses 64-step truncated recurrence, a 16-step burn-in, a
48-step loss suffix, batches of 128 episodes, AdamW and global gradient
clipping at 1.0. Episodes longer than 64 steps carry detached recurrent state
into the next chunk; the state is never reset at a truncation or visibility
boundary. The selected checkpoint minimises validation action MSE. Internal
test loss is calculated once after selection and does not influence training.

After checkpoint reload, the vertical slice evaluates all 225 validation shots
at blackout lengths 0, 5, 10, 15 and 20. This is a development evaluation,
not use of the final test split and not yet a principal multi-seed result.

The 20,000-episode v1 collection completed before the first training attempt.
That attempt then completed 100 epochs but stopped at export because the
project policy identifier had been placed in the checkpoint metadata field
reserved for the runtime architecture identifier. The v1 metrics and failure
log are preserved. The versioned
`structured_n64_k2_full_seed_14303_v2.yaml` correction changes only that
metadata binding, reruns the same training procedure and reuses the immutable
deterministic-mean dataset.

The corrected run completed at epoch 100. Validation action MSE was `0.10859`
and internal-test action MSE was `0.10912`. NumPy/PyTorch maximum error was
`3.8743e-6`, checkpoint reload was exact, and the checkpoint SHA-256 is
`8fbe2171fc4d7272dda0cbdd82adbf8c4bac506aaac21aada62fe9e37a99271f`.
The v1 and v2 metrics files are byte-identical, confirming that the correction
did not change training.

The paired closed-loop validation was weak:

| Blackout steps | Save rate |
| ---: | ---: |
| 0 | 46.2% |
| 5 | 45.8% |
| 10 | 53.3% |
| 15 | 40.4% |
| 20 | 40.9% |
| Overall | 45.3% |

This completes the requested full-data vertical slice, but it does not qualify
the student as a credible controller. The no-blackout result is already too
low, so the next correction should target offline imitation and covariate
shift before running more architecture seeds. Do not interpret the lower
hidden-frame MSE as good blackout behaviour: the loss scale differs across
teacher states and the closed-loop result is the relevant check.

The compact result is
`results/structured_n64_k2_full_seed_14303_v2.json`. Raw evidence remains on
Marvin under:

```text
teacher-datasets/teacher-v3-structured-n64-k2-full-deterministic-2026-08-12-v1
students/structured-n64-k2-full-seed-14303-2026-08-12-v1
students/structured-n64-k2-full-seed-14303-2026-08-13-v2
phase3-structured-n64-k2-full-seed-14303-2026-08-12-v1
phase3-structured-n64-k2-full-seed-14303-2026-08-13-v2
```

The corrected run manifest binds and verifies 50 files totalling 57,828,335
bytes; its SHA-256 is
`5dbbaba2536f5b7307b9103b6bcadf1d4c71badb2fd225447fab0ab0ff92f7a9`.

The follow-up
[`failure diagnostic`](structured_student_failure_diagnostic.md) identifies
the masked initial 16 steps as the primary failure. The student receives no
action loss on those steps even though the commands initialise its recurrent
trajectory and steps 5--15 control the robot. Supplying teacher control only
through step 15 raises no-blackout saves from 104/225 to 224/225 before
handing control back to the student. Student rollouts also have a 16.8-fold
larger 95th-percentile nearest-training-history distance. Exact repeats in the
deterministic dataset have no conflicting targets, so target multimodality is
not established as the primary cause.

## First deterministic shadow-teacher round

Loss on every valid step improved no-blackout control to 151/225 saves but
missed its predeclared 75% gate. The next declared correction collected 20,000
student-controlled episodes and deterministic teacher-mean targets. These were
aggregated one-to-one with the frozen teacher-controlled dataset and used to
retrain the same architecture and seed from scratch.

The corrected checkpoint saves 222/225 no-blackout validation shots (98.7%),
so the gate returns `GO`. Its subsequently opened five-length paired
validation saves 1,107/1,125 episodes (98.4%); at 20 blackout steps it saves
220/225 (97.8%). The compact result and complete hash bindings are in
`results/structured_n64_k2_shadow_round1_v1.json`. The raw datasets,
checkpoint, logs and 1,125 episode rows remain outside Git under the frozen
Marvin experiment directories named there.
