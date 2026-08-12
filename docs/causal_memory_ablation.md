# Causal recurrent-state ablation

Status: **completed**. Decision: **GO**.

This check intervenes on the frozen 700,000-step DreamerV3 teacher rather than
comparing it with another policy. The teacher is evaluated on the same 225
test shots at blackout lengths 0, 5, 10, 15 and 20. One arm carries its
recurrent state normally. The other resets that state exactly when blackout
begins. Both arms use deterministic Dreamer inference: the categorical RSSM
posterior uses its mode and the bounded-normal actor uses its mean.

The code that makes inference deterministic was committed before evaluation.
The canonical run used commit
`700e8f80bac2debb5cb6a9804d8e7f199f0d5c9d` and the pinned container digest
`sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f`.

## Paired result

| Blackout steps | Normal state | Reset at blackout | Paired drop | Paired bootstrap 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 99.6% | 99.6% | 0.0 points | [0.0, 0.0] |
| 5 | 100.0% | 58.2% | 41.8 points | [35.6, 48.4] |
| 10 | 98.2% | 49.3% | 48.9 points | [41.8, 55.6] |
| 15 | 96.9% | 49.8% | 47.1 points | [40.0, 54.2] |
| 20 | 96.0% | 50.2% | 45.8 points | [38.7, 52.9] |

All three predeclared checks passed. At 20 steps, the reset caused a
45.8-point drop, exceeding the required 10 points, and the confidence-interval
lower bound was positive. With no blackout, all 225 episode records were
identical between arms. The result therefore supports the causal claim that
state carried into blackout contributes to the teacher's performance.

## Deterministic replay

The normal-state arm was evaluated twice in independent processes. All 1,125
episode records were identical. Serialising the `episodes` array as sorted,
compact JSON gives SHA-256
`33c19fec5410b4482553bd4420598afddfc095fabe39c3d8adb57c5a88f3451b`
for both runs. The complete result files differ because they contain creation
times and measured inference latencies.

## Canonical result and frozen raw evidence

The compact machine-readable result is
`results/causal_memory_ablation.json`. It is a byte-for-byte copy of the gate
report, contains no episode rows, and has SHA-256
`919a1306453542950b2002a0b797a6279473fe588abc6a4fb1c1021b8e54a67a`.

The three 1,125-episode evaluations, console logs, gate report and checksum
manifest remain outside Git at:

`/home/oliver/experiments/airhockey-memory-distillation/stage-b-causal-memory-ablation-deterministic-2026-08-12-v1`

The directory and its contents are root-owned and read-only. Corrections must
create a new run rather than alter this evidence.

## Verified hashes

| Artefact | SHA-256 |
| --- | --- |
| Frozen teacher actor | `6a672d5b6d7c2b9ca2335f1a85b69280ca7db58deb5e2f56d2ba033ebf636254` |
| Frozen teacher manifest | `b269950b0f10b32c7b411ece890b349d5c1280cc95f6d4864818db4c3ac72c15` |
| Causal gate configuration | `f1052a03a164b6a52fcd2be2f6a31e3b46952cb773b40bb40d7a55abc861ea32` |
| Confirmation evaluation configuration | `8cdfaa1ead0d6525fc899de1b5776cdfca0dce94a3bfbdfa731e52779a2ebc38` |
| Teacher evaluator | `5b1bc9390121d617b39b6a89b0f2d121f19cf550cb3f008358e918cbd2f06b78` |
| Deterministic inference adapter | `0060bb2e1e38a0c2d34930d272e0ba6572360c3a6f629dde62ee2ffa006375ae` |
| Normal-state evaluation | `6cc22fb8035402fa0621d115b3bb9aa0b88084befef2b5e981551373124d29f2` |
| Independent normal-state repeat | `1ac5589dad7967c890652e4217d099ec15571d17cd8eefee7829ce53d42ac710` |
| Reset-at-blackout evaluation | `27017801dd96ef3c8658dec2b2b85aec9b15934b544b14e63c2eb940e2941300` |
| Causal gate and canonical result | `919a1306453542950b2002a0b797a6279473fe588abc6a4fb1c1021b8e54a67a` |
| Frozen checksum manifest | `c9c477f1e0b4141c6f5b65fb2e5e3a46ad81da9b3ff8841ce42e0ae13fe079fe` |

The checksum manifest covers every decision input and output listed above and
passes `sha256sum -c` after the run directory was frozen. Console logs are
retained for diagnosis but are not gate inputs.
