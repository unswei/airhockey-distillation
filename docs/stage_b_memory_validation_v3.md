# Stage B v3 memory validation

Status: **completed**. Decision: **GO**.

Stage B v3 compared the frozen 700,000-step DreamerV3 teacher with a
predeclared, strictly memoryless PPO policy on the observation-aliased
`direct_launch_v3` task. The visible-observation baseline first qualified on
the validation split. Only then were the teacher and selected baseline
evaluated once on the untouched test split.

The result supports a memory requirement in this controlled task. It does not
yet test the proposed structured recurrent students.

## Frozen inputs

The teacher is the validation-selected checkpoint from
`teacher-full-v3-2026-08-12-v1`. Its inference payload remains at
`frozen-teachers/dreamerv3-teacher-v3-step-700000/agent.pkl` under Marvin's
experiment root. The selected memoryless policy is PPO seed 14304 at
`students/feed-forward-ppo-stage-b-v3-seed-14304/model.zip`.

Training and evaluation used code commit
`2f35e92ec0009d97b263cc4e53868fa46b15005c`, configuration
`configs/student/feed_forward_stage_b_v3.yaml`, and container digest
`sha256:de846e25961d2408db73a6b5cfe2afc7ecf0e14e0e8ac837f97212a180e51d5f`.
The policy receives 19 public values and emits two actions. It has no previous
action, observation stack, recurrent state, puck velocity or privileged state.

## Validation-only qualification

All three predeclared PPO seeds were trained for one million environment
steps. Selection used only 225 no-blackout validation shots.

| Training seed | Validation save rate | Mean score |
| ---: | ---: | ---: |
| 14303 | 88.0% | 0.969 |
| 14304 | 92.0% | 1.062 |
| 14305 | 86.2% | 0.937 |

Seed 14304 was selected by the fixed save-rate, mean-score and lower-seed
ordering. It exceeded the 75% credibility threshold, trailed the 99.1% teacher
by 7.1 percentage points rather than more than 10 points, and incurred no
simulator faults. The qualification decision was `GO`, which opened the test
split.

## Held-out paired result

The confirmatory evaluation used 225 test shots at each of five blackout
lengths. Teacher and baseline therefore contributed 1,125 exactly paired
policy comparisons.

| Blackout steps | Teacher save rate | Feed-forward save rate | Paired teacher advantage | Paired bootstrap 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 99.6% | 90.7% | 8.9 points | [5.3, 12.9] |
| 5 | 99.1% | 80.4% | 18.7 points | [13.8, 24.0] |
| 10 | 98.2% | 61.8% | 36.4 points | [30.2, 43.1] |
| 15 | 97.8% | 17.8% | 80.0 points | [74.2, 85.3] |
| 20 | 96.9% | 33.3% | 63.6 points | [56.9, 70.2] |

All five predeclared checks passed:

| Check | Observed | Required |
| --- | ---: | ---: |
| Credible visible feed-forward baseline | 90.7% | at least 75% |
| Comparable no-blackout performance | 8.9-point teacher advantage | at most 10 points |
| Material 20-step teacher advantage | 63.6 points | at least 15 points |
| Advantage growth from 0 to 20 steps | 54.7 points | at least 10 points |
| Lower 95% confidence bound at 20 steps | 56.9 points | above zero |

The feed-forward curve is not monotonic: its save rate is lower at 15 steps
than at 20 steps. The predeclared test concerns the endpoint increase in the
paired teacher advantage, not monotonicity at every intermediate blackout
length. The endpoint effect is large and its paired interval excludes zero.

## Canonical result and immutable raw evidence

The compact machine-readable result is
`results/stage_b_memory_validation_v3.json`. It is a byte-for-byte copy of
the 3.5 KB gate report and contains no episode rows. Its SHA-256 is
`9b074299b5794b4a35af30737fcbba32753a0318f9eda9b11ab492e916a68739`.

The full teacher and feed-forward confirmation files retain the 1,125
episode-level records outside Git at:

`/home/oliver/experiments/airhockey-memory-distillation/stage-b-memory-validation-v3-2026-08-12-v1`

The selected checkpoints and raw JSON evidence are root-owned and non-writable
by Marvin's normal experiment account. Corrections must create a new run
rather than alter these files.

The run, frozen teacher, final baseline models and selection report are also
included in the verified
[`Stage B canonical evidence backup`](stage_b_backup.md).

## Verified hashes

Paths below are relative to
`/home/oliver/experiments/airhockey-memory-distillation` unless marked as a
code-repository path.

| Artefact | Path | SHA-256 |
| --- | --- | --- |
| Stage B v3 configuration | code: `configs/student/feed_forward_stage_b_v3.yaml` | `8cdfaa1ead0d6525fc899de1b5776cdfca0dce94a3bfbdfa731e52779a2ebc38` |
| Frozen teacher actor | `frozen-teachers/dreamerv3-teacher-v3-step-700000/agent.pkl` | `6a672d5b6d7c2b9ca2335f1a85b69280ca7db58deb5e2f56d2ba033ebf636254` |
| Frozen teacher manifest | `frozen-teachers/dreamerv3-teacher-v3-step-700000/manifest.json` | `b269950b0f10b32c7b411ece890b349d5c1280cc95f6d4864818db4c3ac72c15` |
| Teacher selection report | `teacher-full-v3-2026-08-12-v1/validation/selection.json` | `9a5bdda07bf50adf8e33ec6198c678753a7119d8295aef4aa636103913543662` |
| Seed 14303 model | `students/feed-forward-ppo-stage-b-v3-seed-14303/model.zip` | `66aed71125ded94907f0ae00caf7adb1aadda96030aa65b6fd813aa78fb87433` |
| Seed 14303 training result | `students/feed-forward-ppo-stage-b-v3-seed-14303/result.json` | `cf902c3375c3aa9efbdffb43e71ea435b55ddab99acb16a995a92594cd542e7b` |
| Seed 14304 model | `students/feed-forward-ppo-stage-b-v3-seed-14304/model.zip` | `1874930d06164e6ef965acb99a5e031161d20ed5940f8836e97e4991529c42b0` |
| Seed 14304 training result | `students/feed-forward-ppo-stage-b-v3-seed-14304/result.json` | `d8cd8bf0677750575e21d456c15770e79ed968a63745f99c564831577fa28fd0` |
| Seed 14305 model | `students/feed-forward-ppo-stage-b-v3-seed-14305/model.zip` | `3287b669d23998ccbbebe4b5060463f2bc5ef8925eeb4918e4c26abf01148caf` |
| Seed 14305 training result | `students/feed-forward-ppo-stage-b-v3-seed-14305/result.json` | `9c010d0c275f7e7bbaca5ab8dd4f60c390ab7d865273e0f7595b4c16e402bcca` |
| Seed 14303 qualification | `stage-b-memory-validation-v3-2026-08-12-v1/qualification-seed-14303.json` | `a2028295b53f915891cc160ea2427c7b3a834a0fa3658f88169d75fd397496a2` |
| Seed 14304 qualification | `stage-b-memory-validation-v3-2026-08-12-v1/qualification-seed-14304.json` | `91e732c82662eeca66c09b0a9079e5eb05527bbf301b120680dc5036522da699` |
| Seed 14305 qualification | `stage-b-memory-validation-v3-2026-08-12-v1/qualification-seed-14305.json` | `1a21a73053553ee51087baaad1715aa76e6f80428ff8df02355abe4e0eb2a439` |
| Qualification decision | `stage-b-memory-validation-v3-2026-08-12-v1/baseline_qualification.json` | `094f041dc3e2e67ebb9594d6a8207d01974236724942ae37fe60796477db50f5` |
| Teacher confirmation | `stage-b-memory-validation-v3-2026-08-12-v1/teacher_confirmation.json` | `2561cde52150b10f77f09f4afb52cb2ce8ffbd1a70c8211479947931da17bd26` |
| Feed-forward confirmation | `stage-b-memory-validation-v3-2026-08-12-v1/feed_forward_confirmation.json` | `7ae0ce0e0a27dd5d004be468b06b9777aef3ce4739e44fce5bd4bc2b285852c3` |
| Memory gate and canonical result | `stage-b-memory-validation-v3-2026-08-12-v1/memory_gate.json` | `9b074299b5794b4a35af30737fcbba32753a0318f9eda9b11ab492e916a68739` |
| Original run hash manifest | `stage-b-memory-validation-v3-2026-08-12-v1/sha256sums.txt` | `8ccd240f7d1ade529946e366208a9eb4c8c2cd1d341360e391961074f0619eae` |

Training logs are retained with the raw run but are not decision inputs. The
separately predeclared
[`causal recurrent-state ablation`](causal_memory_ablation.md) subsequently
returned `GO`: resetting the teacher state at blackout onset caused a
45.8-point save-rate drop at 20 steps while leaving all no-blackout episode
records unchanged.
