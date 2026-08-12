# Stage B canonical evidence backup

Status: **complete and verified**.

The frozen 700,000-step teacher and canonical Stage B evidence have an
independent copy in the Mac's iCloud Drive. The iCloud provider reported the
container caught up after upload. The local copy is read-only.

The user-facing iCloud Drive path is:

`Research Backups/airhockey-distillation/stage-b-canonical-2026-08-12-v1`

Its resolved local path is:

`/Users/oliver/Library/Mobile Documents/com~apple~CloudDocs/Research Backups/airhockey-distillation/stage-b-canonical-2026-08-12-v1`

The backup contains 57 files occupying 19,245,888 bytes. Its whole-backup
`SHA256SUMS` manifest has SHA-256
`78a2901c3f810f38d1a7044380bfbd9477442501e3379c281092d90218560aa7`.

## Scope

The backup includes:

- the complete frozen DreamerV3 teacher at step 700,000;
- the complete Stage B v3 memory-validation run;
- the complete deterministic causal recurrent-state ablation run;
- the teacher validation-selection report;
- the final model, binding, resolved configuration, result and training log
  for PPO seeds 14303, 14304 and 14305;
- a complete Git bundle of `main` through commit
  `bb362151d405150e7e9e0d84fbfbbc766f575bdf`.

The Git bundle contains the deterministic evaluation implementation commit
`700e8f80bac2debb5cb6a9804d8e7f199f0d5c9d` and has SHA-256
`4773ed5b95101049972f6b6c83b45627338988f9691cc9c2ac9b162f9a26f335`.
`git bundle verify` reports complete history through `main`.

The Marvin-generated `SOURCE_SHA256SUMS` manifest has SHA-256
`57bb0c1faaea31096f774711321656282bf8eef4fa00378e93099e791f6800dd`.
Every copied experiment file passed verification against it after transfer.
The independent whole-backup manifest also passed after the copy was made
read-only.

## Verification and recovery

From the backup directory, verify every stored file with:

```bash
shasum -a 256 -c SHA256SUMS
```

Verify only the experiment copy against the manifest generated on Marvin with:

```bash
cd experiments
shasum -a 256 -c ../SOURCE_SHA256SUMS
```

Restore the code into a new checkout with:

```bash
git clone code/airhockey-distillation-bb362151.bundle restored-code
```

Restore `experiments/` into an empty experiment root. Do not overwrite live
run directories until their hashes have been compared.

This is a compact backup of the canonical evidence, not the full development
history. It deliberately excludes the one-million-step Dreamer training
directory, the 20,000-episode demonstration dataset and intermediate PPO
checkpoints. Future Phase 3 checkpoints and datasets require their own
versioned backup before Marvin becomes their sole copy.
