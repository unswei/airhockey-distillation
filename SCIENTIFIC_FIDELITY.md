# Scientific fidelity of the interactive article

The authoritative source is the complete manuscript and bibliography at Paper
commit 166a4a2154f0896c68497f0070fe9813e0ddf4a4. The article preserves its
question, title, authors, experimental protocol and limited empirical conclusion.
The manuscript source is not edited by the web build.

## Evidence classes

| Visual | Source and computation | Boundary |
| --- | --- | --- |
| Opening KUKA image | Existing poster from the upstream pretrained 2023 self-play reproduction on Marvin; SHA-256 8514f02ea385aa7f4ba7f49f87751dc18733b974031a7ea42bb975fea384827e | Context image of the simulated environment, not evidence from the tracking-loss experiment. Both robots are retained; CSS crops excess floor and the scoreboard. The caption distinguishes upstream self-play from the single-defender experiment. |
| 01: observation/history scrubber | Illustrative straight paths; faithful reconstruction of masking at steps [5,25), 20 ms steps, ten triples including the current sample, episode-start zero padding | Not MuJoCo trajectories, trained actions, contacts or predicted saves. The geometry has arbitrary units. The policy receives no true hidden position or puck velocity. |
| 02: memory-validity and reset comparisons | Frozen Stage B memory-validation and causal-ablation JSON | Separate validation studies. The PPO baseline is not the distilled feed-forward family. Neither teacher estimate is substituted for the principal-test teacher. |
| 03: rank-selectable architecture | Manuscript equations and implementation shapes; analytic counts checked against frozen efficiency records | Architectural reconstruction, not network inference. Each rank denotes a separately trained family. State squares show dimensionality only. |
| Shadow-labelling diagram | Manuscript training protocol; pilot rates from their own frozen summaries | A method schematic. The k=2 pilot is not a principal comparison or evidence of a rank advantage. |
| 04: duration curves and seed readouts | Frozen principal means, stored intervals and all five seed rates | No resampling, seed filtering or synthetic intermediate results. Lines connect the six measured durations only. The 500 ms condition is extrapolation. |
| 05: 400 ms forest plot | Six predeclared paired contrasts reported in the manuscript | The fixed comparisons do not change with the duration selector. No equivalence claim. |
| 06: performance–cost scatter plots | Matched per-seed principal save rates and CPU measurements; stored architecture counts | Five circles per family, no jitter. Diamonds are marginal family medians, which need not equal family means or describe one actual seed. No fitted frontier or winner ranking. |
| Complete table | Frozen means, intervals and efficiency CSV | Overall rates exclude 500 ms. Median and p95 timings are separately aggregated across five seeds. |

Only the lightweight input-bookkeeping and parameter-count formulae are
recomputed in the browser. No trained model, simulator, bootstrap, statistical
test or new experiment runs there. The real-data panels retain all observations
available in the compact frozen summaries.

## Important qualifications

- Identical public observations are explained at blackout onset under the
  fixed-mallet prefix. Later robot observations can diverge after control begins.
  This deliberately narrows the manuscript's broader “during blackout” prose
  to the implemented aliasing guarantee; it does not change the manuscript.
- The schematic shows only puck triples. The full observation has 19 components:
  seven joint positions, seven joint velocities, mallet xy, puck xy, visibility.
  The finite stack retains ten puck triples plus current proprioception.
  The two recurrent action inputs are previous requested commands, not velocity.
- The last visible sample is step 4. It leaves the ten-slot window at step 14
  (280 ms episode time, not 280 ms blackout duration). Masking is temporal,
  not a region of the table. The visible trajectory continues without a kink.
- The state equation uses the manuscript's indices: current observation x_t
  and previous command a_(t−1) update z_(t−1) to z_t. At k=0 the update is
  affine; the encoder and action head remain nonlinear. The rank bound is a
  partial Jacobian statement with current inputs fixed, not a stability proof.
- “Same budget” means equal episode schedules, collection budgets and
  optimisation. Family-specific shadow trajectories, realised episode lengths
  and teacher-query counts are not identical.
- Family uncertainty uses the frozen 10,000-replicate paired hierarchical
  cluster-bootstrap intervals. Single-policy memory checks use their own paired
  episode intervals. A teacher confidence interval is not invented.
- The recurrent plot uses a clearly labelled 90–100% scale; the memory-baseline
  plot uses 0–100%; cost plots use 30–100%. The full numerical values remain
  accessible. Plot heights across different scales must not be compared.
- The principal teacher's 98.2% at 400 ms, the memory-comparison 96.9% and
  causal-reset study's 96.0% belong to distinct evaluations.
- CPU latency compares a verified compiled structured kernel with BLAS-backed
  NumPy baselines on the measured host. Arithmetic counts are the primary
  architecture comparison; no implementation-independent speed claim is made.
- The conclusion is restricted to one deterministic simulated set-piece, one
  frozen teacher and one state size. Near-ceiling performance limits resolution.
  There is no formal equivalence result, hardware deployment, learned-state
  decoding or evidence of a universal advantage for linear recurrence.
- Simple hidden dynamics are labelled as a possible explanation, not a measured
  latent representation. The upstream self-play video is not used as evidence.
- The draft abstract's unfinished CI is not reproduced: the article uses the
  complete interval in the manuscript's Results section and frozen statistics.

## Provenance and maintenance

web/scripts/export-data.mjs checks SHA-256 hashes of seven existing evidence
files before exporting a compact public/data/evidence.json. Its check mode
requires byte-for-byte equality, and runs before every build. No episode-level
records are copied into Git or into the website.

public/data/sources.json identifies the exact manuscript snapshot and hashes
the published bibliography, setup image and scientific-fidelity note. The manuscript
PDF is not currently published; its sources remain in the separate Paper repository.
No figure or result is cleaned up by altering data.

The article is prerendered so its text, default figures, equations and complete
tables remain readable without JavaScript. Interactions are enhancements.
Unit tests cover masking, stack eviction, export means and architecture counts;
browser tests cover controls, evidence retention, responsive overflow, assets,
console errors, keyboard access and the no-JavaScript fallback.
