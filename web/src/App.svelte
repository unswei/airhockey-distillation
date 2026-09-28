<script lang="ts">
  import ObservationHistory from './lib/figures/ObservationHistory.svelte';
  import MemoryEvidence from './lib/figures/MemoryEvidence.svelte';
  import MemoryArchitecture from './lib/figures/MemoryArchitecture.svelte';
  import PerformanceCurves from './lib/figures/PerformanceCurves.svelte';
  import PairedContrasts from './lib/figures/PairedContrasts.svelte';
  import CostFrontier from './lib/figures/CostFrontier.svelte';
  import ComparisonTable from './lib/figures/ComparisonTable.svelte';
  import Math from './lib/Math.svelte';
  import { evidence } from './lib/data';
  import { percentage } from './lib/science';
  const base = import.meta.env.BASE_URL;
  const repo = 'https://github.com/unswei/airhockey-distillation';
  const refs = [
    {
      id: 'hafner',
      authors: 'Hafner et al. (2023)',
      title: 'Mastering Diverse Domains through World Models.',
      url: 'https://arxiv.org/abs/2301.04104',
    },
    {
      id: 'orsula',
      authors: 'Orsula (2024)',
      title: 'Learning to Play Air Hockey with Model-Based Deep Reinforcement Learning.',
      url: 'https://arxiv.org/abs/2406.00518',
    },
    {
      id: 'liu',
      authors: 'Liu et al. (2024)',
      title:
        'A Retrospective on the Robot Air Hockey Challenge: Benchmarking Robust, Reliable, and Safe Learning Techniques for Real-world Robotics.',
      url: 'https://proceedings.neurips.cc/paper_files/paper/2024/hash/12ba5de27afcff1a5c796de4a6392154-Abstract-Datasets_and_Benchmarks_Track.html',
    },
    {
      id: 'rusu',
      authors: 'Rusu et al. (2015)',
      title: 'Policy Distillation.',
      url: 'https://arxiv.org/abs/1511.06295',
    },
    {
      id: 'ross',
      authors: 'Ross, Gordon & Bagnell (2011)',
      title:
        'A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.',
      url: 'https://proceedings.mlr.press/v15/ross11a.html',
    },
    {
      id: 'mastrogiuseppe',
      authors: 'Mastrogiuseppe & Ostojic (2018)',
      title:
        'Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks.',
      url: 'https://doi.org/10.1016/j.neuron.2018.07.003',
    },
    {
      id: 'orvieto',
      authors: 'Orvieto et al. (2023)',
      title: 'Resurrecting Recurrent Neural Networks for Long Sequences.',
      url: 'https://proceedings.mlr.press/v202/orvieto23a.html',
    },
    {
      id: 'stolzenburg',
      authors: 'Stolzenburg et al. (2025)',
      title:
        'Efficient time-series approximation with linear recurrent neural networks: architecture learning and predictive power.',
      url: 'https://doi.org/10.1007/s00521-025-11655-y',
    },
  ];
</script>

<a class="skip-link" href="#question">Skip to the article</a>
<div class="topbar">
  <a class="lab-mark" href="https://unswei.github.io/">UNSW · EVOLVING INTELLIGENCE</a>
  <nav aria-label="Article sections">
    <a href="#question">Question</a><a href="#mechanism">Mechanism</a><a href="#results">Evidence</a
    >
  </nav>
</div>
<main>
  <header class="article-header">
    <p class="eyebrow">Control · Memory · Policy distillation</p>
    <h1>Linear Recurrent Memory Suffices to Distil a World-Model Policy for Robot Air Hockey</h1>
    <p class="standfirst">
      We test whether nonlinear recurrence improves a distilled controller’s ability to defend
      during a loss of puck tracking.
    </p>
    <p class="byline">
      <strong>Olivia Fan</strong> &amp;
      <a href="https://orcid.org/0000-0002-8284-2062"><strong>Oliver Obst</strong></a><br />UNSW
      Sydney
    </p>
    <div class="article-links">
      <span class="paper-pending">Paper (soon)</span><a href={repo}>Code ↗</a><a
        href={base + 'data/evidence.json'}>Frozen results ↗</a
      ><a href="#references">References ↓</a>
    </div>
    <p class="scope-note">
      An interactive companion to the manuscript · Simulation study · Manuscript snapshot: 27
      September 2026
    </p>
  </header>
  <figure class="setup-hero">
    <div class="setup-image">
      <img
        src={base + 'kuka-setup.jpg'}
        width="1280"
        height="720"
        fetchpriority="high"
        alt="Two orange KUKA iiwa arms holding mallets over an air-hockey table in the MuJoCo simulator."
      />
    </div>
    <figcaption>
      <strong>The KUKA iiwa air-hockey environment.</strong> Two-robot self-play from Orsula’s
      air-hockey study <a href="#ref-orsula">[2]</a>. Our experiment uses one defender and a
      directly launched puck.
    </figcaption>
  </figure>
  <section id="question" class="article-section first-section">
    <div class="prose">
      <span class="section-number">01 / Task</span>
      <h2>Defending during a loss of puck tracking</h2>
      <p>
        The defender gets a short view of an incoming puck before its position signal is hidden. It
        must then move the mallet to intercept the shot. Two shots can have the same current
        observation but need different defensive actions, depending on how the puck approached.
      </p>
      <p>
        We use a simulated KUKA iiwa defender and launch the puck directly. This lets us replay the
        same shot across policies and blackout conditions without variation from an opponent’s
        strike. Each episode begins with five visible control steps (100 ms), during which the
        mallet is held fixed, followed by one deterministic tracking blackout.
      </p>
      <p>
        The policy receives robot joint positions and velocities, mallet position, puck position and
        a visibility flag. <strong>It does not receive puck velocity.</strong> During blackout, the puck
        position and visibility flag are set to zero.
      </p>
    </div>
    <ObservationHistory />
    <div class="prose">
      <p>
        The illustrated shots can be distinguished from their earlier positions. A memoryless policy
        has no access to those positions. A ten-step observation stack retains them for a while, but
        loses the last visible sample after ten masked observations.
      </p>
      <p>
        The experiment uses pairs of shots with different incoming trajectories but identical first
        masked public observations. Holding the mallet fixed during the visible prefix prevents the
        policy from recording the trajectory through its own motion. Robot observations can diverge
        once the controllers begin to act.
      </p>
      <details>
        <summary>The task and what counts as a save</summary>
        <p>
          Control runs at 50 Hz in deterministic MuJoCo simulation, with a 2.5 s timeout. The policy
          requests a normalised two-dimensional mallet target, which the unchanged low-level
          controller executes. Returns, arrests and safe deflections count as saves. Concessions,
          misses, unresolved timeouts and simulator or safety faults count as failures. The
          experiment excludes opponent-policy variation, additional sensor noise and full-match
          play.
        </p>
        <p>
          In a separate calibration on 216 shots, inactive and fixed-centre defenders concede 208
          and 209 shots, respectively. A privileged controller with access to the true puck state
          saves 214.
        </p>
      </details>
    </div>
  </section>
  <section class="article-section" id="memory">
    <div class="prose">
      <span class="section-number">02 / Memory requirement</span>
      <h2>Does the teacher use its recurrent state?</h2>
      <p>
        The teacher is a world-model policy trained with DreamerV3 <a href="#ref-hafner">[1]</a>,
        building on model-based air-hockey work <a href="#ref-orsula">[2]</a>. We first check
        whether it relies on information retained before the blackout.
      </p>
      <p>
        We compare the teacher with a memoryless PPO policy trained separately from reward. We then
        evaluate the teacher twice on identical shots, once with its normal recurrent state and once
        with that state reset at blackout onset. The reset experiment tests whether carried state
        contributes to its save rate.
      </p>
    </div>
    <MemoryEvidence />
    <div class="prose">
      <p class="outcome-line">
        Resetting the teacher’s state reduces its save rate. How much recurrent structure does a
        student need to reproduce its behaviour?
      </p>
    </div>
  </section>
  <section class="article-section" id="mechanism">
    <div class="prose">
      <span class="section-number">03 / Student architecture</span>
      <h2>How simple can the memory update be?</h2>
      <p>
        Policy distillation trains a smaller controller to imitate a teacher’s actions <a
          href="#ref-rusu">[4]</a
        >. To study the recurrent update, we keep the state size at 64 values and use the same
        observation encoder and action head across recurrent students.
      </p>
      <p>
        The simplest structured student multiplies each previous state value by a learned
        coefficient, then adds contributions from the encoded observation and previous action.
        Increasing the rank adds a nonlinear correction to this diagonal linear update. We also
        compare it with a conventional GRU-64.
      </p>
    </div>
    <MemoryArchitecture />
    <div class="prose">
      <p>
        <strong>Only the recurrent update is constrained.</strong> The observation encoder and
        action head remain nonlinear at every rank, including <Math tex="k=0" />.
      </p>
      <p class="margin-note">
        The architecture draws on low-rank recurrent dynamics <a href="#ref-mastrogiuseppe">[6]</a>
        and effective linear recurrence <a href="#ref-orvieto">[7]</a>,
        <a href="#ref-stolzenburg">[8]</a>. The present experiment concerns closed-loop control
        under tracking loss, not a general claim about sequence modelling.
      </p>
    </div>
  </section>
  <section class="article-section" id="training">
    <div class="prose">
      <span class="section-number">04 / Training data</span>
      <h2>Labelling the student’s own trajectories</h2>
      <p>
        A student’s actions can take it to states that are poorly represented in teacher-controlled
        data. In the preliminary <Math tex="k=2" /> pilot, training on deterministic teacher targets at
        every valid step gave a no-blackout validation save rate of {percentage(
          evidence.pilot.before,
        )}%.
      </p>
      <p>
        With one round of shadow-teacher labelling, the same seed and architecture reached {percentage(
          evidence.pilot.after,
        )}%. In this collection procedure, the student controls the episode. The teacher follows the
        resulting public observations and previous student commands, maintains its own recurrent
        state, and labels each step with the action it would take. This is a DAgger-style correction
        <a href="#ref-ross">[5]</a>.
      </p>
    </div>
    <figure class="science-figure training-flow">
      <div class="figure-heading">
        <div>
          <span class="figure-number">Protocol / Shared budgets</span>
          <h3>Training and shadow-data collection</h3>
        </div>
        <span class="evidence-label">Method schematic</span>
      </div>
      <ol class="training-stages">
        <li>
          <span class="stage-index">1</span><strong>Common teacher data</strong><b
            >20,000 episodes</b
          >
          <p>713,257 transitions.<br />The same base data for all seven families.</p>
        </li>
        <li>
          <span class="stage-index">2</span><strong>Family-specific shadows</strong><b
            >5 collectors × 4,000 episodes</b
          >
          <p>
            Paired shot and blackout schedules.<br />The frozen teacher labels every valid step.
          </p>
        </li>
        <li>
          <span class="stage-index">3</span><strong>Final students from scratch</strong><b
            >40,000 episodes per family</b
          >
          <p>Five matched seeds.<br />Equal optimisation and episode-loss weighting.</p>
        </li>
      </ol>
      <figcaption>
        All families use the same episode schedules and collection budgets. Their trajectories,
        episode lengths and teacher-query counts differ because each family controls its own
        episodes. Students are supervised only on teacher actions; teacher states, world-model
        latents and privileged puck state are excluded. The k=2 pilot motivated this procedure but
        does not compare ranks.
      </figcaption>
    </figure>
    <div class="prose">
      <details>
        <summary>Training and checkpoint selection</summary>
        <p>
          The common base data are combined with each family’s own 20,000 shadow episodes. Five
          final models per family are trained from scratch. Recurrent training uses 64-step chunks
          with state carried across chunks; loss is applied at every valid step with equal total
          weight per episode.
        </p>
        <p>
          We select checkpoints using offline validation action error. Before opening the principal
          test split, we freeze all 35 final checkpoints, paired validation results and efficiency
          measurements. The test split is evaluated once, with no subsequent retraining or
          checkpoint reselection.
        </p>
      </details>
    </div>
  </section>
  <section class="article-section" id="results">
    <div class="prose">
      <span class="section-number">05 / Test results</span>
      <h2>Does nonlinear recurrence improve save rate?</h2>
      <p>
        We compare seven student families on the same 225 fresh test shots at six blackout
        durations. Each family has five matched training seeds. The 500 ms condition is a
        predeclared extrapolation beyond the 0–400 ms training range.
      </p>
      <p>
        At 400 ms, the distilled feed-forward student saves 44.4% of shots and the ten-step stack
        saves 86.0%. The structured <Math tex="k=0" /> student saves 98.3%, close to the frozen teacher’s
        observed 98.2%.
      </p>
    </div>
    <PerformanceCurves />
    <div class="prose">
      <p>
        The recurrent students have much closer save rates. At 400 ms, GRU-64 saves 97.8%, while the
        structured variants range from 97.9% to 98.8%. The paired comparisons below show the
        uncertainty in these differences.
      </p>
    </div>
    <PairedContrasts />
    <div class="prose">
      <p>
        We find no clear save-rate gain from the low-rank nonlinear correction or the GRU under this
        protocol. The intervals still allow small differences in either direction, so they do not
        establish equivalence.
      </p>
      <details>
        <summary>How to read the uncertainty</summary>
        <p>
          Family intervals come from 10,000 paired hierarchical cluster-bootstrap resamples over
          training seeds and evaluation units. Observation-alias pairs remain together as clusters;
          common resamples preserve the paired comparisons. The plots use the frozen estimates and
          include every seed.
        </p>
        <p>
          The teacher result comes from one frozen policy. The compact result summaries do not
          provide a teacher interval, so none is plotted. The earlier memory and state-reset studies
          use different evaluation protocols; their teacher rates are reported separately.
        </p>
      </details>
    </div>
  </section>
  <section class="article-section" id="cost">
    <div class="prose">
      <span class="section-number">06 / Computational cost</span>
      <h2>What does each recurrent update cost?</h2>
      <p>
        Both <Math tex="k=0" /> and GRU-64 carry 64 recurrent values. The structured student uses 58.5%
        fewer total parameters and multiply–adds per step, and 88.0% fewer recurrent-core parameters.
        Both require the same recurrent-state storage.
      </p>
      <p>
        The structured student also has lower measured latency. This comparison includes an
        implementation difference: structured families use an optimised compiled kernel, whereas the
        GRU and non-recurrent baselines use BLAS-backed NumPy. We use arithmetic counts as the
        primary computational comparison and report latency for these specific implementations.
      </p>
    </div>
    <CostFrontier />
    <ComparisonTable />
  </section>
  <section class="article-section" id="limits">
    <div class="prose">
      <span class="section-number">07 / Interpretation and limits</span>
      <h2>Where might the result change?</h2>
      <p>
        On this defence task, the student with a linear recurrent update retains high save rates
        during blackout. The more complex recurrent updates we tested provide no clear behavioural
        improvement.
      </p>
      <p>
        The comparison uses one simulated set-piece, one teacher and one recurrent state size. Save
        rates are near the ceiling, which limits our ability to distinguish the recurrent families.
        We have not tested full-match play, deployment on a physical robot or other types of sensor
        failure.
      </p>
      <p>
        One possible explanation is that the puck’s hidden motion needs only a compact predictive
        summary. We have not tested this explanation by decoding the learned state, and cannot say
        which physical quantities it represents.
      </p>
      <p class="outcome-line">
        For the tested teacher and task, linear recurrence was sufficient to retain the observed
        blackout performance.
      </p>
      <p>
        Harder hidden dynamics, richer contacts, longer or noisier tracking failures, other teachers
        and real robots could change the comparison. Testing these conditions would help establish
        when more complex recurrence is needed.
      </p>
    </div>
  </section>
  <section class="article-section" id="methods">
    <div class="prose">
      <span class="section-number">Sources / Reproducibility</span>
      <h2>Paper, code and data</h2>
      <p>
        The paper and full methods will be available here soon. On this page, the shot paths are
        illustrative, the architecture diagram follows the implemented update, and the result plots
        use frozen measurements. The browser runs neither the trained policies nor the physics
        simulator.
      </p>
      <div class="resource-links">
        <div class="resource-pending">Manuscript &amp; full methods <span>Soon</span></div>
        <a href={base + 'data/evidence.json'}
          >Frozen numbers &amp; source hashes <span>JSON ↗</span></a
        ><a href={base + 'SCIENTIFIC_FIDELITY.md'}
          >Scientific fidelity &amp; simplifications <span>MD ↗</span></a
        ><a href={repo}>Code &amp; experiment records <span>GitHub ↗</span></a><a
          href={base + 'references.bib'}>Complete bibliography <span>BibTeX ↗</span></a
        >
      </div>
    </div>
  </section>
  <section class="article-section" id="references">
    <div class="prose">
      <span class="section-number">Further reading</span>
      <h2>References</h2>
      <ol class="references">
        {#each refs as ref}<li id={'ref-' + ref.id}>
            <span>{ref.authors}.</span> <a href={ref.url}>{ref.title}</a>
          </li>{/each}
      </ol>
      <p class="margin-note">
        For the wider robotics context, see the Robot Air Hockey Challenge retrospective
        <a href="#ref-liu">[3]</a>. Reference details follow the manuscript.
      </p>
    </div>
  </section>
</main>
<footer class="footer">
  <span>Olivia Fan &amp; Oliver Obst · UNSW Sydney</span><br />Text, figures and tables are also
  available with JavaScript disabled.<br /><a href={base + 'data/sources.json'}
    >Manuscript and asset provenance</a
  >
  · <a href="#question">Back to the question ↑</a>
</footer>
