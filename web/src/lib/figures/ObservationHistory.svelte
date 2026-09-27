<script lang="ts">
  import {
    visibleAt,
    schematicPosition,
    puckInput,
    stackAt,
    STEP_MS,
    MASK_START,
    MASK_LENGTH,
  } from '../science';
  let step = $state(5);
  const shots = [
    { name: 'A', side: 1, colour: '#9a6500' },
    { name: 'B', side: -1, colour: '#008468' },
  ];
  const phase = $derived(
    step < 5 ? 'Visible prefix' : step < 25 ? 'Tracking unavailable' : 'Tracking returns',
  );
  const observed = $derived(stackAt(step, 1).filter((slot) => slot.value[2] === 1).length);
  const px = (s: number) => 20 + schematicPosition(s, 1)[0] * 390;
  const py = (s: number, side: number) => 110 + schematicPosition(s, side)[1] * 175;
  const triple = (side: number) =>
    puckInput(step, side)
      .map((v, i) => (i === 2 ? String(v) : v.toFixed(2)))
      .join(', ');
</script>

<figure
  class="science-figure observation"
  id="observation-figure"
  aria-labelledby="observation-title"
>
  <div class="figure-heading">
    <div>
      <span class="figure-number">01 / Information</span>
      <h3 id="observation-title">What remains in a ten-step history?</h3>
    </div>
    <span class="evidence-label">Schematic + exact input rules</span>
  </div>
  <div class="shot-grid">
    {#each shots as shot}
      <div class="shot" style:--shot-colour={shot.colour}>
        <div class="shot-heading">
          <strong>Shot {shot.name}</strong><span>Illustrative world state</span>
        </div>
        <svg
          viewBox="0 0 420 228"
          role="img"
          aria-label={`Schematic shot ${shot.name} at ${step * STEP_MS} milliseconds; ${phase.toLowerCase()}`}
        >
          <defs
            ><pattern id={`hatch-${shot.name}`} width="6" height="6" patternUnits="userSpaceOnUse"
              ><path d="M0,6 L6,0" stroke="#bbc3bc" stroke-width="1" /></pattern
            ></defs
          >
          <rect x="18" y="16" width="384" height="186" rx="9" class="table-surface" />
          <path d="M210 17 V201" class="table-line" /><circle
            cx="210"
            cy="109"
            r="27"
            class="table-line"
          />
          <path d="M18 58 V162" stroke="#27332f" stroke-width="4" />
          <text x="31" y="220" class="svg-small">defending goal</text>
          <path
            d={`M${px(0)} ${py(0, shot.side)} L${px(5)} ${py(5, shot.side)}`}
            stroke={shot.colour}
            stroke-width="2.5"
            fill="none"
          />
          <path
            d={`M${px(5)} ${py(5, shot.side)} L${px(25)} ${py(25, shot.side)}`}
            stroke={shot.colour}
            stroke-width="2"
            stroke-dasharray="5 5"
            fill="none"
            opacity="0.7"
          />
          <path
            d={`M${px(25)} ${py(25, shot.side)} L${px(28)} ${py(28, shot.side)}`}
            stroke={shot.colour}
            stroke-width="2.5"
            fill="none"
          />
          {#each [0, 1, 2, 3, 4] as point}
            <circle
              cx={px(point)}
              cy={py(point, shot.side)}
              r="3"
              fill={shot.colour}
              opacity={point <= step ? 1 : 0.15}
            />
          {/each}
          <path d={`M${px(5)} 24 V195`} stroke="#bbc3bc" stroke-dasharray="3 5" />
          <text x={px(5) - 8} y="42" text-anchor="end" class="svg-small">blackout onset</text>
          <circle
            cx={px(step)}
            cy={py(step, shot.side)}
            r="8"
            fill={visibleAt(step) ? shot.colour : `url(#hatch-${shot.name})`}
            stroke={shot.colour}
            stroke-width="2.5"
          />
          {#if !visibleAt(step)}<text
              x={px(step)}
              y={py(step, shot.side) + (shot.side === 1 ? -19 : 27)}
              text-anchor="middle"
              class="svg-small">hidden from policy</text
            >{/if}
        </svg>
        <div class="puck-input">
          <span>Current puck input</span><code data-testid={`input-${shot.name}`}
            >({triple(shot.side)})</code
          >
        </div>
        <div class="history-heading">
          <span>Ten-step puck history</span><span>oldest → newest</span>
        </div>
        <div
          class="history-strip"
          aria-label={`Shot ${shot.name}: ${observed} visible samples retained`}
        >
          {#each stackAt(step, shot.side) as slot}
            <div
              class:sample-visible={slot.value[2] === 1}
              class:sample-padding={slot.padding}
              class="history-slot"
              title={slot.padding
                ? 'Episode-start zero padding'
                : `Step ${slot.step}: (${slot.value.join(', ')})`}
            >
              {#if slot.value[2] === 1}
                <svg viewBox="0 0 28 34" aria-hidden="true"
                  ><path
                    d={`M4 ${17 - shot.side * 6} L24 ${17 + shot.side * 6}`}
                    stroke={shot.colour}
                    opacity="0.25"
                  /><circle
                    cx={7 + (slot.step % 5) * 3.5}
                    cy={17 + (slot.step - 2) * shot.side * 2.5}
                    r="2.7"
                    fill={shot.colour}
                  /></svg
                >
              {:else}<span aria-hidden="true">{slot.padding ? '·' : '×'}</span>{/if}
            </div>
          {/each}
        </div>
      </div>
    {/each}
  </div>
  <div class="timeline-control interactive-control">
    <div class="timeline-heading">
      <label for="episode-time">Scrub the episode <span>20 ms per step</span></label><output
        for="episode-time">{step * STEP_MS}<span> ms</span></output
      >
    </div>
    <div class="timeline-track">
      <div class="visible-segment" style="width:17.857%"></div>
      <div class="masked-segment" style="width:71.429%"></div>
      <div class="visible-segment" style="width:10.714%"></div>
    </div>
    <input
      id="episode-time"
      type="range"
      min="0"
      max="28"
      step="1"
      bind:value={step}
      aria-valuetext={`${step * STEP_MS} milliseconds, ${phase}, ${observed} visible samples retained`}
    />
    <div class="timeline-labels">
      <span>0</span><span style="left:17.857%">100 · blackout</span><span style="left:89.286%"
        >500 · visible</span
      >
    </div>
    <div class="timeline-shortcuts">
      <button onclick={() => (step = 4)}>Last visible</button><button onclick={() => (step = 5)}
        >Blackout onset</button
      ><button onclick={() => (step = 14)}>History emptied</button><button
        onclick={() => (step = 25)}>Tracking returns</button
      >
    </div>
  </div>
  <p class="figure-insight" aria-live="polite">
    <strong>{phase}.</strong>
    {#if step < 5}The mallet is held fixed while the policy observes the incoming shot. The two
      shots have different position histories.{:else if step < 14}Both current puck inputs are zero.
      The history still contains {observed} visible samples.{:else if step < 25}Ten successive
      masked samples have replaced every visible puck position in the stack.
    {:else}Visible puck positions enter the history again. Recurrent policies keep their state when
      tracking returns.{/if}
  </p>
  <div class="print-snapshots">
    <p>
      <strong>Reading the sequence:</strong> 80 ms: five visible samples. 100 ms: equal masked puck inputs,
      different histories. 280 ms: all ten retained triples are masked. 500 ms: visibility returns.
    </p>
  </div>
  <figcaption>
    <strong>Illustrative shots with the implemented masking and history rules.</strong> Masking
    starts at step {MASK_START} and lasts {MASK_LENGTH * STEP_MS} ms. Each new sample shifts into the
    ten-slot window, which starts with zero padding. Solid paths show visible intervals and dashed paths
    show blackout; masking depends on time, not a region of the table. The hidden puck is drawn for reference
    but is absent from the policy input. Robot proprioception is omitted. These are schematic paths, without
    simulated actions or save outcomes.
  </figcaption>
</figure>
