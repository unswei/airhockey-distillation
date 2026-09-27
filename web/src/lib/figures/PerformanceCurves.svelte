<script lang="ts">
  import {
    families,
    labels,
    colours,
    dashes,
    principal,
    durations,
    rate,
    interval,
    type Family,
  } from '../data';
  import { percentage } from '../science';
  let selected = $state(20);
  let focused = $state<Family | 'teacher' | null>(null);
  const groups: { name: string; subtitle: string; min: number; members: (Family | 'teacher')[] }[] =
    [
      {
        name: 'Memory baselines',
        subtitle: 'Full scale · 0–100%',
        min: 0,
        members: ['feed_forward', 'finite_stack_10', 'structured_k0', 'teacher'],
      },
      {
        name: 'Recurrent comparison',
        subtitle: 'Expanded scale · 90–100%',
        min: 90,
        members: [
          'structured_k0',
          'structured_k1',
          'structured_k2',
          'structured_k4',
          'gru_n64',
          'teacher',
        ],
      },
    ];
  const x = (s: number) => 46 + (s / 25) * 362;
  const y = (value: number, min: number) => 250 - ((value * 100 - min) / (100 - min)) * 206;
  const path = (family: Family | 'teacher', min: number) =>
    durations.map((s, i) => (i ? 'L' : 'M') + x(s) + ',' + y(rate(family, s), min)).join(' ');
  const band = (family: Family, min: number) =>
    durations
      .map((s, i) => (i ? 'L' : 'M') + x(s) + ',' + y(interval(family, s)[1], min))
      .join(' ') +
    ' ' +
    [...durations]
      .reverse()
      .map((s) => 'L' + x(s) + ',' + y(interval(family, s)[0], min))
      .join(' ') +
    'Z';
</script>

<figure class="science-figure performance" aria-labelledby="performance-title">
  <div class="figure-heading">
    <div>
      <span class="figure-number">04 / Principal result</span>
      <h3 id="performance-title">Save rate across blackout durations</h3>
    </div>
    <span class="evidence-label">Measured · untouched test shots</span>
  </div>
  <div class="curve-grid">
    {#each groups as group}
      <div class="curve-panel">
        <div class="panel-heading">
          <h4>{group.name}</h4>
          <span>{group.subtitle}</span>
        </div>
        <svg
          viewBox="0 0 440 307"
          role="img"
          aria-label={group.name +
            ': save rates across six tested blackout durations. ' +
            group.subtitle +
            '. Full values are in the table below.'}
        >
          <rect x={x(20)} y="34" width={x(25) - x(20)} height="216" fill="#efeee7" />
          <text x="405" y="25" text-anchor="end" class="svg-small">500 ms: extrapolation</text>
          {#each group.min === 0 ? [0, 25, 50, 75, 100] : [90, 92, 94, 96, 98, 100] as tick}<line
              x1="46"
              x2="408"
              y1={y(tick / 100, group.min)}
              y2={y(tick / 100, group.min)}
              stroke="#dce1d7"
            /><text x="35" y={y(tick / 100, group.min) + 4} text-anchor="end" class="axis-text"
              >{tick}</text
            >{/each}
          <text x="46" y="25" class="axis-text">Saves (%)</text>
          {#each group.members as family}
            {#if family !== 'teacher'}<path
                d={band(family, group.min)}
                fill={colours[family]}
                opacity={focused && focused !== family ? 0.02 : 0.075}
              />{/if}
          {/each}
          <line
            x1={x(selected)}
            x2={x(selected)}
            y1="38"
            y2="253"
            stroke="#61786d"
            stroke-width="1"
            stroke-dasharray="3 4"
          />
          {#each group.members as family}
            <g opacity={focused && focused !== family ? 0.18 : 1}>
              <path
                d={path(family, group.min)}
                fill="none"
                stroke={colours[family]}
                stroke-width={family === 'structured_k0' ? 2.6 : 1.8}
                stroke-dasharray={dashes[family]}
              />
              {#each durations as duration}<circle
                  cx={x(duration)}
                  cy={y(rate(family, duration), group.min)}
                  r={selected === duration ? 3.5 : 2.2}
                  fill={colours[family]}
                />{/each}
            </g>
          {/each}
          {#each durations as duration}<text
              x={x(duration)}
              y="271"
              text-anchor="middle"
              class="axis-text">{duration * 20}</text
            >{/each}
          <text x="227" y="297" text-anchor="middle" class="axis-text">Blackout duration (ms)</text>
        </svg>
        <div class="curve-key">
          {#each group.members as family}<button
              class:muted={focused && focused !== family}
              onmouseenter={() => (focused = family)}
              onmouseleave={() => (focused = null)}
              onfocus={() => (focused = family)}
              onblur={() => (focused = null)}
              aria-label={'Highlight ' + labels[family]}
              ><svg viewBox="0 0 20 8" aria-hidden="true"
                ><path
                  d="M0 4H20"
                  stroke={colours[family]}
                  stroke-width="2"
                  stroke-dasharray={dashes[family]}
                /></svg
              >{labels[family]}</button
            >{/each}
        </div>
      </div>
    {/each}
  </div>
  <div class="duration-control interactive-control">
    <div class="duration-title">
      <label for="blackout-duration">Inspect a tested duration</label><output
        for="blackout-duration"
        >{selected * 20} ms{selected === 25 ? ' · extrapolation' : ''}</output
      >
    </div>
    <input
      id="blackout-duration"
      type="range"
      min="0"
      max="25"
      step="5"
      bind:value={selected}
      aria-valuetext={selected * 20 +
        ' milliseconds' +
        (selected === 25 ? ', extrapolation beyond training durations' : '')}
    />
    <div class="duration-ticks">
      {#each durations as d}<button class:chosen={d === selected} onclick={() => (selected = d)}
          >{d * 20}</button
        >{/each}
    </div>
  </div>
  <div class="selected-results">
    <div class="readout-heading">
      <h4>{selected * 20} ms blackout</h4>
      <span>Mean [95% interval] · five seed observations</span>
    </div>
    <div class="readout-grid" aria-live="polite">
      {#each families as family}
        <div class="seed-readout">
          <div>
            <span class="family-dot" style:background={colours[family]}></span><strong
              >{labels[family]}</strong
            ><span class="readout-rate"
              >{percentage(rate(family, selected))}%
              <small
                >[{interval(family, selected)
                  .map((v) => percentage(v))
                  .join(', ')}]</small
              ></span
            >
          </div>
          <div class="seed-values" aria-label="Save rates for seeds 14303 to 14307">
            {#each Object.entries(principal.student_seed_save_rates[family][selected]) as [seed, value]}<span
                title={'Seed ' + seed}
                ><i style:background={colours[family]}></i>{percentage(value)}</span
              >{/each}
          </div>
        </div>
      {/each}
      <div class="seed-readout teacher-readout">
        <div>
          <span class="family-dot" style:background={colours.teacher}></span><strong>Teacher</strong
          ><span class="readout-rate">{percentage(rate('teacher', selected))}%</span>
        </div>
        <p>One frozen teacher policy.</p>
      </div>
    </div>
  </div>
  <details class="figure-data">
    <summary>All recorded means and 95% intervals</summary>
    <div class="table-scroll">
      <table>
        <caption>Save rate (%) · one column per tested blackout duration</caption><thead
          ><tr
            ><th>Family</th>{#each durations as d}<th>{d * 20} ms</th>{/each}</tr
          ></thead
        ><tbody
          >{#each [...families, 'teacher'] as family}<tr
              ><th>{labels[family as Family | 'teacher']}</th>{#each durations as d}<td
                  >{percentage(
                    rate(family as Family | 'teacher', d),
                  )}{#if family !== 'teacher'}<small
                      >[{interval(family as Family, d)
                        .map((v) => percentage(v))
                        .join(', ')}]</small
                    >{/if}</td
                >{/each}</tr
            >{/each}</tbody
        >
      </table>
    </div>
  </details>
  <figcaption>
    Mean save rates over five matched training seeds, evaluated on 225 fresh paired shots at each
    duration. Shading shows 95% hierarchical cluster-bootstrap intervals. Lines connect the six
    measured durations; intermediate durations were not evaluated. The recurrent comparison uses an
    expanded vertical scale. The readout lists all five seed rates in order 14303–14307. The 500 ms
    condition tests extrapolation beyond the 0–400 ms training range.
  </figcaption>
</figure>
