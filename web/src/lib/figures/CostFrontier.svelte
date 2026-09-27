<script lang="ts">
  import { evidence, principal, families, colours, labels, median, type Family } from '../data';
  const row = (f: Family) => evidence.efficiency.find((r) => r.family === f)!;
  const y = (v: number) => 268 - ((v * 100 - 30) / 70) * 220;
  const xs = [(v: number) => 45 + ((v - 8) / 38) * 370, (v: number) => 45 + (v / 32000) * 370];
  const labelPositions: Record<Family, [number, number]> = {
    feed_forward: [80, 246],
    finite_stack_10: [85, 116],
    structured_k0: [117, 32],
    structured_k1: [252, 101],
    structured_k2: [252, 28],
    structured_k4: [252, 69],
    gru_n64: [361, 133],
  };
  const parameterLabels: Record<Family, [number, number]> = {
    feed_forward: [78, 246],
    finite_stack_10: [65, 114],
    structured_k0: [60, 33],
    structured_k1: [242, 106],
    structured_k2: [242, 28],
    structured_k4: [242, 68],
    gru_n64: [352, 128],
  };
  const short = (f: Family) => (f.startsWith('structured') ? 'k=' + f.at(-1) : labels[f]);
</script>

<figure class="science-figure frontier" aria-labelledby="cost-title">
  <div class="figure-heading">
    <div>
      <span class="figure-number">06 / Performance and cost</span>
      <h3 id="cost-title">Save rate against latency and parameter count</h3>
    </div>
    <span class="evidence-label">Measured · five seeds per family</span>
  </div>
  <div class="curve-grid">
    {#each ['CPU latency', 'Total parameters'] as name, i}
      <div class="curve-panel">
        <div class="panel-heading">
          <h4>{name}</h4>
          <span>400 ms blackout · expanded 30–100% scale</span>
        </div>
        <svg
          viewBox="0 0 440 325"
          role="img"
          aria-label={'400 millisecond save rate against ' +
            name +
            '. Five small seed points and directly labelled family medians; full numerical measurements follow in the table.'}
        >
          {#each [40, 60, 80, 100] as tick}<line
              x1="45"
              x2="415"
              y1={y(tick / 100)}
              y2={y(tick / 100)}
              stroke="#dce1d7"
            /><text x="34" y={y(tick / 100) + 4} text-anchor="end" class="axis-text">{tick}</text
            >{/each}
          <text x="45" y="13" class="axis-text">Saves (%)</text>
          {#each families as f}
            {@const seedPoints = principal.efficiency_seed_points[f]}
            {@const rates = principal.student_seed_save_rates[f]['20']}
            {@const cx = xs[i](
              i === 0
                ? median(seedPoints.map((s) => s.median_microseconds))
                : row(f).total_trainable_parameters,
            )}
            {@const cy = y(median(Object.values(rates)))}
            {@const pos = (i === 0 ? labelPositions : parameterLabels)[f]}
            {#each seedPoints as seed}<circle
                cx={xs[i](i === 0 ? seed.median_microseconds : row(f).total_trainable_parameters)}
                cy={y(rates[seed.training_seed])}
                r="2.8"
                fill={colours[f]}
                opacity="0.5"
              />{/each}
            <line
              x1={cx}
              y1={cy}
              x2={pos[0]}
              y2={pos[1] + 4}
              stroke={colours[f]}
              stroke-width=".7"
              opacity=".7"
            />
            <path
              d={'M' + cx + ',' + (cy - 4) + 'l4,4 -4,4 -4,-4 Z'}
              fill={colours[f]}
              stroke="var(--paper)"
              stroke-width="1"
            />
            <text
              x={pos[0]}
              y={pos[1]}
              class="frontier-label"
              fill={colours[f]}
              text-anchor={f === 'gru_n64' ? 'middle' : 'start'}>{short(f)}</text
            >
          {/each}
          {#each i === 0 ? [10, 20, 30, 40] : [0, 10000, 20000, 30000] as tick}<text
              x={xs[i](tick)}
              y="289"
              text-anchor="middle"
              class="axis-text">{i === 0 ? tick : tick / 1000 + 'k'}</text
            >{/each}
          <text x="230" y="315" text-anchor="middle" class="axis-text"
            >{i === 0 ? 'CPU latency (µs per step)' : 'Total trainable parameters'}</text
          >
        </svg>
      </div>
    {/each}
  </div>
  <p class="figure-insight">
    <strong>k=0 uses 12,002 parameters and takes 23.0 µs per step.</strong> GRU-64 uses 28,898 parameters
    and takes 40.1 µs: 58.5% fewer parameters and 42.5% lower measured latency for k=0 under this protocol.
  </p>
  <figcaption>
    Circles show the five seeds and diamonds mark family medians. Coincident points overlap. The
    medians plotted here can differ from the family means: 98.3% for k=0 and 97.8% for GRU-64.
    Latency measures batch-one, float32 inference on an isolated Intel Core Ultra 9 285 CPU core.
    Structured students use a verified compiled kernel; the other families use BLAS-backed NumPy.
    Timings therefore depend on the implementation. They exclude physics, rendering and inverse
    kinematics.
  </figcaption>
</figure>
