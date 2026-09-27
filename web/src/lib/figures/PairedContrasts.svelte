<script lang="ts">
  import { principal, labels } from '../data';
  import { signed } from '../science';
  const keys = [
    'structured_k0_minus_feed_forward',
    'structured_k0_minus_finite_stack_10',
    'structured_k0_minus_gru_n64',
    'structured_k1_minus_structured_k0',
    'structured_k2_minus_structured_k0',
    'structured_k4_minus_structured_k0',
  ];
  const rows = keys.map((k) => principal.planned_differences_at_20_steps[k]);
  const x = (v: number) => 10 + ((v + 5) / 75) * 90;
</script>

<figure class="science-figure contrasts" aria-labelledby="contrasts-title">
  <div class="figure-heading">
    <div>
      <span class="figure-number">05 / Planned comparisons</span>
      <h3 id="contrasts-title">Paired save-rate differences at 400 ms</h3>
    </div>
    <span class="evidence-label">Measured · fixed at 400 ms</span>
  </div>
  <div class="forest-header">
    <span>Left policy − right policy</span><span>Difference in percentage points</span><span
      >Estimate [95% interval]</span
    >
  </div>
  {#each rows as row}
    <div class="forest-row">
      <span class="contrast-name">{labels[row.left]} <span>− {labels[row.right]}</span></span>
      <div class="forest-mark">
        <div class="forest-zero" style:left={x(0) + '%'}></div>
        <div
          class="forest-interval"
          style:left={x(row.percentile_95_interval_points[0]) + '%'}
          style:width={x(row.percentile_95_interval_points[1]) -
            x(row.percentile_95_interval_points[0]) +
            '%'}
        ></div>
        <span class="forest-point" style:left={x(row.estimate_points) + '%'}></span>
      </div>
      <span class="contrast-value"
        >{signed(row.estimate_points)}
        <small>[{row.percentile_95_interval_points.map((v) => signed(v)).join(', ')}]</small></span
      >
    </div>
  {/each}
  <div class="forest-axis">
    <div></div>
    <div>
      {#each [0, 20, 40, 60] as t}<span style:left={x(t) + '%'}>{t}</span>{/each}
    </div>
    <div></div>
  </div>
  <p class="figure-insight">
    For <strong>k=0 minus GRU-64, the interval is [−1.2, +2.5] points</strong>. It includes both a
    small disadvantage and a small advantage for k=0; this comparison does not establish
    equivalence.
  </p>
  <figcaption>
    The six predeclared comparisons reported in the paper. Dots show paired differences and lines
    show their 95% bootstrap intervals. The dashed line marks zero. These results remain at 400 ms
    when you change the duration selector above.
  </figcaption>
</figure>
