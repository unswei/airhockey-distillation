<script lang="ts">
  import { evidence, rate, interval, labels, type Family } from '../data';
  import { percentage } from '../science';
  const n = (value: number) => value.toLocaleString('en-AU');
</script>

<figure class="science-figure complete-table">
  <div class="figure-heading">
    <div>
      <span class="figure-number">Table / Complete comparison</span>
      <h3>Save rates and costs for all seven student families</h3>
    </div>
  </div>
  <p class="table-hint">On a narrow screen, scroll the table horizontally.</p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll the wide table.) -->
  <div
    class="table-scroll"
    tabindex="0"
    role="region"
    aria-label="Complete student comparison; scroll horizontally to see all measurements"
  >
    <table>
      <caption>Save rates and computational costs. Intervals and rates are percentages.</caption
      ><thead
        ><tr
          ><th scope="col">Student</th><th scope="col"
            >400 ms saves<br /><span>[95% interval]</span></th
          ><th scope="col">Overall<br />saves</th><th scope="col">Total<br />params</th><th
            scope="col">Core<br />params</th
          ><th scope="col">Carry<br />bytes</th><th scope="col">Recurrent<br />bytes</th><th
            scope="col">Multiply–<br />adds</th
          ><th scope="col">CPU µs<br />median / p95</th></tr
        ></thead
      ><tbody>
        {#each evidence.efficiency as row}
          {@const f = row.family as Family}
          <tr class:reference-row={f === 'structured_k0'}
            ><th scope="row">{labels[f]}</th><td
              >{percentage(rate(f, 20))}<small
                >[{interval(f, 20)
                  .map((v) => percentage(v))
                  .join(', ')}]</small
              ></td
            ><td>{percentage(row.overall_core_test_save_rate)}</td><td
              >{n(row.total_trainable_parameters)}</td
            ><td>{n(row.core_parameters)}</td><td>{row.total_policy_carry_bytes}</td><td
              >{row.recurrent_memory_bytes}</td
            ><td>{n(row.multiply_adds_per_step)}</td><td
              >{row.cpu_latency_median_microseconds.toFixed(1)} / {row.cpu_latency_p95_microseconds.toFixed(
                1,
              )}</td
            ></tr
          >
        {/each}
      </tbody>
    </table>
  </div>
  <figcaption>
    “Overall” pools the five 0–400 ms conditions and excludes 500 ms. Carry includes the finite
    stack or the recurrent state and previous action, as applicable. For each latency statistic, we
    report the median of the five per-seed measurements. Multiply–adds are analytic counts per
    policy step. Costs are reported for students only.
  </figcaption>
</figure>
