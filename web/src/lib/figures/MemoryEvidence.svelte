<script lang="ts">
  import { evidence } from '../data';
  import { percentage } from '../science';
  const m = evidence.memory['20'],
    a = evidence.ablation['20'];
  const panels = [
    {
      title: 'Teacher versus memoryless PPO',
      study: 'Separate memoryless-PPO comparison',
      rows: [
        { name: 'Teacher', value: m.teacher_save_rate },
        { name: 'Memoryless PPO', value: m.feed_forward_save_rate },
      ],
      result: '+' + percentage(m.paired_teacher_advantage) + ' points',
      ci: m.paired_teacher_advantage_ci95,
      note: 'This PPO baseline is not the distilled feed-forward student below.',
    },
    {
      title: 'Teacher with and without a state reset',
      study: 'Teacher-state reset at blackout onset',
      rows: [
        { name: 'Normal teacher', value: a.recurrent_save_rate },
        { name: 'State reset', value: a.reset_at_blackout_save_rate },
      ],
      result: '−' + percentage(a.paired_save_rate_drop) + ' points after reset',
      ci: a.paired_save_rate_drop_ci95,
      note: 'Without blackout, both runs save 99.6%; all 225 outcome records are identical.',
    },
  ];
</script>

<figure class="science-figure" aria-labelledby="memory-evidence-title">
  <div class="figure-heading">
    <div>
      <span class="figure-number">02 / Two checks</span>
      <h3 id="memory-evidence-title">Two tests of the memory requirement</h3>
    </div>
    <span class="evidence-label">Measured · validation studies</span>
  </div>
  <div class="evidence-grid">
    {#each panels as panel}
      <div class="evidence-panel">
        <h4>{panel.title}</h4>
        <p class="study-label">{panel.study} · 400 ms</p>
        {#each panel.rows as row, i}<div class="bar-row">
            <div><span>{row.name}</span><strong>{percentage(row.value)}%</strong></div>
            <div class="bar-track">
              <span
                style:width={100 * row.value + '%'}
                style:background={i === 0 ? 'var(--green)' : '#ad9e7c'}
              ></span>
            </div>
          </div>{/each}
        <p class="effect-size">{panel.result}</p>
        <p class="effect-ci">
          Paired 95% interval for {panel === panels[0] ? 'advantage' : 'drop'}: [{percentage(
            panel.ci[0],
          )}, {percentage(panel.ci[1])}]
        </p>
        <p class="study-note">{panel.note}</p>
      </div>
    {/each}
  </div>
  <figcaption>
    Each experiment uses 225 paired validation shots per duration. The first compares policies; the
    second changes only the teacher’s carried state. These studies use separate evaluation
    protocols, so their teacher rates are reported separately from the principal test.
  </figcaption>
</figure>
