<script lang="ts">
  import Math from '../Math.svelte';
  import { structuredCounts } from '../science';
  let rank = $state(0);
  let term = $state('memory');
  const counts = $derived(structuredCounts(rank));
  const terms = [
    { id: 'memory', name: 'Carried state', tex: 'Az_{t-1}' },
    { id: 'input', name: 'New input', tex: 'B_x x_t+B_a a_{t-1}+b_z' },
    {
      id: 'innovation',
      name: 'Optional correction',
      tex: 'U\\tanh(Vz_{t-1}+W_x x_t+W_a a_{t-1}+b_r)',
    },
  ];
</script>

<figure class="science-figure architecture" aria-labelledby="architecture-title">
  <div class="figure-heading">
    <div>
      <span class="figure-number">03 / Mechanism</span>
      <h3 id="architecture-title">Varying the nonlinear correction at fixed state size</h3>
    </div>
    <span class="evidence-label">Architecture reconstruction</span>
  </div>
  <div class="rank-control interactive-control" role="group" aria-label="Nonlinear correction rank">
    <span>Correction rank</span>
    {#each [0, 1, 2, 4] as k}<button
        class="rank-button"
        class:selected={rank === k}
        aria-pressed={rank === k}
        onclick={() => (rank = k)}>k = {k}</button
      >{/each}
    <span class="rank-description"
      >{rank === 0 ? 'Linear recurrent update' : rank + '-channel nonlinear correction'}</span
    >
  </div>
  <div class="architecture-flow">
    <div class="network-stage" class:term-highlight={term === 'input'}>
      <span class="diagram-label">Current observation</span>
      <div class="network-box">
        <strong>Observation encoder</strong><span>19 → 64 → 32</span><em>Nonlinear · SiLU</em>
      </div>
      <div class="flow-arrow"><Math tex="x_t" /> <span class="arrow-glyph">→</span></div>
    </div>
    <div class="memory-stage" class:term-highlight={term === 'memory'}>
      <div class="state-heading">
        <strong>64 carried values</strong><span>Always 256 bytes</span>
      </div>
      <div class="state-grid" aria-hidden="true">
        {#each Array(64) as _}<i></i>{/each}
      </div>
      <div class="state-update"><Math tex={'z_{t-1} \\;\\longrightarrow\\; z_t'} /></div>
      <p class="core-explanation">
        {#if rank === 0}Scale each old value, then add the encoded input and previous action.{:else}Keep
          the linear backbone, then add a {rank}-channel nonlinear correction.{/if}
      </p>
      <span class="previous-action">+ previous requested action <Math tex={'a_{t-1}'} /></span>
    </div>
    <div class="network-stage">
      <span class="diagram-label">State + encoded observation</span>
      <div class="network-box">
        <strong>Action head</strong><span>96 → 64 → 2</span><em>Nonlinear · SiLU + tanh</em>
      </div>
      <div class="flow-arrow"><span class="arrow-glyph">→</span> <span>2D mallet target</span></div>
    </div>
  </div>
  <div
    class="innovation-strip"
    class:active={rank > 0}
    class:term-highlight={term === 'innovation'}
  >
    <div>
      <strong>{rank === 0 ? 'No nonlinear correction' : 'Low-rank nonlinear correction'}</strong
      ><span
        >{rank === 0
          ? 'The encoder and action head are still nonlinear.'
          : 'Project → tanh → lift back into the same 64-value state.'}</span
      >
    </div>
    <div class="rank-channels" aria-label={rank + ' nonlinear correction channels'}>
      {#if rank === 0}<span>k = 0</span>{:else}<span>64 →</span>{#each Array(rank) as _}<i
          ></i>{/each}<span>→ 64</span>{/if}
    </div>
  </div>
  <div class="architecture-counts" aria-live="polite">
    <span><strong>{counts.core.toLocaleString('en-AU')}</strong> recurrent-core parameters</span
    ><span><strong>{counts.total.toLocaleString('en-AU')}</strong> total parameters</span><span
      ><strong>64</strong> recurrent state values</span
    >
  </div>
  <div class="equation-explainer">
    <p class="equation-label">The state update, term by term</p>
    <div class="equation-terms">
      <span class="equation-prefix"><Math tex={'z_t ='} /></span>
      {#each terms as item, i}
        {#if i}<span aria-hidden="true">+</span>{/if}
        <button
          class:term-selected={term === item.id}
          class:term-absent={item.id === 'innovation' && rank === 0}
          onclick={() => (term = item.id)}
          aria-pressed={term === item.id}
          aria-label={'Highlight ' + item.name.toLowerCase()}
          ><Math tex={item.tex} /><small
            >{item.id === 'innovation' && rank === 0 ? 'Absent at k = 0' : item.name}</small
          ></button
        >
      {/each}
    </div>
    <details>
      <summary>What does the rank restriction mean?</summary>
      <p>
        The diagonal backbone is <Math tex={'A=\\operatorname{diag}(\\tanh\\alpha)'} />. For fixed
        encoded observation and previous action, the Jacobian of the nonlinear correction has rank
        at most <Math tex="k" />:
      </p>
      <Math
        display
        tex={'\\operatorname{rank}\\!\\left(\\frac{\\partial z_t}{\\partial z_{t-1}}-A\\right)\\leq k.'}
      />
      <p>
        The bound applies to the instantaneous recurrent Jacobian with current inputs fixed. It
        neither bounds the rank of the full feedback policy nor establishes closed-loop stability.
        At <Math tex="k=0" />, the recurrent update is affine; the complete policy is not linear.
      </p>
    </details>
  </div>
  <figcaption>
    Each rank is a separately trained family with the same 64-value state, nonlinear encoder and
    action head. The selector shows how the correction branch changes; it does not modify a
    checkpoint. Squares indicate state dimensions rather than learned values. The previous action
    adds 8 bytes to the carried state, for a total of 264 bytes.
  </figcaption>
</figure>
