<script lang="ts">
  import katex from 'katex';
  let { tex, display = false }: { tex: string; display?: boolean } = $props();
  const html = $derived(
    katex.renderToString(tex, {
      displayMode: display,
      throwOnError: true,
      trust: false,
      output: 'htmlAndMathml',
    }),
  );
</script>

<span class:display-math={display}>{@html html}</span>
