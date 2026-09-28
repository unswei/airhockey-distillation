# Interactive research article

Web companion to the air-hockey memory-distillation paper, built with Svelte,
TypeScript, Vite and KaTeX. The site is static and can be hosted on GitHub Pages.

[Read the interactive article](https://unswei.github.io/airhockey-distillation/).

## Local development

Requires Node.js 22.18 or later; the deployment workflow uses Node 24.

```sh
cd web
npm ci
npm run dev
```

Open [the local article](http://127.0.0.1:5173/airhockey-distillation/).

## Build and test

```sh
npm run check
npm test
npm run build
npm run preview
```

The build writes to `web/dist/`. The production preview runs at
[port 4173](http://127.0.0.1:4173/airhockey-distillation/).

To run browser tests after building:

```sh
npx playwright install chromium
npm run test:browser
```

## Deployment

In the repository’s **Settings → Pages**, select **GitHub Actions** as the
source. Run **Publish interactive article** from the Actions tab on `main`.
Deployment is manual; pushing a commit does not publish the site.

The site uses `/airhockey-distillation/` as its base path, configured in
`vite.config.ts`.

## Contents

- `src/App.svelte`: article text and references.
- `src/lib/figures/`: interactive and static figures.
- `public/data/`: frozen results and source hashes, verified during each build.

The paper PDF is not published yet. The article shows “Paper (soon)”; builds
reject PDFs in `public/` until publication is enabled deliberately.

See [Scientific fidelity](../SCIENTIFIC_FIDELITY.md) for data provenance and
the distinction between measured results and illustrative examples.
