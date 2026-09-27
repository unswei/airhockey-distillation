# Interactive research article

Svelte 5 + TypeScript + Vite, bespoke SVG/HTML figures and KaTeX. The build
prerenders the complete article and hydrates three interactive explanations.
No server, inference, analytics, external fonts or CDN is required at runtime.
D3 is unnecessary for these small, fixed datasets.

The article lives in the existing code repository. It does not modify Paper,
the Python environment, frozen experiments or the test-release gate.

## Build and preview

Use Node 24 LTS (minimum 22.18):

    cd web
    npm ci
    npm run check
    npm test
    npm run build
    npm run preview

Open http://127.0.0.1:4173/airhockey-distillation/ .
For development, npm run dev uses port 5173.

package-lock.json is separate from the experiment uv.lock. Output is web/dist/.
The build fails if frozen source hashes, exported measurements, manuscript
assets or the public fidelity copy change. A sibling Paper checkout and TeX
installation are not needed for a normal build.

## Browser verification

After building:

    npx playwright install chromium
    npm run test:browser

Playwright starts the production preview. For installed local Chrome, use
PLAYWRIGHT_CHANNEL=chrome npm run test:browser. Tests cover interactions,
all 35 seed readouts, fixed contrasts, keyboard controls, assets, no-JavaScript
rendering and 1440/768/390/320 px layouts. Screenshots go to test-results/.

## GitHub Pages deployment

The Vite base is /airhockey-distillation/. The prepared workflow
.github/workflows/article-pages.yml is manual-only; a push does not publish.

1. Review the draft PDF and exported assets as potentially public material.
2. In repository Settings → Pages, select GitHub Actions. Check availability
   for the current repository visibility and account plan. Do not change a
   private repository to public merely to make deployment work.
3. Run “Publish interactive article” from Actions on main.
4. Checks, a production build and browser tests run before deploying only
   web/dist through the github-pages environment.

Intended URL: https://unswei.github.io/airhockey-distillation/ .
This is a deployment target, not an assertion that the site is published.
For another repository name or custom domain, change base in vite.config.ts
and the browser-test base URL.

## Source map

- ../ARTICLE_PLAN.md: approved narrative and ranked interactions.
- ../SCIENTIFIC_FIDELITY.md: evidence classes and interpretation boundaries.
- src/App.svelte: narrative, references and links.
- src/lib/figures/: reusable explanatory units.
- src/lib/science.ts: masking, stack bookkeeping and parameter formulae.
- src/lib/data.ts: typed access and family colours, labels and dashes.
- scripts/export-data.mjs: hash-checked frozen-source export.
- public/data/evidence.json: compact measurements and source hashes.
- public/data/sources.json: manuscript and public-asset hashes.
- public/paper.pdf and references.bib: manuscript snapshot.

## Updating sources

Never silently regenerate from newer results. Review the manuscript, update
explicit pinned hashes/commit and the fidelity note, then npm run data:export.
To update the manuscript snapshot, compile main.tex with latexmk into an
output directory outside Paper. Deliberately update the expected commit and
hashes in the snapshot script, then run:

    node scripts/snapshot-paper.mjs /absolute/path/to/Paper /absolute/path/to/main.pdf

This copies the PDF, bibliography and fidelity note and records their hashes.
It never edits Paper. Rerun all checks and inspect screenshots. The website
must never construct or reopen principal_test.

## Accessibility and design

Warm paper, self-hosted Source Serif 4 and Inter, restrained colour-blind-friendly
colours with redundant labels/dashes. Keyboard/touch controls, no autoplay,
reduced-motion support, independently scrolling tables, MathML equations and
visible numerical information. Text and default figures remain static HTML
without JavaScript; important quantities are never hidden only behind hover.
