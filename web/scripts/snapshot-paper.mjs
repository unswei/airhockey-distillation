// Explicit maintenance operation, not part of a normal build. Never edits Paper.
import { readFile, writeFile, copyFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { resolve, join } from 'node:path';
import assert from 'node:assert/strict';
const [paperDirectory, pdfFile] = process.argv.slice(2);
if (!paperDirectory || !pdfFile)
  throw new Error('Usage: node scripts/snapshot-paper.mjs PAPER_DIRECTORY COMPILED_PDF');
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const paper = resolve(paperDirectory);
const commit = execFileSync('git', ['-C', paper, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
assert.equal(
  commit,
  '166a4a2154f0896c68497f0070fe9813e0ddf4a4',
  'Review the manuscript snapshot before updating it.',
);
assert.equal(
  sha(await readFile(join(paper, 'main.tex'))),
  'bedeb60018eac4b87ed601b588fbb4abfcc6902eb8a6e7ed1c61994ea01b4693',
);
assert.equal(
  sha(await readFile(join(paper, 'references.bib'))),
  '08cd0d15958aae2b3cc6dbfc035abf13c91a857658b8837d82efd88a42f83d3f',
);
assert.equal(
  execFileSync('git', ['-C', paper, 'status', '--porcelain', '--untracked-files=no'], {
    encoding: 'utf8',
  }).trim(),
  '',
  'Paper has tracked changes.',
);
const publicDir = new URL('../public/', import.meta.url);
const setupImage = new URL('../../docs/assets/upstream-self-play-2023-poster.jpg', import.meta.url);
const setupImageHash = '8514f02ea385aa7f4ba7f49f87751dc18733b974031a7ea42bb975fea384827e';
assert.equal(sha(await readFile(setupImage)), setupImageHash, 'Upstream setup poster changed.');
await mkdir(new URL('data/', publicDir), { recursive: true });
await copyFile(setupImage, new URL('kuka-setup.jpg', publicDir));
await copyFile(resolve(pdfFile), new URL('paper.pdf', publicDir));
await copyFile(join(paper, 'references.bib'), new URL('references.bib', publicDir));
await copyFile(
  new URL('../../SCIENTIFIC_FIDELITY.md', import.meta.url),
  new URL('SCIENTIFIC_FIDELITY.md', publicDir),
);
const sourceFiles = execFileSync('git', ['-C', paper, 'ls-files'], { encoding: 'utf8' })
  .trim()
  .split('\n')
  .filter((p) => /\.(tex|bib|cls|sty|pdf|png|jpg)$/.test(p));
const inputs = Object.fromEntries(
  await Promise.all(
    sourceFiles.map(async (path) => [path, sha(await readFile(join(paper, path)))]),
  ),
);
const assets = Object.fromEntries(
  await Promise.all(
    ['paper.pdf', 'references.bib', 'SCIENTIFIC_FIDELITY.md', 'kuka-setup.jpg'].map(
      async (path) => [path, sha(await readFile(new URL(path, publicDir)))],
    ),
  ),
);
await writeFile(
  new URL('data/sources.json', publicDir),
  JSON.stringify(
    {
      schema: 1,
      manuscriptCommit: commit,
      manuscriptInputs: inputs,
      assets,
      setupImage: {
        source: 'docs/assets/upstream-self-play-2023-poster.jpg',
        run: 'upstream-demo-2023-rendered-v2',
        sha256: setupImageHash,
        role: 'Environment context only; upstream two-robot self-play, not a tracking-loss result',
        presentation: 'Unmodified source file; cropped in CSS, retaining both robots and the table',
      },
      compiler: 'TeX Live 2026 / latexmk; no source edits',
      snapshotDate: '2026-09-27',
    },
    null,
    2,
  ) + '\n',
);
console.log(
  'Copied manuscript PDF, bibliography and fidelity note; recorded source and asset hashes.',
);
