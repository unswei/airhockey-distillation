import { readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const root = new URL('../public/', import.meta.url);
assert.deepEqual(
  (await readdir(root, { recursive: true })).filter((path) => /\.pdf$/i.test(path)),
  [],
  'PDF publication is temporarily disabled. Keep manuscript PDFs in the Paper repository.',
);
const manifest = JSON.parse(await readFile(new URL('data/sources.json', root), 'utf8'));
for (const [path, hash] of Object.entries(manifest.assets)) {
  assert.equal(
    createHash('sha256')
      .update(await readFile(new URL(path, root)))
      .digest('hex'),
    hash,
    'Changed asset: ' + path,
  );
}
assert.equal(
  await readFile(new URL('SCIENTIFIC_FIDELITY.md', root), 'utf8'),
  await readFile(new URL('../../SCIENTIFIC_FIDELITY.md', import.meta.url), 'utf8'),
  'Public fidelity note differs from source.',
);
console.log('Manuscript assets and scientific-fidelity copy verified.');
