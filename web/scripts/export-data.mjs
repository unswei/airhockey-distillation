import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = new URL('../../', import.meta.url);
const sources = {
  'results/principal_sweep_v1_statistics.json':
    '55ad1f50399e54487728ade10e6605a993064369a80ec48c0674704f2f378902',
  'results/principal_sweep_v1_efficiency_table.csv':
    'a6e8edc0ebb541c3922b9b589926527519045f370c46c370c78fc749392e9311',
  'results/principal_structured_optimisation_v4.json':
    'a1a5add1c5a5e08b40bec2331e35b9fc9aea140119cd9c0c88a84bec9f3f8c85',
  'results/stage_b_memory_validation_v3.json':
    '9b074299b5794b4a35af30737fcbba32753a0318f9eda9b11ab492e916a68739',
  'results/causal_memory_ablation.json':
    '919a1306453542950b2002a0b797a6279473fe588abc6a4fb1c1021b8e54a67a',
  'results/structured_n64_k2_all_steps_v2.json':
    'c04bc00f8db9ba55887eacabfbe6eb6281ee6bd2c4dbe3421e1a347ab2e21ee4',
  'results/structured_n64_k2_shadow_round1_v1.json':
    '734771381de4766b7fc844a99e5f1e968e8c4a9cafc15a643ffa1186c8dacdd7',
};
const loaded = {};
for (const [path, hash] of Object.entries(sources)) {
  const bytes = await readFile(new URL(path, root));
  assert.equal(
    createHash('sha256').update(bytes).digest('hex'),
    hash,
    `Frozen source changed: ${path}`,
  );
  loaded[path] = path.endsWith('.json') ? JSON.parse(bytes) : bytes.toString('utf8');
}
const statistics = loaded['results/principal_sweep_v1_statistics.json'];
const [header, ...rows] = loaded['results/principal_sweep_v1_efficiency_table.csv']
  .trim()
  .split(/\r?\n/);
const columns = header.split(',');
const efficiency = rows.map((row) =>
  Object.fromEntries(row.split(',').map((v, i) => [columns[i], i ? Number(v) : v])),
);
assert.equal(efficiency.length, 7);
assert.equal(statistics.status, 'completed');
for (const [family, durations] of Object.entries(statistics.student_seed_save_rates)) {
  assert.equal(Object.keys(durations).length, 6);
  for (const [duration, seeds] of Object.entries(durations)) {
    assert.equal(Object.keys(seeds).length, 5);
    const mean = Object.values(seeds).reduce((a, b) => a + b, 0) / 5;
    assert.ok(Math.abs(mean - statistics.student_save_rates[family][duration]) < 1e-12);
  }
}
const before = loaded['results/structured_n64_k2_all_steps_v2.json'];
const after = loaded['results/structured_n64_k2_shadow_round1_v1.json'];
const payload = {
  schema: 1,
  manuscript: {
    commit: '166a4a2154f0896c68497f0070fe9813e0ddf4a4',
    sha256: 'bedeb60018eac4b87ed601b588fbb4abfcc6902eb8a6e7ed1c61994ea01b4693',
  },
  sourceCodeCommit: '1f68358fb08c56e9479a75b7d47fc57873dfd157',
  sources,
  principal: Object.fromEntries(
    [
      'bootstrap',
      'student_save_rates',
      'student_save_rate_95_intervals',
      'student_seed_save_rates',
      'teacher_save_rates',
      'planned_differences_at_20_steps',
      'efficiency_seed_points',
    ].map((key) => [key, statistics[key]]),
  ),
  efficiency,
  memory: loaded['results/stage_b_memory_validation_v3.json'].by_blackout_steps,
  ablation: loaded['results/causal_memory_ablation.json'].by_blackout_steps,
  pilot: {
    before: before.gate.save_rate,
    after: after.no_blackout_gate.save_rate,
    episodes: before.gate.episodes,
    pairedValidation: after.full_paired_validation.save_rate,
  },
};
const output = new URL('../public/data/evidence.json', import.meta.url);
const content = JSON.stringify(payload, null, 2) + '\n';
if (process.argv.includes('--check')) {
  assert.equal(
    await readFile(output, 'utf8'),
    content,
    'Run npm run data:export and review the evidence export.',
  );
  console.log(
    'Frozen sources and complete web export verified (7 families × 5 seeds × 6 durations).',
  );
} else {
  await mkdir(new URL('.', output), { recursive: true });
  await writeFile(output, content);
  console.log(`Exported verified evidence to ${fileURLToPath(output)}`);
}
