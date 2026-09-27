import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { visibleAt, puckInput, stackAt, structuredCounts } from '../src/lib/science.ts';
const data = JSON.parse(readFileSync(new URL('../public/data/evidence.json', import.meta.url)));
test('blackout is [5,25), with 20 masked samples', () => {
  assert.equal(visibleAt(4), true);
  assert.equal(visibleAt(5), false);
  assert.equal(visibleAt(24), false);
  assert.equal(visibleAt(25), true);
  assert.equal(Array.from({ length: 29 }, (_, i) => visibleAt(i)).filter((x) => !x).length, 20);
});
test('hidden triples expose neither puck position nor velocity', () => {
  for (let i = 5; i < 25; i++) {
    assert.deepEqual(puckInput(i, 1), [0, 0, 0]);
    assert.deepEqual(puckInput(i, -1), [0, 0, 0]);
  }
  assert.notDeepEqual(puckInput(4, 1), puckInput(4, -1));
});
test('ten-slot history includes current observation and exact zero padding', () => {
  assert.equal(stackAt(0, 1).filter((x) => x.padding).length, 9);
  assert.equal(stackAt(5, 1).filter((x) => x.value[2]).length, 5);
  assert.equal(stackAt(13, 1).filter((x) => x.value[2]).length, 1);
  assert.equal(stackAt(14, 1).filter((x) => x.value[2]).length, 0);
  assert.equal(stackAt(25, 1).filter((x) => x.value[2]).length, 1);
});
test('architecture counts equal frozen efficiency records at each supported rank', () => {
  for (const k of [0, 1, 2, 4]) {
    const result = structuredCounts(k);
    const row = data.efficiency.find((r) => r.family === `structured_k${k}`);
    assert.equal(result.total, row.total_trainable_parameters);
    assert.equal(result.core, row.core_parameters);
    assert.equal(result.bytes, row.recurrent_memory_bytes);
  }
  assert.throws(() => structuredCounts(3));
});
test('all five seed values remain in the export, and means match frozen means', () => {
  for (const [family, durations] of Object.entries(data.principal.student_seed_save_rates))
    for (const [step, seeds] of Object.entries(durations)) {
      assert.equal(Object.keys(seeds).length, 5);
      assert.ok(
        Math.abs(
          Object.values(seeds).reduce((a, b) => a + b, 0) / 5 -
            data.principal.student_save_rates[family][step],
        ) < 1e-12,
      );
    }
});
