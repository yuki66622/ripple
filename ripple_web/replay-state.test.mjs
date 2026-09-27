import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {clampMinute, visibleObservations, revealedDrawdown, viewSeries} from './replay-state.mjs';
import {replayTitle} from './blocks/replay.mjs';
const data = JSON.parse(readFileSync(new URL('./data/replays.json', import.meta.url)));
const explorer = JSON.parse(readFileSync(new URL('./data/explorer.json', import.meta.url)));
test('all 30 asset replays reveal no future initially, and recover the frozen full actual MDD', () => {
  let count = 0;
  for (const event of data.events) for (const asset of Object.values(event.assets)) {
    count++;
    assert.equal(visibleObservations(asset, 0).length, 31);
    assert.equal(revealedDrawdown(asset, 0), null);
    assert.equal(visibleObservations(asset, 30).length, 61);
    assert.ok(Math.abs(revealedDrawdown(asset, 30) - asset.actual_mdd) < 1e-10);
  }
  assert.equal(count, 30);
});
test('unrevealed observations cannot change visible series, extents or drawdown', () => {
  const original = data.events[0].assets.BTC;
  const mutated = structuredClone(original);
  mutated.observed = mutated.observed.map((value,i) => i >= 36 ? value * .01 : value);
  assert.deepEqual(viewSeries(original,5), viewSeries(mutated,5));
  assert.notEqual(revealedDrawdown(original,30), revealedDrawdown(mutated,30));
});
test('replay headline uses median of per-path drawdowns, never MDD of median price curve', () => {
  const event = data.events[0];
  const mdd = explorer.replays[event.event_id].mdd.BTC;
  const title = replayTitle(event,'BTC',{mdd,revealedMinutes:0});
  assert.ok(title.includes((mdd.p50 * 100).toFixed(2) + '%'));
  assert.ok(title.includes('尚未揭示'));
});
test('time boundaries and invalid partial prices', () => {
  assert.equal(clampMinute(-1),0); assert.equal(clampMinute(31),30);
  assert.equal(clampMinute(NaN),0); assert.equal(clampMinute(4.7),5);
  const asset = structuredClone(data.events[0].assets.BTC);
  asset.observed[31] = -1;
  assert.equal(revealedDrawdown(asset,1),null);
});
