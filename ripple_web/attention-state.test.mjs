import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {partitionAttention, summarizeRevealed} from './attention-state.mjs';

const radar = JSON.parse(readFileSync(new URL('./data/radar.json', import.meta.url)));
const replays = JSON.parse(readFileSync(new URL('./data/replays.json', import.meta.url)));
const replayFor = (event) => replays.events.find((replay) => replay.event_id === event.event_id);
const symbols = (records) => records.map((record) => record.asset);
const expectedFocus = [ ['ADA', 'AVAX', 'LINK'], ['DOGE', 'ADA', 'AVAX'], ['AVAX', 'LINK', 'ADA'] ];

function fixture() {
  const event = {
    event_id: 'example', timestamp: '2026-01-01T00:00:00Z',
    assets: ['D', 'C', 'B', 'A'].map((asset, index) => ({asset, vol_past30: index, actual_mdd: 99})),
  };
  const replay = {
    event_id: event.event_id, timestamp: event.timestamp,
    assets: Object.fromEntries(event.assets.map(({asset}) => [asset, {
      spot: 100, observed: [...Array(31).fill(100), ...Array(30).fill(100)], actual_mdd: 99,
    }])),
  };
  return {event, replay};
}

test('all three frozen events keep exactly three focused and seven deferred, with deterministic display ranks', () => {
  assert.equal(radar.events.length, 3);
  radar.events.forEach((event, index) => {
    const before = structuredClone(event);
    const result = partitionAttention(event);
    assert.deepEqual(symbols(result.focused), expectedFocus[index]);
    assert.equal(result.deferred.length, 7);
    assert.equal(new Set(symbols([...result.focused, ...result.deferred])).size, 10);
    assert.deepEqual(result.ranked.map((record) => record.rank), [1,2,3,4,5,6,7,8,9,10]);
    assert.equal(result.cutoffTied, false);
    for (const record of result.ranked) {
      const {rank, ...sourceFields} = record;
      assert.deepEqual(sourceFields, event.assets.find((source) => source.asset === record.asset));
      assert.notEqual(record, event.assets.find((source) => source.asset === record.asset));
    }
    assert.deepEqual(event, before);
  });
});

test('cutoff ties use lexical symbols and still retain every asset exactly once', () => {
  const {event} = fixture();
  event.assets.forEach((record) => { record.vol_past30 = 0; });
  const result = partitionAttention(event, 2);
  assert.deepEqual(symbols(result.ranked), ['A', 'B', 'C', 'D']);
  assert.deepEqual(symbols(result.focused), ['A', 'B']);
  assert.deepEqual(symbols(result.deferred), ['C', 'D']);
  assert.equal(result.cutoffTied, true);
  assert.deepEqual(partitionAttention({...event, assets: [...event.assets].reverse()}, 2), result);
});

test('past volatility alone controls attention regardless of predictions and future actual fields', () => {
  for (const event of radar.events) {
    const changed = structuredClone(event);
    changed.assets.forEach((record, index) => {
      record.actual_mdd = index * 1000;
      record.mdd_p50 = index % 2 ? Infinity : -1;
      record.mdd_p95 = null;
    });
    const original = partitionAttention(event);
    const updated = partitionAttention(changed);
    assert.deepEqual(symbols(updated.ranked), symbols(original.ranked));
    assert.deepEqual(symbols(updated.deferred), symbols(original.deferred));
  }
});

test('invalid records, duplicate assets, volatility values and empty groups fail explicitly', () => {
  for (const event of [null, {}, {assets: null}, {assets: []}]) {
    assert.throws(() => partitionAttention(event));
  }
  for (const badAsset of [undefined, null, '', ' A', 'a', '<A>', 123]) {
    const {event} = fixture();
    event.assets[0].asset = badAsset;
    assert.throws(() => partitionAttention(event), /valid, unique symbols/);
  }
  const {event} = fixture();
  event.assets[0].asset = event.assets[1].asset;
  assert.throws(() => partitionAttention(event), /unique symbols/);
  for (const value of [undefined, null, NaN, Infinity, -Infinity, -1, '0.5']) {
    const {event} = fixture();
    event.assets[0].vol_past30 = value;
    assert.throws(() => partitionAttention(event), /past volatility/);
  }
  for (const limit of [-1, 0, 4, 5, 1.5, NaN, '3']) {
    assert.throws(() => partitionAttention(fixture().event, limit), /limit/);
  }
});

test('minute zero exposes no actual result and partial replay uses drawdown from running peaks', () => {
  const {event, replay} = fixture();
  replay.assets.A.observed.splice(31, 3, 110, 99, 105); // 10% drawdown from 110, not from spot.
  replay.assets.B.observed.splice(31, 3, 90, 95, 100);
  replay.assets.C.observed.splice(31, 3, 100, 100, 100);
  replay.assets.D.observed.splice(31, 3, 100, 80, 85);
  assert.equal(summarizeRevealed(event, replay, 0, 2), null);
  assert.deepEqual(summarizeRevealed(event, replay, 1, 2), {
    minute: 1, focusedMax: {asset: 'B', value: .1}, deferredMax: {asset: 'C', value: 0},
  });
  assert.deepEqual(summarizeRevealed(event, replay, 3, 2), {
    minute: 3, focusedMax: {asset: 'A', value: .1}, deferredMax: {asset: 'D', value: .2},
  });
});

test('real full replay recovers frozen group maxima while selection stays fixed at every minute', () => {
  const expectedMax = [['ADA', 'LTC'], ['DOGE', 'LINK'], ['ADA', 'DOGE']];
  radar.events.forEach((event, index) => {
    const replay = replayFor(event);
    const saved = structuredClone({event, replay});
    const initial = partitionAttention(event);
    assert.equal(summarizeRevealed(event, replay, 0), null);
    for (let minute = 1; minute <= 30; minute++) {
      const summary = summarizeRevealed(event, replay, minute);
      assert.equal(summary.minute, minute);
      assert.ok(symbols(initial.focused).includes(summary.focusedMax.asset));
      assert.ok(symbols(initial.deferred).includes(summary.deferredMax.asset));
      assert.deepEqual(symbols(partitionAttention(event).focused), expectedFocus[index]);
    }
    const final = summarizeRevealed(event, replay, 30);
    assert.deepEqual([final.focusedMax.asset, final.deferredMax.asset], expectedMax[index]);
    for (const group of ['focusedMax', 'deferredMax']) {
      assert.ok(Math.abs(final[group].value - replay.assets[final[group].asset].actual_mdd) < 1e-10);
    }
    assert.deepEqual({event, replay}, saved);
  });
});

test('hidden future changes, absent future observations and full-horizon actual fields cannot leak into a partial view', () => {
  const event = radar.events[0], replay = replayFor(event);
  const changed = structuredClone(replay);
  for (const asset of Object.values(changed.assets)) {
    asset.observed = asset.observed.slice(0, 36); // Nothing after +5m is available.
    asset.actual_mdd = 999;
    asset.actual_vol = 999;
  }
  assert.deepEqual(summarizeRevealed(event, changed, 5), summarizeRevealed(event, replay, 5));
  assert.throws(() => summarizeRevealed(event, changed, 6), /Incomplete/);
  for (const asset of Object.values(changed.assets)) asset.observed.push(null);
  assert.deepEqual(summarizeRevealed(event, changed, 5), summarizeRevealed(event, replay, 5));
  assert.throws(() => summarizeRevealed(event, changed, 6), /Missing observed price/);
});

test('incomplete or invalid pairings and revealed missing prices throw instead of inventing zero', () => {
  for (const mutation of [
    (replay) => { replay.event_id = 'other'; },
    (replay) => { replay.timestamp = '2026-01-02T00:00:00Z'; },
    (replay) => { delete replay.assets.D; },
    (replay) => { replay.assets.E = replay.assets.D; },
    (replay) => { replay.assets.E = replay.assets.D; delete replay.assets.D; },
    (replay) => { replay.assets.D.observed = null; },
    (replay) => { replay.assets.D.spot = 0; },
    (replay) => { replay.assets.D.observed[30] = 99; },
  ]) {
    const {event, replay} = fixture();
    mutation(replay);
    assert.throws(() => summarizeRevealed(event, replay, 0));
    assert.throws(() => summarizeRevealed(event, replay, 1));
  }
  for (const missing of [undefined, null, NaN, Infinity, -1, 0, '100']) {
    const {event, replay} = fixture();
    replay.assets.D.observed[31] = missing;
    assert.equal(summarizeRevealed(event, replay, 0), null);
    assert.throws(() => summarizeRevealed(event, replay, 1), /Missing observed price/);
  }
  const {event, replay} = fixture();
  for (const minute of [-1, 31, .5, NaN, undefined, '5']) {
    assert.throws(() => summarizeRevealed(event, replay, minute), /minute/);
  }
});
