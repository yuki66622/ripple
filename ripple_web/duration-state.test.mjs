import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {tierForAsset, durationStatus} from './duration-state.mjs';

const duration = JSON.parse(readFileSync(new URL('./data/duration.json', import.meta.url)));
const radar = JSON.parse(readFileSync(new URL('./data/radar.json', import.meta.url)));
const sourceBytes = readFileSync(new URL('../research/shock_radar/seven_checks/result_3_5_7.json', import.meta.url));
const decay = JSON.parse(sourceBytes).results.analysis_5_decay;
const tiers = ['major', 'small'];
const fixture = (curve) => ({
  threshold: 1.2, confirmation_points: 10, reference_horizon_minutes: 120,
  groups: duration.groups,
  events: {example: {groups: {major: {status: 'valid', curve}}}},
});

test('export is reproducible and source hash, cohort counts and distinct summary definitions remain exact', () => {
  const beforeRadar = readFileSync(new URL('./data/radar.json', import.meta.url));
  execFileSync('python3', [fileURLToPath(new URL('./build_duration_data.py', import.meta.url)), '--check']);
  assert.deepEqual(readFileSync(new URL('./data/radar.json', import.meta.url)), beforeRadar);
  assert.deepEqual(readFileSync(new URL('../research/shock_radar/seven_checks/result_3_5_7.json', import.meta.url)), sourceBytes);
  assert.equal(duration.source.sha256, createHash('sha256').update(sourceBytes).digest('hex'));
  assert.equal(duration.source.path, 'research/shock_radar/seven_checks/result_3_5_7.json');
  assert.deepEqual(tiers.map((tier) => duration.groups[tier].reference_minute), [46, 25]);
  assert.deepEqual(tiers.map((tier) => duration.groups[tier].event_median_minute), [36, 30]);
  assert.deepEqual(tiers.map((tier) => duration.groups[tier].unconfirmed_after_120), [19, 11]);
  assert.deepEqual(tiers.map((tier) => duration.groups[tier].label), ['大币组', '小币组']);
  for (const tier of tiers) {
    const group = duration.groups[tier], saved = decay.groups.combined[tier];
    assert.deepEqual([group.n, group.n_expected, group.n_missing], [139, 140, 1]);
    assert.deepEqual(group.median_curve, saved.median_curve);
    assert.equal(group.median_curve.length, 121);
    for (const month of ['2026-01', '2026-02']) {
      assert.equal(group.monthly_reference[month], decay.groups[month][tier].median_curve_confirmation_minutes);
    }
  }
});

test('only the three replay events and saved tier curves are exported, without future confirmation labels', () => {
  assert.deepEqual(Object.keys(duration.events).sort(), radar.events.map((event) => event.event_id).sort());
  for (const event of radar.events) {
    const actual = duration.events[event.event_id];
    const saved = decay.events.find((record) => record.event_id === event.event_id);
    assert.equal(actual.timestamp, event.timestamp);
    for (const tier of tiers) {
      assert.deepEqual(Object.keys(actual.groups[tier]).sort(), ['curve', 'status']);
      assert.deepEqual(actual.groups[tier].curve, saved.tiers[tier].curve);
      assert.equal(actual.groups[tier].status, saved.tiers[tier].status);
    }
  }
});

test('all ten assets map to exactly one known tier, unknown and ambiguous assignments return null', () => {
  for (const asset of ['BTC', 'ETH', 'SOL', 'BNB']) assert.equal(tierForAsset(duration, asset), 'major');
  for (const asset of ['XRP', 'ADA', 'DOGE', 'AVAX', 'LINK', 'LTC']) assert.equal(tierForAsset(duration, asset), 'small');
  for (const asset of ['', 'btc', 'UNKNOWN', null]) assert.equal(tierForAsset(duration, asset), null);
  assert.equal(tierForAsset(null, 'BTC'), null);
  const ambiguous = structuredClone(duration);
  ambiguous.groups.small.assets.push('BTC');
  assert.equal(tierForAsset(ambiguous, 'BTC'), null);
});

test('confirmation occurs at the tenth point including h=0, ends after rebound, and retains its prior history', () => {
  const data = fixture([...Array(10).fill(1.2), 1.200001, ...Array(10).fill(1)]);
  assert.deepEqual(durationStatus(data, 'example', 'major', 0), {
    available: true, minute: 0, ratio: 1.2, streak: 1, confirmed: false, hadConfirmed: false,
  });
  assert.equal(durationStatus(data, 'example', 'major', 8).confirmed, false);
  assert.deepEqual(durationStatus(data, 'example', 'major', 9), {
    available: true, minute: 9, ratio: 1.2, streak: 10, confirmed: true, hadConfirmed: true,
  });
  assert.deepEqual(durationStatus(data, 'example', 'major', 10), {
    available: true, minute: 10, ratio: 1.200001, streak: 0, confirmed: false, hadConfirmed: true,
  });
  assert.equal(durationStatus(data, 'example', 'major', 19).confirmed, false);
  assert.equal(durationStatus(data, 'example', 'major', 20).confirmed, true);
  assert.equal(durationStatus(data, 'example', 'major', 0).hadConfirmed, false); // Rewind has no memory leak.
});

test('future values and future confirmation metadata cannot change a revealed prefix', () => {
  const original = fixture(Array(121).fill(2));
  const changed = structuredClone(original);
  changed.events.example.groups.major.curve = [...Array(6).fill(2), ...Array(115).fill(1)];
  changed.events.example.groups.major.confirmation_minutes = 0;
  assert.deepEqual(durationStatus(changed, 'example', 'major', 5), durationStatus(original, 'example', 'major', 5));
  const curve = changed.events.example.groups.major.curve;
  Object.defineProperty(curve, 6, {get() {throw new Error('Unseen future was read');}});
  assert.equal(durationStatus(changed, 'example', 'major', 5).available, true);
  changed.events.example.groups.major.curve = Array(6).fill(2);
  assert.equal(durationStatus(changed, 'example', 'major', 5).available, true);
  assert.equal(durationStatus(changed, 'example', 'major', 6).available, false);
});

test('missing or invalid known values never become zero or a false confirmation', () => {
  for (const bad of [null, undefined, NaN, Infinity, -1, '1']) {
    const data = fixture([bad, ...Array(120).fill(1)]);
    assert.deepEqual(durationStatus(data, 'example', 'major', 30), {
      available: false, minute: 30, ratio: null, streak: null, confirmed: false, hadConfirmed: false,
    });
  }
  const afterConfirmation = fixture([...Array(10).fill(1), null]);
  assert.deepEqual(durationStatus(afterConfirmation, 'example', 'major', 10), {
    available: false, minute: 10, ratio: null, streak: null, confirmed: false, hadConfirmed: true,
  });
  const zero = durationStatus(fixture(Array(10).fill(0)), 'example', 'major', 9);
  assert.equal(zero.available, true); // A real numeric zero is distinct from missing.
  assert.equal(zero.ratio, 0);
  assert.equal(zero.confirmed, true);
});

test('unavailable event/tier/method and invalid minute boundaries are explicit', () => {
  assert.equal(durationStatus(duration, 'missing', 'major', 0).available, false);
  assert.equal(durationStatus(duration, Object.keys(duration.events)[0], 'unknown', 0).available, false);
  assert.equal(durationStatus(null, 'example', 'major', 0).available, false);
  const data = fixture(Array(121).fill(1));
  assert.equal(durationStatus(data, 'example', 'major', 120).streak, 121);
  data.events.example.groups.major.status = 'boundary_insufficient';
  assert.equal(durationStatus(data, 'example', 'major', 120).available, false);
  for (const minute of [-1, 121, .5, NaN, Infinity, '30', undefined]) {
    assert.throws(() => durationStatus(duration, 'missing', 'major', minute), /integer between 0 and 120/);
  }
});

test('real replay prefixes match the frozen first confirmation without consulting it, and never mutate data', () => {
  const before = structuredClone(duration);
  for (const [eventId] of Object.entries(duration.events)) {
    const saved = decay.events.find((event) => event.event_id === eventId);
    for (const tier of tiers) {
      let first = null;
      for (let minute = 0; minute <= 120; minute++) {
        const status = durationStatus(duration, eventId, tier, minute);
        assert.equal(status.available, true);
        if (status.confirmed && first === null) first = minute;
        assert.equal(status.hadConfirmed, first !== null);
      }
      assert.equal(first, saved.tiers[tier].confirmation_minutes);
    }
  }
  assert.deepEqual(duration, before);
});

test('the first event is not confirmed at the small-tier reference minute or the end of the price replay', () => {
  const eventId = radar.events[0].event_id;
  const atReference = durationStatus(duration, eventId, 'small', 25);
  assert.equal(atReference.ratio.toFixed(3), '1.258');
  assert.equal(atReference.streak, 0);
  assert.equal(atReference.confirmed, false);
  const atReplayEnd = durationStatus(duration, eventId, 'small', 30);
  assert.equal(atReplayEnd.ratio.toFixed(3), '1.132');
  assert.equal(atReplayEnd.streak, 5);
  assert.equal(atReplayEnd.confirmed, false);
});
