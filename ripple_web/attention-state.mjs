import {revealedDrawdown} from './replay-state.mjs';

const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const isAsset = (value) => typeof value === 'string' && /^[A-Z][A-Z0-9]*$/.test(value);
const lexical = (left, right) => left < right ? -1 : left > right ? 1 : 0;

// Freeze attention from past volatility only. Source records are never sorted or edited.
export function partitionAttention(radarEvent, limit = 3) {
  if (!isRecord(radarEvent) || !Array.isArray(radarEvent.assets)) {
    throw new TypeError('Attention requires an event with an assets array.');
  }
  if (!Number.isInteger(limit) || limit < 1 || limit >= radarEvent.assets.length) {
    throw new RangeError('Attention limit must leave both focused and deferred assets.');
  }
  const seen = new Set();
  const ranked = radarEvent.assets.map((record) => {
    if (!isRecord(record) || !isAsset(record.asset) || seen.has(record.asset)) {
      throw new TypeError('Attention assets must be valid, unique symbols.');
    }
    if (!Number.isFinite(record.vol_past30) || record.vol_past30 < 0) {
      throw new TypeError(`Invalid past volatility for ${record.asset}.`);
    }
    seen.add(record.asset);
    return {...record};
  }).sort((left, right) => right.vol_past30 - left.vol_past30 || lexical(left.asset, right.asset));
  ranked.forEach((record, index) => { record.rank = index + 1; });
  return {
    ranked,
    focused: ranked.slice(0, limit),
    deferred: ranked.slice(limit),
    cutoffTied: ranked[limit - 1].vol_past30 === ranked[limit].vol_past30,
  };
}

function validatePair(radarEvent, replayEvent, ranked, minute) {
  if (!isRecord(replayEvent) || typeof radarEvent.event_id !== 'string' || !radarEvent.event_id.trim()
      || radarEvent.event_id !== replayEvent.event_id
      || typeof radarEvent.timestamp !== 'string' || !Number.isFinite(Date.parse(radarEvent.timestamp))
      || radarEvent.timestamp !== replayEvent.timestamp) {
    throw new TypeError('Attention and replay must describe the same dated event.');
  }
  if (!isRecord(replayEvent.assets) || Object.keys(replayEvent.assets).length !== ranked.length) {
    throw new TypeError('Replay must contain exactly the attention asset set.');
  }
  for (const {asset} of ranked) {
    if (!Object.hasOwn(replayEvent.assets, asset)) {
      throw new TypeError(`Missing replay for ${asset}.`);
    }
    const replay = replayEvent.assets[asset];
    if (!isRecord(replay) || !Number.isFinite(replay.spot) || replay.spot <= 0
        || !Array.isArray(replay.observed) || replay.observed.length < 31 + minute
        || replay.observed[30] !== replay.spot) {
      throw new TypeError(`Incomplete or inconsistent replay for ${asset} at +${minute}m.`);
    }
    // Validate only the revealed prefix. Unseen future gaps must not affect an earlier view.
    for (let index = 31; index < 31 + minute; index++) {
      if (!Number.isFinite(replay.observed[index]) || replay.observed[index] <= 0) {
        throw new TypeError(`Missing observed price for ${asset} at +${index - 30}m.`);
      }
    }
  }
}

function maximumRevealed(records, replayEvent, minute) {
  let maximum;
  for (const {asset} of records) {
    const value = revealedDrawdown(replayEvent.assets[asset], minute);
    if (!Number.isFinite(value)) throw new TypeError(`Missing revealed drawdown for ${asset}.`);
    if (!maximum || value > maximum.value || (value === maximum.value && lexical(asset, maximum.asset) < 0)) {
      maximum = {asset, value};
    }
  }
  return maximum;
}

// Actual outcomes are computed from the visible prices, never from full-horizon actual_mdd.
// Both maximum values are fractions (0.01 = 1%), not percentages or basis points.
export function summarizeRevealed(radarEvent, replayEvent, minute, limit = 3) {
  if (!Number.isInteger(minute) || minute < 0 || minute > 30) {
    throw new RangeError('Replay minute must be an integer between 0 and 30.');
  }
  const partition = partitionAttention(radarEvent, limit);
  validatePair(radarEvent, replayEvent, partition.ranked, minute);
  if (minute === 0) return null;
  return {
    minute,
    focusedMax: maximumRevealed(partition.focused, replayEvent, minute),
    deferredMax: maximumRevealed(partition.deferred, replayEvent, minute),
  };
}
