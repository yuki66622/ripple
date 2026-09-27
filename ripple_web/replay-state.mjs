// Arithmetic on an already frozen replay. Never samples, fetches or uses hidden future observations.
export function clampMinute(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(30, Math.round(number))) : 0;
}

export function visibleObservations(asset, minute) {
  return asset.observed.slice(0, 31 + clampMinute(minute));
}

export function revealedDrawdown(asset, minute) {
  const n = clampMinute(minute);
  if (n === 0) return null;
  const prices = [asset.spot, ...asset.observed.slice(31, 31 + n)];
  if (prices.length !== n + 1 || prices.some((v) => !Number.isFinite(v) || v <= 0)) return null;
  let peak = prices[0], result = 0;
  for (const price of prices) {
    peak = Math.max(peak, price);
    result = Math.max(result, (peak - price) / peak);
  }
  return result;
}

export function viewSeries(asset, minute) {
  return {
    minute: clampMinute(minute),
    observed: visibleObservations(asset, minute).map((v) => v / asset.spot - 1),
    lower: asset.p05.map((v) => v / asset.spot - 1),
    median: asset.p50.map((v) => v / asset.spot - 1),
    upper: asset.p95.map((v) => v / asset.spot - 1),
    actualMdd: revealedDrawdown(asset, minute),
  };
}
