// Displays current per-asset state from the same market snapshot; no fetching or inference.
import {el} from '../format.mjs?v=en-r1';
import {tr} from '../i18n.mjs?v=en-r1';

const THRESHOLD = 1.2;
const finiteNonnegative = value => Number.isFinite(value) && value >= 0;

function validState(record) {
  const value = record?.volatility_state;
  if (!value || !finiteNonnegative(value.ratio)
      || !finiteNonnegative(value.recent_std)
      || !Number.isFinite(value.baseline_std) || value.baseline_std <= 0
      || value.reason != null) return null;
  const expected = value.ratio >= THRESHOLD ? 'above_usual' : 'near_usual';
  return value.state === expected ? value : null;
}

function ratioText(ratio) {
  const rounded = ratio.toFixed(2);
  // A rounded value must not put a gray card on the highlighted side of the boundary.
  return ratio < THRESHOLD && rounded === '1.20' ? '<1.20×' : `${rounded}×`;
}

export function renderVolatilityState(host, snapshot, selectedAsset, onSelectAsset, {stale = false} = {}) {
  host.replaceChildren();
  host.append(el('h3', null, tr('当前波动状态', 'Current volatility')));
  host.append(el('p', 'minor', tr('近 10 分钟波动 ÷ 近 120 分钟基线', 'Past 10 min volatility ÷ past 120 min baseline')));

  const invalidFeed = stale || Boolean(snapshot?.stale) || Boolean(snapshot?.error)
    || snapshot?.status === 'error' || snapshot?.status === 'stale';
  const assets = Array.isArray(snapshot?.assets) ? snapshot.assets.filter(asset => typeof asset === 'string' && asset.trim()) : [];
  if (!assets.length) {
    host.append(el('p', 'minor', tr('等待行情', 'Awaiting market data')));
    return;
  }
  const records = new Map(), duplicates = new Set();
  for (const record of Array.isArray(snapshot?.metrics) ? snapshot.metrics : []) {
    if (records.has(record?.asset)) duplicates.add(record.asset);
    records.set(record?.asset, record);
  }
  const grid = el('div', 'volatility-state-grid');
  for (const asset of assets) {
    const value = invalidFeed || duplicates.has(asset) ? null : validState(records.get(asset));
    const state = value?.state || 'unavailable';
    const selected = asset === selectedAsset;
    const label = state === 'above_usual' ? tr('高于平时', 'Above usual')
      : state === 'near_usual' ? tr('接近平时', 'Near usual')
      : invalidFeed ? tr('等待更新', 'Awaiting update') : tr('暂不可用', 'Unavailable');
    const ratio = value ? ratioText(value.ratio) : '—';
    const button = el('button', `volatility-state-asset ${state.replaceAll('_', '-')}${selected ? ' selected' : ''}`);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(selected));
    button.setAttribute('aria-label', `${asset} · ${ratio} · ${label}`);
    button.setAttribute('data-asset', asset);
    button.append(el('span', 'volatility-state-symbol', asset), el('strong', 'volatility-state-ratio', ratio), el('span', 'volatility-state-label', label));
    button.addEventListener('click', () => onSelectAsset?.(asset));
    grid.append(button);
  }
  host.append(grid);
}
