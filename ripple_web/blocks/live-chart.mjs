// Live charts accept already-validated snapshots; they never fetch or invoke a model.
import {el, svg, label, pct, spct, showTip, hideTip} from '../format.mjs?v=en-r1';
import {partitionAttention} from '../attention-state.mjs';
import {tr} from '../i18n.mjs?v=en-r1';

const MINUTE = 60_000;
const positive = (value) => Number.isFinite(value) && value > 0;
const utc = (value) => new Date(value).toISOString().slice(11, 16);
const price = (value) => new Intl.NumberFormat('en-US', {maximumSignificantDigits: 7}).format(value);

function sourceText(source) {
  if (typeof source === 'string' && source.trim()) return source;
  if (source && typeof source === 'object') {
    return [source.exchange, source.market, source.name].filter((v) => typeof v === 'string' && v.trim()).join(' · ') || tr('来源未注明', 'Source unspecified');
  }
  return tr('来源未注明', 'Source unspecified');
}

/** Return fresh arrays, normalized to this snapshot's latest close, without any data repair. */
export function chartSeries(snapshot, asset, prediction = null) {
  if (!snapshot || typeof snapshot.window_id !== 'string' || !snapshot.window_id.trim()
      || !Array.isArray(snapshot.assets) || !snapshot.assets.includes(asset)
      || typeof snapshot.as_of !== 'string' || !snapshot.as_of.endsWith('Z')
      || !Number.isFinite(Date.parse(snapshot.as_of))) {
    throw new TypeError('Live chart requires a dated window and an available asset.');
  }
  const origin = Date.parse(snapshot.as_of), records = snapshot.series?.[asset];
  if (!Array.isArray(records) || records.length !== 61) throw new TypeError(`Expected 61 observed closes for ${asset}.`);
  const spot = records.at(-1)?.close;
  if (!positive(spot)) throw new TypeError(`Invalid latest close for ${asset}.`);
  const observed = records.map((record, index) => {
    const expected = origin - (60 - index) * MINUTE;
    if (!record || typeof record.time !== 'string' || !record.time.endsWith('Z')
        || Date.parse(record.time) !== expected || !positive(record.close)) {
      throw new TypeError(`Incomplete or invalid observed minute for ${asset} at ${index}.`);
    }
    const value = record.close / spot - 1;
    if (!Number.isFinite(value)) throw new TypeError(`Non-finite normalized price for ${asset}.`);
    return {time: new Date(expected).toISOString(), close: record.close, value};
  });
  let predictionState = 'unavailable', forecast = [], forecastMdd = null;
  if (prediction) {
    if (prediction.window_id !== snapshot.window_id
        || Date.parse(prediction.as_of) !== origin) predictionState = 'stale';
    else if (prediction.status === 'running') predictionState = 'running';
    else if (prediction.status === 'error') predictionState = 'error';
    else if (prediction.status === 'ready') {
      const closes = prediction.assets?.[asset]?.close;
      if (!Array.isArray(closes) || closes.length !== 30
          || !closes.every((value) => positive(value) && Number.isFinite(value / spot - 1))) {
        predictionState = 'invalid';
      } else {
        predictionState = 'ready';
        forecast = closes.map((close, index) => ({time: new Date(origin + (index + 1) * MINUTE).toISOString(), close, value: close / spot - 1}));
        const mdd = prediction.assets[asset].mdd;
        if (Number.isFinite(mdd) && mdd >= 0 && mdd < 1) forecastMdd = mdd;
      }
    }
  }
  return {asset, windowId: snapshot.window_id, asOf: new Date(origin).toISOString(), quote: snapshot.quote_currency || tr('报价单位未注明', 'Quote currency unspecified'), source: sourceText(snapshot.source), spot, observed, forecast, predictionState, forecastMdd};
}

// A fixed tick count stays bounded for flat, sub-cent, and extreme-valued series.
function extent(values) {
  const low = Math.min(...values, 0), high = Math.max(...values, 0);
  const padding = Math.max((high - low) * .09, .0001);
  return [low - padding, high + padding];
}

function path(points, x, y) {
  return points.map((point, index) => `${index ? 'L' : 'M'}${x(index).toFixed(2)} ${y(point.value).toFixed(2)}`).join(' ');
}

const forecastNotes = {
  unavailable: tr('等待此窗口预测。', 'Waiting for this window’s forecast.'),
  stale: tr('已有预测属于其他窗口，未叠加。', 'Forecast from another window is not shown.'),
  running: tr('此窗口预测中。', 'Forecasting this window.'),
  error: tr('此窗口预测失败，未叠加。', 'Forecast failed for this window.'),
  invalid: tr('预测价格未通过校验，未叠加。', 'Forecast prices failed validation and are not shown.'),
  ready: tr('未来 30 分钟为 Kronos 单路径预测，无置信区间。', 'Next 30 min: one Kronos forecast path, without a confidence interval.'),
};

function predictionLabel(view) {
  if (view.predictionState === 'ready') return view.forecastMdd === null ? tr('预测回撤不可用', 'Drawdown unavailable') : tr(`预测回撤 ${pct(view.forecastMdd)}`, `Forecast drawdown ${pct(view.forecastMdd)}`);
  return {unavailable: tr('等待预测', 'Awaiting forecast'), stale: tr('新窗口待预测', 'New window pending'), running: tr('预测中', 'Forecasting'), error: tr('预测失败', 'Forecast failed'), invalid: tr('预测不可用', 'Forecast unavailable')}[view.predictionState];
}

function provenance(view) {
  return `${view.asset} · ${view.asOf.replace('T', ' ').replace('.000Z', ' UTC')} · ${view.source} · ${view.quote} · ${forecastNotes[view.predictionState]}`;
}

export function renderLiveChart(host, snapshot, asset, prediction = null) {
  const view = chartSeries(snapshot, asset, prediction);
  const W = 920, H = 290, m = {l: 76, r: 26, t: 24, b: 42};
  const horizon = 90;
  const values = [...view.observed, ...view.forecast].map((point) => point.value);
  const [low, high] = extent(values);
  const x = (index) => m.l + index / horizon * (W - m.l - m.r);
  const y = (value) => m.t + (high - value) / (high - low) * (H - m.t - m.b);
  host.replaceChildren();
  const heading = el('div', 'live-chart-heading');
  heading.style.cssText = 'display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap';
  heading.setAttribute('title', provenance(view));
  heading.setAttribute('aria-label', provenance(view));
  heading.append(el('strong', null, `${price(view.spot)} ${view.quote}`), el('span', 'minor', tr(`预测起点 ${utc(view.asOf)} UTC`, `Forecast starts ${utc(view.asOf)} UTC`)));
  host.append(heading);
  const root = svg('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, role: 'img', 'aria-label': tr(`${asset} 过去 60 分钟价格，相对最新价的变化百分比。${provenance(view)}`, `${asset}, past 60 min prices as percentage changes relative to the latest close. ${provenance(view)}`), 'data-window-id': view.windowId, 'data-prediction-state': view.predictionState}, host);
  root.style.cssText = 'display:block;width:100%;height:auto;min-height:230px';
  for (let index = 0; index <= 4; index++) {
    const value = low + (high - low) * index / 4;
    svg('line', {x1: m.l, x2: W - m.r, y1: y(value), y2: y(value), stroke: '#1D3045', 'stroke-opacity': '.10'}, root);
    label(root, m.l - 12, y(value) + 4, spct(value, Math.abs(high - low) < .01 ? 2 : 1), {'text-anchor': 'end', fill: '#4E6275', 'font-size': 11});
  }
  svg('line', {x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: '#1D3045', 'stroke-opacity': '.25'}, root);
  label(root, m.l, 12, tr('价格变化 · 相对最新价', 'Price change · relative to latest close'), {fill: '#4E6275', 'font-size': 10});
  const ticks = [0, 30, 60, 90];
  for (const index of ticks) {
    const time = Date.parse(view.asOf) + (index - 60) * MINUTE;
    label(root, x(index), H - 17, `${utc(time)}${index === 60 ? tr(' 起点', ' start') : ''}`, {'text-anchor': index === 0 ? 'start' : index === horizon ? 'end' : 'middle', fill: '#4E6275', 'font-size': 11});
  }
  label(root, W - m.r, H - 2, 'UTC', {'text-anchor': 'end', fill: '#4E6275', 'font-size': 9});
  svg('line', {x1: x(60), x2: x(60), y1: m.t, y2: H - m.b, stroke: '#35699A', 'stroke-opacity': '.45', 'stroke-dasharray': '3 4', 'data-origin-marker': ''}, root);
  svg('path', {d: path(view.observed, x, y), fill: 'none', stroke: '#1D3045', 'stroke-width': '2', 'stroke-linejoin': 'round', 'data-series': 'observed', 'data-point-count': 61}, root);
  if (view.forecast.length) {
    svg('path', {d: path([view.observed.at(-1), ...view.forecast], (index) => x(60 + index), y), fill: 'none', stroke: '#35699A', 'stroke-width': '2', 'stroke-dasharray': '5 4', 'data-series': 'forecast', 'data-point-count': 30}, root);
  }
  svg('circle', {cx: x(60), cy: y(0), r: 3, fill: '#35699A'}, root);
  const cross = svg('line', {y1: m.t, y2: H - m.b, stroke: '#1D3045', opacity: 0, 'stroke-opacity': '.35'}, root);
  const hit = svg('rect', {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent'}, root);
  const all = [...view.observed, ...view.forecast];
  hit.addEventListener('pointermove', (event) => {
    const bounds = root.getBoundingClientRect();
    const index = Math.max(0, Math.min(horizon, Math.round(((event.clientX - bounds.left) * W / bounds.width - m.l) / (W - m.l - m.r) * horizon)));
    const point = all[index];
    if (!point) { cross.setAttribute('opacity', 0); hideTip(); return; }
    cross.setAttribute('x1', x(index)); cross.setAttribute('x2', x(index)); cross.setAttribute('opacity', 1);
    showTip(event.clientX, event.clientY, `${asset} · ${utc(point.time)} UTC`, [[index <= 60 ? tr('实际价格', 'Actual price') : tr('单路径预测', 'Single-path forecast'), `${price(point.close)} ${view.quote}`], [tr('相对最新价', 'Relative to latest close'), spct(point.value)], [tr('数据来源', 'Source'), view.source]]);
  });
  hit.addEventListener('pointerleave', () => { cross.setAttribute('opacity', 0); hideTip(); });
  return view;
}

export function renderMarketOverview(host, snapshot, onSelect, prediction = null) {
  const views = snapshot.assets.map((asset) => chartSeries(snapshot, asset, prediction));
  const partition = partitionAttention({assets: snapshot.metrics});
  if (views.length !== 10 || partition.ranked.length !== 10
      || new Set(views.map((view) => view.asset)).size !== 10
      || partition.ranked.some((record) => !snapshot.assets.includes(record.asset))) {
    throw new TypeError('Market overview requires the same ten assets in series and metrics.');
  }
  const top = new Set(partition.focused.map((record) => record.asset));
  const metrics = new Map(partition.ranked.map((record) => [record.asset, record]));
  const [low, high] = extent(views.flatMap((view) => [...view.observed, ...view.forecast].map((point) => point.value)));
  host.replaceChildren();
  const grid = el('div', 'live-market-grid');
  grid.style.cssText = 'display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:14px';
  for (const view of views) {
    const card = el('button', `live-market-asset${top.has(view.asset) ? ' priority' : ''}`);
    card.type = 'button';
    card.style.cssText = `padding:14px 12px;text-align:left;border:1px solid ${top.has(view.asset) ? '#35699A' : '#BAC7D0'};border-radius:3px;background:${top.has(view.asset) ? '#DFE9F0' : '#F0F4F7'};color:#1D3045;cursor:pointer;min-width:0`;
    card.setAttribute('aria-label', tr(`${view.asset}，过去 30 分钟波动 ${pct(metrics.get(view.asset).vol_past30)}，第 ${metrics.get(view.asset).rank}，${predictionLabel(view)}，点击放大。${forecastNotes[view.predictionState]}`, `${view.asset}, past 30 min volatility ${pct(metrics.get(view.asset).vol_past30)}, rank ${metrics.get(view.asset).rank}, ${predictionLabel(view)}. Click to enlarge. ${forecastNotes[view.predictionState]}`));
    card.setAttribute('data-asset', view.asset);
    card.setAttribute('title', provenance(view));
    const title = el('div', 'live-market-title');
    title.style.cssText = 'display:flex;justify-content:space-between;gap:6px;align-items:baseline;font-size:13px';
    title.append(el('strong', null, view.asset), el('span', null, `#${metrics.get(view.asset).rank}`));
    card.append(title);
    const root = svg('svg', {viewBox: '0 0 170 66', width: '100%', height: 66, role: 'img', 'aria-label': tr(`${view.asset}，过去 60 分钟行情及未来 30 分钟预测，同一百分比刻度。${forecastNotes[view.predictionState]}`, `${view.asset}, past 60 min prices and next 30 min forecast on the same percentage scale. ${forecastNotes[view.predictionState]}`), 'data-window-id': view.windowId, 'data-prediction-state': view.predictionState, 'data-y-min': low, 'data-y-max': high}, card);
    root.style.cssText = 'display:block;width:100%;height:66px';
    const x = (index) => 3 + index / 90 * 164, y = (value) => 5 + (high - value) / (high - low) * 56;
    svg('line', {x1: 3, x2: 167, y1: y(0), y2: y(0), stroke: '#8AA1B5', 'stroke-opacity': '.35'}, root);
    svg('line', {x1: x(60), x2: x(60), y1: 5, y2: 61, stroke: '#8AA1B5', 'stroke-opacity': '.55', 'stroke-dasharray': '2 3', 'data-origin-marker': ''}, root);
    svg('path', {d: path(view.observed, x, y), fill: 'none', stroke: top.has(view.asset) ? '#1D3045' : '#657C91', 'stroke-width': '1.8', 'data-series': 'observed', 'data-point-count': 61}, root);
    if (view.forecast.length) {
      svg('path', {d: path([view.observed.at(-1), ...view.forecast], (index) => x(60 + index), y), fill: 'none', stroke: '#35699A', 'stroke-width': '1.8', 'stroke-dasharray': '4 3', 'data-series': 'forecast', 'data-point-count': 30}, root);
    }
    svg('circle', {cx: x(60), cy: y(0), r: 2, fill: '#35699A'}, root);
    const detail = el('div', null, tr(`30 分钟波动 ${pct(metrics.get(view.asset).vol_past30)}`, `30 min volatility ${pct(metrics.get(view.asset).vol_past30)}`));
    detail.style.cssText = 'font-size:11px;color:#4E6275';
    card.append(detail);
    const forecastDetail = el('div', 'live-market-forecast', predictionLabel(view));
    forecastDetail.style.cssText = 'font-size:11px;color:#35699A;margin-top:3px';
    forecastDetail.setAttribute('data-prediction-state', view.predictionState);
    card.append(forecastDetail);
    card.addEventListener('click', () => { if (typeof onSelect === 'function') onSelect(view.asset); });
    grid.append(card);
  }
  host.append(grid);
}
