import test from 'node:test';
import assert from 'node:assert/strict';
import {chartSeries, renderLiveChart, renderMarketOverview} from './blocks/live-chart.mjs';

const assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'LTC'];
function fixture() {
  const as_of = '2026-09-27T12:00:00Z', origin = Date.parse(as_of);
  const snapshot = {
    window_id: 'test-window', as_of, assets: [...assets], quote_currency: 'USDT', source: 'Synthetic test source',
    series: Object.fromEntries(assets.map((asset, index) => [asset, Array.from({length: 61}, (_, minute) => ({time: new Date(origin - (60 - minute) * 60_000).toISOString(), close: 100 + index + minute / 10}))])),
    metrics: assets.map((asset, index) => ({asset, vol_past30: index / 100})),
  };
  const prediction = {status: 'ready', window_id: snapshot.window_id, as_of, path_count: 1, assets: Object.fromEntries(assets.map((asset, index) => [asset, {close: Array.from({length: 30}, (_, minute) => 106 + index + (minute + 1) / 10), mdd: 0}]))};
  return {snapshot, prediction};
}

test('observed and future timestamps are aligned, normalized to the same origin, and never mutate inputs', () => {
  const {snapshot, prediction} = fixture(), before = structuredClone({snapshot, prediction});
  const view = chartSeries(snapshot, 'BTC', prediction);
  assert.equal(view.observed.length, 61);
  assert.equal(view.forecast.length, 30);
  assert.equal(view.predictionState, 'ready');
  assert.equal(view.forecastMdd, 0);
  assert.equal(view.observed[0].time, '2026-09-27T11:00:00.000Z');
  assert.equal(view.observed.at(-1).time, '2026-09-27T12:00:00.000Z');
  assert.equal(view.observed.at(-1).value, 0);
  assert.equal(view.forecast[0].time, '2026-09-27T12:01:00.000Z');
  assert.equal(view.forecast.at(-1).time, '2026-09-27T12:30:00.000Z');
  assert.equal(view.forecast.at(-1).value, 109 / 106 - 1);
  assert.deepEqual({snapshot, prediction}, before);
  view.observed[0].close = 999;
  assert.equal(snapshot.series.BTC[0].close, 100);
});

test('a matching timestamp cannot rescue a different window, nor a matching window a different origin', () => {
  for (const mutate of [p => p.window_id = 'old', p => p.as_of = '2026-09-27T11:59:00Z', p => delete p.as_of]) {
    const {snapshot, prediction} = fixture(); mutate(prediction);
    const view = chartSeries(snapshot, 'BTC', prediction);
    assert.deepEqual(view.forecast, []);
    assert.equal(view.predictionState, 'stale');
  }
});

test('running/error/absent predictions never leak any supplied future prices', () => {
  const {snapshot, prediction} = fixture();
  assert.equal(chartSeries(snapshot, 'BTC').predictionState, 'unavailable');
  for (const status of ['running', 'error', 'idle']) {
    const view = chartSeries(snapshot, 'BTC', {...prediction, status});
    assert.equal(view.forecast.length, 0);
    assert.equal(view.predictionState, status === 'idle' ? 'unavailable' : status);
  }
});

test('bad prediction paths are withheld rather than interpolated or converted into zeros', () => {
  for (const mutate of [p => p.assets.BTC.close.pop(), p => p.assets.BTC.close.push(100), p => delete p.assets.BTC, ...[0, -1, NaN, Infinity, null, '100'].map(value => p => p.assets.BTC.close[3] = value)]) {
    const {snapshot, prediction} = fixture(); mutate(prediction);
    const view = chartSeries(snapshot, 'BTC', prediction);
    assert.equal(view.predictionState, 'invalid');
    assert.equal(view.forecast.length, 0);
  }
});

test('missing, duplicate, wrong-origin or invalid observed closes fail visibly', () => {
  const mutators = [s => s.series.BTC.pop(), s => s.series.BTC.push(s.series.BTC.at(-1)), s => s.series.BTC[4].time = s.series.BTC[3].time, s => s.as_of = '2026-09-27T12:01:00Z', s => s.window_id = '', s => s.assets = ['ETH'], ...[0, -1, NaN, Infinity, null, '100'].map(value => s => s.series.BTC[10].close = value)];
  for (const mutate of mutators) {
    const {snapshot} = fixture(); mutate(snapshot);
    assert.throws(() => chartSeries(snapshot, 'BTC'), TypeError);
  }
});

// Minimal synthetic DOM verifies rendered paths and callbacks without browser/network side effects.
class Element {
  constructor(tag) { this.tagName = tag; this.attributes = {}; this.children = []; this.style = {}; this.events = {}; }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.children.push(node); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(type, callback) { this.events[type] = callback; }
}
function dom() {
  const original = globalThis.document;
  globalThis.document = {createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag)};
  return () => { globalThis.document = original; };
}
const allElements = node => [node, ...node.children.flatMap(allElements)];

test('rendering uses bounded axes even for flat/large series and withholds mixed-window SVG forecast', () => {
  const restore = dom();
  try {
    for (const factor of [0, 1, 1e12]) {
      const {snapshot, prediction} = fixture();
      snapshot.series.BTC.forEach((point, index) => { point.close = factor ? 100 + index * factor : 100; });
      const host = new Element('div');
      renderLiveChart(host, snapshot, 'BTC', prediction);
      const elements = allElements(host);
      assert.equal(elements.filter(el => el.attributes['data-series'] === 'observed').length, 1);
      assert.equal(elements.filter(el => el.attributes['data-series'] === 'forecast').length, 1);
      assert.ok(elements.length < 40, 'Axis tick count is constant, not magnitude-dependent.');
      assert.ok(elements.every(el => Object.values(el.attributes).every(value => !/NaN|Infinity/.test(value))));
      renderLiveChart(host, snapshot, 'BTC', {...prediction, window_id: 'old'});
      assert.equal(allElements(host).filter(el => el.attributes['data-series'] === 'forecast').length, 0);
    }
  } finally { restore(); }
});

test('overview contains all ten real series, only past-volatility top three are highlighted, clicks preserve asset', () => {
  const restore = dom();
  try {
    const {snapshot} = fixture(), host = new Element('div'), clicks = [];
    const saved = structuredClone(snapshot);
    renderMarketOverview(host, snapshot, asset => clicks.push(asset));
    const buttons = allElements(host).filter(el => el.tagName === 'button');
    assert.deepEqual(buttons.map(el => el.attributes['data-asset']), assets);
    assert.deepEqual(buttons.filter(el => el.className.includes('priority')).map(el => el.attributes['data-asset']), ['AVAX', 'LINK', 'LTC']);
    assert.equal(allElements(host).filter(el => el.attributes['data-point-count'] === '61').length, 10);
    buttons[4].events.click();
    assert.deepEqual(clicks, ['XRP']);
  assert.deepEqual(snapshot, saved);
    snapshot.metrics.pop();
    assert.throws(() => renderMarketOverview(host, snapshot, () => {}), /same ten assets/);
  } finally { restore(); }
});

test('all ten small charts use their matching forecasts, one shared scale including future values, and retain source data', () => {
  const restore = dom();
  try {
    const {snapshot, prediction} = fixture(), host = new Element('div');
    prediction.assets.BTC.close[29] = 150;
    const saved = structuredClone({snapshot, prediction});
    renderMarketOverview(host, snapshot, () => {}, prediction);
    const elements = allElements(host), charts = elements.filter(el => el.tagName === 'svg');
    assert.equal(charts.length, 10);
    assert.equal(elements.filter(el => el.attributes['data-series'] === 'observed').length, 10);
    const forecasts = elements.filter(el => el.attributes['data-series'] === 'forecast');
    assert.equal(forecasts.length, 10);
    assert.ok(forecasts.every(el => el.attributes['data-point-count'] === '30' && el.attributes['stroke-dasharray']));
    assert.equal(new Set(charts.map(el => el.attributes['data-y-min'])).size, 1);
    assert.equal(new Set(charts.map(el => el.attributes['data-y-max'])).size, 1);
    assert.ok(Number(charts[0].attributes['data-y-max']) > 150 / 106 - 1, 'Future prices must fit in the common scale.');
    assert.equal(elements.filter(el => 'data-origin-marker' in el.attributes).length, 10);
    assert.equal(elements.filter(el => el.className === 'live-market-forecast' && el.textContent === '预测回撤 0.00%').length, 10);
    assert.ok(charts.every(el => el.attributes['data-window-id'] === snapshot.window_id && el.attributes['data-prediction-state'] === 'ready'));
    assert.deepEqual({snapshot, prediction}, saved);
    // A selected chart expands the same validated prices, not another sampling.
    for (const asset of assets) {
      const expanded = new Element('div');
      const view = renderLiveChart(expanded, snapshot, asset, prediction);
      assert.deepEqual(view.forecast.map(point => point.close), prediction.assets[asset].close);
      assert.equal(allElements(expanded).filter(el => el.attributes['data-series'] === 'forecast').length, 1);
    }
  } finally { restore(); }
});

test('small charts never overlay stale, failed, pending, or invalid forecasts and do not let rejected values stretch the scale', () => {
  const restore = dom();
  try {
    for (const mutate of [p => p.window_id = 'previous', p => p.as_of = '2026-09-27T11:59:00Z', p => p.status = 'running', p => p.status = 'error']) {
      const {snapshot, prediction} = fixture(), host = new Element('div');
      prediction.assets.BTC.close[29] = 1e12;
      mutate(prediction);
      renderMarketOverview(host, snapshot, null, prediction);
      const elements = allElements(host);
      assert.equal(elements.filter(el => el.attributes['data-series'] === 'forecast').length, 0);
      assert.ok(Number(elements.find(el => el.tagName === 'svg').attributes['data-y-max']) < 0.1);
      assert.ok(elements.filter(el => el.tagName === 'button').every(el => !el.attributes['aria-label'].includes('预测回撤 0.00%')));
    }
    const {snapshot, prediction} = fixture(), host = new Element('div');
    prediction.assets.BTC.close[0] = NaN;
    prediction.assets.BTC.close[29] = 1e12;
    renderMarketOverview(host, snapshot, null, prediction);
    const cards = allElements(host).filter(el => el.tagName === 'button');
    assert.equal(allElements(cards[0]).filter(el => el.attributes['data-series'] === 'forecast').length, 0);
    assert.ok(cards[0].attributes['aria-label'].includes('预测不可用'));
    assert.equal(allElements(host).filter(el => el.attributes['data-series'] === 'forecast').length, 9);
    assert.ok(Number(allElements(cards[0]).find(el => el.tagName === 'svg').attributes['data-y-max']) < 0.1);
  } finally { restore(); }
});

test('missing or illegal predicted drawdown is unavailable, never shown as zero or a confidence interval', () => {
  const restore = dom();
  try {
    for (const mdd of [undefined, null, NaN, Infinity, -0.1, 1, '0.05']) {
      const {snapshot, prediction} = fixture(), host = new Element('div');
      prediction.assets.BTC.mdd = mdd;
      assert.equal(chartSeries(snapshot, 'BTC', prediction).forecastMdd, null);
      renderMarketOverview(host, snapshot, null, prediction);
      const card = allElements(host).find(el => el.attributes['data-asset'] === 'BTC');
      assert.ok(card.attributes['aria-label'].includes('预测回撤不可用'));
      assert.equal(allElements(card).filter(el => el.attributes['data-series'] === 'forecast').length, 1, 'Valid prices remain visible when the unused metric is missing.');
      assert.ok(allElements(card).some(el => el.textContent === '预测回撤不可用'));
    }
  } finally { restore(); }
});
