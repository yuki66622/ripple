import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {renderVolatilityState} from './blocks/volatility-state.mjs';

class Element {
  constructor(tag) { this.tagName = tag; this.attributes = {}; this.children = []; this.events = {}; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(type, callback) { this.events[type] = callback; }
}
const elements = node => [node, ...node.children.flatMap(elements)];
const buttons = host => elements(host).filter(node => node.tagName === 'button');
const ratio = button => button.children.find(node => node.className === 'volatility-state-ratio').textContent;
const label = button => button.children.find(node => node.className === 'volatility-state-label').textContent;
const assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'LTC'];
function metric(asset, ratioValue) {
  return {asset, volatility_state: {ratio: ratioValue, state: ratioValue >= 1.2 ? 'above_usual' : 'near_usual', recent_std: ratioValue * .01, baseline_std: .01, reason: null}};
}
function snapshot() { return {assets: [...assets], metrics: assets.map((asset, index) => metric(asset, .7 + index / 10))}; }
function withDom(callback) {
  const original = globalThis.document;
  globalThis.document = {createElement: tag => new Element(tag)};
  try { callback(); } finally { globalThis.document = original; }
}

test('each asset uses its own full-precision ratio and preserves snapshot order', () => withDom(() => {
  const source = snapshot();
  source.metrics = [metric('ETH', 1.1999), metric('SOL', 0), metric('BTC', 1.2), ...source.metrics.slice(3)];
  const host = new Element('div');
  renderVolatilityState(host, source, 'ETH', () => {});
  const cards = buttons(host);
  assert.deepEqual(cards.map(card => card.attributes['data-asset']), assets);
  assert.match(cards[0].className, /above-usual/);
  assert.equal(ratio(cards[0]), '1.20×');
  assert.equal(label(cards[0]), '高于平时');
  assert.match(cards[1].className, /near-usual/);
  assert.equal(ratio(cards[1]), '<1.20×');
  assert.equal(label(cards[1]), '接近平时');
  assert.equal(ratio(cards[2]), '0.00×', 'A measured zero with positive baseline remains valid.');
  assert.equal(cards[1].attributes['aria-pressed'], 'true');
  assert.equal(cards.filter(card => card.attributes['aria-pressed'] === 'true').length, 1);
}));

test('only below-threshold values that round across the boundary use the less-than label', () => withDom(() => {
  for (const [value, expected] of [[1.194, '1.19×'], [1.199999, '<1.20×'], [1.2, '1.20×'], [1.200001, '1.20×'], [2, '2.00×']]) {
    const source = snapshot(), host = new Element('div');
    source.metrics[0] = metric('BTC', value);
    renderVolatilityState(host, source, 'BTC');
    assert.equal(ratio(buttons(host)[0]), expected);
  }
}));

test('stale and errored snapshots never expose old ratios or current classifications', () => withDom(() => {
  for (const flags of [{stale: true}, {error: 'feed disconnected'}, {status: 'error'}, {status: 'stale'}]) {
    const host = new Element('div');
    renderVolatilityState(host, {...snapshot(), ...flags}, 'BTC');
    assert.equal(buttons(host).length, 10);
    for (const card of buttons(host)) {
      assert.match(card.className, /unavailable/);
      assert.equal(ratio(card), '—');
      assert.equal(label(card), '等待更新');
    }
  }
  const host = new Element('div');
  renderVolatilityState(host, snapshot(), 'BTC', null, {stale: true});
  assert.ok(buttons(host).every(card => ratio(card) === '—'));
}));

test('missing, malformed, zero-baseline, duplicate and mismatched records stay unavailable locally', () => withDom(() => {
  const mutators = [
    s => delete s.metrics[0].volatility_state,
    s => s.metrics.shift(),
    s => s.metrics.push(metric('BTC', 2)),
    ...[null, -1, NaN, Infinity, '1.3'].map(value => s => s.metrics[0].volatility_state.ratio = value),
    ...[0, -1, null, NaN].map(value => s => s.metrics[0].volatility_state.baseline_std = value),
    s => s.metrics[0].volatility_state.recent_std = -1,
    s => s.metrics[0].volatility_state.state = 'above_usual',
    s => s.metrics[0].volatility_state.reason = 'zero_baseline',
  ];
  for (const mutate of mutators) {
    const source = snapshot(), host = new Element('div'); mutate(source);
    renderVolatilityState(host, source, 'BTC');
    assert.equal(ratio(buttons(host)[0]), '—');
    assert.equal(label(buttons(host)[0]), '暂不可用');
    assert.equal(ratio(buttons(host)[1]), '0.80×', 'A broken asset must not suppress another asset.');
  }
}));

test('selection callback passes the asset without mutating or reordering input data', () => withDom(() => {
  const source = snapshot(), before = structuredClone(source), host = new Element('div'), clicks = [];
  renderVolatilityState(host, source, 'BTC', asset => clicks.push(asset));
  buttons(host)[7].events.click();
  assert.deepEqual(clicks, ['AVAX']);
  assert.deepEqual(source, before);
  renderVolatilityState(host, source, 'AVAX', asset => clicks.push(asset));
  assert.equal(buttons(host).length, 10, 'Rerender replaces prior cards.');
  assert.equal(buttons(host)[7].attributes['aria-pressed'], 'true');
  assert.deepEqual(source, before);
}));

test('missing snapshots show an unavailable state without fabricated asset metrics', () => withDom(() => {
  const host = new Element('div');
  renderVolatilityState(host, null, 'BTC');
  assert.equal(buttons(host).length, 0);
  assert.ok(elements(host).some(node => node.textContent === '等待行情'));
}));

test('English mode renders the same ratios with translated labels', () => {
  const moduleUrl = new URL('./blocks/volatility-state.mjs', import.meta.url).href;
  const script = `globalThis.location={search:'?lang=en'};
    const {renderVolatilityState}=await import(${JSON.stringify(moduleUrl)});
    ${Element.toString()}
    globalThis.document={createElement:tag=>new Element(tag)};
    const host=new Element('div');
    renderVolatilityState(host,{assets:['BTC','ETH'],metrics:[
      {asset:'BTC',volatility_state:{ratio:1.2,state:'above_usual',recent_std:.012,baseline_std:.01,reason:null}},
      {asset:'ETH',volatility_state:{ratio:1.1999,state:'near_usual',recent_std:.011999,baseline_std:.01,reason:null}}
    ]},'BTC');
    process.stdout.write(JSON.stringify(host));`;
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', script], {encoding: 'utf8'});
  assert.match(output, /Current volatility/);
  assert.match(output, /Past 10 min volatility ÷ past 120 min baseline/);
  assert.match(output, /Above usual/);
  assert.match(output, /Near usual/);
  assert.match(output, /<1.20×/);
});
