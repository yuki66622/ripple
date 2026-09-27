import { el, pct, hideTip } from './format.mjs?v=en-r1';
import { tr, locale, syncLanguageLink } from './i18n.mjs?v=en-r1';
import { clampMinute, revealedDrawdown } from './replay-state.mjs?v=ui-r2';
import { renderReplay, renderReplayLegend } from './blocks/replay.mjs?v=en-r1';
import { partitionAttention, summarizeRevealed } from './attention-state.mjs?v=attention-r1';
import { renderAttentionTime } from './blocks/attention-time.mjs?v=vol-state-r1';
import { renderVolatilityState } from './blocks/volatility-state.mjs?v=vol-state-r1';
import { chartSeries, renderLiveChart, renderMarketOverview } from './blocks/live-chart.mjs?v=en-r1';
import { createLiveClient, publicLiveError } from './live-client.mjs?v=en-r1';
import { radarTitle, renderRadar } from './blocks/radar.mjs?v=en-r1';
import { renderIntegratedEvidence } from './blocks/integrated-evidence.mjs?v=en-r1';

const $ = (id) => document.getElementById(id);
const historyAnchors = new Set(['method-heading', 'history-radar-panel', 'shock-ripple', 'evaluation', 'capabilities', 'acceptance']);
const state = { mode: new URLSearchParams(location.search).get('mode') === 'history' || historyAnchors.has(location.hash.slice(1)) ? 'history' : 'live', event: 0, asset: null, minute: 0, historyDetail: false };
let initialAnchorPending = Boolean(location.hash);
let live = null, prediction = null, liveError = null, predictionError = null;
let replayVisible = false, autoplayConsumed = false;
let data = {}, timer = null, loading = false, durationUnavailable = false;
const historicalEvent = () => data.radar.events[state.event];
const event = () => state.mode === 'live' ? {assets:live?.metrics || []} : historicalEvent();
const replay = () => data.replays.events.find((r) => r.event_id === event().event_id);
const selection = () => partitionAttention(event());
const prettyTime = (s) => s.replace('T', ' ').replace('Z', ' UTC');
const localTime = (s) => new Intl.DateTimeFormat(locale, {hour:'2-digit', minute:'2-digit', hour12:false}).format(new Date(s));
const eventLabel = (ev) => tr(ev.reason_label, ({strongest_systemic:'Strongest systemic shock', strongest_non_systemic:'Strongest single-asset shock', closest_to_median_trigger_strength:'Median-strength shock'})[ev.reason] || 'Historical shock');

function button(text, action, cls) {
  const b = el('button', cls, text); b.type = 'button'; b.addEventListener('click', action); return b;
}
function retry(host, text) {
  const p = el('div', 'attention-error'); p.setAttribute('role', 'status');
  p.append(el('p', null, text), button(tr('重新读取', 'Reload'), load, 'text-button')); host.replaceChildren(p);
}
function validate(radar, replays) {
  if (!radar?.events?.length || !replays?.events?.length) throw new Error(tr('缺少回放数据', 'Replay data is missing'));
  const ids = new Set();
  radar.events.forEach((ev) => {
    if (ids.has(ev.event_id)) throw new Error(tr('重复回放事件', 'Duplicate replay events')); ids.add(ev.event_id);
    const parts = partitionAttention(ev);
    if (parts.ranked.length !== 10) throw new Error(tr('十资产数据不完整', 'The ten-asset dataset is incomplete'));
    const r = replays.events.find((v) => v.event_id === ev.event_id);
    if (!r || r.timestamp !== ev.timestamp || Object.keys(r.assets || {}).length !== 10) throw new Error(tr('事件快照不匹配', 'Event snapshots do not match'));
    for (const a of parts.ranked) {
      if (![a.mdd_p05, a.mdd_p50, a.mdd_p95].every((v) => Number.isFinite(v) && v >= 0) || a.mdd_p05 > a.mdd_p50 || a.mdd_p50 > a.mdd_p95) throw new Error(tr('回撤采样数据不完整', 'Drawdown samples are incomplete'));
      const v = r.assets[a.asset];
      if (!v || !Number.isFinite(v.spot) || v.spot <= 0) throw new Error(tr('资产快照不完整', 'The asset snapshot is incomplete'));
      for (const [key, length] of [['observed', 61], ['p05', 30], ['p50', 30], ['p95', 30]]) {
        if (!Array.isArray(v[key]) || v[key].length !== length || !v[key].every((x) => Number.isFinite(x) && x > 0)) throw new Error(tr('行情序列不完整', 'The price series is incomplete'));
      }
    }
  });
}
function markSelected() {
  document.querySelectorAll('[data-asset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.asset === state.asset)));
  $('evidence-asset').value = state.asset;
}
function chooseAsset(asset, open = true) {
  if (!selection().ranked.some(a => a.asset === asset)) return;
  if (state.mode === 'history') state.historyDetail = true;
  state.asset = asset; hideTip(); markSelected(); drawEvidence();
  if (open) {
    $('evidence-title').focus({ preventScroll: true });
    $('evidence-chart').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }
}
function drawSelection() {
  const { ranked, focused, deferred, cutoffTied } = selection();
  $('focus-list').replaceChildren(...focused.map((a) => {
    const b = button('', () => chooseAsset(a.asset), 'focus-row'); b.dataset.asset = a.asset;
    const title = el('span', 'focus-name', a.asset);
    const value = el('span', 'focus-value', pct(a.vol_past30));
    const bar = el('span', 'attention-bar'); bar.setAttribute('aria-hidden', 'true');
    const fill = el('i'); fill.style.width = `${a.vol_past30 / (ranked[0].vol_past30 || 1) * 100}%`; bar.append(fill);
    const middle = el('span', 'focus-middle'); middle.append(bar);
    b.append(el('span', 'focus-rank', String(a.rank).padStart(2, '0')), title, middle, value, el('span', 'focus-link', tr('看走势 →', "View chart →")));
    b.setAttribute('aria-label', tr(`${a.asset}，优先第${a.rank}，过去30分钟波动${pct(a.vol_past30)}，查看走势`, `${a.asset}, priority ${a.rank}, past 30 min volatility ${pct(a.vol_past30)}. View chart`));
    return b;
  }));
  $('asset-universe').replaceChildren(...ranked.map((a) => {
    const b = button(a.asset, () => chooseAsset(a.asset), `universe-asset${a.rank <= 3 ? ' priority' : ''}`); b.dataset.asset = a.asset;
    b.setAttribute('aria-label', tr(`${a.asset}，第${a.rank}，查看走势`, `${a.asset}, rank ${a.rank}. View chart`)); return b;
  }));
  $('deferred-rows').replaceChildren(...deferred.map((a) => {
    const row = el('tr'); const name = el('th'); name.scope = 'row';
    name.append(button(a.asset, () => chooseAsset(a.asset), 'text-button'));
    row.append(el('td', 'focus-rank', String(a.rank).padStart(2, '0')), name, el('td', null, pct(a.vol_past30)));
    return row;
  }));
  $('evidence-asset').replaceChildren(...ranked.map((a) => {
    const option = el('option', null, tr(`${a.asset} · 第 ${a.rank}`, `${a.asset} · #${a.rank}`)); option.value = a.asset; return option;
  }));
  $('tie-note').hidden = !cutoffTied;
  markSelected();
}
function setEvent(index) {
  if (state.mode !== 'history') return;
  pause(); state.event = index; state.minute = 0; state.historyDetail = false; state.asset = selection().focused[0].asset;
  const eventButtons = $('event-options');
  if (eventButtons.children.length !== data.radar.events.length) {
    eventButtons.replaceChildren(...data.radar.events.map((ev, i) => button('', () => setEvent(i))));
  }
  [...eventButtons.children].forEach((b, i) => {
    const ev = data.radar.events[i];
    b.textContent = `${eventLabel(ev)} · ${ev.timestamp.slice(5, 10).replace('-', '/')}`;
    b.setAttribute('aria-pressed', String(i === state.event));
  });
  $('event-time').textContent = tr(`${prettyTime(event().timestamp)} · 名单固定在此时刻`, `${prettyTime(event().timestamp)} · Rankings fixed at this time`);
  $('event-time').removeAttribute('title');
  $('event-context').textContent = '';
  drawSelection(); setMinute(0);
}
function pause() {
  if (timer !== null) clearInterval(timer); timer = null;
  $('replay-play').textContent = state.minute === 30 ? tr('重新播放', "Replay") : state.minute ? tr('继续回放', "Resume") : tr('播放回放', "Play");
  $('replay-play').setAttribute('aria-pressed', 'false');
}
function play() {
  if (timer !== null) { pause(); return; }
  if (state.minute === 30) setMinute(0);
  $('replay-play').textContent = tr('暂停回放', "Pause"); $('replay-play').setAttribute('aria-pressed', 'true');
  timer = setInterval(() => setMinute(state.minute + 1), 700);
}
function setMinute(value) {
  state.minute = clampMinute(value); hideTip();
  $('replay-time').value = String(state.minute);
  $('replay-time').setAttribute('aria-valuetext', state.minute ? tr(`冲击后${state.minute}分钟`, `${state.minute} minutes after the shock`) : tr('冲击时刻', "Shock time"));
  $('replay-minute').value = state.minute ? tr(`+${state.minute} 分钟`, `+${state.minute} min`) : tr('冲击时刻', "Shock time");
  $('replay-next').disabled = state.minute === 30;
  $('replay-all').disabled = state.minute === 30;
  if (state.minute === 30) pause();
  if (state.mode === 'history' && data.radar) drawEvidence();
}
function drawEvidence() {
  durationUnavailable = !renderAttentionTime($('duration-panel'), data.duration, state.asset);
  if (state.mode === 'live') renderVolatilityState($('volatility-state-panel'), live, state.asset,
    asset => chooseAsset(asset, false), {stale:Boolean(liveError)});
  $('asset-detail').hidden = state.mode === 'history' && !state.historyDetail;
  if (state.mode === 'live') { drawLiveEvidence(); return; }
  $('history-radar-title').textContent = radarTitle(event());
  $('history-radar-context').textContent = tr(`${prettyTime(event().timestamp)} · ${event().origin_assets.join(' / ')} 先触发 · 三个预选历史案例`, `${prettyTime(event().timestamp)} · First triggers: ${event().origin_assets.join(' / ')} · Three preselected historical cases`);
  renderRadar($('history-radar'), event(), {selectedAsset: state.historyDetail ? state.asset : null, revealedMinutes: state.minute, replayEvent: replay(), onSelectAsset: chooseAsset});
  $('model-title').textContent = tr('模型估计 · 未来30分钟回撤中位', "Forecast · Median drawdown over the next 30 min");
  const r = replay(), a = selection().ranked.find((x) => x.asset === state.asset);
  $('evidence-title').textContent = tr(`${a.asset} · 历史走势`, `${a.asset} · Historical prices`);
  $('evidence-context').textContent = tr(`${prettyTime(event().timestamp)} · 冲击后 30 分钟的固定预测`, `${prettyTime(event().timestamp)} · Frozen forecast for the next 30 min`);
  $('window-reason').textContent = tr(`过去 30 分钟波动 ${pct(a.vol_past30)} · 第 ${a.rank} / 10`, `Past 30 min volatility ${pct(a.vol_past30)} · Rank ${a.rank} / 10`);
  $('model-mdd').textContent = pct(a.mdd_p50);
  $('model-range').textContent = tr(`p05–p95：${pct(a.mdd_p05)}–${pct(a.mdd_p95)} · ${a.mdd_n} 条路径 · 未校准`, `p05–p95: ${pct(a.mdd_p05)}–${pct(a.mdd_p95)} · ${a.mdd_n} paths · Uncalibrated`);
  const actual = revealedDrawdown(r.assets[state.asset], state.minute);
  $('actual-mdd').textContent = actual === null ? tr('待揭示', "Not yet revealed") : pct(actual);
  $('actual-scope').textContent = state.minute ? tr(`实际最大回撤 · 截至 +${state.minute} 分钟`, `Actual max drawdown · Through +${state.minute} min`) : tr('实际最大回撤 · 还没有揭示未来', "Actual max drawdown · Future prices not yet revealed");
  renderReplay($('attention-replay'), r, state.asset, { revealedMinutes: state.minute, mdd: { p05: a.mdd_p05, p50: a.mdd_p50, p95: a.mdd_p95 } });
  renderReplayLegend($('attention-legend'), r);
  const revealed = summarizeRevealed(event(), r, state.minute);
  if (!revealed) {
    $('selection-outcome').textContent = tr('播放回放，比较优先三只与其余七只的实际回撤。', "Play the replay to compare actual drawdowns for the top three and the other seven.");
    $('outcome-inspect').hidden = true;
  } else {
    $('selection-outcome').textContent = tr(`截至 +${state.minute} 分钟：优先三只最大回撤 ${revealed.focusedMax.asset} ${pct(revealed.focusedMax.value)}；其余七只 ${revealed.deferredMax.asset} ${pct(revealed.deferredMax.value)}。仅本案例。`, `Through +${state.minute} min: largest drawdown in the top three, ${revealed.focusedMax.asset} ${pct(revealed.focusedMax.value)}; in the other seven, ${revealed.deferredMax.asset} ${pct(revealed.deferredMax.value)}. This case only.`);
    $('outcome-inspect').hidden = false;
    $('outcome-inspect').textContent = tr(`查看 ${revealed.deferredMax.asset} 的回放 →`, `Replay ${revealed.deferredMax.asset} →`);
    $('outcome-inspect').dataset.targetAsset = revealed.deferredMax.asset;
  }
  $('source-details').textContent = tr("Binance 现货 · USDT · 1 分钟 K 线。名单由冲击前 30 分钟波动排序；Kronos-base 读取此前 256 分钟，生成未来 30 分钟路径。回撤中位数先逐路径计算，再对 10 条路径取中位数。预测和实际的未来数据分开保存，拖动回放不重新预测或重排。", "Binance Spot \u00b7 USDT \u00b7 1 min candles. Rankings use volatility in the 30 min before the shock. Kronos-base takes the preceding 256 min and forecasts the next 30 min. Drawdown is calculated for each of 10 paths, then the median is taken. Forecasts and future observations are stored separately. Scrubbing the replay does not rerun forecasts or rankings.");
}

function drawLoadStatus(errors) {
  if (durationUnavailable) errors.duration ||= tr('观察时长资料不完整', "Monitoring-time data is incomplete");
  $('load-status').textContent = Object.keys(errors).length ? tr('部分历史资料暂不可用，可重新读取。', "Some historical data is unavailable. Try reloading.") : '';
  $('load-retry').hidden = !Object.keys(errors).length;
}
async function load() {
  if (loading) return;
  loading = true; pause(); document.body.dataset.loadState = 'loading';
  const names = ['radar', 'replays', 'evaluation', 'duration', 'explorer', 'acceptance'];
  const results = await Promise.allSettled(names.map(async (name) => {
    const res = await fetch(`data/${name}.json`, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json();
  }));
  const next = {}, errors = {};
  results.forEach((r, i) => { if (r.status === 'fulfilled') next[names[i]] = r.value; else errors[names[i]] = r.reason; });
  renderIntegratedEvidence($('research-evidence'), next);
  revealLinkedSection();
  try {
    validate(next.radar, next.replays); data = next;
    if (state.mode === 'history') showHistory();
    else if (live) showLive();
    drawLoadStatus(errors);
    revealLinkedSection();
    document.body.dataset.loadState = Object.keys(errors).length ? 'partial' : 'ready';
  } catch (error) {
    if (state.mode === 'live' && live) { drawLoadStatus(errors); return; }
    data = {}; $('ready-content').hidden = true; $('loading-message').hidden = false;
    retry($('loading-message'), tr('历史数据暂时无法完整读取。请确认本地服务仍在运行后重试。', "Historical data could not be loaded. Check that the local server is running and try again."));
    document.body.dataset.loadState = 'error';
  } finally { loading = false; }
}
function setControls() {
  const isLive = state.mode === 'live';
  $('mode-live').setAttribute('aria-pressed', String(isLive));
  $('mode-history').setAttribute('aria-pressed', String(!isLive));
  $('history-event-bar').hidden = isLive;
  $('focus').hidden = !isLive;
  $('duration-panel').hidden = !isLive;
  $('volatility-state-panel').hidden = !isLive;
  $('method-heading').hidden = isLive;
  $('shock-ripple').hidden = isLive;
  $('source-details-block').hidden = isLive;
  $('research-section').hidden = isLive;
  document.querySelector('.nav .mark').href = isLive ? '#focus' : '#method-heading';
  document.querySelector('.skip-link').href = isLive ? '#focus' : '#method-heading';
  document.querySelectorAll('.nav [data-view]').forEach(link => { link.hidden = link.dataset.view !== state.mode; });
  $('history-radar-panel').hidden = isLive;
  $('history-detail-close').hidden = isLive;
  $('asset-detail').hidden = !isLive && !state.historyDetail;
  $('history-controls').hidden = isLive;
  $('history-timeline').hidden = isLive;
  $('live-refresh').hidden = !isLive;
  $('live-status').hidden = !isLive;
  $('market-overview').parentElement.hidden = !isLive;
  $('data-mode-label').textContent = isLive ? tr('实时行情', "Live market") : tr('历史冲击回放', "Historical shock replay");
  document.body.dataset.mode = state.mode;
}
function showHistory() {
  setControls();
  if (!data.radar) { $('loading-message').hidden = false; $('ready-content').hidden = true; return; }
  $('ready-content').hidden = false; $('loading-message').hidden = true;
  setEvent(Math.min(state.event, data.radar.events.length - 1));
  if (initialAnchorPending) revealLinkedSection();
  requestAnimationFrame(maybeAutoplay);
}
function maybeAutoplay() {
  if (state.mode === 'history' && state.historyDetail && data.radar && replayVisible && !autoplayConsumed && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    autoplayConsumed = true; play();
  }
}
function showLive() {
  setControls();
  if (!live) { $('ready-content').hidden = true; $('loading-message').hidden = false; return; }
  $('ready-content').hidden = false; $('loading-message').hidden = true;
  if (!live.assets.includes(state.asset)) state.asset = selection().focused[0].asset;
  $('asset-count').value = String(live.assets.length);
  $('event-time').textContent = tr(`${localTime(live.as_of)} 更新`, `Updated ${localTime(live.as_of)}`);
  $('event-time').title = tr(`行情截至 ${prettyTime(live.as_of)} · Binance · USDT`, `Market data through ${prettyTime(live.as_of)} · Binance · USDT`);
  $('event-context').textContent = '';
  drawSelection(); drawEvidence();
  if (initialAnchorPending) revealLinkedSection();
  document.body.dataset.liveState = liveError || live.stale ? 'stale' : 'ready';
}
function drawLiveEvidence() {
  if (!live) return;
  const a = live.metrics.find(v => v.asset === state.asset);
  const view = chartSeries(live, state.asset, prediction);
  const matching = prediction?.window_id === live.window_id && prediction?.as_of === live.as_of;
  const ready = view.predictionState === 'ready';
  $('evidence-title').textContent = tr(`${state.asset} · 走势与预测`, `${state.asset} · Prices & forecast`);
  $('evidence-context').textContent = tr(`${localTime(live.as_of)} 更新 · ${a.last_price.toLocaleString(locale,{maximumFractionDigits:6})} ${live.quote_currency}`, `Updated ${localTime(live.as_of)} · ${a.last_price.toLocaleString(locale,{maximumFractionDigits:6})} ${live.quote_currency}`);
  $('window-reason').textContent = '';
  $('model-title').textContent = tr('模型估计 · 未来30分钟回撤', "Forecast · Drawdown over the next 30 min");
  const failed = matching && prediction.status === 'error';
  $('model-mdd').textContent = ready ? (view.forecastMdd === null ? tr('回撤不可用', "Drawdown unavailable") : pct(view.forecastMdd)) : failed ? tr('预测失败', "Forecast failed") : view.predictionState === 'invalid' ? tr('预测不可用', "Forecast unavailable") : (liveError || live.stale) ? tr('暂停更新', "Updates paused") : tr('预测中…', "Forecasting…");
  $('model-range').textContent = tr('原版 Kronos · 单条采样', "Original Kronos · One sampled path");
  $('actual-scope').textContent = tr('实际最大回撤 · 过去30分钟', "Actual max drawdown · Past 30 min");
  $('actual-mdd').textContent = pct(a.mdd_past30);
  $('prediction-status').textContent = liveError || live.stale ? tr('行情连接异常，自动预测暂停', "Market connection lost. Automatic forecasts paused.")
    : failed ? tr('本窗口预测失败，下一窗口继续', "This forecast failed. The next window will run automatically.")
    : predictionError || (ready ? tr(`${live.assets.length} 个资产已预测 · ${prediction.elapsed_seconds.toFixed(1)} 秒`, `${live.assets.length} assets forecast · ${prediction.elapsed_seconds.toFixed(1)} s`)
    : prediction?.status === 'running' && !matching ? tr('上一批完成后自动预测最新窗口', "The latest window will run after the current batch finishes.") : tr(`正在自动预测 ${live.assets.length} 个资产…`, `Forecasting ${live.assets.length} assets…`));
  renderLiveChart($('attention-replay'), live, state.asset, prediction);
  $('attention-replay').dataset.forecastId = ready ? prediction.forecast_id : '';
  $('market-overview-panel').dataset.forecastId = ready ? prediction.forecast_id : '';
  $('attention-legend').textContent = '';
  $('selection-outcome').textContent = liveError || (live.error ? publicLiveError(live.error) : null) || (live.stale ? tr('行情已过期，显示最后有效窗口。', "Market data is stale. Showing the last valid window.") : '');
  $('outcome-inspect').hidden = true;
  $('source-details').textContent = tr(`${live.source} · ${live.quote_currency} · 1分钟K线。排名使用过去30分钟对数收益的总体标准差×√30。按更新频率读取最新窗口，默认5分钟；每个新窗口用256根K线预测未来30分钟，十资产各一条原版Kronos路径，无置信区间。切换资产不重算；旧窗口预测不叠加到新窗口。`, `${live.source} · ${live.quote_currency} · 1 min candles. Rankings use the population standard deviation of the past 30 min log returns × √30. Data refreshes every 5 min by default. Each new window uses 256 candles to forecast the next 30 min: one original Kronos path per asset, with no confidence interval. Selecting an asset does not rerun the model. Forecasts from older windows are not shown on the current chart.`);
  $('overview-scope').textContent = tr('实线 · 行情　虚线 · 未来30分钟预测', "Solid · Observed prices　Dashed · Next 30 min forecast");
  renderMarketOverview($('market-overview'), live, asset => chooseAsset(asset), prediction);
}
const client = createLiveClient({
  onMarket(snapshot) {
    try {
      if (snapshot.schema_version !== 1 || snapshot.assets?.length !== 10 || snapshot.metrics?.length !== 10 || !snapshot.window_id || !snapshot.as_of) throw new Error(tr('十资产行情不完整', "Market data for the ten assets is incomplete"));
      partitionAttention({assets:snapshot.metrics});
      live = snapshot; liveError = snapshot.error ? publicLiveError(snapshot.error) : null;
      $('live-status').textContent = liveError || (snapshot.stale ? tr('行情已过期', "Market data is stale") : tr(`Binance已连接 · ${snapshot.as_of.slice(11,19)} UTC`, `Binance connected · ${snapshot.as_of.slice(11,19)} UTC`));
      if (state.mode === 'live') showLive();
    } catch (error) { liveError = publicLiveError(error.message); $('live-status').textContent = liveError; document.body.dataset.liveState = 'error'; throw error; }
  },
  onPrediction(result) { prediction = result; predictionError = null; if (state.mode === 'live' && live) drawEvidence(); },
  onError(message, {scope = 'market'} = {}) {
    if (scope === 'prediction') {
      predictionError = message;
      if (state.mode === 'live' && live) drawEvidence();
      return;
    }
    liveError = message; $('live-status').textContent = message;
    document.body.dataset.liveState = 'error';
    if (state.mode === 'live' && live) drawEvidence();
    else if (state.mode === 'live') $('loading-message').textContent = tr(`${message} 可点击刷新行情重试，或切换历史回放。`, `${message} Refresh the market data to retry, or open historical analysis.`);
  },
});
function switchMode(mode) {
  pause(); state.mode = mode;
  const url = new URL(location.href);
  if (mode === 'live') {
    url.searchParams.delete('mode'); url.hash = 'focus';
    state.asset = null; showLive(); client.start();
  } else {
    url.searchParams.set('mode', 'history'); url.hash = 'method-heading';
    client.stop(); showHistory();
  }
  history.replaceState(null, '', url);
  syncLanguageLink();
}
$('mode-live').addEventListener('click', () => switchMode('live'));
$('mode-history').addEventListener('click', () => switchMode('history'));
$('history-detail-close').addEventListener('click', () => {
  pause(); state.historyDetail = false; drawEvidence();
  $('history-radar-panel').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
});
$('live-refresh').addEventListener('click', () => { $('live-status').textContent = tr('正在读取最新行情…', "Loading the latest market data…"); client.refresh(); });
$('refresh-interval').addEventListener('change', e => client.setRefreshInterval(Number(e.target.value)));
$('replay-restart').addEventListener('click', () => { pause(); setMinute(0); play(); });
['history-controls', 'history-timeline'].forEach(id => $(id).addEventListener('pointerdown', () => { autoplayConsumed = true; }, {capture:true}));
['history-controls', 'history-timeline'].forEach(id => $(id).addEventListener('keydown', () => { autoplayConsumed = true; }, {capture:true}));
$('evidence-asset').addEventListener('change', (e) => chooseAsset(e.target.value, false));
$('replay-play').addEventListener('click', play);
$('replay-reset').addEventListener('click', () => { pause(); setMinute(0); pause(); });
$('replay-next').addEventListener('click', () => { pause(); setMinute(state.minute + 5); pause(); });
$('replay-all').addEventListener('click', () => { pause(); setMinute(30); });
$('replay-time').addEventListener('input', (e) => { pause(); setMinute(e.target.value); pause(); });
$('outcome-inspect').addEventListener('click', () => chooseAsset($('outcome-inspect').dataset.targetAsset, false));
$('load-retry').addEventListener('click', load);
function revealLinkedSection() {
  const target = document.getElementById(location.hash.slice(1));
  if (!target) return;
  if (target.matches('details')) target.open = true;
  const parent = target.closest('details');
  if (parent) parent.open = true;
  if (initialAnchorPending && !target.closest('[hidden]') && target.getBoundingClientRect().height) {
    initialAnchorPending = false;
    requestAnimationFrame(() => target.scrollIntoView({behavior:'instant'}));
  }
}
addEventListener('hashchange', () => { revealLinkedSection(); syncLanguageLink(); });
new IntersectionObserver(entries => { replayVisible = entries[0].isIntersecting; maybeAutoplay(); }, {threshold:.25}).observe($('attention-replay'));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) pause();
  else if (state.mode === 'live') client.refresh();
});
let resizeTimer;
addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if ((state.mode === 'live' && live) || (state.mode === 'history' && data.radar)) drawEvidence(); }, 100); });
load();
setControls();
if (state.mode === 'live') client.start();
