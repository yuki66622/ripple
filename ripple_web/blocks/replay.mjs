// 03 Replay: 30 minutes before and after a real shock, Kronos' 10-path median and p05–p95 band against the actual path.
import { svg, label, el, spct, pct, MINUS, showTip, hideTip } from "../format.mjs?v=en-r1";
import {tr} from "../i18n.mjs?v=en-r1";
import { viewSeries } from "../replay-state.mjs?v=ui-r2";

export function replayTitle(ev, asset, options = {}) {
  const view = viewSeries(ev.assets[asset], options.revealedMinutes ?? 0);
  const forecast = options.mdd ? tr(`预测回撤中位 ${pct(options.mdd.p50)}`, `Median forecast drawdown ${pct(options.mdd.p50)}`) : tr(`原冻结单路径回撤 ${pct(ev.assets[asset].kronos_mdd)}`, `Original frozen single-path drawdown ${pct(ev.assets[asset].kronos_mdd)}`);
  const actual = view.minute ? tr(`截至 +${view.minute} 分钟实际回撤 ${pct(view.actualMdd)}`, `Actual drawdown through +${view.minute} min: ${pct(view.actualMdd)}`) : tr("实际结果尚未揭示", "Actual results not yet revealed");
  return tr(`${asset}：${forecast}；${actual}`, `${asset}: ${forecast}; ${actual}`);
}

export function renderReplay(host, ev, asset, options = {}) {
  host.replaceChildren();
  const a = ev.assets[asset];
  const view = viewSeries(a, options.revealedMinutes ?? 0);
  const obs = view.observed;  // Only historical30 plus already-revealed future observations.
  const lo5 = view.lower, mid = view.median, hi5 = view.upper;
  const W = Math.max(320, host.clientWidth), H = Math.max(220, host.clientHeight), m = { l: 46, r: 16, t: 16, b: 26 };
  let y0 = Math.min(...obs, ...lo5), y1 = Math.max(...obs, ...hi5);
  const pad = (y1 - y0) * 0.08 || 0.001; y0 -= pad; y1 += pad;
  const x = (i) => m.l + (i / 60) * (W - m.l - m.r), y = (v) => m.t + ((y1 - v) / (y1 - y0)) * (H - m.t - m.b);
  const root = svg("svg", { width: W, height: H, role: "img", "data-revealed-minutes": view.minute, "aria-label": tr(`${asset}：实际行情仅揭示至冲击后${view.minute}分钟；虚线与采样带为冻结预测，未经校准`, `${asset}: actual prices revealed through ${view.minute} min after the shock; dashed line and sample band are frozen, uncalibrated forecasts`) }, host);
  const span = y1 - y0, step = span > 0.04 ? 0.01 : span > 0.02 ? 0.005 : span > 0.008 ? 0.002 : 0.001;
  for (let v = Math.ceil(y0 / step) * step; v <= y1 + 1e-12; v += step) {
    const zero = Math.abs(v) < 1e-9;
    svg("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "#1D3045", "stroke-opacity": zero ? 0.3 : 0.08, "stroke-width": 1 }, root);
    label(root, m.l - 8, y(v) + 3.5, zero ? "0" : (v > 0 ? "+" : MINUS) + Math.abs(v * 100).toFixed(step < 0.001 ? 2 : 1) + "%", { "text-anchor": "end", fill: "#4E6275", "font-size": 10 });
  }
  svg("line", { x1: x(30), x2: x(30), y1: m.t - 6, y2: H - m.b, stroke: "#1D3045", "stroke-opacity": 0.35, "stroke-width": 1 }, root);
  label(root, x(30), H - 8, tr(`${ev.timestamp.slice(11, 16)} 冲击`, `${ev.timestamp.slice(11, 16)} shock`), { "text-anchor": "middle", fill: "#1D3045", "font-size": 10.5 });
  label(root, m.l, H - 8, tr(`${MINUS}30 分`, `${MINUS}30 min`), { fill: "#4E6275", "font-size": 10 });
  label(root, x(60), H - 8, tr("+30 分", "+30 min"), { "text-anchor": "end", fill: "#4E6275", "font-size": 10 });

  const bandPts = [[x(30), y(0)], ...hi5.map((v, j) => [x(31 + j), y(v)])].concat([...lo5].reverse().map((v, j) => [x(60 - j), y(v)]), [[x(30), y(0)]]);
  svg("path", { d: bandPts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join("") + "Z", fill: "#35699A", "fill-opacity": 0.16 }, root);
  const line = (pts, attrs) => svg("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(""), fill: "none", "stroke-linejoin": "round", ...attrs }, root);
  line([[x(30), y(0)], ...mid.map((v, j) => [x(31 + j), y(v)])], { stroke: "#1D3045", "stroke-width": 1.25, "stroke-dasharray": "4 3" });
  line(obs.map((v, i) => [x(i), y(v)]), { stroke: "#1D3045", "stroke-width": 1.5, "data-series": "actual", "data-point-count": obs.length });
  svg("circle", { cx: x(obs.length - 1), cy: y(obs.at(-1)), r: 3, fill: "#B5543C" }, root);
  if (view.minute > 0 && view.minute < 30) {
    svg("line", { x1: x(30 + view.minute), x2: x(30 + view.minute), y1: m.t, y2: H - m.b, stroke: "#B5543C", "stroke-opacity": .45, "stroke-dasharray": "2 4" }, root);
  }

  const cross = svg("line", { y1: m.t - 6, y2: H - m.b, stroke: "#1D3045", "stroke-opacity": 0.35, "stroke-width": 1, opacity: 0 }, root);
  const hit = svg("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, root);
  hit.addEventListener("pointermove", (e) => {
    const b = root.getBoundingClientRect(), i = Math.max(0, Math.min(60, Math.round(((e.clientX - b.left - m.l) / (W - m.l - m.r)) * 60)));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 1);
    const rows = [[tr("实际", "Actual"), i < obs.length ? spct(obs[i]) : tr("尚未揭示", "Not yet revealed")]];
    if (i > 30) rows.push([tr("Kronos 中位", "Kronos median"), spct(mid[i - 31])], ["p05–p95", tr(`${spct(lo5[i - 31])} 至 ${spct(hi5[i - 31])}`, `${spct(lo5[i - 31])} to ${spct(hi5[i - 31])}`)]);
    showTip(e.clientX, e.clientY, `${asset} · ${i === 30 ? tr("冲击时刻", "Shock time") : tr(`${i < 30 ? MINUS : "+"}${Math.abs(i - 30)} 分`, `${i < 30 ? MINUS : "+"}${Math.abs(i - 30)} min`)}`, rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
}

export function renderReplayLegend(host, ev) {
  host.replaceChildren();
  [["k-obs", tr("实际 · 已揭示部分", "Actual · revealed")], ["k-med", tr("逐分钟预测价格中位线", "Pointwise median forecast")], ["k-band", tr("采样带 · 未校准", "Sample band · uncalibrated")]].forEach(([k, t]) => {
    const s = el("span"); s.append(el("i", k), document.createTextNode(t)); host.appendChild(s);
  });
}
