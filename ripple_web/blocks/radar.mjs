// 01 Radar: past-30-minute volatility ranking (the historical_30 rule) beside Kronos' uncalibrated drawdown interval.
// The screen shows bars and ranges only; every number is in the row's hover.
import { el, pct, hoverTip } from "../format.mjs?v=en-r1";
import {tr} from "../i18n.mjs?v=en-r1";
import { revealedDrawdown } from "../replay-state.mjs?v=ui-r2";

export function radarTitle(ev) {
  const top = [...ev.assets].sort((a, b) => b.vol_past30 - a.vol_past30)[0];
  return tr(`过去 30 分钟，${top.asset} 波动最大`, `${top.asset} had the highest volatility in the past 30 min`);
}

export function renderRadar(host, ev, options = {}) {
  host.replaceChildren();
  const rows = [...ev.assets].sort((a, b) => b.vol_past30 - a.vol_past30);
  const vmax = rows[0].vol_past30 || 1;
  const minute = options.revealedMinutes ?? 0;
  const actual = Object.fromEntries(rows.map((a) => [a.asset, options.replayEvent ? revealedDrawdown(options.replayEvent.assets[a.asset], minute) : null]));
  const mmax = Math.max(1e-8, ...ev.assets.map((a) => Math.max(a.mdd_p95, actual[a.asset] ?? 0))) * 1.06;

  const hd = el("div", "row hd");
  const h3 = el("span", null, tr("未来 30 分钟回撤", "Next 30 min drawdown"));
  h3.appendChild(el("span", "tag", tr("未校准", "Uncalibrated")));
  hd.append(el("span"), el("span", null, tr("过去 30 分钟波动", "Past 30 min volatility")), h3);
  host.appendChild(hd);

  rows.forEach((a, i) => {
    const row = el("button", "row");
    row.type = "button";
    row.setAttribute("aria-pressed", String(options.selectedAsset === a.asset));
    row.setAttribute("aria-label", tr(`${a.asset}：过去30分钟波动${pct(a.vol_past30)}，预测回撤中位${pct(a.mdd_p50)}；打开回放`, `${a.asset}: past 30 min volatility ${pct(a.vol_past30)}, median forecast drawdown ${pct(a.mdd_p50)}; open replay`));
    row.addEventListener("click", () => options.onSelectAsset?.(a.asset));
    const bar = el("div", `bar${i === 0 ? " top" : ""}`);
    const fill = el("i"); fill.style.width = `${(a.vol_past30 / vmax) * 100}%`; bar.appendChild(fill);
    bar.appendChild(el("span", "value", pct(a.vol_past30)));
    const band = el("div", "band");
    const ln = el("span", "ln");
    ln.style.left = `${(a.mdd_p05 / mmax) * 100}%`;
    ln.style.width = `${((a.mdd_p95 - a.mdd_p05) / mmax) * 100}%`;
    const mid = el("span", "p50"); mid.style.left = `${(a.mdd_p50 / mmax) * 100}%`;
    band.append(ln, mid);
    band.appendChild(el("span", "value", tr(`${pct(a.mdd_p50)} 中位`, `${pct(a.mdd_p50)} median`)));
    if (actual[a.asset] !== null) {
      const act = el("span", "act"); act.style.left = `${(actual[a.asset] / mmax) * 100}%`;
      band.appendChild(act);
    }
    row.append(el("span", "as", a.asset), bar, band);
    hoverTip(row, a.asset, [
      [tr("过去 30 分钟波动", "Past 30 min volatility"), pct(a.vol_past30)], [tr("回撤区间 p05–p95", "Drawdown p05–p95"), tr(`${pct(a.mdd_p05)} 至 ${pct(a.mdd_p95)}`, `${pct(a.mdd_p05)} to ${pct(a.mdd_p95)}`)],
      [tr("回撤中位", "Median drawdown"), pct(a.mdd_p50)], [tr("已揭示实际回撤", "Revealed actual drawdown"), actual[a.asset] === null ? tr("尚未揭示", "Not yet revealed") : tr(`${pct(actual[a.asset])}（截至+${minute}分）`, `${pct(actual[a.asset])} (through +${minute} min)`)],
    ]);
    host.appendChild(row);
  });

  const foot = el("div", "foot");
  const legend = [["k-rng", tr("Kronos 采样区间", "Kronos sample range")], ["k-p50", tr("逐路径回撤的中位数", "Median of per-path drawdowns")]];
  if (minute) legend.push(["k-act", tr(`实际回撤 · 截至+${minute}分`, `Actual drawdown · through +${minute} min`)]);
  legend.forEach(([k, t]) => {
    const s = el("span"); s.append(el("i", k), document.createTextNode(t)); foot.appendChild(s);
  });
  host.appendChild(foot);
}
