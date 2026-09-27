// 04 Honest evaluation: the main 4-method table, the February re-check, null results, coverage and the worst 10.
import { el, pct, bps, when, hoverTip } from "../format.mjs?v=en-r1";
import { tr } from "../i18n.mjs?v=en-r1";

function methodName(row) {
  const names = {
    kronos_base: "Kronos",
    no_propagation: "No-propagation baseline",
    btc_beta: "BTC-beta baseline",
    historical_30: "Past-30-minute baseline",
  };
  return tr(row.name, names[row.method] ?? row.name);
}

export function evalTitle(ev) {
  const t = ev?.january?.table;
  if (!t?.length) return tr("评测数据暂不可用", "Evaluation data is unavailable");
  const bestMdd = [...t].sort((a, b) => a.mdd_mae_bps - b.mdd_mae_bps)[0];
  const bestVol = [...t].sort((a, b) => b.vol_top3 - a.vol_top3)[0];
  return tr(`这批事件中，${bestMdd.name} 回撤误差最低；波动排序仍由${bestVol.name}领先`, `Across these events, ${methodName(bestMdd)} has the lowest drawdown error; ${methodName(bestVol)} leads volatility ranking`);
}

export function renderEvalTable(host, subHost, febHost, ev) {
  const t = ev.january.table;
  const best = {
    vol_top3: Math.max(...t.map((r) => r.vol_top3)),
    vol_mae_bps: Math.min(...t.map((r) => r.vol_mae_bps)),
    mdd_mae_bps: Math.min(...t.map((r) => r.mdd_mae_bps)),
  };
  const table = el("table");
  const head = table.createTHead().insertRow();
  [[tr("方法", "Method"), ""], [tr("波动 Top3 期望召回", "Volatility Top 3 recall"), tr("越高越好", "Higher is better")], [tr("波动误差 bps", "Volatility error · bps"), tr("越低越好", "Lower is better")], [tr("回撤误差 bps", "Drawdown error · bps"), tr("越低越好", "Lower is better")]].forEach(([h, s], i) => {
    const th = el("th", null, h); if (s) th.appendChild(el("small", null, s)); head.appendChild(th);
    if (i === 1) { th.tabIndex = 0; hoverTip(th, tr("波动 Top3 期望召回", "Volatility Top 3 expected recall"), [[tr("含义", "Meaning"), tr("前三名的期望重合比例", "Expected overlap among the top three")], [tr("并列", "Ties"), tr("按等概率计入，不是三只全对率", "Equal-probability tie scoring; not an exact-match rate")]]); }
  });
  const body = table.createTBody();
  t.forEach((r) => {
    const row = body.insertRow();
    if (r.method === "kronos_base") row.className = "kronos";
    row.insertCell().textContent = r.method === "kronos_base" ? tr(`${r.name}（原版）`, `${methodName(r)} (original)`) : methodName(r);
    [["vol_top3", pct(r.vol_top3)], ["vol_mae_bps", r.vol_mae_bps.toFixed(1)], ["mdd_mae_bps", r.mdd_mae_bps.toFixed(1)]].forEach(([k, v]) => {
      const c = row.insertCell(); c.textContent = v;
      if (Math.abs(r[k] - best[k]) < 1e-12) c.className = "best";
    });
  });
  host.replaceChildren(table);
  subHost.textContent = tr(`一月 ${ev.january.events} 次冲击，同一批事件上比较；● 为每列最好。`, `${ev.january.events} January shocks, compared on the same events. ● marks the best result in each column.`);
  const f = ev.february;
  febHost.replaceChildren();
  febHost.append(tr(`二月观察性复核（${f.events} 次）：回撤误差 `, `February observational check (${f.events} shocks): drawdown error `), el("b", null, tr(`${f.kronos_mdd_bps.toFixed(2)} 对 ${f.hist_mdd_bps.toFixed(2)} bps`, `${f.kronos_mdd_bps.toFixed(2)} vs ${f.hist_mdd_bps.toFixed(2)} bps`)), tr(`；跌冲击贡献从一月约${pct(f.january_down_share, 0)}变为${pct(f.down_share)}，没有复现相同的优势来源。`, `; down shocks contributed ${pct(f.down_share)} of the improvement, versus about ${pct(f.january_down_share, 0)} in January. The same source of improvement did not repeat.`));
  febHost.tabIndex = 0;
  hoverTip(febHost, tr("二月复核", "February check"), [[tr("一月优势来自下跌事件", "January improvement from down shocks"), tr(`约 ${pct(f.january_down_share, 0)}`, `About ${pct(f.january_down_share, 0)}`)], [tr("二月", "February"), pct(f.down_share, 0)],
    [tr("二月波动 Top3 召回", "February volatility Top 3 recall"), tr(`${pct(f.kronos_vol_top3)} 对 ${pct(f.hist_vol_top3)}`, `${pct(f.kronos_vol_top3)} vs ${pct(f.hist_vol_top3)}`)]]);
}

export function renderNulls(listHost, covHost, topHost, topSummary, ev, explorer) {
  const n = ev.nulls;
  const k = n.downside.kronos_base, h = n.downside.historical_30;
  const minutes = (v) => (v == null ? "null" : tr(`${v} 分钟`, `${v} min`));
  const items = [
    { n: pct(n.direction.accuracy), t: tr("涨跌方向", "Price direction"), c: tr("和抛硬币差不多", "Similar to a coin toss"),
      tip: [[tr("猜对", "Correct"), tr(`${n.direction.correct} / ${n.direction.pairs} 个资产事件`, `${n.direction.correct} / ${n.direction.pairs} asset-event pairs`)]] },
    { n: pct(k.top3), t: tr("下行波动 Top3", "Downside volatility Top 3"), c: tr("比过去 30 分钟基线略差", "Slightly worse than the past-30-minute baseline"),
      tip: [[tr("基线", "Baseline"), pct(h.top3)], [tr("误差 Kronos / 基线", "Error: Kronos / baseline"), `${bps(k.mae_bps, 2)} / ${bps(h.mae_bps, 2)}`]] },
    { n: n.radius.max_drawdown == null ? tr("未确立", "Not established") : tr(`${n.radius.max_drawdown} 分`, `${n.radius.max_drawdown} min`), t: tr("回撤预警半径", "Drawdown warning horizon"), c: tr("本次分段比较未建立稳定优势时长", "No horizon showed a consistent advantage"),
      tip: [[tr("回撤", "Drawdown"), minutes(n.radius.max_drawdown)], [tr("波动", "Volatility"), minutes(n.radius.volatility)], [tr("两项同时", "Both"), minutes(n.radius.joint)]] },
  ];
  listHost.replaceChildren(...items.map((it) => {
    const box = el("div", "null");
    box.tabIndex = 0;
    box.append(el("p", "n", it.n), el("p", "t", it.t), el("p", "c", it.c));
    hoverTip(box, it.t, it.tip);
    return box;
  }));

  const c = ev.coverage;
  covHost.replaceChildren();
  const big = el("div");
  big.append(el("p", "n", pct(c.coverage_valid, 1)), el("p", "t", tr("区间过度自信", "Overconfident intervals")));
  const right = el("div");
  right.tabIndex = 0;
  right.appendChild(el("p", "c", tr(`${c.expected} 个资产事件中，仅 ${c.covered} 个实际回撤落入采样区间。只有3个预选事件，不是30个独立样本；区间未经校准。`, `Only ${c.covered} of ${c.expected} asset-event drawdowns fell within the sampled interval. These are 3 preselected events, not 30 independent samples. The intervals are uncalibrated.`)));
  hoverTip(right, tr("p05–p95 覆盖", "p05–p95 coverage"), [[tr("高于区间", "Above interval"), tr(`${c.above} 次`, `${c.above}`)], [tr("低于区间", "Below interval"), tr(`${c.below} 次`, `${c.below}`)], [tr("样本", "Sample"), tr("3 个回放事件 × 10 个资产", "3 replay events × 10 assets")]]);
  const dots = el("div", "dots");
  for (let i = 0; i < c.expected; i++) dots.appendChild(el("i", i < c.covered ? "in" : null));
  right.appendChild(dots);
  covHost.append(big, right);

  const p = ev.top10_population;
  const downCount = ev.top10.filter((r) => r.direction === "down").length;
  const systemicCount = ev.top10.filter((r) => r.systemic).length;
  topSummary.textContent = tr(`${p.events} 次冲击中误差最大的10次：${downCount}次下跌、${systemicCount}次系统性事件`, `The 10 largest errors across ${p.events} shocks: ${downCount} down shocks, ${systemicCount} systemic events`);
  const table = topHost;
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  ["#", tr("时间 (UTC)", "Time (UTC)"), tr("强度 σ", "Magnitude · σ"), tr("系统性", "Systemic"), tr("触发方向", "Direction"), tr("回撤误差 bps（被评分资产平均）", "Drawdown error · bps (mean across scored assets)"), tr("评分资产数", "Assets scored"), tr("最差资产", "Worst asset"), tr("该资产误差 bps", "Asset error · bps")].forEach((h) => head.appendChild(el("th", null, h)));
  const body = table.createTBody();
  ev.top10.forEach((r) => {
    const row = body.insertRow();
    const count = explorer?.evidence.scored_asset_counts_by_timestamp?.[r.timestamp] ?? (r.systemic ? 10 : tr("排除源头", "Origin excluded"));
    [String(r.rank), `2026-${when(r.timestamp)}`, r.strength_sigma.toFixed(1), r.systemic ? tr("是", "Yes") : tr("否", "No"), r.direction === "down" ? tr("下跌", "Down") : tr("上涨", "Up"), r.mae_bps.toFixed(1), String(count), r.worst_asset, r.worst_asset_error_bps.toFixed(1)]
      .forEach((v) => { row.insertCell().textContent = v; });
  });
}
