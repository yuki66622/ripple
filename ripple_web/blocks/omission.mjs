// Display only supplied frozen omission statistics; no ranking, fetching, or inference.
import { el, pct } from "../format.mjs?v=ui-r2";

const percentage = (value, digits = 2) => Number.isFinite(value) ? pct(value, digits) : "未定义";

export function renderOmissions(host, data) {
  host.classList.add("omissions");
  host.replaceChildren();
  host.appendChild(el("p", "omission-scope", "仅适用于已经超车的遗漏记录（资产 × 事件 × 指标），不是所有事件的遗漏率，也不是损失或收益。01/02 合并后，每条遗漏记录等权。"));
  host.appendChild(el("p", "omission-formula", "相对超出 =（漏掉资产实际值 − 已选 Top3 垫底实际值）÷ 垫底实际值。分位数使用未舍入比值线性插值。"));
  const table = el("table", "omission-table");
  const head = table.createTHead().insertRow();
  ["指标", "中位数", "p75", "p90"].forEach((text) => {
    const th = el("th", null, text); th.scope = "col"; head.appendChild(th);
  });
  const body = table.createTBody();
  data.metrics.forEach((metric) => {
    const row = body.insertRow();
    const name = el("th", null, metric.label); name.scope = "row"; row.appendChild(name);
    [percentage(metric.p50), percentage(metric.p75), percentage(metric.p90)]
      .forEach((value) => { row.insertCell().textContent = value; });
  });
  const counts = data.metrics.map((metric) => `${metric.label} ${metric.finite_n}/${metric.n} 条，零分母 ${metric.zero_denominator_count} 条`).join("；");
  host.append(table, el("p", "omission-note", `有限比值 / 超车记录：${counts}。零分母不计入分位数；相对比例与绝对 bps 一起阅读。`));
  const cases = el("div", "omission-cases");
  data.cases.forEach((item) => {
    const article = el("article", "omission-case");
    const selection = String(item.selection);
    const absolute = !/p90/i.test(selection) && /max|largest|最大/i.test(selection);
    const metricLabel = data.metrics.find((metric) => metric.key === item.metric)?.label || item.label;
    article.append(el("h3", null, `${item.asset} · ${metricLabel}`), el("p", "omission-selection", absolute
      ? "按已实现回撤的绝对超出量最大选取，属于结果后选案例。"
      : "接近 p90 的真实记录，不是精确分位点；属于结果后选案例。"));
    const details = el("dl");
    [["时间 UTC", item.timestamp.replace("T", " ").replace(/Z$/, "")],
      ["被漏资产", item.asset], ["已选 Top3", item.selected_assets.join(" / ")],
      ["Top3 垫底实际值", percentage(item.weakest_selected_actual, 4)],
      ["漏掉资产实际值", percentage(item.actual, 4)],
      ["相对超出", percentage(item.ratio)],
      ["绝对超出", Number.isFinite(item.excess_bps) ? `${item.excess_bps.toFixed(2)} bps` : "未定义"]]
      .forEach(([label, value]) => { details.append(el("dt", null, label), el("dd", null, value)); });
    article.appendChild(details); cases.appendChild(article);
  });
  host.append(cases, el("p", "omission-note", "这三个案例用于说明已经发生的遗漏幅度，按实际结果选择，不是新的验证样本，也不代表全部遗漏记录。"));
}
