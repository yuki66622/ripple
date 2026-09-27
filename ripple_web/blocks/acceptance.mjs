// 05 Sealed acceptance: the LoRA fine-tune against the original model and two simple baselines on the sealed month.
// Only the LoRA column carries marks: up/down relative to the original model and to the baselines. Losses stay grey.
import { el, pct, hoverTip, MINUS } from "../format.mjs?v=en-r1";
import { tr } from "../i18n.mjs?v=en-r1";

const COLS = [["lora", "LoRA"], ["original", tr("原版 Kronos", "Original Kronos")], ["dynamic", tr("动态基线", "Dynamic baseline")], ["fixed", tr("固定组合", "Fixed pair")]];
const BASELINES = [["dynamic_naive", tr("动态基线", "Dynamic baseline")], ["fixed_pair", tr("固定组合", "Fixed pair")]];
const pp = (v) => `${v > 0 ? "+" : v < 0 ? MINUS : ""}${Math.abs(v * 100).toFixed(2)}pp`;

export function tierLabel(tier) {
  const labels = { small: "Smaller assets", major: "Major assets" };
  return tr(tier.label, labels[tier.tier] ?? tier.label);
}

function mark(up, name) {
  return el("span", `mk ${up ? "up" : "down"}`, `${up ? "↑" : "↓"} ${name}`);
}

// One mark against the original; one against the baselines when they agree, otherwise one per baseline.
function marks(t) {
  const out = [mark(t.vs.original.estimate > 0, tr("原版", "Original"))];
  const base = BASELINES.filter(([k]) => t.vs[k]);
  const ups = base.map(([k]) => t.vs[k].estimate > 0);
  if (ups.every((u) => u === ups[0])) out.push(mark(ups[0], tr("基线", "Baselines")));
  else base.forEach(([, name], i) => out.push(mark(ups[i], name)));
  return out;
}

export function acceptanceTitle(a) {
  const n = [...new Set(a.tiers.map((t) => t.n))];
  return [tr("03 密封验收", "March holdout evaluation"), tr("Top2 精确命中率", "Top 2 exact-match rate"), tr(`${n.join("/")} 窗口/档`, `${n.join("/")} windows per group`)];
}

// The one sentence under the table, built from the paired differences so the numbers cannot drift from the report.
export function acceptanceLine(a) {
  const num = (v) => String(Number((Math.abs(v) * 100).toFixed(2)));
  const cmp = (t) => tr(`${t.vs.original.estimate > 0 ? "高" : "低"} ${num(t.vs.original.estimate)}pp`, `${num(t.vs.original.estimate)} pp ${t.vs.original.estimate > 0 ? "higher" : "lower"}`);
  const lostAll = a.tiers.every((t) => BASELINES.every(([k]) => !t.vs[k] || t.vs[k].estimate < 0));
  if (!lostAll) return "";
  const [small, big] = a.tiers;
  return tr(`微调在${small.label}上比原版${cmp(small)}，${big.label}${cmp(big)}；两档都赢不了简单基线——波动率排名这题，我们不用模型。`, `Compared with the original, fine-tuning scores ${cmp(small)} on smaller assets and ${cmp(big)} on major assets. Neither group beats the simple baselines, so volatility ranking uses the baseline.`);
}

export function renderAcceptance(titleHost, tableHost, lineHost, noteHost, a) {
  // Separators stay outside the inline-block clauses so the line can break at them.
  titleHost.replaceChildren();
  acceptanceTitle(a).forEach((s, i) => { if (i) titleHost.append(" · "); titleHost.append(el("span", "cl", s)); });
  hoverTip(titleHost, tr("Top2 精确命中", "Top 2 exact match"), [[tr("含义", "Meaning"), tr("预测波动最大的两只，和实际完全相同", "Both predicted top-volatility assets match the actual top two")], [tr("月份", "Month"), a.month], [tr("固定组合", "Fixed pair"), a.fixed_pair.join(" + ")]]);

  const table = el("table");
  const head = table.createTHead().insertRow();
  head.appendChild(el("th"));
  COLS.forEach(([key, name]) => head.appendChild(el("th", key === "lora" ? "lora" : null, name)));
  const body = table.createTBody();
  a.tiers.forEach((t) => {
    const row = body.insertRow();
    row.appendChild(el("th", null, tierLabel(t)));
    COLS.forEach(([key]) => {
      const c = row.insertCell();
      if (t[key] == null) { c.className = "na"; c.textContent = "—"; return; }
      c.appendChild(el("span", "v", pct(t[key])));
      if (key !== "lora") return;
      c.className = "lora";
      c.tabIndex = 0;
      const box = el("span", "mks");
      box.append(...marks(t));
      c.appendChild(box);
      const rows = [[tr("对原版", "vs original"), t.vs.original]].concat(BASELINES.filter(([k]) => t.vs[k]).map(([k, name]) => [tr(`对${name}`, `vs ${name.toLowerCase()}`), t.vs[k]]))
        .map(([k, v]) => [k, tr(`${pp(v.estimate)}（${pp(v.ci95[0])} 至 ${pp(v.ci95[1])}）`, `${pp(v.estimate)} (${pp(v.ci95[0])} to ${pp(v.ci95[1])})`)]);
      hoverTip(c, tr(`${t.label} · LoRA 减对照 · 95% 区间`, `${tierLabel(t)} · LoRA minus comparison · 95% interval`), rows);
    });
  });
  tableHost.replaceChildren(table);

  lineHost.textContent = acceptanceLine(a);
  const dd = [...new Set(a.tiers.map((t) => (t.drawdown_mae_change * 100).toFixed(1)))];
  const sign = (s) => (Number(s) > 0 ? `+${s}` : Number(s) < 0 ? `${MINUS}${s.slice(1)}` : s);
  noteHost.textContent = tr(`回撤误差基本持平（${dd.map((s) => `${sign(s)}%`).join(" / ")}）`, `Drawdown error was nearly unchanged (${dd.map((s) => `${sign(s)}%`).join(" / ")})`);
}
