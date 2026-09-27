// One-page presentation of the existing frozen evidence; no loading or inference.
import { el } from "../format.mjs?v=en-r1";
import { tr } from "../i18n.mjs?v=en-r1";
import { renderEvalTable } from "./evaluation.mjs?v=en-r1";
import { renderAcceptance, tierLabel } from "./acceptance.mjs?v=en-r1";

function section(id, title, level = "h3") {
  const node = el("section", `evidence-section evidence-${id}`);
  node.id = id;
  const heading = el(level, "evidence-heading", title);
  heading.id = `${id}-heading`;
  node.setAttribute("aria-labelledby", heading.id);
  node.appendChild(heading);
  return node;
}

function unavailable(node, message) {
  const notice = el("p", "evidence-unavailable", message);
  notice.setAttribute("role", "status");
  node.appendChild(notice);
}

// Build each data-backed section independently, so one unavailable export does
// not erase the other results. Original renderers retain metric definitions.
export function renderIntegratedEvidence(host, {
  evaluation, explorer, acceptance,
} = {}) {
  host.replaceChildren();
  host.classList.add("integrated-evidence");
  const result = { rendered: [], unavailable: [] };

  const mount = (id, title, source, render, level) => {
    const node = section(id, title, level);
    host.appendChild(node);
    if (!source) {
      unavailable(node, tr(`${title}数据暂不可用。`, `${title} data is unavailable.`));
      result.unavailable.push(id);
      return;
    }
    const content = el("div", "evidence-content");
    try {
      render(content);
      node.appendChild(content);
      result.rendered.push(id);
    } catch {
      unavailable(node, tr(`${title}数据不完整，暂无法显示。`, `${title} data is incomplete.`));
      result.unavailable.push(id);
    }
  };

  mount("evaluation", tr("真实测评依据", "Evaluation results"), evaluation, (content) => {
    const scope = el("p", "minor evidence-scope");
    const table = el("div", "evaltable");
    const february = el("p", "feb");
    renderEvalTable(table, scope, february, evaluation);
    content.append(table, february);
  }, "h2");

  mount('acceptance', tr('LoRA 微调结果', 'LoRA fine-tuning results'), acceptance, content => {
    const title = el('h4', 'acceptance-title');
    const table = el('div', 'acctable');
    const line = el('p', 'acc-line');
    const note = el('p', 'acc-note');
    renderAcceptance(title, table, line, note, acceptance);
    const comparisons = acceptance.tiers.map(t => `${tierLabel(t)} ${t.vs.original.estimate >= 0 ? '+' : '−'}${Math.abs(t.vs.original.estimate * 100).toFixed(2)} pp`);
    line.textContent = tr(`相对原版：${comparisons.join('，')}。两档均低于动态基线。`, `Compared with the original: ${comparisons.join('; ')}. Both groups trail the dynamic baseline.`);
    note.textContent = tr('本次微调没有改善回撤预测。', 'Fine-tuning did not improve drawdown forecasts.');
    content.append(table, line, note);
  });

  mount("capabilities", tr("哪些能力还不能依赖", "Limitations"), evaluation, (content) => {
    const limits = el("ul", "capability-limits");
    [
      tr("暂不能可靠判断未来涨跌。", "This model cannot reliably predict price direction."),
      tr("判断哪些资产的下跌波动更大，模型不如直接参考最近走势。", "Recent price history ranks downside volatility better than the model."),
      tr("还没找到回撤预测能持续优于历史基线的时间范围。", "We have not established how long drawdown forecasts can consistently outperform the historical baseline."),
      tr("预测区间还不可靠，实际回撤经常落在区间外。", "Prediction intervals are unreliable: actual drawdowns often fall outside them."),
    ].forEach(text => limits.appendChild(el("li", null, text)));
    content.appendChild(limits);
  });
  return result;
}
