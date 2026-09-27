// 05 Method discipline: the three data segments, the LoRA rank and the baselines, as small print.
import { el, hoverTip } from "../format.mjs?v=ui-r2";

export function renderMethod(segHost, notesHost, m) {
  segHost.replaceChildren(...m.segments.map((s) => {
    const box = el("div", `seg s${s.id}`);
    box.append(el("p", "id", s.id), el("p", "nm", s.name), el("p", "rg", s.range), el("p", "dt", s.detail));
    return box;
  }));
  const l = m.lora;
  const lora = el("p");
  lora.append(el("b", null, "LoRA"), document.createTextNode(`rank ${l.rank}`));
  lora.tabIndex = 0;
  hoverTip(lora, "LoRA 设置", [["alpha", String(l.alpha)], ["dropout", String(l.dropout)], ["模块", l.targets.join("/")],
    ["学习率", l.lr.toExponential()], ["batch", String(l.batch_size)], ["最多轮数", String(l.max_epochs)]]);
  const base = el("p");
  base.append(el("b", null, "三条基线"), document.createTextNode(m.baselines.join(" · ")));
  notesHost.replaceChildren(lora, base);
}
