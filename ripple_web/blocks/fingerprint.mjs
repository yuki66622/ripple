// 03 Fingerprint search: the first minutes of a shock as a 10-asset fingerprint, and the 3 nearest January events.
// Cards show who triggered and one outcome number; distances, sigma and class detail are in the hover.
import { el, pct, when, dirWord, hoverTip, ASSETS } from "../format.mjs?v=ui-r2";
import { censorSlots, findNearest } from "../matching.mjs?v=ui-r2";

// Ten slots in fixed asset order; only the assets that triggered are named.
function strip(slots) {
  const box = el("div", "strip");
  ASSETS.forEach((a) => {
    const s = slots[a];
    const on = s && s.present;
    const slot = el("div", `slot${on ? " on" : ""}`);
    const g = el("div", "g", on ? (s.direction > 0 ? "▲" : "▼") : "·");
    if (on) g.style.fontSize = `${Math.min(18, 9 + (s.magnitude_sigma || 0))}px`;
    slot.append(g, el("span", null, on ? a : ""));
    box.appendChild(slot);
  });
  return box;
}

function starters(slots) {
  const first = ASSETS.filter((a) => slots[a] && slots[a].present && slots[a].delay_minutes === 0);
  if (!first.length) return "观察前缀内没有起点";
  const signs = new Set(first.map((a) => slots[a].direction));
  return `${first.join("/")} ${signs.size > 1 ? "涨跌混合" : dirWord(slots[first[0]].direction)}`;
}

function classRows(c) {
  const each = c.origins.length > 1 ? "各 " : "";
  return [["起点最常见", `${c.origins.join("/")}（${each}${c.origin_count}/${c.size} 次）`],
    ["触发方向", `${c.up} 涨 / ${c.down} 跌`], ["跟随资产数中位", String(c.median_followers)]];
}

function paint(host, classesHost, subHost, data, interactive) {
  host.replaceChildren();
  const byId = Object.fromEntries(data.classes.map((c) => [c.id, c]));
  const q = el("div", "fpcard q");
  q.append(el("p", "t", `${interactive ? "选中事件" : "固定示例"} · ${when(data.query.timestamp)} UTC`), el("p", "h", starters(data.query.slots)), strip(data.query.slots),
    el("p", "c", `已观察 ${data.query.observed_minutes} 分钟 · ▲ 涨 / ▼ 跌`));
  host.appendChild(q);
  data.results.forEach((r, i) => {
    const c = byId[r.class_id];
    const card = el("div", "fpcard");
    card.tabIndex = 0;
    card.append(el("p", "t", `第 ${i + 1} 像 · ${when(r.timestamp)} UTC`), el("p", "h", starters(r.prefix)),
      strip(r.prefix), el("p", "c", `历史结果 · 随后 30 分钟最深回撤 ${pct(r.actual_max_mdd)}`));
    hoverTip(card, `第 ${i + 1} 像 · 2026-${when(r.timestamp)}`, [["距离", r.distance.toFixed(4)], ["强度", `${r.magnitude_sigma.toFixed(1)}σ`],
      ["历史回撤前三", r.actual_top_mdd.join("、")], ["完整 30 分钟后的类别", c ? `类 ${c.ordinal}（${c.size} 次）` : "未分类"]]);
    host.appendChild(card);
  });
  subHost.textContent = `${interactive ? "一月历史库探索" : "固定示例"}：两边都只比较前 ${data.query.observed_minutes} 分钟，排除事件自身；使用整月案例，不是当时的回测，也不代表未来概率。`;

  const hit = new Set(data.results.map((r) => r.class_id));
  const row = el("div", "classes");
  row.appendChild(el("span", "lead", `一月 ${data.event_count} 次冲击聚成 ${data.classes.length} 类`));
  data.classes.forEach((c) => {
    const chip = el("span", `cls${hit.has(c.id) ? " on" : ""}`, `类 ${c.ordinal} · ${c.size} 次${c.size === 1 ? " · 孤立观察，尚无重复类别证据" : ""}`);
    chip.tabIndex = 0;
    hoverTip(chip, `类 ${c.ordinal}${c.size === 1 ? " · 孤立观察" : ""}`, classRows(c));
    row.appendChild(chip);
  });
  classesHost.replaceChildren(row);
}

export function renderFingerprint(host, classesHost, subHost, oldNearest, explorer, controlsHost) {
  if (!explorer || !controlsHost) {
    if (controlsHost) controlsHost.replaceChildren();
    paint(host, classesHost, subHost, oldNearest, false);
    return;
  }
  const queryId = explorer.demo?.event_id || explorer.events[0]?.event_id;
  const initialMinutes = explorer.demo?.observed_minutes ?? 10;
  const selectLabel = el("label", "control-field", "选择一月历史冲击");
  const select = el("select");
  select.id = "fingerprint-event";
  for (const event of explorer.events) {
    const option = el("option", null, `${when(event.timestamp)} UTC · ${starters(censorSlots(event.slots, 0))}`);
    option.value = event.event_id;
    select.appendChild(option);
  }
  select.value = queryId;
  selectLabel.appendChild(select);
  const rangeLabel = el("label", "control-field", "可观察的时间前缀");
  const controlRow = el("div", "control-row");
  const slider = el("input");
  slider.type = "range"; slider.min = "0"; slider.max = "30"; slider.step = "1";
  slider.id = "fingerprint-prefix"; slider.value = String(initialMinutes);
  const output = el("output"); output.setAttribute("for", slider.id);
  controlRow.append(slider, output); rangeLabel.appendChild(controlRow);
  controlsHost.classList.add("fp-controls");
  controlsHost.replaceChildren(selectLabel, rangeLabel);
  subHost.setAttribute("aria-live", "polite");
  const update = () => {
    const observed = Number(slider.value);
    output.textContent = `${observed} 分钟`;
    slider.setAttribute("aria-valuetext", `冲击后 ${observed} 分钟`);
    try {
      const query = explorer.events.find((event) => event.event_id === select.value);
      if (!query) throw new Error("未找到选中的历史事件");
      const results = findNearest(explorer, query.event_id, observed);
      paint(host, classesHost, subHost, {
        query: { ...query, slots: censorSlots(query.slots, observed), observed_minutes: observed },
        results, event_count: explorer.events.length, classes: explorer.classes,
      }, true);
    } catch (error) {
      host.replaceChildren(el("p", "c", `无法比较这些指纹：${error.message}`));
      classesHost.replaceChildren();
      subHost.textContent = "指纹数据校验未通过，没有产生匹配结果。";
    }
  };
  select.addEventListener("change", update);
  slider.addEventListener("input", update);
  update();
}
