// 02 Contagion: click an asset, the ripple spreads to the others in order of their mean follow-up delay.
// Distance from the centre = mean delay in minutes; node depth = how many times the target followed.
import { svg, label, when, dirWord, hoverTip, reduceMotion, ASSETS } from "../format.mjs?v=en-r1";
import {tr} from "../i18n.mjs?v=en-r1";

const RAMP = ["#A9C6D8", "#6E9DBE", "#35699A", "#153D66"];
const C = 300, R0 = 46, R1 = 236, OUTER = 262;
const MS_PER_MIN = 420;

export function strongestEdge(graph) {
  return [...(graph.edges || [])].sort((a, b) => b.weight - a.weight || b.followup_rate - a.followup_rate || ASSETS.indexOf(a.source) - ASSETS.indexOf(b.source) || ASSETS.indexOf(a.target) - ASSETS.indexOf(b.target))[0] || null;
}

export function contagionTitle(graph, source = strongestEdge(graph)?.source) {
  if (!source) return tr("一月图谱：没有观察到后续触发关联", "January: no subsequent triggers observed");
  const e = strongestEdge({ edges: (graph.edges || []).filter((edge) => edge.source === source) });
  if (!e) return tr(`一月 ${source} 触发后，未观察到其他资产在 30 分钟内跟随`, `January: no other assets triggered within 30 min after ${source}`);
  return tr(`一月 ${e.source} 触发后，${e.target} 最常跟随：${e.source_event_count} 次里 ${e.count} 次`, `January: ${e.target} followed ${e.source} most often — ${e.count} of ${e.source_event_count} events`);
}

export function chainSentences(graph) {
  return (graph.strongest || []).map((s) => {
    const last = Math.max(...s.steps.map((st) => st.delay));
    const directions = new Set(s.steps[0].direction);
    const word = directions.size > 1 ? tr("涨跌混合", "mixed directions") : dirWord(s.steps[0].direction[0]);
    return tr(`${when(s.timestamp).slice(0, 5)} · ${s.origin_assets.join("/")} ${word}：${last} 分钟内另有 ${s.followup_count} 个资产跟随`, `${when(s.timestamp).slice(0, 5)} · ${s.origin_assets.join("/")} ${word}: ${s.followup_count} other assets triggered within ${last} min`);
  });
}

export function renderContagion(host, graph, source, onPick) {
  host.replaceChildren();
  if (!ASSETS.includes(source)) source = strongestEdge(graph)?.source || ASSETS[0];
  const allEdges = graph.edges || [];
  const edges = allEdges.filter((e) => e.source === source);
  const dmax = Math.max(5, ...allEdges.map((e) => e.mean_delay_minutes));
  const wmax = Math.max(1, ...edges.map((e) => e.weight));
  const r = (d) => R0 + (R1 - R0) * Math.sqrt(Math.min(1, d / dmax));
  const others = ASSETS.filter((a) => a !== source);
  const angle = (i) => -90 + (360 / others.length) * i;
  const root = svg("svg", { viewBox: "0 0 600 600", role: "group", "aria-label": tr(`一月全部历史事件中，${source} 触发后其他资产跟随的平均延迟；时间关联，不是因果。各资产可用键盘选择。`, `January events: average delay before other assets triggered after ${source}. Temporal association, not causation. Assets can be selected with the keyboard.`) }, host);

  const ticks = [1, 5, 10].filter((t) => t <= dmax);
  // Between the last node (label runs left) and the first (label runs right): the one gap no node label reaches.
  const tickAng = (-110 * Math.PI) / 180;
  ticks.forEach((t) => {
    svg("circle", { class: "ring", cx: C, cy: C, r: r(t), "vector-effect": "non-scaling-stroke" }, root);
    label(root, C + (r(t) + 3) * Math.cos(tickAng), C + (r(t) + 3) * Math.sin(tickAng), tr(`${t} 分`, `${t} min`), { class: "tk", "text-anchor": "end" });
  });
  svg("circle", { class: "ring", cx: C, cy: C, r: OUTER, "stroke-dasharray": "2 5", "vector-effect": "non-scaling-stroke" }, root);
  const wave = svg("circle", { class: "wave", cx: C, cy: C, r: R0, "stroke-opacity": 0, "vector-effect": "non-scaling-stroke" }, root);

  const drawn = others.map((a, i) => {
    const e = edges.find((x) => x.target === a);
    const ang = (angle(i) * Math.PI) / 180;
    const rr = e ? r(e.mean_delay_minutes) : OUTER;
    const x = C + rr * Math.cos(ang), y = C + rr * Math.sin(ang);
    const spoke = e ? svg("line", { class: "spoke", x1: C, y1: C, x2: x, y2: y, "stroke-width": 1, "stroke-opacity": 0.12 + 0.4 * (e.weight / wmax), "vector-effect": "non-scaling-stroke" }, root) : null;
    const g = svg("g", { class: "node", tabindex: 0, role: "button", "aria-label": tr(`${a}${e ? `：平均 ${e.mean_delay_minutes.toFixed(1)} 分钟，${e.count} 次` : "：未观察到跟随"}，点击设为起点`, `${a}${e ? `: mean delay ${e.mean_delay_minutes.toFixed(1)} min, ${e.count} events` : ": no subsequent triggers observed"}; select as origin`) }, root);
    const tier = e ? Math.min(RAMP.length - 1, Math.floor((e.weight / wmax) * RAMP.length - 1e-9)) : -1;
    svg("circle", { cx: x, cy: y, r: e ? 5 + 5 * (e.weight / wmax) : 4, fill: e ? RAMP[tier] : "none", stroke: e ? "none" : "currentColor", "stroke-opacity": 0.35, "vector-effect": "non-scaling-stroke" }, g);
    const out = Math.cos(ang) >= 0 ? "start" : "end", dx = Math.cos(ang) >= 0 ? 12 : -12;
    label(g, x + dx, y, a, { class: "nl", "text-anchor": out, dy: "0.35em" });
    g.addEventListener("click", () => onPick(a));
    g.addEventListener("keydown", (k) => { if (k.key === "Enter" || k.key === " ") { k.preventDefault(); onPick(a); } });
    if (e) hoverTip(g, `${source} → ${a}`, [[tr("跟随", "Subsequent triggers"), tr(`${e.source_event_count} 次里 ${e.count} 次`, `${e.count} of ${e.source_event_count} events`)], [tr("平均延迟", "Mean delay"), tr(`${e.mean_delay_minutes.toFixed(1)} 分钟`, `${e.mean_delay_minutes.toFixed(1)} min`)]]);
    return { g, spoke, delay: e ? e.mean_delay_minutes : null };
  });

  svg("circle", { cx: C, cy: C, r: 7, fill: "currentColor" }, root);
  label(root, C, C + 12, source, { class: "src", "text-anchor": "middle", dy: "1em" });
  if (!edges.length) label(root, C, C + 43, tr("30 分钟内未观察到跟随", "No subsequent triggers within 30 min"), { class: "tk", "text-anchor": "middle" });

  // Repeat the wave; keep followers visible after their first reveal so targets stay stable.
  const total = dmax * MS_PER_MIN;
  drawn.forEach((d) => { d.g.style.opacity = d.delay == null ? 0.45 : 0; if (d.spoke) d.spoke.style.opacity = 0; });
  if (reduceMotion() || !edges.length) { drawn.forEach((d) => { d.g.style.opacity = d.delay == null ? 0.45 : 1; if (d.spoke) d.spoke.style.opacity = 1; }); return; }
  const t0 = performance.now();
  const step = (now) => {
    if (!root.isConnected) return;
    if (document.hidden || !host.getClientRects().length) { requestAnimationFrame(step); return; }
    const t = Math.max(0, now - t0), minutes = t / MS_PER_MIN;
    const phase = (t % total) / total;
    wave.setAttribute("r", r(phase * dmax));
    wave.setAttribute("stroke-opacity", 0.5 * Math.sin(Math.PI * phase));
    drawn.forEach((d) => {
      if (d.delay == null || minutes < d.delay) return;
      d.g.style.opacity = 1; d.g.style.transition = "opacity .4s"; if (d.spoke) { d.spoke.style.opacity = 1; d.spoke.style.transition = "opacity .4s"; }
    });
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
