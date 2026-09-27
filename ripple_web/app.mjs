// Frozen-data exploration. Loading or interacting never calls a model or live-market API.
import { el, clamp01, headline, ASSETS, hideTip } from "./format.mjs?v=ui-r2";
import { loadDatasets } from "./data-loader.mjs?v=ui-r2";
import { clampMinute } from "./replay-state.mjs?v=ui-r2";
import { radarTitle, renderRadar } from "./blocks/radar.mjs?v=ui-r2";
import { contagionTitle, chainSentences, renderContagion, strongestEdge } from "./blocks/contagion.mjs?v=ui-r2";
import { replayTitle, renderReplay, renderReplayLegend } from "./blocks/replay.mjs?v=ui-r2";
import { renderFingerprint } from "./blocks/fingerprint.mjs?v=ui-r2";
import { evalTitle, renderEvalTable, renderNulls } from "./blocks/evaluation.mjs?v=ui-r2";
import { renderMethod } from "./blocks/method.mjs?v=ui-r2";
import { renderAcceptance } from "./blocks/acceptance.mjs?v=ui-r2";

const $ = (id) => document.getElementById(id);
const LIGHT = [233, 238, 242], DARK = [15, 27, 41];
const NAV_OF = { radar: "radar", contagion: "contagion", replay: "replay", fingerprint: "replay", evaluation: "evaluation", nulls: "evaluation", method: "method", acceptance: "method" };
const state = { event: 0, source: "BTC", asset: null, minute: 0 };
let D = {}, playing = null, wired = false, loading = false;
let autoPlayed = false;

function problem(host, message) {
  const box = el("div", "load-problem");
  box.setAttribute("role", "status");
  const button = el("button", null, "重新读取");
  button.type = "button";
  button.addEventListener("click", main);
  box.append(el("p", null, message), button);
  host.replaceChildren(box);
}

function renderEventButtons() {
  document.querySelectorAll("[data-events]").forEach((host) => {
    host.replaceChildren(...D.radar.events.map((ev, i) => {
      const b = el("button", null, `${ev.reason_label} · ${ev.timestamp.slice(5, 10).replace("-", "/")}`);
      b.type = "button"; b.setAttribute("aria-pressed", String(i === state.event));
      b.addEventListener("click", () => setEvent(i));
      return b;
    }));
  });
}

function stopPlaying() {
  if (playing !== null) clearInterval(playing);
  playing = null;
  $("playback-play").textContent = state.minute === 30 ? "已播放完" : state.minute ? "继续播放" : "播放回放";
  $("playback-play").disabled = state.minute === 30;
  $("playback-play").setAttribute("aria-pressed", "false");
}

function startPlaying() {
  if (playing !== null || !replayEvent()) return;
  autoPlayed = true;
  if (state.minute === 30) setMinute(0);
  $("playback-play").textContent = "暂停回放";
  $("playback-play").disabled = false;
  $("playback-play").setAttribute("aria-pressed", "true");
  playing = setInterval(() => setMinute(state.minute + 1), 650);
}

function restartPlayback() {
  stopPlaying();
  setMinute(0);
  startPlaying();
}

function replayEvent() {
  const id = D.radar?.events[state.event]?.event_id;
  return D.replays?.events.find((ev) => ev.event_id === id);
}

function replayOptions() {
  const ev = replayEvent();
  const assetRisk = D.radar.events[state.event].assets.find((a) => a.asset === state.asset);
  const mdd = D.explorer?.replays[ev.event_id]?.mdd[state.asset] ||
    { p05: assetRisk.mdd_p05, p50: assetRisk.mdd_p50, p95: assetRisk.mdd_p95 };
  return { revealedMinutes: state.minute, mdd };
}

function renderAssetButtons() {
  const ev = replayEvent();
  $("replay-assets").replaceChildren(...Object.keys(ev.assets).map((a) => {
    const b = el("button", null, a);
    b.type = "button"; b.setAttribute("aria-pressed", String(a === state.asset));
    b.addEventListener("click", () => setAsset(a));
    return b;
  }));
}

function setAsset(asset, openReplay = false) {
  if (!replayEvent()?.assets[asset]) return;
  state.asset = asset;
  hideTip();
  [...$("replay-assets").children].forEach((b) => b.setAttribute("aria-pressed", String(b.textContent === asset)));
  drawReplay();
  drawRadar();
  if (openReplay) $("replay").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

function drawRadar() {
  renderRadar($("radar-table"), D.radar.events[state.event], {
    replayEvent: replayEvent(), revealedMinutes: state.minute, selectedAsset: state.asset,
    onSelectAsset: (asset) => setAsset(asset, true),
  });
}

function drawReplay() {
  const ev = replayEvent();
  if (!ev) return;
  const opts = replayOptions();
  headline($("replay-h"), replayTitle(ev, state.asset, opts));
  renderReplay($("replay-chart"), ev, state.asset, opts);
  renderReplayLegend($("replay-legend"), ev);
  $("replay-chart").dataset.forecastId = D.explorer?.replays[ev.event_id]?.forecast_id || ev.event_id;
}

function setMinute(value) {
  state.minute = clampMinute(value);
  hideTip();
  $("replay-time").value = String(state.minute);
  const label = state.minute ? `+${state.minute}分钟` : "冲击时刻";
  $("replay-time-label").value = label;
  $("replay-time").setAttribute("aria-valuetext", label);
  $("playback-next").disabled = state.minute === 30;
  $("playback-all").disabled = state.minute === 30;
  if (state.minute === 30 || playing === null) stopPlaying();
  drawReplay();
  drawRadar();
}

function setEvent(index) {
  stopPlaying();
  state.event = index; state.minute = 0;
  const ev = replayEvent();
  if (!ev) throw new Error("回放事件与雷达数据不匹配");
  if (!ev.assets[state.asset]) state.asset = ev.origin_assets[0];
  renderEventButtons(); renderAssetButtons();
  headline($("radar-h"), radarTitle(D.radar.events[index]));
  $("event-context").textContent = `${ev.timestamp.replace("T", " ").replace("Z", " UTC")} · ${ev.origin_assets.join(" / ")} 先触发 · 三个预选历史案例`;
  setMinute(0);
}

function setSource(asset) {
  if (!D.graph) return;
  state.source = asset;
  $("graph-source").value = asset;
  headline($("contagion-h"), contagionTitle(D.graph, asset));
  renderContagion($("contagion-map"), D.graph, asset, setSource);
}

function wireControls() {
  $("replay-time").addEventListener("input", (e) => { autoPlayed = true; stopPlaying(); setMinute(e.target.value); });
  $("playback-play").addEventListener("click", () => {
    if (playing !== null) { stopPlaying(); return; }
    startPlaying();
  });
  $("playback-restart").addEventListener("click", restartPlayback);
  $("playback-reset").addEventListener("click", () => { autoPlayed = true; stopPlaying(); setMinute(0); });
  $("playback-next").addEventListener("click", () => { autoPlayed = true; stopPlaying(); setMinute(state.minute + 5); });
  $("playback-all").addEventListener("click", () => { autoPlayed = true; stopPlaying(); setMinute(30); });
  $("graph-source").addEventListener("change", (e) => setSource(e.target.value));
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlaying(); });
}

function wireScroll() {
  const nav = $("nav"), sections = [...document.querySelectorAll("main > section")];
  const links = [...document.querySelectorAll(".links a")];
  let queued = false;
  const frame = () => {
    queued = false;
    const k = clamp01((scrollY + innerHeight * .75 - $("evaluation").offsetTop) / (innerHeight * .5));
    const bg = `rgb(${LIGHT.map((v, i) => Math.round(v + (DARK[i] - v) * k)).join(",")})`;
    document.body.style.background = bg; nav.style.background = bg;
    nav.style.color = k > .5 ? "#fff" : "";
    let current = sections[0].id;
    sections.forEach((s) => { if (s.getBoundingClientRect().top < innerHeight * .45) current = s.id; });
    links.forEach((a) => { const active = a.dataset.sec === NAV_OF[current]; a.classList.toggle("on", active); active ? a.setAttribute("aria-current", "location") : a.removeAttribute("aria-current"); });
  };
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(frame); } }, { passive: true });
  frame();
  const observer = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add("in"); }), { threshold: .05 });
  sections.forEach((s) => observer.observe(s));
  const replayObserver = new IntersectionObserver((entries) => {
    if (autoPlayed) { replayObserver.disconnect(); return; }
    if (entries.some((entry) => entry.isIntersecting && entry.intersectionRatio >= .25) && replayEvent() && !document.hidden) {
      autoPlayed = true;
      replayObserver.disconnect();
      if (!matchMedia("(prefers-reduced-motion: reduce)").matches) startPlaying();
    }
  }, { threshold: .25 });
  replayObserver.observe($("replay-chart"));
  let timeout;
  addEventListener("resize", () => { clearTimeout(timeout); timeout = setTimeout(() => { drawReplay(); frame(); }, 120); });
}

async function main() {
  if (loading) return;
  loading = true;
  stopPlaying();
  const { data, errors } = await loadDatasets();
  D = data;
  try {
    if (D.radar?.events?.length && D.replays?.events?.length) {
      $("playback").hidden = false;
      setEvent(Math.min(state.event, D.radar.events.length - 1));
    } else {
      $("radar-h").textContent = "暂时无法读取历史回放";
      $("replay-h").textContent = "回放尚未就绪";
      $("event-context").textContent = "可以重试读取，评测和其他可用部分仍可浏览。";
      $("playback").hidden = true;
      document.querySelectorAll("[data-events]").forEach((host) => host.replaceChildren());
      $("replay-assets").replaceChildren();
      problem($("radar-table"), "回放数据不可用，请确认本地服务仍在运行。");
      problem($("replay-chart"), "缺少冻结回放数据。");
    }
    if (D.graph) {
      $("graph-source").replaceChildren(...ASSETS.map((asset) => { const option = el("option", null, asset); option.value = asset; return option; }));
      if (!D.graph.nodes.some((n) => n.asset === state.source)) state.source = strongestEdge(D.graph)?.source || "BTC";
      $("chains").replaceChildren(...chainSentences(D.graph).map((s) => el("li", null, s)));
      setSource(state.source);
      const perm = D.explorer?.evidence.permutation;
      $("graph-evidence").textContent = perm ? `200次独立时间错位检验：真实${perm.real.nonzero_directed_pair_count}条边，随机中位${perm.null_diagnostics.nonzero_directed_pair_count.median}条；平均延迟${perm.real.observation_mean_delay_minutes.toFixed(2)}分钟，随机中位${perm.null_diagnostics.observation_mean_delay_minutes.median.toFixed(2)}分钟。两项Holm校正p=${perm.tests.nonzero_directed_pair_count.holm_adjusted_p.toFixed(5)}。零模型未保留共同市场时段，只支持时间对齐，不证明因果或预测价值。` : "这是整月时间先后记录，不是当前事件的传播预测。";
    } else { $("contagion-h").textContent = "关联图谱暂不可用"; problem($("contagion-map"), "其余回放与评测不受影响。"); }
    if (D.nearest) renderFingerprint($("fp"), $("fp-classes"), $("fp-sub"), D.nearest, D.explorer, $("fp-controls"));
    else problem($("fp"), "指纹数据暂不可用。");
    if (D.evaluation?.january?.table?.length && D.evaluation.february && D.evaluation.nulls && D.evaluation.coverage && D.evaluation.top10) {
      headline($("eval-h"), evalTitle(D.evaluation));
      renderEvalTable($("eval-table"), $("eval-sub"), $("eval-feb"), D.evaluation);
      renderNulls($("nulls-list"), $("coverage"), $("top10"), $("top10-summary"), D.evaluation, D.explorer);
    } else { errors.evaluation ||= "评测数据不完整"; $("eval-h").textContent = "评测数据暂不可用"; problem($("eval-table"), "回放仍可继续浏览，稍后可重试读取评测。"); }
    if (D.method) renderMethod($("segments"), $("method-notes"), D.method);
    else problem($("segments"), "此前导出的方法说明暂不可用。");
    if (D.acceptance) renderAcceptance($("acc-h"), $("acc-table"), $("acc-line"), $("acc-note"), D.acceptance);
    else { $("acc-h").textContent = "微调结果暂不可用"; problem($("acc-table"), "尚未读到已导出的微调结果，不影响原版模型回放。"); }
    ["fp-h", "nulls-h", "method-h"].forEach((id) => headline($(id), $(id).textContent));
    document.body.dataset.loadState = Object.keys(errors).length ? "partial" : "ready";
  } catch (error) {
    $("radar-h").textContent = "数据无法完整展示";
    problem($("radar-table"), `请重新读取冻结数据：${error.message}`);
    document.body.dataset.loadState = "error";
  } finally {
    if (!wired) { wireControls(); wireScroll(); wired = true; }
    $("radar").classList.add("in");
    loading = false;
  }
}

main();
