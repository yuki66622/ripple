// Formatting, DOM and SVG helpers shared by every block.
import { tr } from './i18n.mjs?v=en-r1';
export const MINUS = "−";
export const NS = "http://www.w3.org/2000/svg";
export const ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"];

export const clamp01 = (v) => Math.max(0, Math.min(1, v));
export const spct = (v, d = 2) => (v > 0 ? "+" : v < 0 ? MINUS : "") + Math.abs(v * 100).toFixed(d) + "%";
export const pct = (v, d = 2) => (v * 100).toFixed(d) + "%";
export const bps = (v, d = 1) => `${v.toFixed(d)} bps`;
export const when = (iso) => iso.slice(5, 16).replace("T", " ");   // "01-24 14:00"
export const dirWord = (d) => (d > 0 ? tr("上涨", "rise") : d < 0 ? tr("下跌", "drop") : tr("持平", "flat"));
export const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

// Headlines wrap between clauses, never inside a Chinese word: each clause is an inline-block.
export function headline(node, text) {
  node.replaceChildren(...text.split(/(?<=[，；：])/).map((part) => el("span", "cl", part)));
}

export function svg(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  if (parent) parent.appendChild(node);
  return node;
}

export function label(parent, x, y, content, attrs = {}) {
  const node = svg("text", { x, y, ...attrs }, parent);
  node.textContent = content;
  return node;
}

// One tooltip for the whole page. Values lead, labels follow; text is always set with textContent.
let tip;

export function showTip(x, y, title, rows) {
  if (!tip) {
    tip = el("div", "tip");
    tip.setAttribute("role", "status");
    document.body.appendChild(tip);
  }
  tip.replaceChildren(el("b", null, title));
  rows.forEach(([k, v]) => {
    const row = el("div", "k");
    row.append(el("span", null, k), el("span", null, v));
    tip.appendChild(row);
  });
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  const left = x + 16 + w > innerWidth - 12 ? x - w - 16 : x + 16;
  tip.style.left = `${Math.max(12, left)}px`;
  tip.style.top = `${Math.max(12, Math.min(y - 12, innerHeight - h - 12))}px`;
}
export const hideTip = () => { if (tip) tip.hidden = true; };

export function hoverTip(node, title, rows) {
  const at = (e) => {
    const r = node.getBoundingClientRect();
    const x = e && e.clientX != null ? e.clientX : r.left + r.width / 2;
    const y = e && e.clientY != null ? e.clientY : r.top;
    showTip(x, y, title, typeof rows === "function" ? rows() : rows);
  };
  node.addEventListener("pointermove", at);
  node.addEventListener("focus", () => at());
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("blur", hideTip);
}
