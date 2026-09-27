// Frozen January fingerprint math. No DOM, network, model, or outcome-based ranking.
export const ASSETS = Object.freeze(["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]);

const blank = () => ({ present: false, direction: null, magnitude_sigma: null, delay_minutes: null });
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const order = (a, b) => a < b ? -1 : a > b ? 1 : 0;

function minutes(value) {
  if (!finite(value) || value < 0 || value > 30) throw new Error("观察时长须在 0–30 分钟内");
  return value;
}

function validateSlots(slots) {
  if (!slots || typeof slots !== "object" || Array.isArray(slots) ||
      Object.keys(slots).length !== ASSETS.length || ASSETS.some((a) => !Object.hasOwn(slots, a))) {
    throw new Error("指纹必须包含固定的十个资产");
  }
  for (const asset of ASSETS) {
    const s = slots[asset];
    const keys = ["present", "direction", "magnitude_sigma", "delay_minutes"];
    if (!s || typeof s !== "object" || Array.isArray(s) || Object.keys(s).length !== keys.length ||
        keys.some((key) => !Object.hasOwn(s, key)) || typeof s.present !== "boolean") {
      throw new Error(`${asset} 指纹格式错误`);
    }
    if (!s.present) {
      if (s.direction !== null || s.magnitude_sigma !== null || s.delay_minutes !== null) {
        throw new Error(`${asset} 未触发时数值必须为空`);
      }
    } else if (![1, -1].includes(s.direction) || !finite(s.magnitude_sigma) || s.magnitude_sigma <= 0 ||
               !finite(s.delay_minutes) || s.delay_minutes < 0 || s.delay_minutes > 30) {
      throw new Error(`${asset} 触发方向、强度或延迟无效`);
    }
  }
}

export function censorSlots(slots, observedMinutes) {
  validateSlots(slots);
  minutes(observedMinutes);
  return Object.fromEntries(ASSETS.map((a) => [a,
    slots[a].present && slots[a].delay_minutes <= observedMinutes ? { ...slots[a] } : blank()]));
}

export function fingerprintDistance(leftSlots, rightSlots) {
  validateSlots(leftSlots); validateSlots(rightSlots);
  let sum = 0;
  for (const asset of ASSETS) {
    const a = leftSlots[asset], b = rightSlots[asset];
    if (!a.present && !b.present) continue;
    if (a.present !== b.present) { sum += 1; continue; }
    const dz = Math.abs(Math.log1p(a.magnitude_sigma) - Math.log1p(b.magnitude_sigma));
    sum += ((a.direction === b.direction ? 0 : 1) + dz / (1 + dz) + Math.abs(a.delay_minutes - b.delay_minutes) / 30) / 3;
  }
  return sum / ASSETS.length;
}

export function findNearest(explorer, eventId, observedMinutes, k = 3) {
  minutes(observedMinutes);
  if (!Number.isInteger(k) || k < 1) throw new Error("返回数量须为正整数");
  if (!explorer || !Array.isArray(explorer.events) || !Array.isArray(explorer.assets) ||
      explorer.assets.join("|") !== ASSETS.join("|")) throw new Error("历史库缺少固定十资产事件集");
  const ids = new Set();
  for (const event of explorer.events) {
    if (!event || typeof event.event_id !== "string" || !event.event_id || ids.has(event.event_id) ||
        typeof event.timestamp !== "string" || !Number.isFinite(Date.parse(event.timestamp))) {
      throw new Error("历史事件标识或时间无效");
    }
    ids.add(event.event_id);
    validateSlots(event.slots);
  }
  const query = explorer.events.find((event) => event.event_id === eventId);
  if (!query) throw new Error("历史库中没有这个事件");
  const prefix = censorSlots(query.slots, observedMinutes);
  return explorer.events.filter((event) => event.event_id !== eventId).map((event) => {
    const candidate = censorSlots(event.slots, observedMinutes);
    return { ...event, prefix: candidate, distance: fingerprintDistance(prefix, candidate) };
  }).sort((a, b) => a.distance - b.distance || order(a.timestamp, b.timestamp) || order(a.event_id, b.event_id)).slice(0, k);
}
