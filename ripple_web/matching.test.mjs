import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { ASSETS, censorSlots, fingerprintDistance, findNearest } from "./matching.mjs";

const absent = () => ({ present: false, direction: null, magnitude_sigma: null, delay_minutes: null });
const slot = (direction = 1, magnitude = 6, delay = 0) => ({ present: true, direction, magnitude_sigma: magnitude, delay_minutes: delay });
const slots = (values = {}) => Object.fromEntries(ASSETS.map((a) => [a, values[a] || absent()]));
const event = (id, timestamp, values = {}) => ({ event_id: id, timestamp, slots: slots(values) });
const explorer = (events) => ({ assets: [...ASSETS], events });

test("distance obeys frozen present, direction, log-strength and delay formula", () => {
  const empty = slots(), left = slots({ BTC: slot() });
  assert.equal(fingerprintDistance(empty, empty), 0);
  assert.equal(fingerprintDistance(empty, left), .1);
  assert.equal(fingerprintDistance(left, left), 0);
  assert.ok(Math.abs(fingerprintDistance(left, slots({ BTC: slot(-1, 6, 15) })) - .05) < 1e-14);
  const dz = Math.abs(Math.log1p(6) - Math.log1p(12));
  assert.ok(Math.abs(fingerprintDistance(left, slots({ BTC: slot(1, 12) })) - dz / (1 + dz) / 30) < 1e-14);
  assert.equal(fingerprintDistance(empty, left), fingerprintDistance(left, empty));
});

test("censor is inclusive at the observed minute and never mutates the frozen vector", () => {
  const original = slots({ BTC: slot(), ETH: slot(-1, 8, 10), SOL: slot(1, 9, 11) });
  const saved = structuredClone(original), prefix = censorSlots(original, 10);
  assert.equal(prefix.ETH.present, true);
  assert.deepEqual(prefix.SOL, absent());
  prefix.BTC.magnitude_sigma = 100;
  assert.deepEqual(original, saved);
});

test("both sides are censored, self is excluded, and prefix changes rankings", () => {
  const data = explorer([
    event("query", "2026-01-01T00:00:00Z", { BTC: slot(), ETH: slot(1, 6, 20) }),
    event("earlier", "2026-01-02T00:00:00Z", { BTC: slot(), ETH: slot(-1, 6, 15) }),
    event("later", "2026-01-03T00:00:00Z", { BTC: slot() }),
  ]);
  const before = structuredClone(data);
  const ten = findNearest(data, "query", 10);
  assert.deepEqual(ten.map((e) => [e.event_id, e.distance]), [["earlier", 0], ["later", 0]]);
  assert.equal(ten[0].prefix.ETH.present, false);
  assert.equal(findNearest(data, "query", 16)[0].event_id, "later");
  assert.equal(findNearest(data, "query", 30)[0].event_id, "earlier");
  assert.deepEqual(data, before);
});

test("ties use timestamp then ID, never input order, outcomes, or class", () => {
  const query = event("query", "2026-01-01T00:00:00Z", { BTC: slot() });
  const a = event("a", "2026-01-02T00:00:00Z", { BTC: slot() });
  const b = event("b", "2026-01-02T00:00:00Z", { BTC: slot() });
  a.actual_max_mdd = 1; a.class_id = "different";
  b.actual_max_mdd = 0; b.class_id = "same";
  assert.deepEqual(findNearest(explorer([b, query, a]), "query", 0).map((e) => e.event_id), ["a", "b"]);
  assert.equal(findNearest(explorer([query, a, b]), "query", 0, 1).length, 1);
});

test("invalid inputs fail explicitly rather than producing fake matches", () => {
  assert.throws(() => censorSlots(slots(), NaN));
  assert.throws(() => censorSlots(slots(), -1));
  assert.throws(() => censorSlots(slots(), 31));
  assert.throws(() => censorSlots({ BTC: slot() }, 10));
  assert.throws(() => censorSlots(slots({ BTC: slot(0) }), 10));
  assert.throws(() => censorSlots(slots({ BTC: slot(1, Infinity) }), 10));
  assert.throws(() => censorSlots(slots({ BTC: slot(1, 6, 31) }), 10));
  assert.throws(() => censorSlots(slots({ BTC: { ...absent(), magnitude_sigma: 0 } }), 10));
  const q = event("q", "2026-01-01T00:00:00Z");
  assert.throws(() => findNearest(explorer([q, q]), "q", 10));
  assert.throws(() => findNearest(explorer([q]), "missing", 10));
  assert.throws(() => findNearest(explorer([q]), "q", 10, 0));
  assert.deepEqual(findNearest(explorer([q]), "q", 10), []);
});

const savedData = new URL("./data/explorer.json", import.meta.url);
test("saved January example reproduces all three frozen matches and distances", { skip: !existsSync(savedData) }, () => {
  const data = JSON.parse(readFileSync(savedData, "utf8"));
  assert.equal(data.events.length, 84);
  const found = findNearest(data, data.demo.event_id, data.demo.observed_minutes);
  assert.deepEqual(found.map((e) => e.event_id), data.demo.expected_matches.map((e) => e.event_id));
  for (let i = 0; i < found.length; i++) {
    assert.ok(Math.abs(found[i].distance - data.demo.expected_matches[i].distance) < 1e-12);
    assert.ok(Object.values(found[i].prefix).every((s) => !s.present || s.delay_minutes <= data.demo.observed_minutes));
  }
});
