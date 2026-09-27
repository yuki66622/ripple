import copy
import json
import unittest

from research.shock_radar.deep_checks import permutation as p


def trigger(asset, index, direction=1, magnitude=6.5):
    return {"asset": asset, "index": index, "timestamp": p.TIMES[index],
            "direction": direction, "magnitude_sigma": magnitude}


class TemporalNullTests(unittest.TestCase):
    def test_tail_ties_plus_one(self):
        self.assertEqual(p.tail_test(3, [1, 3, 3, 4], "upper")["monte_carlo_p"], 4 / 5)
        self.assertEqual(p.tail_test(3, [1, 3, 3, 4], "lower")["monte_carlo_p"], 4 / 5)
        self.assertEqual(p.tail_test(5, [1, 2, 3, 4], "upper")["monte_carlo_p"], 1 / 5)

    def test_empty_delays_stay_in_denominator(self):
        result = p.tail_test(2, [None, None, 1], "lower")
        self.assertEqual(result["monte_carlo_p"], 2 / 4)
        self.assertEqual(result["null_empty_count"], 2)
        self.assertEqual(result["replicates"], 3)
        result = p.tail_test(2, [None] * 4, "lower")
        self.assertEqual(result["monte_carlo_p"], .2)
        self.assertEqual(result["null_p05_p50_p95_finite"], [None] * 3)
        with self.assertRaises(ValueError):
            p.tail_test(2, [], "lower")

    def test_holm_step_down(self):
        self.assertEqual(p.holm_adjust({"a": .005, "b": .03}), {"a": .01, "b": .03})
        self.assertEqual(p.holm_adjust({"a": .04, "b": .041}), {"a": .08, "b": .08})
        self.assertEqual(p.holm_adjust({"a": .004, "b": .004}), {"a": .008, "b": .008})

    def test_shift_preserves_marks_and_cyclic_gaps(self):
        original = [trigger("BTC", 10), trigger("BTC", 110, -1, 7.2), trigger("BTC", 44600)]
        saved = copy.deepcopy(original)
        offsets = dict.fromkeys(p.ASSETS, 0)
        offsets["BTC"] = 100
        shifted = p.shift_triggers(original, offsets)
        self.assertEqual(original, saved)
        self.assertEqual([x["index"] for x in shifted], [60, 110, 210])
        def gaps(seq):
            xs = sorted(t["index"] for t in seq)
            return sorted((xs[(i + 1) % len(xs)] - x) % p.PERIOD for i, x in enumerate(xs))
        self.assertEqual(gaps(original), gaps(shifted))
        self.assertEqual(sorted((t["direction"], t["magnitude_sigma"]) for t in original),
                         sorted((t["direction"], t["magnitude_sigma"]) for t in shifted))

    def test_original_merge_and_graph_boundaries(self):
        # Exact eligibility boundary 255 and future boundary N-31.
        raw = [trigger("BTC", 254), trigger("ETH", 265), trigger("SOL", 265),
               trigger("BNB", 266), trigger("XRP", 275), trigger("DOGE", 276),
               trigger("ADA", 295), trigger("AVAX", 296),
               trigger("LINK", p.PERIOD - 31), trigger("LTC", p.PERIOD - 20)]
        raw.sort(key=lambda t: (t["index"], p.ASSETS.index(t["asset"])))
        events, graph = p.reconstruct(raw)
        self.assertFalse(events[0]["eligible"])
        first = next(e for e in events if e["index"] == 265)
        self.assertEqual(first["origin_assets"], ["ETH", "SOL"])
        self.assertEqual([t["index"] for t in first["member_triggers"]], [265, 265, 266, 275])
        obs = {(e["source"], e["target"]): e for e in graph["edges"]}
        self.assertNotIn(("ETH", "SOL"), obs)
        self.assertEqual(obs["ETH", "ADA"]["observations"][0]["delay_minutes"], 30)
        self.assertNotIn(("ETH", "AVAX"), obs)
        self.assertTrue(next(e for e in events if e["index"] == p.PERIOD - 31)["eligible"])
        self.assertFalse(events[-1]["eligible"])

    def test_zero_shift_exact_frozen_catalog_graph(self):
        catalog, frozen = json.loads(p.CATALOG.read_text()), json.loads(p.GRAPH.read_text())
        events, graph = p.reconstruct(p.shift_triggers(catalog["triggers"], dict.fromkeys(p.ASSETS, 0)))
        self.assertEqual(events, catalog["events"])
        self.assertEqual(graph, {k: v for k, v in frozen.items() if k != "provenance"})
        stats = p.statistics(events, graph)
        self.assertEqual((stats["eligible_event_count"], stats["nonzero_directed_pair_count"],
                          stats["observation_count"]), (84, 68, 120))


if __name__ == "__main__":
    unittest.main()
