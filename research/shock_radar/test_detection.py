"""Synthetic detector tests only; no data files, outputs, model or network."""

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

import numpy as np

from research.shock_radar import detection as d


def times(n):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [(start + timedelta(minutes=i)).isoformat().replace("+00:00", "Z") for i in range(n)]


def feature_fixture(candidates, assets=("A", "B", "C"), n=400):
    """Explicit signal fixtures isolate cooldown/merging from return arithmetic."""
    stamps = times(n)
    parsed = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps]
    closes = np.ones((n, len(assets)))
    r5, sigma = np.full_like(closes, np.nan), np.full_like(closes, np.nan)
    r5[65:], sigma[65:] = 0, 1
    for index, asset, signed_z in candidates:
        r5[index, assets.index(asset)] = signed_z
    return closes, stamps, list(assets), parsed, r5, sigma


class SignalTests(unittest.TestCase):
    def test_sigma_is_population_std_of_exact_lagged_slice(self):
        n = 180
        prices = np.exp(np.cumsum(.001 * np.sin(np.arange(n))))[:, None]
        prices[95:] *= 1.5
        *_, r5, sigma = d._prepare(prices, times(n), ["A"])
        expected = np.std(r5[35:95, 0], ddof=0)
        self.assertEqual(sigma[95, 0], expected)
        self.assertNotAlmostEqual(sigma[95, 0], np.std(r5[36:96, 0], ddof=0))
        self.assertTrue(np.isnan(sigma[:65]).all())
        self.assertEqual(sigma[65, 0], np.std(r5[5:65, 0], ddof=0))

    def test_future_suffix_cannot_change_past_triggers_or_finalized_episodes(self):
        n, cutoff = 400, 240
        prices = np.exp(np.cumsum(.0002 * np.sin(np.arange(n))))[:, None]
        prices[100:] *= 1.05
        prices[180:] *= .94
        original = d.detect_events(prices, times(n), ["A"], 3)
        changed = prices.copy()
        changed[cutoff:] *= np.exp(np.linspace(.4, 1, n - cutoff))[:, None]
        other = d.detect_events(changed, times(n), ["A"], 3)
        prefix = d.detect_events(prices[:cutoff], times(cutoff), ["A"], 3)
        past = lambda result: [t for t in result["triggers"] if t["index"] < cutoff]
        self.assertEqual(past(original), past(other))
        self.assertEqual(past(original), prefix["triggers"])
        # Eligibility needs another 30 closes; compare only settled prefix.
        settled = lambda result: [e for e in result["events"] if e["index"] + 30 < cutoff]
        self.assertEqual(settled(original), settled(prefix))

    def test_zero_sigma_is_counted_undefined_without_epsilon_or_infinite_z(self):
        result = d.detect_events(np.ones((100, 2)), times(100), ["A", "B"], 3)
        self.assertEqual(result["stats"]["zero_sigma_count"], 70)
        self.assertEqual(result["stats"]["raw_candidate_count"], 0)
        self.assertEqual(result["events"], [])

    def test_strict_threshold_and_cooldown_end_is_inclusive_eligible(self):
        candidates = [(t, "A", 7) for t in (65, 66, 94, 95, 96, 124, 125)]
        candidates.append((190, "B", 3))  # Equality is not a crossing.
        result = d._detect(feature_fixture(candidates), 3)
        self.assertEqual([t["index"] for t in result["triggers"]], [65, 95, 125])
        self.assertEqual(result["stats"]["raw_candidate_count"], 7)
        self.assertEqual(result["stats"]["suppressed_candidate_count"], 4)

    def test_fixed_ten_minute_merge_does_not_chain(self):
        result = d._detect(feature_fixture([(260, "A", 7), (270, "B", 8), (280, "C", 9)]), 3)
        self.assertEqual([e["index"] for e in result["events"]], [260, 280])
        self.assertEqual([t["index"] for t in result["events"][0]["member_triggers"]], [260, 270])
        self.assertEqual(result["events"][0]["magnitude_sigma"], 7)

    def test_simultaneous_origins_never_choose_a_fake_unique_source(self):
        result = d._detect(feature_fixture([(260, "A", 7), (260, "B", -8), (260, "C", 9)]), 3)
        event = result["events"][0]
        self.assertIsNone(event["origin_asset"])
        self.assertEqual(event["origin_assets"], ["A", "B", "C"])
        self.assertEqual(event["direction"], "mixed")
        self.assertEqual(event["magnitude_sigma"], 9)
        same = d._detect(feature_fixture([(260, "A", -7), (260, "B", -8)]), 3)["events"][0]
        self.assertEqual(same["direction"], -1)

    def test_systemic_is_posthoc_and_future_magnitude_cannot_move_anchor(self):
        result = d._detect(feature_fixture([(260, "A", 7), (269, "B", 100), (270, "C", 20)]), 3)
        event = result["events"][0]
        self.assertEqual(event["index"], 260)
        self.assertEqual(event["origin_asset"], "A")
        self.assertEqual(event["magnitude_sigma"], 7)
        self.assertTrue(event["systemic_flag"])
        self.assertEqual(event["systemic_asset_count"], 3)
        self.assertEqual(event["systemic_confirmed_at"], times(400)[270])
        partial = d._detect(feature_fixture([(260, "A", 7)], n=265), 3)["events"][0]
        self.assertIsNone(partial["systemic_flag"])
        self.assertIsNone(partial["systemic_confirmed_at"])
        self.assertFalse(partial["systemic_confirmation_complete"])
        self.assertEqual(partial["event_id"], event["event_id"])

    def test_systemic_counts_prior_episode_triggers_but_not_suppressed_candidates(self):
        result = d._detect(feature_fixture([(260, "A", 7), (270, "B", 8), (271, "C", 9), (279, "D", 10)],
                                           assets=("A", "B", "C", "D")), 3)
        second = result["events"][1]
        self.assertEqual(second["index"], 271)
        self.assertEqual(len(second["member_triggers"]), 2)
        self.assertEqual(second["systemic_asset_count"], 3)  # Includes B at 270.
        suppressed = d._detect(feature_fixture([(240, "B", 7), (255, "B", 9), (260, "A", 7), (270, "C", 8)]), 3)
        event = next(e for e in suppressed["events"] if e["index"] == 260)
        self.assertEqual(event["systemic_asset_count"], 2)
        self.assertFalse(event["systemic_flag"])

    def test_eligibility_is_only_history_and_future_timestamp_availability(self):
        for index, eligible in ((254, False), (255, True), (369, True), (370, False)):
            event = d._detect(feature_fixture([(index, "A", 7)], n=400), 3)["events"][0]
            self.assertEqual(event["eligible"], eligible)
            self.assertEqual(bool(event["exclusion_reasons"]), not eligible)

    def test_frozen_grid_rejects_expansion_and_bad_inputs(self):
        for k in (2, 7, True, None):
            with self.assertRaises(ValueError):
                d.detect_events(np.ones((100, 1)), times(100), ["A"], k)
        with self.assertRaises(ValueError):
            d.detect_events(np.ones((100, 2)), times(100), ["A", "A"], 3)
        broken = times(100)
        broken[70] = broken[69]
        with self.assertRaises(ValueError):
            d.detect_events(np.ones((100, 1)), broken, ["A"], 3)
        with self.assertRaises(ValueError):
            d.detect_events(np.zeros((100, 1)), times(100), ["A"], 3)


class CalibrationTests(unittest.TestCase):
    def test_no_feasible_threshold_returns_null_and_all_catalogs(self):
        result = d.calibrate(np.ones((100, 2)), times(100), ["A", "B"])
        self.assertIsNone(result["selected_k"])
        self.assertEqual([r["k"] for r in result["grid"]], [3, 4, 5, 6])
        self.assertEqual(set(result["catalogs"]), {"3", "4", "5", "6"})

    def test_calibration_uses_eligible_counts_and_larger_k_on_tie(self):
        counts = {3: 30, 4: 50, 5: 19, 6: 61}

        def fake_detect(prepared, k):
            return {"k": k, "stats": {"eligible_event_count": counts[k], "merged_event_count": 40}, "events": [], "triggers": []}

        with patch.object(d, "_detect", side_effect=fake_detect):
            result = d.calibrate(np.ones((100, 1)), times(100), ["A"])
        self.assertEqual(result["selected_k"], 4)


if __name__ == "__main__":
    unittest.main()
