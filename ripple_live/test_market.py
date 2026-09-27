"""Closed-bar, metric and failure checks; no network, model or stored data."""

from datetime import datetime, timezone
import math
import unittest
from unittest.mock import patch

from . import market


CUTOFF = 1790513580000


def fixture():
    closes = [100.0] * 225 + [100.0, 110.0, 90.0] + [99.0] * 28
    return exchange_rows(closes)


def exchange_rows(closes):
    rows = []
    for index, close in enumerate(closes):
        opened = CUTOFF - 256 * 60_000 + index * 60_000
        rows.append([opened, str(close), str(close), str(close), str(close), "1",
                     opened + 59_999, str(close), 1, "0", "0", "0"])
    return rows


def rows_from_returns(returns, initial=100.0):
    prices = [initial]
    for value in returns:
        prices.append(prices[-1] * math.exp(value))
    return [{"close": price} for price in prices]


def population_std(values):
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))


class MarketTests(unittest.TestCase):
    def fake_request(self, path, params=None):
        if path == "time":
            return {"serverTime": CUTOFF + 12_000}
        self.assertEqual(path, "klines")
        self.assertIn(params["symbol"], [asset + "USDT" for asset in market.ASSETS])
        self.assertEqual(params["endTime"], CUTOFF - 1)
        self.assertEqual(params["interval"], "1m")
        self.assertEqual(params["limit"], 256)
        return fixture()

    def fetch(self, age=15):
        with patch.object(market, "_request_json", side_effect=self.fake_request), patch.object(
                market, "_utcnow", return_value=datetime.fromtimestamp(CUTOFF / 1000 + age, timezone.utc)):
            return market.fetch_market()

    def test_all_ten_exactly_aligned_and_model_input_compatible(self):
        result = self.fetch()
        w, snap = result["window"], result["snapshot"]
        self.assertEqual(w["assets"], list(market.ASSETS))
        for asset in market.ASSETS:
            self.assertEqual(len(w["histories"][asset]), 256)
            self.assertEqual(len(snap["series"][asset]), 61)
            self.assertEqual(w["histories"][asset][-1]["time"], w["as_of"])
        from model_adapter.adapter import validate_input
        _, _, rows = validate_input(w, {"lookback": 256, "horizon": 30, "path_count": 1})
        self.assertEqual(list(rows), list(market.ASSETS))

    def test_metrics_include_all_thirty_returns_and_use_population_variance(self):
        row = self.fetch()["snapshot"]["metrics"][0]
        returns = [math.log(1.1), math.log(90 / 110), math.log(99 / 90)] + [0.0] * 27
        mean = sum(returns) / 30
        expected = math.sqrt(sum((value - mean) ** 2 for value in returns))
        self.assertAlmostEqual(row["vol_past30"], expected, places=13)
        self.assertAlmostEqual(row["mdd_past30"], 20 / 110)
        self.assertAlmostEqual(row["return_past30"], -0.01)
        self.assertEqual(row["last_price"], 99)

    def test_identity_excludes_fetch_time_and_outputs_are_detached(self):
        a, b = self.fetch(age=15), self.fetch(age=190)
        self.assertEqual(a["window"], b["window"])
        self.assertFalse(a["snapshot"]["stale"])
        self.assertTrue(b["snapshot"]["stale"])
        self.assertEqual(b["snapshot"]["age_seconds"], 190)
        a["snapshot"]["series"]["BTC"][0]["close"] = 1
        self.assertEqual(a["window"]["histories"]["BTC"][-61]["close"], 100)

    def test_volatility_addition_preserves_model_identity_and_original_metrics(self):
        result = self.fetch()
        # Captured before the additive volatility-state change with this fixture.
        self.assertEqual(result["window"]["window_id"],
                         "window_c23567c3abec7037c2e68decd70bd6b832e421e71b59fd63db8f9b1ce9f9deff")
        row = result["snapshot"]["metrics"][0]
        self.assertEqual({key: value for key, value in row.items() if key != "volatility_state"},
                         {"asset": "BTC", "vol_past30": 0.2417300597586146,
                          "mdd_past30": 0.18181818181818182, "last_price": 99.0,
                          "return_past30": -0.010000000000000009})
        self.assertNotIn("volatility_state", result["window"])

    def test_volatility_exact_return_counts_population_std_and_shared_endpoint(self):
        # One old extreme is outside the baseline; the newest is in BOTH windows.
        expected_returns = [0.001 * ((index % 7) - 3) for index in range(119)] + [0.027]
        rows = rows_from_returns([0.8] + expected_returns)
        actual = market._volatility_state(rows)
        expected_recent = population_std(expected_returns[-10:])
        expected_baseline = population_std(expected_returns)
        self.assertAlmostEqual(actual["recent_std"], expected_recent, places=14)
        self.assertAlmostEqual(actual["baseline_std"], expected_baseline, places=14)
        self.assertAlmostEqual(actual["ratio"], expected_recent / expected_baseline, places=12)
        self.assertEqual(actual["state"], "above_usual")
        self.assertIsNone(actual["reason"])
        # Exactly 121 closes suffice. Changing older data cannot affect either std.
        self.assertEqual(actual, market._volatility_state(rows[-121:]))
        rows[0]["close"] = 1e50
        self.assertEqual(actual, market._volatility_state(rows))

    def test_volatility_threshold_includes_exactly_one_point_two(self):
        rows = rows_from_returns([0.001, -0.001] * 60)
        # Isolate equality from log/exp rounding; dispersion itself is checked above.
        for ratio, expected in [(math.nextafter(1.2, 0), "near_usual"),
                                (1.2, "above_usual"),
                                (math.nextafter(1.2, math.inf), "above_usual")]:
            with self.subTest(ratio=ratio), patch.object(market, "pstdev", side_effect=[ratio, 1.0]):
                actual = market._volatility_state(rows)
            self.assertEqual(actual["ratio"], ratio)
            self.assertEqual(actual["state"], expected)

    def test_volatility_same_scale_has_ratio_one_and_is_price_scale_invariant(self):
        rows = rows_from_returns([0.003, -0.003] * 60)
        original = market._volatility_state(rows)
        self.assertAlmostEqual(original["ratio"], 1.0, places=12)
        self.assertEqual(original["state"], "near_usual")
        for scale in (0.0001, 10000):
            actual = market._volatility_state([{"close": row["close"] * scale} for row in rows])
            with self.subTest(scale=scale):
                for key in ("recent_std", "baseline_std", "ratio"):
                    self.assertAlmostEqual(actual[key], original[key], places=12)
                self.assertEqual(actual["state"], original["state"])

    def test_volatility_zero_baseline_has_no_fabricated_ratio(self):
        actual = market._volatility_state([{"close": 100.0}] * 121)
        self.assertEqual(actual, {"ratio": None, "state": "unavailable", "recent_std": 0.0,
                                  "baseline_std": 0.0, "reason": "zero_baseline"})

    def test_volatility_insufficient_or_invalid_data_is_explicitly_unavailable(self):
        for count in (0, 10, 120):
            with self.subTest(count=count):
                actual = market._volatility_state([{"close": 100.0}] * count)
                self.assertEqual(actual["reason"], "insufficient_data")
                self.assertIsNone(actual["ratio"])
        for bad in ({}, None, {"close": None}, {"close": True}, {"close": "100"},
                    {"close": 0}, {"close": -1}, {"close": math.nan}, {"close": math.inf}):
            rows = rows_from_returns([0.001, -0.001] * 60)
            rows[50] = bad
            with self.subTest(bad=bad):
                self.assertEqual(market._volatility_state(rows),
                                 {"ratio": None, "state": "unavailable", "recent_std": None,
                                  "baseline_std": None, "reason": "invalid_series"})
        self.assertEqual(market._volatility_state(None)["reason"], "invalid_series")

    def test_ten_assets_have_independent_volatility_states(self):
        asset_returns = {}
        for index, asset in enumerate(market.ASSETS):
            amplitude = 0.00025 + index * 0.0003
            asset_returns[asset] = [0.001, -0.001] * 55 + [amplitude, -amplitude] * 5

        def request(path, params=None):
            if path == "time":
                return {"serverTime": CUTOFF + 12_000}
            asset = params["symbol"][:-4]
            rows = rows_from_returns([0.0] * 135 + asset_returns[asset])
            return exchange_rows([row["close"] for row in rows])

        with patch.object(market, "_request_json", side_effect=request), patch.object(
                market, "_utcnow", return_value=datetime.fromtimestamp(CUTOFF / 1000 + 15, timezone.utc)):
            result = market.fetch_market()
        states = result["snapshot"]["metrics"]
        self.assertEqual(len(states), 10)
        for row in states:
            returns = asset_returns[row["asset"]]
            expected = population_std(returns[-10:]) / population_std(returns)
            self.assertAlmostEqual(row["volatility_state"]["ratio"], expected, places=11)
            self.assertEqual(row["volatility_state"]["state"],
                             "above_usual" if expected >= 1.2 else "near_usual")
        self.assertEqual(len({row["volatility_state"]["ratio"] for row in states}), 10)
        self.assertEqual({row["volatility_state"]["state"] for row in states},
                         {"above_usual", "near_usual"})

    def test_gap_duplicate_future_bar_wrong_shape_and_partial_window_rejected(self):
        mutations = [lambda rows: rows.pop(), lambda rows: rows[20].__setitem__(0, rows[19][0]),
                     lambda rows: rows[-1].__setitem__(0, CUTOFF),
                     lambda rows: rows[2].__setitem__(6, rows[2][6] + 1),
                     lambda rows: rows[3].pop()]
        for mutate in mutations:
            rows = fixture()
            mutate(rows)
            with self.subTest(mutate=mutate), self.assertRaises(market.MarketError):
                market._parse_rows(rows, "BTC", CUTOFF)

    def test_invalid_price_volume_and_nonfinite_rejected(self):
        for field, bad in [(4, "NaN"), (3, "0"), (2, "99"), (5, "-1"), (7, "0"), (1, True)]:
            rows = fixture()
            rows[0][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(market.MarketError):
                market._parse_rows(rows, "BTC", CUTOFF)

    def test_one_bad_asset_rejects_entire_snapshot(self):
        def request(path, params=None):
            if params and params["symbol"] == "LTCUSDT":
                raise market.MarketError("LTC unavailable")
            return self.fake_request(path, params)
        with patch.object(market, "_request_json", side_effect=request), self.assertRaises(market.MarketError):
            market.fetch_market()

    def test_bad_exchange_clock_rejected(self):
        for clock in [None, {}, {"serverTime": True}, {"serverTime": -1}, {"serverTime": "1790513580000"}]:
            with self.subTest(clock=clock), patch.object(market, "_request_json", return_value=clock), self.assertRaises(market.MarketError):
                market.fetch_market()


if __name__ == "__main__":
    unittest.main()
