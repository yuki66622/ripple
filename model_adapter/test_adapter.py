"""Cheap contract checks; actual model evidence is produced by benchmark.py."""
import copy
import unittest

from .adapter import stable_seed, validate_input, validate_output


def fixture():
    return {"schema_version": 1, "assets": ["BTC", "ETH", "SOL"],
            "interval_seconds": 60, "window_id": "test-window", "source": "explicit-test-fixture",
            "quote_currency": "USDT", "as_of": "2025-01-01T00:02:00Z",
            "histories": {asset: [
                {"time": f"2025-01-01T00:0{i}:00Z", "open": 100, "high": 102,
                 "low": 99, "close": 101, "volume": 2, "amount": 201}
                for i in (1, 2)] for asset in ("BTC", "ETH", "SOL")}}


class AdapterContractTests(unittest.TestCase):
    def test_input_is_not_mutated_and_no_future_rows(self):
        window = fixture()
        saved = copy.deepcopy(window)
        cfg, times, rows = validate_input(window, {"lookback": 2, "path_count": 16})
        self.assertEqual(window, saved)
        self.assertEqual(len(times), 2)
        self.assertEqual(cfg["path_count"], 16)
        self.assertEqual(rows["BTC"][-1]["amount"], 201)

    def test_amount_must_be_present_not_estimated(self):
        window = fixture()
        del window["histories"]["BTC"][0]["amount"]
        with self.assertRaises(KeyError):
            validate_input(window, {"lookback": 2})

    def test_gaps_rejected(self):
        window = fixture()
        window["histories"]["BTC"][0]["time"] = "2025-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "minute gaps"):
            validate_input(window, {"lookback": 2})

    def test_invalid_bounds_and_fractional_path_count_rejected(self):
        with self.assertRaises(ValueError):
            validate_input(fixture(), {"lookback": 2, "path_count": 1.5})
        window = fixture()
        window["histories"]["BTC"][0]["low"] = 102
        with self.assertRaisesRegex(ValueError, "inconsistent OHLC"):
            validate_input(window, {"lookback": 2})

    def test_seed_is_run_and_assignment_independent_but_path_specific(self):
        first = stable_seed(123, "window-a", "SOL", 0)
        self.assertEqual(first, stable_seed(123, "window-a", "SOL", 0))
        others = [stable_seed(123, w, a, p) for w, a, p in
                  (("window-b", "SOL", 0), ("window-a", "ETH", 0), ("window-a", "SOL", 1))]
        self.assertNotIn(first, others)

    def test_all_raw_output_problems_are_reported_without_repair(self):
        raw = [{"path_id": "path-000", "assets": {"BTC": {
            "open": [100, 100], "high": [102, 98], "low": [99, 99],
            "close": [101, 100], "volume": [1, -1], "amount": [100, 100]}}}]
        saved = copy.deepcopy(raw)
        issues = validate_output(raw, 2)
        self.assertEqual(raw, saved)
        self.assertIn("inconsistent OHLC", issues[0])
        self.assertNotIn("negative volume/amount", " ".join(issues))


if __name__ == "__main__":
    unittest.main()
