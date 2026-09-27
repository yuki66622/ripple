"""Data boundaries are tested without network/model dependencies."""

import copy
import io
import json
import unittest
from unittest.mock import patch

from data_pipeline import DataError, load_history, load_truth, load_window, window_from_history
from data_pipeline import pipeline as p


def fixture_history(count=40):
    start = 1735689600
    return {
        "schema_version": 1, "profile_id": "binance_jan2025", "mode": "replay",
        "source": "Binance Spot monthly klines 2025-01", "quote_currency": "USDT",
        "assets": list(p.ASSETS), "interval_seconds": 60,
        "histories": {s: [p._bar(start + (n + 1) * 60, [10, 11, 9, 10, 2, 20])
                          for n in range(count)] for s in p.ASSETS},
        "quality": {"amount_source": "exchange_quote_asset_volume", "amount_is_estimate": False},
    }


class WindowTests(unittest.TestCase):
    def test_default_reserves_truth_and_does_not_leak(self):
        history = fixture_history()
        with patch.object(p, "_history", return_value=history):
            window = load_window(lookback=5)
            truth = load_truth(window)
        self.assertEqual(window["as_of"], "2025-01-01T00:10:00Z")
        self.assertEqual(truth["times"][0], "2025-01-01T00:11:00Z")
        self.assertEqual(truth["times"][-1], "2025-01-01T00:40:00Z")
        self.assertEqual(len(truth["times"]), 30)
        self.assertNotIn("truth", window)
        self.assertEqual(len(window["histories"]["BTC"]), 5)

    def test_canonical_times_and_stable_identity(self):
        history = fixture_history()
        a = window_from_history(history, lookback=5, as_of="2025-01-01T00:10:00Z")
        b = window_from_history(history, lookback=5, as_of="2025-01-01T00:10:00+00:00")
        self.assertEqual(a["window_id"], b["window_id"])
        b["histories"]["BTC"][0]["close"] = 9.5
        self.assertEqual(history["histories"]["BTC"][5]["close"], 10)
        with patch.object(p, "_history", return_value=history):
            with self.assertRaisesRegex(DataError, "window_id"):
                load_truth(b)

    def test_rejects_missing_duplicate_unaligned_and_unknown_assets(self):
        for mutation in ("gap", "duplicate", "unaligned", "asset"):
            with self.subTest(mutation=mutation):
                history = fixture_history()
                if mutation == "gap":
                    history["histories"]["BTC"].pop(4)
                elif mutation == "duplicate":
                    history["histories"]["BTC"][5] = history["histories"]["BTC"][4]
                elif mutation == "unaligned":
                    history["histories"]["ETH"] = history["histories"]["ETH"][1:]
                else:
                    history["histories"]["DOGE"] = history["histories"].pop("SOL")
                with self.assertRaises(DataError):
                    window_from_history(history, lookback=5)

    def test_rejects_non_exact_as_of_and_insufficient_context(self):
        history = fixture_history()
        invalid = ["2025-01-01T00:10:30Z", "2025-01-01T00:10:00", "2025-01-01T01:10:00+01:00",
                   "2025-01-01T00:50:00Z", "2025-01-01T00:02:00Z"]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(DataError):
                window_from_history(history, lookback=5, as_of=value)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(DataError):
                window_from_history(history, lookback=value)

    def test_partial_truth_returns_none_without_padding(self):
        history = fixture_history()
        window = window_from_history(history, lookback=5, as_of="2025-01-01T00:20:00Z")
        with patch.object(p, "_history", return_value=history):
            self.assertIsNone(load_truth(window, horizon=30))

    def test_quote_currency_and_source_cannot_mix(self):
        history = fixture_history()
        window = window_from_history(history, lookback=5)
        for key, value in (("quote_currency", "USD"), ("source", "Other exchange")):
            changed = copy.deepcopy(history)
            changed[key] = value
            with patch.object(p, "_history", return_value=changed):
                with self.assertRaisesRegex(DataError, "provenance mismatch"):
                    load_truth(window)

    def test_invalid_candles_fail(self):
        for values in ([10, 9, 8, 10, 2, 20], [10, 11, 9, float("nan"), 2, 20],
                       [10, 11, 9, 10, -1, 20], [10, 11, 9, 10, 0, 20]):
            with self.assertRaises(DataError):
                p._bar(1735689660, values)

    def test_binance_microsecond_close_and_quote_amount(self):
        raw = ["1735689600000000", "10", "11", "9", "10", "2",
               "1735689659999999", "20.125", "3", "1", "10", "0"]
        result = p._parse_binance(iter([raw]), "BTC")
        self.assertEqual(result[0]["time"], "2025-01-01T00:01:00Z")
        self.assertEqual(result[0]["amount"], 20.125)
        raw[6] = "1735689660000000"
        with self.assertRaises(DataError):
            p._parse_binance(iter([raw]), "BTC")

    def test_kraken_excludes_last_and_future_candle(self):
        raw = [[1735689600 + n * 60, "10", "11", "9", "10", "10.25", "2", 2] for n in range(4)]
        rows = p._parse_kraken(raw, "BTC", 1735689720)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["amount"], 20.5)
        with patch.object(p, "_request_kraken", return_value=raw):
            live = p._kraken_history()
        self.assertTrue(live["quality"]["amount_is_estimate"])
        self.assertEqual(live["quote_currency"], "USD")

    def test_kraken_wrong_asset_or_api_error_rejected(self):
        for payload in ({"error": ["EAPI:failure"]}, {"error": [], "result": {"ETHUSD": [], "last": 1}}):
            with patch.object(p, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
                with self.assertRaises(DataError):
                    p._request_kraken("BTC")


class RealArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = load_history()

    def test_downloaded_archive_coverage(self):
        for symbol in p.ASSETS:
            rows = self.history["histories"][symbol]
            self.assertEqual(len(rows), 44640)
            self.assertEqual(rows[0]["time"], "2025-01-01T00:01:00Z")
            self.assertEqual(rows[-1]["time"], "2025-02-01T00:00:00Z")

    def test_real_default_window_truth_and_mutation_isolation(self):
        window = load_window()
        truth = load_truth(window)
        self.assertEqual(window["as_of"], "2025-01-31T23:30:00Z")
        self.assertEqual(len(window["histories"]["BTC"]), 256)
        self.assertEqual(len(truth["times"]), 30)
        self.assertEqual(truth["times"][-1], "2025-02-01T00:00:00Z")
        window["histories"]["BTC"][0]["close"] = 1
        self.assertNotEqual(load_window()["histories"]["BTC"][0]["close"], 1)


if __name__ == "__main__":
    unittest.main()
