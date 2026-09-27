import copy
import unittest

from forecast_metrics.engine import freeze_forecast, make_holdings, recalculate
from model_adapter import validate_corrections
from model_adapter.corrections import correct_paths, structure_issues


TIMES = ["2025-01-01T00:01:00Z", "2025-01-01T00:02:00Z"]


def raw_fixture():
    return [{"path_id": "path-000", "seeds": {"BTC": 1}, "assets": {"BTC": {
        "open": [100.0, 101.0], "high": [98.0, 103.0], "low": [102.0, 100.0],
        "close": [101.0, 102.0], "volume": [1.0, 2.0], "amount": [101.0, 204.0]}}}]


def forecast_fixture(raw=None):
    raw = raw_fixture() if raw is None else raw
    corrected, audit = correct_paths(raw, TIMES)
    return {"model_revision": "explicit-test-fixture", "prediction_run_id": "test-1",
            "source": "unit-test", "quote_currency": "USDT",
            "as_of": "2025-01-01T00:00:00Z", "interval_seconds": 60,
            "times": TIMES, "spots": {"BTC": 100.0},
            "pairing": "paired_scenarios_not_calibrated_joint_distribution",
            "paths": [{"path_id": path["path_id"], "assets": {asset: {field: values[field] for field in ("close", "high", "low")} for asset, values in path["assets"].items()}} for path in corrected],
            "ohlc_corrections": audit}


class CorrectionTests(unittest.TestCase):
    def test_pure_inversion_changes_only_bounds_and_preserves_raw(self):
        raw = raw_fixture()
        saved = copy.deepcopy(raw)
        paths, audit = correct_paths(raw, TIMES)
        self.assertEqual(raw, saved)
        for name in ("open", "close", "volume", "amount"):
            self.assertEqual(paths[0]["assets"]["BTC"][name], raw[0]["assets"]["BTC"][name])
        self.assertEqual(paths[0]["assets"]["BTC"]["high"], [102, 103])
        self.assertEqual(paths[0]["assets"]["BTC"]["low"], [98, 100])
        self.assertEqual(audit["corrected_candles"], 1)
        self.assertEqual(audit["total_candles"], 2)
        self.assertEqual(audit["correction_rate"], 0.5)
        self.assertEqual(audit["correction_rate_pct"], 50)
        self.assertEqual(audit["max_adjustment_bps"], 4 / 101 * 10000)
        self.assertFalse(audit["records"][1]["was_corrected"])
        self.assertEqual(audit["records"][1]["max_adjustment_bps"], 0)

    def test_open_and_close_outside_are_both_contained_without_changing_them(self):
        raw = raw_fixture()
        values = raw[0]["assets"]["BTC"]
        values.update(open=[105, 95], close=[104, 96], high=[103, 103], low=[100, 100])
        paths, audit = correct_paths(raw, TIMES)
        result = paths[0]["assets"]["BTC"]
        self.assertEqual(result["high"], [105, 103])
        self.assertEqual(result["low"], [100, 95])
        self.assertEqual(result["open"], [105, 95])
        self.assertEqual(result["close"], [104, 96])
        self.assertEqual(audit["corrected_candles"], 2)

    def test_nan_nonpositive_missing_bar_and_field_rejected_before_correction(self):
        edits = [lambda a: a["close"].__setitem__(0, float("nan")),
                 lambda a: a["low"].__setitem__(0, 0),
                 lambda a: a["open"].pop(), lambda a: a.pop("amount")]
        for edit in edits:
            raw = raw_fixture()
            edit(raw[0]["assets"]["BTC"])
            with self.assertRaises(ValueError):
                correct_paths(raw, TIMES)

    def test_complete_audit_validates_then_metrics_accept_forecast(self):
        raw = raw_fixture()
        forecast = forecast_fixture(raw)
        validate_corrections(forecast)
        validate_corrections(forecast, raw)
        frozen = freeze_forecast(forecast)
        holdings = make_holdings(frozen, capital=1000, weights={"BTC": 1})
        metrics = recalculate(frozen, holdings)
        self.assertEqual(metrics["forecast_id"], frozen.identity)

    def test_metadata_and_published_value_tampering_rejected(self):
        mutations = [lambda f: f["ohlc_corrections"].__setitem__("corrected_candles", 0),
                     lambda f: f["ohlc_corrections"].__setitem__("max_adjustment_bps", 0),
                     lambda f: f["ohlc_corrections"]["records"][0].__setitem__("high_adjustment_bps", 0),
                     lambda f: f["ohlc_corrections"]["records"][0]["corrected"].__setitem__("open", 101),
                     lambda f: f["paths"][0]["assets"]["BTC"]["high"].__setitem__(0, 103),
                     lambda f: f["ohlc_corrections"]["records"].pop(),
                     lambda f: f["ohlc_corrections"]["records"].append(copy.deepcopy(f["ohlc_corrections"]["records"][0]))]
        for mutate in mutations:
            forecast = forecast_fixture()
            mutate(forecast)
            with self.assertRaises(ValueError):
                validate_corrections(forecast)

    def test_double_argument_binds_original_to_raw_data(self):
        raw = raw_fixture()
        forecast = forecast_fixture(raw)
        # Internally consistent altered original, but not what model emitted.
        record = forecast["ohlc_corrections"]["records"][0]
        record["original"]["open"] = record["corrected"]["open"] = 100.5
        validate_corrections(forecast)
        with self.assertRaisesRegex(ValueError, "raw model output"):
            validate_corrections(forecast, raw)

    def test_structural_path_and_asset_identity_rejected(self):
        raw = raw_fixture()
        self.assertTrue(structure_issues(raw, 2, expected_assets=["BTC", "ETH"]))
        self.assertTrue(structure_issues(raw, 2, expected_count=2))
        self.assertTrue(structure_issues(raw + copy.deepcopy(raw), 2))

    def test_audit_participates_in_forecast_id(self):
        forecast = forecast_fixture()
        original_id = freeze_forecast(forecast).identity
        del forecast["ohlc_corrections"]
        self.assertNotEqual(freeze_forecast(forecast).identity, original_id)


if __name__ == "__main__":
    unittest.main()
