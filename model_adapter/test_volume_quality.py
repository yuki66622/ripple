import copy
import unittest

from forecast_metrics.engine import freeze_forecast, make_holdings, recalculate
from model_adapter import build_volume_quality, validate_corrections, validate_forecast_output
from model_adapter.corrections import structure_issues
from model_adapter.test_corrections import TIMES, forecast_fixture, raw_fixture
from model_adapter.test_adapter import fixture as input_fixture
from model_adapter.adapter import validate_input


def full_fixture(raw):
    forecast = forecast_fixture(raw)
    forecast["volume_quality"] = build_volume_quality(raw, TIMES)
    return forecast


class VolumeQualityTests(unittest.TestCase):
    def test_negative_values_untouched_and_both_negative_count_once(self):
        raw = raw_fixture()
        raw[0]["assets"]["BTC"].update(volume=[-1, 0], amount=[-100, 0])
        original = copy.deepcopy(raw)
        forecast = full_fixture(raw)
        validate_forecast_output(forecast, raw)
        self.assertEqual(raw, original)
        audit = forecast["volume_quality"]
        self.assertEqual(audit["status"], "volume_forecast_unavailable")
        self.assertEqual(audit["volume_invalid_count"], 1)
        self.assertEqual(audit["volume_invalid_rate"], 0.5)
        self.assertEqual(audit["records"][0]["volume"], -1)
        self.assertEqual(audit["records"][0]["amount"], -100)
        self.assertFalse(audit["records"][0]["volume_valid"])
        self.assertTrue(audit["records"][1]["volume_valid"])

    def test_zero_is_valid_with_complete_zero_rate_audit(self):
        raw = raw_fixture()
        raw[0]["assets"]["BTC"].update(volume=[0, 0], amount=[0, 0])
        forecast = full_fixture(raw)
        validate_forecast_output(forecast)
        audit = forecast["volume_quality"]
        self.assertEqual(audit["status"], "valid")
        self.assertEqual(audit["volume_invalid_count"], 0)
        self.assertEqual(audit["volume_invalid_rate"], 0)
        self.assertEqual(len(audit["records"]), 2)

    def test_nan_inf_missing_and_bad_price_still_reject(self):
        for field, value in (("volume", float("nan")), ("amount", float("inf")),
                             ("volume", -float("inf")), ("open", -1), ("close", 0)):
            raw = raw_fixture()
            raw[0]["assets"]["BTC"][field][0] = value
            self.assertTrue(structure_issues(raw, 2))
            with self.assertRaises(ValueError):
                build_volume_quality(raw, TIMES)
        for mutate in (lambda a: a.pop("volume"), lambda a: a["amount"].pop()):
            raw = raw_fixture()
            mutate(raw[0]["assets"]["BTC"])
            with self.assertRaises(ValueError):
                build_volume_quality(raw, TIMES)

    def test_input_negative_volume_is_still_rejected(self):
        window = input_fixture()
        window["histories"]["BTC"][0]["volume"] = -1
        with self.assertRaises(ValueError):
            validate_input(window, {"lookback": 2})

    def test_full_validator_requires_both_audits_without_migrating_old(self):
        old = forecast_fixture()
        saved = copy.deepcopy(old)
        validate_corrections(old)
        with self.assertRaisesRegex(ValueError, "volume quality audit"):
            validate_forecast_output(old)
        self.assertEqual(old, saved)
        full = full_fixture(raw_fixture())
        del full["ohlc_corrections"]
        with self.assertRaises(ValueError):
            validate_forecast_output(full)

    def test_flag_counts_rates_identity_and_missing_records_tampering_rejected(self):
        mutations = [lambda a: a.__setitem__("volume_invalid_count", 1),
                     lambda a: a.__setitem__("volume_invalid_rate", 0.5),
                     lambda a: a.__setitem__("status", "volume_forecast_unavailable"),
                     lambda a: a["records"][0].__setitem__("volume_valid", False),
                     lambda a: a["records"][0].__setitem__("volume_valid", 1),
                     lambda a: a["records"][0].__setitem__("time", "unknown"),
                     lambda a: a["records"].pop(),
                     lambda a: a["records"].append(copy.deepcopy(a["records"][0]))]
        for mutate in mutations:
            forecast = full_fixture(raw_fixture())
            mutate(forecast["volume_quality"])
            with self.assertRaises(ValueError):
                validate_forecast_output(forecast)

    def test_double_argument_binds_finite_values_to_original(self):
        raw = raw_fixture()
        forecast = full_fixture(raw)
        forecast["volume_quality"]["records"][0]["volume"] = 2
        validate_forecast_output(forecast)
        with self.assertRaisesRegex(ValueError, "raw model output"):
            validate_forecast_output(forecast, raw)

    def test_unused_volume_does_not_change_price_metric_values(self):
        raw = raw_fixture()
        positive = full_fixture(raw)
        raw[0]["assets"]["BTC"].update(volume=[-1, -2], amount=[-10, -20])
        negative = full_fixture(raw)
        results = []
        for forecast in (positive, negative):
            frozen = freeze_forecast(forecast)
            results.append(recalculate(frozen, make_holdings(frozen, 1000, {"BTC": 1})))
        self.assertEqual(positive["paths"], negative["paths"])
        self.assertNotEqual(results[0]["forecast_id"], results[1]["forecast_id"])
        self.assertEqual(results[0]["summary"], results[1]["summary"])


if __name__ == "__main__":
    unittest.main()
