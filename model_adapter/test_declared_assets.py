"""Declared-universe regression checks; fake forecaster, no weights or GPU."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from .adapter import ADAPTER_REVISION, COLUMNS, KronosAdapter, assemble_forecast, stable_seed, validate_input
from .test_adapter import fixture
from .volume_quality import validate_forecast_output

TEN_ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC"]
CONFIG = {"lookback": 2, "horizon": 2, "path_count": 2, "seed": 20260926}


def declared_window(assets=None):
    window = fixture()
    rows = window["histories"]["BTC"]
    window["assets"] = list(TEN_ASSETS if assets is None else assets)
    window["histories"] = {asset: copy.deepcopy(rows) for asset in window["assets"]}
    window["window_id"] = "TEST_ONLY_declared_universe_" + "_".join(window["assets"])
    return window


def raw_fixture(assets, path_count=2):
    return [{"path_id": f"path-{p:03d}", "assets": {asset: {
                "open": [100., 100.], "high": [99., 102.], "low": [101., 99.],
                "close": [100.5, 101.], "volume": [-1., 2.], "amount": [100., 202.],
            } for asset in assets}} for p in range(path_count)]


class FakeForecaster:
    """Synthetic fixed output isolates iteration/seed/interface behavior."""

    def __init__(self):
        self.calls = 0

    def fit(self, frame):
        self.frame = frame
        return self

    def predict(self, fh):
        import pandas as pd
        self.calls += 1
        self.last_index = self.frame.index
        values = {field: [float(self.frame[field].iloc[-1])] * len(fh) for field in COLUMNS}
        return pd.DataFrame(values, index=fh.to_pandas(), columns=COLUMNS)


class FakeRuntimeAdapter(KronosAdapter):
    """No torch import, model loader, checkpoint access or accelerator use."""

    def __init__(self):
        super().__init__(device="cpu")
        import numpy as np
        self.seed_calls = []
        self._np = np
        self._torch = SimpleNamespace(use_deterministic_algorithms=lambda *a: None,
                                      manual_seed=self.seed_calls.append, inference_mode=nullcontext)
        self._forecaster = FakeForecaster()
        self._identity = {"backend": "cpu", "model_revision": "TEST_ONLY_no_model",
                          "adapter_revision": ADAPTER_REVISION}
        self._load_ms = 0.

    def identity(self):
        return dict(self._identity)

    def _synchronize(self):
        pass

    def _load(self, first_frame):
        self.first_loaded_index = first_frame.index
        return 0.


class DeclaredAssetsTests(unittest.TestCase):
    def test_ten_assets_preserve_declared_order_and_input(self):
        window = declared_window()
        before = copy.deepcopy(window)
        _, times, rows = validate_input(window, CONFIG)
        self.assertEqual(list(rows), TEN_ASSETS)
        self.assertEqual(len(times), 2)
        self.assertEqual(window, before)

    def test_one_asset_and_maximum_asset_count_are_valid(self):
        for assets in (["BTC"], [f"TOKEN{i}" for i in range(100)]):
            _, _, rows = validate_input(declared_window(assets), CONFIG)
            self.assertEqual(list(rows), assets)

    def test_empty_duplicate_unordered_and_invalid_symbols_rejected(self):
        for assets in ([], ["BTC", "BTC"], ("BTC",), {"BTC"}, ["btc"], [" BTC"], ["BTC/USD"],
                       [""], [1], ["A" * 21], [f"TOKEN{i}" for i in range(101)]):
            with self.subTest(assets=assets):
                window = fixture()
                window["assets"] = assets
                with self.assertRaisesRegex(ValueError, "ordered list"):
                    validate_input(window, CONFIG)

    def test_histories_must_exactly_match_all_declared_assets(self):
        for mutation in ("missing", "extra", "wrong_type"):
            window = declared_window()
            if mutation == "missing":
                window["histories"].pop("LTC")
            elif mutation == "extra":
                window["histories"]["EXTRA"] = copy.deepcopy(window["histories"]["BTC"])
            else:
                window["histories"] = []
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "complete declared asset set"):
                validate_input(window, CONFIG)

    def test_new_assets_still_require_aligned_complete_finite_positive_prices(self):
        for mutation in ("gap", "nan", "negative_price", "negative_volume", "missing_amount"):
            window = declared_window()
            row = window["histories"]["LTC"][0]
            if mutation == "gap":
                row["time"] = "2025-01-01T00:00:00Z"
            elif mutation == "nan":
                row["close"] = float("nan")
            elif mutation == "negative_price":
                row["low"] = -1
            elif mutation == "negative_volume":
                row["volume"] = -1
            else:
                row.pop("amount")
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                validate_input(window, CONFIG)

    def test_ten_asset_assembly_keeps_raw_and_complete_dual_audit(self):
        window = declared_window()
        raw = raw_fixture(TEN_ASSETS)
        original = copy.deepcopy(raw)
        forecast = assemble_forecast(window, CONFIG, "TEST_ONLY_run", raw, model_revision="TEST_ONLY_no_model")
        validate_forecast_output(forecast, raw)
        self.assertEqual(raw, original)
        self.assertEqual(list(forecast["spots"]), TEN_ASSETS)
        self.assertEqual(forecast["ohlc_corrections"]["total_candles"], 40)
        self.assertEqual(forecast["ohlc_corrections"]["corrected_candles"], 20)
        self.assertEqual(forecast["volume_quality"]["total_candles"], 40)
        self.assertEqual(forecast["volume_quality"]["volume_invalid_count"], 20)
        for path in forecast["paths"]:
            self.assertEqual(list(path["assets"]), TEN_ASSETS)
            for asset in TEN_ASSETS:
                self.assertEqual(path["assets"][asset]["close"], [100.5, 101.])
        self.assertEqual(forecast["pairing"], "paired_scenarios_not_calibrated_joint_distribution")

    def test_missing_new_asset_output_rejects_whole_batch(self):
        raw = raw_fixture(TEN_ASSETS)
        raw[0]["assets"].pop("LTC")
        with self.assertRaisesRegex(ValueError, "wrong asset set"):
            assemble_forecast(declared_window(), CONFIG, "TEST_ONLY_run", raw, model_revision="TEST_ONLY_no_model")

    def test_default_three_asset_audits_remain_unchanged(self):
        assets = ["BTC", "ETH", "SOL"]
        forecast = assemble_forecast(declared_window(assets), CONFIG, "TEST_ONLY_run", raw_fixture(assets),
                                     model_revision="TEST_ONLY_no_model")
        self.assertEqual(list(forecast["spots"]), assets)
        self.assertEqual(forecast["ohlc_corrections"]["total_candles"], 12)
        self.assertEqual(forecast["volume_quality"]["volume_invalid_count"], 6)

    def test_mock_predict_iterates_all_assets_paths_and_preserves_seeds(self):
        for assets in (["BTC", "ETH", "SOL"], TEN_ASSETS, list(reversed(TEN_ASSETS))):
            with self.subTest(assets=assets):
                adapter = FakeRuntimeAdapter()
                window = declared_window(assets)
                result = adapter.predict(window, CONFIG, "TEST_ONLY_run")
                expected_seeds = [stable_seed(CONFIG["seed"], window["window_id"], asset, p)
                                  for p in range(CONFIG["path_count"]) for asset in assets]
                self.assertEqual(adapter.seed_calls, expected_seeds)
                self.assertEqual(adapter._forecaster.calls, len(assets) * CONFIG["path_count"])
                self.assertEqual([c["asset"] for c in result["runtime"]["calls"]], assets * 2)
                self.assertEqual(result["forecast"]["times"], ["2025-01-01T00:03:00Z", "2025-01-01T00:04:00Z"])
                self.assertEqual(adapter.first_loaded_index[0].isoformat(), "2025-01-01T00:00:00+00:00")
                self.assertEqual(result["runtime"]["identity"]["adapter_revision"], ADAPTER_REVISION)
                validate_forecast_output(result["forecast"], result["raw_paths"])

    def test_expanded_window_identity_can_change_existing_asset_seed(self):
        old = declared_window(["BTC", "ETH", "SOL"])
        new = declared_window()
        self.assertNotEqual(stable_seed(CONFIG["seed"], old["window_id"], "BTC", 0),
                            stable_seed(CONFIG["seed"], new["window_id"], "BTC", 0))


if __name__ == "__main__":
    unittest.main()
