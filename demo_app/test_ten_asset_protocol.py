"""The generic model must not weaken the worker's explicit data universe gate."""

import copy
import unittest

from data_pipeline import load_window
from demo_app.worker_stdio import validate_task
from evaluation.runner import _hash


def signed_task(window):
    window = copy.deepcopy(window)
    window["window_id"] = "window_" + _hash({k: v for k, v in window.items() if k != "window_id"})
    task = {"schema_version": 1, "kind": "whole_window_prediction", "fold": 0, "origin_index": 255,
            "window": window, "predict_config": {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926},
            "model_spec": {"model_path": None, "tokenizer_path": None, "model_label": "Kronos-base"}}
    task["task_id"] = "task_" + _hash(task)
    return task


class TenAssetProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ten = load_window("binance_jan2025_10assets")
        cls.three = load_window("binance_jan2025")

    def test_declared_ten_asset_profile_accepted(self):
        validate_task(signed_task(self.ten))

    def test_default_three_profile_still_accepted(self):
        validate_task(signed_task(self.three))

    def test_ten_assets_cannot_masquerade_as_original_profile(self):
        window = copy.deepcopy(self.ten)
        window["profile_id"] = "binance_jan2025"
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_task(signed_task(window))

    def test_missing_seven_assets_cannot_masquerade_as_ten_profile(self):
        window = copy.deepcopy(self.three)
        window["profile_id"] = "binance_jan2025_10assets"
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_task(signed_task(window))

    def test_history_cannot_masquerade_as_live(self):
        window = copy.deepcopy(self.ten)
        window["mode"] = "live"
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_task(signed_task(window))
