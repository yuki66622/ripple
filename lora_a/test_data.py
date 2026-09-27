"""Synthetic-only checks. No real market file, including sealed March, is read."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lora_a import data as d


def _config():
    return {"assets": d.ASSETS, "lookback": 256, "horizon": 30, "stride": 30,
            "train_month": "2026-01", "validation_month": "2026-02", "test_month": "2026-03",
            "training_candles_per_window": 286, "protocol_version": "lora-a-v2.1", "max_epochs": 2,
            "train_stride": 120, "evaluation_stride": 30, "validation_windows": 300, "test_windows": 300,
            "observation_windows_per_day": 8, "train_start": "2026-01-01", "train_end_exclusive": "2026-01-26",
            "observation_start": "2026-01-26", "observation_end_exclusive": "2026-02-01",
            "fixed_small_pair": ["AVAX", "LINK"]}


def _csv(rows=316, month="2026-01"):
    start = int(datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc).timestamp()) * 1_000_000
    records = []
    for i in range(rows):
        opened = start + i * 60_000_000
        records.append(f"{opened},100,102,99,101,2,{opened + 59_999_999},201,5,1,100,0\n")
    return "".join(records).encode()


def _fixture(root, month="2026-01", rows=316):
    (root / "lora_a").mkdir()
    (root / "lora_a/config.json").write_text(json.dumps(_config()))
    directory = root / "research/data-probe" / d.FOLDERS[month]
    directory.mkdir(parents=True)
    raw = _csv(rows, month)
    entries = []
    for asset in d.ASSETS:
        name = f"{asset}USDT-1m-{month}.csv"
        (directory / name).write_bytes(raw)
        entries.append({"asset": asset, "month": month, "pair": asset + "USDT", "status": "verified",
                        "csv_filename": name, "csv_sha256": hashlib.sha256(raw).hexdigest(), "csv_bytes": len(raw),
                        "rows": rows, "official_zip_sha256_verified": True, "zip_crc_verified": True})
    manifest = {"month": month, "status": "verified", "verified_assets": 10, "failed_assets": 0,
                "expected_rows_per_asset": rows, "interval_seconds": 60, "quote_currency": "USDT",
                "assets": list(reversed(d.ASSETS)), "results": entries}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def _memory(rows=406):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    values = np.ones((10, rows, 6), dtype=np.float32)
    values[:, :, 1] = 2
    times = [(start + timedelta(minutes=i + 1)).isoformat().replace("+00:00", "Z") for i in range(rows)]
    return d.MonthData("2026-01", list(d.ASSETS), values, np.zeros((rows, 5), dtype=np.float32), times,
                       list(range(255, rows - 30, 120)), {"role": "train"})


def _locked_fixture(root):
    (root / "lora_a").mkdir()
    def artifact(name, raw):
        path = root / "lora_a" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    method = {"n_scheduled": 300, "n_scored": 300, "n_failed": 0, "volatility_mae": 0.001}
    report_body = {"month": "2026-02", "n_scheduled": 300,
                   "tiers": {tier: {"methods": {"epoch_01": method,
                                                  "original": {**method, "volatility_mae": 0.002}}}
                             for tier in ("major", "small")}}
    report = artifact("february.json", json.dumps(report_body).encode())
    candidate = {"epoch": 1, "name": "epoch_01", "small_mae": 0.001, "validation_report": report}
    return {"status": "locked", "acceptance_authorized": True, "epoch": 1,
            "locked_at_utc": "2026-09-27T00:00:00Z",
            "config": artifact("config.json", json.dumps(_config()).encode()),
            "selected_checkpoint": artifact("epoch_01/adapter/adapter_model.safetensors", b"synthetic checkpoint identity only"),
            "validation_report": report,
            "selection_history": artifact("selection-history.json", json.dumps({"candidates": [candidate], "selected": candidate,
                                                                                  "planned_epochs": 1}).encode()),
            "code": [artifact(name + ".py", b"# synthetic code identity\n")
                     for name in ("data", "metrics", "training", "runner")]}


class SealTests(unittest.TestCase):
    def test_no_mae_improvement_or_incomplete_february_blocks_before_claim(self):
        for change in ("tie", "worse", "major_incomplete", "small_incomplete", "epoch3"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                lock = _locked_fixture(root)
                if change == "epoch3":
                    lock["epoch"] = 3
                else:
                    path = Path(lock["validation_report"]["path"])
                    report = json.loads(path.read_bytes())
                    if change in ("tie", "worse"):
                        report["tiers"]["small"]["methods"]["original"]["volatility_mae"] = 0.001 if change == "tie" else 0.0005
                    else:
                        tier = "major" if change == "major_incomplete" else "small"
                        report["tiers"][tier]["methods"]["original"].update(n_scored=299, n_failed=1)
                    raw = json.dumps(report).encode()
                    path.write_bytes(raw)
                    lock["validation_report"]["sha256"] = hashlib.sha256(raw).hexdigest()
                    history_path = Path(lock["selection_history"]["path"])
                    history = json.loads(history_path.read_bytes())
                    history["selected"]["validation_report"] = lock["validation_report"]
                    history["candidates"][0]["validation_report"] = lock["validation_report"]
                    history_raw = json.dumps(history).encode()
                    history_path.write_bytes(history_raw)
                    lock["selection_history"]["sha256"] = hashlib.sha256(history_raw).hexdigest()
                with self.assertRaises(d.SealedDataError):
                    d._claim_march(lock, root=root)
                self.assertFalse((root / "lora_a/MARCH_UNSEALED.json").exists())

    def test_unfrozen_pair_allowed_for_january_preflight_only(self):
        config = {**_config(), "fixed_small_pair": None}
        d._validate_config(config, allow_unfrozen_pair=True)
        with self.assertRaisesRegex(d.DataError, "pair"):
            d._validate_config(config)

    def test_first_acceptance_claim_is_exclusive_and_repeat_blocks_before_march_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = _locked_fixture(root)
            # Claim helper touches only synthetic artifacts, never a market directory.
            self.assertEqual(d._claim_march(lock, root=root), _config())
            receipt_path = root / "lora_a/MARCH_UNSEALED.json"
            first_receipt = receipt_path.read_bytes()
            receipt = json.loads(first_receipt)
            self.assertEqual(receipt["selected_epoch"], 1)
            canonical = json.dumps(lock, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            self.assertEqual(receipt["selection_lock_canonical_sha256"], hashlib.sha256(canonical).hexdigest())
            original_read = Path.read_bytes
            reads = []
            def allowed_read(path):
                reads.append(path)
                if "research" in path.parts:
                    raise AssertionError("market data path was accessed")
                return original_read(path)
            with patch.object(Path, "read_bytes", allowed_read):
                with self.assertRaisesRegex(d.SealedDataError, "already claimed"):
                    d.load_month("2026-03", root=root, selection_lock=lock)
            self.assertTrue(reads)
            self.assertEqual(receipt_path.read_bytes(), first_receipt)

    def test_selection_history_enforces_minimum_mae_ties_and_epoch_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = _locked_fixture(root)
            history_path = Path(lock["selection_history"]["path"])
            initial = json.loads(history_path.read_bytes())
            first = initial["selected"]
            second = {**first, "epoch": 2, "name": "epoch_02"}
            def save(history):
                raw = json.dumps(history).encode()
                history_path.write_bytes(raw)
                lock["selection_history"]["sha256"] = hashlib.sha256(raw).hexdigest()
            # Equal MAE correctly retains epoch 1.
            save({"candidates": [first, second], "selected": first, "planned_epochs": 2})
            d.validate_selection_lock(lock, root=root)
            # A lower MAE in epoch 2 means epoch 1 cannot remain selected.
            save({"candidates": [first, {**second, "small_mae": 0.0005}], "selected": first, "planned_epochs": 2})
            with self.assertRaisesRegex(d.SealedDataError, "minimum"):
                d.validate_selection_lock(lock, root=root)
            save({"candidates": [first, second], "selected": first, "planned_epochs": 1})
            with self.assertRaisesRegex(d.SealedDataError, "plan"):
                d.validate_selection_lock(lock, root=root)
            save({"candidates": [first, first], "selected": first, "planned_epochs": 2})
            with self.assertRaisesRegex(d.SealedDataError, "plan"):
                d.validate_selection_lock(lock, root=root)

    def test_march_without_lock_refuses_before_any_filesystem_operation(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("unexpected filesystem probe")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected content read")):
            for lock in (None, {}, {"status": "locked"}, {"status": "locked", "acceptance_authorized": False, "epoch": 1}):
                with self.assertRaises(d.SealedDataError):
                    d.load_month("2026-03", selection_lock=lock)

    def test_retired_and_out_of_scope_months_refuse_before_io(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected read")):
            for month in ("2025-01", "2026-04", "2026-09", "../2026-01"):
                with self.assertRaises(d.DataError):
                    d.load_month(month)

    def test_synthetic_lock_validates_unique_artifacts_and_rejects_changes(self):
        # Only lock validation runs; this never invokes a March loader/read.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = _locked_fixture(root)
            self.assertEqual(d.validate_selection_lock(lock, root=root), _config())
            Path(lock["code"][0]["path"]).write_text("# changed code\n")
            with self.assertRaisesRegex(d.SealedDataError, "SHA256"):
                d.validate_selection_lock(lock, root=root)

    def test_missing_code_duplicate_code_and_unauthorized_artifact_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = _locked_fixture(root)
            lock["code"].append(lock["code"][0])
            with self.assertRaisesRegex(d.SealedDataError, "uniquely"):
                d.validate_selection_lock(lock, root=root)
            lock["code"] = lock["code"][:3]
            with self.assertRaisesRegex(d.SealedDataError, "uniquely"):
                d.validate_selection_lock(lock, root=root)
            lock["config"]["path"] = str(root / "elsewhere.json")
            with self.assertRaisesRegex(d.SealedDataError, "declared locations"):
                d.validate_selection_lock(lock, root=root)


class LoadingTests(unittest.TestCase):
    def test_v21_calendar_grids_are_frozen_without_filesystem_access(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("calendar must not probe filesystem")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("calendar must not read data")):
            train = d.calendar_origins("2026-01", "train")
            self.assertEqual(len(train), 298)
            self.assertEqual(train, list(range(255, 35970, 120)))
            self.assertTrue(all(origin - 255 >= 0 and origin + 30 < 36000 for origin in train))
            for month, role, rows in (("2026-02", "validation", 40320), ("2026-03", "test", 44640)):
                actual = d.calendar_origins(month, role)
                candidates = list(range(255, rows - 30, 30))
                expected = [candidates[i * (len(candidates) - 1) // 299] for i in range(300)]
                self.assertEqual(actual, expected)
                self.assertEqual(len(set(actual)), 300)
                self.assertEqual((actual[0], actual[-1]), (candidates[0], candidates[-1]))
                self.assertEqual(actual, d.calendar_origins(month, role))
            observation = d.calendar_origins("2026-01", "observation")
            self.assertEqual(len(observation), 48)
            self.assertEqual(len(set(observation)), 48)
            for day in range(25, 31):
                self.assertEqual(sum(day * 1440 <= origin + 1 < (day + 1) * 1440 for origin in observation), 8)
            self.assertTrue(all(36000 <= origin + 1 and origin + 30 < 44640 for origin in observation))

    def test_invalid_month_role_is_rejected_before_io_or_unseal(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected file read")):
            for month, role in (("2026-03", "train"), ("2026-02", "observation"), ("2026-01", "test")):
                with self.assertRaisesRegex(d.DataError, "role"):
                    d.load_month(month, role=role)

    def test_full_calendar_grid_has_correct_month_boundaries(self):
        self.assertEqual(d._month_grid("2026-01")[1], 44640)
        self.assertEqual(d._month_grid("2026-02")[1], 40320)
        for month, expected_origins in (("2026-01", 1479), ("2026-02", 1335)):
            start, rows = d._month_grid(month)
            origins = list(range(255, rows - 30, 30))
            self.assertEqual(len(origins), expected_origins)
            self.assertGreaterEqual(origins[0] - 255, 0)
            self.assertLess(origins[-1] + 30, rows)
            # The final source candle opens in this month and ends at next-month 00:00.
            self.assertEqual((start + timedelta(minutes=rows - 1)).strftime("%Y-%m"), month)
            self.assertNotEqual((start + timedelta(minutes=rows)).strftime("%Y-%m"), month)

    def test_load_synthetic_month_maintains_config_order_and_start_stamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _fixture(root)
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            # Small synthetic grid tests the loader; full calendar sizing is checked separately.
            with patch.object(d, "_month_grid", return_value=(start, 316)):
                data = d.load_month("2026-01", root=root)
            self.assertEqual(data.assets, d.ASSETS)
            self.assertEqual(data.values.shape, (10, 316, 6))
            self.assertEqual(data.values.dtype, np.float32)
            self.assertEqual(data.raw_values.dtype, np.float64)
            self.assertFalse(data.values.flags.writeable)
            self.assertFalse(data.raw_values.flags.writeable)
            self.assertEqual(data.times[0], "2026-01-01T00:01:00Z")
            self.assertEqual(data.stamps[0].tolist(), [0, 0, 3, 1, 1])
            self.assertEqual(data.stamps[60].tolist(), [0, 1, 3, 1, 1])
            self.assertEqual(data.origins, [255])
            self.assertEqual(data.provenance["role"], "train")
            self.assertEqual(data.provenance["grid"]["origins"], [255])

    def test_csv_hash_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            directory = _fixture(root)
            (directory / "BTCUSDT-1m-2026-01.csv").write_bytes(b"tampered")
            with patch.object(d, "_month_grid", return_value=(datetime(2026, 1, 1, tzinfo=timezone.utc), 316)):
                with self.assertRaisesRegex(d.DataError, "SHA256"):
                    d.load_month("2026-01", root=root)

    def test_missing_rows_gap_and_invalid_ohlc_are_fatal(self):
        raw = _csv(2)
        self.assertEqual(d._parse_csv(raw, "2026-01", expected_rows=2).shape, (2, 6))
        cases = [(raw, 3), (raw + raw.splitlines(keepends=True)[-1], 2),
                 (raw.replace(b",100,102,99,101,", b",100,99,99,101,", 1), 2),
                 (raw.replace(b",100,102,99,101,", b",nan,102,99,101,", 1), 2),
                 (raw.replace(b",2,", b",-2,", 1), 2),
                 (raw.splitlines(keepends=True)[0] * 2, 2)]
        for candidate, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(d.DataError):
                    d._parse_csv(candidate, "2026-01", expected_rows=expected)


class WindowTests(unittest.TestCase):
    def test_january_observation_and_late_targets_cannot_enter_training(self):
        data = _memory(rows=44640)
        # Caller-added origins cannot broaden the frozen training period.
        self.assertIn(36015, data.origins)
        with self.assertRaises(d.DataError):
            d.training_window(data, 36015, "BTC")
        data.provenance["role"] = "observation"
        data.origins = d.calendar_origins("2026-01", "observation")
        self.assertEqual(d.window(data, data.origins[0])["as_of"], "2026-01-26T00:16:00Z")
        with self.assertRaisesRegex(d.DataError, "training role"):
            d.training_window(data, data.origins[0], "BTC")

    def test_scoring_preserves_raw_float64_while_training_uses_float32(self):
        data = _memory()
        data.raw_values = data.values.astype(np.float64)
        exact = 1.0000000123
        data.raw_values[0, 255:286, 3] = exact
        data.values[0, 255:286, 3] = exact
        self.assertEqual(d.window(data, 255)["histories"]["BTC"][-1]["close"], exact)
        self.assertEqual(d.truth(data, 255)["BTC"]["close"][0], exact)
        raw, _ = d.training_window(data, 255, "BTC")
        self.assertEqual(raw.dtype, np.float32)
        self.assertNotEqual(float(raw[255, 3]), exact)
        parsed = d._parse_csv(_csv(1).replace(b",101,2,", b",101.0000000123,2,"), "2026-01", expected_rows=1)
        self.assertEqual(parsed[0, 3], 101.0000000123)

    def test_future_prices_and_provenance_do_not_affect_input_or_identity(self):
        data = _memory()
        before = d.window(data, 255)
        data.values[:, 256:, :] *= 3
        data.provenance["csv_sha256"] = "different future-only file identity"
        self.assertEqual(before, d.window(data, 255))
        self.assertEqual(len(before["histories"]["BTC"]), 256)
        self.assertEqual(before["histories"]["BTC"][-1]["time"], before["as_of"])
        self.assertNotIn("truth", before)
        data.values[0, 255, 3] = 1.5
        self.assertNotEqual(before["window_id"], d.window(data, 255)["window_id"])

    def test_truth_and_training_segment_have_exact_nonoverlapping_bounds(self):
        data = _memory()
        data.values[0, :, 3] = np.arange(406)
        truth = d.truth(data, 255)
        self.assertEqual(truth["BTC"]["close"], list(range(256, 286)))
        raw, stamps = d.training_window(data, 375, "BTC")
        self.assertEqual(raw.shape, (286, 6))
        self.assertEqual(stamps.shape, (286, 5))
        self.assertEqual((raw[0, 3], raw[255, 3], raw[-1, 3]), (120, 375, 405))
        raw[0, 3] = -9
        self.assertEqual(data.values[0, 120, 3], 120)
        for origin in (254, 256, 315, True):
            with self.assertRaises(d.DataError):
                d.window(data, origin)
        data.month = "2026-02"
        with self.assertRaisesRegex(d.DataError, "January"):
            d.training_window(data, 255, "BTC")

    def test_normalization_uses_only_past_256_rows(self):
        raw = np.tile(np.arange(286, dtype=np.float32)[:, None], (1, 6))
        first = d.normalize_training_window(raw)
        raw[256:] = 1e9
        second = d.normalize_training_window(raw)
        np.testing.assert_array_equal(first[:256], second[:256])
        expected = (np.arange(256, dtype=np.float32) - 127.5) / (np.arange(256, dtype=np.float32).std() + 1e-5)
        np.testing.assert_allclose(first[:256, 0], expected)
        self.assertTrue((second[256:] == 5).all())
        self.assertEqual(d.normalize_training_window(np.ones((286, 6), np.float32)).max(), 0)


if __name__ == "__main__":
    unittest.main()
