"""C data tests use only in-memory/synthetic temporary artifacts."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lora_c import data as d


def config():
    return {"protocol_version": "lora-c-v1", "assets": d.ASSETS, "train_month": "2026-01", "validation_month": "2026-04",
            "test_month": "2026-05", "lookback": 256, "horizon": 30, "training_candles_per_window": 286,
            "train_stride": 120, "evaluation_stride": 30, "validation_windows": 300, "test_windows": 300,
            "max_epochs": 2, "train_start": "2026-01-01", "train_end_exclusive": "2026-01-26",
            "selection_metric": "small.max_drawdown_mae", "major_drawdown_stop_relative": .2}


def artifact(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def locked_fixture(root, *, scores=(.01,), major_scores=None, original_major=.02):
    major_scores = major_scores or (.02,) * len(scores)
    folder = "lora_c/runs/synthetic"
    candidates = []
    for epoch, (small, major) in enumerate(zip(scores, major_scores), 1):
        name = f"epoch_{epoch:02d}"
        methods = {}
        for tier, candidate_score, original_score in (("small", small, .02), ("major", major, original_major)):
            base = {"n_scheduled": 300, "n_scored": 300, "n_failed": 0, "n_missing": 0, "coverage": 1,
                    "n_max_drawdown_mae_origins": 300, "n_max_drawdown_mae_asset_origins": 300 * (4 if tier == "major" else 6)}
            methods[tier] = {"n_scheduled": 300,
                             "methods": {"original": {**base, "max_drawdown_mae": original_score},
                                         name: {**base, "max_drawdown_mae": candidate_score}},
                             "paired": {name: {"original": {"n_scheduled": 300, "n_paired": 300, "complete_grid": True,
                                         "max_drawdown_mae_difference": {"n_paired": 300, "complete_grid": True,
                                             "candidate_mean": candidate_score, "reference_mean": original_score,
                                             "estimate": candidate_score - original_score}}}}}
        report = artifact(root, folder + f"/april/{name}-report.json", {"protocol_version": "lora-c-v1", "month": "2026-04", "n_scheduled": 300, "tiers": methods})
        candidates.append({"epoch": epoch, "name": name, "small_mae": small, "validation_report": report})
    best = min(candidates, key=lambda row: (row["small_mae"], row["epoch"]))
    selected = folder + "/" + best["name"]
    now = datetime.now(timezone.utc)
    lock = {"status": "locked", "acceptance_authorized": True, "epoch": best["epoch"],
            "locked_at_utc": (now - timedelta(seconds=10)).isoformat(),
            "config": artifact(root, "lora_c/config.json", config()),
            "environment": artifact(root, folder + "/environment.json",
                                    {"shared_code": [artifact(root, "lora_a/data.py", b"# synthetic reused parser\n")]}),
            "selected_checkpoint": artifact(root, selected + "/adapter/adapter_model.safetensors", b"synthetic adapter"),
            "selected_merged_checkpoint": artifact(root, selected + "/merged/model.safetensors", b"synthetic merged model"),
            "validation_report": best["validation_report"],
            "selection_history": artifact(root, folder + "/selection-history.json",
                                          {"rule": "small.max_drawdown_mae", "candidates": candidates, "selected": best,
                                           "planned_epochs": len(scores), "original_small_mae": .02}),
            "calendar_grid_plan": artifact(root, folder + "/calendar-grid-plan.json",
                                            {"grids": {role: {"month": month, "origins": d.calendar_origins(month, role)}
                                                       for month, role in (("2026-01", "train"), ("2026-04", "validation"), ("2026-05", "test"))}}),
            "code": [artifact(root, "lora_c/" + name + ".py", b"# synthetic code\n")
                     for name in ("data", "acquire", "metrics", "training", "runner")]}
    lock_ref = artifact(root, folder + "/selection-lock.json", lock)
    artifact(root, folder + "/root-review.json", {"selection_lock_sha256": lock_ref["sha256"], "passed": True,
                                                 "reviewed_at_utc": (now - timedelta(seconds=5)).isoformat()})
    return lock


class CalendarTests(unittest.TestCase):
    def test_calendar_grids_have_no_filesystem_access_and_exact_counts(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("no calendar IO")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("no calendar IO")):
            self.assertEqual(d.calendar_origins("2026-01"), list(range(255, 35970, 120)))
            for month, rows in (("2026-04", 43200), ("2026-05", 44640)):
                candidates = list(range(255, rows - 30, 30))
                origins = d.calendar_origins(month)
                self.assertEqual(origins, [candidates[i * (len(candidates) - 1) // 299] for i in range(300)])
                self.assertEqual(len(set(origins)), 300)
                self.assertEqual((origins[0], origins[-1]), (candidates[0], candidates[-1]))

    def test_other_months_and_invalid_roles_refused_before_io(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("no IO")):
            for month in ("2026-02", "2026-03", "2026-06", "2026-07", "2026-08", "2025-01"):
                with self.assertRaises(d.DataError):
                    d.load_month(month)
            with self.assertRaises(d.DataError):
                d.load_month("2026-05", role="train")
            with self.assertRaises(d.SealedDataError):
                d.load_month("2026-05")


class LockTests(unittest.TestCase):
    def test_lock_and_review_validated_without_may_directory_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = locked_fixture(root, scores=(.01, .01))
            self.assertTrue(all(Path(record["path"]).is_relative_to(root / "lora_c") for record in lock["code"]))
            self.assertEqual(lock["epoch"], 1)
            self.assertEqual(d.validate_selection_lock(lock, root=root), config())
            cfg, receipt = d._claim_may(lock, root=root)
            self.assertEqual(cfg, config())
            self.assertFalse((root / "lora_c/data").exists())
            d.consume_may_acquisition(lock, receipt, root=root)
            with self.assertRaisesRegex(d.SealedDataError, "already started"):
                d.consume_may_acquisition(lock, receipt, root=root)
            with self.assertRaisesRegex(d.SealedDataError, "already claimed"):
                d._claim_may(lock, root=root)

    def test_shared_parser_change_is_detected_through_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = locked_fixture(root)
            (root / "lora_a/data.py").write_text("# changed parser\n")
            with self.assertRaisesRegex(d.SealedDataError, "SHA256"):
                d.validate_selection_lock(lock, root=root)

    def test_missing_or_stale_review_blocks_claim(self):
        for mode in ("missing", "wrong_hash", "before_lock", "failed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                lock = locked_fixture(root)
                path = Path(lock["selection_history"]["path"]).parent / "root-review.json"
                if mode == "missing":
                    path.unlink()
                else:
                    review = json.loads(path.read_bytes())
                    if mode == "wrong_hash": review["selection_lock_sha256"] = "0" * 64
                    if mode == "before_lock": review["reviewed_at_utc"] = lock["locked_at_utc"]
                    if mode == "failed": review["passed"] = False
                    path.write_text(json.dumps(review))
                with self.assertRaises(d.SealedDataError):
                    d._claim_may(lock, root=root)
                self.assertFalse((root / "lora_c/MAY_UNSEALED.json").exists())
                self.assertFalse((root / "lora_c/data").exists())

    def test_all_epoch_large_guards_and_no_small_improvement_block(self):
        for options in ({"scores": (.02,)}, {"scores": (.03,)}, {"scores": (.01, .009), "major_scores": (.025, .02)},
                        {"scores": (.01,), "major_scores": (.00001,), "original_major": 0}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                lock = locked_fixture(root, **options)
                with self.assertRaises(d.SealedDataError):
                    d._claim_may(lock, root=root)
                self.assertFalse((root / "lora_c/MAY_UNSEALED.json").exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = locked_fixture(root, major_scores=(0,), original_major=0)
            d.validate_selection_lock(lock, root=root)

    def test_missing_coverage_calendar_tamper_and_environment_hash_block(self):
        for mode in ("coverage", "missing_metric", "missing_pair", "calendar", "environment"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                lock = locked_fixture(root)
                if mode == "environment":
                    Path(lock["environment"]["path"]).write_text("changed")
                elif mode == "calendar":
                    path = Path(lock["calendar_grid_plan"]["path"])
                    plan = json.loads(path.read_bytes())
                    plan["grids"]["test"]["origins"][0] += 30
                    lock["calendar_grid_plan"] = artifact(root, str(path.relative_to(root)), plan)
                else:
                    path = Path(lock["validation_report"]["path"])
                    report = json.loads(path.read_bytes())
                    if mode == "coverage":
                        report["tiers"]["major"]["methods"]["original"]["n_scored"] = 299
                    elif mode == "missing_metric":
                        report["tiers"]["small"]["methods"]["original"]["n_max_drawdown_mae_origins"] = 299
                    else:
                        report["tiers"]["small"]["paired"]["epoch_01"]["original"]["max_drawdown_mae_difference"]["n_paired"] = 299
                    ref = artifact(root, str(path.relative_to(root)), report)
                    lock["validation_report"] = ref
                    history_path = Path(lock["selection_history"]["path"])
                    history = json.loads(history_path.read_bytes())
                    history["selected"]["validation_report"] = ref
                    history["candidates"][0]["validation_report"] = ref
                    lock["selection_history"] = artifact(root, str(history_path.relative_to(root)), history)
                with self.assertRaises(d.SealedDataError):
                    d.validate_selection_lock(lock, root=root)

    def test_artifact_cannot_redirect_into_any_market_data_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = locked_fixture(root)
            lock["environment"]["path"] = str(root / "lora_c/data/binance-2026-05/manifest.json")
            with self.assertRaisesRegex(d.SealedDataError, "scope"):
                d.validate_selection_lock(lock, root=root)
            link = root / "lora_c/environment-link.json"
            link.symlink_to(root / "lora_c/data/binance-2026-05/manifest.json")
            lock["environment"]["path"] = str(link)
            with self.assertRaisesRegex(d.SealedDataError, "symlinks"):
                d.validate_selection_lock(lock, root=root)


class WindowTests(unittest.TestCase):
    def test_inputs_do_not_depend_on_future_and_training_does_not_cross_cutoff(self):
        rows = 406
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        raw = np.ones((10, rows, 6), dtype=np.float64)
        raw[:, :, 1] = 2
        raw[0, 255:, 3] = 1.0000000123
        times = [(start + timedelta(minutes=i + 1)).isoformat().replace("+00:00", "Z") for i in range(rows)]
        data = d.MonthData("2026-01", d.ASSETS.copy(), raw.astype(np.float32), np.zeros((rows, 5), np.float32),
                           times, [255, 375], {"role": "train"}, raw)
        before = d.window(data, 255)
        self.assertEqual(before["histories"]["BTC"][-1]["close"], 1.0000000123)
        raw[:, 256:, :] *= 3
        self.assertEqual(before, d.window(data, 255))
        self.assertEqual(len(d.truth(data, 255)["BTC"]["close"]), 30)
        self.assertEqual(d.training_window(data, 375, "BTC")[0].shape, (286, 6))
        self.assertTrue(all(origin + 30 < 36000 for origin in d.calendar_origins("2026-01")))


if __name__ == "__main__":
    unittest.main()
