"""Ten-asset profile checks: synthetic boundary cases plus real local archives."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from data_pipeline import DataError, load_history, load_truth, load_window, profile_assets, window_from_history
from data_pipeline import pipeline as p
from data_pipeline.test_pipeline import fixture_history
from research.examples.ten_assets_data import verify_archive


def ten_fixture():
    history = fixture_history()
    history["profile_id"] = "binance_jan2025_10assets"
    history["assets"] = list(p.TEN_ASSETS)
    history["histories"] = {s: copy.deepcopy(history["histories"]["BTC"]) for s in p.TEN_ASSETS}
    return history


class TenAssetBoundaryTests(unittest.TestCase):
    def test_lightweight_frozen_profiles_and_unknowns(self):
        with patch.object(p, "_history", side_effect=AssertionError("profile lookup must not read data")):
            self.assertEqual(profile_assets("binance_jan2025_10assets"), p.TEN_ASSETS)
            self.assertEqual(profile_assets("binance_jan2025"), p.ASSETS)
            self.assertEqual(profile_assets("kraken_live"), p.ASSETS)
            for profile in ("unknown", "", None, [], {}):
                with self.subTest(profile=profile), self.assertRaises(DataError):
                    profile_assets(profile)

    def test_duplicate_missing_wrong_order_and_foreign_profile_assets_rejected(self):
        for mutation in ("duplicate", "missing", "order", "foreign_profile", "unknown_profile"):
            history = ten_fixture()
            if mutation == "duplicate":
                history["assets"][-1] = "BTC"
            elif mutation == "missing":
                history["histories"].pop("LTC")
            elif mutation == "order":
                history["assets"][3:5] = ["XRP", "BNB"]
            elif mutation == "foreign_profile":
                history["profile_id"] = "binance_jan2025"
            else:
                history["profile_id"] = "unknown"
            with self.subTest(mutation=mutation), self.assertRaises(DataError):
                window_from_history(history, lookback=5)

    def test_new_assets_gaps_duplicates_and_alignment_are_validated(self):
        for mutation in ("gap", "duplicate", "alignment"):
            history = ten_fixture()
            if mutation == "gap":
                history["histories"]["LINK"].pop(4)
            elif mutation == "duplicate":
                history["histories"]["LTC"][5] = history["histories"]["LTC"][4]
            else:
                history["histories"]["BNB"] = history["histories"]["BNB"][1:]
            with self.subTest(mutation=mutation), self.assertRaises(DataError):
                window_from_history(history, lookback=5)

    def test_ten_asset_truth_is_separate_and_incomplete_stays_pending(self):
        history = ten_fixture()
        with patch.object(p, "_history", return_value=history):
            window = load_window("binance_jan2025_10assets", lookback=5)
            truth = load_truth(window)
            self.assertEqual(set(truth["assets"]), set(p.TEN_ASSETS))
            self.assertTrue(all(row["time"] <= window["as_of"] for rows in window["histories"].values() for row in rows))
            self.assertTrue(all(t > window["as_of"] for t in truth["times"]))
            self.assertNotIn("truth", window)
            late = load_window("binance_jan2025_10assets", lookback=5, as_of="2025-01-01T00:20:00Z")
            self.assertIsNone(load_truth(late))

    def test_old_kraken_fixture_identity_is_unchanged(self):
        raw = [[1735689600 + n * 60, "10", "11", "9", "10", "10.25", "2", 2] for n in range(8)]
        with patch.object(p, "_request_kraken", return_value=raw):
            window = window_from_history(p._kraken_history(), lookback=3)
        self.assertEqual(window["window_id"], "window_66d055f3203ed1802fb39ef10f05d3917555212813f2ab404d1ab476db8bd752")


class TenAssetRealArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = load_history("binance_jan2025_10assets")

    def test_real_complete_month_and_strict_common_grid(self):
        reference = [r["time"] for r in self.history["histories"]["BTC"]]
        self.assertEqual(len(reference), 44640)
        self.assertEqual((reference[0], reference[-1]), ("2025-01-01T00:01:00Z", "2025-02-01T00:00:00Z"))
        for symbol in p.TEN_ASSETS:
            self.assertEqual([r["time"] for r in self.history["histories"][symbol]], reference)

    def test_real_default_and_exact_origin_no_truth_leak(self):
        window = load_window("binance_jan2025_10assets")
        truth = load_truth(window)
        self.assertEqual(window["as_of"], "2025-01-31T23:30:00Z")
        self.assertEqual(len(truth["times"]), 30)
        for symbol in p.TEN_ASSETS:
            self.assertEqual(len(window["histories"][symbol]), 256)
            self.assertEqual(len(truth["assets"][symbol]["close"]), 30)
            self.assertEqual(window["histories"][symbol][-1]["time"], window["as_of"])
        specific = load_window("binance_jan2025_10assets", as_of="2025-01-01T04:16:00Z")
        original = load_window(as_of=specific["as_of"])
        for symbol in p.ASSETS:
            self.assertEqual(specific["histories"][symbol], original["histories"][symbol])

    def test_original_default_and_all_96_frozen_window_contents_unchanged(self):
        self.assertEqual(load_window()["window_id"], "window_aa90f21e271c108a538111941deea349454b4f56d3de5e8b8592e922f72f1517")
        manifest = json.loads((p.DATA_DIR.parents[2] / "evaluation/runs/expanded96-20260926/manifest.json").read_text())
        self.assertEqual(len(manifest["tasks"]), 96)
        for task in manifest["tasks"]:
            self.assertEqual(load_window(as_of=task["window"]["as_of"]), task["window"])

    def test_public_provenance_matches_archives_and_original_three_reused(self):
        manifest = json.loads((p.DATA_DIR / "manifest-ten-assets-verified.json").read_text())
        self.assertEqual(manifest["status"], "verified")
        self.assertEqual(manifest["total_rows"], 446400)
        self.assertEqual([r["asset"] for r in manifest["results"]], list(p.TEN_ASSETS))
        for row in manifest["results"]:
            self.assertEqual(hashlib.sha256((p.DATA_DIR / f"{row['pair']}-1m-2025-01.zip").read_bytes()).hexdigest(), row["archive_sha256"])
            self.assertEqual(row["archive_reused"], row["asset"] in p.ASSETS)


class ArchiveEvidenceTests(unittest.TestCase):
    def archive(self, directory, rows):
        path = Path(directory) / "BNBUSDT-1m-2025-01.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("BNBUSDT-1m-2025-01.csv", "\n".join(",".join(row) for row in rows))
        checksum = hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name
        return path, checksum

    def test_wrong_checksum_or_filename_rejected_before_data_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, checksum = self.archive(directory, [])
            with self.assertRaisesRegex(DataError, "checksum mismatch"):
                verify_archive(path, "BNB", "0" * 64 + "  " + path.name)
            with self.assertRaisesRegex(DataError, "filename"):
                verify_archive(path, "BNB", checksum.replace("BNB", "XRP"))

    def test_width_microsecond_grid_and_partial_month_rejected(self):
        row = ["1735689600000000", "10", "11", "9", "10", "2", "1735689659999999", "20", "3", "1", "10", "0"]
        for mutation in ("width", "timestamp", "duplicate", "partial"):
            rows = [list(row)]
            if mutation == "width":
                rows[0].pop()
            elif mutation == "timestamp":
                rows[0][0] = "1735689600000"
            elif mutation == "duplicate":
                rows *= 2
            with tempfile.TemporaryDirectory() as directory:
                path, checksum = self.archive(directory, rows)
                with self.subTest(mutation=mutation), self.assertRaises(DataError):
                    verify_archive(path, "BNB", checksum)


if __name__ == "__main__":
    unittest.main()
