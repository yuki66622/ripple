"""Synthetic archive and acquisition failures; never perform real downloads."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from lora_c import acquire as a
from lora_c import data as d


def csv_bytes(month="2026-04", rows=2):
    start = int(datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc).timestamp()) * 1_000_000
    return "".join(f"{start+i*60000000},100,102,99,101,2,{start+i*60000000+59999999},201,1,1,100,0\n" for i in range(rows)).encode()


def zipped(asset="BTC", month="2026-04", raw=None, member=None):
    stream = io.BytesIO()
    zip_name, csv_name, _ = a._names(asset, month)
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member or csv_name, raw if raw is not None else csv_bytes(month))
    data = stream.getvalue()
    return data, f"{hashlib.sha256(data).hexdigest()}  {zip_name}\n".encode()


class AcquisitionTests(unittest.TestCase):
    def test_checksum_exact_member_crc_and_csv_structural_failure(self):
        raw, checksum = zipped()
        extracted = a.inspect_archive(raw, checksum, "BTC", "2026-04")
        self.assertEqual(d._parse_csv(extracted, "2026-04", expected_rows=2).shape, (2, 6))
        with self.assertRaises(d.DataError):
            a.inspect_archive(raw, checksum.replace(checksum[:64], b"0" * 64), "BTC", "2026-04")
        bad, check = zipped(member="../BTC.csv")
        with self.assertRaises(d.DataError):
            a.inspect_archive(bad, check, "BTC", "2026-04")
        corrupted = raw.replace(b",100,102,99,101,", b",101,102,99,101,", 1)
        checksum = f"{hashlib.sha256(corrupted).hexdigest()}  BTCUSDT-1m-2026-04.zip\n".encode()
        with self.assertRaisesRegex(d.DataError, "CRC"):
            a.inspect_archive(corrupted, checksum, "BTC", "2026-04")
        for bad in (extracted.splitlines(keepends=True)[0] * 2,
                    extracted.replace(b",100,102,99,101,", b",100,99,99,101,", 1),
                    extracted.replace(b",2,", b",-2,", 1), extracted.splitlines(keepends=True)[0]):
            with self.assertRaises(d.DataError):
                d._parse_csv(bad, "2026-04", expected_rows=2)

    def test_may_or_other_month_cannot_probe_or_download_without_gate(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("no path IO")), \
                patch.object(a, "_fetch", side_effect=AssertionError("no download")):
            for month in ("2026-01", "2026-02", "2026-03", "2026-06", "2026-08", "2026-05"):
                with self.assertRaises(d.DataError):
                    a.acquire_month(month)

    def test_synthetic_full_month_acquisition_and_no_redownload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            responses = {}
            for asset in d.ASSETS:
                raw, checksum = zipped(asset)
                responses[a._url(asset, "2026-04")] = raw
                responses[a._url(asset, "2026-04") + ".CHECKSUM"] = checksum
            with patch.object(a, "_rows", return_value=2), patch.object(a, "_fetch", side_effect=lambda url, limit: responses[url]) as fetch:
                path = a.acquire_month("2026-04", root=root)
                self.assertEqual(fetch.call_count, 20)
                manifest = json.loads(path.read_bytes())
                self.assertEqual(manifest["verified_assets"], 10)
                for entry in manifest["results"]:
                    self.assertEqual([entry[k] for k in ("missing_rows", "missing_rate", "duplicate_rows", "invalid_rows")], [0, 0, 0, 0])
                a.acquire_month("2026-04", root=root)
                self.assertEqual(fetch.call_count, 20)

    def test_bad_raw_evidence_remains_and_partial_directory_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            bad_csv = csv_bytes().replace(b",100,102,99,101,", b",100,99,99,101,", 1)
            raw, checksum = zipped(raw=bad_csv)
            with patch.object(a, "_rows", return_value=2), patch.object(a, "_fetch", side_effect=lambda url, limit: checksum if url.endswith("CHECKSUM") else raw):
                with self.assertRaises(d.DataError):
                    a.acquire_month("2026-04", root=root)
            directory = root / "lora_c/data/binance-2026-04"
            self.assertEqual((directory / "BTCUSDT-1m-2026-04.csv").read_bytes(), bad_csv)
            self.assertEqual(len(list(directory.glob("acquisition-failure-*.json"))), 1)
            failure = json.loads(next(directory.glob("acquisition-failure-*.json")).read_bytes())
            self.assertEqual([failure[k] for k in ("missing_rows", "missing_rate", "duplicate_rows", "invalid_rows")], [None] * 4)
            self.assertFalse((directory / "manifest.json").exists())
            with patch.object(a, "_fetch", side_effect=AssertionError("no repair download")):
                with self.assertRaisesRegex(d.DataError, "cannot auto-resume"):
                    a.acquire_month("2026-04", root=root)

    def test_symlink_directory_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            parent = root / "lora_c/data"
            parent.mkdir(parents=True)
            (parent / "binance-2026-04").symlink_to(root / "outside")
            with patch.object(a, "_fetch", side_effect=AssertionError("no download")):
                with self.assertRaises(d.DataError):
                    a.acquire_month("2026-04", root=root)


if __name__ == "__main__":
    unittest.main()
