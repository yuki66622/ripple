"""Verified C-line Binance acquisition; public CLI is April-only.

No network or filesystem activity occurs at import. May acquisition requires
the loader's already-created one-shot receipt and verified reviewed lock.
"""
from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile

from lora_a.data import ASSETS, DataError, _parse_csv, _safe_path

ROOT = Path(__file__).resolve().parents[1]
MONTHS = ("2026-04", "2026-05")
MAX_ZIP_BYTES, MAX_CSV_BYTES = 24_000_000, 40_000_000


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _rows(month):
    if month not in MONTHS:
        raise DataError("C acquisition allows only April or reviewed May")
    year, month_number = map(int, month.split("-"))
    return calendar.monthrange(year, month_number)[1] * 1440


def _names(asset, month):
    if asset not in ASSETS or month not in MONTHS:
        raise DataError("asset/month outside C acquisition allowlist")
    stem = f"{asset}USDT-1m-{month}"
    return stem + ".zip", stem + ".csv", stem + ".zip.CHECKSUM"


def _url(asset, month):
    return f"https://data.binance.vision/data/spot/monthly/klines/{asset}USDT/1m/{_names(asset, month)[0]}"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch(url, limit):
    allowed = {_url(a, m) + suffix for a in ASSETS for m in MONTHS for suffix in ("", ".CHECKSUM")}
    if url not in allowed:
        raise DataError("URL is outside official frozen C acquisition sources")
    with build_opener(_NoRedirect()).open(Request(url, headers={"User-Agent": "HackUMBC-C-verified-acquisition/1"}), timeout=45) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise DataError("download exceeds frozen byte limit")
    return raw


def _write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def inspect_archive(zip_raw, checksum_raw, asset, month):
    """Pure byte validation: official SHA, exact member, CRC; no dataset IO."""
    zip_name, csv_name, _ = _names(asset, month)
    try:
        match = re.fullmatch(r"([a-fA-F0-9]{64})\s+\*?" + re.escape(zip_name) + r"\s*", checksum_raw.decode("ascii"))
    except UnicodeError:
        match = None
    if not match or len(zip_raw) > MAX_ZIP_BYTES or _sha(zip_raw) != match.group(1).lower():
        raise DataError("official ZIP checksum/filename/byte limit failed")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_raw)) as archive:
            if archive.namelist() != [csv_name]:
                raise DataError("ZIP must contain exactly its expected CSV member")
            info = archive.getinfo(csv_name)
            if info.is_dir() or info.flag_bits & 1 or info.file_size > MAX_CSV_BYTES:
                raise DataError("ZIP CSV member is invalid, encrypted or oversized")
            raw = archive.read(csv_name)  # Full read verifies member CRC.
            if len(raw) != info.file_size:
                raise DataError("ZIP extracted size mismatch")
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError) as exc:
        raise DataError("ZIP structure, decompression or CRC validation failed") from exc
    return raw


def acquire_month(month, *, root=None, selection_lock=None, receipt=None):
    """Acquire exact month or stop at first failure, preserving every raw file."""
    if month not in MONTHS:
        raise DataError("C acquisition permits April or reviewed May only")
    if month == "2026-05":
        # Validate before resolving, creating, stat-ing or reading any May path.
        from .data import consume_may_acquisition
        consume_may_acquisition(selection_lock, receipt, root=root)
    root = Path(root or ROOT).resolve()
    directory = _safe_path(root, root / "lora_c/data" / f"binance-{month}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = _safe_path(root, directory / "manifest.json")
    if manifest_path.exists():
        return manifest_path  # Loader independently verifies every CSV identity/value.
    if any(directory.iterdir()):
        raise DataError("nonempty month directory has no complete manifest; partial acquisition cannot auto-resume or repair")
    results, active_asset = [], None
    try:
        for asset in ASSETS:
            active_asset = asset
            zip_name, csv_name, checksum_name = _names(asset, month)
            zip_path = _safe_path(root, directory / zip_name)
            checksum_path = _safe_path(root, directory / checksum_name)
            csv_path = _safe_path(root, directory / csv_name)
            if checksum_path.exists():
                checksum_raw = checksum_path.read_bytes()
            else:
                checksum_raw = _fetch(_url(asset, month) + ".CHECKSUM", 4096)
                _write_new(checksum_path, checksum_raw)
            if zip_path.exists():
                zip_raw = zip_path.read_bytes()
            else:
                zip_raw = _fetch(_url(asset, month), MAX_ZIP_BYTES)
                _write_new(zip_path, zip_raw)
            csv_raw = inspect_archive(zip_raw, checksum_raw, asset, month)
            if csv_path.exists():
                if _sha(csv_path.read_bytes()) != _sha(csv_raw):
                    raise DataError("existing CSV differs from verified archive; preserved without overwrite")
            else:
                _write_new(csv_path, csv_raw)
            _parse_csv(csv_raw, month, expected_rows=_rows(month))
            results.append({"asset": asset, "pair": asset + "USDT", "month": month, "status": "verified",
                            "rows": _rows(month), "csv_filename": csv_name, "csv_sha256": _sha(csv_raw), "csv_bytes": len(csv_raw),
                            "zip_filename": zip_name, "zip_sha256": _sha(zip_raw), "zip_bytes": len(zip_raw),
                            "checksum_filename": checksum_name, "checksum_sha256": _sha(checksum_raw),
                            "official_zip_sha256_verified": True, "zip_crc_verified": True,
                            "source_url": _url(asset, month), "financial_values_validated": True,
                            "missing_rows": 0, "missing_rate": 0, "duplicate_rows": 0, "invalid_rows": 0})
        manifest = {"schema_version": 1, "month": month, "status": "verified", "assets": ASSETS,
                    "quote_currency": "USDT", "interval_seconds": 60, "verified_assets": 10, "failed_assets": 0,
                    "expected_rows_per_asset": _rows(month), "verified_at": datetime.now(timezone.utc).isoformat(),
                    "validation_scope": "official ZIP SHA256, exact member/CRC, complete UTC microsecond grid, finite OHLCVA and volume consistency",
                    "results": results}
        _write_new(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    except BaseException as exc:
        failure = {"month": month, "status": "failed", "failed_asset": active_asset,
                   "verified_assets": [r["asset"] for r in results], "error_type": type(exc).__name__,
                   "error": str(exc) if isinstance(exc, DataError) else "download/filesystem operation failed; raw evidence retained",
                   "utc": datetime.now(timezone.utc).isoformat(), "no_missing_rows_filled": True,
                   "missing_rows": None, "missing_rate": None, "duplicate_rows": None, "invalid_rows": None,
                   "quality_counts_status": "unknown: stopped at first failure; no zero counts inferred"}
        failure_path = directory / ("acquisition-failure-" + uuid.uuid4().hex + ".json")
        _write_new(failure_path, (json.dumps(failure, indent=2) + "\n").encode())
        raise
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=["2026-04"])
    args = parser.parse_args()
    print(acquire_month(args.month))
