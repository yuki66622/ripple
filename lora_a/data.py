"""Isolated Q1 data access for A-line, with a fail-closed March gate.

No directory discovery and no fallback dataset are used. Calendar features are
from candle starts; public times are exclusive candle ends. Normalization uses
only the historical 256 candles. Nothing reads any data at import time.
"""
from __future__ import annotations

import calendar
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC"]
FIELDS = ("open", "high", "low", "close", "volume", "amount")
LOOKBACK, HORIZON, STRIDE = 256, 30, 30
TRAIN_STRIDE, TRAIN_END, EVALUATION_WINDOWS = 120, 25 * 1440, 300
MONTHS = ("2026-01", "2026-02", "2026-03")
FOLDERS = {"2026-01": "binance-2026-01", "2026-02": "binance-2026-02", "2026-03": "binance-2026-03-sealed"}
MAX_CSV_BYTES = 40_000_000
_HEX = re.compile(r"[0-9a-f]{64}")


class DataError(ValueError):
    """Invalid data is fatal; no skipping, repair, or substitute values."""


class SealedDataError(DataError):
    """March cannot be accessed without the verified selection lock."""


@dataclass
class MonthData:
    month: str
    assets: list[str]
    values: np.ndarray  # (asset, minute, OHLCVA), float32
    stamps: np.ndarray  # (minute, minute/hour/weekday/day/month), float32
    times: list[str]  # UTC candle END
    origins: list[int]
    provenance: dict
    raw_values: np.ndarray | None = None  # float64 prices for unchanged financial scoring


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_path(root: Path, path: Path) -> Path:
    """Reject redirects, including parent symlinks; require project containment."""
    if not path.is_absolute():
        path = root / path
    if path.resolve() != path or not path.is_relative_to(root):
        raise DataError("paths must be canonical project-local paths without symlinks")
    return path


def _artifact(root: Path, record, *, expected: Path | None = None) -> bytes:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise SealedDataError("selection identity requires exactly path and sha256")
    if not isinstance(record["path"], str) or not isinstance(record["sha256"], str) or not _HEX.fullmatch(record["sha256"]):
        raise SealedDataError("selection artifact identity is invalid")
    path = Path(record["path"])
    if not path.is_absolute():
        path = root / path
    # Reject forbidden locations lexically before resolve() could probe them.
    if ".." in path.parts or not path.is_relative_to(root / "lora_a") or (expected is not None and path != expected):
        raise SealedDataError("selection artifacts must be under lora_a at their declared locations")
    path = _safe_path(root, path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SealedDataError("selection artifact is unavailable") from exc
    if _sha(raw) != record["sha256"]:
        raise SealedDataError("selection artifact SHA256 mismatch")
    return raw


def validate_selection_lock(lock, *, root: Path | str | None = None) -> dict:
    """Validate identities without touching March paths or any market dataset."""
    # This first check is deliberately before resolve(), stat(), or open().
    if (not isinstance(lock, dict) or lock.get("status") != "locked"
            or lock.get("acceptance_authorized") is not True
            or isinstance(lock.get("epoch"), bool) or not isinstance(lock.get("epoch"), int)
            or not 1 <= lock["epoch"] <= 2):
        raise SealedDataError("March is sealed: an authorized unique selected checkpoint is required")
    try:
        locked_at = datetime.fromisoformat(lock["locked_at_utc"].replace("Z", "+00:00"))
        if locked_at.utcoffset() != timedelta(0):
            raise ValueError
    except (KeyError, TypeError, AttributeError, ValueError):
        raise SealedDataError("selection lock requires a UTC lock timestamp") from None
    root = Path(root or ROOT).resolve()
    config_raw = _artifact(root, lock.get("config"), expected=root / "lora_a/config.json")
    try:
        config = json.loads(config_raw)
    except (ValueError, UnicodeError):
        raise SealedDataError("locked config is not valid JSON") from None
    _validate_config(config)
    checkpoint = lock.get("selected_checkpoint")
    if not isinstance(checkpoint, dict) or Path(checkpoint.get("path", "")).name != "adapter_model.safetensors":
        raise SealedDataError("one adapter_model.safetensors checkpoint must be selected")
    _artifact(root, checkpoint)
    _artifact(root, lock.get("validation_report"))
    selected = _validate_selection_history(root, lock)
    _validate_february_report(root, lock, selected)
    code = lock.get("code")
    if not isinstance(code, list) or not code:
        raise SealedDataError("selection lock requires code identities")
    names = []
    for record in code:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise SealedDataError("code identity is invalid")
        _artifact(root, record)
        path = Path(record["path"])
        names.append(path if path.is_absolute() else root / path)
    required = {root / "lora_a" / (name + ".py") for name in ("data", "metrics", "training", "runner")}
    if len(names) != len(set(names)) or not required.issubset(names):
        raise SealedDataError("selection lock must uniquely identify data, metrics, training and runner code")
    if "selected_merged_checkpoint" in lock:
        _artifact(root, lock["selected_merged_checkpoint"])
    return config


def _validate_selection_history(root: Path, lock: dict):
    try:
        history = json.loads(_artifact(root, lock.get("selection_history")))
    except (ValueError, UnicodeError) as exc:
        raise SealedDataError("selection history identity or JSON is invalid") from exc
    if not isinstance(history, dict) or not isinstance(history.get("candidates"), list) or not history["candidates"]:
        raise SealedDataError("selection history requires at least one candidate")
    candidates, epochs = history["candidates"], []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise SealedDataError("selection history candidate must be an object")
        epoch, mae = candidate.get("epoch"), candidate.get("small_mae")
        if (isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 2
                or candidate.get("name") != f"epoch_{epoch:02d}"
                or isinstance(mae, bool) or not isinstance(mae, (int, float)) or not math.isfinite(mae) or mae < 0):
            raise SealedDataError("candidate requires a unique epoch, canonical name and finite nonnegative small MAE")
        _artifact(root, candidate.get("validation_report"))
        epochs.append(epoch)
    planned = history.get("planned_epochs")
    if (len(epochs) != len(set(epochs)) or isinstance(planned, bool) or not isinstance(planned, int)
            or not max(epochs) <= planned <= 2):
        raise SealedDataError("selection history epoch plan is inconsistent")
    best = min(candidates, key=lambda candidate: (candidate["small_mae"], candidate["epoch"]))
    selected = history.get("selected")
    if (selected != best or best["epoch"] != lock["epoch"]
            or best["validation_report"] != lock.get("validation_report")):
        raise SealedDataError("locked selection must be the minimum small-tier MAE, using earlier epoch on ties")
    checkpoint_path = Path(lock["selected_checkpoint"]["path"])
    if checkpoint_path.parent.name != "adapter" or checkpoint_path.parent.parent.name != best["name"]:
        raise SealedDataError("selected adapter path must identify the selected epoch")
    if "selected_merged_checkpoint" in lock:
        merged = lock["selected_merged_checkpoint"]
        if not isinstance(merged, dict) or not isinstance(merged.get("path"), str):
            raise SealedDataError("selected merged checkpoint identity is invalid")
        merged_path = Path(merged["path"])
        if (merged_path.name != "model.safetensors" or merged_path.parent.name != "merged"
                or merged_path.parent.parent != checkpoint_path.parent.parent):
            raise SealedDataError("merged checkpoint must belong to the selected adapter epoch")
    return best


def _validate_february_report(root, lock, selected):
    try:
        report = json.loads(_artifact(root, lock["validation_report"]))
    except (ValueError, UnicodeError) as exc:
        raise SealedDataError("February report JSON or identity is invalid") from exc
    if not isinstance(report, dict) or report.get("month") != "2026-02" or report.get("n_scheduled") != EVALUATION_WINDOWS:
        raise SealedDataError("selection requires the frozen 300-window February report")
    for tier in ("major", "small"):
        for method in ("original", selected["name"]):
            try:
                row = report["tiers"][tier]["methods"][method]
            except (KeyError, TypeError):
                raise SealedDataError("February must report both tiers for selected and original checkpoints") from None
            if (not isinstance(row, dict) or row.get("n_scheduled") != EVALUATION_WINDOWS
                    or row.get("n_scored") != EVALUATION_WINDOWS or row.get("n_failed") != 0):
                raise SealedDataError("March requires complete 300-window February coverage in both tiers")
    small = report["tiers"]["small"]["methods"]
    candidate_mae, original_mae = small[selected["name"]].get("volatility_mae"), small["original"].get("volatility_mae")
    if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
            for x in (candidate_mae, original_mae)) or candidate_mae != selected["small_mae"]
            or candidate_mae >= original_mae):
        raise SealedDataError("selected February small-tier MAE must strictly improve on original; March remains sealed")


def _claim_march(lock, *, root=None) -> dict:
    """Consume final acceptance once, before probing any March data path.

    A failed later read/inference intentionally leaves the receipt in place.
    Recovery requires a separately reviewed action, never an automatic retry.
    """
    config = validate_selection_lock(lock, root=root)
    root = Path(root or ROOT).resolve()
    canonical = json.dumps(lock, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt = {"schema_version": 1, "selection_lock_canonical_sha256": _sha(canonical),
               "selected_epoch": lock["epoch"], "selected_checkpoint": lock["selected_checkpoint"],
               "unsealed_at_utc": datetime.now(timezone.utc).isoformat(),
               "policy": "single acceptance claim; engineering recovery requires explicit review"}
    path = _safe_path(root, root / "lora_a/MARCH_UNSEALED.json")
    try:
        with path.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise SealedDataError("March acceptance was already claimed; no automatic rerun is permitted") from None
    return config


def _validate_config(config, *, allow_unfrozen_pair=False):
    required = {"assets": ASSETS, "lookback": LOOKBACK, "horizon": HORIZON, "stride": STRIDE,
                "train_month": "2026-01", "validation_month": "2026-02", "test_month": "2026-03",
                "training_candles_per_window": LOOKBACK + HORIZON,
                "protocol_version": "lora-a-v2.1", "max_epochs": 2, "train_stride": TRAIN_STRIDE,
                "evaluation_stride": STRIDE, "validation_windows": EVALUATION_WINDOWS, "test_windows": EVALUATION_WINDOWS,
                "observation_windows_per_day": 8, "train_start": "2026-01-01", "train_end_exclusive": "2026-01-26",
                "observation_start": "2026-01-26", "observation_end_exclusive": "2026-02-01"}
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in required.items()):
        raise DataError("data configuration differs from the frozen A-line protocol")
    pair = config.get("fixed_small_pair")
    if pair is None and allow_unfrozen_pair:
        return
    if (not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(asset, str) for asset in pair)
            or len(set(pair)) != 2 or not set(pair).issubset(ASSETS[4:])):
        raise DataError("a distinct two-small-asset January-fixed pair is required before validation or acceptance")


def _month_grid(month):
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    return start, calendar.monthrange(start.year, start.month)[1] * 1440


def _role(month, role=None):
    allowed = {"2026-01": ("train", "observation"), "2026-02": ("validation",), "2026-03": ("test",)}
    if month not in allowed:
        raise DataError("only January training/observation, February selection and March acceptance are allowed")
    role = allowed[month][0] if role is None else role
    if role not in allowed[month]:
        raise DataError("month and role do not match the frozen calendar split")
    return role


def _evenly_spaced(candidates, count):
    if len(candidates) < count or count < 2:
        raise DataError("calendar grid cannot supply the frozen number of unique windows")
    return [candidates[i * (len(candidates) - 1) // (count - 1)] for i in range(count)]


def calendar_origins(month: str, role: str | None = None) -> list[int]:
    """Pure time-only v2.1 grid; no filesystem access, financial values or RNG."""
    role = _role(month, role)
    _, rows = _month_grid(month)
    if role == "train":
        return list(range(LOOKBACK - 1, min(rows, TRAIN_END) - HORIZON, TRAIN_STRIDE))
    candidates = list(range(LOOKBACK - 1, rows - HORIZON, STRIDE))
    if role in ("validation", "test"):
        return _evenly_spaced(candidates, EVALUATION_WINDOWS)
    origins = []
    # Origin day uses the public candle-end timestamp, never its open time.
    for day in range(25, 31):
        daily = [origin for origin in candidates if day * 1440 <= origin + 1 < (day + 1) * 1440]
        origins.extend(_evenly_spaced(daily, 8))
    return origins


def _parse_csv(raw: bytes, month: str, *, expected_rows: int) -> np.ndarray:
    """Parse all rows and validate their full grid and OHLCVA before float32."""
    if len(raw) > MAX_CSV_BYTES:
        raise DataError("CSV exceeds the frozen byte limit")
    start, _ = _month_grid(month)
    start_us = int(start.timestamp()) * 1_000_000
    values = np.empty((expected_rows, 6), dtype=np.float64)
    count = 0
    try:
        reader = csv.reader(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline=""), strict=True)
        for i, row in enumerate(reader):
            if i >= expected_rows or len(row) != 12:
                raise DataError(f"row {i + 1}: incorrect row count or CSV width")
            opened = start_us + i * 60_000_000
            if (not re.fullmatch(r"[0-9]{16}", row[0]) or not re.fullmatch(r"[0-9]{16}", row[6])
                    or int(row[0]) != opened or int(row[6]) != opened + 59_999_999):
                raise DataError(f"row {i + 1}: candle timestamps violate the complete UTC month grid")
            try:
                vals = np.asarray([row[j] for j in (1, 2, 3, 4, 5, 7)], dtype=np.float64)
            except (ValueError, OverflowError):
                raise DataError(f"row {i + 1}: nonnumerical OHLCVA") from None
            o, h, low, c, volume, amount = vals
            if not np.isfinite(vals).all() or min(o, h, low, c) <= 0 or min(volume, amount) < 0:
                raise DataError(f"row {i + 1}: invalid finite positive prices/nonnegative volume and amount")
            if not low <= min(o, c) <= max(o, c) <= h or bool(volume) != bool(amount):
                raise DataError(f"row {i + 1}: inconsistent OHLC or zero volume/amount")
            with np.errstate(over="ignore", under="ignore"):
                converted = vals.astype(np.float32)
            if (not np.isfinite(converted).all() or (converted[:4] <= 0).any()
                    or bool(converted[4]) != bool(converted[5])):
                raise DataError(f"row {i + 1}: OHLCVA cannot be represented as finite float32")
            values[i] = vals
            count += 1
    except (UnicodeError, csv.Error):
        raise DataError("CSV encoding or record structure is invalid") from None
    if count != expected_rows:
        raise DataError(f"month has {count} rows; expected {expected_rows}; no missing rows are allowed")
    return values


def load_month(month: str, *, role=None, root=None, selection_lock=None) -> MonthData:
    role = _role(month, role)
    origins = calendar_origins(month, role)  # Freeze calendar selection before any data IO.
    # Even config IO is behind the March lock. Never probe a March path first.
    if month == "2026-03":
        config = _claim_march(selection_lock, root=root)
    else:
        project_root = Path(root or ROOT).resolve()
        config = json.loads(_safe_path(project_root, project_root / "lora_a/config.json").read_bytes())
        _validate_config(config, allow_unfrozen_pair=month == "2026-01")
    root = Path(root or ROOT).resolve()
    directory = _safe_path(root, root / "research/data-probe" / FOLDERS[month])
    manifest_raw = _safe_path(root, directory / "manifest.json").read_bytes()
    try:
        manifest = json.loads(manifest_raw)
    except (ValueError, UnicodeError):
        raise DataError("month manifest is not valid JSON") from None
    start, rows = _month_grid(month)
    if (not isinstance(manifest, dict) or manifest.get("month") != month or manifest.get("status") != "verified"
            or manifest.get("verified_assets") != len(ASSETS) or manifest.get("failed_assets") != 0
            or manifest.get("expected_rows_per_asset") != rows or manifest.get("interval_seconds") != 60
            or manifest.get("quote_currency") != "USDT" or not isinstance(manifest.get("assets"), list)
            or len(manifest["assets"]) != len(ASSETS) or set(manifest["assets"]) != set(ASSETS)):
        raise DataError("manifest does not describe the complete frozen ten-asset month")
    results = manifest.get("results")
    if (not isinstance(results, list) or len(results) != len(ASSETS)
            or any(not isinstance(r, dict) for r in results)
            or {r.get("asset") for r in results} != set(ASSETS)):
        raise DataError("manifest must identify each frozen asset exactly once")
    entries = {r["asset"]: r for r in results}
    seal_entries = None
    if month == "2026-03":
        seal = json.loads(_safe_path(root, root / "research/2026-03-seal.json").read_bytes())
        if seal.get("status") != "sealed" or seal.get("month") != month or seal.get("manifest_sha256") != _sha(manifest_raw):
            raise SealedDataError("March manifest differs from the original external seal")
        seal_files = seal.get("files")
        if (not isinstance(seal_files, list) or len(seal_files) != len(ASSETS)
                or any(not isinstance(e, dict) for e in seal_files)
                or {e.get("asset") for e in seal_files} != set(ASSETS)):
            raise SealedDataError("March seal asset identities are incomplete")
        seal_entries = {r["asset"]: r for r in seal_files}
    all_values, csv_hashes = [], {}
    for asset in config["assets"]:
        entry = entries[asset]
        filename = f"{asset}USDT-1m-{month}.csv"
        digest = entry.get("csv_sha256")
        if (entry.get("status") != "verified" or entry.get("month") != month or entry.get("pair") != asset + "USDT"
                or entry.get("rows") != rows or entry.get("csv_filename") != filename
                or entry.get("official_zip_sha256_verified") is not True or entry.get("zip_crc_verified") is not True
                or not isinstance(digest, str) or not _HEX.fullmatch(digest)):
            raise DataError(f"{asset}: manifest identity or acquisition verification is invalid")
        if seal_entries is not None and any(seal_entries[asset].get(k) != entry.get(k) for k in ("csv_filename", "csv_sha256", "zip_filename", "zip_sha256")):
            raise SealedDataError("March file identities differ from the external seal")
        raw = _safe_path(root, directory / filename).read_bytes()
        if _sha(raw) != digest or entry.get("csv_bytes") != len(raw):
            raise DataError(f"{asset}: CSV SHA256/byte length differs from the verified manifest")
        all_values.append(_parse_csv(raw, month, expected_rows=rows))
        csv_hashes[asset] = digest
    starts = [start + timedelta(minutes=i) for i in range(rows)]
    stamps = np.asarray([[d.minute, d.hour, d.weekday(), d.day, d.month] for d in starts], dtype=np.float32)
    times = [(d + timedelta(minutes=1)).isoformat().replace("+00:00", "Z") for d in starts]
    raw_values = np.stack(all_values)
    values = raw_values.astype(np.float32)
    values.setflags(write=False)
    raw_values.setflags(write=False)
    stamps.setflags(write=False)
    return MonthData(month=month, assets=list(config["assets"]), values=values, stamps=stamps, times=times,
                     origins=origins, raw_values=raw_values,
                     provenance={"manifest_sha256": _sha(manifest_raw), "csv_sha256": csv_hashes, "role": role,
                                 "grid": {"origins": origins.copy(), "n": len(origins), "selection": "calendar only before data access",
                                          "stride_candidates": TRAIN_STRIDE if role == "train" else STRIDE,
                                          "sampling": "all training origins" if role == "train" else "floor(i*(n-1)/(k-1)), including endpoints"},
                                 "rows_per_asset": rows, "raw_time_unit": "microseconds", "public_time": "exclusive candle end",
                                 "split_by": "raw candle-open UTC month", "config_canonical_sha256": _sha(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()),
                                 "source": "Binance official public spot monthly klines", "cast": "float64 financial scoring; validated float32 training"})


def _check_origin(data: MonthData, origin: int):
    if (isinstance(origin, bool) or not isinstance(origin, (int, np.integer))
            or origin < LOOKBACK - 1 or origin + HORIZON >= len(data.times)
            or (origin - (LOOKBACK - 1)) % STRIDE != 0 or origin not in data.origins):
        raise DataError("origin is outside the complete frozen 256-to-30 grid")
    role = _role(data.month, data.provenance.get("role"))
    if origin not in calendar_origins(data.month, role):
        raise DataError("origin is outside this role's predeclared calendar grid")
    if role == "train" and origin + HORIZON >= TRAIN_END:
        raise DataError("January training targets may not cross into January 26")


def window(data: MonthData, origin: int) -> dict:
    """Input-only MarketWindow v1; future prices cannot influence its identity."""
    _check_origin(data, origin)
    section = slice(origin - LOOKBACK + 1, origin + 1)
    prices = data.raw_values if data.raw_values is not None else data.values
    result = {"schema_version": 1, "profile_id": "binance_" + data.month.replace("-", "_") + "_lora_a",
              "source": "Binance official public spot monthly klines", "mode": "replay", "quote_currency": "USDT",
              "interval_seconds": 60, "assets": list(data.assets), "as_of": data.times[origin],
              "quality": {"input_complete": True, "calendar_features": "candle starts", "public_times": "candle ends"},
              "histories": {asset: [{"time": stamp, **{field: float(value) for field, value in zip(FIELDS, row)}}
                                    for stamp, row in zip(data.times[section], prices[a, section])]
                            for a, asset in enumerate(data.assets)}}
    result["window_id"] = "window_" + _sha(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return result


def truth(data: MonthData, origin: int) -> dict:
    """Return exactly the 30 future candles as an asset-to-fields mapping."""
    _check_origin(data, origin)
    prices = data.raw_values if data.raw_values is not None else data.values
    return {asset: {field: prices[a, origin + 1:origin + HORIZON + 1, f].astype(float).tolist()
                    for f, field in enumerate(FIELDS)} for a, asset in enumerate(data.assets)}


def training_window(data: MonthData, origin: int, asset: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw 286-candle January sample; trainer shifts once to 285 token targets."""
    if data.month != "2026-01" or _role(data.month, data.provenance.get("role")) != "train":
        raise DataError("training windows are restricted to January 1-25 training role")
    _check_origin(data, origin)
    if asset not in data.assets:
        raise DataError("unknown training asset")
    section = slice(origin - LOOKBACK + 1, origin + HORIZON + 1)
    return data.values[data.assets.index(asset), section].copy(), data.stamps[section].copy()


def normalize_training_window(raw: np.ndarray) -> np.ndarray:
    """Official past-only normalization; future values never set its scale."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (LOOKBACK + HORIZON, len(FIELDS)) or not np.isfinite(raw).all():
        raise DataError("training sample must contain exactly 286 finite OHLCVA candles")
    past = raw[:LOOKBACK]
    return np.clip((raw - past.mean(axis=0)) / (past.std(axis=0, ddof=0) + 1e-5), -5, 5).astype(np.float32)
