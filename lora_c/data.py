"""C-line isolated data, calendar plans, and reviewed one-shot May access."""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from lora_a.data import ASSETS, FIELDS, MonthData, DataError, _parse_csv, _safe_path, normalize_training_window

ROOT = Path(__file__).resolve().parents[1]
LOOKBACK, HORIZON, TRAIN_END = 256, 30, 36000
MONTHS = ("2026-01", "2026-04", "2026-05")


class SealedDataError(DataError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _role(month, role=None):
    roles = {"2026-01": "train", "2026-04": "validation", "2026-05": "test"}
    if month not in roles or role not in (None, roles[month]):
        raise DataError("C permits only January train, April validation, May reviewed test")
    return roles[month]


def _month_grid(month):
    _role(month)
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    return start, calendar.monthrange(start.year, start.month)[1] * 1440


def calendar_origins(month, role=None):
    role = _role(month, role)
    _, rows = _month_grid(month)
    if role == "train":
        return list(range(255, min(rows, TRAIN_END) - 30, 120))
    candidates = list(range(255, rows - 30, 30))
    if len(candidates) < 300:
        raise DataError("C calendar requires exactly 300 distinct evaluation origins")
    return [candidates[i * (len(candidates) - 1) // 299] for i in range(300)]


def _validate_config(config):
    required = {"protocol_version": "lora-c-v1", "assets": ASSETS, "train_month": "2026-01",
                "validation_month": "2026-04", "test_month": "2026-05", "lookback": 256, "horizon": 30,
                "training_candles_per_window": 286, "train_stride": 120, "evaluation_stride": 30,
                "validation_windows": 300, "test_windows": 300, "max_epochs": 2,
                "train_start": "2026-01-01", "train_end_exclusive": "2026-01-26",
                "selection_metric": "small.max_drawdown_mae", "major_drawdown_stop_relative": 0.2}
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in required.items()):
        raise DataError("configuration differs from the frozen C protocol")


def _artifact(root, reference, *, expected=None, code=False):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str) or not isinstance(reference["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", reference["sha256"]) is None):
        raise SealedDataError("artifact requires a path and SHA256 identity")
    path = Path(reference["path"])
    path = path if path.is_absolute() else root / path
    reusable = {root / "lora_a" / name for name in ("data.py", "training.py", "recovery_state.py", "supervise.py")}
    if (".." in path.parts or path.is_relative_to(root / "lora_c/data")
            or (not path.is_relative_to(root / "lora_c") and not (code and path in reusable))
            or (expected is not None and path != expected)):
        raise SealedDataError("artifact path is outside its permitted C/code scope")
    probe = root
    for part in path.relative_to(root).parts:
        probe = probe / part
        if probe.is_symlink():
            raise SealedDataError("artifact symlinks are prohibited before target resolution")
    _safe_path(root, path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SealedDataError("locked artifact unavailable") from exc
    if _sha(raw) != reference["sha256"]:
        raise SealedDataError("locked artifact SHA256 mismatch")
    return raw


def _json(raw, label):
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise SealedDataError(label + " is not valid JSON") from exc


def _utc(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() != timedelta(0):
            raise ValueError
        return result
    except (AttributeError, TypeError, ValueError):
        raise SealedDataError("explicit UTC timestamp is required") from None


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _candidate_report(root, candidate):
    from .metrics import major_drawdown_guard, selection_gate
    report = _json(_artifact(root, candidate.get("validation_report")), "April report")
    if not isinstance(report, dict) or report.get("month") != "2026-04" or report.get("n_scheduled") != 300:
        raise SealedDataError("selection requires a complete 300-window April report")
    values = {}
    for tier in ("major", "small"):
        for method in ("original", candidate["name"]):
            try:
                row = report["tiers"][tier]["methods"][method]
            except (KeyError, TypeError):
                raise SealedDataError("April report lacks a required tier/model") from None
            if (not isinstance(row, dict) or row.get("n_scheduled") != 300 or row.get("n_scored") != 300
                    or row.get("n_failed") != 0 or row.get("n_max_drawdown_mae_origins") != 300
                    or row.get("n_max_drawdown_mae_asset_origins") != 300 * (4 if tier == "major" else 6)
                    or not _finite(row.get("max_drawdown_mae"))):
                raise SealedDataError("all April candidates and original require full tier coverage and finite MDD MAE")
            values[tier, method] = row["max_drawdown_mae"]
    if values["small", candidate["name"]] != candidate["small_mae"]:
        raise SealedDataError("candidate history MDD differs from April report")
    guard = major_drawdown_guard(report, candidate["name"])
    gate = selection_gate(report, candidate["name"])
    if guard.get("evidence_complete") is not True or guard.get("passed") is not True:
        raise SealedDataError("an April epoch has incomplete paired MDD evidence or exceeded the major-tier 20% stop")
    if gate.get("checks", {}).get("small_vs_original", {}).get("evidence_complete") is not True:
        raise SealedDataError("an April epoch has incomplete small-tier paired MDD evidence")
    return values["small", "original"], gate["passed"]


def validate_selection_lock(lock, *, root=None):
    # Invalid/unapproved lock fails before any filesystem operation.
    if (not isinstance(lock, dict) or lock.get("status") != "locked" or lock.get("acceptance_authorized") is not True
            or isinstance(lock.get("epoch"), bool) or not isinstance(lock.get("epoch"), int) or not 1 <= lock["epoch"] <= 2):
        raise SealedDataError("May remains sealed without an authorized unique checkpoint lock")
    _utc(lock.get("locked_at_utc"))
    root = Path(root or ROOT).resolve()
    config = _json(_artifact(root, lock.get("config"), expected=root / "lora_c/config.json"), "C config")
    _validate_config(config)
    environment = _json(_artifact(root, lock.get("environment")), "environment")
    shared = environment.get("shared_code") if isinstance(environment, dict) else None
    if not isinstance(shared, list):
        raise SealedDataError("environment must bind the reused parser source")
    parser_records = []
    for record in shared:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise SealedDataError("environment shared-code reference is invalid")
        path = Path(record["path"])
        path = path if path.is_absolute() else root / path
        if path == root / "lora_a/data.py":
            parser_records.append(record)
    if len(parser_records) != 1:
        raise SealedDataError("environment must uniquely identify reused lora_a/data.py")
    _artifact(root, parser_records[0], expected=root / "lora_a/data.py", code=True)
    _artifact(root, lock.get("selected_checkpoint"))
    _artifact(root, lock.get("selected_merged_checkpoint"))
    code = lock.get("code")
    if not isinstance(code, list):
        raise SealedDataError("code identities are required")
    paths = []
    for record in code:
        _artifact(root, record, code=True)
        path = Path(record["path"])
        paths.append(path if path.is_absolute() else root / path)
    required = {root / "lora_c" / f"{name}.py" for name in ("data", "acquire", "metrics", "training", "runner")}
    if len(set(paths)) != len(paths) or not required.issubset(paths):
        raise SealedDataError("complete unique C code identities are required")
    plan = _json(_artifact(root, lock.get("calendar_grid_plan")), "calendar plan")
    expected = {role: {"month": month, "origins": calendar_origins(month, role)}
                for month, role in (("2026-01", "train"), ("2026-04", "validation"), ("2026-05", "test"))}
    if not isinstance(plan, dict) or plan.get("grids") != expected:
        raise SealedDataError("locked calendar plan differs from the data-free frozen grids")
    history = _json(_artifact(root, lock.get("selection_history")), "selection history")
    if (not isinstance(history, dict) or history.get("rule") != "small.max_drawdown_mae"
            or not isinstance(history.get("candidates"), list) or not history["candidates"]):
        raise SealedDataError("C selection history and its MDD rule are required")
    candidates, epochs, originals, gates = history["candidates"], [], [], {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise SealedDataError("candidate record is invalid")
        epoch = candidate.get("epoch")
        if (isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 2
                or candidate.get("name") != f"epoch_{epoch:02d}" or not _finite(candidate.get("small_mae"))):
            raise SealedDataError("candidate epoch/name/MDD MAE is invalid")
        original, eligible = _candidate_report(root, candidate)
        originals.append(original)
        gates[epoch] = eligible
        epochs.append(epoch)
    planned = history.get("planned_epochs")
    if (epochs != list(range(1, max(epochs) + 1)) or isinstance(planned, bool) or not isinstance(planned, int)
            or not max(epochs) <= planned <= 2 or any(x != originals[0] for x in originals)
            or history.get("original_small_mae") != originals[0]):
        raise SealedDataError("selection history has inconsistent epochs, budget or original baseline")
    best = min(candidates, key=lambda c: (c["small_mae"], c["epoch"]))
    if (history.get("selected") != best or best["epoch"] != lock["epoch"]
            or best["validation_report"] != lock.get("validation_report") or best["small_mae"] >= originals[0]
            or gates[best["epoch"]] is not True):
        raise SealedDataError("selected checkpoint must be minimum small MDD, ties earlier, and strictly beat original")
    adapter, merged = Path(lock["selected_checkpoint"]["path"]), Path(lock["selected_merged_checkpoint"]["path"])
    if (adapter.name != "adapter_model.safetensors" or adapter.parent.name != "adapter" or adapter.parent.parent.name != best["name"]
            or merged.name != "model.safetensors" or merged.parent.name != "merged" or merged.parent.parent != adapter.parent.parent):
        raise SealedDataError("selected adapter and merged paths must identify the same selected epoch")
    return config


def _reviewed_lock(lock, *, root=None):
    config = validate_selection_lock(lock, root=root)
    root = Path(root or ROOT).resolve()
    history = Path(lock["selection_history"]["path"])
    history = history if history.is_absolute() else root / history
    directory = history.parent
    lock_path = _safe_path(root, directory / "selection-lock.json")
    try:
        raw_lock = lock_path.read_bytes()
    except OSError as exc:
        raise SealedDataError("persisted selection lock is unavailable") from exc
    if _json(raw_lock, "persisted selection lock") != lock:
        raise SealedDataError("supplied selection lock differs from the persisted review target")
    review_path = _safe_path(root, directory / "root-review.json")
    try:
        review_raw = review_path.read_bytes()
    except OSError as exc:
        raise SealedDataError("passed root review is unavailable; May remains sealed") from exc
    review = _json(review_raw, "root review")
    if (not isinstance(review, dict) or review.get("passed") is not True
            or review.get("selection_lock_sha256") != _sha(raw_lock)):
        raise SealedDataError("May requires a passed root review of this exact persisted lock")
    reviewed_at = _utc(review.get("reviewed_at_utc"))
    if not _utc(lock["locked_at_utc"]) < reviewed_at <= datetime.now(timezone.utc):
        raise SealedDataError("root review must follow lock creation and precede the May claim")
    return config, _sha(raw_lock), _sha(review_raw)


def _claim_may(lock, *, root=None):
    config, lock_hash, review_hash = _reviewed_lock(lock, root=root)
    root = Path(root or ROOT).resolve()
    receipt = {"status": "claimed", "selection_lock_sha256": lock_hash, "root_review_sha256": review_hash,
               "selected_epoch": lock["epoch"], "claimed_at_utc": datetime.now(timezone.utc).isoformat()}
    path = _safe_path(root, root / "lora_c/MAY_UNSEALED.json")
    try:
        with path.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise SealedDataError("May already claimed; reruns require separate engineering review") from None
    return config, receipt


def validate_existing_may_claim(lock, receipt, *, root=None):
    # Acquisition is only reachable with the loader's already-created receipt.
    if not isinstance(receipt, dict) or receipt.get("status") != "claimed":
        raise SealedDataError("May download requires the controlled loader's prior claim")
    _, lock_hash, review_hash = _reviewed_lock(lock, root=root)
    root = Path(root or ROOT).resolve()
    stored = _json(_safe_path(root, root / "lora_c/MAY_UNSEALED.json").read_bytes(), "May claim")
    if stored != receipt or stored.get("selection_lock_sha256") != lock_hash or stored.get("root_review_sha256") != review_hash:
        raise SealedDataError("May claim does not identify the reviewed lock")


def consume_may_acquisition(lock, receipt, *, root=None):
    """Consume a reviewed receipt once before any May directory is probed."""
    validate_existing_may_claim(lock, receipt, root=root)
    root = Path(root or ROOT).resolve()
    marker = _safe_path(root, root / "lora_c/MAY_ACQUISITION_STARTED.json")
    try:
        with marker.open("x") as stream:
            json.dump({"selection_lock_sha256": receipt["selection_lock_sha256"],
                       "started_at_utc": datetime.now(timezone.utc).isoformat()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise SealedDataError("May acquisition already started; old receipts cannot replay it") from None


def load_month(month, role=None, *, root=None, selection_lock=None):
    role = _role(month, role)
    origins = calendar_origins(month, role)
    receipt = None
    if month == "2026-05":
        config, receipt = _claim_may(selection_lock, root=root)
    else:
        project = Path(root or ROOT).resolve()
        config = _json(_safe_path(project, project / "lora_c/config.json").read_bytes(), "C config")
        _validate_config(config)
    root = Path(root or ROOT).resolve()
    if month == "2026-01":
        directory = _safe_path(root, root / "research/data-probe/binance-2026-01")
    else:
        from .acquire import acquire_month
        acquire_month(month, root=root, selection_lock=selection_lock, receipt=receipt)
        directory = _safe_path(root, root / "lora_c/data" / f"binance-{month}")
    manifest_raw = _safe_path(root, directory / "manifest.json").read_bytes()
    manifest = _json(manifest_raw, "month manifest")
    start, rows = _month_grid(month)
    if (not isinstance(manifest, dict) or manifest.get("month") != month or manifest.get("status") != "verified"
            or manifest.get("verified_assets") != 10 or manifest.get("failed_assets") != 0
            or manifest.get("expected_rows_per_asset") != rows or manifest.get("interval_seconds") != 60
            or manifest.get("quote_currency") != "USDT" or not isinstance(manifest.get("assets"), list)
            or len(manifest["assets"]) != 10 or set(manifest["assets"]) != set(ASSETS)):
        raise DataError("manifest does not describe the complete frozen C month")
    results = manifest.get("results")
    if (not isinstance(results, list) or len(results) != 10 or any(not isinstance(r, dict) for r in results)
            or {r.get("asset") for r in results} != set(ASSETS)):
        raise DataError("manifest must identify each asset exactly once")
    entries, arrays, hashes = {r["asset"]: r for r in results}, [], {}
    for asset in ASSETS:
        entry = entries[asset]
        filename = f"{asset}USDT-1m-{month}.csv"
        if (entry.get("status") != "verified" or entry.get("month") != month or entry.get("pair") != asset + "USDT"
                or entry.get("rows") != rows or entry.get("csv_filename") != filename
                or entry.get("official_zip_sha256_verified") is not True or entry.get("zip_crc_verified") is not True):
            raise DataError("asset manifest identity/acquisition verification is invalid")
        raw = _safe_path(root, directory / filename).read_bytes()
        if _sha(raw) != entry.get("csv_sha256") or len(raw) != entry.get("csv_bytes"):
            raise DataError("CSV differs from its verified hash/size")
        arrays.append(_parse_csv(raw, month, expected_rows=rows))
        hashes[asset] = _sha(raw)
    starts = [start + timedelta(minutes=i) for i in range(rows)]
    stamps = np.asarray([[t.minute, t.hour, t.weekday(), t.day, t.month] for t in starts], dtype=np.float32)
    times = [(t + timedelta(minutes=1)).isoformat().replace("+00:00", "Z") for t in starts]
    raw_values = np.stack(arrays)
    values = raw_values.astype(np.float32)
    for array in (raw_values, values, stamps):
        array.setflags(write=False)
    return MonthData(month, ASSETS.copy(), values, stamps, times, origins,
                     {"role": role, "manifest_sha256": _sha(manifest_raw), "csv_sha256": hashes,
                      "grid": {"origins": origins.copy(), "n": len(origins), "selection": "calendar only before data access"},
                      "rows_per_asset": rows, "split_by": "raw candle-open UTC month"}, raw_values)


def _check_origin(data, origin):
    role = _role(data.month, data.provenance.get("role"))
    if (isinstance(origin, bool) or not isinstance(origin, (int, np.integer)) or origin not in data.origins
            or origin not in calendar_origins(data.month, role) or origin < 255 or origin + 30 >= len(data.times)):
        raise DataError("origin lies outside the complete frozen C role grid")


def window(data, origin):
    _check_origin(data, origin)
    values = data.raw_values if data.raw_values is not None else data.values
    section = slice(origin - 255, origin + 1)
    result = {"schema_version": 1, "profile_id": "binance_" + data.month.replace("-", "_") + "_lora_c",
              "source": "Binance official public spot monthly klines", "mode": "replay", "quote_currency": "USDT",
              "interval_seconds": 60, "assets": data.assets.copy(), "as_of": data.times[origin],
              "quality": {"input_complete": True, "calendar_features": "candle starts", "public_times": "candle ends"},
              "histories": {asset: [{"time": t, **{f: float(v) for f, v in zip(FIELDS, row)}}
                                    for t, row in zip(data.times[section], values[a, section])]
                            for a, asset in enumerate(data.assets)}}
    result["window_id"] = "window_" + _sha(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return result


def truth(data, origin):
    _check_origin(data, origin)
    values = data.raw_values if data.raw_values is not None else data.values
    return {asset: {field: values[a, origin + 1:origin + 31, f].astype(float).tolist()
                    for f, field in enumerate(FIELDS)} for a, asset in enumerate(data.assets)}


def training_window(data, origin, asset):
    if data.month != "2026-01" or data.provenance.get("role", "train") != "train":
        raise DataError("C training is restricted to January 1-25")
    _check_origin(data, origin)
    section = slice(origin - 255, origin + 31)
    return data.values[data.assets.index(asset), section].copy(), data.stamps[section].copy()
