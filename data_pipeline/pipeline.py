"""Market data contracts; standard library only, with no model dependency.

All times represent UTC candle END (exclusive right boundary). Binance's
inclusive microsecond close timestamp is validated, then normalized to that
boundary. Historical truth is never part of a MarketWindow.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import zipfile
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ASSETS = ("BTC", "ETH", "SOL")
TEN_ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC")
PROFILE_ASSETS = {"binance_jan2025": ASSETS, "binance_jan2025_10assets": TEN_ASSETS, "kraken_live": ASSETS}
FIELDS = ("open", "high", "low", "close", "volume", "amount")
INTERVAL = 60
REPLAY_RESERVE = 30
DATA_DIR = Path(__file__).resolve().parents[1] / "research/data-probe/binance-2025-01"
KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD"}
KRAKEN_RESULT_KEYS = {
    "BTC": {"XXBTZUSD", "XBTUSD", "BTC/USD", "XBT/USD"},
    "ETH": {"XETHZUSD", "ETHUSD", "ETH/USD"},
    "SOL": {"SOLUSD", "SOL/USD"},
}


class DataError(ValueError):
    """Invalid, incomplete or unavailable data; never silently substitute data."""


def profile_assets(profile_id: str) -> tuple[str, ...]:
    """Ordered frozen universe; no filesystem or network activity."""
    if not isinstance(profile_id, str) or profile_id not in PROFILE_ASSETS:
        raise DataError(f"unknown data profile: {profile_id!r}")
    return PROFILE_ASSETS[profile_id]


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch(value: str) -> int:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise DataError("as_of/time must be an ISO UTC candle-end timestamp") from exc
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise DataError("as_of/time must explicitly use UTC")
    if dt.second or dt.microsecond:
        raise DataError("as_of/time must be an exact one-minute candle end")
    return int(dt.timestamp())


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DataError(f"{name} must be a positive integer")
    return value


def _window_id(window: dict) -> str:
    content = {key: value for key, value in window.items() if key != "window_id"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "window_" + hashlib.sha256(encoded).hexdigest()


def _bar(end: int, values) -> dict:
    try:
        numbers = [float(v) for v in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataError("candle contains a non-numerical value") from exc
    if len(numbers) != 6 or not all(math.isfinite(v) for v in numbers):
        raise DataError("candle must contain six finite OHLCVA values")
    o, h, low, c, volume, amount = numbers
    if min(o, h, low, c) <= 0 or volume < 0 or amount < 0:
        raise DataError("prices must be positive; volume and amount nonnegative")
    if not (low <= o <= h and low <= c <= h):
        raise DataError("invalid OHLC ordering")
    if bool(volume) != bool(amount):
        raise DataError("zero volume and zero amount must agree")
    if end % INTERVAL:
        raise DataError("candle timestamp is not on the one-minute grid")
    return {"time": _iso(end), **dict(zip(FIELDS, numbers))}


def _validate_rows(rows: list, symbol: str) -> list[int]:
    if not rows:
        raise DataError(f"{symbol}: no candles")
    epochs = []
    for row in rows:
        try:
            epoch = _epoch(row["time"])
            _bar(epoch, [row[k] for k in FIELDS])
        except (KeyError, TypeError) as exc:
            raise DataError(f"{symbol}: malformed candle") from exc
        if epochs and epoch - epochs[-1] != INTERVAL:
            raise DataError(f"{symbol}: duplicate, unordered or missing candle at {row['time']}")
        epochs.append(epoch)
    return epochs


def _validate_history(history: dict) -> list[int]:
    assets = profile_assets(history.get("profile_id"))
    if history.get("assets") != list(assets) or set(history.get("histories", {})) != set(assets):
        raise DataError("history must contain the profile's unique assets in the declared order")
    if history.get("interval_seconds") != INTERVAL:
        raise DataError("only one-minute candles are supported")
    expected = None
    for symbol in assets:
        epochs = _validate_rows(history["histories"][symbol], symbol)
        if expected is not None and epochs != expected:
            raise DataError("asset candle timestamps are not exactly aligned")
        expected = epochs
    return expected


def _dataset_paths(profile_id="binance_jan2025") -> tuple:
    paths = []
    for symbol in profile_assets(profile_id):
        stem = f"{symbol}USDT-1m-2025-01"
        # The existing archive is authoritative when both archive and an export exist.
        path = DATA_DIR / f"{stem}.zip"
        if not path.is_file():
            path = DATA_DIR / f"{stem}.csv"
        if not path.is_file():
            raise DataError(f"missing local Binance dataset: {stem}.zip or .csv")
        stat = path.stat()
        paths.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(paths)


def _parse_binance(reader, symbol: str) -> list:
    rows = []
    for line_no, raw in enumerate(reader, 1):
        if len(raw) != 12:
            raise DataError(f"{symbol}: CSV row {line_no} has {len(raw)} columns, expected 12")
        try:
            opened = int(raw[0])
            closed = int(raw[6])
        except ValueError as exc:
            raise DataError(f"{symbol}: invalid integer timestamp at row {line_no}") from exc
        # Binance Spot archives from January 2025 use microseconds.
        if opened % 60_000_000 or closed != opened + 60_000_000 - 1:
            raise DataError(f"{symbol}: expected complete 1m microsecond candle at row {line_no}")
        end = opened // 1_000_000 + INTERVAL
        rows.append(_bar(end, [*raw[1:6], raw[7]]))
    _validate_rows(rows, symbol)
    return rows


@lru_cache(maxsize=4)
def _binance_history(paths: tuple, profile_id="binance_jan2025") -> dict:
    assets = profile_assets(profile_id)
    histories = {}
    if len(paths) != len(assets):
        raise DataError("archive count does not match profile assets")
    for symbol, (filename, _, _) in zip(assets, paths):
        path = Path(filename)
        try:
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    expected = f"{symbol}USDT-1m-2025-01.csv"
                    if archive.namelist() != [expected]:
                        raise DataError(f"{symbol}: archive must contain only {expected}")
                    with archive.open(expected) as file:
                        histories[symbol] = _parse_binance(csv.reader(io.TextIOWrapper(file)), symbol)
            else:
                with path.open(newline="", encoding="utf-8") as file:
                    histories[symbol] = _parse_binance(csv.reader(file), symbol)
        except (OSError, zipfile.BadZipFile, UnicodeError) as exc:
            raise DataError(f"{symbol}: cannot read local Binance data: {type(exc).__name__}") from exc
    history = {
        "schema_version": 1, "profile_id": profile_id,
        "source": "Binance Spot monthly klines 2025-01", "mode": "replay",
        "quote_currency": "USDT", "interval_seconds": INTERVAL,
        "assets": list(assets), "histories": histories,
        "quality": {
            "amount_source": "exchange_quote_asset_volume",
            "amount_is_estimate": False, "timestamp_semantics": "UTC candle end (exclusive)",
            "gap_policy": "reject", "synthetic_candles": 0,
            "pairs": {s: f"{s}USDT" for s in assets},
        },
    }
    _validate_history(history)
    return history


def _request_kraken(symbol: str) -> list:
    url = "https://api.kraken.com/0/public/OHLC?" + urlencode({"pair": KRAKEN_PAIRS[symbol], "interval": 1})
    request = Request(url, headers={"User-Agent": "HackUMBC-market-demo/1.0"})
    try:
        with urlopen(request, timeout=8) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise DataError(f"{symbol}: Kraken response exceeds size limit")
        payload = json.loads(raw)
    except (URLError, HTTPError, TimeoutError, OSError, ValueError) as exc:
        raise DataError(f"{symbol}: Kraken request failed ({type(exc).__name__})") from exc
    if not isinstance(payload, dict) or payload.get("error") != []:
        raise DataError(f"{symbol}: Kraken returned an API error or invalid envelope")
    result = payload.get("result", {})
    keys = [k for k in result if k != "last"]
    if len(keys) != 1 or not isinstance(result[keys[0]], list):
        raise DataError(f"{symbol}: unexpected Kraken result shape")
    if keys[0] not in KRAKEN_RESULT_KEYS[symbol]:
        raise DataError(f"{symbol}: Kraken returned the wrong asset/quote pair")
    return result[keys[0]]


def _parse_kraken(raw_rows: list, symbol: str, closed_cutoff: int) -> list:
    # Kraken explicitly says the last row is current/uncommitted, even with since.
    rows = []
    for raw in raw_rows[:-1]:
        if len(raw) != 8:
            raise DataError(f"{symbol}: Kraken OHLC row must contain 8 values")
        try:
            opening = int(raw[0])
            if opening != float(raw[0]):
                raise ValueError("fractional timestamp")
            end = opening + INTERVAL
            vwap, volume = float(raw[5]), float(raw[6])
        except (ValueError, TypeError, OverflowError) as exc:
            raise DataError(f"{symbol}: invalid Kraken timestamp or volume") from exc
        if end > closed_cutoff:
            continue
        # Both inputs are exchange-rounded: label this estimate everywhere.
        rows.append(_bar(end, [*raw[1:5], volume, vwap * volume]))
    _validate_rows(rows, symbol)
    return rows


def _kraken_history() -> dict:
    closed_cutoff = int(datetime.now(timezone.utc).timestamp()) // INTERVAL * INTERVAL
    with ThreadPoolExecutor(max_workers=3) as pool:
        raw = dict(zip(ASSETS, pool.map(_request_kraken, ASSETS)))
    histories = {s: _parse_kraken(raw[s], s, closed_cutoff) for s in ASSETS}
    first = max(rows[0]["time"] for rows in histories.values())
    last = min(rows[-1]["time"] for rows in histories.values())
    if first > last:
        raise DataError("Kraken assets have no common completed time range")
    source_last = {s: histories[s][-1]["time"] for s in ASSETS}
    histories = {s: [row for row in histories[s] if first <= row["time"] <= last] for s in ASSETS}
    history = {
        "schema_version": 1, "profile_id": "kraken_live", "source": "Kraken Spot REST OHLC",
        "mode": "live", "quote_currency": "USD", "interval_seconds": INTERVAL,
        "assets": list(ASSETS), "histories": histories,
        "quality": {
            "amount_source": "estimated_from_exchange_rounded_vwap_times_volume",
            "amount_is_estimate": True, "timestamp_semantics": "UTC candle end (exclusive)",
            "gap_policy": "reject", "synthetic_candles": 0,
            "pairs": {s: f"{s}/USD" for s in ASSETS},
            "source_latest_closed": source_last, "alignment": "common closed interval only",
        },
    }
    _validate_history(history)
    return history


def _history(profile_id: str) -> dict:
    if profile_id in ("binance_jan2025", "binance_jan2025_10assets"):
        return _binance_history(_dataset_paths(profile_id), profile_id)
    if profile_id == "kraken_live":
        return _kraken_history()
    raise DataError(f"unknown data profile: {profile_id!r}")


def load_history(profile_id: str = "binance_jan2025") -> dict:
    """Load aligned, validated history for evaluation. Caller owns returned data.

    Binance: full January 2025. Kraken: only its recent completed OHLC range;
    this endpoint is not a historical backfill API. No truth/input split here.
    """
    return copy.deepcopy(_history(profile_id))


def _window(history: dict, lookback: int, as_of: str | None) -> dict:
    _positive_int(lookback, "lookback")
    assets = profile_assets(history["profile_id"])
    if history.get("assets") != list(assets) or set(history.get("histories", {})) != set(assets):
        raise DataError("history assets do not match the frozen profile")
    rows = history["histories"][assets[0]]
    ends = [row["time"] for row in rows]
    if as_of is None:
        index = len(rows) - (REPLAY_RESERVE if history["mode"] == "replay" else 0) - 1
    else:
        canonical = _iso(_epoch(as_of))
        index = bisect_left(ends, canonical)
        if index == len(ends) or ends[index] != canonical:
            raise DataError(f"as_of {canonical} is not an available candle end")
    if index < lookback - 1:
        raise DataError(f"not enough input candles for lookback={lookback}")
    window = {key: copy.deepcopy(history[key]) for key in
              ("schema_version", "profile_id", "source", "mode", "quote_currency", "interval_seconds", "assets", "quality")}
    # Acquisition progress must not change the identity of the same past window.
    window["quality"].pop("source_latest_closed", None)
    window["as_of"] = ends[index]
    window["histories"] = {
        s: copy.deepcopy(history["histories"][s][index - lookback + 1:index + 1]) for s in assets
    }
    window["window_id"] = _window_id(window)
    return window


def window_from_history(history: dict, *, lookback: int = 256, as_of: str | None = None) -> dict:
    """Create a validated immutable-by-value input slice from loaded history.

    Useful for evaluation workers receiving complete source data. This public
    entry point revalidates caller-provided history rather than trusting it.
    """
    _validate_history(history)
    return _window(history, lookback, as_of)


def load_window(profile_id: str = "binance_jan2025", *, lookback: int = 256, as_of: str | None = None) -> dict:
    """Return only completed past candles. Default replay leaves 30 truth bars."""
    return _window(_history(profile_id), lookback, as_of)


def load_truth(window: dict, *, horizon: int = 30) -> dict | None:
    """Read future candles separately; None means horizon is not yet complete.

    Unknown origin/out-of-retention inputs and inconsistent provenance raise
    DataError. Partial or gapped truth is never returned as a complete label.
    """
    _positive_int(horizon, "horizon")
    try:
        profile_id = window["profile_id"]
        as_of = _iso(_epoch(window["as_of"]))
        identifier = window["window_id"]
    except (KeyError, TypeError) as exc:
        raise DataError("truth requires a MarketWindow with identity and as_of") from exc
    if identifier != _window_id(window):
        raise DataError("MarketWindow content no longer matches its window_id")
    history = _history(profile_id)
    for key in ("source", "quote_currency", "assets", "interval_seconds"):
        if window.get(key) != history[key]:
            raise DataError(f"truth/input provenance mismatch: {key}")
    assets = profile_assets(profile_id)
    ends = [row["time"] for row in history["histories"][assets[0]]]
    index = bisect_left(ends, as_of)
    if index == len(ends) or ends[index] != as_of:
        raise DataError("forecast origin is outside the available truth history")
    if index + horizon >= len(ends):
        return None
    selected = slice(index + 1, index + horizon + 1)
    return {
        "window_id": identifier, "as_of": as_of, "times": ends[selected],
        "assets": {s: {field: [row[field] for row in history["histories"][s][selected]]
                       for field in FIELDS} for s in assets},
    }
