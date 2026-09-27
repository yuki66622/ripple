"""Closed Binance Spot minute candles for the integrated live demo.

Only public market data; no model, key, file reads or synthetic filling.
The exchange clock freezes one completed-minute boundary for all ten requests.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from statistics import pstdev
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC")
BASE_URL = "https://data-api.binance.vision/api/v3/"
SOURCE = "Binance Spot public market data"
PROFILE_ID = "binance_live_10assets"
LOOKBACK = 256
INTERVAL_MS = 60_000
TIMEOUT_SECONDS = 8
MAX_RESPONSE_BYTES = 500_000
STALE_AFTER_SECONDS = 180
VOLATILITY_RECENT_RETURNS = 10
VOLATILITY_BASELINE_RETURNS = 120
VOLATILITY_THRESHOLD = 1.2


class MarketError(ValueError):
    """Public, bounded feed failure; callers must not publish a partial market."""


def _utcnow():
    return datetime.now(timezone.utc)


def _iso_ms(milliseconds):
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_constant(_):
    raise ValueError("Nonfinite JSON constant")


def _request_json(path, params=None):
    url = BASE_URL + path + ("?" + urlencode(params) if params else "")
    request = Request(url, headers={"User-Agent": "Ripple-HackUMBC-public-market/1.0"})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise MarketError("公开行情响应超过大小限制。")
        return json.loads(raw, parse_constant=_reject_constant)
    except HTTPError as exc:
        raise MarketError(f"公开行情接口返回 HTTP {exc.code}。") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise MarketError("公开行情暂时连接失败，请重试。") from exc
    except (ValueError, UnicodeError) as exc:
        raise MarketError("公开行情返回的数据格式无效。") from exc


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise MarketError(f"{label}必须是整数时间戳。")
    return value


def _parse_rows(raw, asset, cutoff_ms):
    if not isinstance(raw, list) or len(raw) != LOOKBACK:
        raise MarketError(f"{asset} 缺少完整的 {LOOKBACK} 根已收盘分钟行情。")
    result = []
    first_open = cutoff_ms - LOOKBACK * INTERVAL_MS
    for index, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != 12:
            raise MarketError(f"{asset} 行情字段不完整。")
        opened = _integer(row[0], asset)
        closed = _integer(row[6], asset)
        expected_open = first_open + index * INTERVAL_MS
        if opened != expected_open or closed != opened + INTERVAL_MS - 1:
            raise MarketError(f"{asset} 分钟行情缺失、重复、未收盘或未对齐。")
        try:
            source_values = [row[1], row[2], row[3], row[4], row[5], row[7]]
            if any(isinstance(value, bool) for value in source_values):
                raise ValueError("Boolean is not a market number")
            values = [float(value) for value in source_values]
        except (TypeError, ValueError, OverflowError) as exc:
            raise MarketError(f"{asset} 行情含非数值字段。") from exc
        if not all(math.isfinite(value) for value in values):
            raise MarketError(f"{asset} 行情含非有限数值。")
        o, h, low, close, volume, amount = values
        if not (0 < low <= o <= h and low <= close <= h and volume >= 0 and amount >= 0):
            raise MarketError(f"{asset} 价格或成交量不合法。")
        if (volume == 0) != (amount == 0):
            raise MarketError(f"{asset} 成交量和成交额零值不一致。")
        result.append({"time": _iso_ms(opened + INTERVAL_MS),
                       **dict(zip(("open", "high", "low", "close", "volume", "amount"), values))})
    return result


def _asset_history(asset, cutoff_ms):
    raw = _request_json("klines", {"symbol": asset + "USDT", "interval": "1m",
                                  "limit": LOOKBACK, "endTime": cutoff_ms - 1})
    return _parse_rows(raw, asset, cutoff_ms)


def _past30(asset, rows):
    prices = [row["close"] for row in rows[-31:]]
    returns = [math.log(b) - math.log(a) for a, b in zip(prices, prices[1:])]
    peak, drawdown = prices[0], 0.0
    for price in prices:
        peak = max(peak, price)
        drawdown = max(drawdown, (peak - price) / peak)
    return {"asset": asset, "vol_past30": pstdev(returns) * math.sqrt(30),
            "mdd_past30": drawdown, "last_price": prices[-1],
            "return_past30": prices[-1] / prices[0] - 1}


def _volatility_state(rows):
    """Compare per-minute dispersion in two windows with the same endpoint.

    The baseline includes the latest ten returns. No horizon scaling is
    applied, so the ratio compares like units rather than sqrt(10)/sqrt(120).
    Market ingestion validates timestamps before this helper is called.
    """
    unavailable = {"ratio": None, "state": "unavailable", "recent_std": None,
                   "baseline_std": None, "reason": "invalid_series"}
    if not isinstance(rows, (list, tuple)):
        return unavailable
    if len(rows) < VOLATILITY_BASELINE_RETURNS + 1:
        return {**unavailable, "reason": "insufficient_data"}
    try:
        prices = [row["close"] for row in rows[-(VOLATILITY_BASELINE_RETURNS + 1):]]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in prices):
            return unavailable
        returns = [math.log(b) - math.log(a) for a, b in zip(prices, prices[1:])]
        recent_std = pstdev(returns[-VOLATILITY_RECENT_RETURNS:])
        baseline_std = pstdev(returns)
    except (KeyError, TypeError, ValueError, OverflowError):
        return unavailable
    if baseline_std == 0:
        return {**unavailable, "recent_std": recent_std, "baseline_std": baseline_std,
                "reason": "zero_baseline"}
    ratio = recent_std / baseline_std
    if not math.isfinite(ratio):
        return unavailable
    return {"ratio": ratio, "state": "above_usual" if ratio >= VOLATILITY_THRESHOLD else "near_usual",
            "recent_std": recent_std, "baseline_std": baseline_std, "reason": None}


def fetch_market():
    """Return detached {window, snapshot}; raise MarketError on any bad asset.

    Window contains exactly 256 completed minutes per asset. Metrics use the
    final 31 closes (30 returns), population standard deviation, sqrt(30).
    Volatility state compares population std of the final 10 and 120 one-minute
    log returns. Snapshot exposes the final 61 observed points; no forecast.
    """
    clock = _request_json("time")
    if not isinstance(clock, dict):
        raise MarketError("交易所时钟不可用。")
    server_ms = _integer(clock.get("serverTime"), "交易所时钟")
    if server_ms <= LOOKBACK * INTERVAL_MS:
        raise MarketError("交易所时钟无效。")
    cutoff_ms = server_ms // INTERVAL_MS * INTERVAL_MS
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="market") as pool:
        histories = dict(zip(ASSETS, pool.map(lambda asset: _asset_history(asset, cutoff_ms), ASSETS)))
    window = {"schema_version": 1, "profile_id": PROFILE_ID, "source": SOURCE,
              "mode": "live", "quote_currency": "USDT", "interval_seconds": 60,
              "assets": list(ASSETS), "as_of": _iso_ms(cutoff_ms), "histories": histories,
              "quality": {"amount_source": "exchange_quote_asset_volume", "amount_is_estimate": False,
                          "timestamp_semantics": "UTC candle end (exclusive)", "gap_policy": "reject",
                          "synthetic_candles": 0, "alignment": "exact shared closed interval",
                          "pairs": {asset: asset + "USDT" for asset in ASSETS}}}
    encoded = json.dumps(window, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    window["window_id"] = "window_" + sha256(encoded).hexdigest()
    fetched = _utcnow()
    age = max(0.0, fetched.timestamp() - cutoff_ms / 1000)
    snapshot = {"schema_version": 1, "window_id": window["window_id"], "source": SOURCE,
                "quote_currency": "USDT", "as_of": window["as_of"],
                "fetched_at": fetched.isoformat().replace("+00:00", "Z"), "assets": list(ASSETS),
                "metrics": [{**_past30(asset, histories[asset]),
                             "volatility_state": _volatility_state(histories[asset])}
                            for asset in ASSETS],
                "series": {asset: [{"time": row["time"], "close": row["close"]}
                                   for row in histories[asset][-61:]] for asset in ASSETS},
                "stale": age > STALE_AFTER_SECONDS, "age_seconds": age, "error": None}
    return {"window": window, "snapshot": snapshot}
