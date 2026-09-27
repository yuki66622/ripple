"""Pure arithmetic. Decimal returns (0.03 = 3%); no model or network calls."""

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from math import fsum, isclose, isfinite, log, sqrt
from statistics import mean, pstdev

VERSION = "deterministic-metrics-v1"
PAIRING = "paired_scenarios_not_calibrated_joint_distribution"
ASSET_METRICS = ("terminal_return", "pnl", "max_drawdown", "volatility",
                 "predicted_high", "predicted_low", "amplitude")
PORTFOLIO_METRICS = ("terminal_return", "pnl", "max_drawdown")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(prefix, value):
    return prefix + sha256(_json(value).encode()).hexdigest()


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number")
    if not isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label}: expected a finite {'positive ' if positive else ''}number")
    return float(value)


def _time(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be an ISO string with timezone")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamp requires timezone")
    return dt


@dataclass(frozen=True)
class FrozenInput:
    """JSON string storage prevents callers mutating nested prediction arrays."""
    canonical_json: str
    identity: str

    @property
    def data(self):
        return json.loads(self.canonical_json)


def freeze_forecast(data):
    """Freeze aligned high/low/close paths; reject rather than repair bad data.

    `as_of` is the observation/spot anchor. `times` are future close instants.
    Each path explicitly supplies all assets on that shared time grid.
    Metadata must identify model, source, quote currency and sampling run.
    """
    data = json.loads(_json(data))
    for field in ("model_revision", "source", "quote_currency", "prediction_run_id"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"Missing {field}")
    if data.get("pairing") != PAIRING:
        raise ValueError("Explicit scenario pairing semantics required")
    anchor = _time(data["as_of"])
    interval = _number(data["interval_seconds"], "interval", positive=True)
    times = [_time(t) for t in data["times"]]
    if len(times) < 2:
        raise ValueError("At least two future closes required for volatility")
    for index, timestamp in enumerate(times, 1):
        if not isclose((timestamp - anchor).total_seconds(), index * interval,
                       rel_tol=0, abs_tol=1e-6):
            raise ValueError("Forecast time grid must be regular and start after as_of")
    spots = data["spots"]
    if not isinstance(spots, dict) or not spots:
        raise ValueError("Nonempty asset spots required")
    for symbol, spot in spots.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Nonempty asset symbols required")
        _number(spot, "spot", positive=True)
    if not isinstance(data["paths"], list) or not data["paths"]:
        raise ValueError("At least one scenario path required")
    seen = set()
    for path in data["paths"]:
        pid = path["path_id"]
        if not isinstance(pid, str) or not pid or pid in seen:
            raise ValueError("Unique nonempty path_id required")
        seen.add(pid)
        if set(path["assets"]) != set(spots):
            raise ValueError("Each scenario must contain the same asset set")
        for series in path["assets"].values():
            for field in ("close", "high", "low"):
                values = series[field]
                if not isinstance(values, list) or len(values) != len(times):
                    raise ValueError("Every price series must match the time grid")
                for value in values:
                    _number(value, field, positive=True)
            for low, close, high in zip(series["low"], series["close"], series["high"]):
                if not low <= close <= high:
                    raise ValueError("Predicted low <= close <= high required")
    return FrozenInput(_json(data), _id("forecast_", data))


def make_holdings(forecast, capital=10000, weights=None):
    """Convert weights to quantities ONCE. Reuse this object on forecast refresh."""
    data = forecast.data
    weights = dict(weights if weights is not None else {"BTC": .5, "ETH": .3, "SOL": .2})
    capital = _number(capital, "capital", positive=True)
    if set(weights) != set(data["spots"]):
        raise ValueError("Weights must match forecast assets")
    for weight in weights.values():
        if _number(weight, "weight") < 0:
            raise ValueError("Long-only weights required")
    if not isclose(fsum(weights.values()), 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Weights must sum to 1")
    config = {
        "conversion_as_of": data["as_of"], "conversion_forecast_id": forecast.identity,
        "quote_currency": data["quote_currency"], "initial_capital": capital,
        "weights": weights, "conversion_prices": data["spots"],
        "quantities": {s: capital * weights[s] / p for s, p in data["spots"].items()},
    }
    for q in config["quantities"].values():
        _number(q, "quantity")
    return FrozenInput(_json(config), _id("holdings_", config))


def _drawdown(values):
    peak, worst = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst


def asset_metrics(spot, close, high, low, quantity):
    """Internal validated path arithmetic; validation lives at snapshot boundary."""
    prices = [spot, *close]
    returns = [log(b) - log(a) for a, b in zip(prices, prices[1:])]
    return {
        "terminal_return": (close[-1] - spot) / spot,
        "pnl": quantity * (close[-1] - spot),
        "max_drawdown": _drawdown(prices),
        "volatility": pstdev(returns) * sqrt(len(returns)),
        "predicted_high": max(high), "predicted_low": min(low),
        "amplitude": (max(high) - min(low)) / spot,
    }


def _summary(values):
    ordered = sorted(values)

    def quantile(q):
        position = (len(ordered) - 1) * q
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        fraction = position - left
        return ordered[left] * (1 - fraction) + ordered[right] * fraction

    return {"mean": mean(values), "p05": quantile(.05), "p50": quantile(.5),
            "p95": quantile(.95)}


def recalculate(forecast, holdings):
    """All paths first, THEN scalar metric summaries. No averaged price baseline."""
    data, config = forecast.data, holdings.data
    spots, quantities = data["spots"], config["quantities"]
    if set(spots) != set(quantities) or config["quote_currency"] != data["quote_currency"]:
        raise ValueError("Holdings assets and quote currency must match forecast")
    if _time(config["conversion_as_of"]) > _time(data["as_of"]):
        raise ValueError("Holdings conversion cannot postdate forecast anchor")
    for quantity in quantities.values():
        if _number(quantity, "quantity") < 0:
            raise ValueError("Long-only quantities required")
    current_value = fsum(quantities[s] * spots[s] for s in spots)
    _number(current_value, "current value", positive=True)
    rows = []
    for path in data["paths"]:
        assets = {s: asset_metrics(spots[s], series["close"], series["high"],
                                   series["low"], quantities[s])
                  for s, series in path["assets"].items()}
        values = [current_value] + [
            fsum(quantities[s] * path["assets"][s]["close"][i] for s in spots)
            for i in range(len(data["times"]))]
        contribution = {s: assets[s]["pnl"] for s in spots}
        pnl = fsum(contribution.values())
        rows.append({"path_id": path["path_id"], "assets": assets, "portfolio": {
            "value_path": values, "terminal_return": pnl / current_value,
            "pnl": pnl, "max_drawdown": _drawdown(values),
            "pnl_contributions": contribution}})
    summary = {
        "assets": {s: {metric: _summary([r["assets"][s][metric] for r in rows])
                       for metric in ASSET_METRICS} for s in spots},
        "portfolio": {metric: _summary([r["portfolio"][metric] for r in rows])
                      for metric in PORTFOLIO_METRICS},
        "pnl_contributions": {s: _summary([r["portfolio"]["pnl_contributions"][s]
                                          for r in rows]) for s in spots},
    }
    result = {
        "forecast_id": forecast.identity, "holdings_id": holdings.identity,
        "metric_version": VERSION, "quote_currency": data["quote_currency"],
        "as_of": data["as_of"], "times": [data["as_of"], *data["times"]],
        "model_revision": data["model_revision"], "source": data["source"],
        "semantics": data["pairing"], "sample_count": len(rows),
        "units": "returns/drawdown/volatility/amplitude: decimal; prices/PnL: quote currency",
        "holdings": config, "current_value": current_value, "paths": rows, "summary": summary,
    }
    result["metrics_id"] = _id("metrics_", result)
    return result


def evaluate_alerts(metrics, drawdown_threshold=.03, top_k=2):
    """Threshold-only operation over cached metrics; ties use competition rank."""
    threshold = _number(drawdown_threshold, "drawdown threshold")
    if not 0 <= threshold <= 1 or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("Threshold must be in [0,1]; top_k must be a nonnegative integer")
    assets = metrics["summary"]["assets"]
    vols = {s: row["volatility"]["mean"] for s, row in assets.items()}
    decisions = {}
    for symbol, row in assets.items():
        rank = 1 + sum(v > vols[symbol] for v in vols.values())
        reasons = []
        if row["max_drawdown"]["mean"] > threshold:
            reasons.append("mean_drawdown_exceeds_threshold")
        if rank <= top_k:
            reasons.append("mean_volatility_top_rank")
        decisions[symbol] = {"volatility_rank": rank, "reasons": reasons,
                             "needs_review": bool(reasons),
                             "label": "需优先检查" if reasons else "未触发检查标记"}
    return {
        "forecast_id": metrics["forecast_id"], "metrics_id": metrics["metrics_id"],
        "policy": {"drawdown_threshold": threshold, "top_k": top_k,
                   "aggregation": "mean_of_path_metrics", "ties": "competition_rank"},
        "assets": decisions,
        "portfolio": {"needs_review": metrics["summary"]["portfolio"]["max_drawdown"]["mean"] > threshold},
    }


def report_payload(metrics, alerts):
    """Numeric source for a future report API; does not call an LLM."""
    if any(metrics[key] != alerts[key] for key in ("forecast_id", "metrics_id")):
        raise ValueError("Report cannot mix different forecasts or holdings calculations")
    return json.loads(_json({"forecast_id": metrics["forecast_id"], "metrics": metrics,
                            "alerts": alerts,
                            "disclosure": "预测条件下的算术结果；采样分位数不是已校准概率。"
                            "固定数量持仓，未计手续费、滑点、交易或再平衡。"}))
