"""Forecast close paths exclusively from the passed historical MarketWindow."""
from __future__ import annotations

import numpy as np


def _replay(spot, returns):
    with np.errstate(over="raise", invalid="raise"):
        path = float(spot) * np.exp(np.cumsum(returns))
    if path.shape != (30,) or not np.all(np.isfinite(path)) or np.any(path <= 0):
        raise ValueError("baseline produced invalid close path")
    return path.tolist()


def predict_baselines(window, origin_assets):
    assets = window["assets"]
    if not origin_assets or not set(origin_assets).issubset(assets):
        raise ValueError("complete known origin_assets required")
    closes = np.array([[r["close"] for r in window["histories"][a]] for a in assets], dtype=float).T
    if closes.shape != (256, len(assets)) or np.any(closes <= 0) or not np.all(np.isfinite(closes)):
        raise ValueError("256 positive finite historical closes per asset required")
    returns = np.diff(np.log(closes), axis=0)
    paths = {"historical_30": {}, "no_propagation": {}}
    for i, a in enumerate(assets):
        paths["historical_30"][a] = _replay(closes[-1, i], returns[-30:, i])
        paths["no_propagation"][a] = (list(paths["historical_30"][a]) if a in origin_assets
                                        else [float(closes[-1, i])] * 30)
    btc_index = assets.index("BTC")
    btc = returns[:, btc_index]
    variance = float(np.var(btc, ddof=0))
    metadata = {"training_or_future_data_used": False, "historical_return_count": 255,
                "driver": "replay past30 BTC one-minute returns, not future truth",
                "btc_variance": variance, "beta": {}}
    errors = {}
    if variance == 0:
        errors["btc_beta"] = "BTC historical variance is zero; beta unavailable"
    else:
        beta = np.mean((returns-returns.mean(axis=0)) * (btc-btc.mean())[:, None], axis=0) / variance
        try:
            paths["btc_beta"] = {a: _replay(closes[-1, i], beta[i] * btc[-30:]) for i, a in enumerate(assets)}
            metadata["beta"] = {a: float(beta[i]) for i, a in enumerate(assets)}
        except (FloatingPointError, ValueError) as exc:
            paths.pop("btc_beta", None)
            errors["btc_beta"] = str(exc)
    return {"paths": paths, "metadata": metadata, "errors": errors}
