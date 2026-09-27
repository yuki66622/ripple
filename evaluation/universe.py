"""Declared evaluation universes and explicit experimental portfolio policy."""

DEFAULT_ASSETS = ("BTC", "ETH", "SOL")
TEN_ASSETS = (*DEFAULT_ASSETS, "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC")
TEN_PROFILE = "binance_jan2025_10assets"
LOCAL_PROFILES = ("binance_jan2025", TEN_PROFILE)
PROFILE_ASSETS = {"binance_jan2025": DEFAULT_ASSETS, "kraken_live": DEFAULT_ASSETS, TEN_PROFILE: TEN_ASSETS}


def asset_list(value):
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(s, str) or not s or s == "equal_weight_assets" for s in value)
            or len(set(value)) != len(value)):
        raise ValueError("an ordered nonempty unique asset list is required")
    return tuple(value)


def portfolio_policy(window):
    assets = asset_list(window["assets"])
    if window.get("profile_id") == TEN_PROFILE:
        if assets != TEN_ASSETS:
            raise ValueError("the ten-asset experiment requires its frozen ordered universe")
        return {"kind": "equal_weight_experiment", "initial_capital": 10000,
                "weights": {s: 1 / len(assets) for s in assets}}
    if set(assets) != set(DEFAULT_ASSETS):
        raise ValueError("a non-default universe requires an explicit experimental portfolio policy")
    return None  # Preserve the original engine's 50/30/20 defaults and IDs.


def report_assets(report):
    """New reports declare their universe; old known profiles remain readable."""
    declared = report.get("assets")
    if declared is not None:
        return asset_list(declared)
    known = PROFILE_ASSETS.get(report.get("profile_id"))
    if known is not None:
        return known
    raise ValueError("verified report is missing its declared asset universe")
