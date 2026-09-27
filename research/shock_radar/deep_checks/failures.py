"""Descriptive failure audit using only the 140 frozen event JSON artifacts.

No model, detector, market-data, network, or training imports are used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC")
MAJOR = frozenset(ASSETS[:4])
METHODS = ("kronos_base", "historical_30", "btc_beta", "no_propagation")
INPUT_DIRS = {
    "2026-01": ROOT / "research/shock_radar/artifacts/january-v1/inference-original-v1",
    "2026-02": ROOT / "research/shock_radar/closure/artifacts/february-v1/inference-original-v1",
}
EXPECTED = {"2026-01": 84, "2026-02": 56}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values):
    return mean(values) if values else None


def mdd(prices):
    if not prices or any(not math.isfinite(x) or x <= 0 for x in prices):
        raise ValueError("MDD requires finite positive prices")
    peak = prices[0]
    result = 0.0
    for price in prices:
        peak = max(peak, price)
        result = max(result, (peak - price) / peak)
    return result


def strength_group(z):
    if not math.isfinite(z) or z < 6:
        raise ValueError("Expected retained k=6 event")
    return "6-8sigma" if z < 8 else "8sigma+"


def summarize_event(data, month):
    event, score = data["event"], data["score"]
    if score["status"] != "scored" or score["actual_status"] != "valid":
        raise ValueError("Event is not scored with valid actuals")
    if any(score["methods"][method]["status"] != "scored" for method in METHODS):
        raise ValueError("Not a complete four-method paired event")
    expected_assets = [a for a in ASSETS if event["systemic_flag"] or a not in event["origin_assets"]]
    if score["scored_assets"] != expected_assets or score["scored_asset_count"] != len(expected_assets):
        raise ValueError("Original scored universe mismatch")
    if not expected_assets:
        raise ValueError("Empty scored universe")
    if event["timestamp"][:7] != month:
        raise ValueError("Month mismatch")
    pred = data["predicted_risk"]["kronos_base"]
    actual = data["actual_risk"]
    if len(data["forecast"]["paths"]) != 1:
        raise ValueError("Original frozen forecast must have exactly one path")
    errors, signed = {}, {}
    for asset in expected_assets:
        # Independently reconstruct prediction MDD from frozen close + spot.
        close = data["forecast"]["paths"][0]["assets"][asset]["close"]
        if len(close) != 30:
            raise ValueError("Expected thirty future closes")
        recomputed = mdd([data["forecast"]["spots"][asset], *close])
        p, y = pred[asset]["max_drawdown"], actual[asset]["max_drawdown"]
        if not (math.isfinite(y) and 0 <= y < 1):
            raise ValueError("Invalid actual MDD")
        if not math.isclose(recomputed, p, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Frozen prediction MDD mismatch")
        signed[asset] = (p - y) * 10000
        errors[asset] = abs(p - y) * 10000
    recomputed_mae = mean(errors.values())
    saved_mae = score["methods"]["kronos_base"]["metrics"]["max_drawdown"]["mae_bps"]
    if not math.isclose(recomputed_mae, saved_mae, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Frozen event mean error mismatch")
    major = [errors[a] for a in expected_assets if a in MAJOR]
    small = [errors[a] for a in expected_assets if a not in MAJOR]
    worst = min(expected_assets, key=lambda a: (-errors[a], ASSETS.index(a)))
    direction = {1: "up", -1: "down", 0: "mixed"}.get(event["direction"])
    if direction is None:
        raise ValueError("Invalid direction")
    return {
        "event_id": event["event_id"], "timestamp": event["timestamp"], "month": month,
        "mae_bps": recomputed_mae, "strength_sigma": event["magnitude_sigma"],
        "strength_tier": strength_group(event["magnitude_sigma"]),
        "systemic": event["systemic_flag"], "direction": direction,
        "origin_assets": event["origin_assets"], "scored_assets": expected_assets,
        "scored_asset_count": len(expected_assets), "major_count": len(major), "small_count": len(small),
        "major_mae_bps": safe_mean(major), "small_mae_bps": safe_mean(small),
        "small_mae_gt_major": mean(small) > mean(major) if major and small else None,
        "worst_asset": worst, "worst_asset_tier": "major" if worst in MAJOR else "small",
        "worst_asset_error_bps": errors[worst],
        "mean_signed_error_bps": mean(signed.values()),
        "underpredicted_asset_count": sum(v < 0 for v in signed.values()),
        "absolute_errors_bps": errors, "signed_errors_bps": signed,
    }


def rank_rows(rows):
    return sorted(rows, key=lambda x: (-x["mae_bps"], x["timestamp"], x["event_id"]))


def describe(rows):
    n = len(rows)
    result = {"n_events": n, "mean_event_mae_bps": safe_mean([r["mae_bps"] for r in rows])}
    for field in ("month", "direction", "systemic", "strength_tier", "worst_asset_tier", "small_mae_gt_major"):
        counts = Counter(str(r[field]).lower() if isinstance(r[field], bool) else str(r[field]) for r in rows)
        result[field] = {key: {"count": count, "denominator": n, "fraction": count/n if n else None}
                         for key, count in sorted(counts.items())}
    # Event-level tier means avoid comparing six small assets against four major by sums.
    for tier in ("major", "small"):
        values = [r[f"{tier}_mae_bps"] for r in rows if r[f"{tier}_mae_bps"] is not None]
        result[f"{tier}_mean_event_mae_bps"] = safe_mean(values)
        result[f"{tier}_available_events"] = len(values)
        result[f"{tier}_scored_asset_pairs"] = sum(r[f"{tier}_count"] for r in rows)
    result["mean_signed_error_bps"] = safe_mean([r["mean_signed_error_bps"] for r in rows])
    result["worst_asset_counts"] = dict(sorted(Counter(r["worst_asset"] for r in rows).items()))
    return result


def run():
    paths = {month: sorted(folder.glob("event_*.json")) for month, folder in INPUT_DIRS.items()}
    counts = {m: len(p) for m, p in paths.items()}
    if counts != EXPECTED:
        raise ValueError(f"Expected {EXPECTED}, found {counts}; no silent exclusion")
    before = {str(p.relative_to(ROOT)): sha256(p) for month in paths for p in paths[month]}
    rows, invalid = [], []
    for month, month_paths in paths.items():
        for path in month_paths:
            try:
                row = summarize_event(json.loads(path.read_text()), month)
                row["source"] = str(path.relative_to(ROOT))
                rows.append(row)
            except (KeyError, TypeError, ValueError) as exc:
                invalid.append({"source": str(path.relative_to(ROOT)), "reason": str(exc)})
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate event IDs")
    if before != {p: sha256(ROOT / p) for p in before}:
        raise ValueError("Frozen input changed during audit")
    ordered = rank_rows(rows)
    # Do not select a favorable subset if the declared paired population fails validation.
    top = ordered[:10] if not invalid and len(rows) == 140 else []
    result = {
        "schema": "b-line-failure-audit-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(top) == 10 else "invalid_input_no_top10_published",
        "model_calls": 0, "raw_market_data_reads": 0,
        "population": {"expected_events": 140, "found_by_month": counts, "validated_events": len(rows),
                       "invalid_events": len(invalid), "invalid": invalid, "missing_events": 140-sum(counts.values())},
        "definition": "Rank original event-level Kronos MDD MAE over original scored universe; systemic=all10, otherwise exclude all origins. Descending error, timestamp+ID tie break. Tiers BTC/ETH/SOL/BNB vs remaining6; compare per-tier means, not sums. Units bps.",
        "full": describe(rows), "top10": describe(top), "top10_rows": top,
        "all_rows": ordered, "input_sha256": before, "input_hashes_unchanged": True,
        "limitations": ["Postselected descriptive worst cases, not causal inference or an out-of-sample failure classifier.",
                        "Absolute error increases with realized movement scale; composition is compared to all140 rather than asserted from top10 alone.",
                        "Actual MDD is the frozen saved truth; predicted MDD and event MAE are independently recomputed without raw prices."],
    }
    with (OUT / "failures.json").open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    fields = ["rank", "timestamp", "month", "mae_bps", "strength_sigma", "strength_tier", "systemic", "direction",
              "major_mae_bps", "major_count", "small_mae_bps", "small_count", "worst_asset", "worst_asset_error_bps", "event_id"]
    with (OUT / "top10.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(top, 1):
            writer.writerow({"rank": rank, **{key: row[key] for key in fields if key != "rank"}})
    print(json.dumps({"status": result["status"], "population": result["population"], "full": result["full"], "top10": result["top10"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
