"""Extract frozen check-5 results for the three existing replay events.

No raw prices, inference or analysis pipeline is read or executed. The 1.2
threshold and ten-point check are frozen in seven_checks/CONTRACT.md section 5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/shock_radar/seven_checks/result_3_5_7.json"
RADAR = ROOT / "ripple_web/data/radar.json"
OUTPUT = ROOT / "ripple_web/data/duration.json"
TIERS = {
    "major": ("大币组", ["BTC", "ETH", "SOL", "BNB"]),
    "small": ("小币组", ["XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC"]),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_curve(curve, label):
    require(isinstance(curve, list) and len(curve) == 121, f"{label}: expected 121 saved points")
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in curve),
            f"{label}: invalid saved ratio")
    return curve


def extract(source, radar, source_hash):
    decay = source["results"]["analysis_5_decay"]
    require(decay["horizon_minutes"] == list(range(121)), "Unexpected saved horizon")
    require(decay["rolling_return_count"] == 10 and decay["baseline_return_count"] == 120
            and decay["baseline_last_return_offset"] == -5, "Unexpected saved method")
    groups = {}
    for tier, (label, assets) in TIERS.items():
        saved = decay["groups"]["combined"][tier]
        require(saved["n"] + saved["n_missing"] == saved["n_expected"], "Invalid saved count identity")
        groups[tier] = {
            "label": label,
            "assets": assets,
            "n": saved["n"],
            "n_expected": saved["n_expected"],
            "n_missing": saved["n_missing"],
            "reference_minute": saved["median_curve_confirmation_minutes"],
            "event_median_minute": saved["recovery"]["median_confirmation_minutes"],
            "unconfirmed_after_120": saved["recovery"]["n_right_censored_gt_120"],
            "monthly_reference": {
                month: decay["groups"][month][tier]["median_curve_confirmation_minutes"]
                for month in ("2026-01", "2026-02")
            },
            "median_curve": validate_curve(saved["median_curve"], f"combined {tier}"),
        }
    by_id = {event["event_id"]: event for event in decay["events"]}
    require(len(by_id) == len(decay["events"]), "Duplicate saved event IDs")
    require(len(radar["events"]) == 3, "Expected exactly three existing replay events")
    events = {}
    for replay in radar["events"]:
        event_id = replay["event_id"]
        require(event_id not in events, "Duplicate replay event ID")
        require(event_id in by_id, f"Missing saved decay event: {event_id}")
        saved = by_id[event_id]
        require(saved["timestamp"] == replay["timestamp"], "Replay/decay timestamp mismatch")
        event_groups = {}
        for tier in TIERS:
            saved_tier = saved["tiers"][tier]
            curve = saved_tier["curve"]
            if saved_tier["status"] == "valid":
                validate_curve(curve, f"{event_id} {tier}")
            else:
                require(curve is None, "Unavailable tier unexpectedly has a curve")
            # Never export per-event confirmation_minutes: the UI computes revealed prefixes.
            event_groups[tier] = {"status": saved_tier["status"], "curve": curve}
        events[event_id] = {"timestamp": saved["timestamp"], "groups": event_groups}
    return {
        "schema_version": 1,
        "threshold": 1.2,
        "confirmation_points": 10,
        "reference_horizon_minutes": decay["horizon_minutes"][-1],
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": source_hash},
        "groups": groups,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the existing export without writing")
    args = parser.parse_args()
    source_bytes, radar_bytes = SOURCE.read_bytes(), RADAR.read_bytes()
    output = extract(json.loads(source_bytes), json.loads(radar_bytes), hashlib.sha256(source_bytes).hexdigest())
    encoded = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    require(SOURCE.read_bytes() == source_bytes and RADAR.read_bytes() == radar_bytes,
            "A frozen input changed during extraction")
    if args.check:
        require(OUTPUT.read_text() == encoded, "duration.json differs from the frozen extraction")
        print("Duration export verified against frozen check-5 results and all three replay IDs.")
    else:
        OUTPUT.write_text(encoded)
        print(f"Exported {len(output['events'])} replay events; frozen sources unchanged.")


if __name__ == "__main__":
    main()
