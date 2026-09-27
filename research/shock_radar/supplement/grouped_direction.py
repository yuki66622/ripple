"""Trigger-side risk comparison and terminal-direction controls, pure arithmetic."""
from statistics import mean

METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]


def trigger_group(direction):
    return "up" if direction == 1 else "down" if direction == -1 else "mixed_or_unknown"


def by_trigger(rows):
    groups = {}
    for group in ("up", "down", "mixed_or_unknown"):
        selected = [r for r in rows if trigger_group(r["direction"]) == group]
        methods = {}
        for m in METHODS:
            scores = [r["saved_score"]["methods"][m]["metrics"] for r in selected]
            methods[m] = {
                "volatility_top3_recall_expected": mean(s["volatility"]["top3_recall_expected"] for s in scores) if scores else None,
                "volatility_mae_bps": mean(s["volatility"]["mae_bps"] for s in scores) if scores else None,
                "max_drawdown_mae_bps": mean(s["max_drawdown"]["mae_bps"] for s in scores) if scores else None,
            }
        groups[group] = {"event_count": len(selected), "event_ids": [r["event_id"] for r in selected], "methods": methods}
    advantages = {}
    for m in METHODS[1:]:
        contributions = {}
        for group, values in groups.items():
            n = values["event_count"]
            delta = (values["methods"][m]["max_drawdown_mae_bps"] - values["methods"]["kronos_base"]["max_drawdown_mae_bps"]) if n else None
            contributions[group] = {"event_count": n, "within_group_advantage_bps": delta,
                                    "contribution_to_all_event_advantage_bps": n / len(rows) * delta if n else None}
        advantages[m] = {"groups": contributions,
                         "total_advantage_bps": sum(c["contribution_to_all_event_advantage_bps"] or 0 for c in contributions.values()) if rows else None}
    return {"groups": groups, "mdd_advantage_decomposition": advantages}


def sign_terminal(spot, closes):
    # Comparing prices gives precisely the sign of the terminal simple return,
    # without cancellation from division near zero.
    return int(closes[-1] > spot) - int(closes[-1] < spot)


def directions(rows):
    pairs = []
    for row in rows:
        known = row["direction"] in (-1, 1)
        for asset in row["assets"]:
            truth = sign_terminal(row["spots"][asset], row["actual"][asset])
            predicted = {m: sign_terminal(row["spots"][asset], row["paths"][m][asset]) for m in METHODS}
            predicted.update(always_continue=row["direction"] if known else None,
                             always_reverse=-row["direction"] if known else None)
            pairs.append({"event_id": row["event_id"], "asset": asset, "actual_sign": truth,
                          "event_direction": row["direction"], "predicted": predicted,
                          "common_binary_eligible": known and truth != 0})
    common = [p for p in pairs if p["common_binary_eligible"]]
    results = {}
    for m in [*METHODS, "always_continue", "always_reverse"]:
        correct = sum(p["predicted"][m] == p["actual_sign"] for p in common)
        valid = [p for p in pairs if p["predicted"][m] is not None]
        nonzero = [p for p in valid if p["actual_sign"] != 0]
        results[m] = {"common_correct": correct, "common_pairs": len(common),
                      "common_binary_accuracy": correct / len(common) if common else None,
                      "common_predicted_flat": sum(p["predicted"][m] == 0 for p in common),
                      "all_three_sign_correct": sum(p["predicted"][m] == p["actual_sign"] for p in valid),
                      "all_three_sign_pairs": len(valid),
                      "all_nonzero_truth_correct": sum(p["predicted"][m] == p["actual_sign"] for p in nonzero),
                      "all_nonzero_truth_pairs": len(nonzero),
                      "unavailable_pairs": len(pairs) - len(valid)}
    model_accuracy = results["kronos_base"]["common_binary_accuracy"]
    return {"per_pair": pairs, "aggregate": {
        "event_count": len(rows), "all_pairs": len(pairs), "common_pairs": len(common),
        "true_flat_pairs": sum(p["actual_sign"] == 0 for p in pairs),
        "mixed_or_unknown_trigger_pairs": sum(p["event_direction"] not in (-1, 1) for p in pairs),
        "methods": results, "random_binary_expected_accuracy": 0.5 if common else None,
        "near_chance_range": [0.45, 0.55],
        "interpretation": "无方向预测能力（本样本未证实），不进 pitch" if model_accuracy is not None and 0.45 <= model_accuracy <= 0.55 else "非五五开也仅为描述性结果，不能由单月样本证明方向预测能力",
        "aggregation": "Pooled asset-event pairs; correlated within/across events, not independent sample count."}}
