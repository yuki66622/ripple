"""Build the compact JSON files the Ripple page reads, straight from the frozen research artifacts.

Every number on the page comes from one of these sources; nothing is typed in by hand.
  radar.json        replays-v1 (historical_30 predicted volatility) + demo-multipath-v2/uncertainty.json
  graph.json        closure/artifacts/january-graph-v1/graph.json
  replays.json      replays-v1 (observed, truth) + demo-multipath-v2 event files (10 Kronos paths per asset)
  nearest.json      january-graph-v1/demo-nearest.json + event-classes.csv + fingerprints.json clusters
  evaluation.json   january-v1 summary.json, february-v1 summary.json, supplement/results-v1.json,
                    deep_checks/coverage.json, deep_checks/top10.csv, deep_checks/failures.json
  method.json       lora_a/config.json + the latest lora_a/runs/*/training-log.jsonl (examples per epoch)
  acceptance.json   lora_a/MARCH_UNSEALED.json -> that run's march/sealed-report.json + verdict.json

Run from the project root:  python3 ripple_web/build_data.py
"""
import csv
import glob
import json
import math
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / "research" / "shock_radar"
OUT = Path(__file__).resolve().parent / "data"
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]
METHOD_NAMES = {"kronos_base": "Kronos", "no_propagation": "不传播基线", "btc_beta": "BTC beta 基线", "historical_30": "过去 30 分钟基线"}
REASONS = {"strongest_systemic": "最强系统性冲击", "strongest_non_systemic": "最强单资产冲击", "closest_to_median_trigger_strength": "中位强度冲击"}


def load(path):
    return json.loads(Path(path).read_text())


def quantile(values, q):
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def replay_files():
    return {load(p)["event"]["event_id"]: load(p) for p in sorted(glob.glob(str(SR / "artifacts/january-v1/replays-v1/event_*.json")))}


def event_meta(ev, reason):
    return {"event_id": ev["event_id"], "timestamp": ev["timestamp"], "origin_assets": ev["origin_assets"],
            "direction": ev["direction"], "magnitude_sigma": ev["magnitude_sigma"], "systemic": ev["systemic_flag"],
            "reason": reason, "reason_label": REASONS.get(reason, reason)}


def build_radar_and_replays(replays):
    unc = load(SR / "closure/artifacts/demo-multipath-v2/uncertainty.json")
    radar, out = [], []
    for u in unc["events"]:
        rp = replays[u["event_id"]]
        meta = event_meta(rp["event"], u["choice"]["reason"])
        hist = rp["methods"]["historical_30"]["predicted_risk"]
        radar.append({**meta, "assets": [{
            "asset": a,
            "vol_past30": hist[a]["volatility"],
            "mdd_p05": u["asset_mdd_quantiles"][a]["p05"], "mdd_p50": u["asset_mdd_quantiles"][a]["p50"],
            "mdd_p95": u["asset_mdd_quantiles"][a]["p95"], "mdd_n": u["asset_mdd_quantiles"][a]["n_valid"],
            "actual_mdd": rp["actual_risk"][a]["max_drawdown"],
        } for a in ASSETS]})
        mp = load(SR / f"closure/artifacts/demo-multipath-v2/{u['event_id']}.json")
        paths = mp["forecast"]["paths"]
        assets = {}
        for a in ASSETS:
            closes = [p["assets"][a]["close"] for p in paths]
            steps = list(zip(*closes))
            assets[a] = {
                "observed": rp["observed"]["close_paths"][a],
                "spot": mp["forecast"]["spots"][a],
                "p05": [quantile(s, .05) for s in steps], "p50": [quantile(s, .5) for s in steps], "p95": [quantile(s, .95) for s in steps],
                "actual_mdd": rp["actual_risk"][a]["max_drawdown"], "actual_vol": rp["actual_risk"][a]["volatility"],
                "kronos_mdd": rp["methods"]["kronos_base"]["predicted_risk"][a]["max_drawdown"],
            }
        out.append({**meta, "observed_times": rp["observed"]["times"], "forecast_times": mp["forecast"]["times"],
                    "path_count": len(paths), "assets": assets})
    return ({"source": "replays-v1 historical_30 + demo-multipath-v2/uncertainty.json", "calibrated": False,
             "coverage_note": unc.get("disclosure"), "events": radar},
            {"source": "replays-v1 + demo-multipath-v2", "events": out})


def build_graph():
    g = load(SR / "closure/artifacts/january-graph-v1/graph.json")
    edges = [{k: e[k] for k in ("source", "target", "weight", "count", "mean_delay_minutes", "median_target_magnitude_sigma",
                                "followup_rate", "source_event_count")} for e in g["edges"]]
    seqs = [{"event_id": s["event_id"], "timestamp": s["timestamp"], "origin_assets": s["origin_assets"],
             "initial_magnitude_sigma": s["initial_magnitude_sigma"], "followup_count": s["distinct_followup_asset_count"],
             "steps": [{"delay": st["delay_minutes"], "assets": [t["asset"] for t in st["triggers"] if t.get("present")],
                        "direction": [t.get("direction") for t in st["triggers"] if t.get("present")]} for st in s["sequence"]]}
            for s in g["strongest_three_sequences"]]
    return {"nodes": g["nodes"], "event_count": g["event_count"], "edges": edges, "strongest": seqs,
            "interpretation": g["interpretation"]}


def build_nearest():
    g = SR / "closure/artifacts/january-graph-v1"
    n = load(g / "demo-nearest.json")
    rows = {r["event_id"]: r for r in csv.DictReader(open(g / "event-classes.csv"))}
    fp = load(g / "fingerprints.json")
    # A class name only describes the most frequent origin and the majority trigger sign of its members,
    # so the page shows those counts next to it instead of implying every member started that way.
    classes = [{"id": c["cluster_id"], "ordinal": int(c["cluster_id"].split("_")[-1]), "name": c["name"], "size": c["size"],
                "origins": c["most_frequent_origins"], "origin_count": max(c["origin_counts"].values()),
                "up": c["trigger_sign_counts"].get("1", 0), "down": c["trigger_sign_counts"].get("-1", 0),
                "median_followers": c["median_followup_count"]} for c in fp["clusters"]["clusters"]]
    def top_assets(risk, key, k=3):
        return sorted(risk, key=lambda a: risk[a][key], reverse=True)[:k]
    res = []
    for r in n["results"]:
        full = r["full_historical_event"]["event"]
        actual = r["saved_risk"]["actual_risk"]
        res.append({"distance": r["distance"], "event_id": r["event_id"], "timestamp": r["timestamp"],
                    "origin_assets": r["origin_assets"], "direction": full["direction"], "magnitude_sigma": full["magnitude_sigma"],
                    "class_id": rows[r["event_id"]]["cluster_id"], "prefix": r["observed_prefix_fingerprint"]["slots"],
                    "actual_top_mdd": top_assets(actual, "max_drawdown"),
                    "actual_max_mdd": max(actual[a]["max_drawdown"] for a in actual)})
    q = n["query"]
    return {"query": {"slots": q["vector"]["slots"], "observed_minutes": q["observed_minutes"], "note": q["selection_reason"],
                      "timestamp": rows[q["exclude_event_id"]]["timestamp"]},
            "results": res, "event_count": fp["clusters"]["event_count"], "classes": classes,
            "interpretation": n.get("interpretation")}


def build_evaluation():
    jan_all = load(SR / "artifacts/january-v1/inference-original-v1/summary.json")["groups"]["all"]
    jan = jan_all["scores"]
    table = [{"method": m, "name": METHOD_NAMES[m],
              "vol_top3": jan[m]["volatility"]["top3_recall_expected"], "vol_mae_bps": jan[m]["volatility"]["mae_bps"],
              "mdd_mae_bps": jan[m]["max_drawdown"]["mae_bps"]} for m in METHODS]
    feb_full = load(SR / "closure/artifacts/february-v1/summary.json")
    fs = feb_full["full_metrics"]["groups"]["all"]["scores"]
    dec = feb_full["fixed_historical_30_decomposition"]
    sup = load(SR / "supplement/results-v1.json")
    term = sup["terminal_direction"]["aggregate"]["methods"]["kronos_base"]
    down = sup["downside"]["aggregate"]["methods"]
    cov = load(SR / "deep_checks/coverage.json")
    top10 = list(csv.DictReader(open(SR / "deep_checks/top10.csv")))
    population = load(SR / "deep_checks/failures.json")["population"]
    return {
        "january": {"events": jan_all["common_paired_event_count"], "table": table},
        "february": {"events": feb_full["eligible_event_count"],
                     "kronos_mdd_bps": fs["kronos_base"]["max_drawdown"]["mae_bps"], "hist_mdd_bps": fs["historical_30"]["max_drawdown"]["mae_bps"],
                     "kronos_vol_top3": fs["kronos_base"]["volatility"]["top3_recall_expected"], "hist_vol_top3": fs["historical_30"]["volatility"]["top3_recall_expected"],
                     "down_share": dec["down_share"], "january_down_share": dec["january_reference_down_share_approx"]},
        "nulls": {"direction": {"correct": term["common_correct"], "pairs": term["common_pairs"], "accuracy": term["common_binary_accuracy"]},
                  "downside": {m: {"top3": down[m]["top3_recall_expected"], "mae_bps": down[m]["mae_bps"]} for m in ("kronos_base", "historical_30")},
                  "radius": sup["radius"]["aggregate"]["advantage_radius_minutes"]},
        "coverage": {**cov["pooled"], "conclusion": cov["conclusion"]},
        "top10_population": {"events": population["validated_events"], "by_month": population["found_by_month"]},
        "top10": [{"rank": int(r["rank"]), "timestamp": r["timestamp"], "month": r["month"], "mae_bps": float(r["mae_bps"]),
                   "strength_sigma": float(r["strength_sigma"]), "systemic": r["systemic"] == "True", "direction": r["direction"],
                   "worst_asset": r["worst_asset"], "worst_asset_error_bps": float(r["worst_asset_error_bps"])} for r in top10],
    }


def training_examples():
    """Examples per epoch as logged by the most recent completed LoRA epoch (None if no run is present)."""
    found = None
    for log in sorted(glob.glob(str(ROOT / "lora_a/runs/*/training-log.jsonl")), key=lambda p: Path(p).stat().st_mtime):
        for line in open(log):
            rec = json.loads(line)
            if rec.get("event") == "training_epoch_complete":
                found = rec["stats"]["expected_examples"]
    return found


def build_method():
    cfg = load(ROOT / "lora_a/config.json")
    day = lambda s: date.fromisoformat(s)
    train_last = day(cfg["train_end_exclusive"]) - timedelta(days=1)
    obs_first, obs_end = day(cfg["observation_start"]), day(cfg["observation_end_exclusive"])
    obs_days = (obs_end - obs_first).days
    examples = training_examples()
    n_assets = len(cfg["assets"])
    train_detail = f"{examples:,} 个样本" if examples else f"步长 {cfg['train_stride']} 分钟的窗口"
    return {"segments": [
        {"id": "01", "name": "训练", "range": f"{cfg['train_start']} 至 {train_last:%m-%d}", "detail": train_detail},
        {"id": "02", "name": "选模", "range": cfg["validation_month"], "detail": f"{cfg['validation_windows']} 个窗口，只用来选 checkpoint"},
        {"id": "03", "name": "密封", "range": cfg["test_month"], "detail": f"{cfg['test_windows']} 个窗口，选模后只打开一次"}],
        "observation": (f"{obs_first:%Y-%m-%d} 至 {(obs_end - timedelta(days=1)):%m-%d} 的 "
                        f"{cfg['observation_windows_per_day'] * obs_days} 个观察点只记录曲线，不参与任何选择"),
        "lora": {"rank": cfg["rank"], "alpha": cfg["alpha"], "dropout": cfg["dropout"], "targets": cfg["target_modules"],
                 "lr": cfg["lr"], "batch_size": cfg["batch_size"], "max_epochs": cfg["max_epochs"]},
        "baselines": [METHOD_NAMES[m] for m in METHODS[1:]]}


def build_acceptance():
    """A-line sealed March acceptance: exact top-2 set hit rate per tier, read from the run that consumed the unseal."""
    unseal = load(ROOT / "lora_a/MARCH_UNSEALED.json")
    run = Path(unseal["selected_checkpoint"]["path"]).parents[2]
    rep = load(run / "march/sealed-report.json")
    verdict = load(run / "march/verdict.json")
    cand = f"epoch_{unseal['selected_epoch']:02d}"
    tiers = []
    for key, label in (("small", "小币"), ("major", "大币")):
        t = rep["tiers"][key]
        m = t["methods"]
        tiers.append({
            "tier": key, "label": label, "assets": t["assets"], "n": t["n_scheduled"],
            "lora": m[cand]["exact_set_hit_rate"], "original": m["original"]["exact_set_hit_rate"],
            "dynamic": m["dynamic_naive"]["exact_set_hit_rate"],
            "fixed": m["fixed_pair"]["exact_set_hit_rate"] if "fixed_pair" in m else None,
            "vs": {ref: {"estimate": v["exact_set_hit_rate_difference"]["estimate"], "ci95": v["exact_set_hit_rate_difference"]["ci95"]}
                   for ref, v in t["paired"][cand].items()},
            "drawdown_mae_change": m[cand]["max_drawdown_mae"] / m["original"]["max_drawdown_mae"] - 1,
        })
    return {"month": rep["month"], "candidate": cand, "fixed_pair": rep["fixed_pair"], "passed": verdict["passed"],
            "interval": rep["intervals"], "tiers": tiers}


def main():
    OUT.mkdir(exist_ok=True)
    replays = replay_files()
    radar, rep = build_radar_and_replays(replays)
    files = {"radar.json": radar, "replays.json": rep, "graph.json": build_graph(), "nearest.json": build_nearest(),
             "evaluation.json": build_evaluation(), "method.json": build_method(), "acceptance.json": build_acceptance()}
    for name, obj in files.items():
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"{name:16s} {len((OUT / name).read_bytes()):>8,d} bytes")
    e = files["evaluation.json"]
    print("jan:", [(r["name"], round(r["vol_top3"] * 100, 2), round(r["vol_mae_bps"], 3), round(r["mdd_mae_bps"], 3)) for r in e["january"]["table"]])
    print("feb:", round(e["february"]["kronos_mdd_bps"], 2), round(e["february"]["hist_mdd_bps"], 2), "down share", e["february"]["down_share"])
    print("direction:", e["nulls"]["direction"], "radius:", e["nulls"]["radius"], "coverage:", e["coverage"]["covered"], "/", e["coverage"]["expected"])
    for t in files["acceptance.json"]["tiers"]:
        print(f"03 {t['tier']}: lora {t['lora']:.4%} original {t['original']:.4%} dynamic {t['dynamic']:.4%} fixed {t['fixed']}",
              "vs original", round(t["vs"]["original"]["estimate"] * 100, 3), "drawdown MAE change", f"{t['drawdown_mae_change']:+.2%}")


if __name__ == "__main__":
    main()
