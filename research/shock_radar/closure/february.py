"""Frozen February B-line replication: prepare, single original-model run, report.

Preparation/reporting never import a model. Only --run may load local weights;
the coordinating agent owns that invocation. All artifacts are exclusive writes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile
import time

import numpy as np

from data_pipeline.pipeline import _parse_binance
from research.shock_radar.baselines import predict_baselines
from research.shock_radar.detection import detect_events
from research.shock_radar.metrics import aggregate_events, risk_metrics, score_event

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "research/data-probe/binance-2026-02"
OUTPUT = HERE / "artifacts/february-v1"
JANUARY_REFERENCE = ROOT / "research/shock_radar/artifacts/january-v1/inference-original-v1/run_manifest.json"
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]
CONFIG = {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926}
SAMPLING = {"T": 1.0, "top_k": 0, "top_p": 0.9, "sample_count": 1, "verbose": False}
MANIFEST_SHA256 = "da9c7841090fd376becd9d51517269c8bb3ac419fa0acc84a1c03a922484e24f"
SOURCES = ["research/shock_radar/closure/february.py", "research/shock_radar/detection.py",
           "research/shock_radar/baselines.py", "research/shock_radar/metrics.py",
           "data_pipeline/pipeline.py", "forecast_metrics/engine.py",
           "model_adapter/adapter.py", "model_adapter/corrections.py",
           "model_adapter/volume_quality.py", "model_adapter/sktime_compat.py"]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_bytes(path, raw):
    """Publish a fully flushed file without replacement, even with concurrent writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic and fails if final name already exists
    finally:
        os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def load_february():
    """Fixed paths and pinned acquisition manifest; every OHLCVA value/grid validated."""
    if DATA.resolve() != DATA or (DATA / "manifest.json").is_symlink():
        raise ValueError("February inputs must not redirect")
    raw_manifest = (DATA / "manifest.json").read_bytes()
    if digest(raw_manifest) != MANIFEST_SHA256:
        raise ValueError("February acquisition manifest hash changed")
    manifest = json.loads(raw_manifest)
    if (manifest["month"] != "2026-02" or manifest["status"] != "verified"
            or manifest["assets"] != ASSETS or manifest["failed_assets"] != 0
            or manifest["expected_rows_per_asset"] != 40320):
        raise ValueError("fixed February ten-asset manifest required")
    records = {r["asset"]: r for r in manifest["results"]}
    if len(manifest["results"]) != 10 or set(records) != set(ASSETS):
        raise ValueError("manifest asset identities must be complete and unique")
    histories, files, shared_times = {}, [], None
    for asset in ASSETS:
        name = f"{asset}USDT-1m-2026-02.csv"
        entry = records[asset]
        if entry["csv_filename"] != name or entry["rows"] != 40320 or entry["status"] != "verified":
            raise ValueError(f"{asset}: invalid February CSV identity")
        path = DATA / name
        if path.is_symlink():
            raise ValueError("February CSV must not redirect")
        raw = path.read_bytes()
        sha = digest(raw)
        if sha != entry["csv_sha256"]:
            raise ValueError(f"{asset}: CSV hash differs from acquisition")
        rows = _parse_binance(csv.reader(io.StringIO(raw.decode("utf-8"))), asset)
        times = [r["time"] for r in rows]
        if (len(rows) != 40320 or times[0] != "2026-02-01T00:01:00Z"
                or times[-1] != "2026-03-01T00:00:00Z"):
            raise ValueError("complete February candle-end minute grid required")
        if shared_times is not None and times != shared_times:
            raise ValueError("cross-asset time grid mismatch")
        shared_times, histories[asset] = times, rows
        files.append({"asset": asset, "csv_filename": name, "csv_sha256": sha})
    return {"assets": list(ASSETS), "times": shared_times, "histories": histories,
            "closes": np.array([[r["close"] for r in histories[a]] for a in ASSETS], dtype=float).T,
            "provenance": {"month": "2026-02", "role": "B_observational_replication_no_training",
                           "manifest_path": str((DATA / "manifest.json").relative_to(ROOT)),
                           "manifest_sha256": digest(raw_manifest), "files": files, "rows_per_asset": 40320,
                           "source": "Binance Spot monthly klines 2026-02", "quote_currency": "USDT"}}


def market_window(data, index):
    if isinstance(index, bool) or not isinstance(index, int) or not 255 <= index < len(data["times"]):
        raise ValueError("window requires 256 observed closes")
    if data["provenance"]["month"] != "2026-02":
        raise ValueError("February market window cannot accept another month")
    value = {"schema_version": 1, "profile_id": "research_binance_202602_shock_v1",
             "source": data["provenance"]["source"], "mode": "replay", "quote_currency": "USDT",
             "interval_seconds": 60, "assets": list(data["assets"]), "as_of": data["times"][index],
             "data_revision": data["provenance"]["manifest_sha256"],
             "histories": {a: data["histories"][a][index-255:index+1] for a in data["assets"]}}
    value = json.loads(canonical(value))  # detach from full dataset; no future truth/classification
    value["window_id"] = "window_" + digest(canonical(value))
    return value


def future_closes(data, index):
    if not 255 <= index or index + 30 >= len(data["times"]):
        raise ValueError("complete observed window plus 30-minute truth required")
    return {a: [r["close"] for r in data["histories"][a][index+1:index+31]] for a in data["assets"]}


def _source_bytes():
    return {name: (ROOT / name).read_bytes() for name in SOURCES}


def prepare(out=OUTPUT):
    out = Path(out)
    if out.exists():
        raise FileExistsError("preparation exists; no overwrite or alternate sample selection")
    data = load_february()
    catalog = detect_events(data["closes"], data["times"], data["assets"], k=6)
    events = [e for e in catalog["events"] if e["eligible"]]
    sources = _source_bytes()
    january_reference = JANUARY_REFERENCE.read_bytes()
    reference_identity = json.loads(january_reference)["model_identity"]
    out.mkdir(parents=True)
    write_json(out / "catalog-k6.json", catalog)
    atomic_bytes(out / "january-reference-manifest.json", january_reference)
    source_hashes = {name: digest(raw) for name, raw in sources.items()}
    for name, raw in sources.items():
        atomic_bytes(out / "code_at_prepare" / name, raw)
    result = {"status": "prepared_no_model_run", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "month": "2026-02", "k": 6, "selection": "all eligible fixed-k6 events; no count recalibration/subsampling",
              "event_count": len(events), "event_ids": [e["event_id"] for e in events],
              "catalog_event_count": len(catalog["events"]), "stats": catalog["stats"],
              "catalog_sha256": digest((out / "catalog-k6.json").read_bytes()),
              "provenance": data["provenance"], "predict_config": dict(CONFIG), "sampling": dict(SAMPLING),
              "method_names": METHODS, "source_sha256": source_hashes,
              "window_profile_id": "research_binance_202602_shock_v1",
              "reference_model_identity": reference_identity,
              "january_reference_manifest_sha256": digest(january_reference),
              "expected_asset_path_calls": len(events) * len(ASSETS)}
    write_json(out / "preparation.json", result)
    return result


def _prepared(out, *, check_live_code=True):
    out = Path(out)
    prep = json.loads((out / "preparation.json").read_bytes())
    raw = (out / "catalog-k6.json").read_bytes()
    if (prep["month"] != "2026-02" or prep["k"] != 6 or prep["predict_config"] != CONFIG
            or prep["sampling"] != SAMPLING or digest(raw) != prep["catalog_sha256"]):
        raise ValueError("preparation/catalog/config identity changed")
    catalog = json.loads(raw)
    events = [e for e in catalog["events"] if e["eligible"]]
    if [e["event_id"] for e in events] != prep["event_ids"] or len(events) != prep["event_count"]:
        raise ValueError("all eligible February events must be preserved")
    reference = (out / "january-reference-manifest.json").read_bytes()
    if (digest(reference) != prep["january_reference_manifest_sha256"]
            or json.loads(reference)["model_identity"] != prep["reference_model_identity"]):
        raise ValueError("frozen January model identity changed")
    for name, expected in prep["source_sha256"].items():
        if digest((out / "code_at_prepare" / name).read_bytes()) != expected:
            raise ValueError("frozen source snapshot changed")
        if check_live_code and digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"source changed after preparation: {name}")
    return prep, catalog, events


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _risk_table(spots, paths):
    return {a: risk_metrics(spots[a], closes) for a, closes in paths.items()}


def _identity_matches(identity, reference):
    if identity.get("sampling") != SAMPLING:
        raise ValueError("sampling differs from frozen January protocol")
    if identity.get("output_policy") != "containment-expand-v1" or identity.get("volume_policy") != "unused-volume-audit-v1":
        raise ValueError("model output policies differ from approved protocol")
    if identity != reference:
        raise ValueError("full model/tokenizer/adapter/runtime identity differs from frozen January run")


def run(out=OUTPUT, device="mps"):
    """Root-only invocation: one attempt per event, immutable raw output before scoring."""
    prep, catalog, events = _prepared(out)
    out = Path(out)
    run_dir = out / "inference-original-v1"
    if run_dir.exists():
        raise FileExistsError("inference already attempted; no implicit retry/resampling")
    data = load_february()
    if data["provenance"] != prep["provenance"]:
        raise ValueError("February data changed after preparation")
    # Lazy imports ensure --prepare and --report never import or call a model.
    from forecast_metrics.engine import freeze_forecast
    from model_adapter import KronosAdapter, PredictionValidationError, validate_forecast_output
    model = ROOT / "scenario-lab/models/Kronos-base"
    tokenizer = ROOT / "scenario-lab/models/Kronos-Tokenizer-base"
    adapter = KronosAdapter(model_path=model, tokenizer_path=tokenizer, device=device)
    identity = adapter.identity()
    _identity_matches(identity, prep["reference_model_identity"])
    run_dir.mkdir()
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "event_count": len(events),
                "event_ids": prep["event_ids"], "preparation_sha256": digest((out / "preparation.json").read_bytes()),
                "catalog_sha256": prep["catalog_sha256"], "model_identity": identity,
                "model_path": str(model), "tokenizer_path": str(tokenizer), "device": device,
                "provenance": data["provenance"], "predict_config": CONFIG,
                "method_names": METHODS, "source_sha256": prep["source_sha256"]}
    write_json(run_dir / "run_manifest.json", manifest)
    started = time.perf_counter()
    for number, event in enumerate(events, 1):
        event_start = time.perf_counter()
        event_id = event["event_id"]
        window = market_window(data, event["index"])
        spots = {a: window["histories"][a][-1]["close"] for a in ASSETS}
        write_json(run_dir / "requests" / (event_id + ".json"),
                   {"window": window, "predict_config": CONFIG, "prediction_run_id": "shock-feb-k6-v1-" + event_id})
        predictions = {m: None for m in METHODS}
        artifact = {"event": event, "window_id": window["window_id"], "forecast_id": None,
                    "forecast": None, "raw_paths": None, "runtime": None, "errors": {}}
        paths = {}
        try:
            baseline = predict_baselines(window, event["origin_assets"])
            paths.update(baseline["paths"])
            artifact.update(baseline_metadata=baseline["metadata"])
            artifact["errors"].update(baseline["errors"])
            for method, values in paths.items():
                predictions[method] = _risk_table(spots, values)
        except Exception as exc:
            artifact["errors"]["baselines"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            output = adapter.predict(window, CONFIG, "shock-feb-k6-v1-" + event_id)
            write_json(run_dir / "raw" / (event_id + ".json"), _safe(output))
            artifact.update(output)
            validate_forecast_output(output["forecast"], output["raw_paths"])
            frozen = freeze_forecast(output["forecast"])
            if len(output["forecast"]["paths"]) != 1:
                raise ValueError("frozen replication requires one raw model path")
            paths["kronos_base"] = {a: series["close"] for a, series in output["forecast"]["paths"][0]["assets"].items()}
            predictions["kronos_base"] = _risk_table(spots, paths["kronos_base"])
            artifact["forecast_id"] = frozen.identity
        except PredictionValidationError as exc:
            raw_failure = {"raw_paths": _safe(exc.raw_paths), "runtime": _safe(exc.runtime),
                           "issues": _safe(exc.issues), "raw_paths_encoding": "explicit nonfinite tags"}
            write_json(run_dir / "raw" / (event_id + ".json"), raw_failure)
            artifact.update(raw_paths=raw_failure["raw_paths"], runtime=raw_failure["runtime"])
            artifact["errors"]["kronos_base"] = {"type": type(exc).__name__, "issues": _safe(exc.issues)}
        except Exception as exc:
            artifact["errors"]["kronos_base"] = {"type": type(exc).__name__, "message": str(exc)}
        # Future truth is a separate object, used only after model invocation.
        actual = _risk_table(spots, future_closes(data, event["index"]))
        score = score_event(event, predictions, actual, ASSETS)
        artifact.update(predicted_risk=predictions, actual_risk=actual,
                        method_close_paths=paths, score=score,
                        end_to_end_seconds=time.perf_counter() - event_start)
        write_json(run_dir / (event_id + ".json"), _safe(artifact))
        print(json.dumps({"completed": number, "total": len(events), "timestamp": event["timestamp"],
                          "kronos_status": score["methods"]["kronos_base"]["status"],
                          "seconds": round(artifact["end_to_end_seconds"], 3)}), flush=True)
    ending_identity = KronosAdapter(model_path=model, tokenizer_path=tokenizer, device=device).identity()
    hashes_after = {f["csv_filename"]: digest((DATA / f["csv_filename"]).read_bytes()) for f in prep["provenance"]["files"]}
    data_stable = all(hashes_after[f["csv_filename"]] == f["csv_sha256"] for f in prep["provenance"]["files"])
    stable = ending_identity == identity and data_stable
    write_json(run_dir / "completion.json", {"status": "complete" if stable else "invalid_identity_changed",
               "event_attempt_count": len(events), "model_identity_unchanged": ending_identity == identity,
               "data_hashes_unchanged": data_stable, "data_hashes_after": hashes_after,
               "end_to_end_seconds": time.perf_counter() - started,
               "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return 0 if stable else 2


def _direction(event):
    return "up" if event["direction"] == 1 else "down" if event["direction"] == -1 else "mixed_or_unknown"


def summarize(catalog, artifacts):
    """All denominators retained; direction comparisons share complete four-method pairs."""
    by_id = {a["event"]["event_id"]: a for a in artifacts}
    if len(by_id) != len(artifacts):
        raise ValueError("duplicate event outputs")
    eligible_ids = {e["event_id"] for e in catalog["events"] if e["eligible"]}
    if not set(by_id) <= eligible_ids:
        raise ValueError("unexpected event output outside frozen February selection")
    scores, events = [], {}
    for event in catalog["events"]:
        event_id = event["event_id"]
        events[event_id] = event
        artifact = by_id.get(event_id)
        if artifact is None:
            status = "missing" if event["eligible"] else "ineligible"
            score = {"event_id": event_id, "eligible": event["eligible"], "systemic_flag": event["systemic_flag"],
                     "actual_status": "not_scored", "methods": {m: {"status": status} for m in METHODS}}
        else:
            if artifact["event"] != event or artifact["score"]["event_id"] != event_id:
                raise ValueError("event output/catalog identity mismatch")
            score = artifact["score"]
        scores.append(score)
    full = aggregate_events(scores, METHODS)
    groups = {"all": full["groups"]["all"]}
    for group in ("up", "down", "mixed_or_unknown"):
        selected = [s for s in scores if _direction(events[s["event_id"]]) == group]
        groups[group] = aggregate_events(selected, METHODS)["groups"]["all"]
    total_n = groups["all"]["common_paired_event_count"]
    all_stats = groups["all"]["scores"]
    total_gap = (all_stats["historical_30"]["max_drawdown"]["mae_bps"] -
                 all_stats["kronos_base"]["max_drawdown"]["mae_bps"]) if total_n else None
    contributions = {}
    for group in ("up", "down", "mixed_or_unknown"):
        selected = groups[group]
        n = selected["common_paired_event_count"]
        delta = (selected["scores"]["historical_30"]["max_drawdown"]["mae_bps"] -
                 selected["scores"]["kronos_base"]["max_drawdown"]["mae_bps"]) if n else None
        contributions[group] = {"event_count": n, "paired_total": total_n,
                                "within_group_gap_bps": delta,
                                "weighted_contribution_bps": n / total_n * delta if n else 0. if total_n else None}
    down_contribution = contributions["down"]["weighted_contribution_bps"]
    share = down_contribution / total_gap if total_gap is not None and total_gap > 0 else None
    return {"month": "2026-02", "groups": groups, "full_metrics": full,
            "eligible_event_count": len(eligible_ids), "output_event_count": len(artifacts),
            "fixed_historical_30_decomposition": {
                "comparator": "historical_30", "denominator": "identical complete paired events; failures retained outside arithmetic",
                "paired_event_count": total_n, "total_advantage_bps": total_gap, "groups": contributions,
                "down_share": share, "down_share_reason": None if share is not None else "total_advantage_not_positive_or_no_pairs",
                "january_reference_down_share_approx": .95, "january_reference_total_gap_bps_approx": 8.477,
                "replication_claim": "observational comparison only; no preselected significance or closeness threshold"}}


def render_report(summary):
    names = {"all": "全部", "up": "上涨触发", "down": "下跌触发", "mixed_or_unknown": "混合/未知"}
    def number(value, percent=False):
        return "—" if value is None else f"{100*value:.2f}%" if percent else f"{value:.3f}"
    lines = ["# 二月 B 线观察性复验", "", f"固定 k=6，所有合格事件 {summary['eligible_event_count']} 个；已有输出 {summary['output_event_count']} 个。原版 Kronos，256→30、每资产一条路径，同一组四种方法；未重新选择阈值。", "",
             "| 触发方向 | 完整配对/合格 | 方法 | MDD MAE (bps) | 波动率 Top3 期望召回 |", "|---|---:|---|---:|---:|"]
    for group, values in summary["groups"].items():
        for method in METHODS:
            stats = values["scores"][method]
            lines.append(f"| {names[group]} | {values['common_paired_event_count']}/{values['eligible_event_count']} | {method} | {number(stats['max_drawdown']['mae_bps'])} | {number(stats['volatility']['top3_recall_expected'], True)} |")
    decomp = summary["fixed_historical_30_decomposition"]
    lines += ["", f"固定比较对象 historical_30：整体 MDD 优势 {number(decomp['total_advantage_bps'])} bps；下跌事件加权贡献占比 {number(decomp['down_share'], True)}。一月参考约 8.477 bps、95%。",
              ""]
    if decomp["down_share"] is None:
        lines.append("本月总优势非正或没有完整配对，因此下跌贡献占比不定义；不能宣称复现了 95%。")
    else:
        lines.append("该占比是分组误差差值的加权分解，不是方向准确率；仅并列观察，不设事后‘复现成功’门槛。")
    lines += ["", "失败与缺失保留在分母，四种方法仅在相同完整配对事件上比较。混合组无事件时保留零分母与空指标；每个事件先平均资产误差，再等权平均事件。一次采样、单月、重叠窗口均限制外推；systemic 为事后标签，未改实时报警或 A 线。", ""]
    return "\n".join(lines)


def _completion_gate(completion, expected_events):
    if (completion.get("status") != "complete"
            or completion.get("model_identity_unchanged") is not True
            or completion.get("data_hashes_unchanged") is not True
            or completion.get("event_attempt_count") != expected_events):
        raise ValueError("complete attempts and unchanged model/data identities required before reporting")


def report(out=OUTPUT):
    out = Path(out)
    prep, catalog, events = _prepared(out)
    run_dir = out / "inference-original-v1"
    manifest = json.loads((run_dir / "run_manifest.json").read_bytes())
    if manifest["preparation_sha256"] != digest((out / "preparation.json").read_bytes()):
        raise ValueError("run preparation identity mismatch")
    _identity_matches(manifest["model_identity"], prep["reference_model_identity"])
    completion = json.loads((run_dir / "completion.json").read_bytes())
    _completion_gate(completion, len(events))
    artifacts = [json.loads(p.read_bytes()) for p in sorted(run_dir.glob("event_*.json"))]
    if len(artifacts) != len(events):
        raise ValueError("every attempted event must retain an output before reporting")
    result = summarize(catalog, artifacts)
    result["provenance"] = prep["provenance"]
    result["run_manifest_sha256"] = digest((run_dir / "run_manifest.json").read_bytes())
    result["event_output_hashes"] = {p.name: digest(p.read_bytes()) for p in sorted(run_dir.glob("event_*.json"))}
    result["completion"] = completion
    write_json(out / "summary.json", result)
    atomic_bytes(out / "FEBRUARY.md", render_report(result).encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.output)
        print(json.dumps({"status": result["status"], "events": result["event_count"], "asset_path_calls": result["expected_asset_path_calls"], "output": str(args.output)}))
    elif args.run:
        return run(args.output, args.device)
    else:
        result = report(args.output)
        print(json.dumps({"eligible": result["eligible_event_count"], "paired": result["groups"]["all"]["common_paired_event_count"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
