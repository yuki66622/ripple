"""Bounded rolling evaluation and portable whole-window task execution.

Use --plan-only for JSON task creation without loading/running a model. Each
task contains past input only and is a suitable unit of two-device dispatch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from data_pipeline import load_history, load_window
from forecast_metrics.engine import freeze_forecast
from .scoring import _validate_pair, score_forecast
from .corrections import extract_quality, summarize_quality, validate_for_publication
from .volume_quality import extract_volume_quality, summarize_volume_quality
from .universe import LOCAL_PROFILES, TEN_PROFILE, portfolio_policy


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def build_manifest(*, profile_id="binance_jan2025", lookback=256, horizon=30,
                   stride=30, limit=4, path_count=1, seed=20260926,
                   model_path=None, tokenizer_path=None, model_label="Kronos-base"):
    """Freeze fixed sktime folds; never infer a model or consume future labels."""
    import pandas as pd
    from sktime.split import SlidingWindowSplitter

    if profile_id not in LOCAL_PROFILES:
        raise ValueError("frozen rolling evaluation requires a declared local Binance profile")
    for value, name, low, high in ((lookback, "lookback", 2, 512), (horizon, "horizon", 2, 128),
                                  (stride, "stride", 1, None), (limit, "limit", 1, None),
                                  (path_count, "path_count", 1, 16), (seed, "seed", 0, 2**32 - 1)):
        _integer(value, name, low, high)
    if lookback < horizon + 1:
        raise ValueError("lookback must include horizon+1 candles for the frozen historical-volatility baseline")
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model_label must be nonempty")
    history = load_history(profile_id)
    rows = history["histories"][history["assets"][0]]
    splitter = SlidingWindowSplitter(fh=list(range(1, horizon + 1)), window_length=lookback,
                                    step_length=stride, start_with_window=True)
    # The splitter receives an index, not future prices. Input windows remain
    # physically separate from truth objects during inference.
    series = pd.Series(range(len(rows)), index=pd.RangeIndex(len(rows)))
    available = max(0, (len(rows) - lookback - horizon) // stride + 1)
    if not available:
        raise ValueError("history has no complete input+truth window")
    spec = {"model_path": str(model_path) if model_path else None,
            "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
            "model_label": model_label}
    config = {"lookback": lookback, "horizon": horizon, "path_count": path_count, "seed": seed}
    tasks = []
    for fold, (train, test) in enumerate(splitter.split(series)):
        if fold >= limit:
            break
        if len(train) != lookback or len(test) != horizon or train[-1] + 1 != test[0]:
            raise RuntimeError("sktime fold violates the fixed-window contract")
        as_of = rows[int(train[-1])]["time"]
        window = load_window(profile_id, lookback=lookback, as_of=as_of)
        task = {"schema_version": 1, "kind": "whole_window_prediction", "fold": fold,
                "origin_index": int(train[-1]), "window": window, "predict_config": dict(config),
                "model_spec": dict(spec)}
        task["task_id"] = "task_" + _hash(task)
        tasks.append(task)
    metadata = {
        "schema_version": 1, "profile_id": profile_id, "config": config, "model_spec": spec,
        "stride": stride, "requested_limit": limit, "available_windows": available,
        "selected_count": len(tasks), "splitter": "sktime.SlidingWindowSplitter",
        "overlapping_truth": stride < horizon,
        "regime": "unclassified; past-only calibration thresholds not frozen",
        "task_ids": [task["task_id"] for task in tasks],
    }
    if profile_id == TEN_PROFILE:
        from .comparison import initial_ranking_baseline
        metadata.update(assets=list(tasks[0]["window"]["assets"]),
                        portfolio_policy=portfolio_policy(tasks[0]["window"]),
                        ranking_baseline=initial_ranking_baseline(tasks[0]["window"], horizon))
    metadata["experiment_id"] = "experiment_" + _hash(metadata)
    return {**metadata, "tasks": tasks}


def _safe_json(value):
    """Preserve rejected nonfinite raw model values with explicit string tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def _write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_safe_json(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    # New names and exclusive creation preserve all attempts, including failures.
    with path.open("x", encoding="utf-8") as file:
        file.write(encoded + "\n")


def save_manifest(manifest, output):
    destination = Path(output) / "manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing != manifest:
            raise FileExistsError("output already contains a different manifest; choose a new output directory")
    else:
        _write_new(destination, manifest)
    return destination


def _validate_task(task):
    expected = "task_" + _hash({key: value for key, value in task.items() if key != "task_id"})
    if task.get("schema_version") != 1 or task.get("kind") != "whole_window_prediction" or task.get("task_id") != expected:
        raise ValueError("task identity/schema mismatch")
    if "truth" in task or "truth" in task.get("window", {}):
        raise ValueError("inference tasks cannot contain future truth")


def _adapter(spec, *, device=None):
    from model_adapter import KronosAdapter
    return KronosAdapter(model_path=spec.get("model_path"), tokenizer_path=spec.get("tokenizer_path"), device=device)


def run_task(task, out_dir, *, adapter=None, score=True, device=None):
    """Run one manifest task, always retaining a unique result/failure artifact.

    score=False is useful for a remote worker without the full history: the
    coordinator can score its returned frozen forecast against local truth.
    Adapter injection is for a resident model or explicit test fixture.
    """
    _validate_task(task)
    attempt_id = "attempt_" + uuid.uuid4().hex
    started = time.perf_counter()
    artifact = {
        "schema_version": 1, "task_id": task["task_id"], "attempt_id": attempt_id,
        "as_of": task["window"]["as_of"], "window_id": task["window"]["window_id"],
        "profile_id": task["window"]["profile_id"], "source": task["window"]["source"],
        "quote_currency": task["window"]["quote_currency"],
        "model_spec": task["model_spec"], "predict_config": task["predict_config"],
        "status": "failed", "started_at": datetime.now(timezone.utc).isoformat(),
        "correction_quality": None, "correction_quality_status": "missing",
        "volume_quality": None, "volume_quality_status": "missing",
    }
    stage = "load_model"
    try:
        adapter = adapter or _adapter(task["model_spec"], device=device)
        stage = "predict"
        result = adapter.predict(task["window"], task["predict_config"], "evaluation:" + task["task_id"])
        artifact["result"] = result
        stage = "correction_metadata"
        artifact["correction_quality"] = extract_quality(result["forecast"])
        if artifact["correction_quality"] is not None:
            artifact["correction_quality_status"] = "unvalidated"
        stage = "volume_metadata"
        artifact["volume_quality"] = extract_volume_quality(result["forecast"])
        if artifact["volume_quality"] is not None:
            artifact["volume_quality_status"] = "unvalidated"
        stage = "validate_forecast_output"
        # Real and fixture runner outputs pass the same validation gate. There is
        # no legacy/no-audit bypass for newly published inference results.
        if not isinstance(result.get("raw_paths"), list) or not result["raw_paths"]:
            raise ValueError("raw_paths are required for correction publication validation")
        validate_for_publication(result["forecast"], result["raw_paths"])
        artifact["correction_quality_status"] = "validated"
        artifact["volume_quality_status"] = "validated"
        stage = "freeze_forecast"
        frozen = freeze_forecast(result["forecast"])
        _validate_pair(task["window"], frozen)
        if len(frozen.data["paths"]) != task["predict_config"]["path_count"]:
            raise ValueError("result path count does not match requested task")
        if len(frozen.data["times"]) != task["predict_config"]["horizon"]:
            raise ValueError("result horizon does not match requested task")
        artifact["forecast_id"] = frozen.identity
        artifact["status"] = "predicted"
        if score:
            stage = "score"
            evaluation = score_forecast(task["window"], frozen.data, frozen.identity,
                                        model_label=task["model_spec"]["model_label"])
            artifact["evaluation"] = evaluation
            artifact["status"] = evaluation["status"]
    except Exception as exc:
        artifact["status"] = "failed"
        if stage == "correction_metadata":
            artifact["correction_quality_status"] = "invalid"
        if stage == "volume_metadata":
            artifact["volume_quality_status"] = "invalid"
        artifact["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        for key in ("raw_paths", "runtime", "issues"):
            if hasattr(exc, key):
                artifact[key] = getattr(exc, key)
        if "raw_paths" in artifact:
            artifact["raw_nonfinite_encoding"] = "Invalid nonfinite floats, if any, are preserved as NaN/Infinity/-Infinity strings."
    artifact["wall_ms"] = (time.perf_counter() - started) * 1000
    destination = Path(out_dir) / "tasks" / task["task_id"] / (attempt_id + ".json")
    artifact["artifact_path"] = str(destination.resolve())
    _write_new(destination, artifact)
    return artifact


def summarize(artifacts, *, requested_count=None, high_rate_threshold=.25):
    """Aggregate once per unique task; callers must select one retry per task."""
    ids = [artifact["task_id"] for artifact in artifacts]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task results require explicit retry selection before aggregation")
    requested = len(artifacts) if requested_count is None else requested_count
    if requested < len(artifacts):
        raise ValueError("requested_count cannot be less than returned task count")
    rows = [row for artifact in artifacts for row in artifact.get("evaluation", {}).get("rows", [])]
    return {
        "status": "complete" if len(artifacts) == requested else "partial",
        "requested_count": requested, "attempted_count": len(artifacts),
        "sample_count": sum(a["status"] == "scored" for a in artifacts),
        "failed_count": sum(a["status"] == "failed" for a in artifacts),
        "pending_count": sum(a["status"] == "pending_truth" for a in artifacts),
        "incomplete_count": sum(a["status"] == "incomplete_truth" for a in artifacts),
        "predicted_only_count": sum(a["status"] == "predicted" for a in artifacts),
        "not_run_count": requested - len(artifacts), "quality_status": "insufficient_evidence",
        "rows": rows, "artifacts": [{"task_id": a["task_id"], "status": a["status"],
                                     "artifact_path": a["artifact_path"]} for a in artifacts],
        "sample_unit": "distinct input windows; not paths/assets/metric rows",
        "correction_quality": summarize_quality(artifacts),
        "volume_quality": summarize_volume_quality(artifacts, high_rate_threshold=high_rate_threshold),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=LOCAL_PROFILES, default="binance_jan2025")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--paths", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--model-label", default="Kronos-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--manifest", help="read an existing frozen task manifest")
    parser.add_argument("--task-id", help="run exactly one task from --manifest")
    parser.add_argument("--no-score", action="store_true", help="remote inference only; coordinator scores later")
    parser.add_argument("--volume-high-rate", type=float, default=.25, help="exploratory evaluator grouping only; not an alarm")
    args = parser.parse_args(argv)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        for task in manifest["tasks"]:
            _validate_task(task)
    else:
        manifest = build_manifest(profile_id=args.profile, lookback=args.lookback, horizon=args.horizon, stride=args.stride,
                                  limit=args.limit, path_count=args.paths, seed=args.seed,
                                  model_path=args.model_path, tokenizer_path=args.tokenizer_path,
                                  model_label=args.model_label)
    manifest_path = save_manifest(manifest, args.out)
    if args.plan_only:
        print(json.dumps({"status": "planned", "manifest": str(manifest_path),
                          "selected_count": manifest["selected_count"], "available_windows": manifest["available_windows"]}))
        return 0
    tasks = manifest["tasks"]
    if args.task_id:
        tasks = [task for task in tasks if task["task_id"] == args.task_id]
        if len(tasks) != 1:
            raise ValueError("requested task_id not found exactly once")
    adapter = None
    artifacts = []
    try:
        for task in tasks:
            # Constructor is light; actual loading remains inside run_task's failure boundary.
            if adapter is None:
                adapter = _adapter(task["model_spec"], device=args.device)
            artifact = run_task(task, args.out, adapter=adapter, score=not args.no_score)
            artifacts.append(artifact)
            print(json.dumps({"task_id": artifact["task_id"], "status": artifact["status"],
                              "wall_ms": artifact["wall_ms"]}), flush=True)
    finally:
        summary = summarize(artifacts, requested_count=len(tasks), high_rate_threshold=args.volume_high_rate)
        summary["experiment_id"] = manifest["experiment_id"]
        summary_path = Path(args.out) / ("summary_" + uuid.uuid4().hex + ".json")
        _write_new(summary_path, summary)
        print(json.dumps({key: value for key, value in summary.items() if key not in ("rows", "artifacts")}), flush=True)
    return 1 if summary["failed_count"] or summary["incomplete_count"] or summary["not_run_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
