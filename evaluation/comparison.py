"""Offline, manifest-denominated comparisons of retained distributed artifacts.

No model, network, remote truth, resampling of predictions or threshold search.
Saved predictions are re-scored against the already-downloaded local truth.
All errors are decimal returns/volatility, not cross-asset dollar errors.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

from forecast_metrics.engine import ASSET_METRICS
from .universe import DEFAULT_ASSETS, LOCAL_PROFILES, TEN_PROFILE, asset_list, portfolio_policy

ASSETS = DEFAULT_ASSETS  # Backward-compatible import for legacy three-asset fixtures.
BASELINES = {"terminal_return": "naive-last", "volatility": "historical-volatility"}
SEGMENTS = ("Jan01-07", "Jan08-14", "Jan15-21", "Jan22-31")
DRAW_DOWN = .03


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("timestamps require a timezone")
    return result.astimezone(timezone.utc)


def _segment(value):
    dt = _time(value)
    if dt.year != 2025 or dt.month != 1:
        raise ValueError("this frozen comparison requires January 2025 origins")
    return SEGMENTS[min((dt.day - 1) // 7, 3)]


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numerical metric required")
    return float(value)


def _equal(a, b):
    return math.isclose(_number(a), _number(b), rel_tol=1e-10, abs_tol=1e-12)


def _manifest_tasks(manifest):
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("a nonempty frozen manifest is required")
    ids, origins, configs, models, profiles, universes = set(), set(), set(), set(), set(), set()
    for task in tasks:
        tid, window = task["task_id"], task["window"]
        origin = _time(window["as_of"])
        _segment(window["as_of"])
        if not isinstance(tid, str) or tid in ids or origin in origins:
            raise ValueError("duplicate manifest task or origin")
        body = {key: value for key, value in task.items() if key != "task_id"}
        if tid != "task_" + hashlib.sha256(_canonical(body).encode()).hexdigest():
            raise ValueError("manifest task content hash mismatch")
        assets = asset_list(window["assets"])
        universes.add(assets)
        if window["interval_seconds"] != 60:
            raise ValueError("the comparison requires minute windows")
        config = task["predict_config"]
        for key in ("lookback", "horizon", "path_count", "seed"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
                raise ValueError("invalid frozen prediction configuration")
        ids.add(tid)
        origins.add(origin)
        configs.add(_canonical(config))
        models.add(_canonical(task["model_spec"]))
        profiles.add(_canonical([window["profile_id"], window["source"], window["quote_currency"]]))
    if len(configs) != 1 or len(models) != 1 or len(profiles) != 1 or len(universes) != 1:
        raise ValueError("one comparison requires the same model, configuration and data profile")
    assets = next(iter(universes))
    if "assets" in manifest and asset_list(manifest["assets"]) != assets:
        raise ValueError("manifest and window asset order differ")
    if tasks[0]["window"]["profile_id"] == TEN_PROFILE:
        if asset_list(manifest.get("assets")) != assets or manifest.get("portfolio_policy") != portfolio_policy(tasks[0]["window"]):
            raise ValueError("ten-asset manifest requires the frozen explicit portfolio policy")
    if manifest.get("selected_count", len(tasks)) != len(tasks):
        raise ValueError("manifest selected_count mismatch")
    if "task_ids" in manifest and manifest["task_ids"] != [t["task_id"] for t in tasks]:
        raise ValueError("manifest task_ids mismatch")
    horizon = tasks[0]["predict_config"]["horizon"]
    ordered = sorted(origins)
    if any((b - a).total_seconds() < horizon * 60 for a, b in zip(ordered, ordered[1:])):
        raise ValueError("this frozen comparison requires non-overlapping future windows")
    return tasks


def competition_top(values, top_k=2):
    """Same strict-greater competition rank as evaluate_alerts; include ties."""
    checked = {s: _number(v) for s, v in values.items()}
    return sorted(s for s, v in checked.items() if 1 + sum(x > v for x in checked.values()) <= top_k)


def initial_ranking_baseline(window, horizon):
    """Freeze a static comparator from the first input's past only, never truth."""
    from .scoring import _historical_volatility
    assets = asset_list(window["assets"])
    vol = {s: _historical_volatility(window["histories"][s], horizon) for s in assets}
    return {"label": "constant-initial-volatility", "assets": competition_top(vol), "as_of": window["as_of"]}


def _ranking_baseline(manifest, tasks):
    first = min(tasks, key=lambda task: _time(task["window"]["as_of"]))
    if first["window"]["profile_id"] == TEN_PROFILE:
        expected = initial_ranking_baseline(first["window"], first["predict_config"]["horizon"])
        if manifest.get("ranking_baseline") != expected:
            raise ValueError("static ranking baseline must equal top2 from the first input's past volatility")
        return expected
    if set(first["window"]["assets"]) != set(ASSETS):
        raise ValueError("non-default universe requires a frozen initial-history ranking baseline")
    return {"label": "constant-ETH-SOL", "assets": ["ETH", "SOL"]}


def _path_mean(metrics, symbol, metric, count):
    paths = metrics["paths"]
    if len(paths) != count or metrics["sample_count"] != count:
        raise ValueError("metric path count mismatch")
    if len({p["path_id"] for p in paths}) != count:
        raise ValueError("duplicate metric path")
    value = mean(_number(p["assets"][symbol][metric]) for p in paths)
    if not _equal(value, metrics["summary"]["assets"][symbol][metric]["mean"]):
        raise ValueError("metric summary is not the mean of pathwise metrics")
    return value


def _verify_numerical_evidence(task, artifact):
    """Recheck evidence from saved raw forecasts and local truth, never infer.

    This comparison is deliberately restricted to the frozen local dataset so
    scorer reuse cannot accidentally call a live market API.
    """
    from data_pipeline import load_window
    from forecast_metrics.engine import freeze_forecast
    from model_adapter import validate_forecast_output
    from evaluation.scoring import score_forecast

    window = task["window"]
    if window.get("profile_id") not in LOCAL_PROFILES:
        raise ValueError("evidence verification supports only the local Binance January profile")
    actual_window = load_window(window["profile_id"], lookback=task["predict_config"]["lookback"], as_of=window["as_of"])
    if actual_window != window:
        raise ValueError("manifest window differs from frozen local historical input")
    result = artifact["result"]
    validate_forecast_output(result["forecast"], result["raw_paths"])
    frozen = freeze_forecast(result["forecast"])
    if frozen.identity != artifact["forecast_id"]:
        raise ValueError("saved forecast content does not match forecast_id")
    verified = score_forecast(window, frozen.data, frozen.identity, model_label=task["model_spec"]["model_label"])
    if verified["status"] != "scored":
        raise ValueError("local evidence re-scoring did not complete: " + str(verified.get("error", verified["status"])))
    original = artifact["evaluation"]
    for key in ("forecast_id", "window_id", "as_of", "model", "path_count", "prediction_metrics", "truth_metrics", "rows", "baseline_artifacts"):
        if verified[key] != original.get(key):
            raise ValueError(f"stored {key} differs from raw forecast/local truth re-scoring")
    if window["profile_id"] == TEN_PROFILE:
        for key in ("assets", "portfolio_policy"):
            if verified[key] != original.get(key):
                raise ValueError(f"stored experimental {key} differs from declared policy")


def _observation(task, artifact, expected_identity, ranking_baseline=None):
    """Check stored paired rows against their pathwise metrics and task identity."""
    window, config = task["window"], task["predict_config"]
    assets = asset_list(window["assets"])
    ranking_baseline = ranking_baseline or {"label": "constant-ETH-SOL", "assets": ["ETH", "SOL"]}
    required_identity = {"model_revision", "tokenizer_revision", "adapter_revision", "sampling",
                         "output_policy", "volume_policy", "backend", "torch_version", "sktime_version",
                         "model_timestamp", "output_timestamp"}
    actual_identity = artifact.get("result", {}).get("runtime", {}).get("identity")
    if not isinstance(expected_identity, dict) or not required_identity <= expected_identity.keys():
        raise ValueError("batch expected_identity is missing or incomplete")
    if actual_identity != expected_identity:
        raise ValueError("runtime identity differs from batch expected_identity (including weights, sampling, adapter, tokenizer and backend)")
    _verify_numerical_evidence(task, artifact)
    for key in ("as_of", "window_id", "profile_id", "source", "quote_currency"):
        if artifact.get(key) != window[key]:
            raise ValueError(f"artifact/manifest {key} mismatch")
    for key in ("predict_config", "model_spec"):
        if artifact.get(key) != task[key]:
            raise ValueError(f"artifact/manifest {key} mismatch")
    for key in ("correction_quality_status", "volume_quality_status"):
        if artifact.get(key) != "validated":
            raise ValueError("scored artifact lacks validated dual-audit publication")
    evaluation = artifact["evaluation"]
    if evaluation["status"] != "scored" or evaluation["sample_count"] != 1:
        raise ValueError("scored artifact lacks a complete one-window evaluation")
    forecast_id, model = artifact["forecast_id"], task["model_spec"]["model_label"]
    for key, expected in (("forecast_id", forecast_id), ("as_of", window["as_of"]),
                          ("window_id", window["window_id"]), ("model", model),
                          ("path_count", config["path_count"])):
        if evaluation.get(key) != expected:
            raise ValueError(f"evaluation identity {key} mismatch")
    prediction, truth = evaluation["prediction_metrics"], evaluation["truth_metrics"]
    if prediction["forecast_id"] != forecast_id or prediction["as_of"] != window["as_of"] or truth["as_of"] != window["as_of"]:
        raise ValueError("metric forecast or origin mismatch")
    if len(prediction["times"]) != config["horizon"] + 1 or prediction["times"] != truth["times"]:
        raise ValueError("prediction and truth horizon mismatch")
    if not artifact.get("result", {}).get("forecast", {}).get("model_revision"):
        raise ValueError("actual model revision unavailable")
    if artifact["result"]["forecast"]["model_revision"] != expected_identity["model_revision"]:
        raise ValueError("forecast weight revision differs from runtime identity")
    expected = {(model, asset, metric) for asset in assets for metric in ASSET_METRICS}
    expected |= {(baseline, asset, metric) for metric, baseline in BASELINES.items() for asset in assets}
    rows = {}
    for row in evaluation["rows"]:
        key = (row["model"], row["symbol"], row["metric"])
        if key not in expected or key in rows:
            raise ValueError("unexpected or duplicate evaluation row")
        if row["as_of"] != window["as_of"] or row["forecast_id"] != forecast_id or row["status"] != "scored":
            raise ValueError("row identity or status mismatch")
        if not _equal(row["absolute_error"], abs(_number(row["predicted"]) - _number(row["actual"]))):
            raise ValueError("stored absolute error mismatch")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("incomplete paired evaluation rows")
    pairs, predicted_vol, actual_vol, past_vol, drawdown = {}, {}, {}, {}, {}
    for asset in assets:
        pairs[asset] = {}
        for metric in ASSET_METRICS:
            row = rows[(model, asset, metric)]
            pred = _path_mean(prediction, asset, metric, config["path_count"])
            actual = _path_mean(truth, asset, metric, 1)
            if not _equal(row["predicted"], pred) or not _equal(row["actual"], actual):
                raise ValueError("row disagrees with pathwise metrics")
            if metric in BASELINES:
                baseline = rows[(BASELINES[metric], asset, metric)]
                if not _equal(baseline["actual"], actual):
                    raise ValueError("unpaired baseline actual")
                if metric == "terminal_return" and not _equal(baseline["predicted"], 0):
                    raise ValueError("naive-last terminal return must be zero")
                pairs[asset][metric] = {"model": pred, "baseline": baseline["predicted"], "actual": actual,
                                         "model_error": row["absolute_error"], "baseline_error": baseline["absolute_error"]}
                if metric == "volatility":
                    predicted_vol[asset], actual_vol[asset], past_vol[asset] = pred, actual, baseline["predicted"]
            elif metric == "max_drawdown":
                drawdown[asset] = {"predicted": pred, "actual": actual,
                                   "predicted_positive": pred > DRAW_DOWN, "actual_positive": actual > DRAW_DOWN}
    return {"pairs": pairs,
            "ranking": {"actual": competition_top(actual_vol), "Kronos": competition_top(predicted_vol),
                        "historical-volatility": competition_top(past_vol), ranking_baseline["label"]: ranking_baseline["assets"]},
            "drawdown": drawdown}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _error_stats(pairs):
    model = mean(p[0] for p in pairs) if pairs else None
    baseline = mean(p[1] for p in pairs) if pairs else None
    wins = sum(a < b and not _equal(a, b) for a, b in pairs)
    ties = sum(_equal(a, b) for a, b in pairs)
    return {"window_count": len(pairs), "model_mae": model, "baseline_mae": baseline,
            "skill": 1 - model / baseline if baseline else None,
            "skill_status": "no_pairs" if not pairs else ("undefined_zero_baseline_mae" if not baseline else "defined"),
            "win": wins, "tie": ties, "loss": len(pairs) - wins - ties,
            "tie_tolerance": {"relative": 1e-10, "absolute": 1e-12}}


def _quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _bootstrap(items, *, replicates, seed):
    days = defaultdict(list)
    for origin, a, b in items:
        days[_time(origin).date().isoformat()].append((a, b))
    result = {"method": "paired resampling of UTC-day blocks; all same-day windows stay together",
              "status": "insufficient_days", "day_count": len(days), "requested_replicates": replicates,
              "valid_replicates": 0, "zero_baseline_replicates": 0, "seed": seed,
              "skill_percentile_95": None, "exploratory": True,
              "disclosure": "Day blocks retain within-day dependence; days need not be independent. This is an exploratory interval, not proof of significance or future performance."}
    if not replicates:
        result["status"] = "disabled"
        return result
    if len(days) < 2:
        return result
    keys = sorted(days)
    rng, skills = random.Random(seed), []
    for _ in range(replicates):
        sample = [pair for day in rng.choices(keys, k=len(keys)) for pair in days[day]]
        denominator = math.fsum(b for _, b in sample)
        if denominator:
            skills.append(1 - math.fsum(a for a, _ in sample) / denominator)
    result.update(status="exploratory" if skills else "undefined_zero_baseline_mae", valid_replicates=len(skills),
                  zero_baseline_replicates=replicates - len(skills),
                  skill_percentile_95=[_quantile(skills, .025), _quantile(skills, .975)] if skills else None)
    return result


def _ranking_stats(observations, strategy, assets=ASSETS):
    exact, precisions, recalls, jaccards, sizes = 0, [], [], [], []
    selected = Counter()
    for item in observations:
        actual, predicted = set(item["ranking"]["actual"]), set(item["ranking"][strategy])
        common = len(actual & predicted)
        exact += actual == predicted
        precisions.append(common / len(predicted))
        recalls.append(common / len(actual))
        jaccards.append(common / len(actual | predicted))
        sizes.append(len(predicted))
        selected.update(predicted)
    return {"window_count": len(observations), "exact_set_hits": exact,
            "exact_set_hit_rate": _ratio(exact, len(observations)),
            "mean_set_precision": mean(precisions) if precisions else None,
            "mean_set_recall": mean(recalls) if recalls else None,
            "mean_jaccard": mean(jaccards) if jaccards else None,
            "mean_selected_assets": mean(sizes) if sizes else None,
            "selection_count_by_asset": {s: selected[s] for s in assets}}


def _confusion(values):
    tp = sum(p and a for p, a in values)
    fp = sum(p and not a for p, a in values)
    fn = sum(not p and a for p, a in values)
    tn = len(values) - tp - fp - fn
    return {"asset_window_count": len(values), "actual_positives": tp + fn, "predicted_positives": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}


def _scope(observations, *, requested, replicates, seed, assets=ASSETS, static_label="constant-ETH-SOL"):
    comparisons = {}
    for metric, baseline in BASELINES.items():
        by_asset = {}
        for asset in (*assets, "equal_weight_assets"):
            items = []
            for observation in observations:
                selected = assets if asset == "equal_weight_assets" else (asset,)
                errors = [observation["pairs"][s][metric] for s in selected]
                items.append((observation["as_of"], mean(e["model_error"] for e in errors), mean(e["baseline_error"] for e in errors)))
            by_asset[asset] = {**_error_stats([(a, b) for _, a, b in items]),
                             "skill_interval": _bootstrap(items, replicates=replicates, seed=seed)}
        comparisons[metric] = {"baseline": baseline, "units": "decimal", "by_asset": by_asset}
    ranking = {strategy: _ranking_stats(observations, strategy, assets)
               for strategy in ("Kronos", "historical-volatility", static_label)}
    ranking_comparisons = {}
    for baseline in ("historical-volatility", static_label):
        deltas = [int(o["ranking"]["Kronos"] == o["ranking"]["actual"]) -
                  int(o["ranking"][baseline] == o["ranking"]["actual"]) for o in observations]
        ranking_comparisons[baseline] = {"paired_windows": len(deltas),
            "exact_set_hit_rate_gain": mean(deltas) if deltas else None,
            "win": sum(d > 0 for d in deltas), "tie": sum(d == 0 for d in deltas),
            "loss": sum(d < 0 for d in deltas)}
    drawdown = {s: _confusion([(o["drawdown"][s]["predicted_positive"], o["drawdown"][s]["actual_positive"])
                               for o in observations]) for s in assets}
    drawdown["all_assets"] = _confusion([(o["drawdown"][s]["predicted_positive"], o["drawdown"][s]["actual_positive"])
                                           for o in observations for s in assets])
    return {"requested_windows": requested, "paired_windows": len(observations),
            "paired_coverage": _ratio(len(observations), requested), "errors": comparisons,
            "top2_ranking": {"rule": "competition rank <= 2 using mean of per-path volatility; ties may select more than two",
                             "principal_baseline": "historical-volatility",
                             "strategies": ranking, "paired_exact_set_comparisons": ranking_comparisons},
            "drawdown_3pct": {"rule": "mean per-path maximum drawdown > 0.03; independently reported from top2",
                               "by_asset": drawdown}}


def compare_records(manifest, records, *, expected_identity=None, bootstrap_replicates=2000, bootstrap_seed=20260926):
    """Compare parsed records ({path, artifact} or {path, parse_error, task_id})."""
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or not 0 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates must be between 0 and 10000")
    tasks = _manifest_tasks(manifest)
    assets = asset_list(tasks[0]["window"]["assets"])
    ranking_baseline = _ranking_baseline(manifest, tasks)
    by_id = {task["task_id"]: [] for task in tasks}
    unassigned = []
    for record in records:
        artifact = record.get("artifact", {})
        tid = artifact.get("task_id", record.get("task_id")) if isinstance(artifact, dict) else record.get("task_id")
        if tid not in by_id:
            unassigned.append({"path": record["path"], "task_id": tid, "reason": record.get("parse_error", "unknown task or non-object artifact")})
        else:
            by_id[tid].append(record)
    observations, accounting = [], []
    for task in tasks:
        window, found = task["window"], by_id[task["task_id"]]
        item = {"task_id": task["task_id"], "window_id": window["window_id"], "as_of": window["as_of"],
                "segment": _segment(window["as_of"]), "artifact_paths": [r["path"] for r in found],
                "status": "missing", "errors": []}
        if len(found) > 1:
            item.update(status="duplicate", errors=["Multiple artifacts for a task; all excluded, no preferred attempt selected."])
            item["attempts"] = [{"path": r["path"], "status": r.get("artifact", {}).get("status") if isinstance(r.get("artifact"), dict) else None,
                                 "error": r.get("parse_error", r.get("artifact", {}).get("error") if isinstance(r.get("artifact"), dict) else None)} for r in found]
        elif found:
            record = found[0]
            if "parse_error" in record:
                item.update(status="invalid_artifact", errors=[record["parse_error"]])
            else:
                artifact = record["artifact"]
                status = artifact.get("status")
                if status in {"failed", "pending_truth", "incomplete_truth", "not_run", "predicted"}:
                    item.update(status=status, errors=[artifact["error"]] if "error" in artifact else [])
                elif status != "scored":
                    item.update(status="invalid_artifact", errors=["unknown artifact status"])
                else:
                    try:
                        item.update(_observation(task, artifact, expected_identity, ranking_baseline), status="scored", forecast_id=artifact["forecast_id"])
                        item["model_revision"] = artifact["result"]["forecast"]["model_revision"]
                        observations.append(item)
                    except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError, IndexError) as exc:
                        item.update(status="invalid_scored", errors=[str(exc)])
        accounting.append(item)
    revisions = {o["model_revision"] for o in observations}
    if len(revisions) > 1:
        for item in observations:
            item.update(status="mixed_model_revision", errors=["Multiple actual weight revisions; cannot pool model comparisons."])
        observations = []
    counts = Counter(item["status"] for item in accounting)
    segments = {}
    for segment in SEGMENTS:
        subset = [o for o in observations if o["segment"] == segment]
        segments[segment] = _scope(subset, requested=sum(a["segment"] == segment for a in accounting),
                                   replicates=bootstrap_replicates, seed=bootstrap_seed,
                                   assets=assets, static_label=ranking_baseline["label"])
    paired = len(observations)
    return {
        "schema_version": 1, "experiment_id": manifest.get("experiment_id"),
        "assets": list(assets), "ranking_baseline": ranking_baseline,
        "expected_identity": expected_identity,
        "status": "no_comparable_results" if not paired else ("complete" if paired == len(tasks) and not unassigned else "partial"),
        "model": tasks[0]["model_spec"]["model_label"], "model_revisions": sorted(revisions),
        "predict_config": tasks[0]["predict_config"], "profile_id": tasks[0]["window"]["profile_id"],
        "source": tasks[0]["window"]["source"], "quote_currency": tasks[0]["window"]["quote_currency"],
        "coverage": {"requested_windows": len(tasks), "paired_windows": paired, "excluded_windows": len(tasks) - paired,
                     "paired_coverage": paired / len(tasks), "status_counts": dict(sorted(counts.items())),
                     "input_artifact_records": len(records), "unassigned_record_count": len(unassigned)},
        "overall": _scope(observations, requested=len(tasks), replicates=bootstrap_replicates, seed=bootstrap_seed,
                          assets=assets, static_label=ranking_baseline["label"]),
        "calendar_segments": segments, "tasks": accounting, "unassigned_records": unassigned,
        "method": {"pairing": f"same manifest task, asset and realized target; all {len(assets) * (len(ASSET_METRICS) + len(BASELINES))} scoring rows required",
                   "evidence_verification": "Full raw forecast dual-audit validation, content hash, exact local input, and re-scoring against independent local Binance truth using the existing engine and baselines; no new inference or network.",
                   "aggregation": "Each path's metrics are calculated before averaging. Equal-weight cross-asset MAE averages all declared asset errors per window, then windows; it is not portfolio error.",
                   "skill": "1 - paired model MAE / paired baseline MAE; null when baseline MAE is zero",
                   "win_tie_loss": "Comparisons of per-window absolute errors; cross-asset counts compare that window's equal-weight average errors.",
                   "failures": "Every manifest task retained. Missing, failed, duplicate or incomplete results are excluded from paired errors and remain explicit in coverage.",
                   "selection": manifest.get("selection", "Use the externally frozen manifest; this tool does not select tasks.")},
        "disclosure": "Exploratory diagnostic January 2025 Binance evidence, not a pristine holdout, trading recommendation, causal claim, or guarantee across future regimes. Conditional errors on successfully paired windows can be selection-biased when failures occur. No forecast-accuracy claim follows from engineering completion."}


def compare_batch(manifest, batch_dir, *, bootstrap_replicates=2000, bootstrap_seed=20260926):
    """Read all retained task files; never rely on summary success counts."""
    batch_dir = Path(batch_dir)
    task_dir = batch_dir / "tasks"
    records = []
    if task_dir.is_dir():
        for path in sorted(task_dir.rglob("*.json")):
            try:
                artifact = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if not isinstance(artifact, dict):
                    raise ValueError("task artifact must be an object")
                records.append({"path": str(path.resolve()), "artifact": artifact})
            except (OSError, UnicodeError, ValueError) as exc:
                records.append({"path": str(path.resolve()), "task_id": path.stem, "parse_error": type(exc).__name__ + ": " + str(exc)})
    expected_identity, summary_error = None, None
    try:
        summary = json.loads((batch_dir / "summary.json").read_text())
        if not isinstance(summary, dict):
            raise ValueError("batch summary must be an object")
        if summary.get("experiment_id") != manifest.get("experiment_id"):
            raise ValueError("batch/manifest experiment_id mismatch")
        expected_identity = summary["expected_identity"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        summary_error = type(exc).__name__ + ": " + str(exc)
    result = compare_records(manifest, records, expected_identity=expected_identity,
                             bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed)
    result["batch_dir"] = str(batch_dir.resolve())
    result["task_directory_present"] = task_dir.is_dir()
    result["batch_summary_error"] = summary_error
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new comparison JSON file; existing files are never overwritten")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260926)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    result = compare_batch(manifest, args.batch_dir, bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
    result["manifest_path"] = str(args.manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "coverage": result["coverage"],
                      "terminal_return": result["overall"]["errors"]["terminal_return"]["by_asset"]["equal_weight_assets"],
                      "volatility": result["overall"]["errors"]["volatility"]["by_asset"]["equal_weight_assets"],
                      "out": str(args.out.resolve())}, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
