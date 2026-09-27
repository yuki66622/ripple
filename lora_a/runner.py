"""Sequential A-line execution. March is reachable only after a written model lock.

Run with the project venv: python -m lora_a.runner --run-dir lora_a/runs/<new-id>
Use --prepare-only or --smoke-only for bounded gates. No remote services.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lora_a/config.json"
CODE_FILES = [ROOT / "lora_a" / name for name in
              ("runner.py", "data.py", "metrics.py", "training.py", "reporting.py", "config.json", "PROTOCOL.md")]
SHARED_CODE_FILES = [ROOT / name for name in ("model_adapter/adapter.py", "model_adapter/sktime_compat.py",
                     "model_adapter/corrections.py", "model_adapter/volume_quality.py", "forecast_metrics/engine.py")]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def verify_initial_identities(directory):
    environment = json.loads((directory / "environment.json").read_text())
    records = environment["code"] + environment["shared_code"] + [environment["config"], environment["base_model"], environment["tokenizer"]]
    for record in records:
        if reference(record["path"]) != record:
            raise RuntimeError("experiment input/code identity changed during execution: " + record["path"])
    for name, version in environment["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("installed package changed during execution: " + name)


def now():
    return datetime.now(timezone.utc).isoformat()


class DeadlineReached(RuntimeError):
    pass


def remaining_seconds(config):
    return (datetime.fromisoformat(config["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()


def ensure_time(config, reserve=0):
    if remaining_seconds(config) <= reserve:
        raise DeadlineReached("insufficient time before frozen morning deadline")


def safe_json(value):
    """Keep rejected nonfinite raw values as explicit tagged evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_raw_value": repr(value)}
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class Journal:
    def __init__(self, directory):
        self.path = directory / "training-log.jsonl"

    def __call__(self, event=None, **fields):
        row = {"utc": now(), **(event if isinstance(event, dict) else {"event": event}), **fields}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with self.path.open("a") as stream:
            stream.write(encoded + "\n")
            stream.flush()
        print(encoded, flush=True)


def clear_inference():
    import torch
    from model_adapter.sktime_compat import _CACHE
    _CACHE.clear()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def score_all(data, config, directory, methods, journal, *, label=None, allow_incomplete=False):
    """Read saved predictions only, without reruns or output-dependent selection."""
    from .data import window, truth
    from .metrics import score_window, summarize
    streams = {}
    for name in methods:
        path = directory / f"{name}-predictions.jsonl"
        if path.exists():
            streams[name] = path.open()
        elif allow_incomplete:
            streams[name] = iter(())
        else:
            raise FileNotFoundError(path)
    rows = []
    try:
        with (directory / ((label or "paired-" + "-".join(methods)) + ".jsonl")).open("x") as out:
            for origin in data.origins:
                w = window(data, origin)
                predictions = {}
                for name, stream in streams.items():
                    raw = next(stream, None)
                    if raw is None and allow_incomplete:
                        predictions[name] = None
                        continue
                    if raw is None:
                        raise ValueError("prediction stream ended before complete grid")
                    item = json.loads(raw)
                    if item["window_id"] != w["window_id"] or item["as_of"] != w["as_of"]:
                        raise ValueError("prediction stream differs from frozen complete grid")
                    predictions[name] = item.get("result") if item["status"] == "scored" else None
                row = score_window(w, truth(data, origin), predictions, fixed_pair=config["fixed_small_pair"])
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if any(next(stream, None) for stream in streams.values()):
                raise ValueError("prediction stream has surplus windows")
    finally:
        for stream in streams.values():
            if hasattr(stream, "close"):
                stream.close()
    report = summarize(rows, model_names=methods)
    report.update({"month": data.month, "config": config, "data_provenance": data.provenance,
                   "prediction_files": {name: reference(directory / f"{name}-predictions.jsonl") for name in methods
                                        if (directory / f"{name}-predictions.jsonl").exists()},
                   "completed_at": now()})
    journal("scoring_complete", month=data.month, methods=methods, n=len(rows))
    return report


def predict_month(data, config, model_path, name, directory, journal):
    from .data import window
    from model_adapter.adapter import KronosAdapter, PredictionValidationError
    directory.mkdir(parents=True, exist_ok=True)
    adapter = KronosAdapter(model_path=model_path, device="mps")
    identity = adapter.identity()
    cfg = {k: config[k] for k in ("lookback", "horizon", "path_count", "seed")}
    started, failed, last_log = time.perf_counter(), 0, time.perf_counter()
    path = directory / f"{name}-predictions.jsonl"
    with path.open("x") as stream:
        for index, origin in enumerate(data.origins, 1):
            ensure_time(config)
            w = window(data, origin)
            item = {"method": name, "window_id": w["window_id"], "as_of": w["as_of"],
                    "origin_index": origin}
            try:
                result = adapter.predict(w, cfg, f"a-{data.month}-{name}-{origin}")
                item.update(status="scored", result=result)
            except PredictionValidationError as exc:
                failed += 1
                item.update(status="failed", error=str(exc), raw_paths=safe_json(exc.raw_paths), runtime=exc.runtime)
            except Exception as exc:
                # Infrastructure/model errors are not hidden as a partial successful month.
                item.update(status="fatal", error=repr(exc))
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                raise
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            elapsed = time.perf_counter() - started
            if index == 1 or time.perf_counter() - last_log >= 30 or index == len(data.origins):
                journal("prediction_progress", month=data.month, model=name, completed=index,
                        total=len(data.origins), failures=failed, elapsed_seconds=elapsed,
                        projected_total_seconds=elapsed / index * len(data.origins))
                last_log = time.perf_counter()
    elapsed = time.perf_counter() - started
    info = {"method": name, "month": data.month, "identity": identity, "n": len(data.origins),
            "failed": failed, "elapsed_seconds": elapsed, "predictions": reference(path)}
    write_new(directory / f"{name}-timing.json", info)
    del adapter
    clear_inference()
    return info


def freeze_environment(directory):
    import torch
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in
                                ("peft", "torch", "transformers", "sktime", "numpy", "pandas", "safetensors", "huggingface-hub")},
                   "peft_import_passed": True, "backend": "mps", "config": reference(CONFIG),
                   "code": [reference(p) for p in CODE_FILES], "shared_code": [reference(p) for p in SHARED_CODE_FILES], "created_at": now(),
                   "base_model": reference(ROOT / "scenario-lab/models/Kronos-base/model.safetensors"),
                   "tokenizer": reference(ROOT / "scenario-lab/models/Kronos-Tokenizer-base/model.safetensors")}
    write_new(directory / "environment.json", environment)


def prepare(directory, config, journal):
    import peft
    import torch
    from .data import load_month, window, truth, calendar_origins
    from .metrics import score_window, summarize
    from .training import load_train_components, smoke_check
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no silent CPU fallback")
    ensure_time(config)
    train = load_month("2026-01", role="train")
    # Gate 1 before baseline selection: smoke updates are always discarded.
    model, tokenizer, optimizer = load_train_components(config, device="mps")
    smoke = smoke_check(model, tokenizer, optimizer, train, config, directory / "smoke")
    write_new(directory / "smoke-gate.json", smoke)
    journal("smoke_gate_passed", peft=peft.__version__, examples=smoke["examples"],
            frozen_parameters_unchanged=smoke["frozen_parameters_unchanged"],
            reload_logits_max_abs_diff=smoke["reload_logits_max_abs_diff"])
    del model, tokenizer, optimizer
    clear_inference()
    # Gates 2/3 use only Jan1-25 training origins; no observation/02/03 selection.
    provisional_pair = sorted(config["tiers"]["small"])[:2]
    initial = [score_window(window(train, t), truth(train, t), {}, fixed_pair=provisional_pair) for t in train.origins]
    actual_sets = [set(row["tiers"]["small"]["actual_set"]) for row in initial]
    pairs = list(combinations(sorted(config["tiers"]["small"]), 2))
    counts = {pair: sum(set(pair) == actual for actual in actual_sets) for pair in pairs}
    best_count = max(counts.values())
    best_pair = min(pair for pair, count in counts.items() if count == best_count)
    if config["fixed_small_pair"] is not None and list(best_pair) != config["fixed_small_pair"]:
        raise RuntimeError("January optimum differs from an already-frozen fixed pair")
    config["fixed_small_pair"] = list(best_pair)
    CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    write_new(directory / "fixed-pair-selection.json", {
        "training_period": [config["train_start"], config["train_end_exclusive"]],
        "origins": train.origins, "n": len(train.origins), "selected_pair": list(best_pair),
        "hits": best_count, "candidates": [{"pair": list(pair), "hits": count} for pair, count in counts.items()],
        "tie_rule": "lexicographically smallest sorted pair", "selection_data": train.provenance,
        "observation_used": False, "february_used": False, "march_used": False})
    train = load_month("2026-01", role="train")
    rows = [score_window(window(train, t), truth(train, t), {}, fixed_pair=best_pair) for t in train.origins]
    baseline = summarize(rows, model_names=[])
    baseline.update(month="2026-01", data_provenance=train.provenance, period="Jan1-25 stride120")
    write_new(directory / "january-baselines.json", baseline)
    from .reporting import markdown_report
    markdown_report(directory / "january-baselines.md", baseline, "preflight",
                    note="训练期探索基线；不是独立测试成绩。固定组合只在该训练期选择。")
    grids = {role: {"month": month, "origins": calendar_origins(month, role)}
             for role, month in (("train", "2026-01"), ("observation", "2026-01"),
                                 ("validation", "2026-02"), ("test", "2026-03"))}
    write_new(directory / "calendar-grid-plan.json", {"policy": "calendar only, floor-equidistant candidate indices", "grids": grids,
              "march_content_accessed": False, "created_at": now()})
    write_new(directory / "config.json", config)
    freeze_environment(directory)
    small = baseline["tiers"]["small"]["methods"]
    journal("preflight_complete", training_origins=len(train.origins), asset_windows=len(train.origins)*10,
            fixed_pair=list(best_pair), dynamic_hit_rate=small["dynamic_naive"]["exact_set_hit_rate"],
            fixed_hit_rate=small["fixed_pair"]["exact_set_hit_rate"], validation_windows=300, test_windows=300)
    return train, config


def complete_report(report, methods):
    return all(report["tiers"][tier]["methods"][method]["n_failed"] == 0
               and report["tiers"][tier]["methods"][method]["n_scored"] == 300
               for tier in ("major", "small") for method in methods)


def write_interim(directory, reason, *, reports=None, selected=None):
    payload = {"status": "interim", "reason": reason, "selected": selected,
               "february_reports": reports or [], "march_accessed": False, "created_at": now(),
               "hyperparameters_changed": False, "automatic_upgrade": False}
    write_new(directory / "interim.json", payload)
    text = "# A 线 v2.1 阶段报告\n\n未进行03密封验收。\n\n原因：" + reason + "\n\n"
    for report in reports or []:
        text += "- 02报告：" + report["path"] + "\n"
    text += "\n不追加训练、不升级rank、不修改验收门。03继续密封。\n"
    with (directory / "interim.md").open("x") as stream:
        stream.write(text)


def observe(data, config, checkpoint, name, directory, journal):
    from .reporting import markdown_report, observation_metrics
    out = directory / "observation"
    predict_month(data, config, checkpoint, name, out, journal)
    methods = ["original"] if name == "original" else ["original", name]
    report = score_all(data, config, out, methods, journal)
    write_new(out / f"{name}-report.json", report)
    markdown_report(out / f"{name}-report.md", report, name,
                    note="01-26至31观察曲线；本报告不参与checkpoint、固定组合或任何参数选择。")
    diagnostics = observation_metrics(report, name)
    journal("observation_only_metrics", dataset="January26-31 fixed48", **diagnostics)
    return {"method": name, "small_mae": report["tiers"]["small"]["methods"][name]["volatility_mae"],
            "major_mae": report["tiers"]["major"]["methods"][name]["volatility_mae"],
            "used_for_selection": False, "diagnostics": diagnostics, "report": reference(out / f"{name}-report.json")}


def execute(directory, *, prepare_only=False):
    from .data import load_month
    from .metrics import gates
    from .training import load_train_components, train_epoch, save_adapter, export_merged
    from .reporting import markdown_report, observation_metrics
    config = json.loads(CONFIG.read_text())
    if config["protocol_version"] != "lora-a-v2.1":
        raise ValueError("v2.1 is the only active protocol")
    directory.mkdir(parents=True, exist_ok=False)
    journal = Journal(directory)
    begin = time.perf_counter()
    candidates, curve, reports = [], [], []
    validation = None
    model = tokenizer = optimizer = None
    try:
        train, config = prepare(directory, config, journal)
        if prepare_only:
            return
        # Smoke is discarded; the measured full epoch IS formal epoch 1.
        model, tokenizer, optimizer = load_train_components(config, device="mps")
        best, best_mae, stale = None, float("inf"), 0
        baseline_timing = None
        for epoch in range(1, config["max_epochs"] + 1):
            verify_initial_identities(directory)
            ensure_time(config)
            name = f"epoch_{epoch:02d}"
            epoch_directory = directory / name
            epoch_directory.mkdir()
            journal("training_epoch_start", epoch=epoch, planned_epochs=config["max_epochs"],
                    timing_is_formal_epoch=True, batch_size=config["batch_size"])
            stats = train_epoch(model, tokenizer, optimizer, train, config, epoch, log_fn=journal)
            write_new(epoch_directory / "training-stats.json", stats)
            adapter_meta = save_adapter(model, epoch_directory / "adapter")
            merged_meta = export_merged(model, epoch_directory / "merged")
            write_new(epoch_directory / "checkpoint.json", {"adapter": adapter_meta, "merged": merged_meta})
            journal("training_epoch_complete", epoch=epoch, stats=stats)
            model.to("cpu")
            tokenizer.to("cpu")
            clear_inference()
            if validation is None:
                validation = load_month("2026-02", role="validation")
                write_new(directory / "validation-grid.json", {"provenance": validation.provenance,
                          "origins": [validation.times[t] for t in validation.origins]})
                observation = load_month("2026-01", role="observation")
            val_dir = directory / "february"
            if baseline_timing is None:
                baseline_timing = predict_month(validation, config, ROOT / "scenario-lab/models/Kronos-base",
                                                "original", val_dir, journal)
                original_report = score_all(validation, config, val_dir, ["original"], journal)
                write_new(val_dir / "original-report.json", original_report)
                journal("observation_only_metrics", dataset="February fixed300", **observation_metrics(original_report, "original"))
                if not complete_report(original_report, ["original"]):
                    raise RuntimeError("original February predictions incomplete; no checkpoint can be fairly selected")
                original_mae = original_report["tiers"]["small"]["methods"]["original"]["volatility_mae"]
            timing = predict_month(validation, config, epoch_directory / "merged", name, val_dir, journal)
            report = score_all(validation, config, val_dir, ["original", name], journal)
            write_new(val_dir / f"{name}-report.json", report)
            markdown_report(val_dir / f"{name}-report.md", report, name, gates(report, name))
            journal("observation_only_metrics", dataset="February fixed300", **observation_metrics(report, name))
            reports.append(reference(val_dir / f"{name}-report.json"))
            if not complete_report(report, ["original", name]):
                write_interim(directory, "02候选或原版存在失败窗口，不能用成功子集选模。", reports=reports)
                journal("interim_complete", reason="incomplete February paired grid", march_accessed=False)
                return
            mae = report["tiers"]["small"]["methods"][name]["volatility_mae"]
            candidate = {"epoch": epoch, "name": name, "small_mae": mae, "validation_report": reports[-1]}
            candidates.append(candidate)
            if mae < best_mae - config["min_delta"]:
                best, best_mae, stale = candidate, mae, 0
            else:
                stale += 1
            if epoch == 1:
                write_new(directory / "microbench.json", {
                    "training_epoch_seconds": stats["elapsed_seconds"], "training_asset_windows": stats["examples"],
                    "steps": stats["batches"], "batch_size": config["batch_size"], "timing_is_epoch1": True,
                    "original_february_300_seconds": baseline_timing["elapsed_seconds"],
                    "candidate_february_300_seconds": timing["elapsed_seconds"], "epoch_upper_bound": 2,
                    "measured_not_full_parameter_estimate": True})
            journal("validation_epoch_complete", epoch=epoch, small_mae=mae, original_small_mae=original_mae,
                    selected_epoch=best["epoch"], stale=stale, n=300)
            # Observation values are isolated from every selection/early-stop branch.
            if not curve:
                curve.append(observe(observation, config, ROOT / "scenario-lab/models/Kronos-base", "original", directory, journal))
            curve.append(observe(observation, config, epoch_directory / "merged", name, directory, journal))
            if stale >= config["patience"]:
                journal("early_stop", reason="February small-tier MAE did not improve", epoch=epoch)
                break
            if epoch < config["max_epochs"]:
                # Reserve a further complete validation/observation; otherwise interim.
                reserve = timing["elapsed_seconds"] + stats["elapsed_seconds"] + timing["elapsed_seconds"] * 48 / 300 + 600
                ensure_time(config, reserve)
                model.to("mps")
                tokenizer.to("mps")
        model = tokenizer = optimizer = None
        clear_inference()
        write_new(directory / "observation-curve.json", {"points": curve, "used_for_selection": False})
        write_new(directory / "selection-history.json", {"candidates": candidates, "selected": best,
                  "rule": config["selection_metric"], "planned_epochs": config["max_epochs"],
                  "original_small_mae": original_mae, "no_improvement_reference": "original on same300 February origins"})
        if best is None or best_mae >= original_mae:
            write_interim(directory, "02两轮最佳小币MAE未优于同300窗口的原版；未观察到改善，不自动解释为欠拟合。", reports=reports, selected=best)
            journal("interim_complete", reason="no February MAE improvement over original", march_accessed=False)
            return
        verify_initial_identities(directory)
        selected = directory / best["name"]
        lock = {"status": "locked", "epoch": best["epoch"], "locked_at_utc": now(), "acceptance_authorized": True,
                "selected_checkpoint": reference(selected / "adapter/adapter_model.safetensors"),
                "selected_merged_checkpoint": reference(selected / "merged/model.safetensors"),
                "config": reference(CONFIG), "code": [reference(p) for p in CODE_FILES],
                "validation_report": best["validation_report"], "selection_history": reference(directory / "selection-history.json"),
                "calendar_grid_plan": reference(directory / "calendar-grid-plan.json")}
        write_new(directory / "selection-lock.json", lock)
        journal("selection_ready_for_root_review", epoch=best["epoch"], elapsed_seconds=time.perf_counter()-begin,
                march_accessed=False)
    except DeadlineReached as exc:
        write_interim(directory, str(exc), reports=reports, selected=candidates[-1] if candidates else None)
        journal("interim_complete", reason="morning deadline", march_accessed=False)
    except BaseException as exc:
        journal("stopped_on_error", error=repr(exc), traceback=traceback.format_exc(), march_accessed=False)
        raise
    finally:
        model = tokenizer = optimizer = None
        clear_inference()


def accept_selected(directory):
    """Separate root-reviewed acceptance, never auto-called from training."""
    from .data import load_month, validate_selection_lock, calendar_origins
    from .metrics import gates
    from .reporting import markdown_report
    journal = Journal(directory)
    started = time.perf_counter()
    try:
        verify_initial_identities(directory)
        lock = json.loads((directory / "selection-lock.json").read_text())
        config = validate_selection_lock(lock)
        best = json.loads((directory / "selection-history.json").read_text())["selected"]
        selected = directory / best["name"]
        timing = json.loads((directory / "microbench.json").read_text())
        # Full same-size two-model March run plus a conservative margin, before IO.
        reserve = (timing["original_february_300_seconds"] + timing["candidate_february_300_seconds"]) * 1.2 + 1200
        if remaining_seconds(config) <= reserve:
            write_interim(directory, "剩余时间不足以完成03双模型300窗口验收，保持密封。", reports=[best["validation_report"]], selected=best)
            journal("interim_complete", reason="insufficient full acceptance budget", march_accessed=False)
            return
        plan = json.loads((directory / "calendar-grid-plan.json").read_text())
        if reference(directory / "calendar-grid-plan.json") != lock["calendar_grid_plan"]:
            raise RuntimeError("locked calendar plan identity changed")
        for role, month in (("train", "2026-01"), ("observation", "2026-01"), ("validation", "2026-02"), ("test", "2026-03")):
            if plan["grids"][role] != {"month": month, "origins": calendar_origins(month, role)}:
                raise RuntimeError("calendar grid differs from predeclared plan: " + role)
        journal("root_acceptance_stage_started", selected_epoch=best["epoch"], remaining_seconds=remaining_seconds(config),
                reserved_seconds=reserve, further_training=False)
        test = load_month("2026-03", role="test", selection_lock=lock)
        if test.origins != plan["grids"]["test"]["origins"] or len(test.origins) != 300:
            raise RuntimeError("March loader did not produce the predeclared300 grid")
        journal("march_unsealed_after_lock", n=300, lock_sha256=digest(directory / "selection-lock.json"))
        test_dir = directory / "march"
        predict_month(test, config, ROOT / "scenario-lab/models/Kronos-base", "original", test_dir, journal)
        predict_month(test, config, selected / "merged", best["name"], test_dir, journal)
        report = score_all(test, config, test_dir, ["original", best["name"]], journal)
        verdict = gates(report, best["name"])
        rates = [report["tiers"][tier]["methods"][best["name"]]["exact_set_hit_rate"] for tier in ("major", "small")]
        verdict["anomalous_accuracy_requires_leakage_review"] = any(rate is not None and rate >= .95 for rate in rates)
        verdict["numerical_gates_passed"] = verdict["passed"]
        verdict["acceptance_status"] = "complete"
        if verdict["anomalous_accuracy_requires_leakage_review"]:
            verdict["passed"] = False
            verdict["acceptance_status"] = "pending_leakage_review"
            verdict["reasons"].append("accuracy >=95%; numerical gates do not constitute acceptance before leakage review")
        write_new(test_dir / "sealed-report.json", report)
        write_new(test_dir / "verdict.json", verdict)
        markdown_report(test_dir / "sealed-report.md", report, best["name"], verdict)
        from .reporting import acceptance_summary
        acceptance_summary(test_dir / "conclusion.md", report, best["name"], verdict)
        journal("acceptance_computed", passed=verdict["passed"], elapsed_seconds=time.perf_counter()-started,
                selected_epoch=best["epoch"], requires_leakage_review=verdict["anomalous_accuracy_requires_leakage_review"],
                further_training=False)
    except BaseException as exc:
        journal("acceptance_stopped_on_error", error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--accept-selected", action="store_true")
    args = parser.parse_args()
    if args.accept_selected:
        if args.prepare_only:
            parser.error("acceptance cannot be combined with preparation")
        accept_selected(args.run_dir.resolve())
    else:
        execute(args.run_dir.resolve(), prepare_only=args.prepare_only)
