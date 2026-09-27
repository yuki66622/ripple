"""Single owner of user state. Models never run for holdings/threshold edits."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

from forecast_metrics.engine import FrozenInput, freeze_forecast, make_holdings, recalculate, evaluate_alerts, report_payload
from model_adapter import validate_corrections, validate_forecast_output
from .storage import Store, canonical, digest, diagnostic_json
from .worker import LocalWorker

PROFILES = [
    {"profile_id": "binance_jan2025", "label": "Binance · 2025年1月回放", "mode": "replay", "quote_currency": "USDT"},
    {"profile_id": "kraken_live", "label": "Kraken · 实时行情", "mode": "live", "quote_currency": "USD"},
]
DEFAULT_CONFIG = {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926}


class APIError(Exception):
    def __init__(self, status, code, message, retryable=False):
        super().__init__(message)
        self.status, self.code, self.message, self.retryable = status, code, message, retryable


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def empty_evaluation():
    return {"status": "pending_truth", "rows": [], "sample_count": 0, "failed_count": 0,
            "pending_count": 0, "quality_status": "insufficient_evidence"}


class Application:
    def __init__(self, root=None, worker=None, loader=None, scorer=None):
        self.lock = threading.RLock()
        self.store = Store(root or Path(__file__).parent / "runtime")
        self.worker = worker or LocalWorker()
        if loader is None:
            from data_pipeline import load_window
            loader = load_window
        self.loader, self.scorer = loader, scorer
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        self.future = None
        self.state = self.store.get("current") or {
            "view_revision": 0, "status": "idle", "job": None, "error": None, "stale": False,
            "profiles": PROFILES, "profile_id": None, "window": None, "forecast": None,
            "metrics": None, "alerts": None, "counters": {"model_runs": 0, "cache_hits": 0},
            "runtime": None, "evaluation": empty_evaluation(),
            "policy": {"drawdown_threshold": .03, "top_k": 2},
        }
        # No silent resume or new inference during restart. Interrupted jobs are retryable.
        if self.state["status"] in ("loading", "predicting"):
            self.state["status"] = "error"
            self.state["stale"] = self.state["forecast"] is not None
            self.state["error"] = {"code": "interrupted", "message": "上次分析被中断，可重新运行。", "retryable": True}
            self.state["job"]["status"] = "interrupted"
            self._save()
        if self.state["forecast"]:
            try:
                f = dict(self.state["forecast"])
                fid = f.pop("forecast_id")
                if freeze_forecast(f).identity != fid:
                    raise ValueError("Snapshot mismatch")
                saved = self.store.get("forecast:" + fid)
                artifact = self.store.read_artifact(saved["artifact"])
                if artifact["forecast"] != f or "raw_paths" not in artifact:
                    raise ValueError("Saved raw prediction is missing or differs")
                self._validate_saved(f, artifact["raw_paths"])
                if self.state["metrics"]["forecast_id"] != fid:
                    raise ValueError("Metrics mismatch")
                metrics = self.state["metrics"]
                holdings = FrozenInput(canonical(metrics["holdings"]), metrics["holdings_id"])
                if recalculate(freeze_forecast(f), holdings) != metrics:
                    raise ValueError("Saved metrics do not match prediction")
                if evaluate_alerts(metrics, **self.state["policy"]) != self.state["alerts"]:
                    raise ValueError("Saved alerts do not match metrics")
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                self.state.update(status="error", forecast=None, metrics=None, alerts=None, window=None, stale=False,
                                  error={"code": "corrupt_state", "message": "保存的结果未通过校验，请重新分析。", "retryable": True})
                self._save()

    def _save(self, extra=None):
        self.state["view_revision"] += 1
        self.store.put_many({"current": self.state, **(extra or {})})

    @staticmethod
    def _validate_saved(forecast, raw_paths):
        if "volume_quality" in forecast:
            validate_forecast_output(forecast, raw_paths)
        else:
            # Retain pre-amendment snapshots under their original stricter rule;
            # do not synthesize evidence that was absent when they were created.
            validate_corrections(forecast, raw_paths)
            if any(value < 0 for path in raw_paths for values in path["assets"].values()
                   for field in ("volume", "amount") for value in values[field]):
                raise ValueError("Legacy output must satisfy its original volume rule")

    def view(self):
        with self.lock:
            result = deepcopy(self.state)
            w = result.get("window")
            if w and w["mode"] == "live":
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(w["as_of"].replace("Z", "+00:00"))).total_seconds()
                result["stale"] = result["stale"] or age > 180
                result["data_age_seconds"] = max(0, age)
            return result

    def _revision(self, body):
        if body.get("expected_view_revision") != self.state["view_revision"]:
            raise APIError(409, "view_changed", "结果已更新，请检查当前结果后再次应用。", True)

    def _busy(self):
        if self.state["status"] in ("loading", "predicting"):
            raise APIError(409, "analysis_busy", "分析正在运行，完成后再应用配置。", True)

    def analyze(self, body):
        if not isinstance(body, dict):
            raise APIError(400, "invalid_request", "请求必须是一个对象。")
        rid = body.get("request_id")
        if not isinstance(rid, str) or not 1 <= len(rid) <= 128:
            raise APIError(400, "invalid_request_id", "缺少有效的请求编号。")
        with self.lock:
            known = self.store.get("request:" + rid)
            request_hash = digest(body)
            if known:
                if known["hash"] != request_hash:
                    raise APIError(409, "request_id_conflict", "同一请求编号不能用于不同参数。")
                return {**self.view(), "job_id": known["job_id"], "idempotent": True}
            self._revision(body)
            self._busy()
            profile = body.get("profile_id", "binance_jan2025")
            if profile not in {p["profile_id"] for p in PROFILES}:
                raise APIError(400, "invalid_profile", "不支持的数据来源。")
            supplied = body.get("predict_config", {})
            if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_CONFIG):
                raise APIError(400, "invalid_config", "预测配置含不支持的字段。")
            config = {**DEFAULT_CONFIG, **supplied}
            if any(isinstance(v, bool) or not isinstance(v, int) for v in config.values()):
                raise APIError(400, "invalid_config", "预测参数必须是整数。")
            if config["lookback"] != 256 or config["horizon"] != 30 or not 1 <= config["path_count"] <= 16 or not 0 <= config["seed"] < 2**31:
                raise APIError(400, "invalid_config", "当前使用256根历史、30根预测；路径数为1至16。")
            force = body.get("force_run", False)
            if not isinstance(force, bool):
                raise APIError(400, "invalid_config", "force_run必须是布尔值。")
            job_id = "job_" + uuid4().hex
            self.state.update(status="loading", error=None,
                              job={"job_id": job_id, "request_id": rid, "status": "loading", "started_at": utcnow()})
            self._save({"request:" + rid: {"hash": request_hash, "job_id": job_id}})
            self.future = self.executor.submit(self._run, profile, config, force, body.get("as_of"), job_id)
            return {**self.view(), "job_id": job_id}

    def _run(self, profile, config, force, as_of, job_id):
        start = time.perf_counter()
        try:
            before_data = time.perf_counter()
            window = self.loader(profile, lookback=config["lookback"], as_of=as_of)
            data_ms = (time.perf_counter() - before_data) * 1000
            identity = self.worker.identity()
            key = "cache:" + digest({"window_id": window["window_id"], "identity": identity, "config": config})
            with self.lock:
                cached = None if force else self.store.get(key)
                self.state["status"] = "predicting"
                self.state["job"]["status"] = "predicting"
                self._save()
                if cached:
                    result = self.store.read_artifact(cached)
                else:
                    self.state["counters"]["model_runs"] += 1
                    self._save()
            if not cached:
                result = self.worker.predict(window, config, "run_" + uuid4().hex)
            if "raw_paths" not in result:
                raise ValueError("Untouched model output is required")
            validate_forecast_output(result["forecast"], result["raw_paths"])
            forecast = freeze_forecast(result["forecast"])
            f = forecast.data
            if (f["as_of"] != window["as_of"] or f["source"] != window["source"]
                    or f["interval_seconds"] != window["interval_seconds"]
                    or f["quote_currency"] != window["quote_currency"] or len(f["paths"]) != config["path_count"]
                    or len(f["times"]) != config["horizon"] or set(f["spots"]) != set(window["assets"])):
                raise ValueError("Prediction does not match request")
            for symbol in window["assets"]:
                if f["spots"][symbol] != window["histories"][symbol][-1]["close"]:
                    raise ValueError("Forecast spot differs from input")
            compute_start = time.perf_counter()
            with self.lock:
                prior_metrics = self.state["metrics"]
                if prior_metrics and self.state["profile_id"] == profile:
                    h = prior_metrics["holdings"]
                    holdings = FrozenInput(canonical(h), prior_metrics["holdings_id"])
                else:
                    holdings = make_holdings(forecast)
                metrics = recalculate(forecast, holdings)
                alerts = evaluate_alerts(metrics, **self.state["policy"])
                runtime = {**result.get("runtime", {}), "data_ms": data_ms,
                           "compute_ms": (time.perf_counter() - compute_start) * 1000,
                           "total_ms": (time.perf_counter() - start) * 1000,
                           "cache_hit": bool(cached)}
                artifact = self.store.write_artifact(result)
                if cached:
                    self.state["counters"]["cache_hits"] += 1
                self.state.update(status="ready", stale=False, error=None, profile_id=profile,
                                  window=window, forecast={**f, "forecast_id": forecast.identity},
                                  metrics=metrics, alerts=alerts, runtime=runtime,
                                  evaluation=empty_evaluation())
                self.state["job"].update(status="ready", finished_at=utcnow())
                self._save({key: artifact, "forecast:" + forecast.identity: {"artifact": artifact, "window": window,
                           "forecast_id": forecast.identity, "job_id": job_id, "predict_config": config,
                           "identity": identity}})
            # Scoring follows publication, and cannot turn a valid prediction into fake model failure.
            try:
                self._score(window, forecast)
                self._score_due_forecasts(forecast.identity)
            except Exception as exc:
                with self.lock:
                    self.state["evaluation"] = {**empty_evaluation(), "status": "failed",
                        "aggregation_status": "failed", "message": "评测汇总失败，已保存预测和单份评分；价格指标不受影响。"}
                    self._save({"evaluation_aggregation_error:" + forecast.identity:
                                {"type": type(exc).__name__, "at": utcnow()}})
        except Exception as exc:
            with self.lock:
                failure = {"type": type(exc).__name__, "at": utcnow()}
                rejected = getattr(exc, "rejected_output", None)
                message = "数据或模型未通过检查，本次分析失败；已有结果未替换。"
                if rejected:
                    try:
                        failure["artifact"] = self.store.write_artifact(diagnostic_json({"rejected_output": rejected,
                            "window": window, "config": config, "job_id": job_id}))
                    except (ValueError, TypeError, OSError):
                        failure["artifact_error"] = "could_not_persist_rejected_output"
                    issues = rejected.get("issues", []) if isinstance(rejected, dict) else []
                    reason = "模型生成了负成交量或负成交额" if any("negative volume/amount" in str(issue) for issue in issues) else "模型输出未通过结构或数值校验"
                    message = reason + "；本批原始结果已保存，未用于计算指标。"
                self.state.update(status="error", stale=self.state["forecast"] is not None,
                                  error={"code": "invalid_prediction" if rejected else "analysis_failed", "message": message, "retryable": True})
                self.state["job"].update(status="failed", finished_at=utcnow())
                self._save({"failure:" + job_id: failure})

    def _score(self, window, forecast):
        try:
            scorer = self.scorer
            if scorer is None:
                from evaluation.scoring import score_forecast
                scorer = score_forecast
            evaluation = scorer(window, forecast.data, forecast.identity)
        except ImportError:
            evaluation = {**empty_evaluation(), "message": "评分器尚未就绪。"}
        except Exception as exc:
            evaluation = {**empty_evaluation(), "status": "failed", "failed_count": 1,
                          "message": "评分失败，预测结果已保留。"}
            with self.lock:
                self.store.put_many({"evaluation_error:" + forecast.identity: {"type": type(exc).__name__}})
        with self.lock:
            evaluation["model_revision"] = forecast.data["model_revision"]
            ids = self.store.get("evaluated_forecasts", [])
            if forecast.identity not in ids:
                ids.append(forecast.identity)
            self.store.put_many({"evaluation:" + forecast.identity: evaluation, "evaluated_forecasts": ids})
            self._refresh_evaluation(ids)

    def _refresh_evaluation(self, ids):
        rows, scored, failed, pending, corrections = [], set(), 0, 0, []
        for fid in ids[-50:]:
            info = self.store.get("forecast:" + fid, {})
            if info.get("window", {}).get("profile_id") != self.state["profile_id"]:
                continue
            value = self.store.get("evaluation:" + fid, {})
            corrections.append({"model_spec": {"model_label": value.get("model", "Kronos-base")},
                "result": {"forecast": {"model_revision": value.get("model_revision", "unavailable")},
                           "runtime": {"identity": info.get("identity", {})}},
                "task_id": fid, "window_id": info.get("window", {}).get("window_id"),
                "as_of": info.get("window", {}).get("as_of"),
                "profile_id": info.get("window", {}).get("profile_id"),
                "source": info.get("window", {}).get("source"),
                "quote_currency": info.get("window", {}).get("quote_currency"),
                "predict_config": info.get("predict_config"),
                "evaluation": value, "status": value.get("status", "pending_truth"),
                "volume_quality": value.get("volume_quality"),
                "volume_quality_status": value.get("volume_quality_status", "missing"),
                "correction_quality": value.get("correction_quality"),
                "correction_quality_status": value.get("correction_quality_status", "missing")})
            if value.get("status") == "scored":
                rows.extend(value.get("rows", []))
                scored.add(value.get("window_id", fid))
            elif value.get("status") == "failed":
                failed += 1
            else:
                pending += 1
        from evaluation.corrections import summarize_quality
        from evaluation.volume_quality import summarize_volume_quality
        self.state["evaluation"] = {"status": "scored" if scored else ("failed" if failed else "pending_truth"),
            "rows": rows, "sample_count": len(scored), "failed_count": failed, "pending_count": pending,
            "quality_status": "insufficient_evidence", "history_limit": 50,
            "correction_quality": summarize_quality(corrections),
            "volume_quality": summarize_volume_quality(corrections),
            "message": "按最近50份预测展示；重复或重叠窗口不是独立样本，尚不据此判断模型有优势。"}
        self._save()

    def _score_due_forecasts(self, current_id):
        # Called only after explicit analysis, never by GET polling. No automatic timer.
        with self.lock:
            ids = self.store.get("evaluated_forecasts", [])[-50:]
        for fid in ids:
            if fid == current_id:
                continue
            with self.lock:
                score = self.store.get("evaluation:" + fid, {})
                info = self.store.get("forecast:" + fid)
            if score.get("status") not in ("pending_truth", "incomplete_truth") or not info:
                continue
            try:
                artifact = self.store.read_artifact(info["artifact"])
                if "raw_paths" not in artifact:
                    raise ValueError("Untouched model output is required")
                self._validate_saved(artifact["forecast"], artifact["raw_paths"])
                f = freeze_forecast(artifact["forecast"])
                end = datetime.fromisoformat(f.data["times"][-1].replace("Z", "+00:00"))
                if end <= datetime.now(timezone.utc):
                    self._score(info["window"], f)
            except (ValueError, OSError, KeyError):
                with self.lock:
                    self.store.put_many({"evaluation:" + fid: {"status": "failed", "rows": []}})

    def _ready(self, body):
        self._revision(body)
        self._busy()
        if not self.state["forecast"]:
            raise APIError(409, "no_forecast", "请先运行一次分析。", True)

    def holdings(self, body):
        with self.lock:
            self._ready(body)
            if body.get("expected_forecast_id") != self.state["forecast"]["forecast_id"]:
                raise APIError(409, "forecast_changed", "预测已更新，请检查后再次应用。", True)
            f = dict(self.state["forecast"])
            f.pop("forecast_id")
            try:
                forecast = freeze_forecast(f)
                holdings = make_holdings(forecast, body["capital"], body["weights"])
                metrics = recalculate(forecast, holdings)
            except (ValueError, KeyError, TypeError) as exc:
                raise APIError(400, "invalid_holdings", "金额必须为正，三个资产权重非负且合计100%。") from exc
            self.state.update(metrics=metrics, alerts=evaluate_alerts(metrics, **self.state["policy"]))
            self._save()
            return self.view()

    def alerts(self, body):
        with self.lock:
            self._ready(body)
            try:
                policy = {"drawdown_threshold": body["drawdown_threshold"], "top_k": body["top_k"]}
                if isinstance(policy["top_k"], bool) or not isinstance(policy["top_k"], int) or not 0 <= policy["top_k"] <= 3:
                    raise ValueError("top_k")
                alerts = evaluate_alerts(self.state["metrics"], **policy)
            except (ValueError, KeyError, TypeError) as exc:
                raise APIError(400, "invalid_policy", "阈值须在0至100%，优先名次数须为0至3。") from exc
            self.state.update(policy=policy, alerts=alerts)
            self._save()
            return self.view()

    def report(self):
        with self.lock:
            if not self.state["metrics"]:
                raise APIError(409, "no_forecast", "请先运行一次分析。")
            result = report_payload(self.state["metrics"], self.state["alerts"])
            result.update(forecast=deepcopy(self.state["forecast"]),
                          ohlc_corrections=deepcopy(self.state["forecast"].get("ohlc_corrections")),
                          evaluation=deepcopy(self.state["evaluation"]),
                          data_quality=deepcopy(self.state["window"].get("quality", {})),
                          runtime=deepcopy(self.state["runtime"]))
            return result

    def wait(self, timeout=240):
        if self.future:
            self.future.result(timeout=timeout)
        return self.view()

    def close(self):
        self.executor.shutdown(wait=True)
        self.worker.close()
        self.store.close()
