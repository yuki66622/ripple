"""Live data cache and explicit, frozen-input model requests."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import uuid

from demo_app.worker import LocalWorker
from forecast_metrics.engine import freeze_forecast
from research.shock_radar.metrics import risk_metrics
from .market import fetch_market

CONFIG = {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926}


class LiveError(Exception):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


class LiveService:
    def __init__(self, fetcher=fetch_market, worker=None, directory=None):
        self.fetcher = fetcher
        self.worker = worker or LocalWorker(timeout=180)
        self.directory = Path(directory or Path(__file__).parent / "runtime")
        self.guard = threading.RLock()
        self.fetch_lock = threading.Lock()
        self.snapshot = None
        self.windows = {}
        self.last_attempt = 0
        self.feed_error = None
        self.prediction = {"status": "idle", "error": None}
        self.predictions = {}

    def market(self):
        with self.fetch_lock:
            if time.monotonic() - self.last_attempt > 20:
                self.last_attempt = time.monotonic()
                try:
                    result = self.fetcher()
                    with self.guard:
                        self.snapshot = copy.deepcopy(result["snapshot"])
                        window = copy.deepcopy(result["window"])
                        self.windows[window["window_id"]] = window
                        while len(self.windows) > 8:
                            old_id = next(iter(self.windows))
                            self.windows.pop(old_id)
                            if self.predictions.get(old_id, {}).get("status") != "running":
                                self.predictions.pop(old_id, None)
                        self.feed_error = None
                except Exception as exc:
                    self.feed_error = f"行情读取失败（{type(exc).__name__}），请重试。"
            with self.guard:
                if self.snapshot is None:
                    raise LiveError(self.feed_error or "行情正在连接，请重试。")
                result = copy.deepcopy(self.snapshot)
                age = datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(result["as_of"].replace("Z", "+00:00")).timestamp()
                result["age_seconds"] = max(0, age)
                result["stale"] = bool(self.feed_error) or age > 180
                result["error"] = self.feed_error
                return result

    def prediction_state(self):
        with self.guard:
            return copy.deepcopy(self.prediction)

    def predict(self, window_id):
        with self.guard:
            if not isinstance(window_id, str) or window_id not in self.windows:
                raise LiveError("这个行情窗口已过期，请刷新行情后分析。", 409)
            if window_id in self.predictions:
                # A failed sample is preserved; no resampling-until-success.
                return copy.deepcopy(self.predictions[window_id])
            if self.prediction["status"] == "running":
                raise LiveError("上一次分析仍在运行，请稍候。", 409)
            window = copy.deepcopy(self.windows[window_id])
            age = datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(window["as_of"].replace("Z", "+00:00")).timestamp()
            if age > 180 or self.feed_error:
                raise LiveError("行情已过期或连接异常，请刷新后分析。", 409)
            self.prediction = {"status": "running", "window_id": window_id,
                               "as_of": window["as_of"], "error": None}
            self.predictions[window_id] = copy.deepcopy(self.prediction)
            threading.Thread(target=self._predict, args=(window,), daemon=True).start()
            return copy.deepcopy(self.prediction)

    def _predict(self, window):
        started = time.monotonic()
        run_id = "ripple-live-" + uuid.uuid4().hex
        artifact = {"window": window, "config": CONFIG, "run_id": run_id}
        try:
            result = self.worker.predict(window, CONFIG, run_id)
            from model_adapter import validate_forecast_output
            validate_forecast_output(result["forecast"], result["raw_paths"])
            frozen = freeze_forecast(result["forecast"])
            forecast = result["forecast"]
            if (any(forecast[key] != window[key] for key in ("source", "quote_currency", "as_of"))
                    or set(forecast["spots"]) != set(window["assets"])
                    or len(forecast["paths"]) != CONFIG["path_count"]
                    or len(forecast["times"]) != CONFIG["horizon"]
                    or any(forecast["spots"][asset] != window["histories"][asset][-1]["close"] for asset in window["assets"])):
                raise ValueError("Prediction does not belong to the requested market window")
            assets = {}
            for asset, values in forecast["paths"][0]["assets"].items():
                metrics = risk_metrics(forecast["spots"][asset], values["close"])
                assets[asset] = {"close": values["close"], "mdd": metrics["max_drawdown"],
                                 "volatility": metrics["volatility"]}
            state = {"status": "ready", "window_id": window["window_id"], "as_of": window["as_of"],
                     "forecast_id": frozen.identity, "assets": assets, "path_count": 1,
                     "model": "Kronos-base", "times": forecast["times"],
                     "elapsed_seconds": time.monotonic() - started, "error": None}
            artifact.update(result=result, public_result=state)
        except Exception as exc:
            state = {"status": "error", "window_id": window["window_id"], "as_of": window["as_of"],
                     "error": f"本次模型分析未能通过校验（{type(exc).__name__}）。行情仍可查看。"}
            artifact.update(error_type=type(exc).__name__, rejected_output=getattr(exc, "rejected_output", None))
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"{run_id}.json").open("x") as handle:
                json.dump(artifact, handle, allow_nan=False)
        except Exception:
            state = {"status": "error", "window_id": window["window_id"], "as_of": window["as_of"],
                     "error": "本次结果未能保存，未发布模型分析。"}
        with self.guard:
            self.prediction = state
            self.predictions[window["window_id"]] = copy.deepcopy(state)

    def close(self):
        self.worker.close()
