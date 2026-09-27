"""Preserve individual Kronos samples through sktime's inference interface.

Input and public output timestamps are UTC candle ends. Model calendar features
use candle starts. Every path uses sample_count=1 because upstream averages that
dimension. Approved containment expands only high/low with a complete audit;
structurally invalid output is rejected and no path is resampled.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time

from .corrections import POLICY, correct_paths, structure_issues
from .volume_quality import VOLUME_POLICY, build_volume_quality, validate_forecast_output

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
MAX_ASSETS = 100
ADAPTER_REVISION = "sktime-kronos-declared-assets-v4"
PAIRING = "paired_scenarios_not_calibrated_joint_distribution"
SAMPLING = {"T": 1.0, "top_k": 0, "top_p": 0.9, "sample_count": 1, "verbose": False}
_MODEL_LOCK = threading.RLock()


class PredictionValidationError(ValueError):
    """Rejected generated data remains available for failure artifacts."""

    def __init__(self, issues, *, raw_paths, runtime):
        self.issues = issues
        self.raw_paths = raw_paths
        self.runtime = runtime
        super().__init__(f"Kronos output rejected ({len(issues)} issues): {issues[0]}")


def stable_seed(seed, window_id, asset, path_index):
    """Seed independent of run ID, worker assignment and asset execution order."""
    text = json.dumps([seed, window_id, asset, path_index], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def _stamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO UTC strings")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError("timestamps must explicitly use UTC")
    return dt


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    return float(value)


def _int(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def validate_input(window, config):
    """Validate before expensive loading; never fill gaps or missing amount."""
    if set(config) - {"lookback", "horizon", "path_count", "seed"}:
        raise ValueError("unknown prediction config fields")
    cfg = {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926, **config}
    for key, lo, hi in (("lookback", 2, 512), ("horizon", 2, 128), ("path_count", 1, 16), ("seed", 0, 2**32 - 1)):
        _int(cfg[key], key, lo, hi)
    if window.get("schema_version") != 1:
        raise ValueError("MarketWindow v1 is required")
    assets = window.get("assets")
    if (not isinstance(assets, list) or not 1 <= len(assets) <= MAX_ASSETS
            or any(not isinstance(asset, str) or re.fullmatch(r"[A-Z0-9]{1,20}", asset) is None for asset in assets)
            or len(set(assets)) != len(assets)):
        raise ValueError("assets must be an ordered list of 1..100 unique uppercase alphanumeric symbols (1..20 characters)")
    if not isinstance(window.get("histories"), dict) or set(window["histories"]) != set(assets):
        raise ValueError("histories must match the complete declared asset set")
    if window.get("interval_seconds") != 60:
        raise ValueError("this adapter contract requires 60-second candles")
    for name in ("window_id", "source", "quote_currency"):
        if not isinstance(window.get(name), str) or not window[name]:
            raise ValueError(f"missing {name}")
    as_of = _stamp(window["as_of"])
    shared_times = None
    frames = {}
    for asset in assets:
        rows = window["histories"][asset]
        if len(rows) != cfg["lookback"]:
            raise ValueError(f"{asset}: history length must equal lookback")
        times, values = [], []
        for i, row in enumerate(rows):
            times.append(_stamp(row["time"]))
            val = {key: _number(row[key], f"{asset}/{i}/{key}") for key in COLUMNS}
            if min(val[key] for key in COLUMNS[:4]) <= 0 or min(val[key] for key in COLUMNS[4:]) < 0:
                raise ValueError(f"{asset}/{i}: prices must be positive; volume/amount nonnegative")
            if not val["low"] <= min(val["open"], val["close"]) <= max(val["open"], val["close"]) <= val["high"]:
                raise ValueError(f"{asset}/{i}: inconsistent OHLC")
            values.append(val)
        if times[-1] != as_of or any(b - a != timedelta(seconds=60) for a, b in zip(times, times[1:])):
            raise ValueError(f"{asset}: history must end at as_of and have no minute gaps")
        if shared_times is not None and times != shared_times:
            raise ValueError("asset histories must have identical timestamps")
        shared_times = times
        frames[asset] = values
    return cfg, shared_times, frames


def validate_output(raw_paths, horizon):
    """Strict final validation; structural errors never reach correction."""
    issues = structure_issues(raw_paths, horizon)
    if issues:
        return issues
    for path in raw_paths:
        for asset, values in path["assets"].items():
            for i in range(horizon):
                row = {key: values[key][i] for key in COLUMNS}
                prefix = f"{path['path_id']}/{asset}/{i}"
                if not row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]:
                    issues.append(f"{prefix}: inconsistent OHLC")
    return issues


def assemble_forecast(window, config, prediction_run_id, raw_paths, *, model_revision):
    """Pure postprocessing of existing raw output; performs no model call."""
    cfg, times, rows = validate_input(window, config)
    assets = list(rows)
    if not isinstance(prediction_run_id, str) or not prediction_run_id:
        raise ValueError("prediction_run_id must be a nonempty string")
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("model_revision must be a nonempty string")
    issues = structure_issues(raw_paths, cfg["horizon"], assets, cfg["path_count"])
    if issues:
        raise ValueError("; ".join(issues))
    public_times = [_iso(times[-1] + timedelta(seconds=60 * (i + 1))) for i in range(cfg["horizon"])]
    corrected_paths, audit = correct_paths(raw_paths, public_times, assets, cfg["path_count"])
    issues = validate_output(corrected_paths, cfg["horizon"])
    if issues:
        raise ValueError("; ".join(issues))
    forecast = {
        "model_revision": model_revision, "prediction_run_id": prediction_run_id,
        "source": window["source"], "quote_currency": window["quote_currency"],
        "as_of": window["as_of"], "interval_seconds": 60, "times": public_times,
        "spots": {asset: rows[asset][-1]["close"] for asset in assets},
        "pairing": PAIRING,
        "paths": [{"path_id": p["path_id"], "assets": {asset: {key: vals[key] for key in ("close", "high", "low")} for asset, vals in p["assets"].items()}} for p in corrected_paths],
        "ohlc_corrections": audit,
        "volume_quality": build_volume_quality(raw_paths, public_times),
    }
    validate_forecast_output(forecast, raw_paths)
    return forecast


class KronosAdapter:
    """One reusable, serialized sktime adapter per process (local weights only)."""

    def __init__(self, *, model_path=None, tokenizer_path=None, device=None):
        self.model_path = Path(model_path or ROOT / "scenario-lab/models/Kronos-base").resolve()
        self.tokenizer_path = Path(tokenizer_path or ROOT / "scenario-lab/models/Kronos-Tokenizer-base").resolve()
        self.device = device or os.environ.get("KRONOS_DEVICE", "auto")
        self._identity = None
        self._forecaster = None
        self._torch = self._pd = self._np = None
        self._load_ms = None

    def identity(self):
        with _MODEL_LOCK:
            if self._identity is None:
                import torch
                if self.device == "auto":
                    backend = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                else:
                    backend = self.device
                if backend not in ("cpu", "mps", "cuda", "cuda:0"):
                    raise ValueError("device must be auto, cpu, mps, cuda or cuda:0")
                if backend == "mps" and not torch.backends.mps.is_available():
                    raise RuntimeError("MPS requested but unavailable; no silent fallback")
                if backend.startswith("cuda") and not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable; no silent fallback")
                hashes = {}
                for label, directory in (("model", self.model_path), ("tokenizer", self.tokenizer_path)):
                    digest = hashlib.sha256()
                    for filename in ("config.json", "model.safetensors"):
                        with (directory / filename).open("rb") as handle:
                            digest.update(filename.encode())
                            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                                digest.update(chunk)
                    hashes[label] = "sha256:" + digest.hexdigest()
                self._identity = {
                    "model_revision": hashes["model"], "tokenizer_revision": hashes["tokenizer"],
                    "adapter_revision": ADAPTER_REVISION, "backend": backend,
                    "sktime_version": importlib.metadata.version("sktime"),
                    "torch_version": torch.__version__, "sampling": dict(SAMPLING),
                    "model_timestamp": "UTC candle start", "output_timestamp": "UTC candle end",
                    "output_policy": POLICY,
                    "volume_policy": VOLUME_POLICY,
                }
            return json.loads(json.dumps(self._identity))

    def _synchronize(self):
        backend = self._identity["backend"]
        if backend == "mps":
            self._torch.mps.synchronize()
        elif backend.startswith("cuda"):
            self._torch.cuda.synchronize()

    def _load(self, first_frame):
        if self._forecaster is not None:
            return 0.0
        started = time.perf_counter()
        identity = self.identity()
        import numpy as np
        import pandas as pd
        import torch
        from .sktime_compat import LocalKronosForecaster
        self._torch, self._pd, self._np = torch, pd, np
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        forecaster = LocalKronosForecaster(
            model_path=str(self.model_path), tokenizer_path=str(self.tokenizer_path),
            device=identity["backend"], columns=COLUMNS, freq="1min",
            predict_kwargs=dict(SAMPLING), deterministic=False,
        )
        forecaster.set_config(remember_data=False)
        # fit binds a historical window and loads/caches weights, not training.
        forecaster.fit(first_frame)
        self._synchronize()
        self._load_ms = (time.perf_counter() - started) * 1000
        self._forecaster = forecaster
        return self._load_ms

    def predict(self, window, config, prediction_run_id):
        cfg, times, rows = validate_input(window, config)
        assets = list(rows)
        if not isinstance(prediction_run_id, str) or not prediction_run_id:
            raise ValueError("prediction_run_id must be a nonempty string")
        with _MODEL_LOCK:
            import pandas as pd
            from sktime.forecasting.base import ForecastingHorizon
            started_total = time.perf_counter()
            model_times = pd.DatetimeIndex([t - timedelta(seconds=60) for t in times], freq="1min")
            frames = {asset: pd.DataFrame(rows[asset], index=model_times, columns=COLUMNS) for asset in assets}
            load_this_call = self._load(frames[assets[0]])
            future_ends = [times[-1] + timedelta(seconds=60 * (i + 1)) for i in range(cfg["horizon"])]
            fh = ForecastingHorizon(pd.DatetimeIndex([t - timedelta(seconds=60) for t in future_ends], freq="1min"), is_relative=False)
            self._torch.use_deterministic_algorithms(True)
            self._synchronize()
            started = time.perf_counter()
            raw_paths, call_timings, interface_issues = [], [], []
            for path_index in range(cfg["path_count"]):
                path = {"path_id": f"path-{path_index:03d}", "assets": {}, "seeds": {}}
                for asset in assets:
                    seed = stable_seed(cfg["seed"], window["window_id"], asset, path_index)
                    random.seed(seed)
                    self._np.random.seed(seed)
                    self._torch.manual_seed(seed)
                    if self._identity["backend"] == "mps":
                        self._torch.mps.manual_seed(seed)
                    elif self._identity["backend"].startswith("cuda"):
                        self._torch.cuda.manual_seed_all(seed)
                    self._synchronize()
                    call_started = time.perf_counter()
                    with self._torch.inference_mode():
                        self._forecaster.fit(frames[asset])
                        output = self._forecaster.predict(fh=fh)
                    self._synchronize()
                    call_timings.append({"asset": asset, "path_index": path_index, "seed": seed,
                                         "fit_predict_ms": (time.perf_counter() - call_started) * 1000})
                    if list(output.columns) != COLUMNS or len(output) != cfg["horizon"] or not output.index.equals(fh.to_pandas()):
                        interface_issues.append(f"{path['path_id']}/{asset}: unexpected sktime output shape/index")
                    path["assets"][asset] = {key: [float(v) for v in output[key]] for key in output.columns}
                    path["seeds"][asset] = seed
                raw_paths.append(path)
            self._synchronize()
            inference_ms = (time.perf_counter() - started) * 1000
            runtime = {
                "load_ms": load_this_call, "resident_load_ms": self._load_ms,
                "inference_ms": inference_ms, "total_ms": (time.perf_counter() - started_total) * 1000,
                "backend": self._identity["backend"], "path_count": cfg["path_count"],
                "lookback": cfg["lookback"], "horizon": cfg["horizon"],
                "calls": call_timings, "identity": self.identity(),
                "path_generation": "sequential sample_count=1; no averaging or rejection resampling",
            }
            issues = interface_issues + structure_issues(raw_paths, cfg["horizon"], assets, cfg["path_count"])
            if issues:
                raise PredictionValidationError(issues, raw_paths=raw_paths, runtime=runtime)
            correction_started = time.perf_counter()
            try:
                forecast = assemble_forecast(window, cfg, prediction_run_id, raw_paths,
                                             model_revision=self._identity["model_revision"])
            except ValueError as exc:
                raise PredictionValidationError([f"forecast output rejected: {exc}"], raw_paths=raw_paths, runtime=runtime) from exc
            runtime["correction_ms"] = (time.perf_counter() - correction_started) * 1000
            runtime["total_ms"] = (time.perf_counter() - started_total) * 1000
            return {"forecast": forecast, "runtime": runtime, "raw_paths": raw_paths}
