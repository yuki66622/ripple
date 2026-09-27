"""One persistent Kronos adapter over bounded JSON-lines stdin/stdout.

No data retrieval or scoring occurs here. All inputs are frozen past-only tasks;
the coordinator owns true future values. stdout is reserved for this protocol.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
import platform
import socket
import sys
import uuid

from evaluation.runner import _hash, _safe_json, _validate_task
from forecast_metrics.engine import freeze_forecast
from model_adapter import validate_forecast_output

PROTOCOL = "kronos-stdio-v1"
MAX_FRAME_BYTES = 32 * 1024 * 1024
IDENTITY_KEYS = ("model_revision", "tokenizer_revision", "adapter_revision", "sampling", "output_policy",
                 "volume_policy", "sktime_version")


def verify_identity(actual, expected):
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError("model identity must be an object")
    for key in IDENTITY_KEYS:
        if key not in expected or actual.get(key) != expected[key]:
            raise ValueError(f"worker identity mismatch: {key}")


def validate_task(task):
    _validate_task(task)
    if set(task) != {"schema_version", "kind", "fold", "origin_index", "window", "predict_config", "model_spec", "task_id"}:
        raise ValueError("unexpected or missing task fields")
    if any(isinstance(task[key], bool) or not isinstance(task[key], int) or task[key] < 0 for key in ("fold", "origin_index")):
        raise ValueError("fold/origin_index must be nonnegative integer metadata")
    window = task["window"]
    if set(window) != {"schema_version", "window_id", "profile_id", "source", "mode", "quote_currency",
                       "as_of", "interval_seconds", "assets", "histories", "quality"}:
        raise ValueError("unexpected or missing MarketWindow fields")
    from data_pipeline import profile_assets
    expected_assets = list(profile_assets(window["profile_id"]))
    expected_mode = "live" if window["profile_id"] == "kraken_live" else "replay"
    if window["assets"] != expected_assets or window["mode"] != expected_mode:
        raise ValueError("declared market assets or mode do not match the profile")
    if not isinstance(window["histories"], dict) or set(window["histories"]) != set(window["assets"]):
        raise ValueError("histories must contain only declared assets")
    quality = window["quality"]
    allowed_quality = {"amount_source", "amount_is_estimate", "timestamp_semantics", "gap_policy",
                       "synthetic_candles", "pairs", "alignment"}
    if not isinstance(quality, dict) or set(quality) - allowed_quality:
        raise ValueError("unexpected MarketWindow quality fields")
    for key, value in quality.items():
        if key == "pairs":
            if not isinstance(value, dict) or set(value) != set(window["assets"]) or any(not isinstance(v, str) for v in value.values()):
                raise ValueError("quality pairs must contain only the declared asset pair strings")
        elif not isinstance(value, (str, bool, int, float)):
            raise ValueError("quality metadata must contain only scalar provenance")
    if set(task["predict_config"]) != {"lookback", "horizon", "path_count", "seed"}:
        raise ValueError("unexpected prediction configuration fields")
    if set(task["model_spec"]) != {"model_path", "tokenizer_path", "model_label"}:
        raise ValueError("unexpected model specification fields")
    for key, value in task["model_spec"].items():
        if value is not None and not isinstance(value, str):
            raise ValueError("model specification values must be strings or null")
    for rows in window["histories"].values():
        for row in rows:
            if set(row) != {"time", "open", "high", "low", "close", "volume", "amount"}:
                raise ValueError("unexpected historical candle fields")
    expected = "window_" + _hash({key: value for key, value in window.items() if key != "window_id"})
    if window.get("window_id") != expected:
        raise ValueError("MarketWindow content does not match window_id")
    # Pure validation, no model load: exact shared grid ending at as_of means
    # every input timestamp is <= as_of, with no future rows or hidden labels.
    from model_adapter.adapter import validate_input
    validate_input(window, task["predict_config"])


def prediction_run_id(task):
    return "evaluation:" + task["task_id"]


def _make_adapter(**kwargs):
    from model_adapter import KronosAdapter
    return KronosAdapter(**kwargs)


class WorkerSession:
    """Protocol state; adapter_factory injection is only used by explicit tests."""

    def __init__(self, *, device="auto", model_path=None, tokenizer_path=None, adapter_factory=None):
        self.device, self.model_path, self.tokenizer_path = device, model_path, tokenizer_path
        self.adapter_factory = adapter_factory or _make_adapter
        self.adapter = None
        self.identity = None
        self.model_spec = None
        self.worker_id = "worker_" + uuid.uuid4().hex
        self.seen = set()
        self.seen_tasks = set()

    def handle(self, request):
        response = {"protocol": PROTOCOL, "worker_id": self.worker_id,
                    "request_id": request.get("request_id") if isinstance(request, dict) else None}
        stage, result = "protocol", None
        try:
            if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
                raise ValueError("unsupported protocol envelope")
            request_id = request.get("request_id")
            if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
                raise ValueError("request_id must be a bounded nonempty string")
            if request_id in self.seen:
                raise ValueError("duplicate request_id; no automatic prediction retry")
            self.seen.add(request_id)
            kind = request.get("kind")
            if kind == "initialize":
                if self.adapter is not None:
                    raise ValueError("worker is already initialized")
                stage = "initialize"
                spec = request["model_spec"]
                if not isinstance(spec, dict):
                    raise ValueError("model_spec must be an object")
                adapter = self.adapter_factory(
                    model_path=self.model_path or spec.get("model_path"),
                    tokenizer_path=self.tokenizer_path or spec.get("tokenizer_path"), device=self.device)
                identity = adapter.identity()
                verify_identity(identity, request["expected_identity"])
                self.adapter, self.identity, self.model_spec = adapter, identity, dict(spec)
                response.update(kind="ready", identity=identity,
                                machine={"hostname": socket.gethostname(), "pid": os.getpid(),
                                         "platform": platform.platform(), "python": platform.python_version()})
                return response
            if kind != "predict" or self.adapter is None:
                raise ValueError("initialize first, then send predict requests")
            task = request["task"]
            response.update(task_id=request.get("task_id"), window_id=request.get("window_id"))
            validate_task(task)
            if task["task_id"] != request.get("task_id") or task["window"]["window_id"] != request.get("window_id"):
                raise ValueError("request/task/window identity mismatch")
            if task["model_spec"] != self.model_spec:
                raise ValueError("task model specification differs from initialized worker")
            if task["task_id"] in self.seen_tasks:
                raise ValueError("duplicate task_id; use a new explicit batch to rerun this task")
            self.seen_tasks.add(task["task_id"])
            stage = "predict"
            result = self.adapter.predict(task["window"], task["predict_config"], prediction_run_id(task))
            stage = "validate_output"
            if not isinstance(result.get("raw_paths"), list) or not result["raw_paths"]:
                raise ValueError("raw_paths are required")
            validate_forecast_output(result["forecast"], result["raw_paths"])
            verify_identity(result["runtime"]["identity"], self.identity)
            if result["forecast"]["model_revision"] != self.identity["model_revision"]:
                raise ValueError("forecast model revision differs from loaded model")
            response.update(kind="result", identity=self.identity, result=result,
                            forecast_id=freeze_forecast(result["forecast"]).identity)
        except Exception as exc:
            response.update(kind="failure", identity=self.identity,
                            error={"stage": stage, "type": type(exc).__name__, "message": str(exc)})
            if result is not None:
                response["result"] = result
            for key in ("raw_paths", "runtime", "issues"):
                if hasattr(exc, key):
                    response[key] = getattr(exc, key)
            response["raw_nonfinite_encoding"] = "Rejected nonfinite floats are represented as NaN/Infinity/-Infinity strings."
        return response


def serve(input_stream, output_stream, error_stream, *, session=None):
    session = session or WorkerSession()
    while True:
        line = input_stream.readline(MAX_FRAME_BYTES + 1)
        if not line:
            return
        if len(line.encode("utf-8")) > MAX_FRAME_BYTES or not line.endswith("\n"):
            raise ValueError("stdin protocol frame exceeds limit or lacks newline")
        try:
            request = json.loads(line)
        except ValueError:
            request = None
        # Python progress and model diagnostics cannot corrupt response framing.
        with redirect_stdout(error_stream):
            response = session.handle(request)
        encoded = json.dumps(_safe_json(response), allow_nan=False, separators=(",", ":")) + "\n"
        if len(encoded.encode("utf-8")) > MAX_FRAME_BYTES:
            raise ValueError("stdout protocol frame exceeds limit")
        output_stream.write(encoded)
        output_stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    args = parser.parse_args(argv)
    # Preserve the original protocol fd, then route even native/C-extension
    # writes to stdout to stderr before model import/loading can log anything.
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        serve(sys.stdin, protocol_output, sys.stderr,
              session=WorkerSession(device=args.device, model_path=args.model_path,
                                    tokenizer_path=args.tokenizer_path))
    finally:
        protocol_output.close()


if __name__ == "__main__":
    main()
