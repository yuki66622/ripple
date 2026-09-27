"""Tonight-only local/SSH batch coordinator; no daemon or deployment actions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

from evaluation.corrections import extract_quality
from evaluation.volume_quality import extract_volume_quality
from evaluation.runner import _hash, _write_new, save_manifest, summarize
from evaluation.scoring import _validate_pair, score_forecast
from forecast_metrics.engine import freeze_forecast
from model_adapter import validate_forecast_output
from .worker_stdio import MAX_FRAME_BYTES, PROTOCOL, prediction_run_id, validate_task, verify_identity

ROOT = Path(__file__).resolve().parents[1]


class TransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class SSHConfig:
    host: str
    cwd: str
    python: str
    known_hosts: str
    control_path: str | None = None
    port: int = 22
    device: str = "auto"
    model_path: str | None = None
    tokenizer_path: str | None = None


def _ssh_option_path(value):
    """Quote at the OpenSSH configuration parser layer, not the local shell."""
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError("invalid SSH option path")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def ssh_command(config):
    """Strict host verification and shell-safe paths; never request passwords."""
    if (not re.fullmatch(r"[A-Za-z0-9_@.:%-]+", config.host or "") or config.host.startswith("-")
            or not isinstance(config.port, int) or isinstance(config.port, bool) or not 1 <= config.port <= 65535):
        raise ValueError("invalid SSH host or port")
    if not config.cwd.startswith("/") or not config.python or "\n" in config.cwd + config.python:
        raise ValueError("remote cwd must be absolute and remote python must be explicit")
    known_hosts = Path(config.known_hosts).expanduser().resolve()
    if not known_hosts.is_file():
        raise ValueError("an existing verified known_hosts file is required")
    command = ["ssh", "-T", "-p", str(config.port), "-o", "BatchMode=yes",
               "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + _ssh_option_path(str(known_hosts)),
               "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
    if config.control_path:
        command += ["-o", "ControlMaster=auto", "-o", "ControlPersist=60",
                    "-o", "ControlPath=" + _ssh_option_path(str(Path(config.control_path).expanduser()))]
    worker = [config.python, "-u", "-m", "demo_app.worker_stdio", "--device", config.device]
    for name, value in (("--model-path", config.model_path), ("--tokenizer-path", config.tokenizer_path)):
        if value:
            worker.extend([name, value])
    remote = "cd " + shlex.quote(config.cwd) + " && exec " + shlex.join(worker)
    return [*command, config.host, remote]


class ProcessTransport:
    """One request at a time over nonblocking pipes, with a total call deadline."""

    def __init__(self, command, *, cwd, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("xb")
        self.closed = False
        try:
            self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.log,
                                            start_new_session=True, bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
        except Exception:
            self.log.close()
            raise

    def call(self, request, timeout):
        if self.closed:
            raise TransportError("worker connection is closed")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        encoded = (json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if len(encoded) > MAX_FRAME_BYTES:
            raise TransportError("request exceeds protocol size limit")
        deadline = time.monotonic() + timeout
        sent, received = 0, bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE)
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close()
                    raise TransportError("worker call timed out; task not retried")
                events = selector.select(remaining)
                if not events:
                    continue
                for key, event in events:
                    if event & selectors.EVENT_WRITE:
                        try:
                            sent += os.write(key.fd, encoded[sent:sent + 65536])
                        except BrokenPipeError as exc:
                            self.close()
                            raise TransportError("worker closed stdin") from exc
                        if sent == len(encoded):
                            selector.unregister(self.process.stdin)
                    elif event & selectors.EVENT_READ:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            self.close()
                            raise TransportError("worker exited before completing its response")
                        received.extend(chunk)
                        if len(received) > MAX_FRAME_BYTES:
                            self.close()
                            raise TransportError("worker response exceeds protocol size limit")
                        if b"\n" in received:
                            line, extra = bytes(received).split(b"\n", 1)
                            if extra:
                                self.close()
                                raise TransportError("unsolicited stdout after protocol response")
                            try:
                                response = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                            except (ValueError, UnicodeError) as exc:
                                self.close()
                                raise TransportError("worker stdout is not a valid JSON protocol frame") from exc
                            if not isinstance(response, dict):
                                self.close()
                                raise TransportError("worker response must be an object")
                            return response

    def close(self):
        if self.closed:
            return
        self.closed = True
        process = getattr(self, "process", None)
        if process is not None:
            for handle in (process.stdin, process.stdout):
                if handle:
                    handle.close()
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=2)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
        self.log.close()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tasks"), list) or not manifest["tasks"]:
        raise ValueError("manifest must contain tasks")
    ids = []
    for task in manifest["tasks"]:
        validate_task(task)
        ids.append(task["task_id"])
        if task["model_spec"] != manifest["model_spec"] or task["predict_config"] != manifest["config"]:
            raise ValueError("manifest must use one fixed model specification and configuration")
    if len(set(ids)) != len(ids) or ids != manifest.get("task_ids") or len(ids) != manifest.get("selected_count"):
        raise ValueError("manifest task list/count mismatch")
    metadata = {key: value for key, value in manifest.items() if key not in ("tasks", "experiment_id")}
    if manifest.get("experiment_id") != "experiment_" + _hash(metadata):
        raise ValueError("manifest experiment identity mismatch")


def _request(kind, **fields):
    return {"protocol": PROTOCOL, "kind": kind, "request_id": "request_" + uuid.uuid4().hex, **fields}


def _verify_envelope(response, request, *, worker_id=None):
    if response.get("protocol") != PROTOCOL or response.get("request_id") != request["request_id"]:
        raise TransportError("response protocol/request_id mismatch")
    if not isinstance(response.get("worker_id"), str) or not response["worker_id"]:
        raise TransportError("response lacks worker identity")
    if worker_id is not None and response["worker_id"] != worker_id:
        raise TransportError("worker identity changed mid-session")
    if request["kind"] == "predict":
        if response.get("task_id") != request["task_id"] or response.get("window_id") != request["window_id"]:
            raise TransportError("response task/window identity mismatch")


def accept_result(task, response, *, expected_identity):
    """Coordinator-side publication boundary; never trust worker-computed scores."""
    result = response["result"]
    verify_identity(response["identity"], expected_identity)
    verify_identity(result["runtime"]["identity"], expected_identity)
    forecast = result["forecast"]
    if forecast.get("model_revision") != expected_identity["model_revision"]:
        raise ValueError("forecast model revision mismatch")
    if forecast.get("prediction_run_id") != prediction_run_id(task):
        raise ValueError("forecast prediction_run_id mismatch")
    if not isinstance(result.get("raw_paths"), list) or not result["raw_paths"]:
        raise ValueError("worker must return complete original raw_paths")
    validate_forecast_output(forecast, result["raw_paths"])
    frozen = freeze_forecast(forecast)
    if response.get("forecast_id") != frozen.identity:
        raise ValueError("worker forecast_id does not match returned content")
    _validate_pair(task["window"], frozen)
    config = task["predict_config"]
    if len(forecast["paths"]) != config["path_count"] or len(forecast["times"]) != config["horizon"]:
        raise ValueError("worker returned an incomplete requested workload")
    if result["runtime"].get("path_count") != config["path_count"]:
        raise ValueError("worker runtime path count mismatch")
    return frozen


def execute_manifest(manifest, *, transports, expected_identity, out_dir, task_timeout=300, init_timeout=90, scorer=None):
    """Run each task once using 1-2 transport factories and score only locally.

    factories are {name: callable(log_path)->transport}. Injection permits
    protocol tests without model inference, deployment or SSH connections.
    """
    validate_manifest(manifest)
    verify_identity(expected_identity, expected_identity)
    if not 1 <= len(transports) <= 2:
        raise ValueError("this bounded batch helper accepts one or two workers")
    if task_timeout <= 0 or init_timeout <= 0:
        raise ValueError("timeouts must be positive")
    score = scorer or score_forecast
    out_dir = Path(out_dir)
    save_manifest(manifest, out_dir)
    run_id = "batch_" + uuid.uuid4().hex
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True)
    tasks = manifest["tasks"]
    names = list(transports)
    # Reserve one first task for each selected worker so a fast startup cannot
    # consume a tiny both-machine pilot before the other worker is initialized.
    initial = {name: tasks[i] for i, name in enumerate(names) if i < len(tasks)}
    remaining = queue.Queue()
    for task in tasks[len(initial):]:
        remaining.put(task)
    artifacts, workers, active_transports = [], {}, {}
    lock = threading.Lock()
    cancelled = threading.Event()
    started = time.perf_counter()

    def store(task, artifact):
        destination = run_dir / "tasks" / (task["task_id"] + ".json")
        artifact["artifact_path"] = str(destination.resolve())
        _write_new(destination, artifact)
        with lock:
            artifacts.append(artifact)

    def base(task, name):
        return {"schema_version": 1, "task_id": task["task_id"], "window_id": task["window"]["window_id"],
                "as_of": task["window"]["as_of"], "model_spec": task["model_spec"],
                "profile_id": task["window"]["profile_id"], "source": task["window"]["source"],
                "quote_currency": task["window"]["quote_currency"],
                "predict_config": task["predict_config"], "worker": name,
                "status": "failed", "correction_quality": None, "correction_quality_status": "missing",
                "volume_quality": None, "volume_quality_status": "missing"}

    def retain_quality(artifact, forecast):
        # Scalar extraction preserves diagnostics, but cannot replace the full
        # model-owned audit/raw-output validation before publication.
        for name, extract in (("correction_quality", extract_quality), ("volume_quality", extract_volume_quality)):
            try:
                quality = extract(forecast)
                artifact[name] = quality
                artifact[name + "_status"] = "missing" if quality is None else "unvalidated"
            except (TypeError, KeyError, ValueError):
                artifact[name + "_status"] = "invalid"

    def worker_loop(name):
        worker = {"status": "initializing", "accepted_tasks": 0, "failed_tasks": 0}
        with lock:
            workers[name] = worker
        if name not in initial:
            worker["status"] = "not_used_no_assigned_task"
            return
        transport, current = None, initial[name]
        stage = "initialize"
        try:
            transport = transports[name](run_dir / "logs" / (name + ".stderr.log"))
            with lock:
                active_transports[name] = transport
            if cancelled.is_set():
                worker["status"] = "cancelled"
                current = None
                return
            initialize = _request("initialize", model_spec=manifest["model_spec"], expected_identity=expected_identity)
            ready = transport.call(initialize, init_timeout)
            worker["initialize_response"] = ready
            _verify_envelope(ready, initialize)
            if ready.get("kind") != "ready":
                raise TransportError("worker initialization failed: " + str(ready.get("error", {})))
            verify_identity(ready.get("identity"), expected_identity)
            worker.update(status="ready", worker_id=ready["worker_id"], identity=ready["identity"], machine=ready.get("machine"))
            while current is not None:
                if cancelled.is_set():
                    worker["status"] = "cancelled"
                    current = None
                    break
                artifact = base(current, name)
                tick = time.perf_counter()
                request = _request("predict", task_id=current["task_id"], window_id=current["window"]["window_id"], task=current)
                artifact["request_id"] = request["request_id"]
                fatal = False
                try:
                    stage = "transport"
                    response = transport.call(request, task_timeout)
                    artifact["response"] = response
                    _verify_envelope(response, request, worker_id=worker["worker_id"])
                    verify_identity(response.get("identity"), expected_identity)
                    if response.get("kind") == "failure":
                        artifact["error"] = response.get("error", {"stage": "worker", "message": "worker failed"})
                        for key in ("raw_paths", "runtime", "issues", "result", "raw_nonfinite_encoding"):
                            if key in response:
                                artifact[key] = response[key]
                        if isinstance(response.get("result"), dict) and isinstance(response["result"].get("forecast"), dict):
                            retain_quality(artifact, response["result"]["forecast"])
                    elif response.get("kind") == "result":
                        artifact["result"] = response["result"]
                        stage = "validate_output"
                        retain_quality(artifact, response["result"]["forecast"])
                        frozen = accept_result(current, response, expected_identity=expected_identity)
                        artifact.update(forecast_id=frozen.identity, correction_quality_status="validated",
                                        volume_quality_status="validated")
                        stage = "score"
                        evaluation = score(current["window"], frozen.data, frozen.identity,
                                           model_label=current["model_spec"]["model_label"])
                        artifact.update(evaluation=evaluation, status=evaluation["status"])
                    else:
                        raise TransportError("unexpected worker response kind")
                except Exception as exc:
                    artifact["status"] = "failed"
                    artifact["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
                    # An audit is not marked validated unless the complete
                    # publication gate passed. Unvalidated diagnostics remain
                    # visible but do not contribute quality statistics.
                    # Quarantine transport/identity/publication violations. A
                    # reported model failure remains visible but can continue.
                    fatal = stage in {"transport", "validate_output"}
                artifact["wall_ms"] = (time.perf_counter() - tick) * 1000
                store(current, artifact)
                worker["failed_tasks" if artifact["status"] == "failed" else "accepted_tasks"] += 1
                current = None
                if fatal:
                    worker["status"] = "quarantined"
                    break
                try:
                    current = remaining.get_nowait()
                except queue.Empty:
                    break
            if worker["status"] not in {"quarantined", "cancelled"}:
                worker["status"] = "completed"
        except Exception as exc:
            worker.update(status="failed", error={"stage": stage, "type": type(exc).__name__, "message": str(exc)})
            if current is not None:
                artifact = base(current, name)
                artifact["error"] = worker["error"]
                store(current, artifact)
                worker["failed_tasks"] += 1
        finally:
            if transport is not None:
                transport.close()
            with lock:
                active_transports.pop(name, None)

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = [pool.submit(worker_loop, name) for name in names]
        try:
            for future in futures:
                future.result()
        except KeyboardInterrupt:
            cancelled.set()
            with lock:
                to_close = list(active_transports.values())
            for transport in to_close:
                transport.close()
            for future in futures:
                future.result()
    accepted_ids = {artifact["task_id"] for artifact in artifacts}
    not_run = []
    for task in tasks:
        if task["task_id"] not in accepted_ids:
            artifact = base(task, "unassigned")
            artifact.update(status="not_run", error={"stage": "dispatch", "message": "No usable worker remained; no task was retried."})
            destination = run_dir / "tasks" / (task["task_id"] + ".json")
            artifact["artifact_path"] = str(destination.resolve())
            _write_new(destination, artifact)
            not_run.append(artifact)
    order = {task["task_id"]: i for i, task in enumerate(tasks)}
    artifacts.sort(key=lambda artifact: order[artifact["task_id"]])
    summary = summarize(artifacts, requested_count=len(tasks))
    summary.update(run_id=run_id, experiment_id=manifest["experiment_id"],
                   batch_wall_ms=(time.perf_counter() - started) * 1000, workers=workers,
                   not_run_tasks=[{"task_id": a["task_id"], "artifact_path": a["artifact_path"]} for a in not_run],
                   execution_scope="startup + transfer + inference + coordinator validation/scoring + artifacts; no automatic retries",
                   speedup_claim="unmeasured; compare complete runs on the identical manifest",
                   interrupted=cancelled.is_set(),
                   expected_identity=expected_identity, recorded_at=datetime.now(timezone.utc).isoformat())
    destination = run_dir / "summary.json"
    summary["summary_path"] = str(destination.resolve())
    _write_new(destination, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", required=True, choices=("local", "remote", "both"))
    parser.add_argument("--python", default=sys.executable, help="local worker Python")
    parser.add_argument("--device", default="auto", help="local worker device")
    parser.add_argument("--task-timeout", type=float, default=300)
    parser.add_argument("--init-timeout", type=float, default=90)
    parser.add_argument("--expected-identity", help="optional pinned identity JSON; otherwise hash local model files")
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--remote-cwd")
    parser.add_argument("--remote-python")
    parser.add_argument("--known-hosts")
    parser.add_argument("--control-path")
    parser.add_argument("--remote-device", default="auto")
    parser.add_argument("--remote-model-path")
    parser.add_argument("--remote-tokenizer-path")
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    transports = {}
    if args.mode in ("local", "both"):
        command = [args.python, "-u", "-m", "demo_app.worker_stdio", "--device", args.device]
        transports["local"] = lambda log: ProcessTransport(command, cwd=ROOT, log_path=log)
    if args.mode in ("remote", "both"):
        if not all((args.ssh_host, args.remote_cwd, args.remote_python, args.known_hosts)):
            parser.error("remote mode requires --ssh-host, --remote-cwd, --remote-python and --known-hosts")
        remote_command = ssh_command(SSHConfig(args.ssh_host, args.remote_cwd, args.remote_python,
                                              args.known_hosts, args.control_path, args.ssh_port,
                                              args.remote_device, args.remote_model_path, args.remote_tokenizer_path))
        transports["remote"] = lambda log: ProcessTransport(remote_command, cwd=ROOT, log_path=log)
    if args.expected_identity:
        expected = json.loads(Path(args.expected_identity).read_text())
    else:
        from model_adapter import KronosAdapter
        spec = manifest["model_spec"]
        expected = KronosAdapter(model_path=spec.get("model_path"), tokenizer_path=spec.get("tokenizer_path")).identity()
    summary = execute_manifest(manifest, transports=transports, expected_identity=expected,
                               out_dir=args.out, task_timeout=args.task_timeout, init_timeout=args.init_timeout)
    print(json.dumps({key: summary[key] for key in ("summary_path", "requested_count", "sample_count",
                                                   "failed_count", "pending_count", "incomplete_count",
                                                   "not_run_count", "batch_wall_ms")}, ensure_ascii=False))
    return 1 if summary["failed_count"] or summary["incomplete_count"] or summary["not_run_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
