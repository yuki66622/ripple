"""Marked fake adapters/transports only; no Kronos inference or SSH connection."""

import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from evaluation.runner import _hash, build_manifest
from evaluation.test_evaluation import FixtureAdapter, RejectedFixture
from forecast_metrics.engine import freeze_forecast
from model_adapter import build_volume_quality
from model_adapter.adapter import ADAPTER_REVISION
from demo_app.distributed import (ProcessTransport, SSHConfig, TransportError,
                                  execute_manifest, ssh_command, validate_manifest)
from demo_app.worker_stdio import PROTOCOL, WorkerSession, serve, validate_task


def fixture_identity():
    return {"model_revision": "TEST_ONLY_numerical_fixture", "tokenizer_revision": "TEST_ONLY_tokenizer",
            "adapter_revision": ADAPTER_REVISION, "sampling": {"sample_count": 1},
            "output_policy": "containment-expand-v1", "sktime_version": "TEST_ONLY_version",
            "volume_policy": "unused-volume-audit-v1",
            "torch_version": "TEST_ONLY_version", "backend": "TEST_ONLY_cpu"}


class ProtocolFixtureAdapter(FixtureAdapter):
    def __init__(self, **kwargs):
        self.calls = 0

    def identity(self):
        return fixture_identity()

    def predict(self, window, config, prediction_run_id):
        self.calls += 1
        result = super().predict(window, config, prediction_run_id)
        result["forecast"]["volume_quality"] = build_volume_quality(result["raw_paths"], result["forecast"]["times"])
        result["runtime"].update(identity=self.identity(), path_count=config["path_count"])
        return result


class NoisyFixtureAdapter(ProtocolFixtureAdapter):
    def predict(self, *args):
        print("TEST_ONLY model progress belongs on stderr")
        return super().predict(*args)


class RejectFirstAdapter(ProtocolFixtureAdapter):
    def predict(self, *args):
        if not self.calls:
            self.calls += 1
            raise RejectedFixture()
        return super().predict(*args)


class NegativeVolumeFixtureAdapter(ProtocolFixtureAdapter):
    def predict(self, *args):
        result = super().predict(*args)
        candle = result["raw_paths"][0]["assets"]["BTC"]
        candle["volume"][0], candle["amount"][0] = -2.5, -7.5
        forecast = result["forecast"]
        forecast["volume_quality"] = build_volume_quality(result["raw_paths"], forecast["times"])
        return result


class FakeTransport:
    def __init__(self, *, modify=None, adapter_factory=ProtocolFixtureAdapter, barrier=None):
        self.session = WorkerSession(adapter_factory=adapter_factory)
        self.modify, self.barrier = modify, barrier
        self.requests, self.closed = [], False

    def call(self, request, timeout):
        self.requests.append(copy.deepcopy(request))
        if request["kind"] == "initialize" and self.barrier is not None:
            self.barrier.wait(timeout=2)
        response = self.session.handle(request)
        if request["kind"] == "predict" and self.modify:
            return self.modify(response)
        return response

    def close(self):
        self.closed = True


def fixture_score(window, forecast, forecast_id, *, model_label):
    """Explicit fixture scorer: lets dispatch tests avoid repeated data reads."""
    return {"status": "scored", "rows": [], "sample_count": 1, "failed_count": 0,
            "pending_count": 0, "quality_status": "TEST_ONLY_fixture", "forecast_id": forecast_id}


def fake_worker_main():
    serve(sys.stdin, sys.stdout, sys.stderr, session=WorkerSession(adapter_factory=NoisyFixtureAdapter))


class DistributedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_manifest(limit=3, horizon=2, path_count=1, model_label="TEST_ONLY_fixture")

    def execute(self, factories, directory, manifest=None):
        return execute_manifest(manifest or self.manifest, transports=factories,
                                expected_identity=fixture_identity(), out_dir=directory,
                                scorer=fixture_score, task_timeout=2, init_timeout=2)

    def test_both_workers_get_whole_tasks_once_and_persist_model(self):
        barrier = threading.Barrier(2)
        left, right = FakeTransport(barrier=barrier), FakeTransport(barrier=barrier)
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute({"local": lambda _: left, "remote": lambda _: right}, directory)
            self.assertEqual(result["sample_count"], 3)
            self.assertEqual(result["failed_count"], 0)
            self.assertEqual(result["not_run_count"], 0)
            self.assertTrue(Path(result["summary_path"]).is_file())
            for item in result["artifacts"]:
                stored = json.loads(Path(item["artifact_path"]).read_text())
                self.assertEqual(stored["correction_quality_status"], "validated")
                self.assertEqual(stored["volume_quality_status"], "validated")
                self.assertEqual(stored["volume_quality"]["status"], "valid")
                self.assertEqual(stored["profile_id"], self.manifest["profile_id"])
        sent = [req for worker in (left, right) for req in worker.requests if req["kind"] == "predict"]
        self.assertEqual(len({req["task_id"] for req in sent}), 3)
        self.assertEqual(len(sent), 3)
        for worker in (left, right):
            self.assertEqual(sum(req["kind"] == "initialize" for req in worker.requests), 1)
            self.assertGreaterEqual(worker.session.adapter.calls, 1)
            self.assertTrue(worker.closed)
        self.assertTrue(all(len(req["task"]["window"]["histories"]["BTC"]) == 256 for req in sent))
        self.assertTrue(all("truth" not in req["task"] for req in sent))

    def test_request_mismatch_quarantines_worker_without_retry(self):
        def alter(response):
            response["request_id"] = "wrong"
            return response
        worker = FakeTransport(modify=alter)
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute({"local": lambda _: worker}, directory)
            self.assertEqual(result["failed_count"], 1)
            self.assertEqual(result["not_run_count"], 2)
            self.assertEqual(result["workers"]["local"]["status"], "quarantined")
            self.assertTrue(all(Path(a["artifact_path"]).is_file() for a in result["not_run_tasks"]))
        self.assertEqual(sum(req["kind"] == "predict" for req in worker.requests), 1)

    def test_coordinator_rechecks_raw_audit_and_forecast_identity(self):
        for field in ("raw", "forecast_id", "model", "volume_policy", "task_id", "prediction_run_id"):
            def alter(response, field=field):
                if field == "raw":
                    response["result"]["raw_paths"][0]["assets"]["BTC"]["close"][0] *= 2
                elif field == "model":
                    response["result"]["runtime"]["identity"]["model_revision"] = "wrong"
                elif field == "volume_policy":
                    response["result"]["runtime"]["identity"]["volume_policy"] = "wrong"
                elif field == "prediction_run_id":
                    response["result"]["forecast"]["prediction_run_id"] = "wrong"
                else:
                    response[field] = "wrong"
                return response
            worker = FakeTransport(modify=alter)
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                result = self.execute({"local": lambda _: worker}, directory)
                self.assertEqual(result["sample_count"], 0)
                self.assertEqual(result["failed_count"], 1)
                self.assertEqual(result["not_run_count"], 2)

    def test_negative_volume_is_audited_and_raw_values_are_preserved(self):
        worker = FakeTransport(adapter_factory=NegativeVolumeFixtureAdapter)
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute({"local": lambda _: worker}, directory)
            self.assertEqual(result["sample_count"], 3)
            self.assertEqual(result["failed_count"], 0)
            for item in result["artifacts"]:
                artifact = json.loads(Path(item["artifact_path"]).read_text())
                quality = artifact["volume_quality"]
                self.assertEqual(artifact["volume_quality_status"], "validated")
                self.assertEqual(quality["status"], "volume_forecast_unavailable")
                self.assertEqual(quality["total_candles"], 6)
                self.assertEqual(quality["volume_invalid_count"], 1)
                self.assertAlmostEqual(quality["volume_invalid_rate"], 1 / 6)
                raw = artifact["result"]["raw_paths"][0]["assets"]["BTC"]
                self.assertEqual((raw["volume"][0], raw["amount"][0]), (-2.5, -7.5))
                self.assertEqual(len(artifact["result"]["forecast"]["volume_quality"]["records"]), 6)
        self.assertEqual(worker.session.adapter.calls, 3)

    def test_worker_rejects_missing_or_incomplete_volume_audit(self):
        for changed in ("missing", "missing_record", "nonfinite_raw"):
            class BadVolumeFixtureAdapter(ProtocolFixtureAdapter):
                def predict(self, *args):
                    result = super().predict(*args)
                    if changed == "missing":
                        result["forecast"].pop("volume_quality")
                    elif changed == "missing_record":
                        result["forecast"]["volume_quality"]["records"].pop()
                    else:
                        result["raw_paths"][0]["assets"]["BTC"]["volume"][0] = float("nan")
                    return result
            worker = FakeTransport(adapter_factory=BadVolumeFixtureAdapter)
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                result = self.execute({"local": lambda _: worker}, directory)
                self.assertEqual(result["sample_count"], 0)
                self.assertEqual(result["failed_count"], 3)
                self.assertEqual(result["not_run_count"], 0)
                artifact = json.loads(Path(result["artifacts"][0]["artifact_path"]).read_text())
                self.assertEqual(artifact["response"]["kind"], "failure")
                self.assertEqual(artifact["error"]["stage"], "validate_output")
                self.assertNotEqual(artifact["volume_quality_status"], "validated")
                if changed == "nonfinite_raw":
                    self.assertEqual(artifact["result"]["raw_paths"][0]["assets"]["BTC"]["volume"][0], "NaN")

    def test_coordinator_rechecks_volume_audit_even_with_rehashed_snapshot(self):
        for changed in ("missing", "missing_record", "record_raw_mismatch", "false_valid"):
            def alter(response, changed=changed):
                forecast = response["result"]["forecast"]
                if changed == "missing":
                    forecast.pop("volume_quality")
                elif changed == "missing_record":
                    forecast["volume_quality"]["records"].pop()
                elif changed == "record_raw_mismatch":
                    response["result"]["raw_paths"][0]["assets"]["BTC"]["volume"][0] = -5
                else:
                    forecast["volume_quality"]["records"][0]["volume_valid"] = False
                response["forecast_id"] = freeze_forecast(forecast).identity
                return response
            worker = FakeTransport(modify=alter)
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                result = self.execute({"local": lambda _: worker}, directory)
                self.assertEqual(result["sample_count"], 0)
                self.assertEqual(result["failed_count"], 1)
                self.assertEqual(result["not_run_count"], 2)
                self.assertEqual(result["workers"]["local"]["status"], "quarantined")
                artifact = json.loads(Path(result["artifacts"][0]["artifact_path"]).read_text())
                self.assertEqual(artifact["error"]["stage"], "validate_output")
                self.assertNotEqual(artifact["volume_quality_status"], "validated")

    def test_reported_model_failure_is_retained_and_other_tasks_continue(self):
        worker = FakeTransport(adapter_factory=RejectFirstAdapter)
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute({"local": lambda _: worker}, directory)
            self.assertEqual(result["failed_count"], 1)
            self.assertEqual(result["sample_count"], 2)
            failed = json.loads(Path(result["artifacts"][0]["artifact_path"]).read_text())
            self.assertEqual(failed["raw_paths"][0]["invalid"], "NaN")
            self.assertIn("issues", failed)
        self.assertEqual(sum(req["kind"] == "predict" for req in worker.requests), 3)

    def test_initialization_failure_preserves_reserved_failure_and_unrun(self):
        def broken(_):
            raise TransportError("TEST_ONLY connection refused")
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute({"remote": broken}, directory)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["not_run_count"], 2)

    def test_mutated_manifest_rejected_before_transport(self):
        value = copy.deepcopy(self.manifest)
        value["tasks"][0]["window"]["histories"]["BTC"][0]["close"] = 1
        with self.assertRaises(ValueError):
            validate_manifest(value)

    def test_rehashed_nested_truth_and_future_input_rows_are_rejected(self):
        for changed in ("quality_truth", "candle_extra", "metadata_truth", "future_row"):
            task = copy.deepcopy(self.manifest["tasks"][0])
            if changed == "quality_truth":
                task["window"]["quality"]["truth"] = {"close": [100, 200]}
            elif changed == "candle_extra":
                task["window"]["histories"]["BTC"][0]["future_close"] = 100
            elif changed == "metadata_truth":
                task["fold"] = {"truth": [100]}
            else:
                task["window"]["histories"]["BTC"][-1]["time"] = "2026-01-01T00:00:00Z"
            task["window"]["window_id"] = "window_" + _hash({k: v for k, v in task["window"].items() if k != "window_id"})
            task["task_id"] = "task_" + _hash({k: v for k, v in task.items() if k != "task_id"})
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_task(task)

    def test_interrupt_stops_new_dispatch_and_keeps_accounting(self):
        class SlowTransport(FakeTransport):
            def call(self, request, timeout):
                if request["kind"] == "predict":
                    threading.Event().wait(.2)
                    if self.closed:
                        raise TransportError("TEST_ONLY interrupted transport")
                return super().call(request, timeout)
        worker = SlowTransport()
        timer = threading.Timer(.05, lambda: os.kill(os.getpid(), signal.SIGINT))
        with tempfile.TemporaryDirectory() as directory:
            timer.start()
            try:
                result = self.execute({"local": lambda _: worker}, directory)
            finally:
                timer.cancel()
                timer.join()
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["not_run_count"], 2)
        self.assertTrue(worker.closed)

    def test_worker_stays_resident_and_duplicate_request_never_reruns(self):
        session = WorkerSession(adapter_factory=ProtocolFixtureAdapter)
        init = {"protocol": PROTOCOL, "kind": "initialize", "request_id": "init",
                "model_spec": self.manifest["model_spec"], "expected_identity": fixture_identity()}
        self.assertEqual(session.handle(init)["kind"], "ready")
        task = self.manifest["tasks"][0]
        request = {"protocol": PROTOCOL, "kind": "predict", "request_id": "one",
                   "task_id": task["task_id"], "window_id": task["window"]["window_id"], "task": task}
        self.assertEqual(session.handle(request)["kind"], "result")
        self.assertEqual(session.handle(request)["kind"], "failure")
        self.assertEqual(session.adapter.calls, 1)
        request["request_id"] = "different-request-same-task"
        duplicate = session.handle(request)
        self.assertEqual(duplicate["kind"], "failure")
        self.assertIn("duplicate task_id", duplicate["error"]["message"])
        self.assertEqual(session.adapter.calls, 1)

    def test_real_fake_process_stdout_is_clean_and_model_is_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            transport = ProcessTransport([sys.executable, "-u", "-c",
                "from demo_app.test_distributed import fake_worker_main; fake_worker_main()"],
                cwd=Path(__file__).resolve().parents[1], log_path=log)
            result = self.execute({"local": lambda _: transport}, Path(directory) / "out")
            self.assertEqual(result["sample_count"], 3)
            self.assertEqual(result["failed_count"], 0)
            self.assertEqual(log.read_text().count("TEST_ONLY model progress"), 3)

    def test_actual_fake_process_timeout_and_stdout_pollution(self):
        scripts = [("import time; time.sleep(10)", "timed out"),
                   ("print('TEST_ONLY noise', flush=True)", "JSON protocol")]
        for script, expected in scripts:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                process = ProcessTransport([sys.executable, "-u", "-c", script],
                                           cwd=directory, log_path=Path(directory) / "log")
                try:
                    with self.assertRaisesRegex(TransportError, expected):
                        process.call({"protocol": PROTOCOL, "request_id": "test"}, .2)
                finally:
                    process.close()
                self.assertIsNotNone(process.process.poll())

    def test_ssh_command_is_strict_and_quotes_remote_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            known = Path(directory) / "My Keys" / "known_hosts"
            known.parent.mkdir()
            known.write_text("TEST_ONLY not a real host key\n")
            config = SSHConfig("user@remote.local", "/Users/example/My Project $(bad)",
                               "venv/bin/python", str(known), str(Path(directory) / "My Control" / "control-%C"))
            command = ssh_command(config)
            self.assertIn("BatchMode=yes", command)
            self.assertIn("StrictHostKeyChecking=yes", command)
            self.assertIn("ControlMaster=auto", command)
            self.assertIn("cd '/Users/example/My Project $(bad)' && exec", command[-1])
            self.assertEqual(command[-2], "user@remote.local")
            # ssh -G expands configuration without opening a network connection.
            resolved = subprocess.run([command[0], "-G", *command[1:-1]], check=True,
                                      capture_output=True, text=True, timeout=5).stdout
            known_line = next(line for line in resolved.splitlines() if line.startswith("userknownhostsfile "))
            control_line = next(line for line in resolved.splitlines() if line.startswith("controlpath "))
            self.assertIn(str(known), known_line)
            self.assertIn(str(Path(directory) / "My Control"), control_line)
            with self.assertRaises(ValueError):
                ssh_command(SSHConfig("-oProxyCommand=bad", "/tmp", "python", str(known)))


if __name__ == "__main__":
    unittest.main()
