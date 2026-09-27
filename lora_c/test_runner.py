"""Pure mocked C control flow: no models, GPUs, downloads or market files."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import json
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from lora_c import metrics, runner, supervise


_REPORT_CACHE = {}


def report_fixture(methods, original, candidate=None):
    """Run real C scoring/summarization over synthetic per-origin evidence."""
    from lora_c.test_metrics import fixture

    candidate = candidate or {}
    cache_key = (tuple(methods), tuple(sorted(original.items())), tuple(sorted(candidate.items())))
    if cache_key in _REPORT_CACHE:
        return deepcopy(_REPORT_CACHE[cache_key])
    window, future, selected, baseline = fixture()
    predictions = {"original": baseline}
    for name in methods:
        if name != "original":
            predictions[name] = selected
    template = metrics.score_window(window, future, predictions)
    rows = []
    anchor = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)
    for index in range(300):
        row = deepcopy(template)
        timestamp = anchor + timedelta(minutes=index)
        row.update(window_id=f"synthetic-control-{index}", as_of=timestamp.isoformat(),
                   utc_day=timestamp.date().isoformat())
        for tier, assets in metrics.TIERS.items():
            for name in methods:
                value = original[tier] if name == "original" else candidate.get(tier, 0.8)
                if (tier == "small" and name != "original"
                        and index >= candidate.get("small_mdd_count", 300)):
                    value = None
                item = row["tiers"][tier]["methods"][name]
                item["max_drawdown_mae"] = value
                item["max_drawdown_absolute_errors"] = {asset: value for asset in assets}
        rows.append(row)
    report = metrics.summarize(rows, model_names=methods)
    _REPORT_CACHE[cache_key] = report
    return deepcopy(report)


def run_scenario(candidates, original=None):
    original = original or {"small": 1.0, "major": 1.0}
    trace, snapshots = [], {}
    modules = {name: ModuleType(name) for name in
               ("lora_c.data", "lora_c.training", "lora_c.reporting", "lora_a.recovery_state")}

    def origins(month, role):
        return list(range(300)) if role != "train" else [255]

    def load_month(month, role):
        trace.append({"kind": "load_data", "month": month})
        return SimpleNamespace(origins=origins(month, role), times=["synthetic"] * 400,
                               provenance={"role": role}, month=month)

    def fresh(*args, **kwargs):
        identifier = sum(row["kind"] == "fresh" for row in trace) + 1
        trace.append({"kind": "fresh", "id": identifier, "device": kwargs["device"]})
        return SimpleNamespace(id=identifier), object(), SimpleNamespace(steps=0)

    def train(model, tokenizer, optimizer, data, config, epoch, **kwargs):
        trace.append({"kind": "train", "epoch": epoch, "model_id": model.id, "initial_steps": optimizer.steps})
        optimizer.steps += 373
        return {"epoch": epoch, "elapsed_seconds": 1, "examples": 2980, "batches": 373}

    def checkpoint(model, path):
        path.mkdir()
        name = "adapter_model.safetensors" if path.name == "adapter" else "model.safetensors"
        (path / name).write_bytes(b"SYNTHETIC MOCK ARTIFACT; NOT MODEL WEIGHTS")
        return {"path": str(path)}

    def save(model, optimizer, config, epoch, path, stats, reference):
        snapshots[str(path)] = optimizer.steps
        path.mkdir()
        trace.append({"kind": "save", "epoch": epoch, "steps": optimizer.steps})
        return {"manifest_sha256": "synthetic"}

    def restore(model, optimizer, path, **kwargs):
        optimizer.steps = snapshots[str(path)]
        trace.append({"kind": "restore", "model_id": model.id, "steps": optimizer.steps,
                      "restore_rng": kwargs["restore_rng"]})
        return {"manifest_sha256": "synthetic", "rng_restored": kwargs["restore_rng"]}

    def predict(data, config, model, name, out, journal, run):
        trace.append({"kind": "predict", "method": name, "month": data.month})
        return {"elapsed_seconds": 1}

    def score(data, config, out, methods, journal):
        candidate = candidates[int(methods[-1].split("_")[-1]) - 1] if len(methods) > 1 else None
        return report_fixture(methods, original, candidate)

    def log(event=None, **fields):
        trace.append({"kind": "journal", "event": event, **fields})

    modules["lora_c.data"].calendar_origins = origins
    modules["lora_c.data"].load_month = load_month
    modules["lora_c.training"].load_train_components = fresh
    modules["lora_c.training"].smoke_check = lambda *args, **kwargs: {"frozen_parameters_unchanged": True, "reload_logits_max_abs_diff": 0}
    modules["lora_c.training"].train_epoch = train
    modules["lora_c.training"].save_adapter = checkpoint
    modules["lora_c.training"].export_merged = checkpoint
    modules["lora_a.recovery_state"].save_training_state = save
    modules["lora_a.recovery_state"].load_training_state = restore
    modules["lora_c.reporting"].markdown_report = lambda *args: None
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = deepcopy(json.loads(runner.CONFIG.read_text()))
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        run = root / "run"
        with patch.dict(sys.modules, modules), patch.object(runner, "CONFIG", config_path), \
                patch.object(runner, "freeze_environment", side_effect=lambda directory:
                             (directory / "environment.json").write_text('{"synthetic_only": true}')), \
                patch.object(runner, "verify_initial_identities"), \
                patch.object(runner, "ensure_time"), patch.object(runner, "clear_inference"), \
                patch.object(runner, "predict_month", side_effect=predict), \
                patch.object(runner, "score_all", side_effect=score), patch.object(runner, "Journal", return_value=log):
            runner.execute(run)
        artifacts = {name: json.loads((run / name).read_text()) for name in
                     ("selection-lock.json", "selection-history.json", "interim.json", "failure.json") if (run / name).exists()}
    return trace, artifacts


class RunnerControlTests(unittest.TestCase):
    def assert_stopped_before_second_epoch(self, trace, artifacts):
        self.assertEqual([row["epoch"] for row in trace if row["kind"] == "train"], [1])
        self.assertNotIn("selection-lock.json", artifacts)
        self.assertIn("interim.json", artifacts)

    def test_original_null_stops_before_candidate_prediction(self):
        for tier in ("small", "major"):
            with self.subTest(tier=tier):
                original = {"small": 1.0, "major": 1.0, tier: None}
                trace, artifacts = run_scenario([{"small": 0.8}, {"small": 0.7}], original)
                self.assert_stopped_before_second_epoch(trace, artifacts)
                self.assertEqual([row["method"] for row in trace if row["kind"] == "predict"], ["original"])

    def test_candidate_null_stops_before_second_epoch(self):
        for tier in ("small", "major"):
            with self.subTest(tier=tier):
                candidate = {"small": 0.8, "major": 1.0, tier: None}
                trace, artifacts = run_scenario([candidate, {"small": 0.7}])
                self.assert_stopped_before_second_epoch(trace, artifacts)

    def test_finite_mean_with_partial_small_mdd_is_not_selection_evidence(self):
        candidate = {"small": 0.8, "major": 1.0, "small_mdd_count": 299}
        report = report_fixture(["original", "epoch_01"], {"small": 1.0, "major": 1.0}, candidate)
        self.assertFalse(metrics.selection_gate(report, "epoch_01")["passed"])
        trace, artifacts = run_scenario([candidate, {"small": 0.7}])
        self.assert_stopped_before_second_epoch(trace, artifacts)

    def test_major_guard_stops_on_first_or_second_epoch(self):
        trace, artifacts = run_scenario([{"small": 0.8, "major": 1.21}, {"small": 0.7}])
        self.assert_stopped_before_second_epoch(trace, artifacts)
        trace, artifacts = run_scenario([{"small": 0.8, "major": 1.0}, {"small": 0.7, "major": 1.21}])
        self.assertNotIn("selection-lock.json", artifacts)
        self.assertIn("interim.json", artifacts)

    def test_tie_selects_earlier_epoch_and_exact_twenty_percent_does_not_stop(self):
        trace, artifacts = run_scenario([{"small": 0.8, "major": 1.2}, {"small": 0.8, "major": 1.0}])
        self.assertEqual(artifacts["selection-lock.json"]["epoch"], 1)
        self.assertTrue(any(row.get("event") == "early_stop" for row in trace))

    def test_second_epoch_uses_fresh_components_and_restored_optimizer_rng(self):
        trace, artifacts = run_scenario([{"small": 0.8, "major": 1.0}, {"small": 0.7, "major": 1.0}])
        self.assertEqual(artifacts["selection-lock.json"]["epoch"], 2)
        trained = [row for row in trace if row["kind"] == "train"]
        self.assertNotEqual(trained[0]["model_id"], trained[1]["model_id"])
        self.assertEqual([row["initial_steps"] for row in trained], [0, 373])
        restored = [row for row in trace if row["kind"] == "restore"]
        self.assertEqual(len(restored), 1)
        self.assertTrue(restored[0]["restore_rng"])
        self.assertEqual([row["steps"] for row in trace if row["kind"] == "save"], [373, 746])
        self.assertEqual({row["month"] for row in trace if row["kind"] == "load_data"}, {"2026-01", "2026-04"})

    def test_complete_but_not_improved_first_epoch_may_continue(self):
        trace, artifacts = run_scenario([{"small": 1.1, "major": 1.0}, {"small": 1.05, "major": 1.0}])
        self.assertEqual([row["epoch"] for row in trace if row["kind"] == "train"], [1, 2])
        self.assertNotIn("selection-lock.json", artifacts)
        self.assertIn("interim.json", artifacts)


class SupervisorControlTests(unittest.TestCase):
    def test_expired_deadline_terminates_then_kills_without_retry(self):
        worker = MagicMock(pid=12345, returncode=None)
        worker.poll.return_value = None
        def wait(timeout):
            if timeout == 30:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            worker.returncode = -9
            return -9
        worker.wait.side_effect = wait
        assertion = MagicMock(pid=23456)
        assertion.poll.return_value = None
        with TemporaryDirectory() as temporary:
            control = Path(temporary)
            (control / "launch.json").write_text(json.dumps({"command": ["NEVER EXECUTE"], "cwd": temporary,
                                                            "deadline_utc": "2000-01-01T00:00:00+00:00"}))
            with patch.object(supervise.signal, "signal"), patch.object(supervise.os, "getpgid", return_value=123), \
                    patch.object(supervise.subprocess, "Popen", side_effect=[worker, assertion]) as launch, \
                    patch.object(supervise.subprocess, "run", side_effect=AssertionError("no telemetry after deadline")):
                with self.assertRaisesRegex(TimeoutError, "deadline"):
                    supervise.serve(control)
            self.assertEqual(launch.call_count, 2)  # One worker and one mocked sleep assertion.
            worker.terminate.assert_called_once()
            worker.kill.assert_called_once()
            assertion.terminate.assert_called_once()
            receipt = json.loads((control / "exit.json").read_text())
            self.assertFalse(receipt["automatic_restart"])
            self.assertEqual(receipt["returncode"], -9)


if __name__ == "__main__":
    unittest.main()
