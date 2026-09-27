"""Authorized crash replay, using the frozen runner unchanged behind two gates.

The original run is immutable. No inference/financial scoring is permitted before
the epoch-one comparison passes. A failed replay is never silently retried.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal

from . import runner


def require_crash_fix(path):
    proof = json.loads(path.read_text())
    required = ("cause_identified", "memory_and_sleep_reviewed", "independent_process_test_passed")
    if any(proof.get(field) is not True for field in required):
        raise RuntimeError("crash diagnosis and execution lifetime fix must pass before recovery")
    for record in proof.get("evidence", []):
        if runner.reference(record["path"]) != record:
            raise RuntimeError("crash-fix evidence changed")
    if not proof.get("evidence"):
        raise RuntimeError("crash-fix evidence references are required")
    return proof


def run(reference_run, directory, crash_audit):
    from . import data, training
    from .recovery_state import compare_epoch1, compare_checkpoint_files, save_training_state

    runner.verify_initial_identities(reference_run)
    require_crash_fix(crash_audit)
    if (runner.ROOT / "lora_a/MARCH_UNSEALED.json").exists():
        raise RuntimeError("March has already been accessed; recovery cannot proceed")
    original_stats = json.loads((reference_run / "epoch_01/training-stats.json").read_text())
    original_config = json.loads((reference_run / "config.json").read_text())
    original_pair = json.loads((reference_run / "fixed-pair-selection.json").read_text())
    original_train_provenance = json.loads((reference_run / "january-baselines.json").read_text())["data_provenance"]
    original_checkpoint = json.loads((reference_run / "epoch_01/checkpoint.json").read_text())
    reference_adapter = reference_run / "epoch_01/adapter/adapter_model.safetensors"
    expected = original_checkpoint["adapter"]["files"][reference_adapter.name]
    if runner.digest(reference_adapter) != expected["sha256"]:
        raise RuntimeError("reference adapter changed")
    passed = False
    merged_passed = False
    original_train_epoch = training.train_epoch
    original_export = training.export_merged
    original_predict = runner.predict_month
    original_prepare = runner.prepare
    original_codes = runner.CODE_FILES
    extra = [Path(__file__).resolve(), runner.ROOT / "lora_a/recovery_state.py", runner.ROOT / "lora_a/supervise.py"]
    runner.CODE_FILES = original_codes + extra

    def prepare_recovery(out, config, journal):
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no CPU fallback")
        if config != original_config:
            raise RuntimeError("config differs from interrupted experiment")
        runner.ensure_time(config)
        train = data.load_month("2026-01", role="train")
        if train.provenance != original_train_provenance or train.origins != original_pair["origins"]:
            raise RuntimeError("January data provenance/order differs")
        # Reuse the already verified gates/baselines; do not calculate new metrics.
        inherited = ("smoke-gate.json", "january-baselines.json", "january-baselines.md",
                     "fixed-pair-selection.json", "calendar-grid-plan.json", "config.json")
        for name in inherited:
            with (out / name).open("xb") as target:
                target.write((reference_run / name).read_bytes())
        runner.write_new(out / "recovery-contract.json", {
            "kind": "authorized_crash_replay", "reference_run": str(reference_run),
            "reference_environment": runner.reference(reference_run / "environment.json"),
            "reference_epoch1_adapter": runner.reference(reference_adapter),
            "reference_epoch1_stats": runner.reference(reference_run / "epoch_01/training-stats.json"),
            "crash_fix": runner.reference(crash_audit), "rtol": 1e-6, "atol": 1e-8,
            "atol_policy": "explicit ordinary torch.allclose default, frozen before comparison",
            "inheritance": {name: runner.reference(reference_run / name) for name in inherited},
            "february_policy": "fresh full frozen300 only after passing replay gate; old125 retained as crash evidence",
            "training_params_changed": False, "march_accessed": False})
        runner.freeze_environment(out)
        journal("recovery_preflight_complete", original_run=str(reference_run),
                new_financial_metrics_computed=False, inherited_smoke_and_baselines=True)
        return train, config

    def guarded_train(model, tokenizer, optimizer, train, config, epoch, log_fn=None):
        nonlocal passed
        if epoch != 1 and not (passed and merged_passed):
            raise RuntimeError("epoch2 forbidden before passing epoch1 replay gate")
        runner.verify_initial_identities(reference_run)
        stats = original_train_epoch(model, tokenizer, optimizer, train, config, epoch, log_fn=log_fn)
        if epoch == 1:
            comparison = compare_epoch1(model, reference_adapter, rtol=1e-6, atol=1e-8)
            fields = ("epoch", "examples", "expected_examples", "batches", "shuffle_seed", "pair_order_sha256")
            comparison["same_training_sequence"] = all(stats[field] == original_stats[field] for field in fields)
            comparison["original_training_identity"] = {key: original_stats[key] for key in fields}
            comparison["replay_training_identity"] = {key: stats[key] for key in fields}
            comparison["passed"] = bool(comparison["passed"] and comparison["same_training_sequence"])
            comparison["checked_at"] = runner.now()
            runner.write_new(directory / "epoch1-replay-comparison.json", comparison)
            if not comparison["passed"]:
                runner.write_interim(directory, "第1轮重放权重或训练顺序未通过冻结allclose一致性门；未进行02新推理、未进入第2轮，03继续密封。")
                raise RuntimeError("epoch1 replay failed allclose/sequence gate; stop without new financial metrics")
            passed = True
            if log_fn:
                log_fn("epoch1_replay_gate_passed", comparison=str(directory / "epoch1-replay-comparison.json"),
                       rtol=1e-6, atol=1e-8, new_financial_metrics_computed=False)
        save_training_state(model, optimizer, config, epoch,
                                    directory / f"epoch_{epoch:02d}" / "training-state", stats,
                                    {"reference_run": str(reference_run), "gate": runner.reference(directory / "epoch1-replay-comparison.json")})
        if log_fn:
            log_fn("complete_training_state_saved", epoch=epoch, path=str(directory / f"epoch_{epoch:02d}" / "training-state"))
        return stats

    def guarded_export(model, path):
        nonlocal merged_passed
        result = original_export(model, path)
        if Path(path).parent.name == "epoch_01":
            comparison = compare_checkpoint_files(Path(path) / "model.safetensors",
                    reference_run / "epoch_01/merged/model.safetensors", rtol=1e-6, atol=1e-8)
            comparison["checked_at"] = runner.now()
            runner.write_new(directory / "epoch1-merged-comparison.json", comparison)
            if not comparison["passed"]:
                runner.write_interim(directory, "第1轮合并权重未通过allclose门；未进行02新推理、未进入第2轮，03继续密封。")
                raise RuntimeError("epoch1 merged checkpoint comparison failed")
            merged_passed = True
            runner.Journal(directory)("epoch1_merged_gate_passed", rtol=1e-6, atol=1e-8,
                                      new_financial_metrics_computed=False)
        return result

    def guarded_predict(*args, **kwargs):
        if not (passed and merged_passed):
            raise RuntimeError("new inference forbidden before epoch1 replay comparison")
        require_crash_fix(crash_audit)
        return original_predict(*args, **kwargs)

    def signal_handler(number, _frame):
        if directory.exists():
            runner.Journal(directory)("termination_signal_received", signal=number)
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    training.train_epoch = guarded_train
    training.export_merged = guarded_export
    runner.prepare = prepare_recovery
    runner.predict_month = guarded_predict
    try:
        runner.execute(directory)
    finally:
        training.train_epoch = original_train_epoch
        training.export_merged = original_export
        runner.prepare = original_prepare
        runner.predict_month = original_predict
        runner.CODE_FILES = original_codes


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--crash-audit", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_run.resolve(), args.run_dir.resolve(), args.crash_audit.resolve())
