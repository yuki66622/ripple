"""Fail-closed recovery orchestration tests; no market data or real models."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lora_a import recover, runner, training, recovery_state


class CrashProofTests(unittest.TestCase):
    def test_missing_proof_and_changed_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            evidence = p / "evidence.json"
            evidence.write_text("{}")
            proof = p / "proof.json"
            value = {"cause_identified": True, "memory_and_sleep_reviewed": True,
                     "independent_process_test_passed": True, "evidence": [runner.reference(evidence)]}
            proof.write_text(json.dumps(value))
            recover.require_crash_fix(proof)
            evidence.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "evidence changed"):
                recover.require_crash_fix(proof)
            value["memory_and_sleep_reviewed"] = False
            proof.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "must pass"):
                recover.require_crash_fix(proof)


class ReplayGatesTests(unittest.TestCase):
    def scenario(self, adapter_pass, merged_pass, epoch2_first=False):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); ref = base / "reference"; out = base / "new"
            adapter = ref / "epoch_01/adapter/adapter_model.safetensors"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b"test-only")
            stats = {"epoch": 1, "examples": 8, "expected_examples": 8, "batches": 1,
                     "shuffle_seed": 7, "pair_order_sha256": "synthetic-order"}
            payloads = {"config.json": {}, "fixed-pair-selection.json": {"origins": [255]},
                        "january-baselines.json": {"data_provenance": {}},
                        "epoch_01/training-stats.json": stats,
                        "epoch_01/checkpoint.json": {"adapter": {"files": {adapter.name:
                                   {"sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()}}}}}
            for name, value in payloads.items():
                (ref / name).write_text(json.dumps(value))

            def fake_export(_model, path):
                path.mkdir(parents=True)
                (path / "model.safetensors").write_bytes(b"synthetic")
                return {}

            def fake_execute(directory):
                directory.mkdir(); (directory / "epoch_01").mkdir()
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    runner.predict_month()
                training.train_epoch(None, None, None, None, {}, 2 if epoch2_first else 1)
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    runner.predict_month()
                training.export_merged(None, directory / "epoch_01/merged")
                runner.predict_month()

            with patch.object(runner, "verify_initial_identities"), patch.object(recover, "require_crash_fix"), \
                 patch.object(runner, "ROOT", base), patch.object(runner, "execute", side_effect=fake_execute), \
                 patch.object(training, "train_epoch", return_value=stats) as train, \
                 patch.object(training, "export_merged", side_effect=fake_export), \
                 patch.object(recovery_state, "compare_epoch1", return_value={"passed": adapter_pass}), \
                 patch.object(recovery_state, "compare_checkpoint_files", return_value={"passed": merged_pass}), \
                 patch.object(recovery_state, "save_training_state") as save, \
                 patch.object(runner, "predict_month") as predict:
                if adapter_pass and merged_pass and not epoch2_first:
                    recover.run(ref, out, base / "proof.json")
                    self.assertEqual(predict.call_count, 1)
                    self.assertEqual(save.call_count, 1)
                else:
                    with self.assertRaises(RuntimeError):
                        recover.run(ref, out, base / "proof.json")
                    predict.assert_not_called()
                    if not adapter_pass or epoch2_first:
                        save.assert_not_called()
                    if epoch2_first:
                        train.assert_not_called()

    def test_adapter_mismatch_blocks_all_new_inference(self):
        self.scenario(False, True)

    def test_merged_mismatch_blocks_all_new_inference(self):
        self.scenario(True, False)

    def test_epoch2_cannot_bypass_gate(self):
        self.scenario(True, True, epoch2_first=True)

    def test_both_gates_required_before_inference(self):
        self.scenario(True, True)


if __name__ == "__main__":
    unittest.main()
