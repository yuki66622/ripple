"""Synthetic CPU LoRA recovery tests; no market data or real checkpoints."""
from copy import deepcopy
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

from lora_a import recovery_state as recovery


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 4)
        self.out_proj = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.out_proj(torch.tanh(self.q_proj(x)))


def components():
    torch.manual_seed(123)
    model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, lora_dropout=0.1,
                          target_modules=["q_proj", "out_proj"], task_type=None))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=4e-5, betas=(0.9, 0.95), weight_decay=0.1)
    return model, optimizer


def step(model, optimizer):
    x = torch.tensor([[0.3, 0.4, 0.7], [1.2, -0.8, 0.1]])
    y = torch.tensor([[0.2, -0.1], [0.5, 0.3]])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def rng_draws():
    return random.random(), np.random.random(4), torch.rand(4)


class CompareTests(unittest.TestCase):
    def test_checkpoint_files_compare_without_full_file_tensor_load(self):
        state = {"weight": torch.ones((3, 4)), "bias": torch.zeros(3)}
        with TemporaryDirectory() as folder:
            reference = Path(folder) / "reference.safetensors"
            candidate = Path(folder) / "candidate.safetensors"
            save_file(state, str(reference))
            save_file(state, str(candidate))
            with patch("safetensors.torch.load_file", side_effect=AssertionError("full tensor load forbidden")):
                exact = recovery.compare_checkpoint_files(candidate, reference)
                self.assertTrue(exact["passed"])
                self.assertTrue(exact["exact_equal"])
                state["weight"][0, 0] += 5e-7
                save_file(state, str(candidate))
                close = recovery.compare_checkpoint_files(candidate, reference)
                self.assertTrue(close["passed"])
                self.assertFalse(close["exact_equal"])
                state["weight"][0, 0] += 1e-3
                save_file(state, str(candidate))
                failed = recovery.compare_checkpoint_files(candidate, reference)
                self.assertFalse(failed["passed"])
                del state["bias"]
                save_file(state, str(candidate))
                self.assertFalse(recovery.compare_checkpoint_files(candidate, reference)["keys_match"])

    def test_exact_near_and_failed_gate_do_not_modify_model(self):
        model, _ = components()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "adapter_model.safetensors"
            save_file(recovery._adapter(model), str(path))
            baseline = recovery.compare_epoch1(model, folder)
            self.assertTrue(baseline["passed"])
            self.assertTrue(baseline["exact_equal"])
            self.assertEqual(baseline["atol"], 1e-8)
            self.assertEqual(baseline["max_abs"], 0)
            target = next(p for name, p in model.named_parameters() if "lora_B" in name)
            with torch.no_grad():
                target.view(-1)[0] = 5e-9
            before = target.clone()
            close = recovery.compare_epoch1(model, path)
            self.assertTrue(close["passed"])
            self.assertFalse(close["exact_equal"])
            self.assertTrue(torch.equal(target, before))
            with torch.no_grad():
                target.view(-1)[0] = 1e-3
            failed = recovery.compare_epoch1(model, path)
            self.assertFalse(failed["passed"])
            self.assertGreater(failed["max_abs"], 1e-4)

    def test_missing_shape_dtype_and_nonfinite_fail_closed(self):
        model, _ = components()
        state = recovery._adapter(model)
        key = next(iter(state))
        variants = []
        missing = deepcopy(state)
        del missing[key]
        variants.append(missing)
        shape = deepcopy(state)
        shape[key] = shape[key].reshape(-1)
        variants.append(shape)
        dtype = deepcopy(state)
        dtype[key] = dtype[key].double()
        variants.append(dtype)
        for value in (float("nan"), float("inf")):
            bad = deepcopy(state)
            bad[key].view(-1)[0] = value
            variants.append(bad)
        with TemporaryDirectory() as folder:
            for index, variant in enumerate(variants):
                path = Path(folder) / f"variant-{index}.safetensors"
                save_file(variant, str(path))
                report = recovery.compare_epoch1(model, path)
                self.assertFalse(report["passed"])
                json.dumps(report, allow_nan=False)


class StateTests(unittest.TestCase):
    def assert_optimizer_equal(self, actual, expected):
        self.assertEqual(actual["param_groups"], expected["param_groups"])
        self.assertEqual(set(actual["state"]), set(expected["state"]))
        for key, fields in actual["state"].items():
            for name, value in fields.items():
                self.assertTrue(torch.equal(value, expected["state"][key][name]))

    def test_complete_restore_reproduces_rng_and_next_update(self):
        model, optimizer = components()
        random.seed(71)
        np.random.seed(81)
        torch.use_deterministic_algorithms(True)
        step(model, optimizer)
        before = recovery._adapter(model)
        optimizer_before = deepcopy(optimizer.state_dict())
        with TemporaryDirectory() as folder:
            path = Path(folder) / "state"
            saved = recovery.save_training_state(model, optimizer, {"seed": 123}, 1, path,
                                                  {"examples": 2}, {"passed": True})
            self.assertTrue((path / "COMPLETE").is_file())
            self.assert_optimizer_equal(optimizer.state_dict(), optimizer_before)
            self.assertTrue(recovery._compare(recovery._adapter(model), before, 0, 0)["exact_equal"])
            expected_draws = rng_draws()
            expected_loss = step(model, optimizer)
            expected_adapter = recovery._adapter(model)
            expected_optimizer = deepcopy(optimizer.state_dict())
            other, other_optimizer = components()
            parameter_ids = [id(p) for p in other.parameters()]
            restored = recovery.load_training_state(other, other_optimizer, path, config={"seed": 123})
            self.assertEqual(parameter_ids, [id(p) for p in other.parameters()])
            self.assertEqual(saved["manifest_sha256"], restored["manifest_sha256"])
            self.assertTrue(restored["rng_restored"])
            actual_draws = rng_draws()
            self.assertEqual(actual_draws[0], expected_draws[0])
            np.testing.assert_array_equal(actual_draws[1], expected_draws[1])
            self.assertTrue(torch.equal(actual_draws[2], expected_draws[2]))
            self.assertEqual(step(other, other_optimizer), expected_loss)
            self.assertTrue(recovery._compare(recovery._adapter(other), expected_adapter, 0, 0)["exact_equal"])
            self.assert_optimizer_equal(other_optimizer.state_dict(), expected_optimizer)
            with self.assertRaises(FileExistsError):
                recovery.save_training_state(model, optimizer, {}, 1, path, {}, {})

    def test_bad_identity_order_and_hash_fail_before_restoration(self):
        model, optimizer = components()
        step(model, optimizer)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "state"
            recovery.save_training_state(model, optimizer, {"seed": 123}, 1, path, {}, {})
            other, other_optimizer = components()
            before = recovery._adapter(other)
            with self.assertRaisesRegex(ValueError, "configuration"):
                recovery.load_training_state(other, other_optimizer, path, config={"seed": 999})
            other_optimizer.param_groups[0]["params"].reverse()
            with self.assertRaisesRegex(ValueError, "order"):
                recovery.load_training_state(other, other_optimizer, path)
            other_optimizer.param_groups[0]["params"].reverse()
            frozen = next(p for p in other.parameters() if not p.requires_grad)
            with torch.no_grad():
                frozen.view(-1)[0] += 1
            with self.assertRaisesRegex(ValueError, "frozen base"):
                recovery.load_training_state(other, other_optimizer, path)
            with (path / "optimizer-rng.pt").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                recovery.load_training_state(other, other_optimizer, path)
            self.assertTrue(recovery._compare(recovery._adapter(other), before, 0, 0)["exact_equal"])
            self.assertFalse(other_optimizer.state)

    def test_uncommitted_and_missing_optimizer_state_rejected(self):
        model, optimizer = components()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "state"
            with self.assertRaisesRegex(ValueError, "complete"):
                recovery.save_training_state(model, optimizer, {}, 1, path, {}, {})
            self.assertFalse(path.exists())
            path.mkdir()
            with self.assertRaises(FileNotFoundError):
                recovery.load_training_state(model, optimizer, path)

    def test_mps_rng_is_saved_and_restored_without_gpu_execution(self):
        model, optimizer = components()
        step(model, optimizer)
        fake_mps_rng = torch.tensor([1, 3, 7], dtype=torch.uint8)
        with TemporaryDirectory() as folder, \
                patch("torch.backends.mps.is_available", return_value=True), \
                patch("torch.mps.get_rng_state", return_value=fake_mps_rng), \
                patch("torch.mps.set_rng_state") as restore_mps:
            path = Path(folder) / "state"
            recovery.save_training_state(model, optimizer, {}, 1, path, {}, {})
            other, other_optimizer = components()
            recovery.load_training_state(other, other_optimizer, path)
            self.assertTrue(torch.equal(restore_mps.call_args.args[0], fake_mps_rng))


if __name__ == "__main__":
    unittest.main()
