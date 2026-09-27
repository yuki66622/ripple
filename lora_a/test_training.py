"""Synthetic CPU checks only: no downloaded weights or market files are read."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lora_a import training


def config():
    return {
        "seed": 20260926, "assets": ["FIXTURE_A", "FIXTURE_B"],
        "train_month": "2026-01", "lookback": 256, "horizon": 30, "stride": 30,
        "protocol_version": "lora-a-v2.1", "max_epochs": 2,
        "train_stride": 120, "train_start": "2026-01-01", "train_end_exclusive": "2026-01-26",
        "training_candles_per_window": 286, "batch_size": 8,
        "lr": 4e-5, "weight_decay": 0.1, "adam_betas": [0.9, 0.95],
        "gradient_clip": 3.0, "scheduler": "constant", "rank": 4,
        "alpha": 8, "dropout": 0.1, "target_modules": list(training.TARGET_MODULES),
    }


def data(n=766, assets=2):
    values = (10 + np.random.default_rng(12).random((assets, n, 6))).astype(np.float32)
    stamps = np.zeros((n, 5), dtype=np.float32)
    stamps[:, 0] = np.arange(n) % 60
    stamps[:, 1] = (np.arange(n) // 60) % 24
    stamps[:, 3:] = 1
    return SimpleNamespace(month="2026-01", assets=["FIXTURE_A", "FIXTURE_B"][:assets],
                           values=values, stamps=stamps, origins=range(255, min(n, 36000) - 30, 120),
                           role="train", provenance={"role": "train"})


def tiny_grid():
    """Only tests may substitute a small grid; runtime calendar checks stay strict."""
    return patch.object(training, "_expected_training_origins",
                        return_value=np.arange(255, 766 - 30, 120, dtype=np.int64))


MODEL_CONFIG = {
    "s1_bits": 2, "s2_bits": 2, "n_layers": 1, "d_model": 16, "n_heads": 4,
    "ff_dim": 32, "ffn_dropout_p": 0.0, "attn_dropout_p": 0.0,
    "resid_dropout_p": 0.0, "token_dropout_p": 0.0, "learn_te": True,
}
TOKENIZER_CONFIG = {
    "d_in": 6, "d_model": 16, "n_heads": 4, "ff_dim": 32, "n_enc_layers": 1,
    "n_dec_layers": 1, "ffn_dropout_p": 0.0, "attn_dropout_p": 0.0,
    "resid_dropout_p": 0.0, "s1_bits": 2, "s2_bits": 2, "beta": 0.05,
    "gamma0": 1.0, "gamma": 1.1, "zeta": 0.05, "group_size": 2,
}


def tiny_components(root):
    from safetensors.torch import save_file
    from sktime.libs.kronos import Kronos, KronosTokenizer

    cfg = config()
    for label, cls, model_config in (("model", Kronos, MODEL_CONFIG),
                                      ("tokenizer", KronosTokenizer, TOKENIZER_CONFIG)):
        folder = Path(root) / label
        folder.mkdir()
        torch.manual_seed(43)
        model = cls(**model_config)
        (folder / "config.json").write_text(json.dumps(model_config))
        save_file(model.state_dict(), str(folder / "model.safetensors"))
        cfg[f"{label}_path"] = str(folder)
    return cfg, training.load_train_components(cfg, device="cpu")


class WindowTests(unittest.TestCase):
    def test_future_never_changes_history_normalization(self):
        raw = np.arange(286 * 6, dtype=np.float32).reshape(286, 6)
        original = raw.copy()
        normalized = training.normalize_training_segment(raw)
        altered = raw.copy()
        altered[256:] = 10_000_000
        other = training.normalize_training_segment(altered)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(normalized[:256], other[:256])
        expected = (raw - raw[:256].mean(0)) / (raw[:256].std(0) + 1e-5)
        np.testing.assert_array_equal(normalized, np.clip(expected, -5, 5).astype(np.float32))
        self.assertTrue((other[256:] == 5).all())

    def test_normalization_constant_columns_shape_and_nonfinite(self):
        np.testing.assert_array_equal(training.normalize_training_segment(np.ones((286, 6))), np.zeros((286, 6)))
        with self.assertRaises(ValueError):
            training.normalize_training_segment(np.ones((285, 6)))
        raw = np.ones((286, 6))
        raw[-1, 3] = np.nan
        with self.assertRaises(ValueError):
            training.normalize_training_segment(raw)

    def test_epoch_coverage_deterministic_shuffle_and_bounds(self):
        month, cfg = data(), config()
        with tiny_grid():
            first = training.epoch_pairs(month, cfg, 1)
            second = training.epoch_pairs(month, cfg, 2)
            repeated = training.epoch_pairs(month, cfg, 1)
        expected = {(a, t) for a in range(2) for t in range(255, 736, 120)}
        self.assertEqual(set(map(tuple, first)), expected)
        self.assertEqual(len(first), len(expected))
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, second))
        features, stamps = training.make_batch(month, [(0, 255), (1, 735)], "cpu")
        self.assertEqual(tuple(features.shape), (2, 286, 6))
        np.testing.assert_array_equal(stamps[0].numpy(), month.stamps[:286])
        np.testing.assert_array_equal(stamps[1].numpy(), month.stamps[480:766])
        np.testing.assert_array_equal(features[1].numpy(), training.normalize_training_segment(month.values[1, 480:766]))
        for invalid in ((0, 254), (1, 736), (2, 255), (0, 285)):
            with self.assertRaises(ValueError):
                training.make_batch(month, [invalid], "cpu")

    def test_reject_incomplete_grid_wrong_month_and_changed_contract(self):
        month, cfg = data(), config()
        month.origins = [255, 285]
        with tiny_grid(), self.assertRaises(ValueError):
            training.epoch_pairs(month, cfg, 1)
        month = data()
        month.month = "2026-02"
        with self.assertRaises(ValueError):
            training.epoch_pairs(month, cfg, 1)
        with self.assertRaises(ValueError):
            training.epoch_pairs(data(), {**cfg, "batch_size": 4}, 1)

    def test_batch_casts_float32_before_history_normalization(self):
        month = data()
        month.raw_values = 1e8 + np.arange(month.values.size).reshape(month.values.shape) * 0.001
        month.values = month.raw_values.astype(np.float32)
        features, _ = training.make_batch(month, [(0, 255)], "cpu")
        expected = training.normalize_training_segment(month.raw_values[0, :286])
        rounded = training.normalize_training_segment(month.values[0, :286])
        np.testing.assert_array_equal(features[0].numpy(), expected)
        np.testing.assert_array_equal(expected, rounded)
        raw = month.raw_values[0, :286]
        double = np.clip((raw - raw[:256].mean(0)) / (raw[:256].std(0) + 1e-5), -5, 5).astype(np.float32)
        self.assertFalse(np.array_equal(expected, double))

    def test_real_calendar_grid_and_observation_boundary(self):
        month, cfg = data(n=44640), config()
        pairs = training.epoch_pairs(month, cfg, 1)
        self.assertEqual(len(pairs), 298 * 2)
        self.assertEqual(set(pairs[:, 1]), set(range(255, 35970, 120)))
        self.assertLess(int(pairs[:, 1].max()) + 30, 36000)
        ten_assets = [f"FIXTURE{i}" for i in range(10)]
        ten = SimpleNamespace(**{**vars(month), "assets": ten_assets,
                                 "values": np.broadcast_to(month.values[:1], (10, 44640, 6))})
        all_pairs = training.epoch_pairs(ten, {**cfg, "assets": ten_assets}, 1)
        self.assertEqual(len(all_pairs), 2980)
        self.assertEqual((len(all_pairs) + 7) // 8, 373)
        training.make_batch(month, [(0, 35895)], "cpu")
        for origin in (35999, 36015, 36135):
            with self.assertRaises(ValueError):
                training.make_batch(month, [(0, origin)], "cpu")
        month.origins = list(month.origins) + [36015]
        with self.assertRaises(ValueError):
            training.epoch_pairs(month, cfg, 1)
        with self.assertRaises(ValueError):
            training.epoch_pairs(data(n=35999), cfg, 1)
        month = data(n=36000)
        month.role = "observation"
        month.provenance["role"] = "observation"
        with self.assertRaises(ValueError):
            training.epoch_pairs(month, cfg, 1)
        with self.assertRaises(ValueError):
            training.make_batch(month, [(0, 255)], "cpu")


class TinyModelTests(unittest.TestCase):
    def test_projection_matching_includes_dependency_and_excludes_other_layers(self):
        from sktime.libs.kronos import Kronos

        model = Kronos(**MODEL_CONFIG)
        names = training.match_attention_modules(model)
        self.assertEqual(len(names), 8)
        for suffix in training.TARGET_MODULES:
            self.assertIn(f"dep_layer.cross_attn.{suffix}", names)
        model.dep_layer.cross_attn.q_proj = torch.nn.Identity()
        with self.assertRaises(ValueError):
            training.match_attention_modules(model)

    def test_one_epoch_frozen_parameters_and_standard_merged_roundtrip(self):
        from safetensors.torch import load_file
        from sktime.libs.kronos import Kronos

        with TemporaryDirectory() as directory:
            cfg, (model, tokenizer, optimizer) = tiny_components(directory)
            self.assertIsNone(model.peft_config["default"].task_type)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            token_before = {n: p.detach().clone() for n, p in tokenizer.named_parameters()}
            logged = []
            with tiny_grid():
                stats = training.train_epoch(model, tokenizer, optimizer, data(), cfg, epoch=1, log_fn=logged.append)
            self.assertEqual(stats["examples"], 10)
            self.assertEqual(stats["batches"], 2)  # Keep the final two-example batch.
            self.assertTrue(np.isfinite(stats["loss"]))
            self.assertEqual(logged[-1]["examples"], 10)
            self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.95))
            for name, param in model.named_parameters():
                if name in frozen:
                    self.assertTrue(torch.equal(frozen[name], param))
                    self.assertIsNone(param.grad)
            for name, param in tokenizer.named_parameters():
                self.assertTrue(torch.equal(token_before[name], param))
                self.assertIsNone(param.grad)
            self.assertFalse(tokenizer.training)
            ids_before = {n: id(p) for n, p in model.named_parameters()}
            state_before = {n: p.detach().clone() for n, p in model.named_parameters()}
            optimizer_before = deepcopy(optimizer.state_dict())
            rng_before = torch.get_rng_state().clone()
            merged_info = training.export_merged(model, Path(directory) / "merged")
            self.assertTrue(model.training)
            self.assertEqual(ids_before, {n: id(p) for n, p in model.named_parameters()})
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            for name, param in model.named_parameters():
                self.assertTrue(torch.equal(state_before[name], param))
            for key, state in optimizer_before["state"].items():
                for field, value in state.items():
                    self.assertTrue(torch.equal(value, optimizer.state_dict()["state"][key][field]))
            merged = Kronos(**MODEL_CONFIG)
            merged.load_state_dict(load_file(str(Path(merged_info["path"]) / "model.safetensors")), strict=True)
            model.eval()
            merged.eval()
            token0 = torch.zeros((1, 5), dtype=torch.long)
            token1 = torch.ones((1, 5), dtype=torch.long)
            torch.manual_seed(62)
            with torch.no_grad():
                reference = model(token0, token1)
            torch.manual_seed(62)
            with torch.no_grad():
                recovered = merged(token0, token1)
            for a, b in zip(reference, recovered):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            with self.assertRaises(FileExistsError):
                training.export_merged(model, Path(directory) / "merged")

    def test_explicit_smoke_save_reload_raw_logits(self):
        with TemporaryDirectory() as directory:
            cfg, components = tiny_components(directory)
            with tiny_grid():
                evidence = training.smoke_check(*components, data(), cfg, Path(directory) / "smoke")
            self.assertGreater(evidence["changed_lora_tensors"], 0)
            self.assertLessEqual(evidence["reload_logits_max_abs_diff"], 1e-6)
            self.assertTrue(evidence["base_files_unchanged"])
            self.assertTrue(evidence["frozen_parameters_unchanged"])
            self.assertEqual(evidence["frozen_parameter_sha256_before"],
                             evidence["frozen_parameter_sha256_after"])
            self.assertTrue(evidence["reload_fresh_before_epoch1"])
            self.assertTrue((Path(evidence["adapter"]["path"]) / "adapter_model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
