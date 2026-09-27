"""Opt-in recovery gates and complete, new-only LoRA training snapshots.

No model, data, GPU job, or recovery action runs on import. A replay comparison
reports evidence only; its caller must stop when ``passed`` is false.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported training-state value: {type(value).__name__}")


def _adapter(model):
    from peft import get_peft_model_state_dict

    return _cpu(get_peft_model_state_dict(model, adapter_name="default"))


def _compare(actual, expected, rtol, atol):
    import torch

    rows = []
    for key in sorted(set(actual) | set(expected)):
        left, right = actual.get(key), expected.get(key)
        row = {"name": key, "present_actual": left is not None,
               "present_reference": right is not None, "passed": False,
               "allclose": False, "exact_equal": False, "max_abs": None}
        if left is not None and right is not None:
            row.update(actual_shape=list(left.shape), reference_shape=list(right.shape),
                       actual_dtype=str(left.dtype), reference_dtype=str(right.dtype),
                       shape_matches=left.shape == right.shape, dtype_matches=left.dtype == right.dtype,
                       actual_finite=bool(torch.isfinite(left).all()),
                       reference_finite=bool(torch.isfinite(right).all()))
            if (row["shape_matches"] and row["dtype_matches"]
                    and row["actual_finite"] and row["reference_finite"]):
                row["max_abs"] = float((left.double() - right.double()).abs().max()) if left.numel() else 0.0
                row["exact_equal"] = torch.equal(left, right)
                row["allclose"] = bool(torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False))
                row["passed"] = row["allclose"]
        rows.append(row)
    return _summary(rows, set(actual), set(expected), rtol, atol)


def _summary(rows, actual_keys, expected_keys, rtol, atol):
    return {"passed": bool(rows) and all(row["passed"] for row in rows),
            "rtol": rtol, "atol": atol, "equal_nan": False,
            "actual_tensor_count": len(actual_keys), "reference_tensor_count": len(expected_keys),
            "keys_match": actual_keys == expected_keys,
            "exact_equal": bool(rows) and all(row["exact_equal"] for row in rows),
            "max_abs": max((row["max_abs"] for row in rows if row["max_abs"] is not None), default=None),
            "per_tensor": rows}


def compare_checkpoint_files(candidate_safetensors, reference_safetensors, rtol=1e-6, atol=1e-8):
    """Read one pair of tensors at a time; never materialize both full models."""
    from safetensors import safe_open

    if any(not math.isfinite(v) or v < 0 for v in (rtol, atol)):
        raise ValueError("rtol and atol must be finite and nonnegative")
    candidate = Path(candidate_safetensors).resolve(strict=True)
    reference = Path(reference_safetensors).resolve(strict=True)
    rows = []
    with safe_open(str(candidate), framework="pt", device="cpu") as left, \
            safe_open(str(reference), framework="pt", device="cpu") as right:
        actual_keys, expected_keys = set(left.keys()), set(right.keys())
        for key in sorted(actual_keys | expected_keys):
            actual = {key: left.get_tensor(key)} if key in actual_keys else {}
            expected = {key: right.get_tensor(key)} if key in expected_keys else {}
            rows.extend(_compare(actual, expected, rtol, atol)["per_tensor"])
            del actual, expected
    report = _summary(rows, actual_keys, expected_keys, rtol, atol)
    return {"kind": "checkpoint-parameter-gate", **report,
            "candidate": {"path": str(candidate), "sha256": _sha(candidate), "bytes": candidate.stat().st_size},
            "reference": {"path": str(reference), "sha256": _sha(reference), "bytes": reference.stat().st_size}}


def compare_epoch1(model, reference_adapter_path, rtol=1e-6, atol=1e-8):
    """Pure gate: compare replayed LoRA against the original local safetensors.

    ``atol=1e-8`` is explicit, matching torch.allclose's default. Missing keys,
    shape/dtype differences and nonfinite values always fail, regardless of
    tolerance. No reference or model tensor is changed.
    """
    from safetensors.torch import load_file

    if any(not math.isfinite(v) or v < 0 for v in (rtol, atol)):
        raise ValueError("rtol and atol must be finite and nonnegative")
    path = Path(reference_adapter_path).resolve(strict=True)
    if path.is_dir():
        path = path / "adapter_model.safetensors"
    expected = load_file(str(path), device="cpu")
    report = _compare(_adapter(model), expected, rtol, atol)
    return {"kind": "epoch1-replay-parameter-gate", **report,
            "reference": {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}}


def _layout(model, optimizer):
    import torch

    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("this recovery format requires the experiment's AdamW")
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("only LoRA parameters may be trainable")
    by_id = {id(value): name for name, value in trainable}
    ordered = [value for group in optimizer.param_groups for value in group["params"]]
    if (len(ordered) != len(by_id) or len({id(value) for value in ordered}) != len(by_id)
            or any(id(value) not in by_id for value in ordered)):
        raise ValueError("optimizer must contain every trainable LoRA parameter exactly once")
    devices = {str(value.device) for _, value in trainable}
    if len(devices) != 1:
        raise ValueError("training parameters must use one device")
    if not devices <= {"cpu", "mps", "mps:0"}:
        raise ValueError("this experiment snapshot supports CPU/MPS only")
    return {"parameters": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                           for name, value in trainable],
            "optimizer_groups": [[by_id[id(value)] for value in group["params"]]
                                 for group in optimizer.param_groups],
            "optimizer_class": "torch.optim.AdamW", "training_device": next(iter(devices))}


def _frozen_sha(model):
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _rng_state():
    import numpy as np
    import torch

    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"name": state[0], "keys": torch.from_numpy(state[1].astype(np.int64)),
                      "pos": state[2], "has_gauss": state[3], "cached_gaussian": state[4]},
            "torch_cpu": torch.get_rng_state(),
            "torch_mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}


def _restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_mps"] is not None:
        torch.mps.set_rng_state(state["torch_mps"])
    torch.use_deterministic_algorithms(state["deterministic_algorithms"], warn_only=state["deterministic_warn_only"])


def _validate_optimizer(state, model, layout):
    import torch

    groups = state["param_groups"]
    expected_groups = layout["optimizer_groups"]
    if len(groups) != len(expected_groups) or any(len(g["params"]) != len(names) for g, names in zip(groups, expected_groups)):
        raise ValueError("saved optimizer group layout differs")
    identifiers = [key for group in groups for key in group["params"]]
    if len(set(identifiers)) != len(identifiers) or set(state["state"]) != set(identifiers):
        raise ValueError("saved optimizer does not have complete per-parameter state")
    parameters = dict(model.named_parameters())
    for group, names in zip(groups, expected_groups):
        for identifier, name in zip(group["params"], names):
            entry, parameter = state["state"][identifier], parameters[name]
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad", False):
                required.add("max_exp_avg_sq")
            if set(entry) != required:
                raise ValueError("saved AdamW state is incomplete or unsupported")
            for key, value in entry.items():
                if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
                    raise ValueError("saved optimizer tensor is invalid")
                if key == "step":
                    if value.numel() != 1 or float(value) < 1 or float(value) % 1:
                        raise ValueError("saved AdamW step is invalid")
                elif value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("saved optimizer moment shape/dtype differs")


def _json_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_training_state(model, optimizer, config, epoch, output_path, training_stats, reference_info):
    """Save a completed epoch to a NEW directory, without changing live state.

    The manifest and COMPLETE marker commit the snapshot only after all files
    have been flushed. An interrupted partial directory cannot be loaded.
    """
    import torch
    from safetensors.torch import save_file

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must identify a completed positive epoch")
    layout = _layout(model, optimizer)
    adapter = _adapter(model)
    if not _compare(adapter, adapter, 0, 0)["passed"]:
        raise ValueError("nonfinite adapter cannot be checkpointed")
    payload = _cpu({"optimizer": optimizer.state_dict(), "rng": _rng_state()})
    _validate_optimizer(payload["optimizer"], model, layout)
    manifest = {"format_version": 1, "kind": "complete-lora-training-state", "epoch": epoch,
                "config": config, "training_stats": training_stats, "reference_info": reference_info,
                "layout": layout, "frozen_model_sha256": _frozen_sha(model),
                "model_training": bool(model.training),
                "model_context": getattr(model, "_lora_a_context", None)}
    json.dumps(manifest, allow_nan=False)  # Reject invalid metadata before writing.
    path = Path(output_path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    save_file(adapter, str(path / "adapter.safetensors"))
    with (path / "adapter.safetensors").open("rb") as stream:
        os.fsync(stream.fileno())
    with (path / "optimizer-rng.pt").open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest["files"] = {name: {"sha256": _sha(path / name), "bytes": (path / name).stat().st_size}
                         for name in ("adapter.safetensors", "optimizer-rng.pt")}
    _json_new(path / "manifest.json", manifest)
    manifest_sha = _sha(path / "manifest.json")
    _json_new(path / "COMPLETE", {"manifest_sha256": manifest_sha})
    return {**manifest, "path": str(path), "manifest_sha256": manifest_sha}


def load_training_state(model, optimizer, input_path, config=None, restore_rng=True):
    """Restore into fresh matching trainable components; reject identities first.

    Uses torch.load(weights_only=True), never unrestricted pickle. Parameter
    object identities are preserved, so optimizer references remain valid.
    RNG restoration requires the original device and MPS availability.
    """
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(input_path).resolve(strict=True)
    commit = json.loads((path / "COMPLETE").read_text())
    manifest_sha = _sha(path / "manifest.json")
    if commit != {"manifest_sha256": manifest_sha}:
        raise ValueError("snapshot completion hash differs")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format_version"] != 1 or manifest["kind"] != "complete-lora-training-state":
        raise ValueError("unsupported training snapshot format")
    if set(manifest["files"]) != {"adapter.safetensors", "optimizer-rng.pt"}:
        raise ValueError("snapshot file set differs")
    for filename, info in manifest["files"].items():
        saved = path / filename
        if saved.stat().st_size != info["bytes"] or _sha(saved) != info["sha256"]:
            raise ValueError("snapshot file checksum differs: " + filename)
    layout = _layout(model, optimizer)
    if layout != manifest["layout"]:
        raise ValueError("trainable parameter order, optimizer grouping, dtype, shape or device differs")
    if config is not None and config != manifest["config"]:
        raise ValueError("frozen training configuration differs")
    if _frozen_sha(model) != manifest["frozen_model_sha256"]:
        raise ValueError("frozen base parameters differ")
    adapter = load_file(str(path / "adapter.safetensors"), device="cpu")
    actual = _adapter(model)
    if set(actual) != set(adapter) or any(actual[k].shape != v.shape or actual[k].dtype != v.dtype
                                       or not bool(torch.isfinite(v).all()) for k, v in adapter.items()):
        raise ValueError("saved adapter key/shape/dtype/finite validation failed")
    payload = torch.load(path / "optimizer-rng.pt", map_location="cpu", weights_only=True)
    _validate_optimizer(payload["optimizer"], model, layout)
    if restore_rng and payload["rng"]["torch_mps"] is not None and not torch.backends.mps.is_available():
        raise ValueError("saved MPS RNG cannot be restored on this runtime")
    result = set_peft_model_state_dict(model, adapter, adapter_name="default")
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise RuntimeError("LoRA state restoration failed")
    optimizer.load_state_dict(payload["optimizer"])
    model.train(manifest["model_training"])
    if manifest["model_context"] is not None:
        model._lora_a_context = manifest["model_context"]
    if restore_rng:
        _restore_rng(payload["rng"])
    return {**manifest, "path": str(path), "manifest_sha256": manifest_sha, "rng_restored": bool(restore_rng)}
