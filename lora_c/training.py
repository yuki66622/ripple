"""Isolated C-line Kronos LoRA training; importing this module runs no jobs.

Only the caller opens market data and decides when to train. Checkpoints are
strict local safetensors, and exports always use new directories. Training uses
the official two-level shifted-token objective, not a financial metric.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj")
LOOKBACK, HORIZON, STRIDE = 256, 30, 30
TRAIN_STRIDE = 120
TRAIN_MINUTES = 25 * 24 * 60
SEGMENT_LENGTH = LOOKBACK + HORIZON


def _validate_config(config):
    frozen = {
        "lookback": LOOKBACK, "horizon": HORIZON, "stride": STRIDE,
        "training_candles_per_window": SEGMENT_LENGTH, "batch_size": 8,
        "lr": 4e-5, "weight_decay": 0.1, "gradient_clip": 3.0,
        "scheduler": "constant", "rank": 4, "alpha": 8, "dropout": 0.1,
        "adam_betas": [0.9, 0.95],
        "protocol_version": "lora-c-v1", "max_epochs": 2,
        "train_stride": TRAIN_STRIDE, "train_start": "2026-01-01",
        "train_end_exclusive": "2026-01-26",
    }
    for key, expected in frozen.items():
        if config.get(key) != expected:
            raise ValueError(f"frozen training setting {key} must be {expected!r}")
    if tuple(config.get("target_modules", ())) != TARGET_MODULES:
        raise ValueError("target_modules must match the frozen attention projections")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if config.get("train_month") != "2026-01":
        raise ValueError("training is authorized only for 2026-01")


def normalize_training_segment(segment):
    """Return float32 features; future candles never influence normalization.

    Uses population standard deviation, matching official Kronos preprocessing.
    The input is never modified. Exactly 256 history + 30 future rows are needed.
    """
    # Installed KronosPredictor casts before computing history mean/std.
    values = np.asarray(segment, dtype=np.float32)
    if values.shape != (SEGMENT_LENGTH, 6):
        raise ValueError(f"training segment must have shape ({SEGMENT_LENGTH}, 6)")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite training candle")
    history = values[:LOOKBACK]
    normalized = (values - history.mean(axis=0)) / (history.std(axis=0) + 1e-5)
    result = np.clip(normalized, -5.0, 5.0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite normalized training candle")
    return result


def _assert_training_role(data):
    if getattr(data, "month", None) != "2026-01":
        raise ValueError("training requires January data")
    provenance_role = getattr(data, "provenance", {}).get("role")
    role = getattr(data, "role", provenance_role)
    if role != "train" or provenance_role not in (None, "train"):
        raise ValueError("training requires the train role, never observation data")


def _expected_training_origins(n_time):
    """Fixed calendar boundary, independent of the full January array length."""
    if n_time < TRAIN_MINUTES:
        raise ValueError("January training data must cover all of January 1–25")
    return np.arange(LOOKBACK - 1, TRAIN_MINUTES - HORIZON, TRAIN_STRIDE, dtype=np.int64)


def epoch_pairs(data, config, epoch):
    """Each (asset index, origin index) exactly once; seed is seed + epoch."""
    _validate_config(config)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch <= config["max_epochs"]:
        raise ValueError("epoch must be 0 (smoke), 1 or 2")
    values, stamps = np.asarray(data.values), np.asarray(data.stamps)
    if values.ndim != 3 or values.shape[2] != 6:
        raise ValueError("MonthData.values must have shape (assets, time, 6)")
    if stamps.shape != (values.shape[1], 5):
        raise ValueError("MonthData.stamps must have shape (time, 5)")
    assets = tuple(data.assets)
    if len(assets) != values.shape[0] or not assets or len(set(assets)) != len(assets):
        raise ValueError("MonthData assets must uniquely match values")
    if tuple(config.get("assets", ())) != assets:
        raise ValueError("MonthData asset order differs from frozen configuration")
    _assert_training_role(data)
    expected = _expected_training_origins(values.shape[1])
    origins = np.asarray(list(data.origins))
    if origins.dtype.kind not in "iu" or not np.array_equal(origins, expected):
        raise ValueError("origins must cover the complete January 1–25 stride-120 grid")
    if not len(expected):
        raise ValueError("no complete training windows")
    pairs = np.array([(a, int(t)) for a in range(len(assets)) for t in expected], dtype=np.int64)
    return pairs[np.random.default_rng(config["seed"] + epoch).permutation(len(pairs))]


def make_batch(data, pairs, device):
    """Materialize only the requested in-memory January segments."""
    import torch

    _assert_training_role(data)
    features, stamps = [], []
    raw_values = getattr(data, "raw_values", None)
    values = np.asarray(data.values if raw_values is None else raw_values)
    if values.shape != np.asarray(data.values).shape:
        raise ValueError("raw_values must match MonthData.values shape")
    n_time = values.shape[1]
    for a, t in pairs:
        a, t = int(a), int(t)
        if (not 0 <= a < len(data.assets) or t < LOOKBACK - 1
                or (t - (LOOKBACK - 1)) % TRAIN_STRIDE
                or t + HORIZON >= min(n_time, TRAIN_MINUTES)):
            raise ValueError("training pair is out of bounds")
        start, stop = t - LOOKBACK + 1, t + HORIZON + 1
        features.append(normalize_training_segment(values[a, start:stop]))
        stamp = np.asarray(data.stamps[start:stop])
        if stamp.shape != (SEGMENT_LENGTH, 5) or not np.isfinite(stamp).all():
            raise ValueError("invalid training time features")
        stamps.append(stamp)
    if not features:
        raise ValueError("cannot materialize an empty batch")
    return (torch.as_tensor(np.stack(features), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(stamps), dtype=torch.float32, device=device))


def match_attention_modules(model):
    """Validate complete predictor attention coverage before PEFT injection."""
    import torch

    matches = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in TARGET_MODULES:
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"LoRA target {name} is not Linear")
            if not (name.startswith("transformer.") or name.startswith("dep_layer.cross_attn.")):
                raise ValueError(f"unexpected projection outside predictor attention: {name}")
            matches.append(name)
    groups = {}
    for name in matches:
        parent, suffix = name.rsplit(".", 1)
        groups.setdefault(parent, set()).add(suffix)
    expected_groups = {f"transformer.{i}.self_attn" for i in range(int(model.n_layers))}
    expected_groups.add("dep_layer.cross_attn")
    if set(groups) != expected_groups or any(parts != set(TARGET_MODULES) for parts in groups.values()):
        raise ValueError("LoRA targets must cover all four projections of every attention layer")
    if "dep_layer.cross_attn" not in groups:
        raise ValueError("dependency cross-attention must be included")
    return sorted(matches)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_info(directory):
    directory = Path(directory).resolve(strict=True)
    files = {}
    for filename in ("config.json", "model.safetensors"):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        files[filename] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return {"path": str(directory), "files": files}


def _load_local(cls, directory):
    from safetensors.torch import load_file

    directory = Path(directory).resolve(strict=True)
    model_config = json.loads((directory / "config.json").read_text())
    model = cls(**model_config)
    model.load_state_dict(load_file(str(directory / "model.safetensors"), device="cpu"), strict=True)
    return model, model_config


def _seed(seed, device):
    import torch

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    elif str(device) == "mps":
        torch.mps.manual_seed(seed)


def _sync(device):
    import torch

    if str(device) == "mps":
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _assert_frozen(model, tokenizer):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("only LoRA parameters may be trainable")
    if any(p.requires_grad for p in tokenizer.parameters()):
        raise RuntimeError("tokenizer must remain frozen")
    return trainable


def _frozen_parameter_hashes(model, tokenizer):
    """Hash frozen RAM tensors one at a time without keeping a full CPU copy."""
    import torch

    hashes = {}
    for label, module in (("predictor", model), ("tokenizer", tokenizer)):
        digest = hashlib.sha256()
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                continue
            value = parameter.detach().cpu().contiguous()
            identity = [name, str(value.dtype), list(value.shape)]
            digest.update(json.dumps(identity, separators=(",", ":")).encode())
            digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
        hashes[label] = digest.hexdigest()
    return hashes


def load_train_components(config, device="mps"):
    """Load local predictor/tokenizer; return (plain PeftModel, tokenizer, AdamW).

    Optional config keys: ``model_path`` and ``tokenizer_path``. Defaults point
    at this project's existing weights. No Hub fallback or download is possible.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from sktime.libs.kronos import Kronos, KronosTokenizer

    _validate_config(config)
    device = torch.device(device)
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be cpu, mps or cuda")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing silent fallback")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent fallback")
    model_path = Path(config.get("model_path", ROOT / "scenario-lab/models/Kronos-base")).resolve()
    tokenizer_path = Path(config.get("tokenizer_path", ROOT / "scenario-lab/models/Kronos-Tokenizer-base")).resolve()
    source = {"model": _checkpoint_info(model_path), "tokenizer": _checkpoint_info(tokenizer_path)}
    _seed(config["seed"], device)
    base, architecture = _load_local(Kronos, model_path)
    tokenizer, _ = _load_local(KronosTokenizer, tokenizer_path)
    matches = match_attention_modules(base)
    base.requires_grad_(False)
    tokenizer.requires_grad_(False)
    # Kronos is not a Transformers causal LM: no task_type or text-model wrapper.
    lora_config = LoraConfig(
        r=config["rank"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
        target_modules=list(TARGET_MODULES), bias="none", task_type=None,
    )
    model = get_peft_model(base, lora_config).to(device)
    model.peft_config["default"].base_model_name_or_path = str(model_path)
    tokenizer = tokenizer.to(device).eval()
    trainable = _assert_frozen(model, tokenizer)
    model._lora_a_context = {
        "source": source, "architecture": architecture, "matched_modules": matches,
        "training_config": deepcopy(dict(config)),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config["lr"],
                                 weight_decay=config["weight_decay"], betas=tuple(config["adam_betas"]))
    return model, tokenizer, optimizer


def _tokenize(tokenizer, batch_x):
    import torch

    with torch.no_grad():
        token0, token1 = tokenizer.encode(batch_x, half=True)
    if token0.shape != batch_x.shape[:2] or token1.shape != token0.shape:
        raise RuntimeError("tokenizer returned an unexpected token grid")
    return token0, token1


def _train_batch(model, tokenizer, optimizer, batch_x, batch_stamp, config):
    import torch

    trainable = _assert_frozen(model, tokenizer)
    optimizer.zero_grad(set_to_none=True)
    token0, token1 = _tokenize(tokenizer, batch_x)
    # Deliberately omit use_teacher_forcing: preserve upstream's default False.
    logits = model(token0[:, :-1], token1[:, :-1], batch_stamp[:, :-1, :])
    loss, s1_loss, s2_loss = model.get_base_model().head.compute_loss(
        logits[0], logits[1], token0[:, 1:], token1[:, 1:]
    )
    losses = [float(x.detach().item()) for x in (loss, s1_loss, s2_loss)]
    if not all(math.isfinite(value) for value in losses):
        raise FloatingPointError("nonfinite official next-token loss")
    loss.backward()
    parameters = [p for _, p in trainable]
    if not any(p.grad is not None for p in parameters):
        raise RuntimeError("LoRA received no gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError("nonfinite LoRA parameter after optimizer update")
    return {"loss": losses[0], "s1_loss": losses[1], "s2_loss": losses[2],
            "gradient_norm": float(grad_norm.detach().item())}


def train_epoch(model, tokenizer, optimizer, data, config, epoch, log_fn=None):
    """Train one complete January 1–25 grid. ``log_fn`` receives JSON-safe dicts.

    Stats include epoch, examples, expected_examples, batches, mean loss/s1_loss/
    s2_loss, elapsed_seconds, examples_per_second, and pair_order_sha256. Losses
    weight the possibly short final batch by number of asset-origin examples.
    """
    device = next(model.parameters()).device
    _sync(device)
    started = time.perf_counter()
    pairs = epoch_pairs(data, config, epoch)
    _assert_frozen(model, tokenizer)
    if any(group["lr"] != config["lr"] or group["weight_decay"] != config["weight_decay"]
           or tuple(group["betas"]) != tuple(config["adam_betas"]) for group in optimizer.param_groups):
        raise ValueError("optimizer violates constant learning rate/weight decay contract")
    _seed(config["seed"] + epoch, device)
    model.train()
    tokenizer.eval()
    totals = {"loss": 0.0, "s1_loss": 0.0, "s2_loss": 0.0}
    count, batches = 0, 0
    batch_size = config["batch_size"]
    expected_batches = math.ceil(len(pairs) / batch_size)
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start + batch_size]
        batch_x, batch_stamp = make_batch(data, batch_pairs, device)
        losses = _train_batch(model, tokenizer, optimizer, batch_x, batch_stamp, config)
        count += len(batch_pairs)
        batches += 1
        for key in totals:
            totals[key] += losses[key] * len(batch_pairs)
        if log_fn is not None and (batches % 50 == 0 or batches == expected_batches):
            _sync(device)
            log_fn({"event": "training_step", "epoch": epoch, "step": batches,
                    "total_steps": expected_batches, "examples": count,
                    "elapsed_seconds": time.perf_counter() - started,
                    "lr": config["lr"], **losses})
    _sync(device)
    elapsed = time.perf_counter() - started
    return {"epoch": epoch, "examples": count, "expected_examples": len(pairs),
            "batches": batches, **{key: total / count for key, total in totals.items()},
            "elapsed_seconds": elapsed, "examples_per_second": count / elapsed,
            "shuffle_seed": config["seed"] + epoch,
            "pair_order_sha256": hashlib.sha256(pairs.astype("<i8").tobytes()).hexdigest()}


def _new_output(path):
    path = Path(path).resolve()
    # Never overwrite even an empty existing directory or an existing checkpoint.
    path.mkdir(parents=True, exist_ok=False)
    return path


def _saved_files(path):
    return {str(file.relative_to(path)): {"sha256": _sha256(file), "bytes": file.stat().st_size}
            for file in sorted(path.rglob("*")) if file.is_file()}


def save_adapter(model, path):
    """Save safetensors LoRA weights to a new directory, with provenance hashes."""
    context = deepcopy(model._lora_a_context)
    path = _new_output(path)
    model.save_pretrained(str(path), safe_serialization=True, save_embedding_layers=False)
    if not (path / "adapter_model.safetensors").is_file():
        raise RuntimeError("PEFT did not save safetensors adapter weights")
    metadata = {"kind": "kronos-lora-adapter", "path": str(path), "context": context,
                "files": _saved_files(path)}
    (path / "training_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {**metadata, "metadata_sha256": _sha256(path / "training_metadata.json")}


def export_merged(model, path):
    """Create an independent CPU model, merge there, and export standard Kronos.

    The live model, adapters, optimizer references, device and train/eval mode
    remain untouched. No source weight or tokenizer files are written.
    """
    import torch
    from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import save_file
    from sktime.libs.kronos import Kronos

    context = deepcopy(model._lora_a_context)
    source = context["source"]["model"]
    if _checkpoint_info(source["path"]) != source:
        raise RuntimeError("base checkpoint changed since training load")
    # Constructing modules consumes RNG even when all weights are loaded later.
    # Keep the live training stream unchanged as well as its tensors and modes.
    with torch.random.fork_rng(devices=[]):
        base, architecture = _load_local(Kronos, source["path"])
        copy_config = deepcopy(model.peft_config["default"])
        # This temporary custom model has no HF name; the export has its own
        # provenance, and does not need a PEFT base-name rewriting warning.
        copy_config.base_model_name_or_path = None
        copy_model = get_peft_model(base, copy_config)
        adapter_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        result = set_peft_model_state_dict(copy_model, adapter_state)
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise RuntimeError("adapter state failed strict LoRA reconstruction")
        merged = copy_model.merge_and_unload(safe_merge=True).eval()
    state = {key: value.detach().cpu().contiguous() for key, value in merged.state_dict().items()}
    if any("lora_" in key or "base_layer" in key for key in state):
        raise RuntimeError("merged export still contains adapter keys")
    path = _new_output(path)
    (path / "config.json").write_text(json.dumps(architecture, indent=2, allow_nan=False) + "\n")
    save_file(state, str(path / "model.safetensors"))
    metadata = {"kind": "merged-kronos", "path": str(path), "source": source,
                "tokenizer": context["source"]["tokenizer"], "files": _saved_files(path),
                "live_training_state_preserved": True}
    (path / "training_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {**metadata, "metadata_sha256": _sha256(path / "training_metadata.json")}


def smoke_check(model, tokenizer, optimizer, data, config, outdir):
    """Explicitly invoked one-batch update + adapter save/reload logit test.

    This intentionally updates the supplied adapter/optimizer once. The caller
    must reload fresh components before epoch 1; the evidence records that fact.
    """
    import torch
    from peft import PeftModel
    from sktime.libs.kronos import Kronos

    pairs = epoch_pairs(data, config, epoch=0)[:config["batch_size"]]
    device = next(model.parameters()).device
    output = _new_output(outdir)
    trainable = _assert_frozen(model, tokenizer)
    before = {name: p.detach().cpu().clone() for name, p in trainable}
    frozen_before = _frozen_parameter_hashes(model, tokenizer)
    model.train()
    tokenizer.eval()
    batch_x, batch_stamp = make_batch(data, pairs, device)
    _seed(config["seed"], device)
    _sync(device)
    started = time.perf_counter()
    losses = _train_batch(model, tokenizer, optimizer, batch_x, batch_stamp, config)
    _sync(device)
    elapsed = time.perf_counter() - started
    changed = [name for name, p in trainable if not torch.equal(before[name], p.detach().cpu())]
    if not changed:
        raise RuntimeError("one-batch smoke update changed no LoRA parameters")
    adapter = save_adapter(model, output / "adapter")
    token0, token1 = _tokenize(tokenizer, batch_x)
    was_training = model.training
    model.eval()
    try:
        _seed(config["seed"], device)
        with torch.no_grad():
            reference = tuple(x.detach().cpu() for x in model(token0[:, :-1], token1[:, :-1], batch_stamp[:, :-1, :]))
        base, _ = _load_local(Kronos, model._lora_a_context["source"]["model"]["path"])
        restored = PeftModel.from_pretrained(base, str(output / "adapter"), is_trainable=False, local_files_only=True).to(device).eval()
        _seed(config["seed"], device)
        with torch.no_grad():
            recovered = tuple(x.detach().cpu() for x in restored(token0[:, :-1], token1[:, :-1], batch_stamp[:, :-1, :]))
        max_diff = max(float((a - b).abs().max().item()) for a, b in zip(reference, recovered))
        if any(not torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(reference, recovered)):
            raise RuntimeError(f"adapter save/reload raw logits differ: max_abs={max_diff}")
    finally:
        model.train(was_training)
    frozen_after = _frozen_parameter_hashes(model, tokenizer)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen predictor or tokenizer parameters changed during smoke check")
    for source in model._lora_a_context["source"].values():
        if _checkpoint_info(source["path"]) != source:
            raise RuntimeError("source checkpoint changed during smoke check")
    evidence = {"kind": "one-batch-training-smoke", "examples": len(pairs),
                "elapsed_seconds": elapsed, **losses, "changed_lora_tensors": len(changed),
                "adapter": adapter, "reload_logits_max_abs_diff": max_diff,
                "reload_logits_atol": 1e-6, "reload_logits_rtol": 1e-5,
                "frozen_parameters_unchanged": True,
                "frozen_parameter_sha256_before": frozen_before,
                "frozen_parameter_sha256_after": frozen_after,
                "base_files_unchanged": True, "reload_fresh_before_epoch1": True}
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence
