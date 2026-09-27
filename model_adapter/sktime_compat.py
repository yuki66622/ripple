"""Local-checkpoint loader compatibility for sktime 1.2.0.

sktime's Kronos PyTorchModelHubMixin soft import checks distribution name
``huggingface_hub``; scikit-base 1.1.1 lists installed ``huggingface-hub``.
Only loading is replaced: constructor, forward, preprocessing and sampling stay
the installed sktime implementation. No library files or weights are modified.
"""
import json
from pathlib import Path
import threading

from sktime.forecasting.kronos import KronosForecaster

_CACHE = {}
_LOCK = threading.RLock()


class LocalKronosForecaster(KronosForecaster):
    """Strict safetensors loading; supports complete local checkpoints only."""

    _config = {"remember_data": False}

    def _load_kronos(self):
        from safetensors.torch import load_file
        from sktime.libs.kronos import Kronos, KronosTokenizer

        directories = (Path(self.tokenizer_path).resolve(), Path(self.model_path).resolve())
        signatures = tuple((str(directory / filename), (directory / filename).stat().st_size,
                            (directory / filename).stat().st_mtime_ns)
                           for directory in directories for filename in ("config.json", "model.safetensors"))
        key = (signatures, self._device)
        with _LOCK:
            if key not in _CACHE:
                loaded = []
                for cls, directory in zip((KronosTokenizer, Kronos), directories):
                    config = json.loads((directory / "config.json").read_text())
                    model = cls(**config)
                    state = load_file(str(directory / "model.safetensors"), device="cpu")
                    model.load_state_dict(state, strict=True)
                    loaded.append(model.to(self._device).eval())
                _CACHE[key] = tuple(loaded)
            return _CACHE[key]
