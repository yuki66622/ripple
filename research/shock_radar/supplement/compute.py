"""Recalculate supplements once from frozen paths; no model loading or calls."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

from ..io import digest, write_json
from .common import load, frozen_inputs
from .grouped_direction import by_trigger, directions
from .downside_radius import downside, radius
from .groups import grouped

HERE = Path(__file__).resolve().parent


def main():
    target = HERE / "results-v1.json"
    if target.exists():
        raise FileExistsError("Existing supplement evidence must not be overwritten")
    names = ["common.py", "compute.py", "grouped_direction.py", "downside_radius.py", "groups.py", "CONTRACT.md"]
    hashes = {n: digest((HERE / n).read_bytes()) for n in names}
    rows, provenance = load()
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "provenance": provenance, "code_sha256": hashes,
              "trigger_direction": by_trigger(rows), "downside": downside(rows),
              "terminal_direction": directions(rows), "radius": radius(rows), "groups": grouped(rows)}
    assert provenance["frozen_inputs_sha256"] == frozen_inputs()
    assert hashes == {n: digest((HERE / n).read_bytes()) for n in names}
    assert not any(n == "torch" or n.startswith("model_adapter") for n in sys.modules)
    result["verification"] = {"frozen_source_hashes_unchanged": True, "source_files_unchanged": True,
                              "torch_or_model_adapter_imported": False, "model_calls": 0}
    write_json(target, result)
    print(json.dumps({"output": str(target), "events": len(rows),
                      "model_calls": 0, "frozen_inputs_unchanged": True}))


if __name__ == "__main__":
    main()
