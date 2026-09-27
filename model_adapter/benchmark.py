"""Bounded actual-model smoke/latency evidence; never launches a full backtest."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from data_pipeline import load_window
from model_adapter import KronosAdapter, PredictionValidationError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--paths", nargs="+", type=int, default=[1, 1, 2, 4])
    parser.add_argument("--output", default="model_adapter/evidence/benchmark-256-30.json")
    args = parser.parse_args()
    if len(args.paths) > 8 or any(n < 1 or n > 16 for n in args.paths):
        parser.error("bounded smoke test permits at most eight runs of 1..16 paths")
    window = load_window(as_of=args.as_of)
    adapter = KronosAdapter()
    result = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "machine": platform.platform(), "window_id": window["window_id"],
              "as_of": window["as_of"], "source": window["source"], "runs": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, paths in enumerate(args.paths):
        run = {"config": {"lookback": 256, "horizon": 30, "path_count": paths, "seed": 20260926}, "index": index}
        started = time.perf_counter()
        try:
            prediction = adapter.predict(window, run["config"], f"smoke-{index}")
            run.update(status="valid", **prediction)
        except PredictionValidationError as exc:
            run.update(status="rejected", issues=exc.issues, raw_paths=exc.raw_paths, runtime=exc.runtime)
        run["wall_ms"] = (time.perf_counter() - started) * 1000
        result["runs"].append(run)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: run[key] for key in ("index", "status", "wall_ms")} | {"path_count": paths, "inference_ms": run["runtime"]["inference_ms"], "issues": len(run.get("issues", []))}), flush=True)
    first_paths = result["runs"][0]["raw_paths"]
    result["same_seed_first_path_identical_across_runs"] = all(run["raw_paths"][:len(first_paths)] == first_paths for run in result["runs"])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"evidence": str(output), "same_seed_first_path_identical_across_runs": result["same_seed_first_path_identical_across_runs"]}))


if __name__ == "__main__":
    main()
