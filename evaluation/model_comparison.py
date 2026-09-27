"""Offline paired financial errors for two separately verified checkpoints.

The CLI revalidates both saved batches with comparison.compare_batch. It never
loads a forecasting model, trains, resamples predictions or picks a best retry.
The pure compare_reports function accepts those already-verified reports.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean

from .comparison import ASSETS, compare_batch
from .corrections import COMPARABLE_IDENTITY_FIELDS, comparison_context
from .universe import report_assets

METRICS = ("terminal_return", "volatility")
SOURCE_FIELDS = ("profile_id", "source", "quote_currency")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric metric required")
    return float(value)


def _equal(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _origin(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.utcoffset() is None:
        raise ValueError("origins require an explicit timezone")
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _index_report(report):
    """Reject pooled/config-ambiguous reports; failed tasks remain indexed."""
    tasks = report.get("tasks")
    assets = report_assets(report)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("a nonempty verified batch report is required on each side")
    requested = report.get("coverage", {}).get("requested_windows")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested != len(tasks):
        raise ValueError("verified report denominator does not match its task ledger")
    config = report.get("predict_config")
    if not isinstance(config, dict) or set(config) != {"lookback", "horizon", "path_count", "seed"}:
        raise ValueError("one complete fixed prediction configuration is required per report")
    if any(not isinstance(report.get(key), str) or not report[key] for key in SOURCE_FIELDS):
        raise ValueError("complete source/profile/quote provenance is required")
    revisions = report.get("model_revisions")
    if not isinstance(revisions, list) or len(revisions) > 1:
        raise ValueError("multiple checkpoint revisions cannot be pooled")
    indexed, task_ids = {}, set()
    for item in tasks:
        origin, task_id = _origin(item["as_of"]), item["task_id"]
        if origin in indexed or task_id in task_ids:
            raise ValueError("duplicate task or origin in verified report; no attempt may be selected")
        if not isinstance(item.get("status"), str) or not isinstance(item.get("window_id"), str):
            raise ValueError("each task needs its input identity and verification status")
        if item["status"] == "scored" and set(item.get("pairs", {})) != set(assets):
            raise ValueError("scored report must cover every declared asset, with no extras")
        indexed[origin] = item
        task_ids.add(task_id)
    counts = dict(Counter(item["status"] for item in tasks))
    if counts != report["coverage"].get("status_counts"):
        raise ValueError("verified report status counts do not match its task ledger")
    if report["coverage"].get("paired_windows") != counts.get("scored", 0):
        raise ValueError("verified report scored denominator does not match its task ledger")
    return indexed


def _controls(report, item):
    identity = report.get("expected_identity")
    context = comparison_context({
        "window_id": item["window_id"], "predict_config": report["predict_config"],
        "result": {"runtime": {"identity": identity}, "forecast": {
            "model_revision": item.get("model_revision")}},
    })
    if context is None:
        return None
    revision = identity["model_revision"]
    if item.get("model_revision") != revision or report.get("model_revisions") != [revision]:
        return None
    return context["identity"]


def _retained(item):
    if item is None:
        return None
    return {key: item[key] for key in ("task_id", "window_id", "status", "forecast_id",
                                       "artifact_paths", "errors", "attempts", "model_revision") if key in item}


def _paired_errors(left, right, assets):
    """Check targets and each model's error; never consume naive error fields."""
    errors = {}
    for asset in assets:
        errors[asset] = {}
        for metric in METRICS:
            lhs, rhs = left["pairs"][asset][metric], right["pairs"][asset][metric]
            actual_left, actual_right = _number(lhs["actual"]), _number(rhs["actual"])
            # Same realized target means the same value, not merely a nearby
            # price/volatility. Source/config/input controls are checked first.
            if actual_left != actual_right:
                raise ValueError(f"target mismatch: {asset}/{metric}")
            row = {"actual": actual_left}
            for name, pair in (("left", lhs), ("right", rhs)):
                estimate, error = _number(pair["model"]), _number(pair["model_error"])
                if error < 0 or not _equal(error, abs(estimate - actual_left)):
                    raise ValueError(f"model error mismatch: {name}/{asset}/{metric}")
                row[name + "_error"] = error
            errors[asset][metric] = row
    return errors


def _statistics(pairs):
    left = mean(a for a, _ in pairs) if pairs else None
    right = mean(b for _, b in pairs) if pairs else None
    ties = sum(_equal(a, b) for a, b in pairs)
    wins = sum(b < a and not _equal(a, b) for a, b in pairs)
    return {
        "window_count": len(pairs), "left_mae": left, "right_mae": right,
        "right_minus_left_mae": right - left if pairs else None,
        "right_vs_left_skill": 1 - right / left if left else None,
        "skill_status": "no_pairs" if not pairs else ("undefined_zero_left_mae" if not left else "defined"),
        "win": wins, "tie": ties, "loss": len(pairs) - wins - ties,
        "win_definition": "right model has lower error on this window",
    }


def compare_reports(left, right):
    """Pair outputs from compare_batch; preserve the union of both task ledgers.

    This pure arithmetic function is also tested with explicitly synthetic
    report fixtures. Production callers should use compare_model_batches/CLI,
    which revalidates saved raw forecasts and independent truth on both sides.
    """
    lhs, rhs = _index_report(left), _index_report(right)
    left_assets, right_assets = report_assets(left), report_assets(right)
    assets = left_assets
    ledger, matched = [], []
    for origin in sorted(set(lhs) | set(rhs)):
        a, b = lhs.get(origin), rhs.get(origin)
        row = {"as_of": origin, "left": _retained(a), "right": _retained(b), "reasons": []}
        if a is None or b is None:
            row["status"] = "only_right" if a is None else "only_left"
        elif a["status"] != "scored" or b["status"] != "scored":
            row.update(status="unavailable_counterpart", reasons=["Both sides must independently pass batch verification and scoring."])
        elif left_assets != right_assets:
            row.update(status="incompatible_universe", reasons=["The ordered declared asset universes must match exactly."])
        elif any(left[k] != right[k] for k in SOURCE_FIELDS):
            row.update(status="incompatible_source", reasons=[k for k in SOURCE_FIELDS if left[k] != right[k]])
        elif left["predict_config"] != right["predict_config"]:
            row.update(status="incompatible_config", reasons=["Prediction configuration, including seed, must match exactly."])
        elif a["window_id"] != b["window_id"]:
            row.update(status="incompatible_input", reasons=["Same origin has different historical input identity."])
        else:
            control_a, control_b = _controls(left, a), _controls(right, b)
            if control_a is None or control_b is None or control_a != control_b:
                reasons = (["Missing or inconsistent complete runtime controls."] if control_a is None or control_b is None
                           else [key for key in COMPARABLE_IDENTITY_FIELDS if control_a[key] != control_b[key]])
                row.update(status="incompatible_controls", reasons=reasons)
            else:
                try:
                    row["errors"] = _paired_errors(a, b, assets)
                    row["status"] = "matched"
                    matched.append(row)
                except (KeyError, TypeError, ValueError) as exc:
                    row.update(status="incompatible_target_or_error", reasons=[str(exc)])
        ledger.append(row)
    counts = Counter(row["status"] for row in ledger)
    errors = {}
    for metric in METRICS:
        by_asset = {}
        for asset in (*assets, "equal_weight_assets"):
            chosen = assets if asset == "equal_weight_assets" else (asset,)
            pairs = [(mean(row["errors"][s][metric]["left_error"] for s in chosen),
                      mean(row["errors"][s][metric]["right_error"] for s in chosen)) for row in matched]
            by_asset[asset] = _statistics(pairs)
        errors[metric] = {"units": "decimal", "by_asset": by_asset}
    complete = (len(matched) == len(lhs) == len(rhs)
                and left.get("status") == right.get("status") == "complete")
    revision_left = (left.get("expected_identity") or {}).get("model_revision")
    revision_right = (right.get("expected_identity") or {}).get("model_revision")
    same_revision = isinstance(revision_left, str) and bool(revision_left) and revision_left == revision_right
    return {
        "schema_version": 1,
        "assets": list(assets),
        "status": "complete" if complete else ("partial" if matched else "no_comparable_results"),
        "quality_status": "descriptive_only; completion is not model superiority",
        "same_model_revision": same_revision,
        "self_comparison": same_revision and complete,
        "left": {key: left.get(key) for key in ("experiment_id", "model", "model_revisions", "expected_identity",
                    "assets", "predict_config", *SOURCE_FIELDS, "coverage", "status", "batch_dir", "batch_summary_error", "unassigned_records")},
        "right": {key: right.get(key) for key in ("experiment_id", "model", "model_revisions", "expected_identity",
                    "assets", "predict_config", *SOURCE_FIELDS, "coverage", "status", "batch_dir", "batch_summary_error", "unassigned_records")},
        "coverage": {
            "left_requested_windows": len(lhs), "right_requested_windows": len(rhs),
            "union_origins": len(ledger), "matched_windows": len(matched),
            "left_unmatched_windows": len(lhs) - len(matched), "right_unmatched_windows": len(rhs) - len(matched),
            "only_left_windows": counts["only_left"], "only_right_windows": counts["only_right"],
            "unavailable_counterpart_windows": counts["unavailable_counterpart"],
            "incompatible_windows": sum(value for key, value in counts.items() if key.startswith("incompatible_")),
            "matched_fraction_of_left": len(matched) / len(lhs), "matched_fraction_of_right": len(matched) / len(rhs),
            "status_counts": dict(sorted(counts.items())),
        },
        "errors": errors, "tasks": ledger,
        "method": {
            "pairing": "same UTC origin AND window content hash, source/profile/quote, full prediction config and strict runtime controls; same realized targets",
            "required_runtime_controls": list(COMPARABLE_IDENTITY_FIELDS),
            "aggregation": "model_error from each side only; equal mean over all declared asset errors per window, then mean across matched windows; not a mean of asset skills",
            "skill": "1 - right MAE / left MAE; null if left MAE is zero; positive means lower right error on matched windows",
            "win_tie_loss": "right versus left per-window errors; ties use relative tolerance 1e-10 and absolute tolerance 1e-12",
            "selection": "Union ledger retained. No resampling, retries, best-attempt selection, or changes to model predictions.",
            "self_comparison": "Same actual weight revision with completely matched inputs and controls; this is a same-checkpoint control, not a fine-tuning result.",
        },
        "disclosure": "Descriptive paired errors only; engineering completion or one fixed sampling seed does not prove fine-tuning improves forecasting. Failure-conditioned matched subsets may be biased. This comparison is not a pristine holdout or a guarantee of future performance. No trained checkpoint is created here.",
    }


def compare_model_batches(left_manifest, left_batch_dir, right_manifest, right_batch_dir):
    """Verify both saved batches against their own manifests before pairing."""
    left = compare_batch(left_manifest, left_batch_dir, bootstrap_replicates=0)
    right = compare_batch(right_manifest, right_batch_dir, bootstrap_replicates=0)
    return compare_reports(left, right)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-manifest", type=Path, required=True)
        parser.add_argument(f"--{side}-batch-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new JSON file; never overwrite retained evidence")
    args = parser.parse_args(argv)
    left_manifest = json.loads(args.left_manifest.read_text())
    right_manifest = json.loads(args.right_manifest.read_text())
    result = compare_model_batches(left_manifest, args.left_batch_dir, right_manifest, args.right_batch_dir)
    result["left"]["manifest_path"] = str(args.left_manifest.resolve())
    result["right"]["manifest_path"] = str(args.right_manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": result["status"], "coverage": result["coverage"],
                      "out": str(args.out.resolve())}, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
