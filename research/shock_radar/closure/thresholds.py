"""Fixed January MDD alarm calibration; immutable predictions, no model imports."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
from math import isfinite
from numbers import Real
from pathlib import Path
import sys

THRESHOLDS = (.01, .02, .03, .05)
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "artifacts/thresholds-v1"
ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC")


def _mdd(values, name):
    values = list(values)
    if any(isinstance(x, bool) or not isinstance(x, Real) or not isfinite(x) or not 0 <= x <= 1
           for x in values):
        raise ValueError(f"{name} must contain finite MDD values in [0,1]")
    return values


def evaluate_thresholds(predicted, actual):
    """All four strict thresholds; precision/recall/F1 undefined only at zero denominators."""
    predicted, actual = _mdd(predicted, "predicted"), _mdd(actual, "actual")
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have identical pair counts")
    rows = []
    for threshold in THRESHOLDS:
        tp = fp = fn = tn = 0
        for p, a in zip(predicted, actual):
            pp, ap = p > threshold, a > threshold
            if pp and ap:
                tp += 1
            elif pp:
                fp += 1
            elif ap:
                fn += 1
            else:
                tn += 1
        predicted_positive, actual_positive = tp + fp, tp + fn
        f1_denominator = 2 * tp + fp + fn
        reasons = []
        if actual_positive < 10:
            reasons.append("fewer_than_ten_actual_positives")
        if predicted_positive < 10:
            reasons.append("fewer_than_ten_predicted_positives")
        rows.append({
            "threshold": threshold, "comparison": ">", "asset_pair_count": len(actual),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "predicted_positives": predicted_positive, "actual_positives": actual_positive,
            "precision": tp / predicted_positive if predicted_positive else None,
            "precision_reason": None if predicted_positive else "no_predicted_positives",
            "recall": tp / actual_positive if actual_positive else None,
            "recall_reason": None if actual_positive else "no_actual_positives",
            "f1": 2 * tp / f1_denominator if f1_denominator else None,
            "f1_reason": None if f1_denominator else "no_actual_or_predicted_positives",
            "recommendation_eligible": not reasons, "recommendation_exclusion_reasons": reasons,
        })
    eligible = [r for r in rows if r["recommendation_eligible"]]
    # Rational comparison preserves exact F1 ties before the lower-threshold tie break.
    chosen = min(eligible, key=lambda r: (-Fraction(2*r["tp"], 2*r["tp"]+r["fp"]+r["fn"]), r["threshold"])) if eligible else None
    return {"asset_pair_count": len(actual), "rows": rows,
            "recommended_threshold": chosen["threshold"] if chosen else None,
            "recommendation_reason": "highest_f1_among_supported_thresholds" if chosen else "no_threshold_has_ten_actual_and_ten_predicted_positives",
            "recommendation_rule": "actual positives >=10 AND predicted positives >=10; highest exact F1; ties lower threshold",
            "existing_product_threshold": .03, "product_rule_changed": False}


def load_frozen_pairs():
    """Revalidate January truth and saved scores, then read the exact saved MDDs."""
    from ..supplement.common import BATCH, ROOT, frozen_inputs, load
    from ..io import digest

    rows, provenance = load()  # January-only loader; imports no model or inference runner.
    if len(rows) != 84:
        raise ValueError("exactly 84 frozen January events required")
    pairs = []
    for row in rows:
        if row["assets"] != list(ASSETS):
            raise ValueError("frozen ordered ten-asset universe required")
        path = BATCH / (row["event_id"] + ".json")
        raw = path.read_bytes()
        if digest(raw) != provenance["frozen_inputs_sha256"][str(path.relative_to(ROOT))]:
            raise ValueError("event changed after validation")
        artifact = json.loads(raw)
        for asset in ASSETS:  # Deliberately all ten: alarm calibration does not exclude origins.
            pairs.append({"event_id": row["event_id"], "timestamp": row["timestamp"], "asset": asset,
                          "predicted_mdd": artifact["predicted_risk"]["kronos_base"][asset]["max_drawdown"],
                          "actual_mdd": artifact["actual_risk"][asset]["max_drawdown"]})
    if len(pairs) != 840 or len({(r["event_id"], r["asset"]) for r in pairs}) != 840:
        raise ValueError("840 unique event/asset pairs required")
    if frozen_inputs() != provenance["frozen_inputs_sha256"]:
        raise ValueError("original evidence changed during threshold loading")
    return pairs, provenance


def render_report(result):
    recommended = result["recommended_threshold"]
    message = ("四个固定阈值均未满足阳性数量门槛，本次建议为空。" if recommended is None else
               f"冻结规则选出 **{recommended:.0%}** 作为一月样本内探索性建议。现有 **3% 产品规则不变**。")
    lines = ["# 一月冻结预测：回撤告警阈值检查", "", message, "",
             "84 个事件 × 全部 10 个资产，共 840 组；包含起源资产。预测与真实标签分别使用各自未来 30 分钟收盘路径的最大回撤，均按严格大于阈值判定，比较前不四舍五入。", "",
             "| 阈值 | TP | FP | FN | TN | 预测阳性 | 实际阳性 | Precision | Recall | F1 | 满足推荐门槛 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    def metric(value):
        return "N/A" if value is None else f"{value:.2%}"
    for row in result["rows"]:
        lines.append(f"| {row['threshold']:.0%} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
                     f"{row['predicted_positives']} | {row['actual_positives']} | {metric(row['precision'])} | "
                     f"{metric(row['recall'])} | {metric(row['f1'])} | {'是' if row['recommendation_eligible'] else '否'} |")
    lines += ["", "推荐先要求实际阳性和预测阳性都至少 10 组，再取最高 F1；并列选更低阈值。四行全部保留，没有依结果删样本。", "",
              "Precision 无预测阳性时为空；Recall 无实际阳性时为空。F1 = 2TP / (2TP + FP + FN)：存在错误但 TP=0 时为 0，实际与预测均无阳性时为空。具体原因保存在 JSON。", "",
              "这是同一批一月事件内的探索性校准，不是独立验证或概率校准。资产与相邻事件相关，840 组不能视为 840 个独立样本；没有部署阈值、修改演示规则或重新推理。", "",
              "结果与逐组数值：[results.json](artifacts/thresholds-v1/results.json)、[pairs.csv](artifacts/thresholds-v1/pairs.csv)。源数据、原预测及评分已重验，原证据哈希前后未变。", ""]
    return "\n".join(lines)


def generate(output_dir=OUTPUT, report_path=HERE / "THRESHOLDS.md"):
    from ..io import digest, write_json
    from ..supplement.common import frozen_inputs

    output_dir, report_path = Path(output_dir), Path(report_path)
    if output_dir.exists() or report_path.exists():
        raise FileExistsError("threshold artifacts/report already exist; never overwrite evidence")
    code_paths = [Path(__file__), HERE / "CONTRACT.md"]
    code_hashes = {p.name: digest(p.read_bytes()) for p in code_paths}
    pairs, provenance = load_frozen_pairs()
    result = evaluate_thresholds([r["predicted_mdd"] for r in pairs], [r["actual_mdd"] for r in pairs])
    result.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
                   "event_count": 84, "assets": list(ASSETS), "model": "Kronos-base",
                   "prediction_path_count": 1, "horizon_minutes": 30, "provenance": provenance,
                   "code_sha256": code_hashes, "model_calls": 0,
                   "disclosure": "January within-sample exploration; correlated asset-event pairs; no product rule change."})
    for row in result["rows"]:
        row["event_count"] = 84
    if provenance["frozen_inputs_sha256"] != frozen_inputs():
        raise ValueError("frozen source evidence changed during threshold calculation")
    if code_hashes != {p.name: digest(p.read_bytes()) for p in code_paths}:
        raise ValueError("threshold code/contract changed during calculation")
    if any(name == "torch" or name.startswith("model_adapter") for name in sys.modules):
        raise RuntimeError("threshold generation must not import a model runtime")
    result["verification"] = {"source_hashes_unchanged": True, "original_scores_revalidated": True,
                              "torch_or_model_adapter_imported": False, "all_origin_assets_included": True}
    report = render_report(result)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "pairs.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    result["pairs_csv_sha256"] = digest((output_dir / "pairs.csv").read_bytes())
    write_json(output_dir / "results.json", result)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(report)
    return result


def main():
    result = generate()
    print(json.dumps({key: result[key] for key in ("status", "event_count", "asset_pair_count", "rows", "recommended_threshold", "model_calls")}))


if __name__ == "__main__":
    main()
