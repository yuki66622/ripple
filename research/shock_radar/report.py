"""Generate the January report and frozen replays from a completed saved run.

No inference, network, training, other-month data, or new event selection.
Run explicitly after summary.json exists. Every output uses exclusive creation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median

from forecast_metrics.engine import freeze_forecast

from .io import ROOT, digest, future_closes, load_january, market_window, write_json
from .metrics import (RISK_METRICS, _average_ranks, _top_weights,
                      aggregate_events, risk_metrics, score_event)

ARTIFACTS = ROOT / "research/shock_radar/artifacts/january-v1"
METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]
LABELS = {"kronos_base": "Kronos-base", "no_propagation": "不传播基线",
          "btc_beta": "BTC beta 基线", "historical_30": "过去 30 分钟基线"}
GROUP_LABELS = {"all": "全部事件", "systemic": "系统性事件（事后分类）",
                "non_systemic": "非系统性事件（事后分类）"}
METRIC_LABELS = {"max_drawdown": "最大回撤", "volatility": "波动率"}


def _read(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, digest(raw)


def _same(expected, actual, name):
    """Tolerate float roundoff only, never missing fields or changed coverage."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"{name}: object fields differ")
        for key in expected:
            _same(expected[key], actual[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{name}: sequence differs")
        for i, (left, right) in enumerate(zip(expected, actual)):
            _same(left, right, f"{name}[{i}]")
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError(f"{name}: numerical evidence differs")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"{name}: evidence differs")


def load_evidence(batch_dir=ARTIFACTS / "inference-original-v1"):
    """Require completion first; bind selection, stored artifacts, and Jan truth."""
    batch_dir = Path(batch_dir).resolve()
    summary_path = batch_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run is not complete: summary.json must exist before reporting")
    summary, summary_hash = _read(summary_path)
    manifest, manifest_hash = _read(batch_dir / "run_manifest.json")
    selection, selection_hash = _read(ARTIFACTS / "selection-approved-k6.json")
    policy, policy_hash = _read(ARTIFACTS / "replay-selection-policy.json")
    catalog, catalog_hash = _read(ARTIFACTS / "catalog-k6.json")
    if (summary.get("status") != "complete" or summary.get("model_identity_unchanged") is not True
            or summary.get("model_identity") != manifest.get("model_identity")):
        raise ValueError("Completed run with unchanged model identity required")
    if (selection.get("catalog_sha256") != catalog_hash or selection.get("selected_k") != 6
            or selection.get("authorization") != "用 k=6 的全部 84 次，接受样本数超出原范围"
            or selection.get("original_automatic_selected_k") is not None
            or selection.get("original_count_gate_passed") is not False
            or selection.get("subsampled") is not False):
        raise ValueError("Frozen all-84 January authorization required")
    events = [event for event in catalog["events"] if event["eligible"]]
    ids = [event["event_id"] for event in events]
    if (len(events) != 84 or len(set(ids)) != 84 or ids != selection.get("event_ids")
            or any(item.get("event_count") != 84 for item in (summary, manifest, selection))):
        raise ValueError("The complete approved 84-event denominator is required")
    if (manifest.get("selection") != selection or manifest.get("replay_selection_policy") != policy
            or manifest.get("method_names") != METHODS or summary.get("method_names") != METHODS
            or manifest.get("predict_config") != selection.get("predict_config")
            or summary.get("predict_config") != selection.get("predict_config")
            or selection.get("predict_config") != {"lookback": 256, "horizon": 30,
                                                   "path_count": 1, "seed": 20260926}):
        raise ValueError("Selection, method list or frozen prediction configuration changed")
    if (policy.get("median_reference_population") != "all_84_eligible_events"
            or policy.get("selected_before_inference") is not True):
        raise ValueError("The pre-inference all-84 median replay policy is required")
    chosen = selection.get("replays_selected_before_inference", [])
    if (len(chosen) != 3 or len({row["event_id"] for row in chosen}) != 3
            or any(row["event_id"] not in ids for row in chosen)):
        raise ValueError("Exactly the three distinct preselected replay IDs are required")
    actual_ids = {path.stem for path in batch_dir.glob("event_*.json")}
    if actual_ids != set(ids):
        raise ValueError("Missing or unexpected event artifacts; no successful subset report")
    for name, expected in manifest["code_sha256"].items():
        if Path(name).name != name or digest((batch_dir / "code_at_run" / name).read_bytes()) != expected:
            raise ValueError("Saved run source hash differs")
    for name in ("io.py", "metrics.py"):
        if digest((ROOT / "research/shock_radar" / name).read_bytes()) != manifest["code_sha256"][name]:
            raise ValueError("Report arithmetic/data reader changed since the saved run")

    data = load_january()  # Sole market-data entry; never the legacy profile loader.
    if data["provenance"] != manifest["provenance"]:
        raise ValueError("January data provenance differs from the inference run")
    artifacts, hashes, scores = {}, {}, []
    for event in events:
        event_id, index = event["event_id"], event["index"]
        artifact, hashes[event_id] = _read(batch_dir / (event_id + ".json"))
        _same(event, artifact["event"], event_id + ".event")
        window = market_window(data, index)
        if artifact.get("window_id") != window["window_id"] or event["timestamp"] != window["as_of"]:
            raise ValueError("Saved event does not bind to the January input window")
        spots = {a: window["histories"][a][-1]["close"] for a in data["assets"]}
        actual = {a: risk_metrics(spots[a], closes) for a, closes in future_closes(data, index).items()}
        _same(actual, artifact["actual_risk"], event_id + ".actual_risk")
        predictions = {}
        for method in METHODS:
            stored = artifact["predicted_risk"][method]
            if stored is None:
                if method not in artifact["errors"]:
                    raise ValueError("Unavailable prediction must retain its failure evidence")
                predictions[method] = None
                continue
            paths = artifact["method_close_paths"][method]
            if set(paths) != set(data["assets"]) or any(len(p) != 30 for p in paths.values()):
                raise ValueError("Successful method requires ten complete 30-minute paths")
            predictions[method] = {a: risk_metrics(spots[a], paths[a]) for a in data["assets"]}
            _same(predictions[method], stored, event_id + "." + method)
        if predictions["kronos_base"] is not None:
            forecast = artifact["forecast"]
            if (freeze_forecast(forecast).identity != artifact["forecast_id"]
                    or artifact["runtime"]["identity"] != manifest["model_identity"]
                    or len(forecast["paths"]) != 1
                    or forecast["times"] != data["times"][index+1:index+31]):
                raise ValueError("Kronos forecast identity or horizon mismatch")
            _same({a: forecast["paths"][0]["assets"][a]["close"] for a in data["assets"]},
                  artifact["method_close_paths"]["kronos_base"], event_id + ".kronos_closes")
        score = score_event(event, predictions, actual, data["assets"])
        _same(score, artifact["score"], event_id + ".score")
        scores.append(score)
        artifacts[event_id] = artifact
    recomputed = aggregate_events(scores, methods=METHODS)
    for key, value in recomputed.items():
        _same(value, summary[key], "summary." + key)
    coverage_path = ARTIFACTS / "event_coverage.json"
    coverage, coverage_hash = _read(coverage_path) if coverage_path.exists() else (None, None)
    return {"data": data, "summary": summary, "manifest": manifest, "selection": selection,
            "policy": policy, "artifacts": artifacts, "coverage": coverage, "batch_dir": batch_dir,
            "hashes": {"summary_sha256": summary_hash, "run_manifest_sha256": manifest_hash,
                       "selection_sha256": selection_hash, "catalog_sha256": catalog_hash,
                       "replay_selection_policy_sha256": policy_hash,
                       "event_coverage_sha256": coverage_hash, "event_sha256": hashes,
                       "report_generator_sha256": digest(Path(__file__).read_bytes())}}


def _rankings(table, assets):
    """Keep declared order and numerical tied ranks; do not pick tie winners."""
    result = {}
    for metric in RISK_METRICS:
        values = [table[a][metric] for a in assets]
        ranks = _average_ranks([-value for value in values])
        weights, cutoff_ties = _top_weights(values, 3)
        result[metric] = {"universe": list(assets), "higher_value_means_higher_risk": True,
                          "cutoff_tie_count": cutoff_ties,
                          "entries": [{"asset": a, "value": v, "average_rank_desc": r,
                                       "top3_inclusion_weight": w}
                                      for a, v, r, w in zip(assets, values, ranks, weights)]}
    return result


def build_replay(evidence, choice):
    """Retain a failed selected event; never search for a better replacement."""
    data, manifest = evidence["data"], evidence["manifest"]
    artifact = evidence["artifacts"][choice["event_id"]]
    event, score = artifact["event"], artifact["score"]
    index, assets = event["index"], data["assets"]
    times = data["times"][index-30:index+31]
    if len(times) != 61:
        raise ValueError("Selected replay lacks its frozen 61-minute observed window")
    observed = {a: [row["close"] for row in data["histories"][a][index-30:index+31]] for a in assets}
    methods = {}
    for method in METHODS:
        risk = artifact["predicted_risk"][method]
        available = risk is not None and score["methods"][method]["status"] == "scored"
        methods[method] = {
            "status": score["methods"][method]["status"],
            "future_close_paths": artifact["method_close_paths"].get(method) if available else None,
            "future_times": times[31:], "predicted_risk": risk,
            "rankings_all_assets": _rankings(risk, assets) if available else None,
            "rankings_scored_assets": _rankings(risk, score["scored_assets"]) if available else None,
            "score": score["methods"][method], "error": artifact["errors"].get(method),
        }
    forecast = artifact.get("forecast") or {}
    return {
        "schema_version": 1, "purpose": "preselected January exploration replay; no model calls",
        "selection": choice, "replay_selection_policy": evidence["policy"],
        "status": score["status"], "event": event, "forecast_id": artifact.get("forecast_id"),
        "window_id": artifact["window_id"], "assets": assets, "as_of": event["timestamp"],
        "classification": {"systemic_flag": event["systemic_flag"],
                           "available_at": event["systemic_confirmed_at"],
                           "available_at_t0": False, "posthoc": True,
                           "note": "Uses retained triggers through t0+10; never supplied to the model."},
        "observed": {"times": times, "close_paths": observed, "point_count": 61,
                     "timestamp_semantics": "UTC candle end"},
        "past_at_t0": {"times": times[:31], "close_paths": {a: p[:31] for a, p in observed.items()},
                       "point_count": 31, "includes_t0": True},
        "future_truth": {"times": times[31:], "close_paths": {a: p[31:] for a, p in observed.items()},
                         "point_count": 30, "available_at_t0": False},
        "methods": methods, "actual_risk": artifact["actual_risk"],
        "actual_rankings_all_assets": _rankings(artifact["actual_risk"], assets),
        "actual_rankings_scored_assets": _rankings(artifact["actual_risk"], score["scored_assets"]),
        "scoring_universe": score["scored_assets"], "score": score,
        "baseline_metadata": artifact["baseline_metadata"], "errors": artifact["errors"],
        "data_identity": manifest["provenance"], "model_identity": manifest["model_identity"],
        "predict_config": manifest["predict_config"], "runtime": artifact.get("runtime"),
        "ohlc_audit": forecast.get("ohlc_corrections"), "volume_flags": forecast.get("volume_quality"),
        "audit_note": "Exact saved numerical audits; finite negative volume/amount are not price failures. No UI flag implied.",
        "source_artifact": str(evidence["batch_dir"] / (event["event_id"] + ".json")),
        "evidence_hashes": {key: value for key, value in evidence["hashes"].items() if key != "event_sha256"},
        "source_artifact_sha256": evidence["hashes"]["event_sha256"][event["event_id"]],
        "code_at_run_sha256": manifest["code_sha256"],
    }


def _link(label, path):
    return f"[{label}](<{Path(path).resolve()}>)"


def _number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_report(evidence, replay_paths):
    summary, selection, coverage = evidence["summary"], evidence["selection"], evidence["coverage"]
    groups = summary["groups"]
    lines = ["# 冲击后风险排序：一月探索报告", "",
             f"已完成全部 {summary['event_count']} 次预先锁定事件；四方法共同可比事件为 "
             f"{groups['all']['common_paired_event_count']} 次。下列结果衡量未来 30 分钟风险大小与排序，不能证明冲击传播的因果关系。", ""]
    # Concrete signed differences avoid a flattering overall verdict on mixed metrics.
    for metric in RISK_METRICS:
        entries = groups["all"]["scores"]
        model = entries["kronos_base"][metric]
        controls = [(m, entries[m][metric]) for m in METHODS[1:] if entries[m][metric]["mae_bps"] is not None]
        if model["mae_bps"] is None or not controls:
            lines.append(f"- {METRIC_LABELS[metric]}：缺少共同配对数据，不能比较。")
            continue
        best_name, best = min(controls, key=lambda row: row[1]["mae_bps"])
        delta = model["mae_bps"] - best["mae_bps"]
        relation = "高" if delta > 0 else "低" if delta < 0 else "相同"
        difference = f"{relation} {abs(delta):.3f} bps" if delta else relation
        top_name, top = max(controls, key=lambda row: row[1]["top3_recall_expected"])
        top_delta = 100 * (model["top3_recall_expected"] - top["top3_recall_expected"])
        lines.append(f"- {METRIC_LABELS[metric]}：Kronos MAE 比误差最低的 {LABELS[best_name]} {difference}；"
                     f"Top3 期望召回与召回最高的 {LABELS[top_name]} 的差值为 {top_delta:+.2f} 个百分点。")
    lines += ["", "## 事件覆盖", "", "| 组别 | 预选事件 | 四方法共同配对 | 未进入配对 |",
              "|---|---:|---:|---:|"]
    for group, info in groups.items():
        lines.append(f"| {GROUP_LABELS[group]} | {info['catalog_event_count']} | {info['common_paired_event_count']} | "
                     f"{info['catalog_event_count']-info['common_paired_event_count']} |")
    lines += ["", "| 方法 | 已评分 / 全部事件 | 失败 | 缺失或其他状态 |", "|---|---:|---:|---|"]
    for method in METHODS:
        counts = groups["all"]["method_status_counts"][method]
        other = {k: v for k, v in counts.items() if k not in ("scored", "failed")}
        lines.append(f"| {LABELS[method]} | {counts.get('scored',0)} / {summary['event_count']} | "
                     f"{counts.get('failed',0)} | {json.dumps(other,ensure_ascii=False) if other else '0'} |")
    if coverage:
        lines += ["", f"事件覆盖 {coverage['utc_days_with_events']} 个 UTC 日；未来区间有 "
                  f"{coverage['overlapping_future_interval_pairs']} 对重叠，形成 "
                  f"{coverage['connected_future_interval_blocks']} 个连通块。这些计数不证明样本独立。"]
    for group in ("all", "systemic", "non_systemic"):
        info = groups[group]
        lines += ["", f"## {GROUP_LABELS[group]}（共同配对 n={info['common_paired_event_count']}）"]
        for metric in RISK_METRICS:
            lines += ["", f"**{METRIC_LABELS[metric]}**", "",
                      "| 方法 | MAE（bps，越低越好） | Spearman | 相关系数 defined n / 配对 n | Top3 期望召回 |",
                      "|---|---:|---:|---:|---:|"]
            for method in METHODS:
                s = info["scores"][method][metric]
                recall = "N/A" if s["top3_recall_expected"] is None else f"{s['top3_recall_expected']:.2%}"
                lines.append(f"| {LABELS[method]} | {_number(s['mae_bps'])} | {_number(s['spearman'])} | "
                             f"{s['spearman_defined_events']} / {s['event_count']} | {recall} |")
            undefined = [f"{LABELS[m]}：{json.dumps(info['scores'][m][metric]['spearman_undefined_reasons'],ensure_ascii=False)}"
                         for m in METHODS if info["scores"][m][metric]["spearman_undefined_events"]]
            if undefined:
                lines += ["", "未定义原因：" + "；".join(undefined) + "。"]
    lines += ["", "## 如何解释", "",
              "- MAE 先在每个事件的评分资产内平均，再对事件等权平均；1 bps = 0.0001。所有方法的 MAE 与 Top3 使用同一共同配对事件集。",
              "- 系统性事件评十个资产；非系统性事件排除所有同分钟起源资产。十资产分别使用各自历史输入，不互相传入冲击。",
              "- Spearman 使用平均并列名次；恒定向量为 N/A，不记作零。各方法 defined n 可能不同，不能把这些相关系数当作完全相同样本上的胜负证据。",
              "- Top3 是处理并列后得到的期望召回，不是三只全对率。全平预测的期望值为 3/N；非系统性组的不传播基线通常正是这种情形。",
              "- 最大回撤沿当前价格与未来 30 个 close 计算；波动率为 30 个对数收益率的总体标准差乘 √30，未年化。",
              "- 仅一月、每资产一条随机路径、固定 seed；这是探索性结果，不是独立留出集表现、校准不确定性或稳定收益证明。",
              "- systemic 标签要到 t0+10 才能确认，在 t0 不可知；它只用于事后分组，不能声称当时已经判断出系统性传播。",
              "- 原 k∈{3,4,5,6} 的 20–60 事件门槛均未通过，自动 selected_k 保持 null。随后用户明确授权 k=6 全部 84 次；没有截取、替换或按结果挑事件。", "",
              "## 三个固定回放", "",
              "回放在推理前按触发元数据选定：最强系统性、最强非系统性、最接近全部 84 次事件初始冲击强度中位数的剩余事件。失败样例照样保留，不另挑成功样例。", "",
              "| 预选理由 | UTC 事件时间 | 状态 | 回放 |", "|---|---|---|---|"]
    for choice, path in zip(selection["replays_selected_before_inference"], replay_paths):
        artifact = evidence["artifacts"][choice["event_id"]]
        lines.append(f"| {choice['reason']} | {artifact['event']['timestamp']} | {artifact['score']['status']} | {_link('JSON',path)} |")
    lines += ["", "每份回放含十资产 t0−30…t0+30 的 61 个真实 close、当时可见 31 点与未来真值 30 点的明确分隔、四方法路径/风险/排名、完整评分及数据/模型身份和原审计字段。", "",
              "## 可复核证据", "",
              "报告从已完成批次读取并重新核对：本地一月真值、保存路径的确定性风险、逐事件评分、共同分母汇总及 forecast_id。没有再次推理。", ""]
    timings = summary.get("per_event_seconds", [])
    if timings:
        lines += [f"完整批次耗时 {summary['end_to_end_seconds']:.2f} 秒；"
                  f"单事件 min / median / max 为 {min(timings):.2f} / {median(timings):.2f} / {max(timings):.2f} 秒"
                  f"（{len(timings)} 次）。包含本次执行过程的端到端开销，非独占设备性能基准。", ""]
    lines += ["| 保存的质量审计 | 有审计事件 / 全部事件 | 总根数 | 涉及根数 |", "|---|---:|---:|---:|"]
    for field, count_field, label in (("ohlc_corrections", "corrected_candles", "OHLC containment 校正"),
                                       ("volume_quality", "volume_invalid_count", "有限负 volume / amount")):
        audits = [(artifact.get("forecast") or {}).get(field) for artifact in evidence["artifacts"].values()]
        present = [audit for audit in audits if isinstance(audit, dict)
                   and isinstance(audit.get("total_candles"), int) and isinstance(audit.get(count_field), int)]
        lines.append(f"| {label} | {len(present)} / {summary['event_count']} | "
                     f"{sum(a['total_candles'] for a in present)} | {sum(a[count_field] for a in present)} |")
    lines += ["", "审计计数只描述保存输出的结构质量，不用于删除事件或证明预测更准；有限负成交量字段不改变价格指标。", "",
              "- " + _link("完整数值汇总", evidence["batch_dir"] / "summary.json"),
              "- " + _link("运行配置与数据/模型/代码身份", evidence["batch_dir"] / "run_manifest.json"),
              "- " + _link("全部 84 次授权与固定回放选择", ARTIFACTS / "selection-approved-k6.json"),
              "- " + _link("回放中位数选择口径", ARTIFACTS / "replay-selection-policy.json"),
              "- " + _link("冻结定义", ROOT / "research/shock_radar/DEFINITION.md"),
              "", f"生成时间：{datetime.now(timezone.utc).isoformat()}。"]
    return "\n".join(lines) + "\n"


def generate(batch_dir=ARTIFACTS / "inference-original-v1", report_path=ROOT / "research/shock_radar/REPORT.md",
             replays_dir=ARTIFACTS / "replays-v1"):
    report_path, replays_dir = Path(report_path).resolve(), Path(replays_dir).resolve()
    if report_path.exists() or replays_dir.exists():
        raise FileExistsError("Report and replay directory must be new; existing evidence is never overwritten")
    evidence = load_evidence(batch_dir)
    choices = evidence["selection"]["replays_selected_before_inference"]
    replay_paths = [replays_dir / (choice["event_id"] + ".json") for choice in choices]
    replays = [build_replay(evidence, choice) for choice in choices]
    report = render_report(evidence, replay_paths)
    # Validate serializability before publishing any artifact.
    for replay in replays:
        json.dumps(replay, allow_nan=False)
    replays_dir.mkdir(parents=True, exist_ok=False)
    for path, replay in zip(replay_paths, replays):
        write_json(path, replay)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(report)
    return {"report": str(report_path), "replays": [str(p) for p in replay_paths],
            "events": evidence["summary"]["event_count"],
            "paired_events": evidence["summary"]["groups"]["all"]["common_paired_event_count"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=ARTIFACTS / "inference-original-v1")
    parser.add_argument("--report", type=Path, default=ROOT / "research/shock_radar/REPORT.md")
    parser.add_argument("--replays-dir", type=Path, default=ARTIFACTS / "replays-v1")
    args = parser.parse_args()
    print(json.dumps(generate(args.batch_dir, args.report, args.replays_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
