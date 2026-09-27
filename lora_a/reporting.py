"""Readable, tier-separated paired reports. No model or market-data access."""
from pathlib import Path


def fmt(value, percent=False):
    if value is None:
        return "N/A"
    return f"{100*value:.3f}%" if percent else f"{value:.7g}"


def interval_text(item, percentage=False):
    interval = item.get("ci95")
    if not interval:
        return "N/A"
    # The metrics module uses [lower, upper] percentile endpoints.
    if isinstance(interval, dict):
        interval = [interval.get("lower"), interval.get("upper")]
    if percentage:
        return f"[{interval[0]*100:+.3f}pp, {interval[1]*100:+.3f}pp]"
    return f"[{fmt(interval[0])}, {fmt(interval[1])}]"


def observation_metrics(report, candidate):
    """Read-only observation record; no access to checkpoint/gate decisions."""
    result = {"candidate": candidate, "additional_metrics_decision_use": "none",
              "volatility_mae_role": "February small-tier MAE still selects checkpoints; January observation MAE never does", "tiers": {}}
    for tier, section in report["tiers"].items():
        row = section["methods"][candidate]
        original = section["methods"].get("original", {})
        error = row.get("max_drawdown_mae")
        baseline = original.get("max_drawdown_mae")
        paired_n = row["n_scored"] if candidate == "original" else 0
        if candidate != "original":
            paired = section.get("paired", {}).get(candidate, {}).get("original", {}).get("max_drawdown_mae_difference")
            if paired is not None:
                error, baseline = paired.get("candidate_mean"), paired.get("reference_mean")
                paired_n = paired.get("n_paired", 0)
            else:
                error = baseline = None
        relative = error / baseline - 1 if error is not None and baseline is not None and baseline > 0 else None
        result["tiers"][tier] = {
            "n": row["n_scored"], "n_scheduled": row["n_scheduled"], "failed": row["n_failed"],
            "max_drawdown_mae": error, "original_max_drawdown_mae": baseline,
            "n_drawdown_paired": paired_n,
            "relative_drawdown_mae_change": relative,
            "drawdown_mae_absolute_change": error - baseline if error is not None and baseline is not None else None,
            "drawdown_worse_by_20pct_or_more": error is not None and baseline is not None and baseline > 0 and error >= baseline * 1.2,
            "relative_change_note": "original MAE is zero or unavailable" if relative is None else None,
            "volatility_mae": row["volatility_mae"], "volatility_top3_recall": row.get("volatility_top3_recall"),
        }
    return result


def markdown_report(path, report, candidate, gate_result=None, *, note=""):
    lines = [f"# {report['month']} A 线 v2.1 报告", "", f"候选：{candidate}。原版与候选共用固定起点、每资产1路径和采样种子。", ""]
    if report.get("fixed_pair"):
        lines.extend(["冻结固定组合：" + " + ".join(report["fixed_pair"]) + "（只由01-01至25确定）。", ""])
    if note:
        lines.extend([note, ""])
    if gate_result is not None:
        conclusion = "待泄露复核" if gate_result.get("acceptance_status") == "pending_leakage_review" else ("通过" if gate_result["passed"] else "不过")
        lines.extend(["验收结论：**" + conclusion + "**。", ""])
        lines.extend(["- " + reason for reason in gate_result.get("reasons", [])])
        lines.append("")
    lines.extend(["| 档位 | 方法 | 完成/计划起点 | 精确集合命中 | 平均命中只数 | 波动率 MAE | 失败数 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for tier, section in report["tiers"].items():
        for method, row in section["methods"].items():
            lines.append(f"| {tier} | {method} | {row['n_scored']}/{row['n_scheduled']} | {fmt(row['exact_set_hit_rate'],True)} | {fmt(row['mean_hit_count'])} | {fmt(row['volatility_mae'])} | {row['n_failed']} |")
    lines.extend(["", "| 档位 | 候选减对照 | 配对 n | 集合命中差（百分点）及95% CI | 平均命中只数差及95% CI | MAE差及95% CI |",
                  "|---|---|---:|---:|---:|---:|"])
    for tier, section in report["tiers"].items():
        for baseline, pair in section.get("paired", {}).get(candidate, {}).items():
            hit = pair["exact_set_hit_rate_difference"]
            softer = pair["mean_hit_count_difference"]
            mae = pair["volatility_mae_difference"]
            def delta(item, pct=False):
                value = item.get("estimate")
                text = ("N/A" if value is None else f"{value*100:+.3f}pp") if pct else fmt(value)
                return text + " " + interval_text(item, pct)
            lines.append(f"| {tier} | {candidate} − {baseline} | {pair['n_paired']} | {delta(hit,True)} | {delta(softer)} | {delta(mae)} |")
    if any(row.get("max_drawdown_mae") is not None for section in report["tiers"].values() for row in section["methods"].values()):
        lines.extend(["", "## 只观察：回撤与Top3诊断", "", "新增回撤/Top3指标和告警不参与loss、早停、选模或验收；02小币波动率MAE仍是选模指标，01观察集全部只记录。", "",
                      "| 档位 | 方法 | 回撤MAE | 波动率MAE | 波动Top3期望召回 |", "|---|---|---:|---:|---:|"])
        for tier, section in report["tiers"].items():
            for method, row in section["methods"].items():
                lines.append(f"| {tier} | {method} | {fmt(row.get('max_drawdown_mae'))} | {fmt(row['volatility_mae'])} | {fmt(row.get('volatility_top3_recall'),True)} |")
        if candidate in report["tiers"]["small"]["methods"]:
            observations = observation_metrics(report, candidate)
            for tier, item in observations["tiers"].items():
                if item["drawdown_worse_by_20pct_or_more"]:
                    lines.extend(["", f"观察告警：{tier} 回撤MAE比原版高 {item['relative_drawdown_mae_change']*100:.2f}%；仅记录，不自动停训或调参。"])
                elif item["relative_change_note"] and item["max_drawdown_mae"] is not None:
                    lines.extend(["", f"{tier} 回撤MAE相对变化未定义（原版为0或不可用），绝对变化 {fmt(item['drawdown_mae_absolute_change'])}。"])
    lines.extend(["", "命中差越大越好，MAE差越小越好；区间按UTC日起点分块bootstrap，仅供参考，不是通过门槛。单月、单路径结果不证明长期泛化或统计非劣。", "",
                  "小币门：点估计严格超过动态、冻结固定对和原版。大币门：点估计不低于动态−2pp。完整配对不足不能通过。", "",
                  "并列沿用competition rank≤2，可能超选；以下分别披露预测/实际超选窗口频率：", ""])
    for tier, section in report["tiers"].items():
        for method, row in section["methods"].items():
            n = row["n_scored"]
            lines.append(f"- {tier}/{method}：预测超选 {row['n_selected_over_two']}/{n}；实际超选 {row['n_actual_over_two']}/{n}。")
    with Path(path).open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def acceptance_summary(path, report, candidate, verdict):
    conclusion = "待泄露复核" if verdict.get("acceptance_status") == "pending_leakage_review" else ("通过" if verdict["passed"] else "不过")
    lines = ["# 03 密封验收：" + conclusion, "", f"唯一候选：{candidate}。预先冻结300个等距起点，原版与候选同路径预算、同种子。", "",
             "| 档位 | 候选命中率 | 动态naive | 固定组合 | 原版 | 候选失败/计划 |", "|---|---:|---:|---:|---:|---:|"]
    for tier, section in report["tiers"].items():
        methods = section["methods"]
        row = methods[candidate]
        rate = lambda name: fmt(methods[name]["exact_set_hit_rate"], True) if name in methods else "—"
        lines.append(f"| {tier} | {rate(candidate)} | {rate('dynamic_naive')} | {rate('fixed_pair')} | {rate('original')} | {row['n_failed']}/{row['n_scheduled']} |")
    lines.extend(["", "固定组合：" + " + ".join(report.get("fixed_pair", [])) + "。小币须严格胜三个对照；大币允许较动态低至2pp。", ""])
    for key, item in verdict["checks"].items():
        lines.append(f"- {key}：{'通过' if item['passed'] else '不过'}；配对n={item.get('n_paired',0)}，命中差 {fmt(item.get('exact_set_hit_rate_difference'),True)}。")
    if verdict.get("acceptance_status") == "pending_leakage_review":
        lines.extend(["", "异常高分触发泄露复核；在窗口、时间与月份隔离检查完成前不认定通过。"])
    lines.extend(["", "配对置信区间、MAE与失败/并列披露见 [完整报告](sealed-report.md)；CI不作为额外门槛。单月单路径结果，不宣称长期泛化。", "",
                  "验收后不再训练、选模或调整参数。"])
    with Path(path).open("x") as stream:
        stream.write("\n".join(lines) + "\n")
