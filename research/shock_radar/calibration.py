"""Freeze the original four-k calibration and every catalog before inference."""
from datetime import datetime, timezone
import csv
from pathlib import Path

from .detection import calibrate
from .io import ROOT, digest, load_january, write_json

OUTPUT = ROOT / "research/shock_radar/artifacts/january-v1"


def write_catalog(path, events):
    fields = ["timestamp", "origin_asset", "direction", "magnitude_sigma", "systemic_flag",
              "event_id", "index", "systemic_confirmed_at", "eligible", "exclusion_reasons"]
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = {key: event.get(key) for key in fields}
            row["origin_asset"] = "|".join(event["origin_assets"])
            row["exclusion_reasons"] = "|".join(event["exclusion_reasons"])
            writer.writerow(row)


def main():
    if OUTPUT.exists():
        raise FileExistsError("calibration output already exists; original evidence cannot be replaced")
    data = load_january()
    result = calibrate(data["closes"], data["times"], data["assets"])
    OUTPUT.mkdir(parents=True)
    contract = (ROOT / "research/shock_radar/CONTRACT.md").read_bytes()
    with (OUTPUT / "definitions_at_calibration.md").open("xb") as stream:
        stream.write(contract)
    files = {}
    for key, catalog in result["catalogs"].items():
        write_json(OUTPUT / f"catalog-k{key}.json", catalog)
        write_catalog(OUTPUT / f"catalog-k{key}-all.csv", catalog["events"])
        write_catalog(OUTPUT / f"catalog-k{key}-eligible.csv", [e for e in catalog["events"] if e["eligible"]])
        p = OUTPUT / f"catalog-k{key}.json"
        files[key] = {"json": p.name, "sha256": digest(p.read_bytes())}
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "ready_for_inference" if result["selected_k"] is not None else "no_k_meets_frozen_count_range",
               "selected_k": result["selected_k"], "selection_rule": result["selection_rule"],
               "target_count": [20, 60], "grid": result["grid"], "assets": data["assets"],
               "provenance": data["provenance"], "model_calls": 0, "catalogs": files,
               "definition_sha256": digest(contract),
               "code_sha256": {name: digest((ROOT / "research/shock_radar" / name).read_bytes())
                                for name in ("io.py", "detection.py", "calibration.py")},
               "disclosure": "Count-only January exploration; no transmission outcomes used for k selection. Edge exclusions are based solely on fixed input/target timestamps. Systemic is a posthoc label, not online knowledge."}
    write_json(OUTPUT / "calibration.json", summary)
    rows = ["# 冲击目录与 k 标定：2026-01", "", "| k | 原始候选 | 冷却后触发 | 合并事件 | 可评测事件 | 时间边界排除 | 系统性事件（可评测） |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["grid"]:
        systemic = sum(e["eligible"] and e["systemic_flag"] is True for e in result["catalogs"][str(row["k"])]["events"])
        rows.append(f"| {row['k']} | {row['raw_candidate_count']} | {row['retained_trigger_count']} | {row['merged_event_count']} | {row['eligible_event_count']} | {row['excluded_event_count']} | {systemic} |")
    rows += ["", f"原冻结网格自动选择：{result['selected_k']}。目标为 20–60 个可评测合并事件；没有按预测结果筛选或截取样本。", "",
             "四档目录均已保存：每档有全部事件 CSV、可评测事件 CSV、含逐资产触发与归因元数据的 JSON。origin_asset 中 | 表示同一分钟共同最早触发，不能据此声称因果源头。", "",
             "输入固定 256 分钟，目标未来 30 分钟；仅因月初历史不足或月末未来不足排除，原始事件仍保留。sigma=0 计数四档均为 0。", "",
             "原阈值门未通过时，不自动推理、不扩网格、不凑样本。等待用户明确调整；校准证据原样保留。2 月未读，3 月继续密封。", ""]
    with (OUTPUT / "calibration_report.md").open("x") as stream:
        stream.write("\n".join(rows))
    print({"status": summary["status"], "selected_k": summary["selected_k"],
           "counts": {row["k"]: row["eligible_event_count"] for row in result["grid"]}})


if __name__ == "__main__":
    main()
