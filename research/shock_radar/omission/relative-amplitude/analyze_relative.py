"""Read one frozen CSV; compute relative omission quantiles without event/model access."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = OUT.parent / "missed_assets.csv"
SOURCE_SHA = "5d474e9904f01987b34619d2c1e74515a0a74b74432ec8dd83eddb4514add78e"


def quantile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA, "Frozen CSV changed"
    groups = {"volatility": [], "max_drawdown": []}
    keys = set()
    for source_row in csv.DictReader(io.StringIO(raw.decode())):
        row = dict(source_row)
        key = (row["event_id"], row["metric"], row["asset"])
        assert key not in keys, "Duplicate metric-asset-event record"
        keys.add(key)
        for field in ("weakest_selected_actual", "actual", "excess_bps"):
            row[field] = float(row[field])
            assert math.isfinite(row[field]), "Nonfinite input"
        low, actual = row["weakest_selected_actual"], row["actual"]
        assert actual > low >= 0, "Input must be a strictly overtaking record"
        assert math.isclose((actual - low) * 10000, row["excess_bps"], rel_tol=1e-12, abs_tol=1e-12)
        row["selected_assets"] = row["selected_assets"].split("|")
        row["ratio"] = (actual - low) / low if low else None
        groups[row["metric"]].append(row)

    metrics, cases = [], []
    for metric, label in (("volatility", "波动"), ("max_drawdown", "回撤")):
        group = groups[metric]
        finite = [r for r in group if r["ratio"] is not None]
        values = [r["ratio"] for r in finite]
        summary = {"key": metric, "label": label, "n": len(group), "finite_n": len(finite),
                   "zero_denominator_count": len(group) - len(finite),
                   **{name: quantile(values, q) for name, q in (("p50", .5), ("p75", .75), ("p90", .9))}}
        metrics.append(summary)
        if finite:
            case = min(finite, key=lambda r: (abs(r["ratio"] - summary["p90"]), r["timestamp"], r["event_id"], r["asset"]))
            cases.append({**case, "label": label, "selection": "接近 p90 的真实记录"})
    largest = min(groups["max_drawdown"], key=lambda r: (-r["excess_bps"], r["timestamp"], r["event_id"], r["asset"]))
    cases.append({**largest, "label": "回撤", "selection": "回撤绝对超出幅度最大的记录"})
    result = {
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA},
        "formula": "(actual - weakest_selected_actual) / weakest_selected_actual",
        "quantile_method": "linear interpolation at (n-1)*q; unrounded ratios; zero denominators reported separately",
        "metrics": metrics, "cases": cases,
        "conclusion": "两项中位数都超过10%；不能把超车概括为边缘差异。回撤p90的相对超出量超过100%，尾部存在明显遗漏。",
        "population_note": "仅对已经超车的资产×事件×指标记录等权统计，不是全部事件的分布。",
        "case_selection_note": "案例按已实现结果选取，用于解释幅度，不作为独立验证样本；近p90记录不等于插值得到的精确分位点。",
        "publication_status": "requires_user_confirmation_before_pitch",
    }
    assert SOURCE.read_bytes() == raw, "Source bytes changed during analysis"
    body = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for path in (OUT / "result.json", ROOT / "ripple_web/data/omission.json"):
        with path.open("x") as f:
            f.write(body)
    print(json.dumps({"metrics": metrics, "case_count": len(cases), "source_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
