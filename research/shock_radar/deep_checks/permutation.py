"""Prespecified January temporal null; frozen trigger metadata only, no inference.

The null preserves each asset's cyclic trigger pattern, not common market regimes.
The graph code receives new synthetic episodes, never price data or model outputs.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import inf, isfinite
from pathlib import Path
from statistics import mean

import numpy as np

from research.shock_radar.detection import _merge
from research.shock_radar.closure.graph_fingerprints import ASSETS, build_graph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CATALOG = ROOT / "artifacts/january-v1/catalog-k6.json"
GRAPH = ROOT / "closure/artifacts/january-graph-v1/graph.json"
PERIOD = 44640
SEED = 20260927
REPLICATES = 200
EPOCH = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
PARSED = [EPOCH + timedelta(minutes=i) for i in range(PERIOD)]
TIMES = [t.isoformat().replace("+00:00", "Z") for t in PARSED]


def hashes():
    inputs = [CATALOG, GRAPH, ROOT / "detection.py",
              ROOT / "closure/graph_fingerprints.py", HERE / "CONTRACT.md", Path(__file__)]
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in inputs}


def shift_triggers(triggers, offsets):
    """A fixed cyclic shift per asset preserves all its marks and cyclic gaps."""
    if set(offsets) != set(ASSETS):
        raise ValueError("Exactly ten asset offsets are required")
    if any(type(v) is not int or not 0 <= v < PERIOD for v in offsets.values()):
        raise ValueError("Offsets must be integer minutes in [0,44639]")
    result = []
    seen = set()
    for trigger in triggers:
        asset, index = trigger["asset"], trigger["index"]
        if asset not in offsets or type(index) is not int or not 0 <= index < PERIOD:
            raise ValueError("Invalid frozen trigger asset/index")
        if (asset, index) in seen or trigger["timestamp"] != TIMES[index]:
            raise ValueError("Duplicate or misaligned frozen trigger")
        seen.add((asset, index))
        shifted = (index + offsets[asset]) % PERIOD
        result.append({**trigger, "index": shifted, "timestamp": TIMES[shifted]})
    order = {a: i for i, a in enumerate(ASSETS)}
    return sorted(result, key=lambda t: (t["index"], order[t["asset"]]))


def reconstruct(triggers):
    """Use the unchanged frozen episode merger and graph constructor."""
    events = _merge(triggers, TIMES, PARSED, 6)
    eligible = [e for e in events if e["eligible"]]
    _, graph, _ = build_graph(eligible, triggers)
    return events, graph


def statistics(events, graph):
    delays = [o["delay_minutes"] for edge in graph["edges"] for o in edge["observations"]]
    return {
        "nonzero_directed_pair_count": len(graph["edges"]),
        "observation_count": len(delays),
        "observation_mean_delay_minutes": mean(delays) if delays else None,
        "edge_unweighted_mean_delay_minutes": mean(e["mean_delay_minutes"] for e in graph["edges"]) if delays else None,
        "merged_event_count": len(events),
        "eligible_event_count": graph["event_count"],
        "excluded_event_count": sum(not e["eligible"] for e in events),
        "source_event_count": sum(n["source_event_count"] for n in graph["nodes"]),
    }


def tail_test(real, null, tail):
    """Plus-one Monte Carlo test including ties; missing delays use +infinity."""
    if tail not in ("upper", "lower") or not null or real is None or not isfinite(real):
        raise ValueError("A finite real statistic, nonempty null and valid tail are required")
    values = [inf if x is None else x for x in null]
    if any(not isfinite(x) and x != inf for x in values):
        raise ValueError("Invalid null statistic")
    extreme = sum(x >= real if tail == "upper" else x <= real for x in values)
    finite = [x for x in values if isfinite(x)]
    q = np.quantile(finite, [.05, .5, .95], method="linear").tolist() if finite else [None] * 3
    return {
        "real": real, "tail": tail, "replicates": len(values),
        "as_or_more_extreme_count": extreme,
        "monte_carlo_p": (1 + extreme) / (1 + len(values)),
        "null_less_than_real_count": sum(x < real for x in values),
        "null_equal_real_count": sum(x == real for x in values),
        "null_greater_than_real_count": sum(x > real for x in values),
        "empirical_cdf_at_real": sum(x <= real for x in values) / len(values),
        "null_mean_finite": mean(finite) if finite else None,
        "null_min_finite": min(finite) if finite else None,
        "null_p05_p50_p95_finite": q,
        "null_max_finite": max(finite) if finite else None,
        "null_finite_count": len(finite), "null_empty_count": len(values) - len(finite),
    }


def holm_adjust(pvalues):
    """Holm step-down adjusted p values in original name order."""
    if not pvalues or any(not isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("Nonempty probabilities required")
    ordered = sorted(pvalues, key=lambda name: (pvalues[name], name))
    adjusted, running = {}, 0.0
    for i, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * pvalues[name]))
        adjusted[name] = running
    return {name: adjusted[name] for name in pvalues}


def main():
    before = hashes()
    catalog, frozen = json.loads(CATALOG.read_text()), json.loads(GRAPH.read_text())
    if catalog["k"] != 6 or len(catalog["triggers"]) != 216:
        raise ValueError("Unexpected frozen January catalog")
    if frozen["provenance"]["catalog_sha256"] != before[str(CATALOG.relative_to(ROOT))]:
        raise ValueError("Graph/catalog provenance mismatch")
    zero = shift_triggers(catalog["triggers"], dict.fromkeys(ASSETS, 0))
    events, graph = reconstruct(zero)
    if events != catalog["events"]:
        raise ValueError("Zero shift must exactly reproduce all original merged events")
    if graph != {k: v for k, v in frozen.items() if k != "provenance"}:
        raise ValueError("Zero shift must reproduce every frozen graph field")
    real = statistics(events, graph)
    assert (real["eligible_event_count"], real["nonzero_directed_pair_count"], real["observation_count"]) == (84, 68, 120)
    rng = np.random.default_rng(SEED)
    replicates = []
    for i in range(REPLICATES):
        offsets = {a: int(x) for a, x in zip(ASSETS, rng.integers(0, PERIOD, size=len(ASSETS)))}
        shifted = shift_triggers(catalog["triggers"], offsets)
        synthetic_events, synthetic_graph = reconstruct(shifted)
        replicates.append({"replicate": i + 1, "synthetic_offsets_minutes": offsets,
                           **statistics(synthetic_events, synthetic_graph)})
    tests = {}
    for name, tail in [("nonzero_directed_pair_count", "upper"), ("observation_mean_delay_minutes", "lower")]:
        tests[name] = tail_test(real[name], [r[name] for r in replicates], tail)
    adjusted = holm_adjust({name: row["monte_carlo_p"] for name, row in tests.items()})
    for name, row in tests.items():
        row.update(holm_adjusted_p=adjusted[name], significant_holm_alpha_05=adjusted[name] <= .05)
    diagnostic = {}
    for key in real:
        vals = [row[key] for row in replicates if row[key] is not None]
        diagnostic[key] = {"min": min(vals) if vals else None, "median": float(np.median(vals)) if vals else None,
                           "max": max(vals) if vals else None, "n_nonempty": len(vals)}
    after = hashes()
    if before != after:
        raise ValueError("Inputs or source code changed during deterministic analysis")
    result = {
        "analysis": "Prespecified January independent per-asset circular-shift null",
        "seed": SEED, "replicate_count": REPLICATES, "period_minutes": PERIOD,
        "timestamp_origin": TIMES[0], "rng": "numpy.random.default_rng / PCG64",
        "numpy_version": np.__version__, "alpha": .05,
        "multiple_testing": "Holm correction over exactly two primary endpoints",
        "null": "Uniform integer offset [0,44639] per asset independently; preserve retained-trigger marks and cyclic gaps; reconstruct original fixed-anchor 10-minute episodes and eligibility; original graph rule",
        "zero_shift_exact_catalog_and_graph_match": True,
        "retained_trigger_count": len(zero), "retained_per_asset": dict(Counter(t["asset"] for t in zero)),
        "real": real, "tests": tests, "null_diagnostics": diagnostic,
        "replicates": replicates, "source_sha256_before": before, "source_sha256_after": after,
        "source_unchanged": True, "model_calls": 0, "market_data_reads": 0,
        "limitations": [
            "This is a conditional temporal-alignment test, not a price-generated market null; monthly shifts do not preserve common daily market regimes.",
            "Synthetic eligible/source-event counts vary because original episode merging is reapplied. The test is not conditioned on 84 episodes and estimates no per-edge causal effect.",
            "Graph observations may share target triggers and episodes; no independence of the 120 observations is asserted.",
            "200 replicates limit raw p resolution to 1/201; only two aggregate tests, no inference that every edge is stable, causal or predictive.",
            "Timeline rotation preserves cyclic gaps, while eligibility and graph follow-ups use linear month boundaries exactly as in the original graph.",
        ],
        "conclusion": ("跨资产时间聚集显著超过该循环平移零模型；不能据此声称因果传染、每条边稳定或具有预测价值。"
                       if any(t["significant_holm_alpha_05"] for t in tests.values())
                       else "图谱可能是噪声，涟漪动画仅作示意。"),
    }
    output = HERE / "permutation.json"
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "real": real, "tests": tests, "null_diagnostics": diagnostic,
                      "conclusion": result["conclusion"]}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
