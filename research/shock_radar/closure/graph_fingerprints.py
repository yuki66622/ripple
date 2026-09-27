"""Frozen January temporal associations and prefix-censored nearest-event demo.

Only saved January metadata/predictions are read. No model, data reader, or
other-month imports. Temporal edges and similarities do not establish causality.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import isfinite, log1p
from numbers import Real
from pathlib import Path
from statistics import mean, median

ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
HERE = Path(__file__).resolve().parent
JANUARY = HERE.parent / "artifacts/january-v1"
OUTPUT = HERE / "artifacts/january-graph-v1"
EDGE_FIELDS = ["source", "target", "count", "mean_delay_minutes", "median_target_magnitude_sigma",
               "source_event_count", "followup_rate"]
METHOD = {
    "clustering": "average-linkage agglomerative; exactly min(4,N) clusters",
    "distance": "mean over ten assets: both absent=0; presence mismatch=1; both present=mean(direction mismatch, abs(log1p(z1)-log1p(z2))/(1+abs(log1p(z1)-log1p(z2))), abs(delay1-delay2)/30)",
    "ordering": "events ordered by timestamp,event_id; linkage ties by member index tuples; medoid ties earlier event",
    "retrieval": "compare query with every historical fingerprint censored at the same observed_minutes; no cluster filter",
    "interpretation": "Temporal association, not causal contagion. Similarities are not probabilities or forecasts.",
}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _minutes(value):
    value = _finite(value, "observed_minutes")
    if not 0 <= value <= 30:
        raise ValueError("observed_minutes must be within [0,30]")
    return value


def absent_slot():
    return {"present": False, "direction": None, "magnitude_sigma": None, "delay_minutes": None}


def validate_vector(vector, observed_minutes=30):
    observed_minutes = _minutes(observed_minutes)
    if (not isinstance(vector, dict) or set(vector) != {"assets", "slots"}
            or vector["assets"] != ASSETS or not isinstance(vector["slots"], dict)
            or set(vector["slots"]) != set(ASSETS)):
        raise ValueError("Fingerprint requires the exact ordered ten assets and ten slots")
    for asset in ASSETS:
        slot = vector["slots"][asset]
        if not isinstance(slot, dict) or set(slot) != set(absent_slot()) or type(slot["present"]) is not bool:
            raise ValueError("Each slot requires present/direction/magnitude_sigma/delay_minutes")
        if not slot["present"]:
            if slot != absent_slot():
                raise ValueError("Absent trigger values must be null, not invented measurements")
            continue
        if type(slot["direction"]) is not int or slot["direction"] not in (-1, 1):
            raise ValueError("Observed trigger direction must be integer +/-1")
        if _finite(slot["magnitude_sigma"], "magnitude_sigma") <= 0:
            raise ValueError("Observed trigger magnitude must be positive")
        delay = _finite(slot["delay_minutes"], "delay_minutes")
        if not 0 <= delay <= observed_minutes:
            raise ValueError("Fingerprint contains a trigger after the observed prefix")
    return vector


def censor(vector, observed_minutes):
    validate_vector(vector)
    observed_minutes = _minutes(observed_minutes)
    return {"assets": list(ASSETS), "slots": {
        a: dict(s) if s["present"] and s["delay_minutes"] <= observed_minutes else absent_slot()
        for a, s in vector["slots"].items()}}


def fingerprint_distance(left, right):
    validate_vector(left)
    validate_vector(right)
    values = []
    for a in ASSETS:
        x, y = left["slots"][a], right["slots"][a]
        if not x["present"] and not y["present"]:
            values.append(0.)
        elif x["present"] != y["present"]:
            values.append(1.)
        else:
            dz = abs(log1p(x["magnitude_sigma"]) - log1p(y["magnitude_sigma"]))
            values.append(mean((float(x["direction"] != y["direction"]), dz/(1+dz),
                                abs(x["delay_minutes"] - y["delay_minutes"])/30)))
    return mean(values)


def _at(timestamp, minutes):
    return (datetime.fromisoformat(timestamp.replace("Z", "+00:00")) + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def build_graph(events, triggers):
    """First retained target in (t0,t0+30], one observation per source-event/pair."""
    events = sorted(events, key=lambda e: (e["timestamp"], e["event_id"]))
    if len({e["event_id"] for e in events}) != len(events):
        raise ValueError("Duplicate selected event")
    by_asset = {a: [] for a in ASSETS}
    lookup = {}
    for trigger in triggers:
        a, t = trigger["asset"], trigger["index"]
        if a not in ASSETS or type(t) is not int or (a, t) in lookup:
            raise ValueError("Invalid or duplicate retained trigger")
        if type(trigger["direction"]) is not int or trigger["direction"] not in (-1, 1):
            raise ValueError("Invalid trigger sign")
        if _finite(trigger["magnitude_sigma"], "magnitude_sigma") <= 0:
            raise ValueError("Invalid trigger magnitude")
        lookup[a, t] = trigger
        by_asset[a].append(trigger)
    for sequence in by_asset.values():
        sequence.sort(key=lambda t: t["index"])
    positions = {a: [t["index"] for t in by_asset[a]] for a in ASSETS}
    opportunities = Counter()
    observations = {(a,b): [] for a in ASSETS for b in ASSETS}
    timelines, fingerprints = [], []
    for event in events:
        t0, origins = event["index"], event["origin_assets"]
        actual_origins = [a for a in ASSETS if (a,t0) in lookup]
        if not event.get("eligible") or set(origins) != set(actual_origins) or not origins:
            raise ValueError("Selected origins must match all retained simultaneous triggers")
        if len(origins) != len(set(origins)):
            raise ValueError("Duplicate origin")
        slots = {a: absent_slot() for a in ASSETS}
        selected = {a: lookup[a,t0] for a in origins}
        followups = {}
        for a in ASSETS:
            if a in origins:
                continue
            j = bisect_right(positions[a], t0)
            if j < len(by_asset[a]) and by_asset[a][j]["index"] <= t0 + 30:
                followups[a] = by_asset[a][j]
        selected.update(followups)
        grouped = {}
        for a, trigger in selected.items():
            delay = trigger["index"] - t0
            if trigger["timestamp"] != _at(event["timestamp"], delay):
                raise ValueError("Trigger timestamp/index mismatch")
            slots[a] = {"present": True, "direction": trigger["direction"],
                        "magnitude_sigma": trigger["magnitude_sigma"], "delay_minutes": delay}
            grouped.setdefault(delay, []).append({"asset": a, **slots[a]})
        vector = {"assets": list(ASSETS), "slots": slots}
        validate_vector(vector)
        for origin in origins:
            opportunities[origin] += 1
            for target, trigger in followups.items():
                observations[origin,target].append({
                    "event_id": event["event_id"], "event_timestamp": event["timestamp"],
                    "origin_assets": list(origins), "source_direction": lookup[origin,t0]["direction"],
                    "source_magnitude_sigma": lookup[origin,t0]["magnitude_sigma"],
                    "target_timestamp": trigger["timestamp"], "target_direction": trigger["direction"],
                    "target_magnitude_sigma": trigger["magnitude_sigma"], "delay_minutes": trigger["index"]-t0,
                })
        timeline = {"event_id": event["event_id"], "timestamp": event["timestamp"],
                    "origin_assets": list(origins), "initial_magnitude_sigma": event["magnitude_sigma"],
                    "distinct_followup_asset_count": len(followups),
                    "sequence": [{"delay_minutes": delay, "timestamp": _at(event["timestamp"], delay),
                                  "triggers": sorted(items, key=lambda item: ASSETS.index(item["asset"]))}
                                 for delay, items in sorted(grouped.items())]}
        timelines.append(timeline)
        fingerprints.append({"event_id": event["event_id"], "timestamp": event["timestamp"],
                             "origin_assets": list(origins), "followup_count": len(followups),
                             "classification_available_at": event.get("systemic_confirmed_at"),
                             "full_fingerprint_available_at": _at(event["timestamp"],30), "fingerprint": vector})
    rows, edges = [], []
    for source in ASSETS:
        for target in ASSETS:
            obs = observations[source,target]
            row = {"source": source, "target": target, "count": len(obs),
                   "mean_delay_minutes": mean(o["delay_minutes"] for o in obs) if obs else None,
                   "median_target_magnitude_sigma": median(o["target_magnitude_sigma"] for o in obs) if obs else None,
                   "source_event_count": opportunities[source],
                   "followup_rate": len(obs)/opportunities[source] if source != target and opportunities[source] else None}
            rows.append(row)
            if obs:
                edges.append({**row, "weight": len(obs), "observations": obs})
    strongest = sorted((t for t in timelines if t["distinct_followup_asset_count"]),
                       key=lambda t: (-t["distinct_followup_asset_count"], -t["initial_magnitude_sigma"], t["timestamp"], t["event_id"]))[:3]
    graph = {"nodes": [{"asset": a,"source_event_count": opportunities[a]} for a in ASSETS],
             "event_count": len(events), "edges": edges, "event_timelines": timelines,
             "strongest_three_sequences": strongest, "strongest_available_count": len(strongest),
             "interpretation": "Observed temporal association only; tied-time assets stay a set. Sequences do not prove hop-by-hop causation. Overlapping event windows can share triggers."}
    return rows, graph, fingerprints


def cluster_fingerprints(fingerprints):
    items = sorted(fingerprints, key=lambda x: (x["timestamp"],x["event_id"]))
    n, target = len(items), min(4,len(items))
    if len({x["event_id"] for x in items}) != n:
        raise ValueError("Duplicate fingerprint event")
    distances = {(i,j): fingerprint_distance(items[i]["fingerprint"],items[j]["fingerprint"])
                 for i in range(n) for j in range(i+1,n)}
    def dist(i,j):
        return 0. if i == j else distances[min(i,j),max(i,j)]
    active = {i: (i,) for i in range(n)}
    links = dict(distances)
    next_id, merges = n, []
    while len(active) > target:
        ids = sorted(active, key=lambda key: active[key])
        _, _, _, a, b = min((links[min(a,b),max(a,b)],active[a],active[b],a,b)
                            for p,a in enumerate(ids) for b in ids[p+1:])
        members = tuple(sorted(active[a]+active[b]))
        merges.append({"members": [items[i]["event_id"] for i in members], "distance": links[min(a,b),max(a,b)]})
        for other in ids:
            if other in (a,b):
                continue
            value = (len(active[a])*links[min(a,other),max(a,other)] + len(active[b])*links[min(b,other),max(b,other)]) / len(members)
            links[min(next_id,other),max(next_id,other)] = value
        del active[a], active[b]
        active[next_id] = members
        next_id += 1
    groups = sorted(active.values())
    silhouette = {}
    if 1 < len(groups) < n:
        for members in groups:
            for i in members:
                if len(members) == 1:
                    silhouette[i] = 0.
                    continue
                a = mean(dist(i,j) for j in members if j != i)
                b = min(mean(dist(i,j) for j in other) for other in groups if other != members)
                silhouette[i] = (b-a)/max(a,b) if max(a,b) else 0.
    clusters = []
    for ordinal,members in enumerate(groups,1):
        medoid = min(members, key=lambda i: (sum(dist(i,j) for j in members),i))
        counts = Counter(a for i in members for a in items[i]["origin_assets"])
        top_origins = [a for a in ASSETS if counts[a] == max(counts.values())]
        signs = Counter(s["direction"] for i in members for s in items[i]["fingerprint"]["slots"].values() if s["present"])
        sign = "mixed" if signs[1] == signs[-1] else "up" if signs[1] > signs[-1] else "down"
        breadth = median(items[i]["followup_count"] for i in members)
        name = "/".join(top_origins) + " 先触发 / " + {"mixed":"涨跌混合","up":"上涨","down":"下跌"}[sign] + f" / 后续广度中位数 {breadth:g}"
        clusters.append({"cluster_id": f"cluster_{ordinal}","name":name,"size":len(members),
                         "event_ids":[items[i]["event_id"] for i in members],"medoid_event_id":items[medoid]["event_id"],
                         "most_frequent_origins":top_origins,"origin_counts":dict(counts),
                         "majority_trigger_sign":sign,"trigger_sign_counts":{str(k):v for k,v in signs.items()},
                         "median_followup_count":breadth,
                         "silhouette":mean(silhouette[i] for i in members) if silhouette else None})
    return {"method":METHOD,"event_count":n,"cluster_count":len(clusters),"clusters":clusters,
            "silhouette":mean(silhouette.values()) if silhouette else None,
            "silhouette_reason":None if silhouette else "requires 1 < cluster_count < event_count",
            "merge_history":merges,"labels_available":"Full fingerprint and cluster labels require the complete t0+30 observation."}


def nearest_events(index, vector, observed_minutes, exclude_event_id=None, k=3):
    """Pure in-memory retrieval; historical future cannot influence the distance."""
    observed_minutes = _minutes(observed_minutes)
    validate_vector(vector,observed_minutes)
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(index,dict) or index.get("assets") != ASSETS or not isinstance(index.get("events"),list):
        raise ValueError("Index requires the frozen ten-asset universe and event records")
    seen, candidates = set(), []
    for event in index["events"]:
        event_id = event["event_id"]
        if event_id in seen:
            raise ValueError("Duplicate index event_id")
        seen.add(event_id)
        prefix = censor(event["fingerprint"],observed_minutes)
        if event_id == exclude_event_id:
            continue
        candidates.append({"distance":fingerprint_distance(vector,prefix),"event_id":event_id,
                           "timestamp":event["timestamp"],"origin_assets":event["origin_assets"],
                           "classification_available_at":event["classification_available_at"],
                           "observed_minutes":observed_minutes,"observed_prefix_fingerprint":prefix,
                           "full_historical_event":{**event["full_historical_event"],
                               "future_labelled":True,"not_used_for_distance":True},
                           "saved_risk":event["saved_risk"]})
    return sorted(candidates,key=lambda x:(x["distance"],x["timestamp"],x["event_id"]))[:k]


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def _write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def generate():
    if OUTPUT.exists() or (HERE / "FINGERPRINTS.md").exists():
        raise FileExistsError("Graph outputs already exist; frozen evidence will not be overwritten")
    selection, selection_hash = _read(JANUARY / "selection-approved-k6.json")
    catalog, catalog_hash = _read(JANUARY / "catalog-k6.json")
    if catalog_hash != selection["catalog_sha256"] or selection["selected_k"] != 6:
        raise ValueError("Frozen January catalog does not match approved selection")
    events = [e for e in catalog["events"] if e["eligible"]]
    if len(events) != 84 or [e["event_id"] for e in events] != selection["event_ids"]:
        raise ValueError("All and only the selected84 January events are required")
    rows, graph, fingerprints = build_graph(events,catalog["triggers"])
    clusters = cluster_fingerprints(fingerprints)
    cluster_by_id = {eid:c["cluster_id"] for c in clusters["clusters"] for eid in c["event_ids"]}
    event_map = {e["event_id"]:e for e in events}
    prediction_hashes, records = {}, []
    for fingerprint in fingerprints:
        eid = fingerprint["event_id"]
        path = JANUARY / "inference-original-v1" / (eid+".json")
        artifact, prediction_hashes[eid] = _read(path)
        if artifact["event"] != event_map[eid]:
            raise ValueError("Saved prediction/event identity mismatch")
        records.append({**fingerprint,"cluster_id":cluster_by_id[eid],
                        "full_historical_event":{"event":event_map[eid],"fingerprint":fingerprint["fingerprint"],
                            "cluster_id":cluster_by_id[eid],"full_fingerprint_available_at":fingerprint["full_fingerprint_available_at"]},
                        "saved_risk":{"forecast_id":artifact.get("forecast_id"),"predicted_risk":artifact["predicted_risk"],
                                      "actual_risk":artifact["actual_risk"],"score_status":artifact["score"]["status"],
                                      "source_artifact":str(path),"source_sha256":prediction_hashes[eid],
                                      "actual_risk_available_at":fingerprint["full_fingerprint_available_at"]}})
    provenance = {"catalog_sha256":catalog_hash,"selection_sha256":selection_hash,
                  "prediction_sha256":prediction_hashes,"source":"frozen January selected84 and all retained k6 triggers",
                  "generator_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    index = {"schema_version":1,"assets":ASSETS,"events":records,"clusters":clusters,"method":METHOD,
             "provenance":provenance,"interpretation":METHOD["interpretation"]}
    # Chronologically first event: deterministic demo choice, independent of outcomes.
    demo_event = records[0]
    query = {"vector":censor(demo_event["fingerprint"],10),"observed_minutes":10,
             "exclude_event_id":demo_event["event_id"],"k":3,
             "selection_reason":"chronologically first eligible January event; self excluded"}
    results = nearest_events(index,query["vector"],10,query["exclude_event_id"],3)
    # Verify sources are unchanged before publishing any derived output.
    if _read(JANUARY / "catalog-k6.json")[1] != catalog_hash or _read(JANUARY / "selection-approved-k6.json")[1] != selection_hash:
        raise ValueError("Source catalog/selection changed during generation")
    for eid,expected in prediction_hashes.items():
        if _read(JANUARY / "inference-original-v1" / (eid+".json"))[1] != expected:
            raise ValueError("Saved January prediction changed during generation")
    OUTPUT.mkdir(parents=True,exist_ok=False)
    with (OUTPUT / "edges.csv").open("x",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=EDGE_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    cluster_names = {c["cluster_id"]:c["name"] for c in clusters["clusters"]}
    with (OUTPUT / "event-classes.csv").open("x",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=["event_id","timestamp","cluster_id","cluster_name"])
        writer.writeheader()
        writer.writerows({"event_id":r["event_id"],"timestamp":r["timestamp"],
                          "cluster_id":r["cluster_id"],"cluster_name":cluster_names[r["cluster_id"]]}
                         for r in records)
    for name,value in (("graph.json",{**graph,"provenance":provenance}),
                       ("fingerprints.json",{"assets":ASSETS,"events":fingerprints,"clusters":clusters,"provenance":provenance}),
                       ("index.json",index),("demo-query.json",query),("demo-nearest.json",{
                           "query":query,"results":results,"interpretation":METHOD["interpretation"]})):
        _write(OUTPUT / name,value)
    write_document(graph,clusters,rows,results)
    return {"output":str(OUTPUT),"event_count":len(events),"edge_rows":len(rows),
            "positive_edges":len(graph["edges"]),"edge_observations":sum(r["count"] for r in rows),
            "cluster_sizes":[c["size"] for c in clusters["clusters"]],"silhouette":clusters["silhouette"],
            "demo_result_count":len(results),"source_hashes_unchanged":True}


def write_document(graph,clusters,rows,results):
    link = lambda name,label: f"[{label}](<{OUTPUT / name}>)"
    lines = ["# 一月冲击时序图与历史形态检索", "",
             f"已生成 {graph['event_count']} 个事件、100 个有序资产对、{len(graph['edges'])} 条非零时序关联边。"
             "边表示某资产先触发后，另一资产在 30 分钟内首次出现保留触发；不是因果传播。", "",
             "## 四类固定聚类", "", "| 形态名称 | 事件数 | 代表事件 UTC | silhouette |", "|---|---:|---|---:|"]
    timestamps = {e["event_id"]:e["timestamp"] for e in graph["event_timelines"]}
    for c in clusters["clusters"]:
        s = "N/A" if c["silhouette"] is None else f"{c['silhouette']:.4f}"
        lines.append(f"| {c['name']} | {c['size']} | {timestamps[c['medoid_event_id']]} | {s} |")
    lines += ["", f"整体 silhouette：{clusters['silhouette']}。四类由冻结距离与 average linkage 得到；未根据结果调参，类大小与分离质量不保证均衡或有效。",
              "名称只描述最常见先触发资产（并列全保留）、全部已观察触发的多数方向、后续资产数中位数。完整形态及类别要到 t0+30 才可得。", "",
              "## 最宽的三个观察序列", "", "| UTC 起点 | 后续不同资产数 | 同分钟触发集合 → 后续集合 |", "|---|---:|---|"]
    for timeline in graph["strongest_three_sequences"]:
        chain = " → ".join(f"+{p['delay_minutes']}m {{{','.join(t['asset'] for t in p['triggers'])}}}" for p in timeline["sequence"])
        lines.append(f"| {timeline['timestamp']} | {timeline['distinct_followup_asset_count']} | {chain} |")
    if not graph["strongest_three_sequences"]:
        lines.append("| N/A | 0 | 没有非空后续序列 |")
    lines.append("")
    for timeline in graph["strongest_three_sequences"]:
        later = [p["delay_minutes"] for p in timeline["sequence"] if p["delay_minutes"] > 0]
        lines.append(f"- {timeline['timestamp']}：{'、'.join(timeline['origin_assets'])} 同时先触发，"
                     f"第 {min(later)} 至 {max(later)} 分钟观察到 {timeline['distinct_followup_asset_count']} 个其他资产的首次后续触发；这是先后记录。")
    lines += ["", "同分钟起源是共同起源，不生成零延迟边；每个起源单独计一次 source-event opportunity。目标只取其首次保留触发，排除全部起源；可以跨原事件片段。重叠窗口可能复用同一触发，不能当独立证据。", "",
              "## Demo：匹配三个历史事件", "",
              "输入十资产指纹和已观察分钟数。历史指纹也裁剪到相同分钟，再在全部事件中检索，不按完整聚类过滤。输出距离、当时可见前缀、未来明确标记的历史完整事件及保存的风险结果。", "",
              "```python", "from research.shock_radar.closure.graph_fingerprints import nearest_events",
              "matches = nearest_events(index, vector, observed_minutes=10, exclude_event_id=event_id, k=3)", "```", "",
              "```bash", "scenario-lab/.venv/bin/python -m research.shock_radar.closure.graph_fingerprints nearest \\",
              "  --index research/shock_radar/closure/artifacts/january-graph-v1/index.json \\",
              "  --query research/shock_radar/closure/artifacts/january-graph-v1/demo-query.json", "```", "",
              "vector 为 `{assets: [固定十资产顺序], slots: {资产: {present, direction, magnitude_sigma, delay_minutes}}}`。缺失槽的后三值必须 null；观测触发必须正有限幅度、方向 ±1、延迟不超过 observed_minutes。", "",
              "真实示例取时间最早的一月事件、观察到第 10 分钟并排除自身；返回 " + str(len(results)) + " 个邻居。相似度不是概率，也不能证明未来会重复。", "",
              "- " + link("edges.csv","完整 100 行边表（含机会数与条件比例）"),
              "- " + link("graph.json","全部事件时序与逐边证据"),
              "- " + link("fingerprints.json","原始指纹、四类及完整合并过程"),
              "- " + link("event-classes.csv","84 行事件—类别表"),
              "- " + link("index.json","Demo 索引"), "- " + link("demo-query.json","真实查询包"),
              "- " + link("demo-nearest.json","三个检索结果"), "",
              "生成仅使用冻结的一月元数据与已保存预测，没有模型调用。源文件哈希生成前后核对一致。"]
    with (HERE / "FINGERPRINTS.md").open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    commands.add_parser("build")
    nearest = commands.add_parser("nearest")
    nearest.add_argument("--index",type=Path,required=True)
    nearest.add_argument("--query",type=Path,required=True)
    nearest.add_argument("--out",type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = generate()
    else:
        index,_ = _read(args.index); query,_ = _read(args.query)
        result = {"results":nearest_events(index,query["vector"],query["observed_minutes"],query.get("exclude_event_id"),query.get("k",3)),
                  "interpretation":METHOD["interpretation"]}
        if args.out:
            _write(args.out,result)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
