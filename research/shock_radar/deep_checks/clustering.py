"""Frozen k=4 leave-one-out stability, with no model or market-data access."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from statistics import mean, median

from ..closure import graph_fingerprints as frozen

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "closure/artifacts/january-graph-v1/fingerprints.json"
OUTPUT = HERE / "clustering.json"
THRESHOLDS = {"median_ari_min": .9, "p10_ari_min": .8, "singleton_persistence_min": .9}


def partition(clusters):
    result = {}
    for cluster in clusters:
        label = cluster["cluster_id"]
        if label in result or not cluster["event_ids"]:
            raise ValueError("Cluster labels must be unique and each class nonempty")
        result[label] = set(cluster["event_ids"])
        if len(result[label]) != len(cluster["event_ids"]):
            raise ValueError("Duplicate item in cluster")
    if sum(map(len,result.values())) != len(set().union(*result.values())):
        raise ValueError("A partition cannot put an event in multiple clusters")
    return result


def canonical_partition(groups):
    return {frozenset(items) for items in groups.values() if items}


def adjusted_rand(left, right):
    """Exact contingency-pair ARI; label names have no role in the result."""
    left_labels = {item:label for label,items in left.items() for item in items}
    right_labels = {item:label for label,items in right.items() for item in items}
    if (sum(map(len,left.values())) != len(left_labels)
            or sum(map(len,right.values())) != len(right_labels)
            or set(left_labels) != set(right_labels)):
        raise ValueError("ARI requires partitions of the same unique items")
    n = len(left_labels)
    if n < 2:
        return 1.
    choose2 = lambda count: count*(count-1)//2
    contingency = Counter((left_labels[item],right_labels[item]) for item in left_labels)
    joint = sum(choose2(nij) for nij in contingency.values())
    a = sum(choose2(len(items)) for items in left.values())
    b = sum(choose2(len(items)) for items in right.values())
    expected = a*b/choose2(n)
    denominator = .5*(a+b)-expected
    if denominator == 0:
        return 1. if canonical_partition(left)==canonical_partition(right) else 0.
    return (joint-expected)/denominator


def matched_jaccard(reference, candidate):
    """One-to-one maximum-sum Jaccard matching; absent references are null."""
    names = [name for name,items in reference.items() if items]
    targets = sorted(candidate)
    if len(names)>len(targets):
        raise ValueError("Not enough candidate classes for one-to-one matching")
    def value(a,b):
        return len(reference[a]&candidate[b])/len(reference[a]|candidate[b])
    options = permutations(targets,len(names))
    chosen = min(options,key=lambda labels:(-sum(value(a,b) for a,b in zip(names,labels)),labels))
    mapping = dict(zip(names,chosen))
    result = {}
    for name,items in reference.items():
        match = mapping.get(name)
        result[name] = {"reference_size_on_common_items":len(items),"matched_cluster_id":match,
                        "matched_cluster_size":len(candidate[match]) if match is not None else None,
                        "intersection":len(items&candidate[match]) if match is not None else None,
                        "union":len(items|candidate[match]) if match is not None else None,
                        "jaccard":value(name,match) if match is not None else None,
                        "status":"matched" if match is not None else "absent_after_own_deletion"}
    return result


def singleton_outcome(groups, singleton_id, omitted_id):
    if singleton_id == omitted_id:
        if any(singleton_id in items for items in groups.values()):
            raise ValueError("Deleted singleton still in candidate partition")
        return {"status":"absent_after_own_deletion","retention_trial":False,"isolated":None}
    containing = [items for items in groups.values() if singleton_id in items]
    if len(containing)!=1:
        raise ValueError("Retained singleton must occur in exactly one candidate class")
    return {"status":"retained","retention_trial":True,"isolated":len(containing[0])==1,
            "containing_cluster_size":len(containing[0])}


def linear_quantile(values,p):
    ordered=sorted(values)
    if not ordered:
        return None
    position=(len(ordered)-1)*p
    lower=int(position); upper=min(lower+1,len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def _stats(values):
    values=list(values)
    return {"n":len(values),"min":min(values) if values else None,
            "mean":mean(values) if values else None,"median":median(values) if values else None,
            "p10":linear_quantile(values,.1),"max":max(values) if values else None}


def engineering_pass(ari_median,ari_p10,persistence):
    return all(value is not None and value>=cutoff for value,cutoff in
               ((ari_median,.9),(ari_p10,.8),(persistence,.9)))


def analyze(payload):
    items=sorted(payload["events"],key=lambda e:(e["timestamp"],e["event_id"]))
    ids=[e["event_id"] for e in items]
    if len(ids)!=84 or len(set(ids))!=84:
        raise ValueError("Frozen January84 required; no event filtering")
    original=partition(payload["clusters"]["clusters"])
    if len(original)!=4 or set().union(*original.values())!=set(ids):
        raise ValueError("Original four-class partition must cover all84")
    rerun=frozen.cluster_fingerprints(items)
    if canonical_partition(original)!=canonical_partition(partition(rerun["clusters"])):
        raise ValueError("Full exact algorithm rerun differs from the frozen partition")
    singleton_classes=[name for name,values in original.items() if len(values)==1]
    if len(singleton_classes)!=1:
        raise ValueError("Expected exactly the original one singleton class")
    singleton_class=singleton_classes[0]
    singleton_id=next(iter(original[singleton_class]))
    runs=[]
    for omitted_id in ids:
        remaining=[item for item in items if item["event_id"]!=omitted_id]
        result=frozen.cluster_fingerprints(remaining)
        candidate=partition(result["clusters"])
        reference={label:values-{omitted_id} for label,values in original.items()}
        if len(candidate)!=4 or len(set().union(*candidate.values()))!=83:
            raise ValueError("Every leave-one-out rerun must retain83 items at k4")
        runs.append({"omitted_event_id":omitted_id,"common_event_count":83,"cluster_count":4,
                     "ari":adjusted_rand(reference,candidate),
                     "matched_jaccard":matched_jaccard(reference,candidate),
                     "singleton":singleton_outcome(candidate,singleton_id,omitted_id),
                     "partition":{label:sorted(values) for label,values in candidate.items()}})
    ari=_stats(r["ari"] for r in runs)
    retained=[r for r in runs if r["singleton"]["retention_trial"]]
    isolated=sum(r["singleton"]["isolated"] for r in retained)
    persistence=isolated/len(retained) if retained else None
    passed=engineering_pass(ari["median"],ari["p10"],persistence)
    class_summaries={}
    for label,values in original.items():
        observations=[r["matched_jaccard"][label]["jaccard"] for r in runs]
        class_summaries[label]={"original_size":len(values),
                                "jaccard":_stats(value for value in observations if value is not None),
                                "absent_after_own_deletion_count":sum(v is None for v in observations)}
    return {"status":"complete","analysis":"frozen_k4_leave_one_out",
            "event_count":84,"full_rerun_exact_match":True,
            "original_partition":{label:sorted(values) for label,values in original.items()},
            "original_cluster_sizes":{label:len(values) for label,values in original.items()},
            "method":payload["clusters"]["method"],"leave_one_out":runs,
            "summary":{"rerun_count":84,"common_items_per_rerun":83,"ari":ari,
                       "per_class_matched_jaccard":class_summaries,
                       "singleton":{"original_class":singleton_class,"event_id":singleton_id,
                                    "retention_trials":len(retained),"own_deletion_absent_count":84-len(retained),
                                    "isolated_retained_count":isolated,"persistence":persistence},
                       "engineering_thresholds":THRESHOLDS,"engineering_stability_pass":passed,
                       "singleton_replication_supported":False,
                       "interpretation":("算法在此次删一检验下通过预先冻结的工程稳定性门槛；单例仍只是一个孤立观察，没有重复类证据。"
                                         if passed else "算法未通过预先冻结的工程稳定性门槛；保留原四类，单例需标注不稳定。单例仍没有重复类证据。")},
            "definitions":{"ari":"Label-invariant contingency-pair adjusted Rand index on common83 items.",
                           "class_matching":"One-to-one maximum sum of class Jaccards; absent reference class excluded from assignment and recorded null.",
                           "p10":"Linear interpolation at (n-1)*0.1.",
                           "thresholds":"Descriptive engineering checks, not significance tests or predictive validation.",
                           "singleton":"Deleting the singleton itself is absent, not a failed retention trial and cannot recover/validate that class.",
                           "taxonomy":"Original custom distance/average linkage/k4 unchanged; no k3 retuning."}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError("clustering.json exists; frozen result will not be overwritten")
    raw=SOURCE.read_bytes(); before=sha256(raw).hexdigest(); payload=json.loads(raw)
    algorithm_path=Path(frozen.__file__)
    algorithm_hash=sha256(algorithm_path.read_bytes()).hexdigest()
    if algorithm_hash!=payload["provenance"]["generator_sha256"]:
        raise ValueError("Frozen graph algorithm source changed since original generation")
    if payload["clusters"]["method"]!=frozen.METHOD:
        raise ValueError("Frozen distance/linkage declaration changed")
    result=analyze(payload)
    if sha256(SOURCE.read_bytes()).hexdigest()!=before or sha256(algorithm_path.read_bytes()).hexdigest()!=algorithm_hash:
        raise ValueError("Read-only source changed during leave-one-out checks")
    result["provenance"]={"fingerprints_path":str(SOURCE),"fingerprints_sha256":before,
                          "algorithm_source_path":str(algorithm_path),"algorithm_source_sha256":algorithm_hash,
                          "analysis_source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
                          "input_hashes_unchanged":True,"model_calls":0,"market_data_reads":0}
    with OUTPUT.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"output":str(OUTPUT),"summary":result["summary"]},ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
