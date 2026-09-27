"""Frozen seven-checks analyses 3, 5 and 7; no model calls or data downloads.

All real inputs enter through common.load after the method lock is verified.
Pure array helpers can be tested without opening any experiment artifacts.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
from pathlib import Path

import numpy as np

MONTHS = ("2026-01", "2026-02")
MAX_DELAY = 30
HORIZON = 120
RECOVERY_POINTS = 10
RECOVERY_LEVEL = 1.2
CONTROL_DAYS = 28
MIN_CONTROLS = 3


def _number(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "p90": None, "p99": None, "proportion_le_30": None}
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Delay samples must be a finite vector")
    return {"n": len(values), "mean": float(values.mean()),
            "p90": float(np.quantile(values, .90, method="linear")),
            "p99": float(np.quantile(values, .99, method="linear")),
            "proportion_le_30": float(np.mean(values <= MAX_DELAY))}


def graph_delay_analysis(graph):
    """Distinguish equal-edge means from the pooled, dependent observations."""
    edge_means, delays = [], []
    for edge in graph["edges"]:
        observations = edge["observations"]
        if not observations or edge["source"] == edge["target"]:
            raise ValueError("Frozen graph edges must be positive non-self edges")
        if edge["count"] != len(observations) or edge["weight"] != len(observations):
            raise ValueError("Graph count/weight/observation mismatch")
        edge_delays = [o["delay_minutes"] for o in observations]
        if not all(_number(x) and 0 < x <= MAX_DELAY for x in edge_delays):
            raise ValueError("Generator delay domain is (0,30]")
        saved = edge["mean_delay_minutes"]
        if not _number(saved) or not math.isclose(saved, float(np.mean(edge_delays)), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Saved edge mean does not reproduce its observations")
        edge_means.append(saved)
        delays.extend(edge_delays)
    return {"month": "2026-01", "positive_edge_means_equal_weight": _distribution(edge_means),
            "pooled_edge_observations": _distribution(delays),
            "delay_definition": "target retained trigger index minus source episode t0; all sources are co-origins at t0",
            "observation_window_minutes": {"lower_exclusive": 0, "upper_inclusive": 30},
            "excluded_from_delay_samples": ["zero-count edges", "no follow-up in 30 minutes", "follow-up after 30 minutes", "simultaneous co-origins"],
            "dependence": "Co-origins and overlapping event windows can repeat the same target trigger; observations are not independent.",
            "february": None, "combined": None,
            "availability_reason": "Only the frozen January graph is authorized as the graph source; no February graph is constructed.",
            "lead_time_conclusion": "图谱仅描述，不承诺提前量",
            "tail_limit": "No inference outside 30 minutes; 30 minus delay is not operational warning lead time."}


def recovery_confirmation(curve):
    """Return the tenth consecutive <=1.2 point, never its run's first point."""
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 1 or not np.isfinite(curve).all():
        raise ValueError("Recovery requires a finite one-dimensional curve")
    streak = 0
    for h, value in enumerate(curve):
        streak = streak + 1 if value <= RECOVERY_LEVEL else 0
        if streak == RECOVERY_POINTS:
            return h
    return None


def recovery_summary(confirmations):
    """Administrative right censoring at 120; median of all event times."""
    values = list(confirmations)
    recovered = sorted(v for v in values if v is not None)
    if not all(type(x) is int and 9 <= x <= HORIZON for x in recovered):
        raise ValueError("Recovery confirmation must be an integer in [9,120]")
    n = len(values)
    median = recovered[(n-1)//2] if n and len(recovered) >= (n+1)//2 else None
    return {"n": n, "n_confirmed": len(recovered), "n_right_censored_gt_120": n-len(recovered),
            "median_confirmation_minutes": median,
            "median_status": "estimated" if median is not None else (">120" if n else "no_valid_events"),
            "median_definition": "Earliest h with cumulative confirmed recovery >=50% of all valid event-tier curves; no averaging of censored times.",
            "confirmation_minutes": values}


def decay_event(closes, index, assets):
    """Compute per-asset 121-point curves using only the event's own month."""
    closes = np.asarray(closes, dtype=np.float64)
    if closes.ndim != 2 or closes.shape[1] != len(assets):
        raise ValueError("Close array must have one column per frozen asset")
    if index < 125 or index + HORIZON >= len(closes):
        return {"status": "boundary_insufficient", "assets": {}}
    records = {}
    for j, asset in enumerate(assets):
        sample = closes[index-125:index+HORIZON+1,j]
        if not np.isfinite(sample).all() or not (sample>0).all():
            records[asset] = {"baseline":None,"status":"invalid_close_window","volatility":None,"ratio":None}
            continue
        # 121 closes -> 120 returns, the last ending at t0-5 (not t0).
        baseline_returns = np.diff(np.log(closes[index-125:index-4,j]))
        b = float(np.std(baseline_returns,ddof=0)*math.sqrt(10))
        # 131 closes -> 130 returns -> 121 windows of ten returns.
        recent_returns = np.diff(np.log(closes[index-10:index+HORIZON+1,j]))
        rolling = np.lib.stride_tricks.sliding_window_view(recent_returns,10)
        volatility = np.std(rolling,axis=-1,ddof=0)*math.sqrt(10)
        assert baseline_returns.shape == (120,) and volatility.shape == (121,)
        records[asset] = {"baseline": b, "status": "zero_baseline" if b == 0 else "valid",
                          "volatility": volatility.tolist(),
                          "ratio": None if b == 0 else (volatility/b).tolist()}
    return {"status": "available", "assets": records}


def weighted_median_curves(curves, weights):
    """Exactly np.median of repeated event rows, including even-size averaging."""
    curves = np.asarray(curves, dtype=np.float64)
    weights = np.asarray(weights)
    if curves.ndim != 2 or weights.ndim != 2 or weights.shape[1] != len(curves):
        raise ValueError("Curve/weight dimensions do not match")
    if not np.isfinite(curves).all() or not np.isfinite(weights).all() or (weights < 0).any() or not (weights == np.floor(weights)).all():
        raise ValueError("Finite curves and integer nonnegative bootstrap counts required")
    output = np.full((len(weights),curves.shape[1]),np.nan)
    totals = weights.sum(axis=1).astype(np.int64)
    valid = totals > 0
    if not valid.any():
        return output
    w = weights[valid].astype(np.int64,copy=False)
    lower = (totals[valid]-1)//2
    upper = totals[valid]//2
    for h in range(curves.shape[1]):
        order = np.argsort(curves[:,h], kind="stable")
        cumulative = np.cumsum(w[:,order],axis=1)
        lo = np.argmax(cumulative > lower[:,None],axis=1)
        hi = np.argmax(cumulative > upper[:,None],axis=1)
        output[valid,h] = (curves[order[lo],h]+curves[order[hi],h])/2
    return output


def _ids(event):
    return {k:event[k] for k in ("event_id", "month", "day", "timestamp", "index", "global_index")}


def _groups(events):
    return {**{m:[e for e in events if e["month"] == m] for m in MONTHS}, "combined":list(events)}


def decay_analysis(bundle, common):
    events = bundle["events"]
    records = []
    for event in events:
        base = decay_event(bundle["monthly"][event["month"]]["closes"],event["index"],common.ASSETS)
        record = {**_ids(event), **base, "tiers":{}}
        for tier, assets in common.TIERS.items():
            if base["status"] != "available":
                record["tiers"][tier] = {"status":base["status"],"curve":None,"confirmation_minutes":None}
                continue
            bad = {a:base["assets"][a]["status"] for a in assets if base["assets"][a]["ratio"] is None}
            if bad:
                record["tiers"][tier] = {"status":"invalid_asset_baseline","asset_reasons":bad,"curve":None,"confirmation_minutes":None}
                continue
            curve = np.median([base["assets"][a]["ratio"] for a in assets],axis=0)
            record["tiers"][tier] = {"status":"valid","curve":curve.tolist(),"confirmation_minutes":recovery_confirmation(curve)}
        records.append(record)
    lookup = {r["event_id"]:r for r in records}
    groups = {}
    for group, selected_events in _groups(events).items():
        groups[group] = {}
        weights = common.bootstrap_weights(selected_events) if selected_events else None
        for tier in common.TIERS:
            selected = [lookup[e["event_id"]]["tiers"][tier] for e in selected_events]
            valid = [i for i,s in enumerate(selected) if s["status"] == "valid"]
            excluded = Counter(s["status"] for s in selected if s["status"] != "valid")
            n_days = len({selected_events[i]["day"] for i in valid})
            item = {"n_expected":len(selected_events),"n":len(valid),"n_missing":len(selected_events)-len(valid),
                    "n_days":n_days,"exclusion_counts":dict(excluded),"stability_eligible":len(valid)>=10 and n_days>=5,
                    "median_curve":None,"pointwise_ci95":None,"median_curve_confirmation_minutes":None,
                    "recovery":recovery_summary([selected[i]["confirmation_minutes"] for i in valid])}
            if valid:
                curves = np.asarray([selected[i]["curve"] for i in valid])
                median = np.median(curves,axis=0)
                item.update(median_curve=median.tolist(),
                    median_curve_confirmation_minutes=recovery_confirmation(median))
                if item["stability_eligible"]:
                    draws = weighted_median_curves(curves,weights[:,valid])
                    usable = np.isfinite(draws).all(axis=1)
                    item.update(pointwise_ci95=np.quantile(draws[usable],[.025,.975],axis=0,method="linear").T.tolist() if usable.any() else None,
                        bootstrap_usable=int(usable.sum()),bootstrap_empty=int((~usable).sum()))
            groups[group][tier] = item
    return {"horizon_minutes":list(range(121)),"rolling_return_count":10,"baseline_return_count":120,
            "baseline_last_return_offset":-5,"baseline_scaling":"std(ddof=0) * sqrt(10)",
            "event_tier_aggregation":"median of all required tier asset ratios; any zero baseline excludes that event-tier",
            "event_aggregation":"median across valid events","bands":"95% pointwise day-block intervals, not simultaneous confidence bands",
            "right_censoring":"Unconfirmed recovery remains >120; never replaced by 120 or silently discarded.",
            "groups":groups,"events":records}


def observed_volume_window(data, end, asset_index):
    """Historical CSV volumes only; model future quality flags are irrelevant."""
    if end < 29 or end >= len(data["volumes"]):
        return {"status":"boundary_insufficient","sum":None}
    v=np.asarray(data["volumes"])[end-29:end+1,asset_index]
    a=np.asarray(data["amounts"])[end-29:end+1,asset_index]
    inferred=np.isfinite(v)&np.isfinite(a)&(v>=0)&(a>=0)&((v==0)==(a==0))
    explicit_mask=data.get("volume_invalid_explicit")
    explicit_bad=bool(explicit_mask is not None and np.asarray(explicit_mask)[end-29:end+1,asset_index].any())
    inferred_bad=not bool(inferred.all())
    provided_inferred=data.get("volume_invalid_inferred")
    if provided_inferred is not None:
        inferred_bad |= bool(np.asarray(provided_inferred)[end-29:end+1,asset_index].any())
    supplied=data.get("observed_volume_valid",data.get("volume_valid"))
    supplied_bad=bool(supplied is not None and not np.asarray(supplied)[end-29:end+1,asset_index].all())
    if explicit_bad or inferred_bad or supplied_bad:
        return {"status":"volume_invalid","sum":None,"explicit_invalid":explicit_bad,
                "inferred_invalid":inferred_bad,"supplied_invalid":supplied_bad}
    return {"status":"valid","sum":float(v.sum()),"explicit_invalid":False,"inferred_invalid":False,"supplied_invalid":False}


def _weekend(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z","+00:00")).weekday() >= 5


def normalize_direction(value):
    """Catalog directions are numeric +/-1 (0 for mixed); keep labels explicit."""
    if isinstance(value,(bool,np.bool_)):
        raise ValueError("Boolean is not an event direction")
    if value in (1,"up"):
        return "up"
    if value in (-1,"down"):
        return "down"
    if value in (0,"mixed"):
        return "mixed"
    raise ValueError(f"Unsupported event direction {value!r}")


def volume_asset(event, data, asset_index, trigger_indices):
    t=int(event["global_index"])
    observed=observed_volume_window(data,t,asset_index)
    controls=[]; rejected=Counter()
    weekday_type=_weekend(event["timestamp"])
    trigger_indices=np.asarray(trigger_indices,dtype=np.int64)
    if len(trigger_indices)>1 and (np.diff(trigger_indices)<0).any():
        raise ValueError("Control shock indices must be sorted")
    for days_back in range(1,CONTROL_DAYS+1):
        c=t-days_back*1440
        if c < 0 or c>=len(data["times"]):
            rejected["outside_authorized_data"] += 1; continue
        if c-120 < 0 or c+120 >= t or c+120 >= len(data["times"]):
            rejected["context_boundary_insufficient"] += 1; continue
        if not np.asarray(data["detector_known"])[c-120:c+121].all():
            rejected["detector_coverage_unknown"] += 1; continue
        if _weekend(data["times"][c]) != weekday_type:
            rejected["weekday_weekend_mismatch"] += 1; continue
        # Fixed calendar minute offsets require the shared loader's continuous grid.
        if data["times"][c][11:16] != event["timestamp"][11:16]:
            raise ValueError("Control does not preserve the same UTC minute")
        lo=np.searchsorted(trigger_indices,c-120,side="left")
        hi=np.searchsorted(trigger_indices,c+120,side="right")
        if hi>lo:
            rejected["shock_within_control_context"] += 1; continue
        window=observed_volume_window(data,c,asset_index)
        if window["status"] != "valid":
            rejected["control_"+window["status"]] += 1
            for kind in ("explicit_invalid","inferred_invalid","supplied_invalid"):
                if window.get(kind): rejected["control_"+kind] += 1
            continue
        controls.append({"days_before":days_back,"global_index":c,"timestamp":data["times"][c],"volume_sum":window["sum"]})
    baseline=float(np.median([c["volume_sum"] for c in controls])) if len(controls)>=MIN_CONTROLS else None
    reasons=[]
    if observed["status"] != "valid": reasons.append("event_"+observed["status"])
    if len(controls)<MIN_CONTROLS: reasons.append("fewer_than_3_controls")
    if baseline is not None and baseline<=0: reasons.append("zero_control_baseline")
    if observed["sum"] is not None and observed["sum"]<=0: reasons.append("zero_event_volume")
    value=None if reasons else float(math.log(observed["sum"])-math.log(baseline))
    return {"status":"valid" if not reasons else "excluded","exclusion_reasons":reasons,
            "event_window":observed,"n_controls":len(controls),"control_median":baseline,
            "log_ratio":value,"control_rejection_counts":dict(rejected),"controls":controls}


def volume_analysis(bundle, common):
    triggers=sorted(t["global_index"] for t in bundle["triggers"])
    records=[]
    for event in bundle["events"]:
        assets={a:volume_asset(event,bundle["combined"],j,triggers) for j,a in enumerate(common.ASSETS)}
        record={**_ids(event),"direction":normalize_direction(event["direction"]),
                "direction_original":event["direction"],"assets":assets,"tiers":{}}
        for tier,members in common.TIERS.items():
            invalid={a:assets[a]["exclusion_reasons"] for a in members if assets[a]["log_ratio"] is None}
            record["tiers"][tier]={"status":"excluded" if invalid else "valid","asset_reasons":invalid,
                "mean_log_ratio":None if invalid else float(np.mean([assets[a]["log_ratio"] for a in members]))}
        records.append(record)
    lookup={r["event_id"]:r for r in records}; groups={}
    for month,events in _groups(bundle["events"]).items():
        groups[month]={}
        for tier in common.TIERS:
            groups[month][tier]={}
            for direction in ("all","up","down","mixed"):
                selected=events if direction=="all" else [e for e in events if normalize_direction(e["direction"])==direction]
                values=[lookup[e["event_id"]]["tiers"][tier]["mean_log_ratio"] for e in selected]
                summary=common.mean_ci(values,selected,adjusted=month=="combined" and direction in ("up","down"))
                estimate=summary.get("estimate")
                summary["geometric_volume_multiple"]=None if estimate is None else float(math.exp(estimate))
                for key in ("ci95","adjusted_ci"):
                    interval=summary.get(key)
                    summary["geometric_"+key]=None if interval is None else [float(math.exp(x)) for x in interval]
                counts=Counter(); control_counts=Counter(); window_counts=Counter()
                for event in selected:
                    for a in common.TIERS[tier]:
                        asset=lookup[event["event_id"]]["assets"][a]
                        window_counts['event_windows_expected']+=1
                        window_counts['event_windows_'+asset['event_window']['status']]+=1
                        window_counts['control_candidates_expected']+=CONTROL_DAYS
                        window_counts['control_windows_valid']+=asset['n_controls']
                        control_counts.update(asset['control_rejection_counts'])
                        counts.update(asset["exclusion_reasons"])
                        for flag in ("explicit_invalid","inferred_invalid","supplied_invalid"):
                            if asset["event_window"].get(flag): counts["event_"+flag]+=1
                summary["asset_event_exclusion_counts"]=dict(counts)
                summary["window_audit_counts"]=dict(window_counts)
                summary["control_rejection_counts"]=dict(control_counts)
                summary["missing_policy"]="Any missing required tier asset excludes that event-tier; all expected events remain in the denominator metadata."
                groups[month][tier][direction]=summary
    candidates={}
    for tier in common.TIERS:
        candidates[tier]={}
        for direction in ("up","down"):
            results=[groups[m][tier][direction] for m in MONTHS]
            combined=groups['combined'][tier][direction]
            eligible=all(r.get("stability_eligible",False) for r in [*results,combined])
            above=eligible and all(r['estimate']>0 for r in results) and combined.get('adjusted_ci') is not None and combined['adjusted_ci'][0]>0
            candidates[tier][direction]={"candidate_volume_expansion":bool(above) if eligible else None,
                "reason":"Both month estimates positive and individually stable; combined Bonferroni lower bound >0" if above else "Sparse, inconsistent monthly directions, or combined adjusted positive lower-bound criterion not met",
                "not_a_product_rule":True}
    return {"window":"30 minute volumes ending at t0, including the triggering window",
        "control_rule":"Previous 28 calendar days, same UTC minute and weekday/weekend class; no retained trigger of any asset within control +/-120 minutes",
        "minimum_controls":3,"baseline":"median of control 30-minute base-asset volume sums, within the same asset",
        "observed_validity":"Historical raw volume and quote amount finite/nonnegative/zero-consistent; no future forecast validity filter",
        "raw_explicit_volume_valid_field_present":bundle["provenance"]["raw_explicit_volume_valid_field_present"],
        "implementation_notes":["Control exclusion is exactly c-120 through c+120, not the expanded 30-minute volume window.",
            "The full control context must have detector_known=True; each month's first 65 indices are unknown.",
            "Raw 12-column CSV has no explicit validity flag; explicit-invalid counts are zero, derived invalidity is separate.",
            "Numeric catalog direction +1/-1/0 maps explicitly to up/down/mixed.",
            "Only the four combined up/down by tier contrasts receive the shared family-of-26 adjustment; individual months show ordinary intervals and must have positive estimates and sufficient support."],
        "aggregation":"mean log ratio across all required tier assets, then equal-weight event mean; geometric multiple is exp(mean log ratio)",
        "missing_policy":"No epsilon, no imputation, no bridging missing data, no model future-volume filter",
        "groups":groups,"candidate_checks":candidates,"events":records}


def analyze(bundle, common_module=None):
    if common_module is None:
        from . import common as common_module
    return {"analysis_3_graph_delay":graph_delay_analysis(bundle["graph"]),
            "analysis_5_decay":decay_analysis(bundle,common_module),
            "analysis_7_observed_volume":volume_analysis(bundle,common_module)}



def main():
    from . import common
    if (common.HERE/'result_3_5_7.json').exists():
        raise FileExistsError('Analysis 3/5/7 already completed; no overwrite')
    bundle=common.load()
    result=analyze(bundle,common)
    common.write_result('result_3_5_7.json',result,bundle)
    print({'result':str(common.HERE/'result_3_5_7.json'),'events':len(bundle['events']),'new_model_calls':0,'plotting':'separate saved-result renderer; experiment dependencies unchanged'})

if __name__ == '__main__':
    main()
