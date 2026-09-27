"""Read frozen predictions and January truth; never import or call a model."""
from pathlib import Path
import json
import math

from forecast_metrics.engine import freeze_forecast
from ..io import ROOT, digest, load_january, market_window, future_closes
from ..metrics import risk_metrics, score_event, aggregate_events

BASE = ROOT / "research/shock_radar/artifacts/january-v1"
BATCH = BASE / "inference-original-v1"
METHODS = ["kronos_base", "no_propagation", "btc_beta", "historical_30"]


def same(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, float):
        assert not isinstance(b, bool) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-13)
    else:
        assert a == b


def frozen_inputs():
    files = [BASE / "selection-approved-k6.json", BASE / "catalog-k6.json",
             BATCH / "run_manifest.json", BATCH / "summary.json"]
    files += sorted(BATCH.glob("event_*.json"))
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files}


def load():
    hashes = frozen_inputs()
    selection = json.loads((BASE / "selection-approved-k6.json").read_text())
    manifest = json.loads((BATCH / "run_manifest.json").read_text())
    summary = json.loads((BATCH / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["model_identity_unchanged"]
    assert manifest["selection"] == selection and manifest["method_names"] == METHODS
    assert len(selection["event_ids"]) == 84 == len(set(selection["event_ids"]))
    assert {p.stem for p in BATCH.glob("event_*.json")} == set(selection["event_ids"])
    assert digest((BASE / "catalog-k6.json").read_bytes()) == selection["catalog_sha256"]
    data = load_january()
    assert data["provenance"] == manifest["provenance"]
    catalog = json.loads((BASE / "catalog-k6.json").read_text())
    catalog_events = {e["event_id"]: e for e in catalog["events"] if e["eligible"]}
    rows, scores = [], []
    for event_id in selection["event_ids"]:
        a = json.loads((BATCH / (event_id + ".json")).read_text())
        event = a["event"]
        assert event == catalog_events[event_id]
        window = market_window(data, event["index"])
        assert a["window_id"] == window["window_id"]
        assert a["forecast_id"] == freeze_forecast(a["forecast"]).identity
        assert a["forecast"]["times"] == data["times"][event["index"]+1:event["index"]+31]
        spots = {asset: window["histories"][asset][-1]["close"] for asset in data["assets"]}
        assert spots == a["forecast"]["spots"]
        actual = future_closes(data, event["index"])
        paths = a["method_close_paths"]
        assert set(paths) == set(METHODS)
        for method in METHODS:
            assert set(paths[method]) == set(data["assets"])
            assert all(len(p) == 30 for p in paths[method].values())
        assert paths["kronos_base"] == {s: v["close"] for s, v in a["forecast"]["paths"][0]["assets"].items()}
        actual_risk = {s: risk_metrics(spots[s], actual[s]) for s in data["assets"]}
        predictions = {m: {s: risk_metrics(spots[s], paths[m][s]) for s in data["assets"]} for m in METHODS}
        same(actual_risk, a["actual_risk"])
        same(predictions, a["predicted_risk"])
        score = score_event(event, predictions, actual_risk, data["assets"])
        same(score, a["score"])
        assert score["status"] == "scored"
        scores.append(score)
        rows.append({"event_id": event_id, "timestamp": event["timestamp"],
                     "direction": event["direction"], "magnitude_sigma": event["magnitude_sigma"],
                     "systemic_flag": event["systemic_flag"], "origin_assets": event["origin_assets"],
                     "assets": list(data["assets"]), "scored_assets": score["scored_assets"],
                     "spots": spots, "actual": actual, "paths": paths, "saved_score": score})
    recomputed = aggregate_events(scores, METHODS)
    for k, v in recomputed.items():
        same(v, summary[k])
    assert hashes == frozen_inputs()
    return rows, {"frozen_inputs_sha256": hashes, "january": data["provenance"],
                  "event_count": len(rows), "model_calls": 0,
                  "source_model_identity": manifest["model_identity"],
                  "source_predict_config": manifest["predict_config"]}
