"""Actual local HTTP/model integration check. No fake data or implicit substitutions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from uuid import uuid4


def verify(base, output):
    if urlsplit(base).hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Local demo only")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Verification evidence already exists; choose a new output path")
    evidence = {"started_at": datetime.now(timezone.utc).isoformat(), "checks": [],
                "view_conflict_retries": 0, "status": "running"}
    owned_forecast_id = None

    def call(path, body=None):
        req = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                      headers={} if body is None else {"Content-Type": "application/json"})
        with urlopen(req, timeout=20) as response:
            return json.load(response)
    def mutate(path, body, *, target_forecast_id=None, max_attempts=5):
        """Retry only a confirmed optimistic-lock conflict, never a model failure.

        A view_changed response occurs before application. Reusing an analyze
        request ID is safe on that branch; ambiguous network failures are not
        retried. Holdings always keep the originally intended forecast identity.
        """
        for attempt in range(max_attempts):
            current = call("/api/state")
            if target_forecast_id and (current.get("forecast") or {}).get("forecast_id") != target_forecast_id:
                raise RuntimeError("Forecast changed during verification; intended mutation was not applied")
            request = {**body, "expected_view_revision": current["view_revision"]}
            if path == "/api/holdings":
                request["expected_forecast_id"] = target_forecast_id
            try:
                response = call(path, request)
            except HTTPError as exc:
                try:
                    code = json.load(exc).get("error", {}).get("code")
                except (ValueError, AttributeError, TypeError):
                    code = None
                if exc.code != 409 or code != "view_changed" or attempt + 1 == max_attempts:
                    raise
                evidence["view_conflict_retries"] += 1
                time.sleep(.05)
                continue
            if target_forecast_id and (response.get("forecast") or {}).get("forecast_id") != target_forecast_id:
                raise RuntimeError("Mutation response changed the intended forecast identity")
            return response, request
        raise RuntimeError("Unreachable mutation retry state")

    def settled(job_id):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            state = call("/api/state")
            if (state.get("job") or {}).get("job_id") != job_id:
                raise RuntimeError("A different analysis replaced the verification job")
            if state["status"] == "ready":
                return state
            if state["status"] == "error":
                raise RuntimeError(state["error"])
            time.sleep(.15)
        raise TimeoutError("Analysis did not finish")
    def analyze(force):
        body = {"request_id": "verify:" + uuid4().hex, "profile_id": "binance_jan2025",
                "force_run": force, "predict_config": {"lookback": 256, "horizon": 30, "path_count": 1, "seed": 20260926}}
        response, submitted = mutate("/api/analyze", body)
        return settled(response["job_id"]), submitted

    def scored(forecast_id):
        """A ready model result may still have scoring in flight."""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = call("/api/state")
            if (state.get("forecast") or {}).get("forecast_id") != forecast_id:
                raise RuntimeError("Forecast changed while waiting for replay scoring")
            if state["status"] == "error":
                raise RuntimeError(state["error"])
            evaluation = state["evaluation"]
            rows = [r for r in evaluation.get("rows", []) if r.get("forecast_id") == forecast_id
                    and r.get("status") == "scored" and r.get("metric") == "volatility"]
            pairs = {(r.get("symbol"), r.get("model")) for r in rows}
            required = {(symbol, model) for symbol in ("BTC", "ETH", "SOL")
                        for model in ("Kronos-base", "historical-volatility")}
            if evaluation.get("status") == "scored" and required <= pairs:
                return evaluation
            time.sleep(.15)
        raise TimeoutError("Current replay forecast did not produce complete Kronos/baseline volatility scores")

    failure = None
    try:
        first, request = analyze(True)
        fid = owned_forecast_id = first["forecast"]["forecast_id"]
        count = first["counters"]["model_runs"]
        audit = first["forecast"]["ohlc_corrections"]
        assert audit["total_candles"] == 90 and len(audit["records"]) == 90, "Incomplete correction audit"
        evidence["checks"].append("real_model_to_metrics")
        read = call("/api/state")
        assert read["forecast"]["forecast_id"] == fid and read["counters"]["model_runs"] == count
        evidence["checks"].append("readonly_state")
        response = call("/api/analyze", request)
        assert response["idempotent"] and response["counters"]["model_runs"] == count
        evidence["checks"].append("request_idempotency")
        changed, _ = mutate("/api/holdings", {"capital": 12000, "weights": {"BTC": .5, "ETH": .3, "SOL": .2}}, target_forecast_id=fid)
        assert changed["forecast"]["forecast_id"] == fid and changed["counters"]["model_runs"] == count
        assert abs(changed["metrics"]["current_value"] - 12000) < 1e-7
        evidence["checks"].append("holdings_only_arithmetic")
        metrics_id = changed["metrics"]["metrics_id"]
        changed, _ = mutate("/api/alerts", {"drawdown_threshold": .04, "top_k": 1}, target_forecast_id=fid)
        assert changed["metrics"]["metrics_id"] == metrics_id and changed["counters"]["model_runs"] == count
        evidence["checks"].append("threshold_only_alerts")
        cached, _ = analyze(False)
        owned_forecast_id = cached["forecast"]["forecast_id"]
        assert owned_forecast_id == fid and cached["counters"]["model_runs"] == count
        assert cached["metrics"]["holdings"]["quantities"] == changed["metrics"]["holdings"]["quantities"]
        evidence["checks"].append("prediction_cache_and_fixed_quantities")
        rerun, _ = analyze(True)
        owned_forecast_id = rerun["forecast"]["forecast_id"]
        assert owned_forecast_id != fid and rerun["counters"]["model_runs"] == count + 1
        evidence["checks"].append("forced_rerun_new_snapshot")
        scored(owned_forecast_id)
        evidence["checks"].append("current_forecast_replay_scoring_ready")
    except Exception as exc:
        failure = exc
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        # Restoration runs even after a failed check. Never overwrite another
        # forecast, and never replace the original failure with a cleanup error.
        if owned_forecast_id:
            try:
                mutate("/api/holdings", {"capital": 10000, "weights": {"BTC": .5, "ETH": .3, "SOL": .2}}, target_forecast_id=owned_forecast_id)
                mutate("/api/alerts", {"drawdown_threshold": .03, "top_k": 2}, target_forecast_id=owned_forecast_id)
                evidence["defaults_restored"] = True
                report = call("/api/report")
                assert owned_forecast_id == report["forecast_id"] == report["metrics"]["forecast_id"] == report["alerts"]["forecast_id"]
                assert report["ohlc_corrections"] == report["forecast"]["ohlc_corrections"]
                evidence["checks"].append("report_same_snapshot_with_correction_audit")
                evidence.update(forecast_id=report["forecast_id"],
                                correction_quality={k: v for k,v in report["ohlc_corrections"].items() if k != "records"},
                                runtime=report["runtime"], evaluation_status=report["evaluation"]["status"],
                                metrics_id=report["metrics"]["metrics_id"])
            except Exception as exc:
                evidence["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
                if failure is None:
                    failure = exc
        else:
            evidence["defaults_restored"] = False
            evidence["cleanup_note"] = "No owned forecast was established; no unrelated result was modified."
        evidence.update(finished_at=datetime.now(timezone.utc).isoformat(), status="failed" if failure else "passed")
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        except Exception as exc:
            evidence["evidence_write_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if failure is None:
                failure = exc
                evidence["status"] = "failed"
        print(json.dumps(evidence, ensure_ascii=False, allow_nan=False))
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.url, args.output)
