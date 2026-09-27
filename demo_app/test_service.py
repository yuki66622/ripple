"""Business invariants tested with an explicitly synthetic worker, never product fallback."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
import json
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from forecast_metrics.engine import PAIRING
from model_adapter.corrections import correct_paths
from model_adapter import build_volume_quality
from .service import APIError, Application
from .server import make_handler


def fixture_loader(profile, *, lookback, as_of=None):
    anchor = datetime.fromisoformat((as_of or "2025-01-02T00:00:00Z").replace("Z", "+00:00"))
    spots = {"BTC": 100., "ETH": 50., "SOL": 10.}
    histories = {a: [{"time": (anchor - timedelta(minutes=255-i)).isoformat().replace("+00:00", "Z"),
                      "open": p, "high": p, "low": p, "close": p, "volume": 1., "amount": p}
                     for i in range(256)] for a, p in spots.items()}
    return {"schema_version": 1, "window_id": "test:" + profile + str(anchor), "profile_id": profile,
            "source": "explicit_synthetic_test:" + profile, "mode": "replay", "quote_currency": "USDT" if profile == "binance_jan2025" else "USD",
            "as_of": histories["BTC"][-1]["time"], "interval_seconds": 60, "assets": list(spots), "histories": histories, "quality": {"amount_source": "fictional"}}


class FakeWorker:
    def __init__(self):
        self.calls, self.bad = 0, False
    def identity(self):
        return {"model_revision": "synthetic-test-worker", "backend": "test"}
    def predict(self, window, config, prediction_run_id):
        self.calls += 1
        t = datetime.fromisoformat(window["as_of"].replace("Z", "+00:00"))
        spots = {a: rows[-1]["close"] for a, rows in window["histories"].items()}
        times = [(t + timedelta(minutes=i+1)).isoformat().replace("+00:00", "Z") for i in range(30)]
        assets = {a: {"open": [p]*30, "close": [p*(1+(i+1)/1000) for i in range(30)],
                     "high": [p*1.1]*30, "low": [p*.9]*30, "volume": [1.]*30, "amount": [p]*30}
                  for a,p in spots.items()}
        raw = [{"path_id": str(i), "assets": deepcopy(assets)} for i in range(config["path_count"])]
        corrected, audit = correct_paths(raw, times)
        paths = [{"path_id": p["path_id"], "assets": {a: {k:v[k] for k in ("high", "low", "close")}
                  for a,v in p["assets"].items()}} for p in corrected]
        if self.bad:
            paths[0]["assets"]["BTC"]["low"][0] = 1000
        return {"forecast": {"model_revision": "synthetic-test-worker", "prediction_run_id": prediction_run_id,
                "source": window["source"], "quote_currency": window["quote_currency"], "as_of": window["as_of"],
                "interval_seconds": 60, "times": times, "spots": spots, "pairing": PAIRING,
                "paths": paths, "ohlc_corrections": audit,
                "volume_quality": build_volume_quality(raw, times)}, "raw_paths": raw,
                "runtime": {"inference_ms": 1, "backend": "test"}}
    def close(self):
        pass


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.worker = FakeWorker()
        self.app = Application(self.tmp.name, self.worker, fixture_loader,
                               scorer=lambda *_: {"status": "scored", "rows": [], "quality_status": "insufficient_evidence"})
        self.counter = 0
    def tearDown(self):
        self.app.close()
        self.tmp.cleanup()
    def run_job(self, **overrides):
        self.counter += 1
        body = {"request_id": str(self.counter), "profile_id": "binance_jan2025",
                "expected_view_revision": self.app.view()["view_revision"], **overrides}
        self.app.analyze(body)
        return self.app.wait(), body

    def test_cache_force_and_request_id(self):
        first, body = self.run_job()
        self.assertEqual(first["status"], "ready")
        self.app.analyze(body)
        self.assertEqual(self.worker.calls, 1)
        second, _ = self.run_job()
        self.assertEqual(first["forecast"]["forecast_id"], second["forecast"]["forecast_id"])
        self.assertEqual(self.worker.calls, 1)
        third, forced = self.run_job(force_run=True)
        self.assertNotEqual(first["forecast"]["forecast_id"], third["forecast"]["forecast_id"])
        self.app.analyze(forced)
        self.assertEqual(self.worker.calls, 2)

    def test_edits_never_predict_and_threshold_keeps_metrics(self):
        s, _ = self.run_job()
        fid = s["forecast"]["forecast_id"]
        s = self.app.holdings({"capital": 20000, "weights": {"BTC": .3, "ETH": .4, "SOL": .3},
                              "expected_forecast_id": fid, "expected_view_revision": s["view_revision"]})
        mid = s["metrics"]["metrics_id"]
        s = self.app.alerts({"drawdown_threshold": .04, "top_k": 1, "expected_view_revision": s["view_revision"]})
        self.assertEqual(s["metrics"]["metrics_id"], mid)
        self.assertEqual(self.app.report()["forecast_id"], fid)
        self.assertEqual(self.worker.calls, 1)
        q = s["metrics"]["holdings"]["quantities"]
        s, _ = self.run_job(as_of="2025-01-02T00:01:00Z")
        self.assertEqual(s["metrics"]["holdings"]["quantities"], q)

    def test_invalid_output_keeps_previous_snapshot(self):
        s, _ = self.run_job()
        self.worker.bad = True
        bad, _ = self.run_job(force_run=True)
        self.assertEqual(bad["status"], "error")
        self.assertTrue(bad["stale"])
        self.assertEqual(bad["forecast"]["forecast_id"], s["forecast"]["forecast_id"])
        self.worker.bad = False
        recovered, _ = self.run_job(force_run=True)
        self.assertEqual(recovered["status"], "ready")

    def test_revision_and_id_conflicts(self):
        s, body = self.run_job()
        with self.assertRaises(APIError) as caught:
            self.app.analyze({**body, "force_run": True})
        self.assertEqual(caught.exception.status, 409)

    def test_nan_evidence_finishes_failure_and_allows_retry(self):
        from .worker import WorkerError
        original = self.worker.predict
        def invalid(*args):
            raise WorkerError({"type": "PredictionValidationError", "rejected_output": {
                "raw_paths": [{"invalid": float("nan")}], "issues": ["nonfinite"]}})
        self.worker.predict = invalid
        state, _ = self.run_job()
        self.assertEqual(state["status"], "error")
        failure = self.app.store.get("failure:" + state["job"]["job_id"])
        artifact = self.app.store.read_artifact(failure["artifact"])
        self.assertEqual(artifact["rejected_output"]["raw_paths"][0]["invalid"], {"nonfinite_float": "NaN"})
        self.worker.predict = original
        self.assertEqual(self.run_job()[0]["status"], "ready")

    def test_reject_different_output_interval(self):
        original = self.worker.predict
        def invalid(*args):
            result = original(*args)
            f = result["forecast"]
            f["interval_seconds"] = 300
            origin = datetime.fromisoformat(f["as_of"].replace("Z", "+00:00"))
            f["times"] = [(origin+timedelta(minutes=5*(i+1))).isoformat() for i in range(30)]
            return result
        self.worker.predict = invalid
        self.assertEqual(self.run_job()[0]["status"], "error")

    def test_missing_or_forged_correction_evidence_rejected(self):
        original = self.worker.predict
        for change in ("missing_raw", "missing_audit", "forged_original", "missing_volume_audit", "forged_volume"):
            with self.subTest(change=change):
                def invalid(*args):
                    result = original(*args)
                    if change == "missing_raw":
                        del result["raw_paths"]
                    elif change == "missing_audit":
                        del result["forecast"]["ohlc_corrections"]
                    elif change == "missing_volume_audit":
                        del result["forecast"]["volume_quality"]
                    elif change == "forged_volume":
                        result["raw_paths"][0]["assets"]["BTC"]["volume"][0] = -1
                    else:
                        result["raw_paths"][0]["assets"]["BTC"]["open"][0] += 1
                    return result
                self.worker.predict = invalid
                self.assertEqual(self.run_job(force_run=True)[0]["status"], "error")
        self.worker.predict = original
        self.assertEqual(self.run_job()[0]["status"], "ready")

    def test_restore_requires_untouched_raw_artifact(self):
        state, _ = self.run_job()
        fid = state["forecast"]["forecast_id"]
        saved = self.app.store.get("forecast:" + fid)
        artifact = self.app.store.read_artifact(saved["artifact"])
        artifact["raw_paths"][0]["assets"]["BTC"]["open"][0] += 1
        saved["artifact"] = self.app.store.write_artifact(artifact)
        self.app.store.put_many({"forecast:" + fid: saved})
        self.app.close()
        self.worker = FakeWorker()
        self.app = Application(self.tmp.name, self.worker, fixture_loader)
        self.assertEqual(self.app.view()["error"]["code"], "corrupt_state")
        self.assertIsNone(self.app.view()["forecast"])
        self.assertEqual(self.worker.calls, 0)

    def test_negative_unused_volume_publishes_identical_price_metrics(self):
        first, _ = self.run_job()
        original = self.worker.predict
        def negative(*args):
            result = original(*args)
            result["raw_paths"][0]["assets"]["BTC"]["volume"][0] = -1
            result["raw_paths"][0]["assets"]["BTC"]["amount"][0] = -100
            result["forecast"]["volume_quality"] = build_volume_quality(result["raw_paths"], result["forecast"]["times"])
            return result
        self.worker.predict = negative
        state, _ = self.run_job(force_run=True)
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["metrics"]["summary"], first["metrics"]["summary"])
        quality = state["forecast"]["volume_quality"]
        self.assertEqual(quality["volume_invalid_count"], 1)
        self.assertEqual(quality["total_candles"], 90)
        self.assertEqual(quality["status"], "volume_forecast_unavailable")
        self.assertEqual(quality["records"][0]["volume"], -1)
        self.assertFalse(quality["records"][0]["volume_valid"])

    def test_evaluation_aggregation_failure_does_not_reject_prediction(self):
        def fail(*_):
            raise ValueError("explicit injected aggregation failure")
        self.app._refresh_evaluation = fail
        state, _ = self.run_job()
        self.assertEqual(state["status"], "ready")
        self.assertIsNone(state["error"])
        self.assertIsNotNone(state["metrics"])
        self.assertEqual(state["evaluation"]["aggregation_status"], "failed")
        fid = state["forecast"]["forecast_id"]
        self.assertEqual(self.app.store.get("evaluation:" + fid)["status"], "scored")

    def test_evaluation_profile_isolation(self):
        self.app.scorer = lambda window, *_: {"status": "scored", "window_id": window["window_id"],
            "rows": [{"source": window["source"]}]}
        self.run_job()
        state, _ = self.run_job(profile_id="kraken_live")
        self.assertEqual(state["evaluation"]["sample_count"], 1)
        self.assertEqual([r["source"] for r in state["evaluation"]["rows"]], ["explicit_synthetic_test:kraken_live"])

    def test_ready_edits_work_during_scoring(self):
        started, finish = threading.Event(), threading.Event()
        def blocked(*_):
            started.set()
            finish.wait(5)
            return {"status": "scored", "rows": []}
        self.app.scorer = blocked
        self.app.analyze({"request_id": "blocking", "profile_id": "binance_jan2025", "expected_view_revision": 0})
        try:
            self.assertTrue(started.wait(2))
            state = self.app.view()
            self.assertEqual(state["status"], "ready")
            changed = self.app.alerts({"drawdown_threshold": .04, "top_k": 1, "expected_view_revision": state["view_revision"]})
            self.assertEqual(changed["policy"]["top_k"], 1)
        finally:
            finish.set()
            self.app.wait()
        with self.assertRaises(APIError) as caught:
            self.app.alerts({"drawdown_threshold": .1, "top_k": 1, "expected_view_revision": 0})
        self.assertEqual(caught.exception.status, 409)

    def test_restart_is_readonly_and_cache_persists(self):
        s, _ = self.run_job()
        self.app.close()
        self.worker = FakeWorker()
        self.app = Application(self.tmp.name, self.worker, fixture_loader)
        self.assertEqual(self.app.view()["forecast"], s["forecast"])
        self.assertEqual(self.worker.calls, 0)
        restored, _ = self.run_job()
        self.assertEqual(restored["forecast"], s["forecast"])
        self.assertEqual(self.worker.calls, 0)

    def test_http_readonly_security_and_invalid_json(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            before = self.app.view()
            for _ in range(2):
                with urlopen(base + "/api/state") as response:
                    self.assertEqual(json.load(response), before)
            for path in ("/.env", "/../CONTRACT.md", "/scenario-lab/models/Kronos-base/config.json"):
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base + path)
                self.assertEqual(caught.exception.code, 404)
            req = Request(base + "/api/analyze", data=b"{}", headers={"Content-Type": "application/json", "Origin": "https://example.invalid"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(req)
            self.assertEqual(caught.exception.code, 403)
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + "/api/state", headers={"Host": "evil.invalid"}))
            self.assertEqual(caught.exception.code, 403)
            req = Request(base + "/api/analyze", data=b'{"x":NaN}', headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(req)
            self.assertEqual(caught.exception.code, 400)
            self.assertEqual(self.worker.calls, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
