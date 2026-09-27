"""Offline contract checks for live caching, frozen jobs and loopback routes.

All market/model values are synthetic. No sockets, historical files or model
weights are opened. Run: python -m unittest ripple_live.test_service -v
"""
import copy
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
from statistics import pstdev
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from model_adapter.adapter import assemble_forecast
from forecast_metrics.engine import freeze_forecast
from ripple_live import server
from ripple_live.service import LiveError, LiveService


ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]


def market_fixture(window_id="window-a", *, age_seconds=30):
    anchor = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    stamp = lambda value: value.isoformat().replace("+00:00", "Z")
    histories = {}
    for index, asset in enumerate(ASSETS):
        price = 100.0 + index
        histories[asset] = [
            {"time": stamp(anchor - timedelta(minutes=255 - i)),
             "open": price, "high": price, "low": price, "close": price,
             "volume": 1.0, "amount": price}
            for i in range(256)
        ]
    window = {"schema_version": 1, "window_id": window_id,
              "profile_id": "synthetic-test", "mode": "live", "source": "synthetic-test",
              "quote_currency": "USDT", "interval_seconds": 60,
              "assets": ASSETS[:], "as_of": stamp(anchor), "histories": histories}
    snapshot = {"schema_version": 1, "window_id": window_id, "source": "synthetic-test",
                "quote_currency": "USDT", "as_of": window["as_of"],
                "fetched_at": stamp(datetime.now(timezone.utc)), "assets": ASSETS[:],
                "metrics": [{"asset": a, "vol_past30": 0.0, "mdd_past30": 0.0,
                             "last_price": histories[a][-1]["close"], "return_past30": 0.0}
                            for a in ASSETS],
                "series": {a: [{"time": b["time"], "close": b["close"]}
                               for b in histories[a][-61:]] for a in ASSETS},
                "stale": False, "error": None, "age_seconds": age_seconds}
    return {"window": window, "snapshot": snapshot}


def synthetic_output(window, config, run_id):
    assets = {}
    for asset in window["assets"]:
        spot = window["histories"][asset][-1]["close"]
        closes = [spot * factor for factor in ([1.01, .99, .98] + [1.] * 27)]
        assets[asset] = {"open": closes[:], "close": closes[:], "high": closes[:],
                         "low": closes[:], "volume": [-1.] + [1.] * 29,
                         "amount": [-spot] + [spot] * 29}
    raw = [{"path_id": "path-0", "assets": assets}]
    forecast = assemble_forecast(window, config, run_id, raw,
                                 model_revision="synthetic-test-weights")
    return {"forecast": forecast, "raw_paths": raw}


class DeferredThread:
    """Hold a submitted job so races can be tested deterministically."""
    def __init__(self, *, target, args, daemon):
        self.target, self.args, self.daemon = target, args, daemon
        self.started = False

    def start(self):
        self.started = True

    def finish(self):
        if not self.started:
            raise AssertionError("job was never started")
        self.target(*self.args)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tick = 100.0
        self.addCleanup(patch.stopall)
        patch("ripple_live.service.time.monotonic", side_effect=lambda: self.tick).start()
        self.jobs = []

        def schedule(**kwargs):
            job = DeferredThread(**kwargs)
            self.jobs.append(job)
            return job

        patch("ripple_live.service.threading.Thread", side_effect=schedule).start()
        self.packet = market_fixture()
        self.fetcher = Mock(return_value=self.packet)
        self.worker = Mock()
        self.worker.predict.side_effect = synthetic_output
        self.service = LiveService(fetcher=self.fetcher, worker=self.worker,
                                   directory=Path(self.temp.name) / "artifacts")

    def refresh(self, packet):
        self.tick += 21
        self.fetcher.return_value = packet
        return self.service.market()

    def test_market_cache_preserves_identity_and_owns_returned_values(self):
        first = self.service.market()
        first["series"]["BTC"][0]["close"] = -900
        self.packet["snapshot"]["metrics"][0]["last_price"] = -800
        self.packet["window"]["histories"]["BTC"][-1]["close"] = -700
        second = self.service.market()
        self.assertEqual(self.fetcher.call_count, 1)
        self.assertEqual(second["window_id"], "window-a")
        self.assertEqual(second["series"]["BTC"][0]["close"], 100.)
        self.assertEqual(second["metrics"][0]["last_price"], 100.)
        self.assertEqual(self.service.windows["window-a"]["histories"]["BTC"][-1]["close"], 100.)
        self.worker.predict.assert_not_called()

    def test_failed_refresh_retains_observations_with_visible_stale_error(self):
        before = self.service.market()
        self.tick += 21
        self.fetcher.side_effect = TimeoutError("private provider details")
        after = self.service.market()
        self.assertEqual(after["window_id"], before["window_id"])
        self.assertEqual(after["series"], before["series"])
        self.assertTrue(after["stale"])
        self.assertIn("TimeoutError", after["error"])
        self.assertNotIn("private provider details", after["error"])
        with self.assertRaises(LiveError) as caught:
            self.service.predict(before["window_id"])
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(self.jobs, [])

    def test_first_fetch_failure_is_503_not_empty_or_fabricated_snapshot(self):
        self.fetcher.side_effect = TimeoutError()
        with self.assertRaises(LiveError) as caught:
            self.service.market()
        self.assertEqual(caught.exception.status, 503)
        self.assertIsNone(self.service.snapshot)

    def test_successful_refresh_clears_connection_error(self):
        self.service.market()
        self.tick += 21
        self.fetcher.side_effect = TimeoutError()
        self.assertTrue(self.service.market()["stale"])
        self.fetcher.side_effect = None
        fresh = self.refresh(market_fixture("window-b"))
        self.assertFalse(fresh["stale"])
        self.assertIsNone(fresh["error"])

    def test_unknown_expired_and_old_windows_cannot_start_model(self):
        self.service.market()
        for window_id in (None, [], "not-fetched"):
            with self.subTest(window_id=window_id), self.assertRaises(LiveError) as caught:
                self.service.predict(window_id)
            self.assertEqual(caught.exception.status, 409)
        stale = self.refresh(market_fixture("old", age_seconds=181))
        self.assertTrue(stale["stale"])
        with self.assertRaises(LiveError):
            self.service.predict("old")
        self.assertEqual(self.jobs, [])

    def test_window_cache_is_bounded_and_evicted_id_rejected(self):
        self.service.market()
        for number in range(8):
            self.refresh(market_fixture(f"window-{number}"))
        self.assertLessEqual(len(self.service.windows), 8)
        with self.assertRaises(LiveError) as caught:
            self.service.predict("window-a")
        self.assertEqual(caught.exception.status, 409)

    def test_prediction_captures_input_and_cannot_be_retargeted_by_refresh(self):
        self.service.market()
        self.assertEqual(self.service.predict("window-a")["status"], "running")
        captured = copy.deepcopy(self.jobs[0].args[0])
        self.service.windows["window-a"]["histories"]["BTC"][-1]["close"] = 9999
        self.refresh(market_fixture("window-b"))
        self.assertEqual(self.jobs[0].args[0], captured)
        self.assertEqual(self.service.prediction_state()["window_id"], "window-a")
        self.jobs[0].finish()
        self.assertEqual(self.service.prediction_state()["window_id"], "window-a")
        self.assertEqual(self.worker.predict.call_args.args[0], captured)

    def test_same_request_deduplicates_running_and_different_request_conflicts(self):
        self.service.market()
        first = self.service.predict("window-a")
        self.assertEqual(self.service.predict("window-a"), first)
        self.refresh(market_fixture("window-b"))
        with self.assertRaises(LiveError) as caught:
            self.service.predict("window-b")
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(len(self.jobs), 1)

    def test_public_result_uses_real_arithmetic_and_keeps_audited_volume_private(self):
        self.service.market()
        self.service.predict("window-a")
        self.jobs[0].finish()
        state = self.service.prediction_state()
        self.assertEqual(state["status"], "ready")
        self.assertEqual(set(state["assets"]), set(ASSETS))
        self.assertEqual(state["path_count"], 1)
        self.assertEqual(len(state["times"]), 30)
        btc = state["assets"]["BTC"]
        self.assertEqual(btc["close"], [101., 99., 98.] + [100.] * 27)
        self.assertAlmostEqual(btc["mdd"], 3 / 101)
        prices = [100., *btc["close"]]
        vol = pstdev([math.log(b) - math.log(a) for a, b in zip(prices, prices[1:])]) * math.sqrt(30)
        self.assertAlmostEqual(btc["volatility"], vol)
        self.assertEqual(set(btc), {"close", "mdd", "volatility"})
        records = list(self.service.directory.glob("*.json"))
        self.assertEqual(len(records), 1)
        artifact = json.loads(records[0].read_text())
        self.assertEqual(state["forecast_id"], freeze_forecast(artifact["result"]["forecast"]).identity)
        self.assertEqual(artifact["result"]["forecast"]["volume_quality"]["volume_invalid_count"], 10)
        state["assets"]["BTC"]["close"][0] = -1
        self.assertEqual(self.service.prediction_state()["assets"]["BTC"]["close"][0], 101.)

    def test_completed_windows_are_reused_even_after_another_forecast(self):
        self.service.market()
        self.service.predict("window-a")
        self.jobs[-1].finish()
        original = self.service.prediction_state()
        self.refresh(market_fixture("window-b"))
        self.service.predict("window-b")
        self.jobs[-1].finish()
        self.assertEqual(self.service.predict("window-a"), original)
        self.assertEqual(len(self.jobs), 2, "same frozen input must never resample")

    def test_failed_sample_remains_failed_after_another_window_runs(self):
        self.worker.predict.side_effect = ValueError("invalid sampled prices")
        self.service.market()
        self.service.predict("window-a")
        self.jobs[-1].finish()
        failed = self.service.prediction_state()
        self.assertEqual(failed["status"], "error")
        self.assertEqual(self.service.predict("window-a"), failed)
        self.worker.predict.side_effect = synthetic_output
        self.refresh(market_fixture("window-b"))
        self.service.predict("window-b")
        self.jobs[-1].finish()
        self.assertEqual(self.service.predict("window-a"), failed)
        self.assertEqual(len(self.jobs), 2, "failed A must not be resampled after B")

    def test_invalid_price_output_rejected_but_market_remains_available(self):
        def bad_output(*args):
            result = synthetic_output(*args)
            result["forecast"]["paths"][0]["assets"]["BTC"]["close"][0] = -1
            return result
        self.worker.predict.side_effect = bad_output
        snapshot = self.service.market()
        self.service.predict("window-a")
        self.jobs[0].finish()
        state = self.service.prediction_state()
        self.assertEqual(state["status"], "error")
        self.assertNotIn("assets", state)
        self.assertEqual(self.service.market()["series"], snapshot["series"])

    def test_unpersisted_forecast_is_never_published(self):
        self.service.directory.write_text("not a directory")
        self.service.market()
        self.service.predict("window-a")
        self.jobs[0].finish()
        state = self.service.prediction_state()
        self.assertEqual(state["status"], "error")
        self.assertNotIn("forecast_id", state)


class FakeConnection:
    def __init__(self, request):
        self.input = io.BytesIO(request)
        self.output = bytearray()

    def makefile(self, mode, *_):
        if mode != "rb":
            raise AssertionError("unexpected socket mode")
        return self.input

    def sendall(self, value):
        self.output.extend(value)


class RouteTests(unittest.TestCase):
    """Run BaseHTTPRequestHandler against bytes, without opening a socket."""
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.web = (Path(self.temp.name) / "web").resolve()
        self.web.mkdir()
        (self.web / "index.html").write_text("<main>synthetic test page</main>")
        (self.web / "data").mkdir()
        (self.web / "data" / "example.json").write_text('{"fixture":true}')
        outside = Path(self.temp.name) / "outside.json"
        outside.write_text('{"private":"must not be served"}')
        (self.web / "escape.json").symlink_to(outside)
        self.web_patch = patch.object(server, "WEB", self.web)
        self.web_patch.start()
        self.addCleanup(self.web_patch.stop)
        self.service = Mock()
        self.service.market.return_value = {"window_id": "window-a", "stale": False}
        self.service.prediction_state.return_value = {"status": "idle", "error": None}
        self.service.predict.return_value = {"status": "running", "window_id": "window-a"}

    def request(self, path, *, method="GET", headers=None, payload=b""):
        fields = {"Host": "127.0.0.1:5176", **(headers or {})}
        if method == "POST":
            fields.setdefault("Content-Type", "application/json")
            fields.setdefault("Content-Length", str(len(payload)))
        text = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in fields.items()) + "\r\n"
        connection = FakeConnection(text.encode() + payload)
        server.make_handler(self.service)(connection, ("127.0.0.1", 43210), SimpleNamespace(server_port=5176))
        headers_raw, body = bytes(connection.output).split(b"\r\n\r\n", 1)
        status = int(headers_raw.split(b" ")[1])
        return status, headers_raw.decode(), body

    def test_only_declared_static_directories_and_suffixes_are_served(self):
        self.assertEqual(self.request("/")[0], 200)
        self.assertEqual(self.request("/data/example.json")[0], 200)
        for path in ("/../outside.json", "/%2e%2e/outside.json", "/escape.json",
                     "/data/../index.html", "/.env", "/service.py", "/runtime/forecast.json",
                     "/data/nested/private.json", "/api/unknown"):
            with self.subTest(path=path):
                status, _, body = self.request(path)
                self.assertEqual(status, 404)
                self.assertNotIn(b"must not be served", body)

    def test_host_and_origin_reject_nonlocal_requests_before_model_or_feed(self):
        self.assertEqual(self.request("/api/live", headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.request("/api/predict", method="POST",
            headers={"Origin": "https://attacker.example"}, payload=b'{"window_id":"window-a"}')[0], 403)
        self.service.market.assert_not_called()
        self.service.predict.assert_not_called()

    def test_live_errors_and_prediction_status_are_not_cached_by_browser(self):
        status, headers, body = self.request("/api/prediction")
        self.assertEqual(status, 200)
        self.assertIn("Cache-Control: no-store", headers)
        self.assertEqual(json.loads(body)["status"], "idle")
        self.service.market.side_effect = LiveError("first load unavailable", 503)
        status, _, body = self.request("/api/live")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"], "first load unavailable")

    def test_predict_accepts_only_bounded_json_window_request(self):
        payload = b'{"window_id":"window-a"}'
        status, _, body = self.request("/api/predict", method="POST", payload=payload)
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(body)["window_id"], "window-a")
        self.service.predict.assert_called_once_with("window-a")
        for bad in (b"", b"{}", b"[]", b"not json", b'{"window_id":"a","extra":1}'):
            with self.subTest(payload=bad):
                self.assertEqual(self.request("/api/predict", method="POST", payload=bad)[0], 400)
        self.assertEqual(self.request("/api/predict", method="POST", payload=payload,
                                     headers={"Content-Length": "2048"})[0], 400)
        self.assertEqual(self.request("/api/predict", method="POST", payload=payload,
                                     headers={"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(self.request("/api/other", method="POST", payload=payload)[0], 404)
        self.assertEqual(self.service.predict.call_count, 1)

    def test_prediction_conflict_is_409(self):
        self.service.predict.side_effect = LiveError("expired window", 409)
        status, _, body = self.request("/api/predict", method="POST", payload=b'{"window_id":"old"}')
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(body)["error"], "expired window")


if __name__ == "__main__":
    unittest.main()
