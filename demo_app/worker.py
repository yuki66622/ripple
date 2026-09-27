"""Resident isolated local model process; timeout cannot poison the next request."""
import multiprocessing as mp
import queue
import threading
import time


def _serve(inbox, outbox):
    adapter = None
    while True:
        message = inbox.get()
        if message is None:
            return
        try:
            if adapter is None:
                from model_adapter import KronosAdapter
                adapter = KronosAdapter()
            if message["kind"] == "identity":
                value = adapter.identity()
            else:
                value = adapter.predict(message["window"], message["config"], message["prediction_run_id"])
            outbox.put({"ok": True, "value": value})
        except Exception as exc:
            # Parent maps this bounded type-only information to a public error.
            reply = {"ok": False, "type": type(exc).__name__}
            if hasattr(exc, "raw_paths"):
                reply["rejected_output"] = {"raw_paths": exc.raw_paths,
                                             "issues": getattr(exc, "issues", []),
                                             "runtime": getattr(exc, "runtime", {})}
            outbox.put(reply)


class WorkerError(RuntimeError):
    def __init__(self, reply):
        super().__init__("Model worker rejected output: " + reply["type"])
        self.rejected_output = reply.get("rejected_output")


class LocalWorker:
    def __init__(self, timeout=180):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.process = None
        self.context = mp.get_context("spawn")

    def _start(self):
        self.inbox, self.outbox = self.context.Queue(), self.context.Queue()
        self.process = self.context.Process(target=_serve, args=(self.inbox, self.outbox), daemon=True)
        self.process.start()

    def _call(self, message):
        with self.lock:
            if self.process is None or not self.process.is_alive():
                self._stop()
                self._start()
            self.inbox.put(message)
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    reply = self.outbox.get(timeout=min(.2, max(.001, deadline - time.monotonic())))
                    if not reply["ok"]:
                        raise WorkerError(reply)
                    return reply["value"]
                except queue.Empty:
                    if time.monotonic() >= deadline or not self.process.is_alive():
                        self._stop()
                        raise TimeoutError("Model worker timed out or exited")

    def identity(self):
        return self._call({"kind": "identity"})

    def predict(self, window, config, prediction_run_id):
        return self._call({"kind": "predict", "window": window, "config": config,
                           "prediction_run_id": prediction_run_id})

    def _stop(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            for q in (self.inbox, self.outbox):
                q.close()
            self.process = None

    def close(self):
        with self.lock:
            self._stop()
