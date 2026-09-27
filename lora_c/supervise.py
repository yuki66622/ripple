"""C-only detached supervisor with external deadline; never retries a worker."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def serve(control):
    """Own a separate session, file output, exit receipt and bounded assertion."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    spec = json.loads((control / "launch.json").read_text())
    write_new(control / "supervisor.json", {"pid": os.getpid(), "ppid": os.getppid(),
              "pgid": os.getpgrp(), "started_at": stamp(), "automatic_restart": False})
    worker = assertion = None
    received = []

    def on_signal(number, _frame):
        received.append({"signal": number, "at": stamp()})
        if worker is not None and worker.poll() is None:
            worker.send_signal(number)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with (control / "worker.log").open("x") as output, (control / "resources.jsonl").open("x") as resources:
        try:
            env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
            worker = subprocess.Popen(spec["command"], cwd=spec["cwd"], env=env,
                                      stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                      close_fds=True)
            write_new(control / "worker.json", {"pid": worker.pid, "supervisor_pid": os.getpid(),
                      "pgid": os.getpgid(worker.pid), "started_at": stamp(), "command": spec["command"]})
            assertion = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(worker.pid)],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=output, close_fds=True)
            write_new(control / "sleep-assertion.json", {"pid": assertion.pid, "worker_pid": worker.pid,
                      "scope": "prevent idle sleep only while this worker exists"})
            while worker.poll() is None:
                left = (datetime.fromisoformat(spec["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()
                if left <= 0:
                    raise TimeoutError("C external six-hour watchdog deadline reached; no restart")
                sample = {"at": stamp(), "pid": worker.pid}
                try:
                    result = subprocess.run(["/bin/ps", "-p", str(worker.pid), "-o", "rss=,vsz=,etime="],
                                            capture_output=True, text=True, timeout=5)
                    sample.update(rss_kib_vsz_kib_elapsed=result.stdout.strip(), ps_exit=result.returncode)
                except (subprocess.TimeoutExpired, OSError) as exc:
                    # Optional telemetry must not kill the supervisor or assertion.
                    sample["monitor_error"] = repr(exc)
                resources.write(json.dumps(sample) + "\n")
                resources.flush()
                try:
                    worker.wait(timeout=min(30, max(0.1, left)))
                except subprocess.TimeoutExpired:
                    pass
            code = worker.returncode
            write_new(control / "exit.json", {"at": stamp(), "returncode": code,
                      "signal": -code if code < 0 else None, "supervisor_signals": received,
                      "automatic_restart": False})
            return code if code >= 0 else 128 - code
        except BaseException as exc:
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=10)
            write_new(control / "supervisor-error.json", {"at": stamp(), "error": repr(exc),
                      "worker_returncode": worker.returncode if worker else None,
                      "supervisor_signals": received, "automatic_restart": False})
            if not (control / "exit.json").exists():
                write_new(control / "exit.json", {"at": stamp(), "returncode": worker.returncode if worker else None,
                          "supervisor_error": repr(exc), "automatic_restart": False})
            raise
        finally:
            if assertion is not None and assertion.poll() is None:
                assertion.terminate()
                assertion.wait(timeout=10)


def launch(control, command):
    if not command:
        raise ValueError("explicit worker command required")
    config = json.loads((ROOT / "lora_c/config.json").read_text())
    run = Path(command[command.index("--run-dir") + 1]).resolve()
    if not run.is_relative_to(ROOT / "lora_c/runs"):
        raise ValueError("supervised run must be C-local")
    start = json.loads((run / "run-start.json").read_text())["utc"] if (run / "run-start.json").exists() else stamp()
    deadline = datetime.fromisoformat(start) + timedelta(seconds=config["max_wall_seconds"])
    if deadline <= datetime.now(timezone.utc):
        raise TimeoutError("C deadline already expired")
    control.mkdir(parents=True, exist_ok=False)
    write_new(control / "launch.json", {"command": command, "cwd": str(ROOT),
              "launcher_pid": os.getpid(), "launcher_pgid": os.getpgrp(), "at": stamp(), "deadline_utc": deadline.isoformat()})
    with (control / "supervisor.log").open("x") as output:
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "serve",
                                  "--control-dir", str(control)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, close_fds=True,
                                 start_new_session=True)
    write_new(control / "launch-receipt.json", {"supervisor_pid": child.pid, "at": stamp(),
              "new_session_requested": True})
    print(json.dumps({"control": str(control), "supervisor_pid": child.pid}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["launch", "serve"])
    parser.add_argument("--control-dir", type=Path, required=True)
    known, command = parser.parse_known_args()
    if command[:1] == ["--"]:
        command = command[1:]
    control = known.control_dir.resolve()
    if known.mode == "launch":
        launch(control, command)
    else:
        if command:
            parser.error("serve takes no worker command")
        sys.exit(serve(control))
