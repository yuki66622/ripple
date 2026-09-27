"""Supervisor failures must not orphan its worker or lose exit evidence."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from lora_a.supervise import serve


class SupervisionTests(unittest.TestCase):
    def test_optional_memory_probe_timeout_does_not_abort_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "launch.json").write_text(json.dumps({"command": ["synthetic"], "cwd": tmp}))
            worker = Mock(pid=123, returncode=0)
            worker.poll.side_effect = [None, 0]
            assertion = Mock(pid=124)
            assertion.poll.return_value = 0
            with patch("lora_a.supervise.signal.signal"), patch("lora_a.supervise.os.getpgid", return_value=999), \
                 patch("lora_a.supervise.subprocess.Popen", side_effect=[worker, assertion]), \
                 patch("lora_a.supervise.subprocess.run", side_effect=subprocess.TimeoutExpired("ps", 5)):
                self.assertEqual(serve(p), 0)
            self.assertEqual(json.loads((p / "exit.json").read_text())["returncode"], 0)
            self.assertIn("monitor_error", json.loads((p / "resources.jsonl").read_text()))
            worker.terminate.assert_not_called()

    def test_assertion_start_failure_stops_worker_and_records_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "launch.json").write_text(json.dumps({"command": ["synthetic"], "cwd": tmp}))
            worker = Mock(pid=123, returncode=-15)
            worker.poll.return_value = None
            with patch("lora_a.supervise.signal.signal"), patch("lora_a.supervise.os.getpgid", return_value=999), \
                 patch("lora_a.supervise.subprocess.Popen", side_effect=[worker, OSError("synthetic assertion error")]):
                with self.assertRaises(OSError):
                    serve(p)
            worker.terminate.assert_called_once()
            worker.wait.assert_called_once()
            self.assertEqual(json.loads((p / "exit.json").read_text())["returncode"], -15)
            self.assertTrue((p / "supervisor-error.json").exists())


if __name__ == "__main__":
    unittest.main()
