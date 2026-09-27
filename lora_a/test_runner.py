"""Synthetic tests of the pre-unseal identity boundary; no market/model access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lora_a.runner import accept_selected, reference, verify_initial_identities


class IdentityBoundaryTests(unittest.TestCase):
    def test_changed_code_blocks_acceptance_before_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "code.py"
            artifact.write_text("original\n")
            ref = reference(artifact)
            environment = {"code": [ref], "shared_code": [], "config": ref,
                           "base_model": ref, "tokenizer": ref, "packages": {}}
            (root / "environment.json").write_text(json.dumps(environment))
            verify_initial_identities(root)
            artifact.write_text("changed after training\n")
            with patch("lora_a.data.load_month") as load:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    accept_selected(root)
                load.assert_not_called()

    def test_changed_package_blocks_acceptance_before_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "code.py"
            artifact.write_text("original\n")
            ref = reference(artifact)
            (root / "environment.json").write_text(json.dumps({"code": [ref], "shared_code": [],
                  "config": ref, "base_model": ref, "tokenizer": ref, "packages": {"fictional": "1.0"}}))
            with patch("lora_a.runner.importlib.metadata.version", return_value="2.0"), patch("lora_a.data.load_month") as load:
                with self.assertRaisesRegex(RuntimeError, "package changed"):
                    accept_selected(root)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
