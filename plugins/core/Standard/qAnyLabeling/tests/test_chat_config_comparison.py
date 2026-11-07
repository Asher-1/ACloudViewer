"""Exercise the real comparator with captured native fixtures and a pinned source."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ChatConfigComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ["QANY_CHAT_UPSTREAM"]).resolve(strict=True)
        cls.artifacts = Path(os.environ["QANY_CHAT_ARTIFACTS"]).resolve(strict=True)
        cls.script = Path(__file__).with_name("compare_chat_config_upstream.py")
        cls.defaults = json.loads(
            (cls.artifacts / "chat-config-defaults-native.json").read_text())

    def compare(self, mutate=None):
        defaults = copy.deepcopy(self.defaults)
        if mutate is not None:
            mutate(defaults["providers"])
        with tempfile.TemporaryDirectory(prefix="qal-provider-gate-") as name:
            target = Path(name)
            for filename in (
                "chat-config-roundtrip-native.json",
                "chat-capabilities-native.json",
                "chat-model-search-native.json",
                "chat-model-favorite-native.json",
                "chat-model-refresh-native.json",
                "chat-model-geometry-native.json",
            ):
                (target / filename).symlink_to(self.artifacts / filename)
            (target / "chat-config-defaults-native.json").write_text(
                json.dumps(defaults), encoding="utf-8")
            env = dict(os.environ)
            env.pop("LD_LIBRARY_PATH", None)
            result = subprocess.run(
                [sys.executable, str(self.script), "--upstream",
                 str(self.upstream), "--artifacts", str(target)],
                env=env, text=True, capture_output=True)
            receipt = target / "chat-config-upstream.json"
            return result, json.loads(receipt.read_text()) if receipt.exists() else None

    def test_source_defaults_and_valid_local_extension_pass(self):
        result, receipt = self.compare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertCountEqual(receipt["local_provider_checks"],
                              ["api_address", "api_key", "api_docs_url",
                               "api_key_url", "model_docs_url"])

    def test_changed_upstream_address_is_rejected(self):
        result, _ = self.compare(
            lambda providers: providers["openai"].update(api_address="invalid"))
        self.assertNotEqual(result.returncode, 0)

    def test_missing_upstream_provider_is_rejected(self):
        result, _ = self.compare(lambda providers: providers.pop("openai"))
        self.assertNotEqual(result.returncode, 0)

    def test_missing_local_provider_is_rejected(self):
        result, _ = self.compare(lambda providers: providers.pop("local"))
        self.assertNotEqual(result.returncode, 0)

    def test_unexpected_provider_is_rejected(self):
        result, _ = self.compare(
            lambda providers: providers.update(unregistered={"api_address": ""}))
        self.assertNotEqual(result.returncode, 0)

    def test_wrong_local_address_is_rejected(self):
        result, _ = self.compare(
            lambda providers: providers["local"].update(api_address="aicore://wrong"))
        self.assertNotEqual(result.returncode, 0)

    def test_local_credentials_are_rejected(self):
        result, _ = self.compare(
            lambda providers: providers["local"].update(api_key="nonempty-test-value"))
        self.assertNotEqual(result.returncode, 0)

    def test_unexpected_local_field_is_rejected(self):
        result, _ = self.compare(
            lambda providers: providers["local"].update(ignored_fact=True))
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
