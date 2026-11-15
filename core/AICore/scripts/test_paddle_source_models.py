#!/usr/bin/env python3
"""Tests for paddle_source_models: the same-weight Paddle baseline resolver."""

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import paddle_source_models as psm


def make_archive(root: Path, params: bytes) -> Path:
    archive = root / "src.tar"
    with tarfile.open(archive, "w") as tar:
        for name, data in (("model_infer/inference.pdiparams", params),
                           ("model_infer/inference.pdmodel", b"program"),
                           ("model_infer/inference.pdiparams.info", b"info")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


class ProvisionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.yml = self.root / "inference.yml"
        self.yml.write_text("Global:\n  model_name: PP-OCRv4_mobile_det\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_matching_digest_yields_paddlex_model_dir(self):
        params = b"source-weights"
        archive = make_archive(self.root, params)
        target = psm.provision("ch_PP-OCRv4_det_infer",
                               hashlib.sha256(params).hexdigest(),
                               self.root / "cache", self.yml, archive.as_uri())
        self.assertEqual((target / "inference.pdiparams").read_bytes(), params)
        self.assertTrue((target / "inference.pdmodel").is_file())
        self.assertEqual((target / "inference.yml").read_text(), self.yml.read_text())
        self.assertFalse((target / "inference.pdiparams.info").exists())
        # A second call reuses the verified cache (no archive needed).
        archive.unlink()
        again = psm.provision("ch_PP-OCRv4_det_infer",
                              hashlib.sha256(params).hexdigest(),
                              self.root / "cache", self.yml, archive.as_uri())
        self.assertEqual(again, target)

    def test_digest_mismatch_fails_closed(self):
        archive = make_archive(self.root, b"other-checkpoint")
        with self.assertRaises(SystemExit) as raised:
            psm.provision("ch_PP-OCRv4_det_infer", "0" * 64, self.root / "cache",
                          self.yml, archive.as_uri())
        self.assertIn("!= GGUF ocr.source.sha256", str(raised.exception))

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(SystemExit):
            psm.provision("unknown_infer", "0" * 64, self.root / "cache", self.yml)


if __name__ == "__main__":
    unittest.main()
