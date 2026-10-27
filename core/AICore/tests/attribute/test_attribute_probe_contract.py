#!/usr/bin/env python3
"""真实公共 probe 的属性门回归；模型和 pinned source 参照由外部测试资产注入。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class AttributeProbeContract(unittest.TestCase):
    def test_wrong_source_attribute_fails_even_when_every_probability_matches(self):
        model = Path(os.environ['AICORE_TEST_ATTRIBUTE_PERSON_GGUF'])
        identity_path = Path(os.environ['AICORE_TEST_ATTRIBUTE_PERSON_REFERENCE'])
        identity = json.loads(identity_path.read_text())
        reference = Path(os.environ['AICORE_TEST_ATTRIBUTE_PROBABILITIES'])
        image = Path(os.environ['AICORE_TEST_ATTRIBUTE_IMAGE'])
        self.assertEqual(identity['attributes']['Gender'], 'Male', '要求 pinned source 的 bus 输入')
        identity['attributes']['Gender'] = 'Female'
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.json'
            source.write_text(json.dumps(identity))
            report = Path(folder) / 'probe.json'
            command = [sys.executable, str(Path(__file__).with_name('test_attribute_probe.py')),
                '--library', str(args.library.resolve()), '--model', str(model), '--image', str(image),
                '--reference', str(reference), '--reference-sha256', hashlib.sha256(reference.read_bytes()).hexdigest(),
                '--source-identity', str(source), '--source-identity-sha256', hashlib.sha256(source.read_bytes()).hexdigest(),
                '--backend', 'cpu', '--quant', 'f32', '--warmups', '1', '--repeats', '2', '--json-out', str(report)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertTrue(report.is_file(), completed.stdout + completed.stderr)
            result = json.loads(report.read_text())
            self.assertLess(result['maximum_output_error'], .001)
            self.assertFalse(result['attributes_match'])
            self.assertFalse(result['pass'])
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(result['source_sha256'], identity['source_sha256'])
            self.assertEqual(result['source_identity_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--library', type=Path, required=True)
    args, remainder = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *remainder])
