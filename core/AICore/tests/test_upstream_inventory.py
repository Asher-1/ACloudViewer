"""Exercise upstream inventory extraction and fail-closed alignment checks."""
import copy
import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools/upstream_inventory.py"


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), "The executable upstream inventory tool is missing")
        spec = importlib.util.spec_from_file_location("upstream_inventory", TOOL)
        self.inventory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.inventory)

    def anylabeling_sources(self):
        return {
            "anylabeling/configs/models.yaml": (
                '- model_name: florence-large\n  config_file: ":/florence.yaml"\n'
                '- model_name: tiny-ocr\n  config_file: ":/tiny.yaml"\n'),
            "anylabeling/configs/auto_labeling/florence.yaml": (
                'type: florence2\nname: florence-large-ft\n'
                'model_path: microsoft/Florence-2-large-ft\n'
                'max_new_tokens: 1024\ndo_sample: false\nnum_beams: 3\n'),
            "anylabeling/configs/auto_labeling/tiny.yaml": (
                'type: ppocr_v6\nname: tiny-ocr\ndet_model_path: https://example.org/det.onnx\n'
                'rec_model_path: https://example.org/rec.onnx\nrec_char_dict_path: dict.txt\n'
                'use_angle_cls: true\ndrop_score: 0.5\n'),
            "anylabeling/configs/auto_labeling/retired.yaml": 'type: retired\n',
            "anylabeling/services/auto_labeling/model_manager.py": (
                'def _get_model_class(model_type):\n'
                '    if model_type == "florence2":\n'
                '        from .florence2 import Florence2\n        return Florence2\n'
                '    elif model_type == "ppocr_v6":\n'
                '        from .ppocr_v6 import PPOCRv6\n        return PPOCRv6\n'
                '    elif model_type == "custom_only":\n'
                '        from .custom import Custom\n        return Custom\n'),
            "anylabeling/services/auto_labeling/florence2.py": (
                'raise RuntimeError("Extraction must not execute upstream code")\n'
                'TASK_MAPPING = {"caption": "<CAPTION>", "ocr": "<OCR>"}\n'),
            "anylabeling/services/auto_labeling/ppocr_v6.py": 'class PPOCRv6: pass\n',
            "anylabeling/services/auto_labeling/custom.py": 'class Custom: pass\n',
        }

    def rows(self):
        sources = self.anylabeling_sources()
        return self.inventory.extract_anylabeling(sources.__getitem__)["models"]

    def test_active_registry_excludes_retired_yaml_and_keeps_registry_alias(self):
        result = self.inventory.extract_anylabeling(self.anylabeling_sources().__getitem__)
        self.assertEqual([row["id"] for row in result["models"]],
                         ["anylabeling/florence-large", "anylabeling/tiny-ocr"])
        self.assertEqual(result["models"][0]["config_name"], "florence-large-ft")
        self.assertEqual([row["family"] for row in result["custom_loaders"]], ["custom_only"])
        self.assertEqual(result["models"][0]["task_tokens"],
                         {"caption": "<CAPTION>", "ocr": "<OCR>"})

    def test_defaults_and_assets_preserve_beams_false_and_dependency_roles(self):
        rows = self.rows()
        self.assertEqual(rows[0]["defaults"],
                         {"max_new_tokens": 1024, "do_sample": False, "num_beams": 3})
        self.assertEqual(rows[0]["assets"],
                         {"model_path": "microsoft/Florence-2-large-ft"})
        self.assertEqual(rows[1]["assets"], {
            "det_model_path": "https://example.org/det.onnx",
            "rec_model_path": "https://example.org/rec.onnx",
            "rec_char_dict_path": "dict.txt"})
        self.assertEqual(rows[1]["defaults"]["drop_score"], 0.5)

    def test_duplicate_registered_model_is_rejected(self):
        sources = self.anylabeling_sources()
        sources["anylabeling/configs/models.yaml"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.inventory.extract_anylabeling(sources.__getitem__)

    def test_duplicate_yaml_key_is_rejected_before_it_can_hide_checkpoint(self):
        sources = self.anylabeling_sources()
        sources["anylabeling/configs/auto_labeling/florence.yaml"] += 'model_path: substitute\n'
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.inventory.extract_anylabeling(sources.__getitem__)

    def test_registered_model_without_loader_is_rejected(self):
        sources = self.anylabeling_sources()
        sources["anylabeling/configs/auto_labeling/tiny.yaml"] = 'type: unsupported\n'
        with self.assertRaisesRegex(ValueError, "loader"):
            self.inventory.extract_anylabeling(sources.__getitem__)

    def test_resource_path_traversal_is_rejected(self):
        sources = self.anylabeling_sources()
        sources["anylabeling/configs/models.yaml"] = (
            '- model_name: escaped\n  config_file: ":/../../private.yaml"\n')
        with self.assertRaisesRegex(ValueError, "config_file"):
            self.inventory.extract_anylabeling(sources.__getitem__)

    def test_converter_registry_keeps_text_and_projector_roles_without_import(self):
        text = ('raise RuntimeError("Do not import torch or upstream modules")\n'
                'TEXT_MODEL_MAP: dict[str, str] = {"Qwen2ForCausalLM": "qwen"}\n'
                'MMPROJ_MODEL_MAP = {"SmolVLMForConditionalGeneration": "smolvlm"}\n')
        rows = self.inventory.extract_converters(text)
        self.assertEqual([(row["role"], row["architecture"], row["module"]) for row in rows],
                         [("text", "Qwen2ForCausalLM", "qwen"),
                          ("mmproj", "SmolVLMForConditionalGeneration", "smolvlm")])

    def test_duplicate_converter_architecture_in_same_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.inventory.extract_converters(
                'TEXT_MODEL_MAP = {"A": "one", "A": "two"}\nMMPROJ_MODEL_MAP = {}\n')

    def test_architecture_registry_ignores_comment_and_marks_quantize_only_clip(self):
        text = ('static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n'
                ' { LLM_ARCH_CLIP, "clip" }, // dummy\n'
                ' // { LLM_ARCH_FAKE, "fake" },\n'
                ' { LLM_ARCH_QWEN2, "qwen2" },\n};\n'
                'const char *unrelated = "not-an-architecture";\n')
        rows = self.inventory.extract_architectures(text)
        self.assertEqual([(r["architecture"], r["loader_kind"]) for r in rows],
                         [("clip", "quantize-only"), ("qwen2", "model")])

    def test_cmake_tool_inventory_retains_nested_backend_conditions(self):
        rows = self.inventory.extract_cmake_subdirectories(
            'if (NOT EMSCRIPTEN)\nadd_subdirectory(quantize)\n'
            'if(GGML_RPC)\nadd_subdirectory(rpc)\nendif()\nendif()\n')
        self.assertEqual(rows, [
            {"directory": "quantize", "conditions": ["NOT EMSCRIPTEN"]},
            {"directory": "rpc", "conditions": ["NOT EMSCRIPTEN", "GGML_RPC"]}])

    def test_routes_keep_methods_and_conditional_alternative_handlers(self):
        rows = self.inventory.extract_routes(
            '// ctx_http.post("/fake", handler);\n'
            'ctx_http.get ("/health", ex_wrapper(routes.get_health));\n'
            'ctx_http.post("/v1/embeddings", ex_wrapper(routes.post_embeddings_oai));\n'
            'ctx_http.get("/cors-proxy", ex_wrapper(proxy_handler_get));\n'
            'ctx_http.get("/cors-proxy", ex_wrapper(res_403));\n')
        self.assertEqual(rows, [
            {"method": "GET", "path": "/health", "handlers": ["routes.get_health"]},
            {"method": "POST", "path": "/v1/embeddings", "handlers": ["routes.post_embeddings_oai"]},
            {"method": "GET", "path": "/cors-proxy", "handlers": ["proxy_handler_get", "res_403"]}])

    def test_loader_return_must_match_the_imported_class(self):
        sources = self.anylabeling_sources()
        path = "anylabeling/services/auto_labeling/model_manager.py"
        sources[path] = sources[path].replace("return Florence2", "return Substitute")
        with self.assertRaisesRegex(ValueError, "loader return"):
            self.inventory.extract_anylabeling(sources.__getitem__)

    def test_converter_classes_preserve_registered_aliases_examples_and_roles(self):
        self.assertTrue(callable(getattr(self.inventory, "extract_converter_classes", None)),
                        "Converter class registration extraction is missing")
        rows = self.inventory.extract_converter_classes(
            '@ModelBase.register("SmolVLM", "Idefics3")\n'
            '@ModelBase.example("HuggingFaceTB/SmolVLM-Instruct")\n'
            'class Vision(MmprojModel): pass\n'
            '@ModelBase.register("Qwen2")\nclass Text(TextModel): pass\n')
        self.assertEqual(rows, [
            {"class": "Vision", "bases": ["MmprojModel"], "architectures": ["SmolVLM", "Idefics3"],
             "examples": ["HuggingFaceTB/SmolVLM-Instruct"]},
            {"class": "Text", "bases": ["TextModel"], "architectures": ["Qwen2"], "examples": []}])

    def test_mtmd_inventory_uses_registered_library_sources_not_disk_presence(self):
        self.assertTrue(callable(getattr(self.inventory, "extract_mtmd_sources", None)),
                        "Registered mtmd source extraction is missing")
        rows = self.inventory.extract_mtmd_sources(
            'add_library(mtmd mtmd.cpp models/qwen3a.cpp models/llava.cpp)\n'
            '# models/not-built.cpp\nadd_executable(debug models/debug.cpp)\n')
        self.assertEqual(rows, ["tools/mtmd/models/qwen3a.cpp", "tools/mtmd/models/llava.cpp"])

    def test_cmake_else_and_elseif_keep_mutually_exclusive_conditions(self):
        rows = self.inventory.extract_cmake_subdirectories(
            'if(A)\nadd_subdirectory(one)\nelseif(B)\nadd_subdirectory(two)\n'
            'else()\nadd_subdirectory(three)\nendif()\n')
        self.assertEqual(rows[1]["conditions"], ["NOT ((A)) AND (B)"])
        self.assertEqual(rows[2]["conditions"], ["NOT ((A) OR (B))"])

    def test_coverage_does_not_accept_python_true_as_numeric_default_one(self):
        expected = self.rows()
        declared = self.declarations()
        declared[0]["defaults"] = dict(declared[0]["defaults"], do_sample=0)
        errors = self.inventory.compare_coverage(expected, declared)
        self.assertTrue(any("default" in error for error in errors), errors)

    def test_cli_never_overwrites_an_existing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            output.write_bytes(b"retained receipt")
            result = subprocess.run([sys.executable, str(TOOL), "--anylabeling", directory,
                                     "--llama-archive", directory, "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Never overwrite", result.stderr)
            self.assertEqual(output.read_bytes(), b"retained receipt")

    def test_archive_rejects_wrong_digest_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "wrong.tar.gz"
            archive.write_bytes(b"not the approved archive")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.inventory.ArchiveSource(archive)

    def declarations(self):
        return [dict(row, consumer="src/consumer.cpp", status="UNVERIFIED") for row in self.rows()]

    def test_configuration_match_without_real_consumer_remains_missing(self):
        declared = self.declarations()
        declared[0]["consumer"] = ""
        errors = self.inventory.compare_coverage(self.rows(), declared)
        self.assertTrue(any("consumer" in error for error in errors), errors)

    def test_missing_extra_duplicate_wrong_checkpoint_role_and_default_fail_closed(self):
        expected = self.rows()
        original = self.declarations()
        mutations = [
            (original[1:], "missing"),
            (original + [dict(original[0], id="anylabeling/unregistered")], "extra"),
            (original + [original[0]], "duplicate"),
        ]
        for key, value, fragment in [
            ("family", "smolvlm", "family"),
            ("assets", {"model_path": "SmolVLM"}, "checkpoint"),
            ("assets", {"projector_path": "microsoft/Florence-2-large-ft"}, "role"),
            ("defaults", {"max_new_tokens": 1024, "do_sample": False, "num_beams": 1}, "default"),
        ]:
            candidate = copy.deepcopy(original)
            candidate[0][key] = value
            mutations.append((candidate, fragment))
        for candidate, fragment in mutations:
            with self.subTest(fragment=fragment):
                errors = self.inventory.compare_coverage(expected, candidate)
                self.assertTrue(any(fragment in error.lower() for error in errors), errors)
        self.assertEqual(self.inventory.compare_coverage(expected, original), [])

    def manifest(self, det_url="https://example.org/det.onnx", det_sha="a" * 64,
                 scenario_inputs=("det", "rec", "images")):
        return {"input_assets": [
                    {"id": "det", "url": det_url, "sha256": det_sha},
                    {"id": "rec", "url": "https://example.org/rec.onnx", "sha256": "b" * 64},
                    {"id": "images", "url": "https://example.org/images.zip", "sha256": "c" * 64}],
                "scenarios": [{"id": "tiny-pipeline", "task": "ocr",
                                "input_ids": list(scenario_inputs)}]}

    def derive(self, manifest):
        rows = self.rows()
        declared = self.inventory.derive_manifest_coverage(rows, manifest)
        return {row["id"]: row for row in declared}, self.inventory.compare_coverage(rows, declared)

    def test_exact_source_dependencies_are_declarations_not_execution_evidence(self):
        declared, errors = self.derive(self.manifest())
        self.assertEqual(declared["anylabeling/tiny-ocr"]["declared_consumer"], "manifest:tiny-pipeline")
        self.assertEqual(declared["anylabeling/tiny-ocr"]["binding"]["unverified_local_roles"],
                         ["rec_char_dict_path"])
        # A Hugging Face repository id has no pinned revision or digest here.
        self.assertIsNone(declared["anylabeling/florence-large"]["consumer"])
        self.assertCountEqual(errors, ["missing inference consumer: anylabeling/florence-large",
                                       "missing inference consumer: anylabeling/tiny-ocr"])

    def test_input_ids_alone_never_prove_an_inference_consumer(self):
        declared, errors = self.derive(self.manifest())
        self.assertIsNone(declared["anylabeling/tiny-ocr"]["consumer"])
        self.assertIn("missing inference consumer: anylabeling/tiny-ocr", errors)

    def test_alias_digest_claim_without_local_bytes_is_not_bound(self):
        manifest = self.manifest(det_url="https://example.org/v3/det.onnx")
        alias = {"url": "https://example.org/det.onnx", "sha256": "a" * 64}
        rows = self.inventory.derive_manifest_coverage(self.rows(), manifest, aliases=[alias])
        self.assertIsNone(rows[1]["binding"]["roles"]["det_model_path"])

    def test_alias_with_corrupt_local_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "det.onnx"
            path.write_bytes(b"corrupt model")
            alias = {"url": "https://example.org/det.onnx", "sha256": "a" * 64,
                     "verified_file": str(path), "bytes": path.stat().st_size}
            with self.assertRaisesRegex(ValueError, "alias.*SHA-256"):
                self.inventory.derive_manifest_coverage(
                    self.rows(), self.manifest(det_url="https://example.org/v3/det.onnx"), aliases=[alias])

    def test_substitute_checkpoint_with_same_file_name_does_not_bind(self):
        declared, errors = self.derive(self.manifest(det_url="https://mirror.example/det.onnx"))
        self.assertIsNone(declared["anylabeling/tiny-ocr"]["consumer"])
        self.assertIsNone(declared["anylabeling/tiny-ocr"]["binding"]["roles"]["det_model_path"])
        self.assertIn("missing inference consumer: anylabeling/tiny-ocr", errors)

    def test_scenario_covering_only_some_roles_is_partial_not_a_consumer(self):
        declared, _ = self.derive(self.manifest(scenario_inputs=("det", "images")))
        binding = declared["anylabeling/tiny-ocr"]["binding"]
        self.assertIsNone(declared["anylabeling/tiny-ocr"]["consumer"])
        self.assertEqual(binding["partial_scenarios"], ["tiny-pipeline"])

    def test_source_pin_without_valid_digest_does_not_bind(self):
        for digest in ["", "A" * 64, "a" * 63, None]:
            with self.subTest(digest=digest):
                declared, _ = self.derive(self.manifest(det_sha=digest))
                self.assertIsNone(declared["anylabeling/tiny-ocr"]["consumer"])

    def test_conflicting_pins_for_one_source_url_are_rejected(self):
        manifest = self.manifest()
        manifest["input_assets"].append(
            {"id": "det-copy", "url": "https://example.org/det.onnx", "sha256": "d" * 64})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.inventory.derive_manifest_coverage(self.rows(), manifest)

    def test_identical_byte_alias_binds_only_to_the_pin_with_the_same_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "det.onnx"
            path.write_bytes(b"same checkpoint bytes")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = self.manifest(det_url="https://example.org/v3/det.onnx", det_sha=digest)
            rows = self.rows()
            alias = {"url": "https://example.org/det.onnx", "sha256": digest,
                     "verified_file": str(path), "bytes": 21}
            declared = {row["id"]: row for row in
                        self.inventory.derive_manifest_coverage(rows, manifest, aliases=[alias])}
            self.assertEqual(declared["anylabeling/tiny-ocr"]["binding"]["roles"]["det_model_path"], "det")
            self.assertEqual(declared["anylabeling/tiny-ocr"]["declared_consumer"], "manifest:tiny-pipeline")
            unknown = dict(alias, sha256="e" * 64)
            declared = {row["id"]: row for row in
                        self.inventory.derive_manifest_coverage(rows, manifest, aliases=[unknown])}
            self.assertIsNone(declared["anylabeling/tiny-ocr"]["declared_consumer"])
            truncated = dict(alias, bytes=20)
            with self.assertRaisesRegex(ValueError, "alias bytes"):
                self.inventory.derive_manifest_coverage(rows, manifest, aliases=[truncated])

    def test_alias_that_contradicts_a_direct_pin_is_rejected(self):
        alias = {"url": "https://example.org/rec.onnx", "sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.inventory.derive_manifest_coverage(self.rows(), self.manifest(), aliases=[alias])

    def test_cli_rejects_mixing_hand_declarations_with_manifest_derivation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            result = subprocess.run([sys.executable, str(TOOL), "--anylabeling", directory,
                                     "--llama-archive", directory, "--coverage", directory,
                                     "--manifest", directory, "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not allowed with", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
