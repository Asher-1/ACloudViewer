"""Compare native OCR discovery/page normalization with pinned upstream methods.

Consumes fixtures emitted by test_ppocr_document. No remote service, inference
or GUI acceptance is inferred from this protocol comparison.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, required=True)
args = parser.parse_args()
source = args.upstream / "anylabeling/views/labeling/ppocr/pipeline.py"
render = args.upstream / "anylabeling/views/labeling/ppocr/render.py"
fixture = args.artifacts / "remote-pipeline-contract.json"
data = json.loads(fixture.read_text())
methods = {
    "_is_ppocr_pipeline_model",
    "_model_display_name",
    "_collect_pipeline_models",
    "_select_pipeline_model",
    "_normalize_page_result",
    "_extract_prediction_page",
    "_normalize_block_data",
    "_normalize_points",
    "_sanitize_bbox",
    "_points_to_bbox",
}
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "PPOCRPipeline")
selected = [
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods
]
assert len(selected) == len(methods)
namespace = {
    "PPOCR_PIPELINE_CAPABILITY_KEY": "ppocr_pipeline",
    "PPOCRPipelineModel": lambda **kw: SimpleNamespace(**kw)
}
render_tree = ast.parse(render.read_text())
normalize = next(
    n for n in render_tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "normalize_block_label")
namespace["re"] = re
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=[
                ast.ImportFrom(module="__future__",
                               names=[ast.alias(name="annotations")],
                               level=0), normalize
            ],
                       type_ignores=[])), str(render), "exec"), namespace)
# Execute the actual upstream implementations; unrelated imports/UI are omitted.
selected_cls = ast.ClassDef(name="Pipeline",
                            bases=[],
                            keywords=[],
                            body=selected,
                            decorator_list=[])
module = ast.Module(body=[
    ast.ImportFrom(module="__future__",
                   names=[ast.alias(name="annotations")],
                   level=0), selected_cls
],
                    type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
pipeline = namespace["Pipeline"]()
pipeline.pipeline_model = "a-last"
pipeline._current_result_pipeline_model = lambda: "a-last"
pipeline.data_manager = SimpleNamespace(root_dir=Path("/fixture"))
models = pipeline._collect_pipeline_models(
    json.loads(data["models_response"])["data"])
actual_models = [{
    "model_id": m.model_id,
    "display_name": m.display_name
} for m in models]
assert actual_models == data["models"], (actual_models, data["models"])
page = pipeline._normalize_page_result(data["raw_result"],
                                       Path("/fixture/files/transparent.png"),
                                       100, 120)
blocks = page["prunedResult"]["parsing_res_list"]
page["markdown"]["text"] = "\n\n".join(
    b.get("block_content", "") for b in blocks if b.get("block_content"))
# The native transport also crops image-only blocks. Compare that map separately
# in its Qt test; normalization here precedes upstream _collect_block_image_paths.
actual = data["document"]["layoutParsingResults"][0]
actual["markdown"].pop("images", None)
assert page == actual, (page, actual)
assert pipeline._select_pipeline_model(models) == "a-last"
pipeline.pipeline_model = "missing"
assert pipeline._select_pipeline_model(models) == "z-first"
receipt = {
    "verdict": "PASS",
    "comparisons": [
        "ordered_capability_models", "preferred_model", "fallback_model",
        "page_and_block_normalization"
    ],
    "upstream_pipeline_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "upstream_render_sha256": hashlib.sha256(render.read_bytes()).hexdigest(),
    "native_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
}
(args.artifacts / "remote-pipeline-upstream-comparison.json"
).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
