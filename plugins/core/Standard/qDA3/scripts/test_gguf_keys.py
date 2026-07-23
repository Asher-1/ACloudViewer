import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.gen_gguf_keys_header as G


def test_canonical_header_matches_source():
    canonical = (REPO / "core/AICore/src/depth/depth_gguf_keys.h").read_text()
    assert canonical == G.render_canonical(), (
        "depth_gguf_keys.h is stale; run scripts/gen_gguf_keys_header.py"
    )
    assert 'AICORE_DEPTH_KV_VIT_EMBED_DIM "depthanything3.vit.embed_dim"' in canonical
    assert 'AICORE_DEPTH_ARCH "depthanything3"' in canonical


def test_head_max_depth_key_present():
    import scripts.gguf_keys as K
    assert K.KV["head.max_depth"] == "depthanything3.head.max_depth"


def test_header_has_max_depth_macro():
    h = (REPO / "core/AICore/src/depth/depth_gguf_keys.h").read_text()
    assert 'AICORE_DEPTH_KV_HEAD_MAX_DEPTH "depthanything3.head.max_depth"' in h
