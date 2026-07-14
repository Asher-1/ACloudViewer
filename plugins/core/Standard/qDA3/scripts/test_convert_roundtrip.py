import os, subprocess, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models/depth-anything-base-f32.gguf"
GIANT_DIR = ROOT / "models/DA3-GIANT"
GIANT_GGUF = ROOT / "models/depth-anything-giant-f32.gguf"


def _kvstr(field):
    return bytes(field.parts[field.data[-1]]).decode()


def _kvnum(field):
    return field.parts[field.data[-1]][0]


@pytest.mark.skipif(not (ROOT / "models/DA3-BASE").exists(), reason="weights not downloaded")
def test_convert_and_read_back():
    subprocess.check_call([sys.executable, str(ROOT / "scripts/convert_da3_to_gguf.py")])
    import gguf
    r = gguf.GGUFReader(str(GGUF))
    keys = {f.name for f in r.fields.values()}
    assert "depthanything3.vit.embed_dim" in keys
    assert "depthanything3.vit.depth" in keys
    names = {t.name for t in r.tensors}
    assert "vit.patch_embed.weight" in names
    assert "vit.blk.0.attn_qkv.weight" in names
    assert "vit.blk.11.mlp_fc2.weight" in names
    # DualDPT depth-head main-path tensors must be present.
    for h in (
        "head.norm.weight",
        "head.proj.0.weight",
        "head.proj.3.weight",
        "head.resize.0.weight",
        "head.resize.3.weight",
        "head.scratch.layer1_rn.weight",
        "head.scratch.rn4.rc2.c1.weight",
        "head.scratch.rn1.rc1.c1.weight",
        "head.scratch.rn1.out.weight",
        "head.scratch.out1.weight",
        "head.scratch.out2a.weight",
        "head.scratch.out2b.weight",
    ):
        assert h in names, f"missing head tensor {h}"
    # No aux-head tensors should leak into M2's GGUF.
    assert not any("_aux" in n for n in names), "aux-head tensors must be skipped in M2"
    # M3 camera pose decoder (cam_dec) tensors must be present.
    for c in (
        "cam.bb0.weight",
        "cam.bb2.weight",
        "cam.fc_t.weight",
        "cam.fc_q.weight",
        "cam.fc_fov.weight",
    ):
        assert c in names, f"missing cam tensor {c}"
    # Regression guard: DA3-BASE backbone has exactly 207 tensors; the DualDPT
    # depth head adds 62 main-path tensors (norm 2 + projects 8 + resize 6 +
    # layer*_rn 4 + refinenet1..3 30 + refinenet4 6 + output_conv1/2 6).
    # The cam_dec MLP adds 10 tensors (bb0/bb2/fc_t/fc_q/fc_fov, weight+bias each).
    BACKBONE, HEAD_MAIN, CAM = 207, 62, 10
    assert len(r.tensors) == BACKBONE + HEAD_MAIN + CAM, (
        f"expected {BACKBONE + HEAD_MAIN + CAM} tensors, got {len(r.tensors)}")


@pytest.mark.skipif(not GIANT_GGUF.exists(), reason="giant GGUF not converted")
def test_giant_reads_back():
    """The GIANT GGUF (SwiGLU backbone + DualDPT head + cam_dec + GSDPT gs_head)
    must read back with the SwiGLU and gs tensors and the giant config KV.
    Converting the giant is slow (5.4GB), so this asserts against an already
    converted GGUF rather than running the converter."""
    import gguf
    r = gguf.GGUFReader(str(GIANT_GGUF))
    kv = {f.name: f for f in r.fields.values()}
    names = {t.name for t in r.tensors}
    for t in (
        "vit.blk.0.mlp_w12.weight",
        "vit.blk.39.mlp_w3.weight",
        "gs.merger.0.weight",
        "gs.scratch.out2b.weight",
    ):
        assert t in names, f"missing giant tensor {t}"
    # no SwiGLU checkpoint should carry the classic fc1/fc2 MLP tensors.
    assert not any(".mlp_fc" in n for n in names), "SwiGLU giant must not have mlp_fc tensors"
    assert _kvstr(kv["depthanything3.vit.ffn_type"]) == "swiglu"
    assert _kvnum(kv["depthanything3.vit.depth"]) == 40
    assert _kvnum(kv["depthanything3.vit.embed_dim"]) == 1536
    assert _kvnum(kv["depthanything3.gs.output_dim"]) == 38
