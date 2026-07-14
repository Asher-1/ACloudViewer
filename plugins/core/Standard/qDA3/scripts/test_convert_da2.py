import os, pytest, numpy as np

# (encoder, checkpoint, output gguf, gguf name, expected embed_dim)
CASES = [
    ("vits", "models/da2/depth_anything_v2_vits.pth", "dumps/da2_vits_test.gguf", "Depth-Anything-V2-Small", 384),
    ("vitb", "models/da2/depth_anything_v2_vitb.pth", "dumps/da2_vitb_test.gguf", "Depth-Anything-V2-Base", 768),
    ("vitl", "models/da2/depth_anything_v2_vitl.pth", "dumps/da2_vitl_test.gguf", "Depth-Anything-V2-Large", 1024),
]


@pytest.mark.parametrize("enc,ckpt,out,name,embed_dim", CASES)
def test_convert_da2_roundtrip(enc, ckpt, out, name, embed_dim):
    if not os.path.exists(ckpt):
        pytest.skip(f"{enc} .pth not downloaded")
    import gguf
    from scripts.da2_reference import load_da2_model
    from scripts.convert_da2_to_gguf import write_da2_gguf
    os.makedirs("dumps", exist_ok=True)
    net = load_da2_model(enc, ckpt)
    stats = write_da2_gguf(net, enc, out, name)
    assert stats["unmapped"] == 0 and stats["backbone"] > 0 and stats["head"] > 0
    r = gguf.GGUFReader(out)
    kv = {f.name: f for f in r.fields.values()}
    def s(n): return bytes(kv[n].parts[kv[n].data[-1]]).decode()
    assert s("depthanything3.arch") == "depthanything2"
    assert s("depthanything3.img.resize_mode") == "lower_bound"
    def u(n): return int(kv[n].parts[kv[n].data[-1]][0])
    assert u("depthanything3.vit.embed_dim") == embed_dim
    assert u("depthanything3.head.output_dim") == 1
    assert u("depthanything3.img.resize_target") == 518
    # every depth_head.* and pretrained.* tensor must be mapped (no silent drop)
    names = {t.name for t in r.tensors}
    assert "vit.patch_embed.weight" in names and "head.scratch.out2b.weight" in names
