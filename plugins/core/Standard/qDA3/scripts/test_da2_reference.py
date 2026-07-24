import os, numpy as np, pytest

CKPT = "models/da2/depth_anything_v2_vitl.pth"

def test_fixed_input_shape():
    from scripts.da2_reference import fixed_input
    x = fixed_input(16)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32

@pytest.mark.skipif(not os.path.exists(CKPT), reason="vitl .pth not downloaded")
def test_load_and_forward_vitl():
    import torch
    from scripts.da2_reference import load_da2_model, fixed_input
    net = load_da2_model("vitl", CKPT)
    x = torch.from_numpy(fixed_input(16))[None]   # (1,3,224,224)
    with torch.no_grad():
        depth = net(x)
    assert depth.shape == (1, 224, 224)
