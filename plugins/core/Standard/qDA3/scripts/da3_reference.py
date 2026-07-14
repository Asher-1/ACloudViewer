#!/usr/bin/env python3
"""Build the reference DA3-BASE model and a single fixed N=1 input fixture.
Shared by convert_da3_to_gguf.py and dump_reference.py so weights and the test
input are identical across conversion and parity dumps.

ENV CAVEAT (not captured by the pinned scripts/requirements.txt): importing the
DA3 api also requires `opencv-python-headless` (cv2) and `evo` to be installed,
and relies on `torchvision` being un-importable so the stub below takes effect
(the only torchvision wheel compatible with this torch build is ABI-broken). A
fresh venv must additionally run:
    pip install opencv-python-headless evo
    pip uninstall -y torchvision   # if a mismatched torchvision is present
The stub assumes torchvision is never genuinely needed on the backbone path
(true here: fixed_input normalizes in numpy, so T.Normalize is never exercised)."""
import sys, numpy as np, torch
sys.path.insert(0, "/tmp/da3-src/src")

PATCH = 14
# Fixed small resolution: multiple of patch, fast on CPU. k=16 -> 224x224 -> 16x16 patches.
FIX_K = 16
FIX_H = FIX_W = PATCH * FIX_K

# Normalization values are VERIFIED from the DA3 source (not from a config file —
# there is no preprocessor_config.json in the checkpoint).
# Source: depth_anything_3/utils/io/input_processor.py:56
#   NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# The processor does T.ToTensor() (which scales uint8 -> [0,1] and converts HWC->CHW)
# then applies the Normalize above. The resize policy (default process_res=504,
# method="upper_bound_resize": resize longest side to 504 then round each dim to a
# multiple of PATCH=14) is NOT exercised here because the fixed input is already
# 224x224 (a multiple of 14); it is documented for the converter/dump scripts.
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_model(model_dir="models/DA3-BASE"):
    """Return (model, net) on CPU in eval/f32. net is the DepthAnything3Net.

    `model` is the DepthAnything3 API wrapper (nn.Module + PyTorchModelHubMixin);
    from_pretrained loads model.safetensors into the underlying network. The
    DepthAnything3Net is at `model.model`; its DINOv2 backbone is `net.backbone`
    and the ViT is `net.backbone.pretrained`.
    """
    # Importing depth_anything_3.api at module load drags in optional, heavy
    # subsystems we never use for plain model building/parity:
    #   - the GLB/Gaussian-Splat *video export* chain needs moviepy==1.0.3, which
    #     is unavailable here (only moviepy 2.x exists, and it conflicts with
    #     numpy>=2 / the pinned torch build).
    #   - the InputProcessor needs torchvision, whose only installable wheel
    #     (0.27.0) is ABI-incompatible with this torch 2.12.0+cpu build
    #     (`operator torchvision::nms does not exist`).
    # Neither is exercised here: load_model only builds the net + loads weights,
    # and fixed_input() does normalization manually in numpy. So we shadow these
    # broken/unused modules with lightweight stubs before importing the api.
    # (cv2/opencv IS installed and used by the real input processor, so it is not
    # stubbed.)
    import types
    if "depth_anything_3.utils.export" not in sys.modules:
        _exp = types.ModuleType("depth_anything_3.utils.export")
        def _export_unavailable(*a, **k):
            raise RuntimeError("depth_anything_3 export is stubbed in da3_reference")
        _exp.export = _export_unavailable
        sys.modules["depth_anything_3.utils.export"] = _exp
    if "torchvision" not in sys.modules:
        class _AnyTransform:
            def __init__(self, *a, **k): pass
            def __call__(self, x): return x
        class _TransformsStub(types.ModuleType):
            def __getattr__(self, _name): return _AnyTransform
        _tv = types.ModuleType("torchvision")
        _tvt = _TransformsStub("torchvision.transforms")
        _tv.transforms = _tvt
        sys.modules["torchvision"] = _tv
        sys.modules["torchvision.transforms"] = _tvt

    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_dir)
    net = model.model if hasattr(model, "model") else model   # unwrap to DepthAnything3Net
    net = net.eval().float()
    return model, net

def fixed_input(seed=0):
    """Deterministic normalized input tensor (1,1,3,H,W) and the raw uint8 image (H,W,3)."""
    rng = np.random.default_rng(seed)
    raw = (rng.integers(0, 256, size=(FIX_H, FIX_W, 3), dtype=np.uint8))
    img = raw.astype(np.float32) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    t = torch.from_numpy(img).permute(2, 0, 1)[None, None]   # (1,1,3,H,W)
    return t.contiguous(), raw

def fixed_input_multiview(S=2, seed=0):
    """Deterministic S-view normalized input (1,S,3,224,224) + list of raw uint8 (224,224,3).
    Each view is a DIFFERENT structured (non-noise) image so cross-view attention has real signal."""
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    views = []; raws = []
    for v in range(S):
        yy, xx = np.mgrid[0:FIX_H, 0:FIX_W].astype(np.float32)
        # vary the pattern per view deterministically
        r = (np.sin((xx + v * 23) / 30.0) * 0.5 + 0.5)
        g = (np.cos((yy + v * 17) / 40.0) * 0.5 + 0.5)
        b = ((xx + yy + v * 50) / (2 * FIX_W)) % 1.0
        arr = np.stack([r, g, b], -1)
        arr[60:120, 60 + v * 20:120 + v * 20, :] = 0.9
        raw = (arr * 255).astype(np.uint8); raws.append(raw)
        x = (raw.astype(np.float32) / 255.0 - mean) / std
        views.append(torch.from_numpy(x).permute(2, 0, 1))
    t = torch.stack(views, 0)[None].contiguous()   # (1,S,3,224,224)
    return t, raws

def fixed_input_multiview_distinct(S=4, seed=0):
    """Deterministic S-view input (1,S,3,224,224) + raw uint8 list, with STRONGLY
    distinct per-view content so reference-view selection (S>=3, saddle_balanced)
    yields a NON-trivial b_idx (i.e. != 0). This actually exercises the reorder /
    restore permutation, unlike fixed_input_multiview whose near-identical phase-
    shifted views always select view 0 (an identity reorder). seed is accepted for
    API symmetry but the content is fully deterministic from the view index."""
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    yy, xx = np.mgrid[0:FIX_H, 0:FIX_W].astype(np.float32)
    views = []; raws = []
    for v in range(S):
        f = 5.0 + v * 7.0
        r = np.sin(xx / f + v * 1.3) * 0.5 + 0.5
        g = np.cos(yy / (f * 0.6) + v * 0.7) * 0.5 + 0.5
        b = ((xx * np.cos(0.5 * v) + yy * np.sin(0.5 * v)) / FIX_W) % 1.0
        arr = np.stack([r, g, b], -1).astype(np.float32) * (0.4 + 0.18 * v)
        arr[15 + v * 14:75 + v * 14, 25:85 + v * 28, :] = (v % 3) / 2.0
        arr = np.clip(arr, 0, 1)
        raw = (arr * 255).astype(np.uint8); raws.append(raw)
        x = (raw.astype(np.float32) / 255.0 - mean) / std
        views.append(torch.from_numpy(x).permute(2, 0, 1))
    t = torch.stack(views, 0)[None].contiguous()
    return t, raws

if __name__ == "__main__":
    _, net = load_model()
    # Structural assertions: net is the DepthAnything3Net with a 12-block, 768-dim ViT.
    assert isinstance(net, torch.nn.Module)
    assert len(net.backbone.pretrained.blocks) == 12, len(net.backbone.pretrained.blocks)
    assert net.backbone.pretrained.embed_dim == 768, net.backbone.pretrained.embed_dim
    x, raw = fixed_input()
    print("model + fixture ready:", x.shape, "raw", raw.shape)
