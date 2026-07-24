# LightGlue / ALIKED GGUF Models

GGUF weights for **LightGlue sparse matching** and, upstream in [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML), a native **ALIKED feature extractor**. Inference runs through AICore (ggml): CPU / CUDA / Vulkan.

## Model families

| Family | GGUF prefix | Role | In qLightGlue GUI |
|--------|-------------|------|-------------------|
| **Matcher** | `aliked-lightglue-*`, `sift-lightglue-*` | LightGlue transformer only | Matcher stage (when wired) |
| **Extractor** | `aliked-n16rot-*` | ALIKED CNN (keypoints + 128-D desc) | Planned via AICore C API |

The matcher files do **not** contain extractor weights. End-to-end ALIKED matching needs **both** an extractor GGUF and a matcher GGUF (COLMAP splits ONNX extractor + ONNX matcher the same way).

## End-to-end matching (qLightGlue plugin)

| Matcher GGUF | Feature extractor | Status |
|--------------|-------------------|--------|
| `sift-lightglue-*.gguf` | OpenCV RootSIFT (C++) | **Supported** — no Python/ONNX |
| `aliked-lightglue-*.gguf` | ALIKED CNN | **Matcher in plugin**; extractor via LightGlue-GGML (below) |

COLMAP reference (runtime, no Python):

- SIFT features: classical C++ (VLFeat / OpenCV)
- ALIKED features: ONNX Runtime (`aliked-n16rot.onnx`)
- LightGlue matcher: ONNX → **GGML matcher** in ACloudViewer

LightGlue-GGML (2026-07) adds a **native ALIKED extractor** with PyTorch parity and a CUDA VRAM pipeline. Porting into `core/AICore` + qLightGlue is tracked in LightGlue-GGML `cpp/PHASE3.md`.

### ALIKED CUDA pipeline (LightGlue-GGML)

Recent optimizations on `--device cuda --ggml-cnn`:

| Optimization | Effect |
|--------------|--------|
| DCN fused weights + workspace cache (`GpuPipelineCache`) | No per-frame `cudaMalloc` / H→D for deform conv |
| Score head single fused GGML graph | 4-layer conv+SELU chain → one `ggml_backend_graph_compute` |
| GPU DKD + SDDH | NMS / block Top-K (256×32) / descriptor head stay on device |
| SDDH per-keypoint VRAM workspace (`AlikedSddhScratch`) | Avoids thread-local stack OOB; NaN-safe bilinear sampling |
| CUDA pool / upsample / crop | Removes GGML ping-pong on small spatial ops |
| SDDH weights resident in VRAM | Warmup uploads descriptor-head weights once |
| GGML + custom-kernel stream sync | `ggml_backend_synchronize` + `cudaDeviceSynchronize` at boundaries |

**Parity (1024 px long edge, 1024 keypoints, RTX 3060):** kpt median ≈ **0.003 px**, descriptor cosine median ≈ **0.9996** vs PyTorch ALIKED.

**Latency (median, same setup):** CUDA GGML extract ≈ **517–531 ms** vs PyTorch ≈ **905–1006 ms** (~**1.7–2.0×** faster; best runs exceed 2×). CPU reference extract remains slower; use CUDA for interactive paths.

**Robustness:** pure-noise inputs (e.g. random JPEG) no longer trigger SDDH OOB at high keypoint counts; invalid/NaN keypoints and offsets are clamped or skipped.

Build / verify (upstream repo):

```bash
cmake -S . -B build_cuda -DLIGHTGLUE_GGML_CUDA=ON
cmake --build build_cuda --target lightglue-ggml -j4
./build_cuda/lightglue-ggml extract models/aliked-n16rot-f32.gguf IMAGE.jpg out.akout \
  --ggml-cnn --device cuda --max-keypoints 1024 --resize 1024
python scripts/verify_aliked_ggml.py --image IMAGE.jpg --model models/aliked-n16rot-f32.gguf \
  --binary build_cuda/lightglue-ggml --ggml-cnn
```

Convert extractor GGUF locally: `python scripts/convert_aliked_to_gguf.py models/aliked-n16rot-f32.gguf`

## Download — LightGlue matcher

Release: [cloudViewer_downloads / LightGlue](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/LightGlue)

Cache: `~/cloudViewer_data/extract/lightglue_models/` (or `aicore_lightglue_model_cache_dir()`)

| Download | Matcher for | Quant | Notes |
|----------|-------------|-------|-------|
| [`sift-lightglue-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f16.gguf) | SIFT | F16 | **default for plugin matching** |
| [`sift-lightglue-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-q8_0.gguf) | SIFT | Q8_0 | smaller; ~93% recall vs F32 |
| [`sift-lightglue-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f32.gguf) | SIFT | F32 | reference |
| [`aliked-lightglue-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-f16.gguf) | ALIKED | F16 | matcher only |
| [`aliked-lightglue-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-q8_0.gguf) | ALIKED | Q8_0 | matcher only |
| [`aliked-lightglue-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-f32.gguf) | ALIKED | F32 | reference |

## Download — ALIKED extractor

Not yet on the LightGlue release tag; build from [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML) or copy prebuilt artifacts into the same cache directory once published.

| File | Quant | Size (approx.) | Notes |
|------|-------|----------------|-------|
| `aliked-n16rot-f32.gguf` | F32 | ~2.7 MB | parity reference |
| `aliked-n16rot-f16.gguf` | F16 | ~1.4 MB | recommended for CUDA |

GGUF key: `aliked` · 128-D descriptors · default resize long edge 1024 · DKD + SDDH postprocess metadata embedded.

## Matcher architecture

- GGUF key: `lightglue`
- Inputs: keypoints, row-major descriptors, image sizes; SIFT also uses scale + orientation (radians)
- Outputs: mutual match index pairs + scores
- Backends: CPU / CUDA / Vulkan via AICore ggml

## Dev-only fixtures

`scripts/extract_aliked_features.py` generates LGINP01 test fixtures for AICore contract tests — **not used by the GUI plugin**.
