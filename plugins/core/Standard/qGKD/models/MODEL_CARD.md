# Model Card — GKDT-L (General Keypoint Detection Transformer)

| | |
|---|---|
| Task | General (open-world) keypoint detection: text prompts, 1-shot visual prompts, or both |
| Model | GKDT-L (DINOv3-L vision tower + dinotxt text tower + adaptation net + KG transformer + detection head) |
| Paper | GKDT: General Keypoint Detection Transformer (ECCV 2026), [arXiv:2607.00752](https://arxiv.org/pdf/2607.00752) |
| Source runtime | [General-Keypoint-Detection-GGML](https://github.com/Asher-1/General-Keypoint-Detection-GGML) `cpp_ggml` (in-tree port under `core/AICore/src/tasks/gkd/`) |
| Weights | https://huggingface.co/Asher-1/GKD_GGUF (`gkd_fullset-<quant>.gguf`, 4 quantizations; the legacy q4_0 build is deprecated upstream and not cataloged) |
| Digests | Pinned in `core/AICore/include/aicore/asset_digests.h` (HF LFS SHA-256) |
| Input | One query image (any aspect ratio; center-crop bbox optional), prompts; support image + keypoints for the 1-shot visual mode |
| Output | Per prompt: keypoint (source-image pixels + normalized -1..1 ROI coords) and heatmap peak score (0..~1) |
| Inference square | 384 (DINOv3 ViT-L/16, 24×24 patch grid, 96×96 heatmap) |

## Published files

| File | Size | Notes |
|---|---|---|
| `gkd_fullset-f32.gguf` | 3.31 GiB | full precision reference |
| `gkd_fullset-f16.gguf` | 1.66 GiB | near-lossless; best Vulkan latency |
| `gkd_fullset-q8_0.gguf` | 905 MiB | near-lossless (≤ 0.002 score diff) |
| `gkd_fullset-q4_K.gguf` | 483 MiB | **recommended**: exact coordinates, ~0.03 score diff |

Upstream parity statement (RTX 4090, official demos): every backend × quantization
reproduces the stock PyTorch model to ≤ 0.0004 px mean on the multimodal /
visual demos and 0.29 px mean on the bbox-ROI demo; f32/f16 track confident
keypoints at 0.003 px mean with zero argmax flips.

## Multi-object mode

qGKD composes the existing YOLO task (open-vocabulary YOLO-World models from
`aicore_yolo_model_count(AICORE_YOLO_ROLE_WORLD)`) as the box detector, then
runs GKD per box with the user's keypoint texts — the same top-down structure
as the official pipeline. The box IS the GKD coordinate system, so detector
quality propagates 1:1 (measured upstream: matched IoU 0.865 vs the official
GroundingDINO boxes; dense scenes suffer).

## License

GKDT source codes, models, and the MegaKPT dataset are free for **academic
research and educational purposes only; commercial use is prohibited**
(upstream README §8). The GGUF conversion inherits this restriction.

## AICore integration

- C API: `core/AICore/include/aicore/gkd_capi.h` (`aicore_gkd_*`)
- Catalog: `aicore_gkd_model_count/at/default_index/model_by_filename`
- Validation: `gkd` scenario in `core/AICore/scripts/validation_manifest.json`
  (light tier = `gkd_fullset-q4_K.gguf`; `--full` covers all four files)
- No ggml patch was required: every op the three graphs use (flash attention,
  interpolate/BILINEAR, fill, get_rows, concat, …) already exists in the
  patched ggml the other tasks share.
