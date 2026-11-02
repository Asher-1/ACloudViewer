# qQuerySplat — Multi-View Image-to-3DGS Prediction

Predicts a full 3D Gaussian Splatting scene from a small set of unposed
images: the in-tree VGGT-Omega encoder (shared with the MapAnything
plugin — same published GGUFs, one implementation) exports cached-layer
tokens, the QuerySplat decoder graph turns them into 524288 gaussians, and
the built-in gsplat-semantics CPU rasterizer renders the predicted input
views. Optional test-time optimization (TTO) fine-tunes the decoder KV
memory against the input pixels (L1 + SSIM + LPIPS, AdamW).

Everything runs locally through `libAICore` — no Python at runtime.

## Usage

1. Open the plugin from the AI menu: **QuerySplat Gaussian Prediction**.
2. Add 2-16 images of one scene (use **Use Test Data** to auto-fill the
   first frames of any shared multi-view scene bundle — the same 13 official
   scene fixtures the MapAnything plugin consumes; courtyard's first four
   frames are byte-identical to the upstream QuerySplat eval inputs).
3. Pick a decoder quantization tier (q8_0 is the recommended default; the
   VGGT-Omega encoder resolves from the shared MapAnything model cache).
4. Optionally enable TTO (20 steps is the official default).
5. **Run**: the plugin imports the gaussian centers + colors as a point
   cloud, the predicted cameras as camera sensors, the depth rasters and
   the gsplat renders as images, exports the standard 3DGS PLY, and writes
   a JSON run summary.

## Models

| File | Role | Size |
|---|---|---|
| `querysplat-paper-q8_0.gguf` | decoder, recommended | 1.31 GB |
| `querysplat-paper-f16.gguf` | decoder, staged as f32 | 2.30 GB |
| `querysplat-paper-f32.gguf` | decoder, parity reference | 4.47 GB |
| `lpips-vgg-f32.gguf` | LPIPS metric (TTO only) | 59 MB |
| `vggt-omega-1b-512-{q8_0,f16,f32}.gguf` | encoder (shared with qMapAnything) | 1.3-4.6 GB |

Published at <https://huggingface.co/Asher-1/querysplat_vggto> (decoder +
LPIPS) and <https://huggingface.co/Asher-1/map_anything_vggt> (encoder);
downloads go through the shared model downloader with pinned SHA-256
verification. Test images are shared with qMapAnything: the
`map_anything_vggt_test_data` scene bundles
(<https://github.com/Asher-1/cloudViewer_downloads/releases/tag/map_anything_vggt_test_data>)
are fetched once through `ecvTestDataRepository` and consumed by both
plugins.

## Headless batch

```bash
ACloudViewer -SILENT -QUERY_SPLAT \
    IMAGE img1.png IMAGE img2.png IMAGE img3.png IMAGE img4.png \
    OUT_DIR ./out QUERY_SPLAT_HELP
```

`HELP` prints the full option schema
(`acloudviewer.aicore.batch/1`); the same schema is exposed through the
JSON-RPC `aicore.batch` surface and the MCP tools.

## Accuracy

The default f32-accuracy profile reproduces the upstream torch chain:
cpp-vs-GT render PSNR 26.85 dB (f32) / 26.84 (f16) / 26.89 (q8_0) on the
official reference scenes — see `core/AICore/scripts/validation_manifest.json`
for the regression matrix and `core/AICore/README.md` for the task
contract.
