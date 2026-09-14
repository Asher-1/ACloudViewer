# qLingbotMap — LingBot-Map Streaming 3D Reconstruction

Streaming RGB-D 3D reconstruction from an ordered image sequence, powered by
the [LingBot-Map](https://technology.robbyant.com/lingbot-map) Geometric
Context Transformer (GCT) running natively on ggml (CPU / CUDA / Vulkan /
Metal — device auto-pick follows the AICore runtime order).

The plugin feeds every frame of a folder through the official crop
preprocessing (aspect-preserving bicubic resize to a patch-grid-snapped
width, default 518), streams them through the persistent-KV-cache GCT graph,
and writes the reconstruction into the DB tree:

- `LingbotMap_<model>_<device>` group with one colored point cloud per frame
  (depth back-projected with the frame intrinsics, camera-to-world pose
  applied, visibility-confidence filtered, optionally sky-filtered),
- `LingbotMap_trajectory` camera-center polyline points.

## Requirements

- `AICore_ENABLED=ON` (the engine lives in `core/AICore/src/tasks/lingbot/`
  and is exposed through `aicore/lingbot_capi.h`).
- A LingBot-Map GGUF from
  [Asher-1/lingbot-map-gguf](https://huggingface.co/Asher-1/lingbot-map-gguf).
  The dialog lists the published catalog (q8 recommended; f16/f32 higher
  fidelity) and downloads missing files automatically with pinned SHA-256
  ingestion (`lingbot_models/` under the shared data root). The optional
  native sky-segmentation GGUF (`lingbot-map-skyseg-*`) is downloaded the
  same way.

## Usage

1. Plugin → `LingBot-Map Reconstruction`.
2. Pick the model (or point to a custom GGUF) and the ordered image folder
   (`--max frames` limits the stream length).
3. Optional: enable native sky segmentation (outdoor scenes).
4. Run. Progress, the resolved backend, and per-frame streaming are logged;
   cancellation stops after the current frame.

## Memory notes

The release KV-cache profile (scale=8, window=64) needs ~21 GiB of GPU
memory at 518×294 with the q8 model. Keep long sequences bounded with
`Max frames` and prefer the q8 deployment format on 12–24 GiB GPUs.

## License

Apache-2.0 (LingBot-Map upstream; GGUF deployment). See
`models/MODEL_CARD.md` for the full asset table.
