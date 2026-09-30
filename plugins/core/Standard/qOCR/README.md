# qOCR — PP-OCR Text Recognition

PP-OCRv4 text detection and recognition over the AICore `ocr` task, built on
the first-party Paddle inference → GGUF conversion
(`core/AICore/tools/convert_paddle_ocr_gguf.py`).

## Workspace

The **OCR Workspace** action opens the plugin dialog (a pure view in the
shared AICore plugin shape, same split as qRMBG):

- **Recognition language / Quantization** — picks the det/rec GGUF pair from
  the published catalog (det is shared and script-agnostic).
- **Device / Threads** — the shared runtime row. Device entries come from
  `aicore_device_count()` ("auto" follows the platform order: CUDA → Vulkan →
  CPU on Linux/Windows, Metal → CPU on macOS); the request is passed through
  `aicore_ocr_options.device` and resolved by the AICore runtime lease
  registry. GPU-unavailable requests fall back to CPU with a console note.
- **Try sample data** — one-click official PaddleOCR demo image through the
  shared `ecvTestDataRepository` (Dataset::Ppocr).
- **Download missing models** — the unified `ecvModelDownloader` pipeline with
  pinned SHA-256 verification and the shared progress bar.
- **Run OCR / Cancel** — the run executes on an `OCRWorker` thread: the
  device task queue is acquired first (serialized against every other AICore
  task), the context is created on the worker thread and released on the main
  thread, and cooperative cancellation is bound for the run. Progress is
  stage-based with the busy (indeterminate) bar — the aligned fallback for
  backends without per-node progress — plus a 10 s heartbeat in the console.
- **Add result to DB** — exports an annotated image (boxes + transcriptions
  drawn over the source) to the DB tree with metadata (transcription list,
  det/rec models, device, stage timings, source path); named through
  `ecvPluginDbNaming` so repeated runs stay separable.

All log output goes through the shared application console
(`aicore_inference_log` → `CVLog`); the plugin embeds no private log widget.

## Models

Published at the [PP-OCR release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/PP-OCR)
and pinned in `core/AICore/include/aicore/asset_digests.h`; every entry is
exposed by `test_catalog_dump_urls --json` for the shared download/cache
pipeline. Recognition dictionaries are embedded in the rec GGUFs
(`ocr.dictionary`), so a model file is self-contained.
