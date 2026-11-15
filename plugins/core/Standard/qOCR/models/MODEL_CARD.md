# qOCR Model Card

## Models

PP-OCR family GGUF conversions of the PaddleOCR / PaddleX pipelines
(X-AnyLabeling-GGML recipes, pinned at `745f6c355f94`): detection (DBNet),
direction classification (cls), recognition (CRNN/SVTR dictionaries),
layout classification (PP-LCNet) and table structure (SLANet_plus).

| Stage | Architecture | Task |
|-------|--------------|------|
| det | DBNet (script-agnostic; one det row serves ch/en/japan) | text-box quadrangles |
| cls | PaddleOCR direction classifier | 0°/180° line orientation |
| rec | CRNN / SVTR recognizer per script dictionary | text lines (scored) |
| layout | PP-LCNet (PaddleX layout) | text/table/figure regions + reading order |
| table | SLANet_plus | HTML table structure + OCR-cell matching |

## Files

Published in the `PP-OCR` release of the ACloudViewer mirror; the sizes
below were measured from the local cache (catalog digests are pinned in
`core/AICore/include/aicore/asset_digests.h`, consumed through the
`aicore_ocr` / `aicore_table` catalogs — no plugin-private URL/digest
table):

| Family | Files (f16 shown; f32/q8_0 variants published alike) | f16 size |
|--------|------------------------------------------------------|----------|
| det (shared) | `ppocr-det-ch.gguf`, `ppocrv4-source-det.gguf`, `ppocrv5-mobile-det.gguf`, `ppocrv6-{tiny,small}-det.gguf` | 2.4–4.8 MB |
| rec zh | `ppocr-rec-ch.gguf`, `ppocrv4-source-rec.gguf` | 7.3 / 8.2 MB |
| rec en | `ppocr-rec-en.gguf` | 4.3 MB |
| rec ja | `ppocrv3-japan-rec.gguf` | 8.0 MB |
| rec zh (v5/v6) | `ppocrv5-mobile-rec.gguf`, `ppocrv6-{tiny,small}-rec.gguf` | 4.0–15.2 MB |
| cls | `ppocr-cls.gguf` | 0.4 MB |
| layout | `pplcnet-layout.gguf` | 3.9 MB |
| table | `slanet-plus-table.gguf` | 3.9 MB |

Download base (owned by the AICore OCR/table catalogs):
`https://github.com/Asher-1/cloudViewer_downloads/releases/download/PP-OCR/<filename>`.
Cache directory: `ocr_models/` under the shared data root (see
`aicore_ocr_model_cache_dir`); every file is SHA-256 digest-verified at
ingestion.

## Recipes (det + rec + cls pairings)

`ppocr-v4-ch` (default, Chinese + English auto), `ppocr-v4-en` (English),
`ppocr-v4-source` (source-precision det), `ppocr-v3-japan` (Japanese),
`ppocr-v5-mobile`, `ppocr-v6-tiny`, `ppocr-v6-small`, `ppocr-v6-medium`.
The quantization tier applies uniformly to every enabled stage (the same
uniform tier the table GT gate measures).

## Backends

- CUDA (primary validated backend)
- Vulkan / CPU / Metal through the shared ggml runtime
  (`aicore_runtime_*`; device auto = Cuda > Vulkan > CPU)

## Accuracy & performance evidence

- Table GT gate: cropped-page TEDS/TEDS-S against the PaddleX reference
  (`core/AICore/scripts/validate_all.py`, table task rows in
  `validation_manifest.json`); the cell-join A/B is documented in
  `core/AICore/src/tasks/table/README.md`.
- Sample-data showcase matrix (every recipe measured with `bench_ocr` /
  `bench_table --layout-regions`, cuda f16, plus f32/q8_0 spot checks):
  see the plugin README "Try sample data" section.
- Batch automation surface: `-OCR_RECOGNIZE` headless command (RESULT_JSON
  schema in `OCRBatchCommand.h`), whitelisted through
  `AICORE_BATCH_COMMANDS` (see `agent-integration/tests/test_integration.py`).
