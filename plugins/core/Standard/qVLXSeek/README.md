# qVLXSeek — VLX-Seek fine-grained grounding

Open-vocabulary, region-retrieval grounding on images, powered by
**VLX-Seek 1.5-10B** running through AICore's native GGML task
(`vlx_seek`):

```text
image ──► WeDetect-Uni proposals (f32) ──► 100 candidate boxes
image ──► Qwen3.5-VL tower + C-RADIOv4/SimpleFPN + HFRE ──► region tokens
region tokens + task prompt ──► Qwen3.5-10B (pinned llama.cpp qwen3_5)
        ──► `<ground>label</ground><objects><obj2><obj5></objects>`
        ──► boxes resolved from the proposal set
```

Because the model retrieves candidate regions instead of generating
coordinates, multi-object answers stay short, stable and fast to decode, and
absent targets are rejected with an explicit answer instead of a hallucinated
box.

## Actions

- **VLX-Seek grounding** — dialog with the seven upstream tasks
  (`detection`, `grounding_single`, `counting`, `reasoning_detection`,
  `brief_region_caption`, `region_ocr`, `vqa`), `en`/`zh` prompt templates,
  precision choice from the published catalog, and a Use-test-data action
  that fills the upstream demo image from the shared data repository.
- **-VLXSEEK_RUN** — headless batch command (`-VLXSEEK_RUN
  IMAGE1 IMAGE2 -TASK detection -TEXT "person" -LANG en -PRECISION q8_0
  -MAX_NEW_TOKENS 1024 -DEVICE auto -THREADS 0`). Writes
  `<stem>_vlxseek.json` plus an annotated `<stem>_vlxseek_vis.png` per image
  and the shared `RESULT_JSON` manifest. `HELP:1` prints the option schema.

## Models

All GGUFs come from the pinned revision of
[Asher-1/VLX-Seek-1.5-10B-GGUF](https://huggingface.co/Asher-1/VLX-Seek-1.5-10B-GGUF)
through the AICore task catalog (`aicore_vlx_seek_model_*`): the f32
WeDetect-Uni proposal network (the official `.pth` dtype), the merged vision
stack (Qwen3.5-VL tower + CAbstractor + RADIOv4 + SimpleFPN), and the
Qwen3.5-10B hybrid LLM. One family = one detector + one vision + one LLM row;
`f32`/`f16`/`q8_0` digests live in `core/AICore/include/aicore/asset_digests.h`.
The q8_0 set is the recommended default (~11.5 GiB VRAM; f16 needs ~21 GiB).

First Run downloads the three files of the selected family into
`~/cloudViewer_data/extract/vlx_seek_models` through the shared
`ecvModelDownloader` with pinned SHA-256 verification.

The official test images (two upstream demos + seven benchmark originals)
ship as `vlx_seek_test_images.zip` on the same model repository; the
**Use test data** action fetches and extracts them through
`ecvTestDataRepository::ensureDatasetReady` into
`~/cloudViewer_data/extract/vlx_seek_test_images` and fills the demo image.

## Notes

- Results land in the DB tree as an annotated image
  (`VLXSeek_<model>_<stem>`) carrying the answer, task and timing metadata.
- Inference runs on a worker thread; heavy inference never touches the GUI
  thread. All messages go to the application console.
- Accuracy/performance evidence: run the `test_vlx_probe` probe against the
  upstream `vlx-cli` reference (see
  `core/AICore/tests/vlx/prepare_vlx_upstream_reference.py`) — greedy-token
  identity and box deltas are the release gates.
