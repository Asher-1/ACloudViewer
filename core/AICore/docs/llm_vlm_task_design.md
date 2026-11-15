# Local LLM / VLM task contract (llama-backed)

Status: **implemented** (`aicore/llm_capi.h`, `src/tasks/llm/`). The public
header, the task catalog, `validation_manifest.json` and the probe output are
the executable source of truth; this page records the contract and why it has
this shape.

## 1. Scope

The pinned llama.cpp archive (`707616fe…`, llama 0.7.0-dev, embedded ggml
0.27.0) builds `libllama` and `libmtmd` next to AICore's ggml. The `llm` task
exposes local chat generation over them through the single libAICore, so a
plugin (qAnyLabeling Chat/VQA) can choose a local provider instead of an
OpenAI-compatible HTTP endpoint.

Out of scope: training/LoRA, server mode, tool calling, audio, embeddings,
and any Florence-2 equivalence claim (a chat VLM answering about an image is
not Florence caption/grounding/segmentation).

## 2. One task, one header

A VLM is a chat LLM plus one more owned asset (the mmproj projector). Both
share the tokenizer, chat template, KV cache, sampler, stop/cancel logic and
timing; only prefill differs (image chunks are encoded by mtmd). A second
`vlm_capi.h` would duplicate every one of those facts, so there is exactly one
header, `aicore/llm_capi.h`:

- `aicore_llm_ctx*` owns one text model, an optional projector and the KV
  state. `aicore_llm_supports_vision(ctx)` reports whether images are accepted.
- `aicore_llm_options` (struct_size/abi_version validated): `model_path`,
  `mmproj_path`, `device` (`auto`, `cpu`, `cuda[:N]`, `vulkan[:N]`, `metal`),
  `threads`, `context_tokens` (0 ⇒ min(train ctx, 4096)), `batch_tokens`
  (0 ⇒ 512), `flash_attn` (-1 auto / 0 / 1), `image_max_tokens`.
- `aicore_llm_request` carries everything per call: full message history,
  borrowed `aicore_image_view` images (stride + RGB/RGBA/GRAY/BGR/BGRA
  preserved; packed to RGB inside the task), `max_tokens`, `temperature`
  (0 ⇒ greedy), `top_p`, `seed`, stop sequences and an optional stream
  callback. Sampling is a request property, not a context property, because
  one loaded model serves greedy gates and sampled chat alike.
- Generation is **stateless**: KV memory is cleared before every prefill, so a
  result never depends on a previous request. Calls on one context are
  serialized by the caller; `aicore_llm_cancel` is thread-safe.
- Return codes: 0 success (stop/length), 1 cancelled (partial result still
  returned, `finish_reason = CANCELLED`), -1 error (`last_error`). The stream
  callback returning non-zero is a cancel. Streamed fragments are complete
  UTF-8 code points; text that may still become a stop sequence is held back,
  so the concatenated stream always equals the final text.
- `aicore_llm_result` is independently owned (`text`, `token_ids`,
  `prompt_tokens`, `generated_tokens`, `prefill_ms`, `decode_ms`, common
  `timings`) and released with `aicore_llm_result_free`.
- `aicore_llm_shutdown()` delegates to `aicore_runtime_shutdown()`.

### Device and runtime ownership

llama allocates its own backend buffers; it cannot adopt a `BackendLease`. The
task therefore resolves the device from the ggml registry that AICore already
loaded (`load_backends_once`) and passes it as `llama_model_params.devices`
(`{dev, nullptr}`; CPU is `{nullptr}` with `n_gpu_layers = 0`). `auto`/`gpu`
follow the platform auto order and fall back to CPU; an explicitly requested
device that does not exist is a typed error, never a silent fallback. The
projector uses the same device. `llama_backend_init` runs once per process;
`llama_log_set` is given AICore's ggml log bridge because it re-installs the
ggml log callback.

## 3. Timing contract

`aicore_pipeline_timings` with truthful `valid_fields`:

| Stage | Meaning |
|---|---|
| `preprocess_ms` | chat template + tokenization + image preprocessing |
| `inference_ms` | prefill (text + image encode) + decode loop |
| `postprocess_ms` | stop-sequence trim + result construction |
| `e2e_ms` | API entry until the result is ready |

`prefill_ms` / `decode_ms` on the result split `inference_ms`.

## 4. Models

Catalog (`src/tasks/llm/model_catalog.cpp`, cache folder `llm_models`), every
row revision-pinned on Hugging Face with its LFS SHA-256 in
`asset_digests.h` and exported by `test_catalog_dump_urls --json`:

| Family | Files | Role |
|---|---|---|
| SmolVLM-256M-Instruct | Q8_0, f16 model + Q8_0, f16 mmproj | vision |
| SmolVLM-500M-Instruct | Q8_0 (**default**), f16 model + Q8_0, f16 mmproj | vision |
| Qwen2.5-0.5B-Instruct | q8_0, fp16 | text |

SmolVLM-500M Q8_0 is the default because it is the model upstream
`tools/mtmd/tests.sh` gates on. A vision model's entry names its recommended
projector (`mmproj_filename`); a mismatched projector fails the load.

## 5. Gates

Truth is llama.cpp itself: `tests/llm/prepare_upstream_reference.py` builds
`llama-mtmd-cli` and `llama-completion` **pristine** (no AICore patches) from
the same pinned archive with the production backend set, and the runner
passes it as `--reference llama_upstream=<bin dir>`. Without it the probe exits
77 and the runner counts the row as missing.

`test_llm_probe` (manifest rows `llm-vqa-newspaper`, `llm-vqa-describe`,
`llm-chat-capital`, `llm-chat-poem`; every catalog model is consumed) requires:

1. `upstream_text_match == 1`: greedy text equals the upstream tool on the same
   model, projector, image, prompt and device (CPU/CUDA/Vulkan).
2. `answer_contains_hit == 1` where declared — `new york` is the upstream
   `tests.sh` acceptance for the newspaper VQA, `paris` for the capital chat.
3. Probe-owned invariants (exit code): identical text and token ids across
   warmups + repeats and after context recreation, streamed == final text,
   stream-callback cancel returns a strict prefix with `CANCELLED`, and a stop
   sequence trims the output.

Quantization: Q8_0 and f16 rows are gated independently against upstream on
the same weights; no cross-quantization equality is claimed.

Performance: the probe reports `e2e_ms_p50/p95`, `prefill_ms_p50`,
`decode_ms_p50` and `decode_tokens_per_s_p50`; speed claims follow the
controlled A/B rule (same host/model/input/backend, `--baseline-build`).

```bash
python3 core/AICore/tests/llm/prepare_upstream_reference.py --build build_app
python3 core/AICore/scripts/validate_all.py --build build_app --backend cuda \
  --tasks llm --reference llama_upstream=build_app/Testing/framework-references/llama-upstream/build/bin
```

## 6. Plugin integration (qAnyLabeling)

Chat/VQA gains a "Local (llama)" provider beside the HTTP providers. The
worker creates and frees the context on its own thread, runs at most one
generation, maps the UI stop button to `aicore_llm_cancel`, and resolves model
files through the catalog + `ecvModelDownloader`; no plugin model table.
