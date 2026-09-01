---
name: acloudviewer-aicore-plugin
description: ACloudViewer AICore architecture and plugin integration contract. Use for C ABI changes, new inference tasks, plugin workers, image/result/timing paths, ggml patches, lifecycle fixes, parity tests, and performance work.
---

# ACloudViewer AICore Plugin Contract

Use this skill for every change that crosses `core/AICore/` and an AI plugin.
It records architectural invariants and failure lessons; current code, CMake,
the ggml patch manifest, and test output remain the executable source of truth.

Read these scoped rules as needed:

- `.agents/rules/acloudviewer-ggml-aicore.mdc`: ExternalProject and backend rules.
- `.agents/rules/acloudviewer-plugin-dev.mdc`: Qt plugin structure and UI rules.
- `.agents/skills/ggml-upgrade/SKILL.md`: ggml version upgrade and controlled A/B.

## 1. Evidence Before Conclusions

Never infer the current architecture from an old document or benchmark.

1. Inspect the public header, implementation, consumer, test, and CMake target.
2. Resolve versions and patch counts from the files that own them:

   ```bash
   rg -n 'GGML_VERSION|GGML_SHA256' 3rdparty/ggml/ggml.cmake
   rg -n 'file:' 3rdparty/ggml/patches/manifest.yaml
   ```

3. Treat current `ctest` output and timestamp-aligned `LastTest.log` as evidence;
   `LastTestsFailed.log` may be stale.
4. Record model digest/name, input, backend, quantization, resolved device,
   hardware, driver, warmups, iterations, revision, and dirty state with every
   accuracy or latency result.
5. Mark missing assets/platforms as unverified. Exit 77 is a skip, not a pass.
6. Never publish a speedup percentage without a same-hardware, same-model,
   same-input, same-build-mode controlled A/B.

Do not embed source line numbers, current patch counts, or benchmark snapshots
as normative rules. They drift. Use symbols and reproducible commands.

## 2. Irreducible Architecture

The shortest supported data path is:

```text
QImage/storage owned by caller
  -> borrowed aicore_image_view (pointer + shape + stride + format)
  -> one task-owned preprocess/upload
  -> one model session using a shared BackendLease
  -> typed task result
  -> plugin rendering/export
```

The architecture has five boundaries:

| Boundary | Contract |
|---|---|
| Packaging | One monolithic `libAICore` target; do not split per-task shared libraries |
| Public ABI | C headers under `core/AICore/include/aicore/`; no STL, Qt, OpenCV, exceptions, or ggml types |
| Input | Borrowed, read-only, row-stride-aware `aicore_image_view` for decoded images |
| Output | Typed result structs/handles with explicit ownership; JSON/encoded files are compatibility or export only |
| Runtime | Process services own discovery/queues/cleanup; each context owns model/session/cache state |

`ggml` is a private implementation dependency linked through `3rdparty_ggml`.
Plugins link `AICore` and include only public `aicore/*` headers.

## 3. Public C ABI

### Required task surface

Every task must expose or deliberately document the equivalent of:

```c
int aicore_<task>_abi_version(void);
aicore_<task>_options* aicore_<task>_options_new(void);
void aicore_<task>_options_free(aicore_<task>_options* options);
aicore_<task>_ctx* aicore_<task>_load_opts(...);
int aicore_<task>_is_ready(const aicore_<task>_ctx* ctx);
const char* aicore_<task>_last_error(const aicore_<task>_ctx* ctx);
void aicore_<task>_free(aicore_<task>_ctx* ctx);
void aicore_<task>_free_buffer(void* buffer);
int aicore_<task>_last_pipeline_timings(
        const aicore_<task>_ctx* ctx, aicore_pipeline_timings* timings);
void aicore_<task>_shutdown(void);
```

Rules:

- Opaque contexts only. A context is either ready or carries a queryable error.
- More than six related inputs belong in an options/request struct.
- Output arrays belong to a typed result. Provide one unambiguous release path.
- Setters are NULL-safe no-ops; teardown/free functions are NULL-safe.
- Return `0` for success and `-1` for contract/runtime errors unless an existing
  ABI documents a different compatibility value.
- Breaking signatures, layouts, ownership, or removed symbols require an ABI
  version bump and contract-test update.
- Preserve old path/RGB/JSON APIs only as thin wrappers around the typed path.
  New plugins must not call them in frame inference.

### ABI export boundary

Only public `aicore_*` symbols may leave `libAICore`. Diagnostic and white-box
symbols belong in a test-only library or executable, never the production DSO.

```bash
bash core/AICore/tests/check_no_legacy_symbols.sh \
  build_app/bin/libAICore.so core/AICore/include/aicore
```

Any test that needs private graph stages must link a separate test target. Do
not weaken the export map to make a white-box test convenient.

## 4. Image Input Contract

`aicore/image_view.h` is the common decoded-image contract. It supports RGB8,
RGBA8, GRAY8, BGR8, and BGRA8 with explicit `row_stride_bytes`.

### Caller rules

- The view borrows memory for the synchronous call; ownership never transfers.
- `row_stride_bytes` may exceed `width * bytes_per_pixel`.
- Map native QImage storage when possible. On little-endian systems,
  `Format_ARGB32` memory is BGRA8.
- Convert unsupported or premultiplied formats once, retain that QImage for the
  call, and pass its real `bytesPerLine()`.
- Never assume `QImage::bits()` is tightly packed. Grayscale8 and RGB888 rows
  may be aligned beyond logical width.

### Core rules

- Validate data, dimensions, format, overflow, and minimum stride centrally.
- Read rows/channels directly into persistent preprocess staging.
- Do not first pack RGB and then convert it again to CHW.
- A compatibility wrapper may create a tight RGB view and call the typed API;
  the typed API must remain the implementation owner.

For output pixels, wrap read-only storage with an explicit-stride QImage while
the source lives. If later code mutates the image, deep-copy row by row. A
single contiguous `memcpy(image.bits(), src, width * height)` is invalid when
Qt pads scanlines.

## 5. Typed Results and Hot-Path Purity

The inference hot path is decoded image -> numeric tensors -> typed result.
These operations are forbidden between input submission and result delivery:

- temporary image files or save/reload cycles;
- JSON serialize followed by JSON parse;
- packed RGB `QByteArray` scratch when an image view can express the layout;
- encoded PNG/JPEG as an internal handoff;
- N full-resolution masks before confidence/top-k selection;
- queued-signal copies of large depth/mask buffers without an owner handle.

JSON remains valid for model metadata, persistence, RPC, user-requested export,
and backward-compatible cold APIs. PNG remains valid for DB metadata or file
export after inference. State this boundary explicitly in reviews.

Typed results should expose only data the consumer needs. Read back decoder
tensors, filter detections, then materialize selected masks. Reuse context
scratch capacity across frames; do not `shrink_to_fit` in a hot loop.

## 6. Common Timing Contract

All pipelines report `aicore_pipeline_timings` from
`aicore/pipeline_timing.h`:

- `preprocess_ms`: validation, resize/normalize, and graph-input preparation;
- `inference_ms`: upload, graph execution, synchronization, and required readback;
- `postprocess_ms`: typed native result construction;
- `serialization_ms`: optional compatibility encoding/serialization only;
- `e2e_ms`: API entry until the native result is ready.

Consumers must inspect `valid_fields`; an unmeasurable stage stays zero with
its bit clear. Do not invent stage boundaries or compare graph-only time with
end-to-end time. Plugin queue/display latency is a separate outer measurement.

## 7. Runtime, Lifecycle, and Mutable State

`aicore_runtime_shutdown()` is the common idempotent cleanup entry. Per-task
shutdown functions delegate to it for ABI compatibility. It may purge inactive
leases and registered caches, but must never destroy a live context.

Rules:

- Each public context owns its model, allocator/scheduler, graph cache, scratch,
  last error, and last timing state.
- Physical backend handles may be shared through `BackendLease`; execution is
  serialized by the resolved-device queue when required.
- Cache keys must include every compatibility dimension: device/backend,
  tensor shape/type, model/weight identity, options affecting graph structure,
  and an owner/session generation where allocator reuse is possible.
- Context teardown invalidates only cache entries owned by that context or its
  weights. Never clear another live context's state.
- Process configuration must be explicit options. Task code must not use
  `getenv`/`setenv` for flow control.
- The only sanctioned environment boundary is the shared data-root reader and
  the centralized ggml environment bridge.

GPU cached-graph ordering is an invariant:

```text
build graph -> allocate/bind graph buffers -> upload current input
            -> execute that same bound graph -> read required outputs
```

Allocating or rebinding after input upload can invalidate/clear the input. A
single successful forward does not prove cache safety; run at least two
forwards and exercise context destruction/recreation.

Long-running workers use caller-owned cancel tokens and the resolved-device
queue. Legacy process cancellation/global inference locks are compatibility
only.

## 8. Plugin Worker Contract

- Keep model contexts on the inference worker thread.
- Live inference uses one running job plus at most one latest pending frame.
- Bind every result to source identity and generation; discard stale results
  after seek, source change, stop, or model reload.
- Consumer-driven video calls `completeFrameProcessing()` on success, failure,
  cancellation, stale-result, and invalid-input paths so playback cannot stall.
- Preserve decoded resolution; preprocessing belongs to AICore.
- Convert QImage only when its format cannot be represented by image view.
- Use `ecvTestDataRepository`; do not create plugin-private download/extract
  state machines.
- Use shared plugin UI helpers. UI design details live in
  `.agents/rules/acloudviewer-plugin-dev.mdc`, not in this architecture skill.

## 9. Build and ggml Source of Truth

AICore intentionally remains one shared library. Register a new task in the
existing monolithic `core/AICore/CMakeLists.txt` source list and private include
paths; do not create a per-task shared library.

ggml is an ExternalProject. Durable changes are ordered patch files referenced
by `3rdparty/ggml/patches/manifest.yaml`. Never edit or commit extracted sources
under `build*/ggml/`.

Patch flow:

1. Read the pinned version and replay the complete current manifest.
2. Make an experimental change only in a disposable extracted tree.
3. Generate the minimal semantic patch against the fully patched base.
4. Add it to the manifest in dependency order.
5. Force a clean ExternalProject replay and build.
6. Verify existing pipelines, not only the new operator.

```bash
rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{install,done}
cmake --build build_app --target ext_ggml -j4
```

Do not copy an upstream patch verbatim when earlier ACloudViewer patches touch
the same code. Merge it semantically and prove a clean replay.

## 10. Validation Ladder

Validation scales from contracts to numerical truth:

1. **Static/public boundary**: export whitelist, no ggml in public headers, no
   task-local environment control, no production absolute paths.
2. **Contract**: ABI version, NULL safety, options defaults/setters, lifecycle,
   typed ownership, image formats/stride, timing `valid_fields`, shutdown.
3. **Repeated inference**: at least two forwards on one context and context
   destroy/recreate, especially for cached GPU graphs.
4. **Backend parity**: fixed input, CPU reference versus each GPU backend with
   task-specific metrics and explicit tolerances.
5. **Upstream truth**: compare against the originating framework/model output.
   CPU/GPU agreement alone can preserve a shared bug.
6. **End-to-end performance**: same asset/hardware/protocol, warmups, multiple
   samples, p50/p95, stage timings, deterministic fingerprint where valid.
7. **Plugin workflow**: still image, DB image, video/camera, cancellation,
   stale-frame handling, export, and teardown.

The complete asset-driven runner is:

```bash
# Build every probe and run the default complete gate. Missing/corrupt models
# are SHA-256 checked and downloaded to ~/cloudViewer_data/extract first.
cmake --build build_app --target aicore-validate-all -j1

# Equivalent direct invocation.
python3 core/AICore/scripts/validate_all.py \
  --build build_app --backend cuda \
  --output build_app/Testing/aicore_validation.json
```

The manifest is `core/AICore/scripts/validation_manifest.json`. A complete
claim requires every requested pipeline x model x quantization x backend row.
The default command is the release/regression gate: catalog, download, digest,
missing scenario, exit 77, accuracy, stability, and performance failures all
make it fail. It never silently shrinks the matrix to the local cache.

Use incomplete mode only to continue local diagnosis when a model cannot be
downloaded or a probe returns exit 77:

```bash
python3 core/AICore/scripts/validate_all.py \
  --build build_app --backend cuda --allow-incomplete \
  --output build_app/Testing/aicore_validation-incomplete.json
```

This mode still attempts every selected download and verifies every available
model. It skips scenarios that depend on unavailable models, records the exact
model/scenario in JSON and Markdown, emits verdict `INCOMPLETE`, and returns
success when the remaining rows pass. An `INCOMPLETE` report is never evidence
for a regression-free, release, parity, or speedup claim. `--offline` disables
network access; combine it with `--allow-incomplete` only for cache-local
diagnosis. `--tasks` narrows the requested task set and is not a full gate.

### One-click regression integration for a new task or plugin

Every new AICore task, and every plugin that introduces a new AICore model or
pipeline, must extend the same runner in the owning change:

1. Register every published GGUF filename, stable download URL, cache folder,
   and SHA-256 in the task runtime catalog and `aicore/asset_digests.h`. Make
   `test_catalog_dump_urls --json` expose it; never rely on local file discovery.
2. Add a task probe that owns numeric accuracy, repeated-forward stability,
   and common pipeline timing. Register the probe target in
   `core/AICore/tests/CMakeLists.txt` and make `aicore-validate-all` depend on it.
3. Add every pipeline x model x quantization row to
   `scripts/validation_manifest.json`. Declare exact owned/covered/required
   globs so a new or unconsumed model fails coverage.
4. Add runner tests for catalog parsing, cache destination, failed download,
   scenario expansion, and incomplete filtering. Verify catalog URLs and prove
   that every catalog model is consumed by at least one expanded scenario.
5. Run the default full command on each affected real backend. For an operator
   optimization, retain before/after builds and add `--baseline-build` for an
   interleaved controlled A/B. Archive JSON and Markdown reports with hardware,
   model digests, revision, dirty state, warmups, iterations, p50/p95, accuracy,
   and stability evidence.

For TRELLIS, `src/tasks/trellis/model_catalog.cpp` is the single source of
truth for every published `Asher-1/Trellis2-models` Hugging Face GGUF. It
exports the resolver URL, exact LFS size, and SHA-256 through
`aicore_trellis_model_entry`; qTrellis must adapt that entry and must not carry
a second model table. A new published f16/q8/f32 file requires a catalog row,
an `asset_digests.h` entry, and a mandatory manifest consumer. Published
Trellis pipeline files must never be marked `local_only`: the default one-click
gate downloads them into `~/cloudViewer_data/extract/trellis_models`. RMBG is
a shared task dependency and belongs only in `rmbg_models`; a Trellis consumer
must obtain that directory through the RMBG cache API rather than creating a
second Trellis copy.

Do not merge a new task/model/plugin as covered merely because its standalone
CTest skips cleanly. The default one-click gate must download it and execute it;
only an explicitly requested `--allow-incomplete` run may omit it.

YOLO additionally has a model matrix runner:

```bash
python3 core/AICore/tests/yolo/run_yolo_model_matrix.py --help
```

Separate these claims:

| Evidence | What it proves | What it does not prove |
|---|---|---|
| Contract test | ABI/lifecycle behavior | model accuracy or GPU safety |
| CPU/GPU parity | backend agreement | upstream correctness |
| Stable hash | deterministic output | semantic accuracy |
| Graph timing | backend execution cost | plugin end-to-end latency |
| Controlled A/B | change-specific regression/gain | other hardware/platforms |

## 11. Required Gates

Run the gates relevant to the blast radius:

```bash
cmake --build build_app --target AICore -j4
cmake --build build_app --target aicore-contract-tests -j4
ctest --test-dir build_app -L capi --output-on-failure -j1

bash core/AICore/tests/check_no_legacy_symbols.sh \
  build_app/bin/libAICore.so core/AICore/include/aicore
bash core/AICore/tests/check_no_env_getenv.sh core/AICore/src
python3 core/AICore/tests/check_capi_coverage.py
```

Build and test each touched plugin helper target. For GPU work, run the real
model parity/performance test on every affected backend and quantization.

## 12. Cross-Platform Rules

- No production hard-coded `/home/...`, drive-letter, or developer checkout
  paths. Resolve assets through options, the shared data root, or repository
  test fixtures.
- No arithmetic on `void*`; cast to `uint8_t*` and use `size_t` offsets.
- Guard compiler attributes and platform APIs.
- Route Qt 5/6 divergent APIs through `QtCompat.h` where a shared wrapper exists.
- Do not redefine command-line macros such as `NOMINMAX` or
  `_USE_MATH_DEFINES`.
- Linux success is not Windows/macOS evidence. Run all platform CI before a
  cross-platform completion claim.

## 13. Review Checklist

- [ ] The task remains inside the monolithic `libAICore`.
- [ ] Public ABI uses opaque contexts, explicit options, typed results, one
      ownership rule, `last_error`, readiness, timing, and real shutdown.
- [ ] Decoded image APIs accept `aicore_image_view` with real stride and all
      required channel orders.
- [ ] The plugin hot path contains no file round-trip, JSON round-trip, encoded
      image handoff, or avoidable tight-RGB scratch.
- [ ] Compatibility APIs wrap the typed implementation, not the reverse.
- [ ] Context/cache invalidation is owner-scoped and repeated inference is tested.
- [ ] Timing fields have common semantics and honest `valid_fields`.
- [ ] No environment-controlled task logic or production absolute paths.
- [ ] ggml changes exist only as a cleanly replayable manifest patch.
- [ ] Contract tests cover every new export and the export whitelist passes.
- [ ] Numerical gates cover affected task/backend/quantization rows with fixed
      assets; upstream truth is distinguished from backend parity.
- [ ] Every published model is in the validation catalog with URL, cache path,
      SHA-256, and at least one manifest scenario consumer.
- [ ] The default `aicore-validate-all` gate downloads and executes all requested
      rows; only explicit `--allow-incomplete` reports skipped model scenarios.
- [ ] Performance claims include controlled A/B evidence and p50/p95.
- [ ] Still-image and live plugin paths preserve ownership, generation,
      cancellation, and consumer-driven completion.
- [ ] Documentation updates the owning source only and links to it elsewhere.
