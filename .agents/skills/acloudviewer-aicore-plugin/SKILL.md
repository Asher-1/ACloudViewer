---
name: acloudviewer-aicore-plugin
description: ACloudViewer AICore architecture and plugin integration contract. Use for public C ABI, new inference tasks, model catalogs, plugin workers, image/result/timing paths, ggml patches, lifecycle fixes, parity tests, and performance work.
---

# ACloudViewer AICore Plugin Contract

Use this skill whenever a change crosses `core/AICore/` and an AI plugin. The
current public headers, CMake, task catalogs, ggml patch manifest, validation
reports, and tests are the executable source of truth. This skill is the
integration gate, not a replacement for those owners.

## Read the Right Detail

- [validation-gates.md](references/validation-gates.md): manifest schema,
  accuracy thresholds, developer/release profiles, and evidence claims.
- [reuse-contract.md](references/reuse-contract.md): ownership, C ABI, data
  flow, and plugin boundary check.
- [plugin-workflow.md](references/plugin-workflow.md): classification, worker
  lifetime, live-frame completion, UI and automation workflow.
- `.agents/rules/acloudviewer-ggml-aicore.mdc`: ExternalProject, patch replay,
  backend loading, and platform constraints.
- `.agents/rules/acloudviewer-plugin-dev.mdc`: Qt plugin structure and UI rules.
- `.agents/skills/ggml-upgrade/SKILL.md`: pinned ggml upgrade and controlled A/B.

## First Principles

The shortest supported chain is:

```text
caller-owned decoded image
  -> borrowed aicore_image_view
  -> task-owned preprocess/session
  -> typed result and common timing
  -> plugin rendering/export
```

Every proposed abstraction must answer four questions:

1. What stable input/output/lifecycle contract does a caller need?
2. Which single module owns each model, device, cache, timing, and error fact?
3. What evidence distinguishes correctness, parity, stability, and speed?
4. What is the smallest change that satisfies that contract?

Do not add a layer merely because two files look similar. Promote a helper only
when at least two real consumers share semantics, ownership, error behavior,
and lifecycle, or when it enforces a named cross-cutting invariant.

## Non-Negotiable Architecture

- Keep one monolithic `libAICore`; never create a shared library per task.
- Keep the compiled task contract build-invariant. Do not add public
  `AICore_USE_<TASK>` switches; plugin switches control optional consumers,
  while task-specific backend dependencies stay in private build wiring.
- Plugins link `AICore` and include only public `core/AICore/include/aicore/*`.
- Public C headers do not expose Qt, STL, OpenCV, exceptions, or ggml types.
- Use opaque contexts, explicit options, typed results, release functions,
  `last_error`, readiness, common timing, and real shutdown.
- Extensible structs validate `struct_size` and `abi_version`.
- Decoded images cross the ABI as borrowed, read-only, stride-aware
  `aicore_image_view`; preserve RGB/RGBA/GRAY/BGR/BGRA format.
- Typed results are the inference path. JSON, encoded images, paths, and files
  are compatibility/export boundaries only.
- A context owns model/session/cache state. Backend handles may be shared only
  through the existing `BackendLease`; invalidation is owner-scoped.
- Per-task shutdown delegates to `aicore_runtime_shutdown()` and never destroys
  live contexts.
- Options configure task behavior. Production task logic does not read private
  environment variables or developer-machine absolute paths.
- ggml changes are ordered files in `3rdparty/ggml/patches/manifest.yaml`, not
  edits in extracted `build*/ggml/` trees.

## Classify Before Coding

| Change | Landing | Must not do |
|---|---|---|
| Existing-task UI/workflow | Plugin worker/UI using current ABI | Clone task, graph, catalog, cache, or backend |
| Model variant | Owning task catalog/loader, digest, probe, manifest | Add a product-name task or plugin model table |
| New inference contract | One task under `src/tasks/<task>/` plus public C ABI | Put inference in a Qt worker or task shared library |
| UI composition | Plugin orchestration of public task APIs | Private task-to-task coupling or file/JSON transport |
| Reusable headless composition | First-class AICore pipeline contract | Hide it in one plugin helper |

Before a new task, write its input ownership, typed output/release path,
options, errors, timing boundaries, backend matrix, assets, and numeric gate.
If these cannot be stated, the task contract is not ready.

## Ownership Map

| Fact | Owner | Consumer rule |
|---|---|---|
| Device discovery, queues, leases, cancellation, cleanup | AICore runtime/common | Use the public runtime/backend API |
| Image representation | `aicore/image_view.h` | Borrow native storage and actual stride |
| Graph, preprocess, postprocess, session/cache | Task context | Treat it as an opaque service |
| Published model role/URL/digest/cache folder | Task catalog/cache API + `asset_digests.h` | Read exported catalog/cache entries |
| Download transport, integrity, shared data root | AICore catalog + shared plugin services | Use catalog metadata with `ecvModelDownloader` / `ecvAssetIntegrity` |
| UI/DB/render/export | Plugin | Keep Qt/application types outside ABI |
| Accuracy/stability/performance | Task probe + validation manifest | Extend the common runner |

The plugin must not implement a private downloader, cache root, model URL/SHA
table, backend registry, device lock, scheduler, graph, task-specific
preprocessing, or second validation runner. A plugin may orchestrate the shared
`ecvModelDownloader` with URL/digest metadata read from an AICore catalog.
Model-specific code remains task-private; Qt-free generic runtime helpers belong
in AICore common code; Qt-aware reuse belongs in shared plugin API.

## Worker and UI Contract

Contexts are created and destroyed on the worker thread. Results carry source
identity and generation; stale results are discarded before DB/UI mutation.
Live input is consumer-driven with at most one running frame and one latest
pending frame. Every success, cancel, error, stale, and teardown path calls the
shared completion hook exactly once. Heavy inference never runs on the Qt GUI
thread.

Reuse `ecvAICoreRuntimeHelpers.h`, `ecvAICoreUiHelper.h`, shared video
components, `ecvTestDataRepository`, and the existing plugin interfaces. Add
the normal `AddPlugin` target, `info.json`, resources, README, build/index rows,
and existing JSON-RPC/CLI mappings when automation is in scope.

### AICore plugin CMake closure

Keep each AICore standard plugin structurally consistent with the maintained
plugin targets: `AddPlugin(NAME ...)`, followed by `add_subdirectory(include)`
and `add_subdirectory(src)`, with headers and implementation files registered
through those two target-owned `CMakeLists.txt` files. Add the plugin's own
`include/` directory privately and link every library that its sources consume
directly. `AddPlugin` already closes the common `CVCoreLib`, `CVPluginAPI`, and
`CVPluginStub` ABI dependencies, but a plugin that calls `CVPluginAPI` services
directly should also list `CVPluginAPI` in its own `target_link_libraries` so
the source-level dependency stays explicit and survives future changes to
transitive links. Shared `video_base` remains optional and is linked only when
the target exists; it must not be the hidden owner of the plugin's API or Qt
dependencies.

### Shared plugin sample data

When a plugin has a "Use test data" / demo-fill action, register its dataset in
`ecvTestDataRepository` and use that repository's shared archive download,
SHA-256 verification, cache-under-`cloudViewer_data/download`, and extraction
into `cloudViewer_data/extract` flow. Do not add a plugin downloader, private
cache directory, or plugin-specific ZIP extraction. Prefer the upstream's
recommended demo image as the deterministic first-fill choice, even when the
archive also contains larger evaluation datasets. Pin the archive digest and
the selected extracted member digest/size in the common validation manifest;
have the plugin action select and display that same extracted file. Add focused
coverage for cached/offline reuse, automatic fill, selected path, and missing
or invalid data handling. This requirement applies to newly integrated AICore
plugins as well as existing plugin workflows.

Model and sample caches honor the same global `CLOUDVIEWER_DATA_ROOT` override,
falling back to `~/cloudViewer_data`. Test a first download in an isolated empty
data root; require the initial demo to be absent, then verify the retained ZIP,
extracted member, and automatic UI fill. Do not count a warm default cache as
evidence of a cold download.

Reference inputs and truth data are external test assets, not source files.
Declare a stable remote archive and its digest in the common asset manifest, or
generate a transient reference under the build directory or `TMPDIR` from a
pinned upstream revision. Tests must accept the resolved asset/reference path,
fail clearly when it is unavailable, and record the revision and digest used.
Never check in generated logits, model outputs, or machine-specific truth
fixtures merely to make a local probe pass. A plugin may choose the upstream
recommended demo member for its first-fill action, but that choice belongs in
the plugin's manifest and documentation rather than in this general contract.

## Required Integration Surfaces

| Classification | Required surfaces |
|---|---|
| Existing-task consumer | Plugin target/registration, worker/UI, helper/workflow tests, docs |
| Model variant | Catalog, stable URL/cache/digest, catalog dump, probe, structured manifest row, runner tests |
| New task | Public header, task implementation, monolithic CMake/test closure, catalog/digest, probe, manifest, ABI tests |
| New plugin | CMake option/registration, `info.json`, resources, README, focused worker/UI tests |
| Automation | Existing JSON-RPC and `cli-anything-acloudviewer` command mapping/tests |

Do not claim coverage when any row is absent. A catalog model is incomplete
until it is exposed by `test_catalog_dump_urls --json` and consumed by an
expanded manifest scenario.

## Validation Ladder

Run the narrowest relevant checks first, then the real matrix:

```bash
cmake --build build_app --target AICore -j4
cmake --build build_app --target aicore-contract-tests -j4
ctest --test-dir build_app -L capi --output-on-failure -j1
python3 core/AICore/tests/check_capi_coverage.py
python3 core/AICore/tests/check_plugin_aicore_boundaries.py --root . --strict
```

For release, parity, or speed claims use the release profile with a same-host
baseline; see [validation-gates.md](references/validation-gates.md). Run real
checkpoints on every claimed backend and quantization. `--allow-incomplete`,
exit 77, synthetic outputs, contract tests, parity alone, and stable hashes do
not prove release accuracy or performance.

For every AICore-dependent plugin, also verify real headless inference before
the still/live/cancel/stale/error/teardown UI workflow. Record model and input
digests, backend/device, quantization, build revision, dirty state, warmups,
iterations, p50/p95, accuracy checks, and stability result.

## Accuracy and Performance Gates

Accuracy gates are task-specific numeric or explicitly probe-owned invariants;
the prose `accuracy_gate` label is not a threshold. New manifest rows must use
the structured `accuracy` object, and the release profile rejects every
selected legacy-prose row. Distinguish upstream/golden correctness from
CPU/backend parity. A quantized weight file does not prove lower activation,
attention, or KV-cache allocation and does not imply lower latency.

For quantized models, record both the error against the independent framework
checkpoint and any error added relative to the same weights in upstream.
Declare absolute numeric bounds and quantization allowances explicitly in the
owning manifest. Preserve failing metrics in the report; agreement with an
upstream quantized output alone cannot replace independent correctness checks.
Check added error per output as well as aggregate scores so improvements in
one class cannot hide a regression in another.
When intentionally changing activation arithmetic, expose the choice through
context options, identify both modes in reports, and verify the original mode's
upstream compatibility separately. The new default must still meet its absolute
independent-checkpoint accuracy and original-baseline performance bounds. Keep
its per-output upstream differences visible; improved aggregate accuracy never
proves that every output improved.

Performance claims require same hardware, model, input, backend, build mode,
and sampling protocol. Use p50/p95, repeated processes, warmups, and a
controlled candidate/baseline A/B. Never publish a speedup from a single wall
clock or from different model/input/quantization rows.

## Review Checklist

- [ ] Change classified as existing consumer, model variant, new task, or pipeline.
- [ ] One owner exists for model, cache, device, timing, error, and UI facts.
- [ ] No plugin private ggml/task include, catalog source, downloader
      implementation, backend, env, absolute path, or frame file/JSON round trip.
- [ ] Public ABI has ownership, release, error, readiness, timing, version, and
      shutdown behavior; image stride/format is preserved.
- [ ] Context lifecycle, repeated inference, context recreation, cancellation,
      stale generation, and teardown are covered.
- [ ] Every model has stable URL/cache/digest/catalog-dump/manifest consumer.
- [ ] Structured accuracy checks identify their reference and required metrics.
- [ ] Real backend/model evidence exists; parity is not called truth.
- [ ] Release performance has controlled A/B, p50/p95, and matching protocol.
- [ ] `check_plugin_aicore_boundaries.py --strict` passes repository-wide.
- [ ] Documentation updates the owning source and links here rather than
      copying another long checklist.
