---
name: acloudviewer-aicore-plugin
description: ACloudViewer AICore architecture and plugin integration contract. Use for public C ABI, new inference tasks, model catalogs, plugin workers, batch CLI/JSON-RPC/MCP automation, image/result/timing paths, ggml patches, lifecycle fixes, parity tests, and performance work.
---

# ACloudViewer AICore Plugin Contract

Use this skill whenever a change crosses `core/AICore/` and an AI plugin. The
current public headers, CMake, task catalogs, ggml patch manifest, validation
reports, and tests are the executable source of truth. This skill is the
integration gate, not a replacement for those owners.

## Engineering Text Language

- Use English for all new or modified engineering text in repository files:
  source comments, documentation, diagnostic messages, CMake descriptions,
  tests, and temporary validation scripts. Conversation language does not
  change this file policy; temporary probes have no language exemption.
- Preserve genuine Unicode test/input data and existing localized resources.
  Prefer language-appropriate Unicode escapes for new test literals without
  changing their runtime content. Source escapes must decode to real Unicode
  and NUL characters, not literal backslash sequences. Do not translate payloads
  to pass a text scan.
- Wording cleanup must not rewrite historical reports, frozen models or
  baselines, or numeric/performance acceptance gates.

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
- ggml changes are ordered files in `3rdparty/llama/llama-patches/manifest.yaml`, not
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

Plugin inference and download messages go to the application's existing
console through `CVLog` / `aicore_inference_log::log`. No plugin-local log
widget, nested log scroll area, or duplicate history buffer is permitted;
absence of `std::cout` does not prove compliance. Keep input/sample actions and
run/cancel/stage/progress visible at the supported minimum window size; scroll
configuration only. A recoverable sample-cache miss starts shared preparation,
not Error logging (GUI-thread `CVLog::Error` can open a modal dialog). Report
real download/integrity/extraction failures, and discard results after closure.

| Tempting shortcut | Required action |
|---|---|
| Keep or share a private log widget | Delete it; route messages to the existing application console |
| Silence modal errors globally or change verbosity | Reject. Start shared preparation on a recoverable cache miss without Error logging; preserve Error for real failures |
| Shrink a fixed-height whole-page scroll | Reject. Fix input/actions/progress outside the configuration scroll; verify their visible geometry |
| Certify a first download using a warm cache or local demo | Reject. Cold evidence requires a previously absent root, one real shared ZIP download and verified extracted-member autofill; warm reuse is a separate test |
| Hide the horizontal scrollbar to disguise clipped fields | Reject. Measure minimum widths, make fields shrink or reflow, and verify zero horizontal overflow with every field accessible |
| Add plugin-private archive download or extraction | Reject. Call the existing repository's ensureDatasetReady and consume its datasetReady signal |

If the sample cache is absent, call shared preparation: absence is not a failed
operation and must not open an Error dialog. If transport, digest verification
or extraction fails, keep the application's genuine Error path. An absent cache
is not a reason to ask users to dismiss an Error dialog before downloading.

For a narrow form, catalog text belongs in the combo popup; its longest label
must not force the form's minimum width. Keep full options and their identities,
shorten redundant labels or reflow rows, and test actual viewport/field geometry.
Changing scrollbar visibility alone does not fix clipping.

Closing or rejecting detaches the sample consumer. Reopening restores enabled
actions from current busy/running state; it must not retain a disabled button
from a completed preparation or accept results into a still-closed window.

Deadline pressure does not waive any of these constraints.

Contexts are created and destroyed on the worker thread. Results carry source
identity and generation; stale results are discarded before DB/UI mutation.
Live input is consumer-driven with at most one running frame and one latest
pending frame. Every success, cancel, error, stale, and teardown path calls the
shared completion hook exactly once. Heavy inference never runs on the Qt GUI
thread.

Qt5/Qt6 dual compatibility is mandatory in plugin UI code: every call to a
version-divergent Qt API goes through `core/include/QtCompat.h` instead of
`QT_VERSION` branches, deprecated Qt5-only methods, or plugin-local shims.
This covers (catalog in the header comment) QRegExp/QRegularExpression,
QStringRef/QStringView, split behavior, font metrics, text codec, endl,
mouse/wheel/drop event positions, QMap insertMulti/unite, QVariant type
queries, tab stops, container range constructors, atomic relaxed access,
BGR888, font family fallback, and QDataStream versions. QtCompat.h is the
single owner of those facts: missing wrappers are added there, never per
plugin. This lives entirely in the Qt layer and does not relax the ABI rule
above - nothing Qt crosses the public C ABI.

Reuse `ecvAICoreRuntimeHelpers.h`, `ecvAICoreUiHelper.h`, shared video
components, `ecvTestDataRepository`, and the existing plugin interfaces. Add
the normal `AddPlugin` target, `info.json`, resources, README, and build/index
rows. Every AICore plugin also exposes a headless batch command through
`ecvAICoreBatchCommand.h`; established commands may use its schema-only
compatibility bridge while retaining their proven runtime parser.

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
| Automation | Native batch command + `HELP RESULT_JSON` schema, `aicore.batch` whitelist, Python AI registry/CLI, `AICORE_BATCH_COMMANDS` test row, four generic MCP tools, L1–L5 tests, docs; CLI-Anything lands first ([plugin-workflow.md](references/plugin-workflow.md#automation-delivery)) |

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
python3 -m pytest agent-integration/tests/test_integration.py -v \
  -k "level1 or level2 or level5"
# With the packaged binary and CLI harness installed:
python3 -m pytest agent-integration/tests/test_integration.py -v \
  -k "level3_aicore_schema"
# Against a running GUI (ACV_RPC_PORT on the GUI, ACV_RPC_URL on the client):
ACV_RPC_URL=ws://127.0.0.1:6001 python3 -m pytest \
  agent-integration/tests/test_integration.py -v -k "level4"
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

- [ ] New or modified engineering text is English, including comments and
      temporary scripts; genuine Unicode payloads and historical evidence remain intact.
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
- [ ] `ai list` exactly matches every registered AICore batch command.
- [ ] Every registry task returns a non-empty `acloudviewer.aicore.batch/1`
      schema from the installed plugin on Linux, macOS, and Windows.
- [ ] JSON-RPC `aicore.batch` and MCP list/schema/run surfaces are covered;
      ordinary plugin passthrough remains isolated in a headless child.
- [ ] RPC handlers that spawn a child use `runChildProcess()`; none calls
      `QProcess::waitForFinished()` on the GUI thread.
- [ ] Qt5/Qt6-divergent UI APIs go through `core/include/QtCompat.h` — no
      `QT_VERSION` branches, Qt5-only calls, or plugin-local shims; both the
      Qt5 (Ubuntu 22.04 CI) and Qt6 (noble) paths compile.
- [ ] Documentation updates the owning source and links here rather than
      copying another long checklist.
