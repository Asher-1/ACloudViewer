# Plugin Workflow Checklist

## Classification

Classify before editing files:

- Existing task consumer: plugin/UI/worker only.
- Model variant: task catalog, digest, probe and manifest; no new ABI.
- New task: genuinely new stable input/output/lifecycle contract and one
  public C ABI under the monolithic `libAICore`.
- UI composition: orchestrate public task APIs; do not couple private tasks.
- Reusable headless composition: define and validate a first-class pipeline.

Write down input shape and ownership, typed output and release path, options,
errors, timing fields, backend matrix, model assets, and numeric gate before
calling a task ready.

## Worker and Lifetime

Create AICore contexts on the worker thread and destroy them on that same
thread. Use the shared runtime shutdown helper for process cleanup; a plugin
must not tear down live contexts or the process backend registry. Keep source
identity and generation with each request/result. Drop stale results before
mutating DB/UI state.

A `QThread` worker's completion signal is emitted from inside `run()`, so a
queued `deleteLater()` can destroy the worker before `run()` returns, and
`~QThread` aborts on a running thread. Every worker destructor that may run in
that window starts with `if (isRunning()) { requestTaskCancel(); wait(); }`
before releasing its cancel token or context, and scoped guards inside `run()`
end before the completion signal. `check_plugin_aicore_boundaries.py --strict`
reports QThread workers whose destructor does not wait.

For live input, use a consumer-driven queue: at most one running frame and one
latest pending frame. The producer never fetches a new frame before inference
and rendering complete. Every completion path, including cancel, error, stale,
and teardown, calls the shared completion hook exactly once.

## Image and Result Ownership

Pass a borrowed `aicore_image_view` with actual dimensions, row stride, and
RGB/RGBA/GRAY/BGR/BGRA format. Do not force a tight RGB copy unless the task
contract truly requires it. Typed results stay owned by their context until the
documented release function; copy only the data needed by the UI.

File paths, JSON blobs, encoded images, and temporary exports are not frame
inference interfaces. Compatibility wrappers must call the typed implementation
and never become the core path.

## UI and Automation

Keep heavy inference off the Qt GUI thread. Reuse `ecvAICoreRuntimeHelpers.h`,
`ecvAICoreUiHelper.h`, shared video components, `ecvTestDataRepository`, and
the existing plugin interface. Add `info.json`, resources, CMake registration,
README, `BUILD.md` and plugin index rows incrementally.

Agent access is mandatory for every AICore plugin, not a later UI follow-up:

1. Register one native headless command and reuse the GUI's inference core.
2. Use `ecvAICoreBatchCommand.h` for argument grammar, inputs, model/catalog
   resolution, manifest, and `HELP RESULT_JSON` schema. An established command
   may use `handleSchemaHelp()` while its runtime parser remains unchanged.
3. Add the command to JSON-RPC `aicore.batch`'s whitelist (`kCommands` in
   `rpcAICoreBatch`, `qJSonRPCPlugin/src/JsonRPCMethodsAI.cpp`) and the Python
   `cli_anything/acloudviewer/core/ai_registry.py`; do not copy option
   defaults into Python.
4. Add one `(task, COMMAND, header)` row to `AICORE_BATCH_COMMANDS` in
   `agent-integration/tests/test_integration.py`. That table is the reference
   the gates compare against: L1 requires the header define, a registered
   command class and set-equality with `kCommands`; L2/L5 require
   set-equality with `ai_registry.py`; L3/L4 query each row's HELP schema.
   A row that is missing anywhere fails; nothing is generated.
5. Verify Click `ai list/schema/run`, MCP `list_ai_tasks`, `get_ai_schema`,
   `run_ai_batch`, and ordinary `run_plugin_command` passthrough.
6. Update `agent-integration` command mapping, quick reference, and tests.
   Adding an MCP tool or RPC method changes counts that L1/L5 compare with
   every README, OpenClaw manifest, and harness doc; update them together.

The binary HELP schema is the option source of truth. The Python registry owns
only stable task name, native command, summary, and input shape. Read
`agent-integration/README.md` and the CLI quick reference; do not invent
undocumented binary arguments.

An RPC handler that runs a child process (CLI, batch, Colmap) returns
`jsonrpc::runChildProcess()`'s deferred result
(`qJSonRPCPlugin/src/JsonRPCChildProcess.h`); its `finish` callback builds the
answer when the child ends. The handler never waits on the GUI thread:
`QProcess::waitForFinished()` freezes the GUI, and a nested `QEventLoop`
re-enters the dispatcher so the outer reply and deadline wait for any inner
child. ACloudViewer children take `headlessChildEnvironment()`
(platform-correct `QT_QPA_PLATFORM`); `timeout_ms` is bounded by
`childTimeout()`.

### Automation delivery

The harness lives in a separate repository, and every ACloudViewer workflow
checks out `Asher-1/CLI-Anything` at its default branch. Order matters:

1. In CLI-Anything, edit `acloudviewer/agent-harness/.../skills/SKILL.md`
   only, then run `python3 .github/scripts/sync_root_skills.py`; the root
   `skills/cli-anything-acloudviewer/SKILL.md` is a generated mirror checked
   by `validate_root_skills.py`.
2. Run the harness suite (`pytest cli_anything/acloudviewer/tests`), then
   merge/push CLI-Anything.
3. Only then push the ACloudViewer change whose L1/L2/L5 rows expect the new
   registry entry; pushing it first turns every agent-integration job red.

## Workflow Acceptance

Verify the real headless probe first, then the plugin workflow for still,
live, cancellation, stale generation, errors, and teardown. A contract test or
synthetic output is not end-to-end evidence. Include the exact model digest,
input fixture, backend/device, build revision, and report path in the result.

Automation acceptance is fail-closed at four boundaries:

- L1: command header, shared schema contract, RPC whitelist, and ownership check.
- L2/L5: exact Python registry plus Click and MCP surfaces without a binary,
  on Linux, macOS, and Windows.
- L3: every registry task resolves from each packaged Linux/macOS/Windows
  binary and returns a non-empty `acloudviewer.aicore.batch/1` HELP schema.
- L4: the packaged GUI runs under the offscreen QPA (Linux CI, Xvfb + Mesa)
  and serves `aicore.batch` HELP for every row while staying responsive.

A missing packaged plugin, stale registry row, absent `RESULT_JSON`, or schema
mismatch is a failure, not a skip.
