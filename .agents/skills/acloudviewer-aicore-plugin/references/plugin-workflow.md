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

Agent access extends the existing JSON-RPC and `cli-anything-acloudviewer`
mapping. Read `agent-integration/README.md` and the CLI quick reference; do
not invent undocumented binary arguments.

## Workflow Acceptance

Verify the real headless probe first, then the plugin workflow for still,
live, cancellation, stale generation, errors, and teardown. A contract test or
synthetic output is not end-to-end evidence. Include the exact model digest,
input fixture, backend/device, build revision, and report path in the result.
