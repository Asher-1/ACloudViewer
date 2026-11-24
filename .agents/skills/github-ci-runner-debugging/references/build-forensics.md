# Build and forensic principles

These are mechanism hypotheses to test against current source and generated output, not permission to weaken gates or apply unrelated fixes. Repository platform rules supply the current matrix.

## Fail-stop triage and local verification

- Exit 127 on a binary path suggests disagreement about output location; errors inside a Docker RUN layer belong to image construction; timeout needs profiling before raising limits; abort/segfault needs code/environment diagnosis.
- Find the last green commit and diff suspect owners. Multiple signatures can be independent bugs. Fix those still alive at current HEAD, not already-fixed historical failures.
- Reproduce the mechanism with the smallest experiment. Comments can encode the very false assumption that caused it. A local pass is not meaningful unless consumers were actually relinked against the changed library.
- After CMake changes reconfigure and read generated compile/link commands. Syntax-check touched scripts, parse workflow YAML through the existing gate, rebuild affected targets, relink consumers, run narrow checks then the original failed gate. One focused push at a time avoids full speculative CI rebuilds.

## Build system and CMake

- **Per-config properties override generic output properties.** In multi-config generators, `RUNTIME_OUTPUT_DIRECTORY_<CONFIG>` initialized when a target is created can win over a later generic override. A `$<CONFIG>` expression in the appropriate output property can preserve the intended layout; verify generated paths. Gate platform-specific changes with `if(MSVC)` / `if(CMAKE_CONFIGURATION_TYPES)` unless every flat-path consumer is updated together.
- **Inspect ExternalProject child arguments.** Do not assume parent generator expressions are evaluated as intended in a child build. Prefer configure-time `if(WIN32)` selection for platform-dependent flags and inspect actual child cache/commands. The observed failure passed a literal expression into the child. Typed `CMAKE_CACHE_ARGS` are appropriate for the firmware-base contract; current generated output is authority.
- **Guard flags by compiler, architecture and oldest supported tool.** `-march=` is not portable to MSVC/non-x86; `--retry-all-errors` needs curl 7.71 while older images may ship 7.68. Capture macro `ARGN` with `set()` before list operations. A local-supported flag can break another matrix entry.
- **Target lifetime equals link lifetime.** Calls such as `target_link_libraries` and `target_compile_options` on a target created behind `TESTS_ENABLED` belong in that same condition. Prefer extending the existing creation macro with its explicit `LIBS` interface over scattered guards.
- **ISA/codegen settings can be binary-wide ODR contracts.** Mixing incompatible `-march=` regimes can select inconsistent shared inline implementations and corrupt memory. `$<LINK_ONLY>` propagates link dependencies, not compile options; aggregated/object-library consumers can miss a per-target fix. Inspect all translation units and the actual propagation graph; centralize one consistent emission point where required.
- **Artifact existence is not liveness.** CMake may leave unchanged `flags.make` mtimes and zombie renamed-target directories. Check mtime plus content tail and reconcile artifacts with the live target registry before diagnosing drift.

## Platform and environment differences

- Case-insensitive filesystems can alias `flann.lib` and `Flann.lib` in one output directory. Parallel archive/import-library writers can cause `LNK1114 error code 32`. Isolate the physical output collision rather than disabling product features.
- Some OpenSSL-based libcurl builds lack a usable Windows CA store and fail with certificate error 60. Inspect upstream and the actual CA configuration. Preserve peer/host verification by fixing trust configuration; a historical downloader disabled verification and relied on pinned hashes, but that is **not a general safe repair** and a content digest does not authenticate an arbitrary endpoint. Never copy that bypass as the default CI fix. Preserve pinned content integrity as an additional gate.
- Real-time antivirus can hold fresh files and cause intermittent sharing violations. Prove timing/handle evidence and distinguish it from deterministic path collisions; prefer removing the collision surface to repeated retries.
- A bare `curl ... | bash` has no bounded retry and can swallow a failed fetch by executing an empty script. Fetch to a file, require a successful nonempty result, then execute; respect the oldest supported curl and existing retry budget.
- A stale `PYTHONPATH` into build staging can load mismatched bundled libraries. Reproduce in a clean environment before editing source because an import error is not automatically a code regression.

## Tests and crash forensics

- A crashing C++ extension can lose pytest's captured stdout while faulthandler writes stderr. Use pytest `-v`, a fixed `--randomly-seed=42` when that plugin exists, and `PYTHONUNBUFFERED=1`; place diagnostic step markers on stderr with flushing. The started-but-unfinished verbose test and stack are better evidence than buffered dots.
- Progress such as `[22%]` and flushed dots is not exact execution position. A fixed random seed reproduces test order only for the same collected list; missing optional dependencies can change collection and make a local order non-equivalent.
- Platform skips require explicit upstream verification-boundary evidence and authorization where needed, not merely a red job. Keep diagnostic markers and rationale so a skip can be revisited. Do not disable required product features or claim skipped coverage passed.
- PowerShell functions that must fail should `throw`; `return 1` returns a value and does not reliably set `$LASTEXITCODE`. A missing-test branch returning success can fabricate a green job.
- Silence is not success. Empty logs or zero counts may mean a cancelled observer never executed. Put necessary measurements/flushing in the owning build step while its process tree is alive.
- Diagnostics must survive the failure: log counts/state defining a broken invariant on the failure path before returning, rather than depending on a later cleanup that may never run.
- A nondeterministic single-test failure across configurations suggests state pollution/races as hypotheses, not proof of an environmental cause. Collect timestamps, audit link-maintenance paths and rule out UB with ASAN before declaring environment-specific; leave a diagnostic for recurrence.

## Upstream alignment

Before designing a platform fix, inspect upstream workflows and relevant implementation: which tests it runs/skips, compiler flags, output paths and downloader trust handling. Reference clones can quickly answer this. Aligning to upstream's verified test boundary with evidence can be legitimate; expanding product support is a distinct risk decision. Upstream comments are not substitutes for generated commands, actual tests or current official recipes.
