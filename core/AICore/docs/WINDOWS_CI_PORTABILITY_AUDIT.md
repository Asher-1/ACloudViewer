# Windows CI portability audit

This audit separates reproduced portability defects from preventive changes.
The failed app build in run [42493688868](https://github.com/Asher-1/ACloudViewer/actions/runs/42493688868)
does not establish a failure in the later test-launch helpers. Its compiler
diagnostic has not been retrieved, so these fixes are not evidence of that
run's exact root cause. During the original investigation the API refresh was rate limited; the
log request also lacked the required access.

## Review of the questioned changes

| File | Decision | Evidence and scope |
| --- | --- | --- |
| `util/ci_utils.ps1` | Withdraw `Invoke-PythonChecked` and the associated edits. | Explicit native status checks can improve failure propagation, but no retrieved diagnostic ties this wrapper to the failed build. Existing call sites need their own demonstrated failure before changing this contract. |
| `scripts/build_win_app.ps1`, `scripts/build_win_wheel.ps1` | Withdraw this review's edits. | No reproduced defect justifies changing their script completion behavior as part of this fix. |
| `libs/Reconstruction/cmake/CMakeHelper.cmake` | Keep environment delegation; withdraw forced per-config executable relocation. | Its original raw inherited Windows `PATH` is split as a CMake list. A standalone CTest reproduction with `alpha;beta;gamma` retained only `bin;alpha`. The shared helper preserves inherited values and the caller's existing COLMAP test directory. This affects test launch, not compilation. |
| `cmake/CloudViewerOutputDirectories.cmake` | Keep the shared environment helper and the examples' explicit config properties. Withdraw unrelated global layout/MinSizeRel expansion and AICore test relocation. | The original plugin helper already escaped Windows `PATH` correctly. Sharing its implementation avoids divergent fixes; it does not establish an existing plugin failure. AICore's `ENVIRONMENT_DEBUG`/`ENVIRONMENT_RELEASE` properties do not configure CTest environments. A Ninja Multi-Config reproduction also shows that initialized config-specific output properties override a generic target output property; explicit examples properties restore the directory already promised by that helper. |
| `.github/workflows/windows.yml` | Keep the separately added CuMesh smoke work under review; add only a focused foundation CTest step in this follow-up. | Installing CUDA Torch and compiling/loading a CuMesh probe adds coverage of an optional branch that was disabled when prerequisites were absent. The normal app CI explicitly disables AICore CUDA, so the standalone smoke exercises an additional configuration. It adds download/build cost and modifies the reused Conda environment. The existing setup-miniconda activation supplies Python directly; redundant `conda run` wrappers were removed. The new three-test selection covers Unicode file/directory I/O and host loop fallbacks that neither `tests.exe` nor the C ABI selection runs. The whole path still needs a hosted Windows run. These changes are not fixes proven necessary by the original app-build diagnostic. |

The standalone CMake reproduction also checked that repeated environment-helper
calls do not duplicate modifications and that pre-existing test modifications
are preserved. Its generated Release example location was
`bin/examples/Release/example`.

CMake documents the runtime environment operation in
[`ENVIRONMENT_MODIFICATION`](https://cmake.org/cmake/help/latest/prop_test/ENVIRONMENT_MODIFICATION.html)
and the config-specific target property in
[`RUNTIME_OUTPUT_DIRECTORY_<CONFIG>`](https://cmake.org/cmake/help/latest/prop_tgt/RUNTIME_OUTPUT_DIRECTORY_CONFIG.html).

## Runtime contracts

Every executable launched by Windows CI must satisfy all three contracts:

1. **Location**: multi-config builds may place an executable under
   `bin/<config>/`, `bin/<test-group>/<config>/`, or a legacy flat test
   directory. Discovery must cover the configured configuration and use the
   `.exe` suffix.
2. **Loader search**: Windows has no ELF-style RPATH. The process `PATH` must
   preserve Conda/system DLL directories and include the build configuration's
   DLL directory before the executable starts. Python ctypes additionally uses
   `os.add_dll_directory`; those registrations must live with the loaded library.
3. **Failure propagation**: native tools, CTest, and Python subprocesses must
   return their exit code to the CI step before a shell helper changes
   `$LASTEXITCODE` or deactivates an environment.

CTest uses `ENVIRONMENT_MODIFICATION` with `path_list_prepend` for the shared
runtime paths. This preserves the runner's existing `PATH` and lets CTest use
the host-native path-list separator; the project minimum CMake version supports
this property.

## Findings and fixes

- AICore CTest rows were configured before registration and used
  `ENVIRONMENT_DEBUG`/`ENVIRONMENT_RELEASE`, which are not configuration-scoped
  CTest properties. Runtime setup now runs after `add_test()` and uses the
  shared helper in `cmake/CloudViewerOutputDirectories.cmake`.
- The old COLMAP MSVC test setup embedded the runner's semicolon-separated `PATH` in a
  CMake list. CMake split it at configure time, dropping most DLL locations.
  The helper now prepends paths at test execution time.
- `validate_all.py` only searched flat Linux-style directories. It now resolves
  CMake configurations and searches Windows multi-config directories, including
  `.exe` binaries, while constructing the platform loader variable (`PATH`,
  `DYLD_LIBRARY_PATH`, or `LD_LIBRARY_PATH`).
- AICore OCR/table/fixture probes unconditionally set `LD_LIBRARY_PATH`. They
  now use a shared platform loader helper, so Windows probes update `PATH`.
- The Paddle reference process retains Windows system and framework `PATH`
  entries while excluding directories containing the tested AICore/ggml DLLs.
- Several standalone parity/diagnostic tools assumed POSIX paths or symlinks.
  They now use the platform loader variable, copy staged fixtures on Windows,
  and default diagnostic output to the system temporary directory.
- Trellis diagnostic tools now accept an explicit runtime DLL or a build root,
  `bin/`, or configuration directory and resolve the AICore runtime from
  multi-config output layouts. Windows loads the DLL through the native loader;
  Unix retains its global Torch preload for CuMesh.
- The separately changed Trellis CuMesh has a native Windows CMake path using Torch's package
  metadata, Python's imported target, Torch's CUDA libraries, and staged
  runtime DLLs. Windows CUDA CI builds and runs a dedicated
  smoke target intended to verify this link/runtime dependency path. It has not
  yet provided native Windows acceptance evidence in this review.
- Plugin tests that load Qt/AICore/plugin DLLs now receive the matching
  configuration and Conda directories on `PATH`; archive/catalog filename
  handling normalizes Windows separators before extracting basenames.
- The ggml upgrade and MapAnything parity runners now discover `.exe` files in
  configuration-specific test directories and prepend `PATH` on Windows.
- Reference fixtures that only need byte identity now use regular copies; tests
  that specifically validate symlink rejection retain an explicit skip when the
  host cannot create symlinks.

## Verification performed

- The three questioned PowerShell files have no diff from HEAD.
- The Windows workflow parses as YAML.
- The current shared-runtime, validation-runner, OCR, relation and portable
  YOLO-launcher selections run 252 Python tests (two expected skips).
- CVCoreLib, AICore, the SAM3 diagnostics and the affected contract/catalog
  executables build and link. Twenty-three focused CTest rows pass, including the
  foundation file/mapping, backend, GKD, SAM3D, ALIKED, catalog and Python
  orchestration contracts. The foundation executable links CVCoreLib without
  AICore. An earlier disposable Linux sparse-file probe also read a byte at
  2 GiB + 4097 bytes.
- The affected CMake targets `test_querysplat_npz` and
  `test_cvpluginapi_ui_helpers` build and pass their Release CTest rows.
- No native Windows runner is available in this checkout. A GitHub Actions
  rerun is still required to validate compiler, SDK, Vulkan, and DLL behavior
  on the hosted Windows image.

The full local CTest selection also contains unrelated worktree failures in
the model-downloader and qInstantMesh tests; those are reported separately and
are not used as evidence against the runtime-path changes above.

The final local evidence is retained under
`build_app/Testing/windows_portability_review/`: `final-build.log`,
`filesystem-build.log`, `parallel-build.log`, `ctest.log`, `python-tests.log`,
and `plugin-boundaries.log`. The final scan finds no host OS adapter branches
in owned task C++ sources. The explicit compiler guard for OCR AVX2 dispatch
is an intentional ISA boundary, described below.

## Shared platform ownership

- `scripts/runtime_environment.py` owns executable/library naming, CMake
  configuration selection, build-layout discovery, Qt runtime discovery,
  loader environment changes, and ctypes DLL-search handle lifetime. OCR,
  validation, ReID, Lingbot, tagging, InstantMesh and Trellis callers reuse it.
  Library lookup/load preserves symlink spelling because ggml discovers
  backends beside the loaded AICore path.
- `core/include/FileSystem.h` / `core/src/FileSystem.cpp` own UTF-8 file
  opening, 64-bit stream positioning (`FSeek`/`FTell`), checked size discovery
  (`FSize`), and `ReadOnlyMappedFile` cleanup. AICore already links CVCoreLib;
  consolidation adds no library dependency. The temporary AICore-private
  `file_io.*` and `mapped_file.*` implementations have been removed.
  Eight task source files previously used `fseek`/`ftell`, including casts of
  GGUF offsets to `long`. On MSVC this truncates positions above 2 GiB even in
  x64 builds. The same fix now covers `CFile::CurPos`, `CFile::GetFileSize`, and
  `FReadToBuffer`. The regression lives in `core/tests/test_filesystem.cpp`
  and does not need the AI runtime to compile. It covers a UTF-8 filename,
  positions beyond 2 GiB without allocating gigabytes, position restoration,
  file sizes, reading, mapping reopen/failure/reset, and repeated close.
  GKD and SAM3D retain their original mapping/fallback policies; Windows
  continues using buffered reads. No new Windows mapping path is enabled.
- The 42 direct narrow-character `fopen` calls in 19 owned task files now use
  `FileSystem::FOpen`, so model/export/debug paths use the existing Windows
  UTF-8 opening boundary. The upstream `ggml_fopen` calls remain because that
  API already provides its own UTF-8 platform contract.
- `src/common/ggml_backend_utils.hpp` owns native backend discovery and the
  legacy Vulkan extension-symbol fallback. ALIKED no longer implements a
  second POSIX loader. SAM3D uses the existing portable string utility instead
  of a task-local `_strnicmp` macro; SAM3's unused macro is removed.
- `cmake/CloudViewerOutputDirectories.cmake` owns test loader environment
  construction; task/test declarations supply only their target or extra path.

- `src/common/capi_utils.hpp` / `.cpp` already provide `dup_cstr`; six
  catalogs now reuse it instead of maintaining `_strdup`/`strdup` branches.
  Allocation and the existing process-lifetime catalog storage remain the same.
- CVCoreLib's existing `core/include/Parallel.h` owns pragma syntax, OpenMP
  collapse/SIMD capability and capped runtime thread queries. The temporary
  AICore `common/parallel.hpp` has been removed. ALIKED, OpenPCDet, QuerySplat
  and the other owned host kernels consume the common directives. OpenMP 2.0
  retains outer-loop parallelism, newer runtimes collapse both axes, and
  serial builds emit no directive. SIMD capability is independent of collapse.
- `include/aicore/export.h` owns import/export, deprecation and printf-format
  attributes. GKD, RFDetr and common logging share the format attribute.
  SAM3/TRELLIS retain their task build/static switches but no OS/compiler
  detection. Supported shared-library declaration attributes are unchanged.
- The Gaussian attention safety limit now queries the actual backend through
  the common backend module: CPU and patched Metal are unbounded; other GPU
  backends retain the existing 65535-token cap. No task tests `__APPLE__`.
- Python reference-loader isolation, ELF dependency inspection and the native
  preload mechanism now live in `scripts/runtime_environment.py`. Callers
  still choose which reference to isolate, which CUDA toolkit to match, and
  which Torch dependency chain to preload. The SAM3 acceptance probe uses Qt's
  portable string and process APIs instead of `_strnicmp` and Linux `popen`.

## Follow-up validation and remaining boundaries

The follow-up Python selections ran 252 tests (two expected skips), including
reference isolation, Windows DLL-directory lifetime/filtering, build layouts,
OCR, relation, and validation-runner behavior. Public export declarations match
pre-change output in seven supported build/consumer modes. A C11 translation
unit accepts the runtime header. Standalone directive probes compile and run in
serial, OpenMP 2.0, and modern OpenMP modes. These are compatibility checks,
not model accuracy or performance evidence.

Owned task platform adapters and Python loader services use the shared owners
below. Task headers no longer repeat compiler-specific logging attributes;
the explicit OCR compiler/architecture guard selects an ISA implementation.

| Owner | Native responsibility |
| --- | --- |
| Core `FileSystem` | UTF-8 paths, environment/home paths, stream offsets, POSIX mapping |
| AICore `common/ggml_backend_utils.hpp` | Module discovery, optional symbols, CPU thread discovery, backend preference and scoped Metal setup |
| Core `Parallel.h` | Compiler directive syntax, OpenMP collapse/SIMD capability and capped runtime thread queries |
| AICore `include/aicore/export.h` | Symbol, deprecation and printf-format attributes |
| AICore `scripts/runtime_environment.py` | Artifact names/layout, loader environment, ctypes handles, reference isolation, ELF/Mach-O inspection |
| CMake/backend dependency modules | Toolchain flags, SDK/imported targets, platform-specific backend selection |

Third-party source trees retain their upstream platform branches. ISA-specific
SIMD kernels retain their instruction-set dispatch. The manual Bash scripts
for ELF-symbol inspection, CUDA artifact comparison and upstream golden
regeneration are Unix tools, are excluded from native Windows CTest where
necessary, and are not Windows acceptance entrypoints. The YOLO matrix launcher now lives in Python and uses shared artifact/loader
services; its Bash entrypoint only forwards arguments. It retains one process
per model and the existing thread and comparison policy. Native Windows/macOS builds, Windows CuMesh
smoke and the full real-model/backend matrix remain unverified.

## Second foundation review

The AICore-private parallel header duplicated an existing foundation owner and
has been removed. `core/include/Parallel.h` now owns the directives and caller
runtime queries used by all owned task loops. Its new `GetMaxThreads(cap)`
preserves each call site's thread ceiling and observes the calling OpenMP
runtime's limit. It does not replace `EstimateMaxThreads()`'s physical-core
policy. `GetOpenMPThreadNum()` is local to the calling translation unit: using
the exported core function for per-thread scratch indexing would be incorrect
if core were built serially and the consumer with OpenMP. Existing exported
functions remain available.

The review also found unconditional SIMD/collapse directives that had not been
covered by the earlier helper. OpenMP 2.0 now retains outer-loop parallelism;
SIMD is enabled separately by the advertised OpenMP version or the explicit
MSVC experimental flag. This follows the distinct capabilities documented by
[MSVC's OpenMP compiler options](https://learn.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170).
Per-output reductions, schedules and task feature switches are preserved.
GKD/RFDetr's two compiler guards for printf checking now share
`AICORE_PRINTF_FORMAT` with common runtime logging. This preserves compiler
diagnostics and has no runtime or C ABI layout effect.
Remaining `_OPENMP` branches in the sparse kernels choose parallel algorithms;
tagging retains its explicit option-controlled runtime setup. These are task
policies, rather than duplicate OS/compiler adapters.

Two additional portability defects were corrected:

- Forty-one C++ string-path stream constructions still bypassed UTF-8 file
  opening. They now use `FileSystem::OpenInputStream`/`OpenOutputStream`, as do
  the two streams in foundation `CopyFile`. Native `std::filesystem::path`
  streams retain their native path overload. String/native conversions use
  `u8path`/`u8string`, including diagnostic and export directories.
  Foundation joining, existence checks, copying, listing, working-directory
  and deletion functions now preserve the same UTF-8 contract. Directory
  iteration uses the filesystem implementation instead of narrow `dirent`
  calls; Windows attributes and CRT operations use wide APIs. The Win32
  `CopyFile` macro no longer rewrites the foundation declaration or definition.
  The regression constructs a real native Unicode path before converting it
  to the string boundary, then checks file content and directory operations.
- Mach-O loader discovery expected `path` immediately after `cmd LC_RPATH`.
  Actual `otool -l` output inserts `cmdsize`. The parser now tracks the load
  command until its path arrives; the regression includes `cmdsize`, spaces,
  `@loader_path`, and a non-RPATH command that must be ignored.

Verification for this review is retained under
`build_app/Testing/windows_portability_review/`:

- `parallel-mode-checks.log`: the foundation probe and production ALIKED and
  deterministic sparse-convolution kernels pass in three modes, six executions
  in total. Serial uses no OpenMP flags; the OpenMP 2.0 branch is selected by
  redefining `_OPENMP` while using GCC/libgomp; modern mode uses `-fopenmp`.
  This exercises fallback syntax and arithmetic, not native MSVC acceptance.
- `final-host-tests-build.log`, `native-path-build.log` and
  `unicode-directory-build.log`: foundation, production AICore, directly
  compiled InstantMesh, SAM3 diagnostics and affected contract targets build.
  `final-production-build.log` records the final production/foundation rebuild;
  `format-attributes-build.log` records the subsequent attribute consolidation.
- `parallel-foundation-ctest.log` and `unicode-directory-ctest.log`: 23 focused
  tests pass; the latter repeats the three foundation/host-kernel rows after
  expanding the Unicode directory regression.
- `python-foundation-review.log`: 252 Python tests run, two expected skips.
- `final-plugin-boundaries.log`: the strict public plugin/AICore boundary scan
  passes. Preprocessor nesting passes for 615 owned/foundation files. The sole
  direct compiler/host guard in owned task C++ sources/headers is the deliberate
  GCC/Clang x86 AVX2 dispatch guard in `ocr_runtime.cpp`; no host OS adapter or
  logging-attribute guard remains in task code.
- `format-attribute-probe.log`: C11 and C++17 compilation accepts valid calls,
  rejects a deliberate printf type mismatch, and accepts the attribute-free
  fallback. This confirms the shared declaration remains C-compatible and
  retains compiler checking.
- The file-service regression defines a Windows-style object-like `CopyFile`
  macro before including the public foundation header, then links and calls
  the real function. A separate forced-macro executable also passes; its probe
  output is `filesystem-win32-macro-probe.log`.

Windows Release CPU CI now selects exactly the foundation file, foundation
parallel and production-host-kernel tests through CTest. This is additional
native regression coverage; it does not demonstrate a successful hosted run.
No PowerShell build-script wrapper or redundant `conda run` was reintroduced.

The workflow uses `setup-miniconda@v3` with
`activate-environment: cloudViewer`. Its official Windows examples use both
`powershell` and `pwsh` directly after activation, without `conda run`:
[setup-miniconda v3](https://github.com/conda-incubator/setup-miniconda/blob/v3/README.md#example-2-other-shells).
An explicitly profile-disabled or otherwise non-initialized shell would have
a different activation contract; this workflow does not select one.

Pending native acceptance:

- [ ] Hosted Windows application/contract tests and CuMesh build/load smoke.
- [ ] Native macOS build and Metal validation.
- [ ] Complete real-model/backend matrix; local contract passes do not replace it.
