# Standard plugin model cache audit

Audit date: 2026-11-09. Base checkout: `890e8fc8c`, with the accompanying
uncommitted fixes. Scope: all 48 `q*` plugin directories in
`plugins/core/Standard`, including their workers, model projections, download
callbacks, and the shared cache/transport implementation. The `base` directory
is shared infrastructure rather than another plugin.

The source review found 22 plugins with catalog model workflows and 26 without
automatic catalog-model downloads. It did not find a normal Run path that
explicitly requires every published quantization/version of a task. It did find
unnecessary pipeline dependencies, mismatched validation, continuation bugs,
and alternate cache-root fallbacks. Those findings are fixed below.

This is a source audit plus focused Linux transport/UI regression evidence.
It is **not** a complete native Linux/macOS/Windows end-to-end inference matrix.

## Required behavior and ownership

For an operation, the required set is the selected catalog entry plus the
dependencies its selected pipeline actually consumes. A missing file queues
only that set. Successful ingestion resumes the initiating operation; failure
or cancellation discards the pending operation and permits an explicit retry.
A valid deployed file does not need transport, including after dialog/process
recreation or an explicit download-button click.

Cache roots and model identities remain owned by the public AICore catalogs and
`aicore_*_model_cache_dir` APIs. Qt projection helpers can adapt those values;
they must not own a second model inventory or choose a private fallback root.
`ecvModelDownloader` owns transport and ingestion, and `ecvAssetIntegrity` owns
the shared disk evidence. Changed files with pinned evidence are rejected.
Ordinary UI access uses cheap checks and ledger size/mtime evidence, without
rehashing multi-GB models on each click. Legacy files without a ledger retain
the documented magic/size fallback; they are not all SHA-256 verified by Run.

Explicit batch requests to download `MODEL all` are distinct user operations.
Catalog enumeration alone and ordinary single-selection Run must not trigger
that operation.

## Findings and fixes

| Finding | Trigger and previous consequence | Resolution / source |
|---|---|---|
| SAM3D dependency and continuation gaps | Single-object downloads included an unused SAM3 model; selected RMBG was omitted; Run/preview lacked a download-and-resume path. | One selected dependency projection drives checks and downloads. Preview needs only selected SAM3 for concept prompts; supplied masks need none. Single-object RMBG uses only its selected precision. Queue completion resumes Generate or Preview, and failure/cancel restores controls. [Sam3dDialog.cpp](../../../plugins/core/Standard/qSAM3D/src/Sam3dDialog.cpp) |
| TRELLIS unnecessary texture downloads | Geometry with PBR disabled still required fields 6–9 from the preset. | Required files exclude the texture chain when disabled; manual download and Run use the same missing-file set. [TrellisDialog.cpp](../../../plugins/core/Standard/qTrellis/src/TrellisDialog.cpp) |
| TRELLIS operation lost after download | Full mode re-entered `onRun()`, which selected GeometryOnly. | Persist the initiating mode before checking files and resume `runWithMode(m_runMode)`. |
| FaceDetect wrong continuation / selection | A Live or test-gallery download resumed batch Run; a landmark filename was used to repopulate the detector combo; a cached landmark could override the selected version. | Capture the original action, preserve the detector combo across landmark completion, and resolve the selected landmark first. Download requests now carry the same pinned SHA-256 anchor as cache checks. [FaceDetectDialog.cpp](../../../plugins/core/Standard/qFaceDetect/src/FaceDetectDialog.cpp) |
| LingbotMap skyseg did not resume | Mapping paused for an enabled sky-segmentation dependency without recording the pending Run. | Record pending Run before downloading skyseg, then recheck the selected pipeline. [LingbotMapDialog.cpp](../../../plugins/core/Standard/qLingbotMap/src/LingbotMapDialog.cpp) |
| AnyLabeling small GGUF rejected | The 1 MiB floor rejected valid small ALIKED GGUFs, including the approximately 714 KiB q8 extractor. | Main model and Loma matcher/extractor requests/checks admit the shared 64 KiB floor, with pinned ingestion validation. [AnnotationWindow.cpp](../../../plugins/core/Standard/qAnyLabeling/src/AnnotationWindow.cpp), [LomaBridgeDialog.cpp](../../../plugins/core/Standard/qAnyLabeling/src/LomaBridgeDialog.cpp) |
| InstantMesh small pipeline stages rejected | All three synthesizers are below the private 1 MiB ingestion floor: q8 264,864 bytes, f16 412,384 bytes, f32 819,424 bytes. A successful transfer was deleted and reported as failure. | Use the shared request default and the existing catalog SHA-256 anchor. Cross-check the catalog against the common floor. [InstantMeshDialog.cpp](../../../plugins/core/Standard/qInstantMesh/src/InstantMeshDialog.cpp) |
| Duplicate cache-root policy | DeepLSD, LightGlue, FaceDetect helpers, FreeSplatter, and its face-capture path constructed alternate model roots if the public API failed. | Remove those fallback roots and use the public owner API. Packaged cascade XML and image/session working directories remain separate non-GGUF assets. |
| Presence-only Run/download gates | GKD, LingbotMap, OpenPCDet, RFDetr, RMBG, YOLO, VLXSeek, and QuerySplat LPIPS could treat an existing changed model as ready. | Reuse shared `isModelVerified`; selected detector/text dependencies are checked individually. No plugin hash ledger is introduced. [ecvAssetIntegrity.cpp](../../../libs/CVPluginAPI/src/ecvAssetIntegrity.cpp) |
| Repeated manual/queued download | Calling the shared downloader directly always began transport, even for an intact destination. | Resolve intact destinations first and emit success with zero HTTP requests. Cache-hit completion is synchronous and documented; caller continuation state is initialized before the call. [ecvModelDownloader.cpp](../../../libs/CVPluginAPI/src/ecvModelDownloader.cpp) |
| Stale continuation after failure | GKD, LingbotMap, RFDetr, RMBG, and YOLO retained an operation after a failed/cancelled fetch. | Clear pending operations on failure. Live continuation re-enters its availability check before starting, covering selection changes during a fetch. |
| Windows path encoding | Narrow environment strings and native `.string()` cache paths could disagree with Qt's UTF-8 decoding for non-ASCII profiles/overrides. | Read environment values through the shared CVCore filesystem boundary, reuse CVTools for UTF-8/wide conversion, and build/return paths with `u8path` / `generic_u8string`. [FileSystem.cpp](../../src/FileSystem.cpp). Native Windows verification remains pending. |

No independent GGUF HTTP implementation was found in plugin production code.
AnyLabeling's `RemoteServerClient`, `PpocrServiceClient`, and
`ChatServiceClient` transport inference/service requests; those are not model
cache implementations. Local chat assets use the shared downloader/catalog.

## Per-plugin dependency review

The entry files below are relative to `plugins/core/Standard`. Each row was
reviewed for selected-entry resolution, dependent file checks, download
completion, and reuse on a subsequent Run. “Selected” includes the catalog's
declared precision pairing; it does not imply every stage has every precision.

| Plugin | Required models for the selected operation | Source entry / result |
|---|---|---|
| qAnyLabeling | Selected task model; only task-specific partners such as OCR detector, chosen text encoder/tagger, selected landmarks, or automatic relation-object proposals. Local chat uses the selected LLM and its declared projector. Loma uses selected matcher/extractor roles. | `src/AnnotationWindow.cpp`, `AnnotationChat.cpp`, `LomaBridgeDialog.cpp`; small-model gates fixed. |
| qDA3 | Selected depth model; metric partner only for a mode that needs it. | `src/DA3Dialog.cpp::ensureAllModelsAvailable`; selected queue and resume retained. |
| qDeepLSD | One selected DeepLSD model. | `src/DeepLSDDialog.cpp`; private root fallback removed. |
| qFaceDetect | Selected detector; selected dense landmark only in DenseLandmarks mode. Live/test-gallery detector checks do not require dense landmarks. | `src/FaceDetectDialog.cpp`; continuation, anchor, and selection fixes. |
| qFreeSplatter | One selected FreeSplatter model; face-capture detector only when that feature is used. | `src/FreeSplatterDialog.cpp`, `FaceCaptureWidget.cpp`; public roots only. |
| qGKD | Selected GKDT; selected YOLO-World detector and text encoder only for object-dependent modes. | `src/GKDDialog.cpp::ensureModelAvailable`, `ensureYoloModelAvailable`; shared presence checks and failure reset. |
| qInstantMesh | Selected preset/quantization chain, with only the selected RMBG entry if enabled. | `src/InstantMeshDialog.cpp`, `InstantMeshModelCatalog.cpp`; selected closure retained, invalid 1 MiB/16 MiB private floors removed. |
| qLightGlue | Selected matcher and its matching ALIKED extractor when required; other matcher variants are unnecessary. | `src/LightGlueDialog.cpp`; public root retained, private fallback removed. |
| qLingbotMap | Selected mapping model; skyseg only if enabled. | `src/LingbotMapDialog.cpp`; skyseg continuation fixed. |
| qMapAnything | One selected model. | `src/MapAnythingDialog.cpp::ensureModelAvailable`; selected resume chain retained. |
| qOCR | Selected recipe/precision detector and recognizer; same-tier layout/table only when enabled. | `src/OCRDialog.cpp::pumpQueue`, `updateRunState`; download completion enables the selected recipe without requiring alternate recipes/tiers. |
| qOpenPCDet | One selected catalog model or explicit custom local model. | `src/OpenPCDetDialog.cpp`; shared presence checks. |
| qQuerySplat | Selected decoder and shared VGGT encoder; LPIPS only when TTO is enabled. | `src/QuerySplatDialog.cpp::ensureModelsAvailable`; LPIPS cache gate fixed. |
| qRAM | One selected RAM/RAM++/Tag2Text entry. | `src/RAMDialog.cpp::ensureModel`; selected ingestion/resume retained. |
| qRFDetr | One selected detector for image/live. | `src/RFDetrDialog.cpp`; shared checks, failure reset, Live recheck. |
| qRMBG | One selected RMBG version for image/live. | `src/RMBGDialog.cpp`; shared checks, failure reset, Live recheck. |
| qRelateAnything | Selected relation model; student/tokenizer only for custom predicates; selected detector/text only when automatic objects are needed. | `src/RelateAnythingDialog.cpp::ensureModel`; selected closure retained. |
| qSAM3 | Selected image/video model. | `src/SAM3Dialog.cpp`, `VideoTab.cpp`; selected pending actions retained. |
| qSAM3D | Four selected-dtype core weights plus fixed f16 MoGe; mesh decoder only for mesh output; selected RMBG only for single-object removal; selected SAM3 only for concept prompts. Preview excludes 3D and RMBG. | `src/Sam3dDialog.cpp::requiredModelFiles`; dependency and continuation regression coverage added. |
| qTrellis | Selected preset and precision chain; texture stages only if enabled; one shared RMBG when enabled. Fixed f16 stages in a mixed q8 pipeline are declared dependencies, not extra versions. | `src/TrellisDialog.cpp::missingPresetFiles`; geometry-only and Full continuation coverage added. |
| qVLXSeek | Selected vision precision, matching LLM, fixed f32 WeDetect; OCR detector/recognizer only for the OCR proposal option. | `src/VLXSeekDialog.cpp::ensureModelsAvailable`; shared presence check. |
| qYOLO | Selected task-head model; selected text encoder only when text prompting requires it. Visual/prompt-free modes omit it. Live catalog excludes text-conditioned families. | `src/YOLODialog.cpp::ensureModelAvailable`, `YOLOLiveWidget.cpp::populateAllModels`; shared checks, failure reset, Live recheck. |

The other 26 plugins have no automatic catalog-model cache workflow:
q3DMASC, qAnimation, qBroom, qCSF, qCanupo, qCloudLayers,
qColorimetricSegmenter, qCompass, qCork, qFacets, qG3Point, qHoughNormals,
qJSonRPCPlugin, qM3C2, qMPlane, qMasonry, qPCL, qPCV, qPoissonRecon,
qPythonRuntime, qRANSAC_SD, qSIBR, qSRA, qSpirula, qTreeIso, qVoxFall.
User-supplied classifier files (for example Canupo/3DMASC), Gaussian scenes,
and arbitrary user Python scripts are not automatic catalog downloads.

## Verification and remaining acceptance

Focused checks use isolated temporary data roots, avoiding changes to the
user's existing model cache. The HTTP test serves generated GGUF-shaped bytes
on a local TCP server and checks their actual SHA-256. It covers first fetch,
repeat fetch without HTTP, persisted evidence with a new downloader instance,
changed-file repair, digest rejection and retry, Range/206 resume, size-only
ingestion, and offline cache hits under a Unicode/space-containing path.

SAM3D/TRELLIS dialog tests use explicitly synthetic UI ledger fixtures. They
prove dependency selection, cached repeated Run, cancellation/retry, and
operation continuation at the download-completion boundary. They do not load
those fixtures as neural networks and do not establish numerical accuracy.

Run the focused checks with:

```bash
ctest --test-dir build_app --output-on-failure \
  -R '^test_(data_root|cvpluginapi_asset_integrity|cvpluginapi_model_downloader|cvpluginapi_ui_helpers|qsam3d_model_selection|qtrellis_model_selection|qtrellis_catalog|qinstantmesh_catalog)$'
python3 core/AICore/tests/check_plugin_aicore_boundaries.py --root . --strict
```

Current-run raw logs are retained in `build_app/Testing/model_cache_audit/`.
The associated source changes do not modify ggml or inference mathematics.
Initial Linux verification used the existing Release build with Qt 5.15
(`/opt/qt515`); CUDA and Vulkan were enabled in the build, but these cache/UI
tests do not execute either inference backend. The 8 focused CTests passed,
the strict boundary check passed, and all 15 modified plugin targets built:
QANYLABELING, QDEEPLSD, QFACEDETECT, QFREESPLATTER, QGKD, QLIGHTGLUE,
QLINGBOTMAP, QOPENPCDET, QQUERYSPLAT, QRFDETR, QRMBG, QSAM3D, QTRELLIS,
QVLXSEEK, and QYOLO (each with the `_PLUGIN` target suffix).

Evidence files: `build-verified.log`, `tests-verified.log`,
`boundaries-final.log`, and `inventory.json` in that artifact directory.

- [x] Review all 48 plugin directories and all 22 model-related workflows.
- [x] Focused Linux cache/path/transport/dialog/catalog regressions (8/8).
- [x] Strict plugin/AICore boundary checker.
- [x] Final integration rebuild of all 15 modified plugins and expanded focused test run.
- [ ] Native macOS and Windows build/test, including Unicode profile/override paths.
- [ ] Native application cold-download-to-inference and repeated cached Run for
  every affected task/mode using published real models; this is needed before
  claiming complete three-platform end-to-end acceptance.


## Follow-up: shared services and platform boundaries

The storage unit is an artifact identified by its published digest, not a
plugin or a complete family of model versions. Model catalogs own identity and
selected pipeline dependencies. Downloading, cached readiness, extraction and
UI continuation are separate responsibilities:

| Owner | Responsibility | Consumers |
|---|---|---|
| CVCore `filesystem::LocateDataRoot` | One UTF-8 root policy; native home-directory resolution via Qt | AICore model caches, CVPluginAPI sample caches, library datasets |
| CVTools `Char2Wchar` / `Wchar2Char` | UTF-8/native-wide conversion, including supplementary Unicode | Common environment and file-open boundaries |
| `ecvAssetIntegrity` | Shared cheap checks, pinned digests and persistent ingestion evidence | Model and ordinary-file readiness checks |
| `ecvModelDownloader` | HTTP, resume, streamed digest validation, atomic promotion and warm-cache bypass | All catalog-model transports and repository archive/file transports |
| `ecvTestDataRepository::ensureDatasetReady` | Resolve cached extraction, cached ZIP, download and final `datasetReady` | Standard-plugin dataset preparation |

Composition through these existing services is preferable to a new plugin
base class. Plugins already participate in QObject/plugin inheritance and
have distinct operation queues, dependencies and UI lifetimes. Inheritance
would couple those lifetimes to storage and would not eliminate duplicate
transport implementations. Queue selection stays local; storage behavior is
shared. The public legacy repository entry points remain compatible, but no
Standard-plugin production consumer now manually chains `startDownload` and
`extractDataset` or their intermediate completion signals.

FaceDetect's background dataset-registration worker continues to compose the
same downloader and the repository's static ZIP extraction function, because
its extraction/registration progress runs off the GUI thread. Its forwarding-
only ZIP wrappers were removed; it owns no alternate ZIP implementation.
DeepLSD/LightGlue keep family/catalog identity filters but delegate cache
validity to `isModelVerified` rather than maintaining their own size/magic
policy. Model catalog adapters remain projections of public task catalogs.
Those adapters also preserve task identity: InstantMesh and TRELLIS publish
different `dino_*` bytes under matching basenames, so the InstantMesh adapter
uses its owning catalog's digest rather than the other task's basename pin.

Additional lifecycle fixes found during consolidation:

- DA3 cached-ZIP preparation now reaches the same input-fill callback as a
  first download. Connections are installed once, rather than accumulated on
  every click.
- Repository completion clears the accepted request before broadcasting, so
  a callback can synchronously start the next dataset. Cancellation no longer
  clears the busy flag of a replacement request started inside its callback.
- Dataset extraction checks consumed members rather than merely the existence
  of a destination directory. RAM and RelateAnything check their registered
  image/video sample members; Image2Mesh checks both consumer subtrees.
- LingbotMap paired masks are prepared even when only their ZIP is cached.
  The initiating dataset is retained while the scene selector is disabled.
- LingbotMap single-file videos are looked up as files, with shared integrity
  evidence, rather than searched as directories. A verified cached download
  is materialized before pending Run resumes, avoiding recursive retries
  against an input path that was never populated.
- The shared downloader now honors an explicit zero-byte floor for ordinary
  pinned files, consistently with warm-cache readiness. Its default remains
  64 KiB for models. A small real-HTTP ZIP fixture covers digest ingestion,
  repeated download bypass and shared extraction.
- Cross-checking all 678 exported catalog rows against download policies
  identified the InstantMesh synthesizer size mismatch above. Catalog rows
  without published sizes cannot establish a size-based audit by themselves.

### Windows necessity review

`std::getenv` exists on all three platforms, but its Windows narrow result is
not a UTF-8 contract. `setenv`/`unsetenv` are POSIX interfaces, not the MSVC CRT
mutation interface. The shared filesystem implementation therefore retains a
single Windows environment boundary (`_wgetenv` / `_wputenv_s`) and reuses
CVTools for conversion. The AICore-local `utf8_environment` implementation and its manual Win32 conversion
were removed. `GetEnvVar` / `SetEnvVar` now own native environment access.
Test fixtures use the same setter rather than mixing narrow Qt mutation with wide reads.
Production cache resolution does not mutate the environment. The ggml common
environment bridge also delegates ASCII backend-option mutation to that
setter, eliminating its separate Windows/POSIX branches.

CVTools' wide conversion pair now uses Qt's Unicode conversion, replacing the
deprecated codecvt implementation that does not correctly represent UTF-16
surrogate pairs on Windows. Legacy CVTools `ToQString`/`FromQString` include
locale/GBK behavior on Windows and are not substitutes for an explicit UTF-8
AICore path contract.

The common `FOpen` implementation now reuses that conversion pair. Minizip
opens UTF-8 paths through `FOpen` on every platform, eliminating a separate
Windows-only ZIP opener. Three depth exporters use `FOpenForWrite` in the
existing FileSystem API, which centralizes UTF-8 path opening and preserves the
existing POSIX 0644 creation mode; no AICore-only file-open wrapper remains.
Six traditional plugins (Animation, 3DMASC, PCL, RANSAC_SD, Cork and
PoissonRecon) use `QThread::msleep` instead of duplicate `Sleep`/`usleep`
branches.

Retained native boundaries have concrete API differences: DLL discovery
versus `dladdr`, CRT case-insensitive string names, Python executable/library
layout and Windows ACL checks, RPC child-process ownership/termination, and
`_commit` versus `fsync` for importer durability. Upstream SIBR/vendor platform
implementations were not rewritten as application-level utilities. Video
hover's Windows-specific 100 ms versus 66 ms throttle is a performance policy,
not an OS API requirement; no new benchmark was run to validate or change it.
The existing POSIX-only stderr suppression around OpenCV decode uses native
file descriptors and remains disabled on Windows. GNU-only `-no-pie` options
in the touched test setup are guarded from Darwin linkers.

### What model readiness proves

`QFile::exists(path)` proves only that a filesystem entry exists. The shared
model predicate also requires a regular readable candidate meeting the common
64 KiB floor and GGUF magic. It looks up a registry digest by filename when
available and accepts matching ingestion evidence only while size and
modification time match. Changed pinned evidence is rejected consistently by Run and transport,
so a corrupt/changed cache cannot pass one gate and fail a different plugin's
private gate. Warm checks avoid hashing gigabytes on the GUI thread.

This is not full verification of every existing GGUF: legacy files without a
ledger and custom filenames retain documented cheap-check fallback. Metadata
checks also cannot detect byte changes that preserve both size and mtime.
Downloaded pinned assets undergo the actual streamed digest comparison.
Model tensor compatibility and successful inference require real-model tests.

### Remote CI evidence and added coverage

The GitHub Actions API was checked on 2026-11-09 for branch
`users/dahailu/v3.9.5`. Windows run
[42493688868](https://github.com/Asher-1/ACloudViewer/actions/runs/42493688868)
is for base HEAD `890e8fc8cb0886f9f40cb23ca592728c639a5f98` and concluded
**failure**. Its CPU Release application job failed at
[Build ACloudViewer App](https://github.com/Asher-1/ACloudViewer/actions/runs/42493688868/job/128771980108);
subsequent C++ and native Python-runtime tests were skipped. The CUDA Release
job was cancelled. Both Debug jobs display success but skip application build
and tests. Wheel build/tests succeeded, which does not establish plugin GUI
cache correctness. The failed step includes application build, AICore contract
tests, install and package; the step name alone does not identify which part
failed. Job log access returned HTTP 403, so no specific compiler/test failure
is asserted here.

Previously the `cvpluginapi` label excluded these cache regressions from the
normal fast tier, and the contract tier selected only `capi`. Shared integrity,
HTTP/ZIP transport, dataset preparation, Unicode root/file access, and the
SAM3D/TRELLIS selected-model dialog tests now participate in the existing
`aicore-contract-tests` gate, with binary dependencies registered through its
shared CMake function. This is coverage for the next CI run, not evidence that
uncommitted changes passed native Windows or macOS tests.

The existing qInstantMesh dialog/plugin CTests and qOpenPCDet dialog CTest are
explicitly disabled on Windows in their CMake files with offscreen-deadlock
or stability rationales. Those are coverage gaps, not passing native GUI
results; the rationales were not reproduced on a Windows host in this audit.
The InstantMesh catalog size-floor regression now also joins the contract gate.

Local follow-up evidence is recorded in `consolidation-configure.log`,
`consolidation-final-build.log`, `consolidation-shared-build.log`,
`consolidation-downloader-build.log`, `consolidation-tests.log`,
`consolidation-ci-coverage.log`, and `consolidation-boundaries.log` under
`build_app/Testing/model_cache_audit`.
The final CMake registration was reconfigured successfully and the seven new
or expanded contract tests were checked for both `capi` labels and actual
contract-target binary dependencies (`consolidation-contract-registration.json`).
The synthetic repository test covers cached ZIP preparation, warm reuse,
synchronous next-dataset callbacks, incomplete-extraction repair, failure and
retry, single-file video materialization and cancellation that immediately
starts a replacement request. It does not claim published-asset inference.

Follow-up Linux results: all 22 catalog-model plugin targets and the six
traditional plugin targets affected by the sleep consolidation built
successfully. The focused suite passed 13/13; after fixing the newly found
InstantMesh floor, its catalog, dialog and plugin suite passed 3/3 (15 distinct
tests across both runs). The latter rebuild/results are recorded in
`consolidation-instantmesh-build.log` and `consolidation-instantmesh-tests.log`.
The strict plugin/AICore boundary check passed. These are cache, selection,
workflow and build checks, with synthetic inputs where explicitly documented;
they do not execute a complete real-model/backend inference matrix. Native
Windows/macOS acceptance and all-plugin real-model Run acceptance remain the
unchecked items above.

The follow-up review removed the newly introduced AICore-private
`common/file_io.hpp`. `cloudViewer::utility::filesystem::FOpenForWrite` now owns both the
shared file-opening boundary and the POSIX creation-mode policy; Windows
delegates to `FOpen` so CVTools UTF-8 conversion remains implemented once. The
Unicode filename/data-root test exercises that shared owner and checks the
POSIX permission invariant. Final Linux output is recorded in
`file-io-consolidation-final-build.log` and
`file-io-consolidation-final-tests.log` under
`build_app/Testing/model_cache_audit` (12/12 tests passed).

## Final pre-commit review

The final review additionally found and repaired the following cases:

- Retry backoff was reported as idle by the shared downloader. A replacement
  request could overwrite the pending request, and Cancel could leave its
  retry timer alive. Backoff now retains ownership until completion or
  cancellation; retry notifications observe the busy state. Cancellation
  preserves the cancelled destination across callbacks.
- Download preparation could outlive a closed dialog. Model downloaders now
  receive dialog completion; explicit Cancel paths discard pending operations.
  TRELLIS enables Cancel during preparation, locks selection controls, and
  removes a duplicate sample-button connection. SAM3D and TRELLIS reject late
  completion callbacks after cancellation/closure. Sample callbacks also
  discard detached requests instead of refilling closed dialogs. GKD detaches
  its sample request without cancelling another plugin's shared preparation.
- VLXSeek installed a lambda with `Qt::UniqueConnection` on every sample click.
  Its connection is now installed once in the constructor and gated by the
  active sample request. A synthetic cached-ZIP UI regression covers initial
  extraction/fill, warm reuse, idle broadcasts and closure during preparation.
- Linux runtime inspection lost library paths containing spaces. The shared
  Python loader parser now retains the complete resolved path. Remaining YOLO
  JSONL, MapAnything PLY and FreeSplatter tool string-path streams reuse the
  existing UTF-8 filesystem boundary. The independent tracker YAML reader uses
  the native `std::filesystem::u8path` stream overload. Unicode PLY/YAML tests
  cover the affected consumers.
- OCR layout tests still called a removed digest function, and the Python
  provenance test could select static `libclang.a` for `ctypes`. Tests now use
  the shared digest function and discover a shared libclang. A missing
  generated MapAnything reference skips only its dependent test, preserving
  independent reference-isolation and source-age tests.
- Reinstalling CUDA dependencies could copy a runtime onto itself. The shared
  bundler now reuses that file, and an actual copy failure propagates instead
  of being counted as a successful deployment. Both cases have ELF regressions.

Current-source evidence is stored under `build_app/Testing/precommit-review`.
The Python suite passes 263 tests with 143 subtests; three explicitly optional
real-reference tests are skipped (two OCR orientation tests and one generated
MapAnything twin test). Linux CuMesh was rebuilt with CUDA 12.6 / Torch 2.7.1;
both link/GPU tests pass, as does loading the bridge and its CUDA closure from
an installed directory containing spaces. The production mesh exporter also
compiles with `TRELLIS2_HAVE_CUMESH=1`, and the bridge header passes a C11 check.
These results do not establish native Windows/macOS acceptance or a complete
real-model inference matrix; the unchecked acceptance items above still apply.

Final Linux verification passed: the entire configured application/plugin/test
build (`build-final.log`), 31/31 fast CTests (`ctest-fast-final.log`), 58/58
selected interface/cache/workflow CTests (`ctest-contracts-final.log`), and 5/5
focused helper/UI CTests (`ctest-helpers-final.log`, overlapping the contract
selection). The FreeSplatter console tool also builds with the PIC flags
required when including the shared filesystem boundary's Qt headers. All 43
modified Python files and four modified shell scripts pass syntax checks;
both working-tree and index whitespace checks pass. `manifest.json` records
source hashes; `contract-selection.txt` lists the exact contract selection.
