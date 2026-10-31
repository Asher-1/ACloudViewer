# qInstantMesh — InstantMesh image-to-3D (GGML)

Single-image to textured 3D mesh via native GGML inference. In-tree port of
[InstantMesh-GGML](https://github.com/Asher-1/InstantMesh-GGML) (MIT), itself
a C++/ggml port of [TencentARC/InstantMesh](https://github.com/TencentARC/InstantMesh)
(Apache-2.0).

## Pipeline

```
image ──▶ RMBG-2.0 matting (optional, in-tree RMBG task)
      ──▶ Zero123++ multi-view diffusion (75 steps, RefOnly double-forward)
      │      cond: CLIP-Vision embeds + scheduler constants (GGUF)
      │      unet: 75×{w-pass 64² + r-pass 120×80} ── VAE decode ──▶ 6×320² views
      ──▶ DINO ViT-B/16 encoder (adaLN camera modulation)
      ──▶ TriplaneTransformer (16 layers, 1024 dim) ──▶ triplane [3,80,64,64]
      ──▶ OSGDecoder MLPs (sdf / deformation / FlexiCubes weights)
      ──▶ FlexiCubes isosurface extraction (host C++)
      ──▶ xatlas UV unwrap + net_rgb texture bake (2048 atlas)
      ──▶ typed mesh/per-corner UV/RGBA material directly into DB; GLB and OBJ/MTL/PNG are export boundaries
```

The "Geometry only" preset skips the Zero123++ diffusion and replicates the
input image as the six views (the upstream benchmark path — faster, lower
geometric quality).

## Models

Published at <https://huggingface.co/Asher-1/InstantMesh-GUFF> (14 GGUFs;
the catalog in `core/AICore/src/tasks/instantmesh/model_catalog.cpp` is the
single source of truth for URLs, sizes and SHA-256). DINO files sharing names with
Trellis use their respective catalog digests; the shared digest table keys them by
`instantmesh_models/<filename>`, never by interchangeable basenames:

| Role | Files | Notes |
|---|---|---|
| cond | `zero123pp_cond_f32.gguf` | CLIP-Vision + pipeline constants (only f32) |
| unet / vae | `zero123pp_{unet,vae}_{f16,f32}.gguf` | no q8 release |
| dino / transformer / synthesizer | `*_f16/_q8/_f32.gguf` | recommended chain: f16 |
| rmbg | **not here** | owned by the in-tree RMBG task catalog (same LFS objects) |

Weights are cached under `~/cloudViewer_data/extract/instantmesh_models/`;
RMBG under `extract/rmbg_models/` (shared with qRMBG/qTrellis). The dialog
downloads missing files through the shared `ecvModelDownloader` with the
pinned SHA-256 anchors. Both caches honor `CLOUDVIEWER_DATA_ROOT`.
RMBG precision choices come from the shared catalog; the GUI passes the selected
file to the worker, while batch defaults to the shared recommended model.
Missing or corrupt RMBG is never silently skipped; batch `DOWNLOAD 1` prepares
dependencies through the shared downloader.

Inference and download logs use the application's shared console. Input/sample
controls and execution/progress stay fixed while only configuration scrolls.
The 640x480 and 800x600 layout checks require zero horizontal overflow, fully
accessible fields and readable complete catalog labels in combo popups.
Default step/guidance values remain visible in their numeric fields; the compact
checkbox labels select whether those defaults are used.
GLB textures use the common AICore PNG encoder; no task-private stb is included.

## Sample data

The "Try sample data" button reuses the shared **Image2Mesh** dataset
(`image_to_mesh_data.zip` from the cloudViewer_downloads release) — the same
bundle qTrellis/qSAM3D use — and fills the candidate-image combo from
`examples_images/`. It calls shared `ensureDatasetReady(Image2Mesh)`, preferring
decodable `cartoon_panda.png` from that ZIP, otherwise the first sorted
 decodable candidate. The selected member is 1517334 bytes with SHA-256
`c82fea6ac66b782b2aa1c6bd133447b5f54f688c7eb44998c4b00f190d47b2b7`.
The existing `T.png` numerical reference and acceptance gates remain unchanged.
Closed/rejected windows stop receiving preparation results; reopening restores
sample controls according to the current preparation/running state. The shared repository marks extracted
Image2Mesh ready only with a decodable candidate, so corrupt PNGs do not block
recovery. Explicit real cold-cache validation (the root must not exist yet;
not registered in default offline CTest):

```sh
QT_QPA_PLATFORM=offscreen <build>/bin/plugin_test/test_qinstantmesh_dialog --sample-cache <new-absolute-data-root>
```

This mode invokes the actual sample button and checks pinned release ZIP/default
member digests, autofill, repeated clicks, warm reuse without another download,
corrupt-member fallback/offline restoration from the retained ZIP, corrupt-ZIP
rejection and closure during preparation. JSON receipts go to stdout. With
`QT_QPA_PLATFORM=xcb` and `ACV_UI_TEST_SCREENSHOTS=<new-directory>`, this mode
also saves real panda top/bottom configuration captures at both tested sizes.
Widget/sample workflow tests are not native material-display acceptance.

## Backend notes

The plugin and its test targets do not directly use QtSql. `CV_DB_LIB` supplies
scene entities (point clouds/meshes), not a SQL database. Explicit `Qt::Sql` links
were removed; the application may still have unrelated SQL consumers.

- The single llama/ggml source owner retains the current pin. `GGML_HINT_STRICT_F32`
  is an existing shared patch extension, not an official API; InstantMesh adds no
  second ggml, private RMBG engine or cuDNN.
- CUDA/Vulkan accuracy requires actual backend patch chains and operator/model
  gates; hints or successful loading alone do not prove acceptance. No private
  environment variables or global fp16 latch alter other tasks' device behavior.
- Explicit unavailable CUDA/Vulkan/Metal requests fail; only `auto` permits CPU
  fallback. Results report the actual backend. InstantMesh and shared RMBG use
  `BackendLease` with canonical device IDs, retaining the selected GPU index;
  display names such as `Vulkan0` must not feed task device selection.
- Metal retains shared migration gates; Linux tests do not replace native macOS acceptance.

## Batch (headless)

```
ACloudViewer -SILENT -INSTANTMESH_GENERATE PRESET full QUANT f16 IMAGE image.png OUT_DIR out_dir
ACloudViewer -SILENT -INSTANTMESH_GENERATE HELP RESULT_JSON instantmesh-schema.json
```

Options: `PRESET full|geometry`, `QUANT f16|q8|f32`, `USE_RMBG 0|1`,
`RMBG_MODEL path`, `STEPS n`, `GUIDANCE x`, `SEED n`, `GRID_RES n`, `GRID_SCALE x`,
`TEXTURE 0|1`, `TEXTURE_RES n`. `SEED` accepts unsigned 64-bit integers (including
0); negative values and overflow fail explicitly. Validate arguments before model
checks/downloads. OBJ preserves vertex colors; textured output adds per-corner
`vt`, relative MTL and PNG. GLB export splits UV-seam vertices at the boundary
and converts glTF image-coordinate conventions.
JSON records common `timings`, separate `load_ms`/`export_ms`, actual backend and
exported files. `glb_bake_ms` is worker GLB encoding cost; `file_export_ms` includes
OBJ/MTL/PNG/GLB/preview writes. `export_ms` sums both with submillisecond timing,
separate from inference latency. Progress-boundary `stage_ms` must not be summed
to replace common `e2e_ms`. Strip previews use plugin CPU triangle rasterization
and per-corner UV sampling of real RGBA atlases. Tests cover sparse triangle
surfaces, seams, top/bottom rows and alpha; thumbnails do not replace native GUI
material-display acceptance.

Automation uses the generic toolchain (`ACV_BINARY` selects an isolated build or installed package):

```sh
cli-anything-acloudviewer --json --mode headless ai list
cli-anything-acloudviewer --json --mode headless ai schema instantmesh
cli-anything-acloudviewer --json --mode headless ai run instantmesh --image image.png --out-dir out --option PRESET=full --option QUANT=f16
```

## Tests

- `core/AICore/tests/instantmesh/test_instantmesh_capi_contract.cpp` — ABI /
  options / preprocess / GLB / catalog contract (no assets).
- `tests/test_instantmesh_catalog.cpp` — preset resolution, quantization
  chain substitution, catalog coverage (14 files, RMBG excluded).
- `test_instantmesh_c_header.c` — public C11 header and ABI 2 struct initialization.
- `test_instantmesh_lifecycle.cpp` — real-model warm forwards, seed=0, error/cancel
  retries, chunked field queries, mid/final texture-chunk cancellation and context
  release/recreation; full behavior uses only 2 steps, not an accuracy gate.
- `tests/test_instantmesh_{worker,dialog,plugin}.cpp` — shared cancellation token,
  stale results, shared sample fill, RMBG selection, typed DB materials, OBJ export
  and native command schema/error contracts. Plugin tests accept a data root and
  backend for real geometry batch runs; not packaged-child-process acceptance.

- `test_instantmesh_scheduler.cpp` — diffusers references generated by pinned upstream
  `parity_scheduler.py --out <build>/instantmesh_reference`; retains original gates:
  tables `<1e-6`, add_noise/scale `<1e-5`, 75-step walk `<1e-4`. The shared runner
  takes `--reference instantmesh_scheduler=<reference>/scheduler`; missing exits 77,
  corrupt exits nonzero. Consumers authenticate revision, generator digest,
  seed/steps/dimensions and every reference digest/size. Real subprocesses in
  `test_instantmesh_references.py` test tampering, missing/truncated/NaN and valid references.
- `test_instantmesh_components.cpp` — production CLIP/VAE/UNet versus independent
  transformers/diffusers outputs; retains CLIP `<1e-3`, VAE encode `<5e-4` / decode
  `<5e-3`, UNet f16 `<0.35` and per-output RMSE. Pinned `parity_clip_vision.py` /
  `parity_vae.py` generate references with explicit `--snapshot` and `--out <build>`.
  Python runs only in isolated tests, never production. Original checkpoint
  revision/digest and reference identity must match. Shared runner inputs are
  `--reference instantmesh_clip=<reference>/clip_vision` and
  `--reference instantmesh_vae=<reference>/vae`; component success is not complete
  six-view/texture acceptance. UNet references use pinned `dump_unet_stages.py`,
  seed=7, B=2, H=120, W=80 w/r double-forward, with explicit-path build-directory
  adaptation and optional staged reruns omitted. Record original/adapted script
  identities; preserve upstream fixtures and fixed `/tmp` files. Authenticate
  Zero123++ checkpoint and TencentARC white-background override revisions and LFS
  sizes/digests first; consumers also check official `ReferenceOnlyAttnProc` source
  digest, config, shape and each reference file. Shared runner input is
  `--reference instantmesh_unet=<reference>/unet`; this original gate excludes
  64² condition and the 75-step trajectory.
  The same probe's `synth-load-contract [device]` uses valid GGUF containers for
  complete four-branch success, missing RGB bias, out-of-range plane_dim, zero
  hidden/layers, wrong metadata types and net_weight input shape. Errors must
  release models; retries without an error buffer still reject them.
  `test_instantmesh_synthesizer_load_{cpu,backend}` needs no external models and is
  neither an independent numeric nor complete C ABI gate. `synth-layout-contract`
  uses independently hand-calculated 100/100.1 for two-batch corner-major inputs;
  CPU/Vulkan red/green reports are retained. `synth` consumes authenticated official
  four-image grid=8, plane=64×64 references and same-weight upstream outputs.
  SDF/deformation/RGB retain original frozen gates; weight uses correct corner-major
  same-weight PyTorch quantization references and separately reports absolute,
  porting and raw old-layout output differences. Real subprocesses in
  `test_instantmesh_synth_reference_contract` reject receipt/scene tampering,
  relaxed gates, bad outputs and widened compatibility exceptions.
  Shared scenario `instantmesh-synthesizer-reference` requires four explicit roots:
  `instantmesh_synth`, `instantmesh_synth_baseline`, `instantmesh_geometry`,
  `instantmesh_weight`. Missing references or unfrozen CUDA/Metal rows stay INCOMPLETE.
- `geometry_reference.hpp` — the same component probe calls production DINO/LRM
  for blue_cat/cute_horse/fox/robot. Official geometry checkpoints and pinned
  `prep_input.py` / `parity_dino.py` / `parity_lrm.py` generate independent references.
  RefDino concatenates six uncoupled original B=1 forwards. Input is direct bicubic
  224, not yet production 320→224 preprocessing. LRM consumes independent DINO
  features to isolate transformer error, not total DINO→LRM error.
  The 48 original-upstream CPU/Vulkan × f32/f16/q8 captures record GGUF, input,
  source tree, binary, actual backend and log identities. Frozen per-view/plane
  max_abs/RMSE bounds are `1.10×error+1e-5`; same-weight output-difference budgets
  are `0.10×error+1e-5`, entered into the shared manifest before candidate sampling
  and never relaxed after candidate failures. Runner inputs:
  `--reference instantmesh_geometry=<four-image-reference-root>` and
  `--reference instantmesh_geometry_baseline=<authenticated-upstream-output-root>`;
  scenarios `instantmesh-dino-reference` / `instantmesh-lrm-reference`. Missing
  CUDA/Metal baselines return INCOMPLETE. Contract subprocesses cover checkpoint,
  NaN, missing/corrupt baseline, backend identity and missing/invalid/relaxed
  per-output gates. Missing capture receipts exit 77; partial pytest skips cannot
  establish acceptance.
- `test_instantmesh_dino_convolution.cpp` — production DINO source with independent
  constant/double-precision oracles for F32 weights not exactly representable in
  half precision, F16 patch small-term accumulation and F32/F16 linear. A zero
  residual block preserves patches and feeds camera input into the real graph.
  Frozen `<2e-5` passed CPU/CUDA/Vulkan; separately detects F32 staging and F16
  accumulation loss. Weight precision is not accumulation precision. Task-private
  GEMM in `numerics.hpp` requests official F32 accumulation and shared STRICT_F32
  staging. DINO/LRM and Synth SDF/deformation/RGB CPU graphs call
  `preserve_cpu_matmul_dispatch()`; `instantmesh_cpu_matmul_` retains original
  vec-dot dispatch without changing global LLAMAFILE, attention, GPU nodes or
  other tasks. All five backend manifests register the same CPU patch. Temporary
  valid GGUF containers cover missing DINO tensors and invalid attention-head
  metadata: loading fails and frees resources, including retries without error
  buffers. `dino_load` checks positive/divisible hparams and all required tensor
  names before graph construction; this is not tensor-shape, other-component or
  complete GGUF-error coverage.
- `test_instantmesh_vae_convolution.cpp` — the same independent double convolution
  oracle compiles real VAE/UNet helpers for non-square, multi-batch, 16-channel and
  small-term F32 GEMM/mixed-im2col cases, frozen `<2e-5`. Both F32 GEMMs request
  the shared strict hint; mixed convolutions retain F16 im2col with official
  `ggml_prec_set_acc(..., GGML_PREC_F32)`. `test_instantmesh_vae_stages` is a manual
  CPU/GPU node diagnostic, not an independent accuracy/performance gate.
- `test_instantmesh_rmbg_{f32,f16,q8}` — real shared-model composition loading,
  cancellation and inference retries after cancellation.
- `test_instantmesh_timing_boundary` — large padded BGRA packing belongs in C ABI E2E.
- `test_instantmesh_catalog_consistency` — real catalog-dump subprocess digests/sizes
  match the owning catalog.
- `instantmesh_backend_ab.py` — one load per device, warm-context warmups/repeated
  forwards, unmodified common timing samples, actual backend, p50/p95 and raw
  samples. `--lib` accepts a library or build/install directory; `--warmups/--iterations`
  follow the shared runner protocol. Backend comparison is not controlled
  baseline/candidate A/B. Manifest registers geometry/full × f32/f16/q8 six chains;
  default light includes full/q8 and the same ZIP's `T.png`. Complete independent
  chain references remain open: report `independent_reference_complete=0` and fail
  accuracy; hashes/bboxes cannot turn this into PASS.

Historical isolated Linux Qt5 CLIP/VAE/UNet component gates passed CPU, CUDA and
Vulkan. VAE decode max_abs: `0.003019`, `0.003345`, `0.002943`; UNet: `0.000636`,
`0.000695`, `0.002315`. This is component accuracy evidence, not full-chain,
quantization-added-error or performance acceptance; old failures and gates remain.
The first four-image DINO/LRM CPU matrix passed only DINO f32 and LRM q8; all six
Vulkan rows failed. After DINO staging/task-private GEMM accumulation fixes, all
three Vulkan DINO precisions passed four images with zero same-weight per-output
difference. Vulkan LRM then met absolute gates but exceeded frozen same-weight
budgets; those FAIL reports remain. Approved compatibility-arithmetic restoration
uses shared pristine transposed-convolution dispatch for `instantmesh_lrm_convt_`,
bypassing SAM3 fast path/F32 scalar override, without attention expansion or
diagnostic environment variables.
Same-source/config CPU single-variable experiments passed six rows only with
LLAMAFILE=OFF; ON reproduced four failures. Global `use_ref` also changed attention
and still failed four rows; that experimental test option was withdrawn, never
entered production ABI. Isolated owner `build_instantmesh_cpu_compat_08dea4ce`
retains LLAMAFILE=ON and uses CPU node tags for original dispatch.
`component-build/instantmesh-components-cpu-vulkan-green.xml` records 24 passes:
CPU/Vulkan × DINO/LRM × f32/f16/q8, original CLIP/VAE/UNet gates and three convolution
oracles. All four-image per-view/plane same-weight max_abs/RMSE differences are 0;
absolute/compatibility gates remain frozen. `instantmesh-shared-operators-cpu-compat.xml`
records 16 shared-operator/source-owner passes; five-manifest clean replay, full
reverse verification and repeat application have separate receipts. Metal replay
is not native-device acceptance. These isolated component results use CPU variants=ON
and do not replace the existing main build, full production C ABI/model chains,
performance or platform acceptance. `instantmesh-dino-load-red.xml` and green
reports remain in the isolated build.
`instantmesh-resume-final-regression.xml` records 27 passes, explicitly excluding
six DINO/LRM accuracy rows and geometry reference contracts;
`instantmesh-resume-vulkan-matrix.xml` records three DINO passes and three LRM FAILs.
Historical 201 L1/L2/L5 passes in `Testing/instantmesh-resume-agent-contract.xml`
do not prove packaged L3 or GUI/RPC L4 acceptance. Manual shared-runner invocation
requires matching `AICORE_QT_LIBRARY_DIR` / `AICORE_CUDA_LIBRARY_DIR`; missing CUDA
modules must fail, never label CPU fallback as GPU success.
Production Qt5 CPU/CUDA/Vulkan v2 build exited 0 for AICore, InstantMesh/RPC plugins
and App; this is not a full-test or installation receipt. Full-contract failures
for missing Attribute input and four unauthorized OpenPCDet environment reads
remain; the allowlist was not expanded. Before frozen-scene identity checks,
real negatives produced 5 FAILs; production contract v4 after checks records 11
CTest passes with no skips, including 23 Synth reference cases, 6 catalog cases
and worker/dialog/plugin offscreen contracts, not native GUI material acceptance.
Pinned build-directory OSGDecoder/real FlexiCubes generated independent four-image
grid=8, plane=64×64 component references, excluding production deformation
tanh/empty-SDF postprocessing, full diffusion or UV texture. Fresh original-tool
compilation and frozen-ggml authentication preceded 24 Synthesizer baseline rows.
Original F32 net_weight max_abs versus independent truth was 0.151–0.181;
feature-major single-variable diagnosis reproduced original output to about 5e-8,
confirming corner→feature permutation differs from official decoder layout.
The user authorized an explicit exception only for this weight layout: production
uses corner-major and dequantizes published weights in the small weight graph to
retain F32 activations. Old baselines, failures and every other output gate remain.
The correct isolated CPU owner passed three Synth precisions; Vulkan f16 SDF still
exceeded original compatibility bounds. Extra CPU FAILs from an incorrectly reused
old owner also remain. Four-image real-source node traces matched the first three
layers and final-layer input; the first difference was final single-row F16 SDF
GEMM. Repeating only the final weight row in a test graph avoided swapped matvec
and restored each original value. Production patch 0005 preserves dispatch only
for `instantmesh_sdf_matmul_`, registered in both Vulkan manifests. New-owner
numeric/shared gates and production updates still require verification; diagnosis
is not fix acceptance.
The preceding resume rebuilt contract tools in the original production directory:
`instantmesh-contract-resume-v7.xml` records 18 passes (15 prior contracts and 3
shared helpers); explicit rebuilt binaries/authenticated references yielded 83
passes in `instantmesh-reference-pytest-resume-v6.xml`. Earlier 83 SKIPs without
environment paths remain and do not count. Corrupt-image readiness RED and fixed
GREEN reports are both retained. `reference/cold-sample-v2.json` records one real
empty-root download, four ready events and zero failures. ZIP/T.png hashes/sizes
match the shared manifest, including corrupt-member fallback/offline restoration,
not native material acceptance.
Real Qt5 xcb GUI `instantmesh-rpc-mcp-green-v6.xml` records 16 passes for InstantMesh
CLI/native schema, generic MCP schema, RPC HELP/catalog, invalid seed/overflow and
asynchronous-child responsiveness. Negatives first reproduced rejected seeds still
creating OUT_DIR. Shared prepare now has an optional empty-default validator;
existing InstantMesh checks run before mkdir, HELP/LIST_MODELS skip generation
validation, and other consumers' default flow is unchanged. Broad RPC schema
reports retain 19 disabled-plugin FAILs. Broad L1/L2/L5
`instantmesh-agent-resume-v3.xml` records 201 PASS/1 FAIL for unavailable LLM native
schema in this isolated build; historical 201 passes do not overwrite it, and the
plugin allowlist is not widened. These build-directory automation results do not
prove packaged L3, real-model RPC L4 or the complete task set.
The independently configured `build_instantmesh_production_sdf_08dea4ce` build is
paused for the SQL/language cleanup. Its queued 34 component and 16 shared SDF
checks remain pending; isolated component results do not prove this production
build passes. Full acceptance still requires independent references, 14 GGUFs and
three RMBG precisions, 75-step latents/six views/textures, controlled same-host
p50/p95 A/B, real GUI materials and native Qt5/Qt6 platform evidence. Contracts,
hashes/bboxes, exit 77 or missing references cannot establish complete accuracy
or performance.

SQL/language cleanup verification: the reconfigured plugin and four test targets
were rebuilt without QtSql include/define/link entries. `instantmesh-english-contract-v1.xml`
records 16 CTest passes (including the shared runner and four reference-contract
suites); `instantmesh-english-agent-v1.xml` records 12 scoped CLI/RPC/MCP and
entry-point tests passing. `instantmesh-english-shared-v2.xml` records 3 shared
helper passes; the v1 shared-test selection matched no tests and is not evidence.
Five-chain forward/reverse/idempotence checks passed in
`reference/english-patch-replay-v1/receipt.json`; replay is not native-device
acceptance. Source/test/documentation and local validation-driver
text is English; original reports, reference receipts and frozen numeric gates
remain unchanged. Translation changes adapter/patch hashes, so historical receipts
are not rewritten to claim fresh evidence. Reconfiguration refreshed the old
production build's disposable fetched source after the patch-comment hash changed;
verification reused installed backend libraries without a third-party rebuild.
The paused new-production/numeric queue was not resumed.

### Linux continuation evidence (supersedes the paused status above)

The production queue was resumed without changing the frozen gates or old
receipts. The independent F32 diffusion generator completed all 75 steps and
recorded 538 authenticated members (531 float payloads and seven PNGs). Its
actual environment was torch 2.6.0, diffusers 0.39.0 and transformers 5.14.1;
this does not claim the versions in the upstream historical requirements file
were installed. The reference remains diffusion-only, not complete
geometry, texture or performance evidence.

Input diagnosis reproduced HWC/NCHW disagreement, missing downsampling
antialiasing, the incorrect VAE bicubic filter and truncated white-alpha
composition. Zero123++ now uses task-private uint8 bilinear VAE and bicubic
CLIP inputs with adaptive int16 coefficients, separable byte rounding and PIL
paste composition. DINO and the compatibility RGB resizer retain their prior
interpolation. `preprocess-final-green-v3.xml` records 26 passing checks;
`preprocessing-reference-v2.log` records zero max_abs/RMSE for both actual T.png
encoder inputs against the authenticated independent tensors.

The natural noise stream separately failed nine Torch CPU comparisons. Low
24-bit MT19937 uniforms and 16-value Box-Muller pairing now match the independent
input contract without adding a framework production dependency.
`rng-reference-green-v2.xml` records 35 preprocessing/noise checks passing;
`rng-authenticated-v1.xml` additionally checks all 151 independent 75-step noise
payloads. Their maximum error is 4.768372e-7, below the new fixed 1e-6 noise-input gate;
this input gate was not widened and does not replace the historical output gates.
These results do not validate the final latents or generated views.

`instantmesh-current-v4.xml` records 48 production CTest passes, including the
new authenticated noise input gate, original component limits, shared RMBG
precisions and short lifecycle cases. `automation-broad-v4.xml` records 222
build-directory L1/L2/L5 and native schema checks passing. The v1 broad configure
FAIL remains preserved: qMapAnything linked a nonexistent Eigen target when
using vendored Eigen; both consumers now use the existing `3rdparty_eigen3`
provider. No consumer was disabled to pass the broad gate.

Complete 75-step output accuracy, six-chain/14-model coverage, controlled
same-host release-profile A/B, native GUI materials, isolated installer L3 and
real-model RPC L4, and Windows/macOS acceptance remain separate unfinished
gates. `independent_reference_complete` remains zero in the full-chain probe;
input equality and build-directory schema checks do not close those gates.

The next default full/f16 CUDA run aborted after diffusion: the decoder's
3900 MiB graph reservation failed, and unchecked tensor upload then asserted.
The retained `headless-default-full-diag-v2.trace` and `headless-gdb-v4.log`
identify that path. Encoder/decoder allocation failures now release their graph
resources and return an error; two injected-allocation tests first aborted and
then passed. A layer/row-tiled decoder experiment passed a 3 GiB per-graph
acceptance check, not a physical process-peak limit: the test wrapper allocated
before inspecting the reservation. It failed its fixed same-weight numerical check (`max_abs=0.00560694933`,
`RMSE=8.269765e-5`). That production experiment was withdrawn; its output and
failure remain, and the original decoder graph and component gates are unchanged.

The context now parks its inactive CLIP weights in host storage after conditioning
and restores them before a subsequent forward. Transactional tensor copies preserve
GGUF metadata, dtype/layout and exact bytes without reopening model files, changing
the requested inference backend, or resetting shared backend/RMBG state.
`instantmesh-current-v5.xml` records 53 passes (CPU/Vulkan component/lifecycle
checks and f32/f16/q8 weight-copy checks); a separate CUDA f16 weight-copy test
also passed. `automation-broad-v5.xml` records 222 passes. These are not complete
output/performance evidence. The fresh `headless-default-full-v5.json` reports
`std::bad_alloc` rather than an abort; its cause remains under investigation.
No default full-chain, native material, installed-package or cross-platform
acceptance is claimed from these receipts.

Later default headless CUDA evidence (`headless-default-full-v7.json`) records
one successful full/f16 75-step run with 11,838 vertices, 23,648 triangles and
an exported texture. Inactive CLIP, UNet and VAE weights now use context-owned
exact host storage between stages; this is not a controlled performance A/B.
Historical v6 missing-GoogleTest, v7 missing-translation and v8 CUDA-bundler
failures remain. Real ELF regression tests reproduced four bundler failures
before the path/transitive-dependency/failure-propagation correction and then
passed all five cases. v9 and v10 each completed normal Qt IFW creation,
isolated installation and 21 installed-schema checks. v10 additionally records
54 CTest passes and 222 broad L1/L2/L5/build-schema passes, without disabling
consumers or changing accuracy gates.

Native installed v9 diagnosis (`native-gui-v3/gui.log`) completed the real
full/f16 CUDA worker, but its typed result never reached DB. A real cross-thread
signal test reproduced `Cannot queue arguments of type 'InstantMeshRunResult'`;
worker construction now registers that type before connections are made.
`gui-queued-red-v1.xml` and `gui-queued-green-v1.xml` retain the failing and
three-test passing results. Installed v10 (`native-gui-v5/scene-result.json`)
then placed the real 11,864-vertex/23,700-triangle typed mesh in DB. The first
v10 driver assumed the mesh itself carried the source-prefixed name; that
receipt failed while the source name was on its associated cloud. Subsequent
matching uses that existing child identity rather than importing an exported
mesh. This does not change the production result path.

The v10 native front/back/left screenshots are gray and remain material FAIL
evidence. The plugin populated only the legacy texture cache, whereas VTK
consumes the diffuse texture-map list. `gui-diffuse-red-v1.xml` reproduced that
missing renderer contract; the plugin now registers the same cached owned
image as DIFFUSE without writing/reloading a texture file. Three plugin/worker/
dialog checks passed in `gui-diffuse-green-v1.xml`; fresh installed visual
acceptance is still required. The v10 complete L4 run is also retained as FAIL:
129 passed, five failed and 12 setup errors. Delaunay explicitly lacked CGAL,
PCD export failed, and the real InstantMesh RPC child reported VAE encode
failure. Neither the passing subset nor installed L3 closes the broad L4 gate.
Full independent output accuracy, six-chain/14-model coverage, controlled
same-host release-profile A/B and Windows/macOS acceptance remain unfinished.
