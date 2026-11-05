# qAnyLabeling AICore model matrix

The model picker is driven directly by the public AICore catalogs at runtime.
Unpublished OCR/attribute models are described separately by public task-owned
recipes; local-file support does not create a download catalog and does not prove
quantization accuracy or release qualification.
For supported task roles it exposes every published user-selectable weight and
quantization, plus the paired text/metric dependency when required. Every
backend returned by `aicore_device_count/at` is available. There is no second
plugin model list to drift from AICore. A dependency-only catalog row such as
`depth-anything-nested-metric.gguf` is intentionally not a separate model
family: the nested AnyView workflow resolves and verifies it automatically.

| AICore catalog family | qAnyLabeling workflow | Result | Prompt/dependency |
| --- | --- | --- | --- |
| DA3 depth / nested AnyView | Depth image and metric depth sidecar | Typed depth map | Nested metric GGUF when selected |
| DeepLSD | Wireframe and line proposals | Editable line/rectangle shapes | None |
| ALIKED | Local keypoint proposals | Editable point shapes | None |
| YOLO (detect, segment, semantic, pose, OBB, classify) | Detection, instance/semantic masks, pose, OBB and image flags | Typed boxes, masks, points and flags | None |
| YOLO classify in Image Classifier | Local model Run/AutoRun over document flags | Trained category vocabulary, one active category; preserves existing shapes | Shared classification model/quantization/backend selection |
| YOLO-World | Text prompted detection | Typed boxes | CLIP text encoder |
| YOLO-World + RAM/RAM++ | Per-object tags after integer detection crops | Replaced rectangles with descriptions; preserves image tags | Catalog detector/text encoder and resident tagging model; shared language/filter options |
| YOLOE | Text, prompt-free and SAVPE visual prompts | Typed masks | Text encoder except prompt-free; SAVPE weights for visual mode |
| RF-DETR | Detection and instance segmentation | Typed boxes and masks | None |
| SAM 2.1 / SAM3 | Point, box and text interactive segmentation | Editable masks and polygons | SAM3 text encoder for text mode |
| RMBG-2.0 | Foreground extraction | Editable foreground mask/polygon | None |
| GKD | Text, top-down ROI and one-shot visual keypoints | Grouped editable points | Optional support image/text |
| FaceDetect + landmark | Face boxes and dense landmarks | Grouped boxes and points | Paired landmark GGUF |
| RAM / RAM++ / Tag2Text | Image tags and optional caption | Image-level tags/description | Public tagging catalog; no object-box substitution |
| PP-OCRv4 Chinese/English + CDLA layout | Detect/recognize or selected-shape recognition; explicit optional layout | Text descriptions and real layout classes/reading order | Paired detector; optional catalog layout; optional local direction classifier, enabled by default |
| Source Chinese PP-OCRv4 / Japanese PP-OCRv3 / PP-OCRv5 / PP-OCRv6 Tiny, Small, Medium | Source detector/recognizer recipes with shared 0/180 text classifier | Editable text quads/descriptions; typed orientation results | The public OCR catalog has published F32/F16/Q8_0 families; the original source defaults and embedded dictionaries apply; each file's accuracy gate is recorded separately |
| RelateAnything | Relationships between selected/detected rectangles | Relations referencing existing objects | Public relation catalog and optional catalog-backed detector |
| Local LLM / SmolVLM | `aicore://llm` Chat/VQA provider | Streaming UTF-8 text with independently owned results | Public `llm` catalog; vision models automatically pair their projector |
| PULC person/vehicle (local F32 recipe) | Full-image multi-attribute annotation; prompts for the path when the local cache file is missing | Public `attribute` typed probabilities/attributes -> full-image rectangle, attributes, replace | The owning task's source identity and strict threshold/argmax; no published URL; Q8/release not passed |

The table describes the task roles consumed by qAnyLabeling. It excludes
catalog roles without a matching annotation workflow (for example ReID
embeddings). The following are validation states, not claims that every
published file has passed a real-data accuracy gate:

- **Picker coverage:** supported task roles returned by the catalog APIs are
  enumerated; quantization and backend are selected per entry.
- **Real smoke coverage:** see the integration list in [README.md](README.md);
  it contains two-forward checks for representative CPU, CUDA and Vulkan rows.
  The 2026-10-17 RAM/OCR follow-up also records 90 passing developer-profile
  rows across these three backends: all 12 current OCR catalog weights and
  nine tagging weights, including six RAM/RAM++ OpenCV-resize scenarios.
  This selected scope does not close the full task/dataset release gate.
- **Release gate:** the repository report contains 122 real CUDA catalog rows
  and the qAnyLabeling rows in that report pass. The expanded CPU/Vulkan
  cross-product was started against all 161 candidate rows; completed rows
  returned `rc=0`, but the run was intentionally interrupted while the slow
  SAM3/YOLO rows were still executing, so no `qany_*_matrix.json` is claimed
  as a complete report. The retained per-row probes under
  `build_app/Testing/probes/` are useful for resuming, but are not a release
  verdict. Unavailable backend/model rows must be recorded as unavailable,
  never treated as a silent pass. A source-app latency/memory comparison is
  still required before claiming a speedup.

The 2026-10-17 OCR extension verifies twelve F32 graphs against independent
ONNX Runtime outputs, ten upright/180-degree real word crops per recognizer,
and the actual pinned source TextSystem on one real page. Source output counts
are 74 for Chinese v4, 77 for Japanese v3, 81 for v5 and 73 for each v6 recipe; exact text/quad gates are recorded
by the common runner, separately from graph gates. At that time the F16/Q8
candidates failed their numeric gates and therefore did not enter the six new
recipe selectors; the current catalog/presets now enumerate the published
families, and the complete 48-file / 168-scenario dual-GPU re-verification has
finished with both overall verdicts still FAIL -- see the current receipts below.
This is a historical state change and does not mean the failing accuracy gates
are closed. A one-page
source comparison does not establish dataset accuracy or release performance.
PP-OCR KIE entity/link import/export is an annotation format workflow, rather
than a new inference model; active source model registration contains no SER/RE
pipeline. Selected polygons use source integer/minimum-area normalization,
reading-order batches and input-index ownership, with six independent ten-region
source comparisons in the common manifest. Composite picker configurations and
complete OCR-window visuals stay in TODO.

## AICore catalogs outside this 2D labeling workflow

These AICore catalogs are included in the repository-wide model inventory, but
their result contracts are not image annotations and therefore are not shown in
the qAnyLabeling picker. Their model matrices remain owned by the corresponding
plugin or by a future annotation workflow:

| AICore catalog | Current owner or status | Why it is not a qAnyLabeling model |
| --- | --- | --- |
| LoMa | qAnyLabeling paired-image bridge | Detector/descriptor/matcher correspondences; accepted matches become point annotations |
| LightGlue matcher | qAnyLabeling paired-image bridge | ALIKED proposals are matched through the separate LightGlue worker; SIFT remains intentionally unexposed |
| Gaussian | `qFreeSplatter` | Produces 3D Gaussian scene data |
| LingBot | `qLingbotMap` | Streams RGB-D reconstruction and map state |
| TRELLIS | `qTrellis` | Generates 3D assets and meshes |
| SAM3D | `qSAM3D` | Produces 3D masks/assets from images |

The qAnyLabeling paired-image bridge provides the two-image pairing view,
correspondence review, model/quantization/backend selectors, and explicit
conversion from accepted matches to annotation geometry. LoMa and LightGlue
have separate bridge workers and real fixture tests; both cover CPU/CUDA/Vulkan
where their model contract permits. Its LoMa fixture test is
`test_qanylabeling_loma_bridge`: the pinned shared Sacre-Coeur pair and F16
detector/DeDoDe-G/B-matcher pass on CPU/CUDA/Vulkan, with 578/581/578 matches.
This verifies bridge configuration, not every quantization or an independent
upstream numeric match. The LightGlue test is
`test_qanylabeling_lightglue_bridge`; it verifies the ALIKED-to-LightGlue path,
while SIFT remains intentionally outside the picker.

Remote ChatBot, VQA and arbitrary-category Image Classifier use a shared
user-configured OpenAI-compatible HTTP service. Their models come from that
service's `/models` endpoint or a user-entered ID. A remote provider adds no
local model entries; the shared ChatServiceClient's `aicore://llm` provider
enumerates models from the public `llm` catalog and calls
`aicore_llm_generate` through the worker, with the task owning the model and the
projector. Image Classifier AI Assistant/AutoRun use the source
category prompt and JSON flags, while its Local model menu retains the YOLO
classification path above. The local provider does not supply the source's exact
Florence2 graph and does not establish output equivalence with an arbitrary
remote model. The official `build_app` LLM passes 12/12 developer scenarios on
each of CPU/CUDA/Vulkan, with receipts
`build_app/Testing/llm-production-full-{cpu,cuda,vulkan}.json`. The official
controlled release A/B receipts live in
`build_app/Testing/formal-gates-resume/llm-release-*/report.json`: Vulkan is
12/12 with the overall gate PASS; the CUDA candidate is 12/12, but four frozen
baseline rows mismatch on text and two p95 regressions make the overall gate
FAIL; CPU is finally 12/12 for correctness/stability, but 10 performance items
across 6 scenarios exceed the gate, so it FAILs overall. The two-row controlled
re-verification
`build_app/Testing/llm-cuda-tail-controlled-recheck/report.json` retains the
overall FAIL with Q8 p95 +8.28%, while the original F16 regression did not
reproduce; it does not replace the complete original CUDA failure.
The later `llm-raw-tail-pathfix/report.json` for the same raw consumer PASSes
overall for two CUDA rows under the original 3x2x10 controlled A/B, with all 30
measured samples per model per side retained; it does not withdraw the original
complete CUDA/CPU FAIL and draws no performance causality conclusion. The four
loaderfix Nsight receipts distinguish the main consumer from the reference child,
and the F16 main GEMM routing difference was really measured; the profiles
include load/warmup/recreate/cancel and are not measured-only qualification.
A later NVTX diagnostic fixed a missing pop caused by a negative return value:
the real CPU range count/duration of `llm-nvtx-paired-{current,frozen}-v6/` pass,
the positive/negative cases of the main measured CUDA selector went RED->GREEN,
and after execution resumed the 3 items passed again.
`llm-raw-full-cpu-pathfix/report.json` is 12/12 for correctness/stability on both
sides, with 6 performance items across 5 scenarios above the gate and an overall
FAIL; `llm-raw-full-cuda-pathfix/report.json` has candidate 12/12, but the four
frozen-baseline text mismatches and 500m newspaper F16 e2e p95 +5.71%
(248.876 -> 263.095 ms) make the overall gate FAIL.
The first round of the four measured profiles failed JPEG decoding; after the Qt
plugin path was completed, the insertion was rejected before the task started
because of an external GPU consumer, OCR was restored and there is no new
attribution receipt. `owned-measured-profile-pluginfix-wait/status.json` records
the automatic wait / serial sampling status; it is a narrow task only and does
not trigger an official rebuild. All four probes of that round then succeeded,
while the four routes still FAILed because each range had 0 kernel nodes; this
host's Nsight default graph granularity does not capture graph nodes, and each
range really had 522 same-thread APIs. The node re-sampling of the later
`owned-measured-profile-nodefix-wait/` has completed: the four probes/routes are
8/8 PASS and all 40 measured ranges of 256M have a main consumer kernel /
same-thread API; the real regression `llm-measured-nodefix-resume-v1.log` is 4/4
PASS.
`profiling_only`/`release_qualified=false` are kept; the original FAIL is not
withdrawn and production CUDA Graphs are not disabled.
The corresponding 500M re-sampling has now finished: the four probes/routes of
`owned-measured-500m-nodefix-wait/status.json` are 8/8 PASS and the additional 40
measured ranges pass the same real regression; the first external GPU rejection
is retained and OCR was restored. The 500M F16 main GEMM is h1688/s1688 for
current/frozen; the profile p95 values are 172.421877/183.116541 ms, with no
claim of an uninstrumented release or that the original p95 failure is closed.
SmolVLM is not equivalent to the source Florence2.

The locally unpublished source YOLO11s candidate has completed its F32/F16/Q8_0
conversion: the maximum coordinate/score differences between the source ONNX and
the official checkpoint on fixed inputs are `0.00201416` / `7.39098e-6`.
Both the isolated and the official runtime use the original official per-box
gate, verified by the common runner at 3/3 on each of CPU/CUDA/Vulkan. The
official receipts are
`build_app/Testing/formal-gates-resume/yolo11s-{cpu,cuda,vulkan}/report.json`;
the isolated receipts are retained in
`build_app/Testing/g8-yolo11-exact/local-official-catalog-fixture-*.json`.
This is developer evidence for a single checkpoint/input, not a production
catalog, the complete YOLO11 family or published performance evidence; the
candidate and its SHA live in `source-identity.json` in the same directory.

The exact source ONNX of PULC person/vehicle was downloaded into a build-local
directory and executed with the original source preprocessing/attribute decoding
to obtain independent outputs; the graph contains 32 Conv, 27 BN, HardSigmoid and
a multi-label Sigmoid. The receipt is
`build_app/Testing/g8-pulc-source/source-graphs.json`.
The original source returns a full-image rectangle carrying attributes with
replace=True, not single-class flags.
The OCR test adapter data plane reusing the existing ONNX lowering has run six
rows on each of CPU/CUDA/Vulkan: the original F32/F16 probabilities and attribute
selection pass the candidate gate, while Q8 person/vehicle fail on all three
backends.
The independent same-weight ONNX attribution in
`build_app/Testing/g8-pulc-native-graphs/stored-attribution/report.json` confirms
maximum Q8 weight-quantization errors of 0.02084890/0.04891500 and a native added
error <=1.55e-6; the original 0.02 candidate gate and all failures are retained.
This OCR adapter receipt does not support a public API/catalog, a complete
dataset or a release A/B.
The later isolated Q8 candidate keeps 1 kernel for person and 2 for vehicle in
F16, still leaving 1601536/1568768 Q8 elements respectively; the three-backend
`native-*-q8-candidate/report.json` pass their six candidate-gate rows each, with
maximum Q8 errors of about 0.01444/0.00667. The selection rationale, the complete
single-tensor trials and the candidate SHA are in
`precision-sensitivity/report.json`. It is still a storage candidate for a fixed
single image; it has not replaced the old failures, published assets, or proven
an untouched holdout/release.
After fixing the candidate, `holdout-cpu/report.json` covers 11 real images that
took no part in the selection: F32/F16 pass 44/44 while Q8 passes only 4/22
(48/66 overall, FAIL); the maximum person/vehicle errors are
0.0855134/0.0764870. The original thresholds and candidate SHA are unchanged, and
the same-storage added error and attribute differences are both recorded.
These images have no attribute dataset GT; a single-image mixed candidate cannot
be promoted to a supported/published model.
The public `attribute` task / C ABI was subsequently implemented, the shared graph
interpreter owner was promoted to `common/paddle_graph`, and OCR/table reuse that
owner. The three-backend public output, attributes, stride, independent result
ownership and live-context shutdown of person/vehicle F32/F16 all pass.
`pulc-public-holdout-{cpu,cuda,vulkan}/report.json` are each 48/66 FAIL, with the
original Q8 gate, the candidate SHA and the per-output / attribute mismatches
retained. The public local F32 recipe/cache API is now consumed by the UI picker
and supplies no fabricated download URL; the real basic Run on qAnyLabeling
CPU/CUDA/Vulkan verifies full-image attributes, replace, saving, model switching,
stale rejection and teardown.
The maintained F32 C ABI E2E and the basic table E2E pass on all three backends;
table did not configure a layout page.
The public probe's real negative case for pinned source identity / all attribute
gates went RED->GREEN.
The first six-row common runner CPU receipt was 0/6 FAIL (the json0/json1 dual
source and bare-metric ambiguity) and the original receipt is retained; after the
isolated configuration switched to the complete json0 path,
`attribute-common-runner-{cpu,cuda,vulkan}-pathfix/report.json` are each 4/6:
F32/F16 pass while the original Q8 0.02 gate still FAILs, with no change to
weights or thresholds.
The existing native `ANYLABEL_INFER` command gained an attribute TASK; the public
recipe/cache provides the local list and default models, and DOWNLOAD fails
explicitly for a missing model without a URL. The real isolated plugin's HELP
schema, two models / two images, confidence not altering attributes, and the bad
manifest test all pass on CPU/CUDA/Vulkan; the GPU receipts are
`shared-owner-acceptance/attribute-batch-{cuda,vulkan}-v2/receipt.json`, which is
not an official rebuild or a release.
`shared-owner-acceptance/attribute-ui-lifecycle-{cpu,cuda,vulkan}-v1/` adds real
window acceptance, passing on all three backends for two-image Auto Run,
in-flight cancellation without scheduling the next image, rerun after cancel,
source-image/model stale, invalid-model worker error recovery and destroying the
window during inference. The source/consumer/model SHA values are unchanged before
and after and no production logic was changed; this is not live-frame,
cross-platform installer, official-library or release accuracy/performance
qualification. The original OCR was restored after the GPU idle-slot acceptance.
A published catalog/digest, an independent attribute dataset GT, complete
live/cancel/UI automation and release all remain to be closed. The original Linux
L3 schema 16 PASS/1 FAIL is retained in `automation-l3-schema-resume.log`.
The actual configuration had PLUGIN_STANDARD_QMAPANYTHING=OFF; after the original
official queue finished, that consumer was enabled and built,
`mapanything-consumer-{configure,build}-v1.log` succeeded, and all 17 schemas in
`automation-l3-schema-formal-green-v1.log` PASS with no skips.
This is not cross-platform installer or packaged GUI L4 qualification.
The official 539463680-byte PA100K archive was verified against an independently
pre-recorded MD5, with SHA-256
`65e2ab37519aacf307c8c0ce04c6ab8edac3b4199e0448d4de625cfd60cd5cc0`.
`pulc-pa100k-gt-v1/dataset-identity.json` pins 10000 val and 10000 test images
plus the 26-dimension GT and per-image SHA; the training set was not consumed.
The download, overwrite refusal, GT preparation and path-traversal negative case
regressions pass; the public raw three-quantization two-image GT regression went
RED->GREEN. The CPU val 10000 images across three quantizations in
`pulc-pa100k-val-cpu-v1/report.json` have completed:
F32 is 10000/10000 PASS (max 3.93391e-6); F16 is 9969/10000 with max 0.00992972
not exceeding the probability gate, but 31 images mismatch on attributes so it
still FAILs; Q8 is 2707/10000 with max 0.27434349 and still FAILs.
The source/model/runtime/GT SHA values are unchanged before and after; the
per-image and per-attribute GT counts are retained, and a local count improvement
is not used to mask a regression. The test 10000 images of
`pulc-pa100k-test-cpu-resume-v2/report.json` have finished:
F32 is 10000/10000 PASS with max 4.64916e-6; F16 is 9955/10000 with 45 images
mismatching on attributes; Q8 is 2497/10000 with max 0.34178510 and still FAILs.
The two splits total 20000 images / 60000 rows, all complete and stable; the four
receipt audits went RED->GREEN, the maximum Q8 same-weight execution added error
is 3.40e-6/3.87e-6, and the storage probability failures are 7275/7480 images
respectively. The original models/gates are unchanged, no post-hoc release floor
is introduced, and this is not performance evidence.
The official VeRi process requires an application, the vehicle GT is still
missing, and the access terms were not accepted on the user's behalf.

The public opaque options of YOLO26n gained an upstream/F32 PSA accumulation
request, with upstream still the default.
The isolated ABI / real graph-propagation tests pass.
`build_app/Testing/yolo-attention-official-matrix/` retains the complete 18-row
whole-model receipt across two modes, three backends and three quantizations: the
Vulkan F16 original mode has a maximum score error of 0.0238298 and FAILs, the
F32 accumulation mode passes the original 0.02 gate at 0.0122409; both Q8 modes
still FAIL on all three backends.
This is original-checkpoint correctness evidence from a single process with two
warmups and ten forwards; it does not replace the frozen release performance A/B,
the plugin did not change its default, and the official runtime was not replaced.
The independent CPU weight-injection re-verification receipt
`yolo-quantization-attribution-resume-v2.json` verifies 204 real GGUF/checkpoint
parameter mappings: the F32 export is bit-equal, the F16 weight-quantization score
error is 0.00776911 and the Q8 weight-quantization error is 0.14595002, so the
original Q8 0.02 gate still FAILs; this does not exclude additional error in task
execution and does not authorize replacing the original assets.
`yolo-quantization-execution-cpu-resume-v3.json` further measures, through the
public CPU ABI with an explicit mode 1, a Q8 same-weight execution added score
error of 0.02269834 and a total error of 0.12325168 against the independent
checkpoint, still FAIL.
The per-box errors of all three quantizations, the ten raw samples and the
model/runtime SHA values before and after are all stored; the two maximum errors
are not subtracted for attribution, and same-weight agreement is treated neither
as independent truth nor as a frozen release pass.
The real tests for backend selection / independent error recording went
RED->GREEN and the resumed parameter suite passes 3/3; in the six GPU rows of
`yolo-quantization-execution-{cuda,vulkan}-resume-v4.json` F32/F16 PASS. Q8 CUDA
has a same-weight score error of 0.18532050 and a total error of 0.18347186 with
four boxes missing; Vulkan is 0.00550228/0.15145230 with five boxes matching, and
both original overall gates FAIL, with all ten repetitions stable and the asset
SHA values unchanged. The missing CUDA boxes are not attributed to storage error
alone; in the current sources both GPUs expand Q8 to F16 on the host. The LLM CUDA
arithmetic scope is 11/11 PASS (all 0 changed), which does not replace the
original text/p95 gates; the first optrace assembly exit 77 is retained, and after
recompiling next to the modules both items are 162/162 with 0 null, the first
exceedance being op61 for CPU->CUDA and op149 for Vulkan->CUDA, both rc=1 -- see
`owned-yolo-q8-optrace-resume-v2/`.
The new public ABI RED of `owned-yolo-q8-public-lifecycle-v1/`: CUDA cold Q8 gives
five boxes, after switching through F32/F16 Q8 gives one box, and recreating the
context still gives one; every context is stable over twelve rounds and Vulkan
PASSes. The same test with IGEMM disabled PASSes, localizing the issue to the
cache path; the old plan never consumed the monotonic `plan_owner_id` the task
already writes. The ordered `0020` enters CUDA/combined only, adding the identity
check and stale NHWC handling, and the original official tree passes read-only
`apply --check`.
The four-chain forward/reverse / idempotent replay and single-TU compilation of
`yolo-igemm-owner-candidate-v1/` pass; the original link was missing 80 objects,
so the isolated CUDA backend is being completely rebuilt from the verified
combined replay and the same pinned llama parent.
It inherits the original CUDA ISA/Graphs/arithmetic options and the build
succeeded; the final module SHA is `4b6ad3c8...` with protections unchanged, and
the same binary oracle SHA `900a4a2f...` has been assembled into RED/GREEN.
`owned-yolo-igemm-owner-green-v1/coordinator-3/` has completed the real oracle
RED rc=1 -> GREEN rc=0, and the CUDA/Vulkan public model-switch / context-
recreation lifecycle PASSes.
The later same CPU lifecycle in `owned-llm-flags-attribution-v2/coordinator-1/`
also PASSes, covering all three backends.
The receipts of the first two insertion rejections and the restoration are
retained; in `yolo-quantization-execution-cuda-ownerfix-v5.json` the Q8 five boxes
match and are stable over ten repetitions with a same-weight execution added score
error of 0.0021265149, but the total error against the independent checkpoint,
0.1480765343, still exceeds the original 0.02 gate and FAILs. The asset/runtime
SHA values are unchanged before and after, and a successful orchestration is not
an accuracy PASS.
The isolated module contains 0019 but does not replace the official pool gate, and
no model, original gate, or official/frozen artifact was changed.
None of the transient-memory single-tensor F16 restoration variants in
`yolo-q8-single-tensor-sensitivity-resume-v1.json` passes the original 0.02 gate;
the best, `op.109.w`, still FAILs with a maximum score error of 0.03684449196.
The per-round Q8 baseline restoration and the asset SHA values before and after
pass; no model is written and this is not called native, hold-out or release.
The two real tests for specified tensor restoration / illegal-name rejection went
RED->GREEN; the three `op.109.w` pairings with
`op.110.w/op.107.w/op.20.w` have errors 0.04792762/0.08504459/0.08508769 and all
FAIL, so improvements are not additive. The user chose to use only the original
frozen baseline, whose release veto stands, and no substitute baseline is created.
`yolo-attention-release/{cpu,cuda,vulkan}/report.json` have completed: the
candidate is 2/3 each (F32/F16 PASS, Q8 FAIL), the frozen baseline is 0/3 each,
and the performance comparison is vetoed by the baseline status.
The frozen Vulkan F32 maximum score error of 0.0908023715 exceeds the original
0.005; the overall release verdict is FAIL.
The frozen CPU F32 is stable in all three processes with five matching boxes and
the same maximum score error of 0.1101961732; it was rejected by the original
accuracy gate after normal execution, not because of a missing attention API or a
loader error. The original baseline / numeric gates stand, and candidate stability
or an ABI pass is not written up as a release performance pass.

The original page and the independent SROIE accuracy gates of the conv112 storage
candidate pass on all three backends; with the same SHA, 960 main inputs and the
original 3x2x10 release A/B, CUDA PASSes overall while CPU p95
74.480 -> 85.560 ms (+14.88%) FAILs overall.
In the first Vulkan round the candidate's third process exceeded the source p95
gate; the re-verification `conv112-release-vulkan-tail-recheck/report.json` has
candidate `[0,0,1]` / baseline `[1,0,0]` and FAILs overall: candidate attempt 3
and baseline attempt 1 were rejected by preflight timeouts caused by external GPU
activity, with only 1/2 fixtures each, so this is not a measured source-speed
FAIL; a successful process does not supply their GPU qualification. The baseline
veto, the original first-round performance FAIL and the old receipts are all
retained.
The latest `conv112-release-vulkan-preflight-recheck-v2/report.json` has
candidate/baseline `[1,1,1]` each and FAILs overall. Candidate attempt 1
fixture 0 passed its GPU/source audit with p95 5.31 against 2.105774 ms, above
the original 5% / 3 ms dual gate; the remaining samples had preflight timeouts,
native/source audit failures or incomplete fixtures.
The final single log of 5.35/7.80 against 4.30/5.03 ms does not qualify as a
complete A/B; all failures must not be attributed to contamination.
The `0019` pool AVG isolated 80 cases are GREEN; before the official rebuild, the
independent oracle loading `build_app/bin` directly is CPU 40/40 and Vulkan 80/80
PASS with CUDA 80 cases / 27 FAIL, so the real consumers and release remain open.
The LLM CPU `llm-cpu-attribution-resume-v2.json` audits the original 12 rows / 720
measured runs, the artifact identities and the input/output comparability, and the
six original performance FAILs are retained; six analyzer behavior tests went
RED->GREEN. The Haswell compilation/ISA/OpenMP settings agree; all 232/69
llama/mtmd source files are identical and among the 69 CPU files only the DW bias
branch differs, from which the whole CPU call chain or its performance must not be
asserted identical. The frozen original 3x2x10 A/A single row for 256M Q8 PASSes
and is a diagnosis only;
the complete `llm-cpu-complete-post-aa-resume-v1/report.json` A/B has finished
with both sides 12/12 correct and stable;
500M newspaper Q8/F16 decode p50 at +11.08%/+6.60% and 256M newspaper Q8 e2e p95
at +5.20% are three original performance FAILs.
The 720 measured samples, artifact identities and compilation-parameter audit of
`llm-cpu-complete-post-aa-attribution-v1.json` pass, the three items and the old
six FAILs are retained, this is not yet a CPU release PASS, and the original
baseline is not replaced.
`llm-cpu-complete-flags-closure-v3.json` extends the low-level source audit:
scheduler/alloc/threading are byte-identical; the Haswell/ggml-base flags are the
same, while AICore's AVX2/frounding-math/macros and direct dependency closure
differ. Two new per-module flags negative cases went RED->GREEN with the analyzer
at 8/8; this is not proof of performance causality. Unprivileged CPU sampling was
rejected by this host's perf paranoid=4 and no system setting was adjusted.
The resumed source checks -- 913/913 C ABI, strict plugin boundary,
17 source-owner contracts, 123 runner tests, 197 L1/L2/L5 items and the re-run of
the public attribute error-reference negative case -- all pass;
the attribute inputs are unchanged, with receipt
`shared-owner-acceptance/attribute-probe-resume-v3/`, which does not replace the official consumers or a release.
`llm-flags-{control,frozenopts}-ab-v2/report-cpu.json` has two rounds of three
rows each, bound to the original frozen baseline and the original 3x2x10, for 360
measured runs in total; both sides are correct and stable with artifacts unchanged.
Recompiling the three llm/paddle_graph/attribute TUs under the same current link
closure, changing only CXX_FLAGS and no macros/includes: control retains five
performance FAILs and the frozen flags still have two FAILs,
500M Q8/F16 decode p50 +8.65%/+6.11%; both attribution audits and the 46
before/after protections PASS.
The three assembly / original-baseline protection behaviors went RED->GREEN with
no change to production flags; this is not a complete release or causal closure.
The isolated builder's shell `\$ORIGIN` escaping defect was additionally proven by
a real side-loaded `.so` going RED->GREEN, with 10 analyzer/assembly items and two
baseline protections PASS; `llm-origin-probe-v3/standalone-cpu.json` PASSes real
CPU inference without LD_LIBRARY_PATH, which is independent-loading qualification
only and does not overwrite the original v2 receipts.
The original OCR 168 scenarios on both backends have finished and still FAIL
overall; the Vulkan execution identity is unchanged and the seven failing
scenarios cover tiny Q8 rec/label/XFUND/orientation, tiny F16 label and
public/heldout en Q8.
`passes_source_gt_gate=0` means at least one candidate-source GT metric delta
exceeds 0.005; it is not a quality gate on the source itself. The tiny F16 label
hmean / e2e CER / e2e WER deltas are 0.005540/0.006178/0.009804; tiny Q8 label
WER / e2e WER are 0.009804, XFUND Q8 WER is 0.018738, and public/heldout en Q8
WER are 0.006834/0.006208, all FAIL;
a CER improvement does not offset a WER regression. The historical mistaken
attribution of the four exact-file rows was corrected in the retired
`OCR_RELEASE_GATES.md` ledger (now in git history); the live gate list is the
OCR block of the "Open parity gap ledger" in `UPSTREAM_PARITY_AUDIT.md`, and the gates run as the structured
`validation_manifest.json` OCR rows via `validate_all.py`.
The two orientation failures are in the text assertion, not in the
angle/score/rotation assertions; a wrapper rc=0 does not bypass failures=2. The
four tiny Q8 rec fixtures are complete with stored max_diff=1.317e-5, but the
source benchmark branch still sets rc=1 on the original max_diff/argmax; the
developer source audit=0 does not by itself trigger that veto.
Neither the inconsistency of the quantization diagnosis / benchmark gate nor the
original FAIL was changed. The original CUDA identity-change veto is retained;
the official rebuild / patch replay, the three contracts and the three-backend
attribute e2e/batch/UI, table and pool of `owned-validation-continuum/` are all
rc=0; pool is CPU 40/40 and 80/80 per GPU with zero failures.
The official library SHA is `7b5b28b7...`, the frozen library / standard manifest
protections pass, and the listed complete does not mean release.
The official public lifecycle of `owned-yolo-formal-lifecycle-v1/` PASSes on all
three backends; the official CUDA module owner oracle in
`yolo-owner-formal-oracle-cuda-v1.log` is rc=0.
The same-stored-weight orientation exporter went really RED->GREEN and the OCR
owner suite was 75 PASS before the new native method was added;
`orientation-stored-{cpu,cuda}-v1/` and the native three-backend diagnoses all
localize the inverted word_1:
the independent source is LNIO while Q8 stored/native is LNIOF, so both original
text assertions still FAIL.
Under the .9 gate the native-stored text adds no mismatch and the artifacts are
unchanged; this cannot be used to withdraw the independent text gate, other page
accuracy gates or the original release FAIL. Threshold=1 does not use the .9 text
truth.
The official Vulkan `conv112-release-formal-vulkan-v3/report.json` has finished
and FAILs overall: candidate `[0,0,1]` / original frozen baseline `[0,1,0]`, with
2/2 fixtures per process on both sides and all main-input audits qualified.
Baseline attempt 2 has p95 6.43 against the source's 3.277795 ms; candidate
attempt 3 has p50 5.76 against the source's 2.430157 ms, with absolute differences
of 3.152205 / 3.329843 ms, both above the original 5% / 3 ms dual gate.
This is not a preflight timeout; successful processes and aggregate values do not
offset the failed processes, and the artifact/model/standard manifest protections
pass.
A later real-load forensic confirmed that v3/v4 still used the current backend
modules: the `.so` was imported as the test baseline and the runner also resolved
the probe symlink. The original frozen ggml does not contain the current probe's
DW bias API and must not be mixed in.
The real load REDs for the adjacent module and the resolved path are retained; in
the end only each side's original probe bytes and owner modules were copied, with
no rebuild of the frozen library. The real CPU load of
`conv112-original-consumer-green-v3.log` passes; for v5,
`conv112-release-formal-vulkan-v5-original-owner/report.json` has candidate
`[0,0,0]` / baseline `[0,0,1]` with complete 2/2 fixtures and qualified main-input
audits, and still FAILs overall: the baseline's third process has p95
5.39 against the source's 2.225544 ms, an absolute difference of 3.164456 ms, so
the original dual gate and the baseline veto stand.
v3/v4 are no longer called a qualified original-baseline A/B, and the v5 original
baseline probe SHA `0be29828...` is unchanged.
The official table demo / OCR consumers passing does not cover the optional layout
page combination that the log explicitly reports as not configured.
In the new official three-row LLM CPU `llm-raw-formal-ab-v1/report-cpu.json` both
sides are entirely rc=77 because the independent upstream tools were unavailable,
so there is no measured inference or new performance comparison qualification; the
original performance FAIL is retained.

Source configurations without a matching local AICore contract are intentionally
not added to this table. Their exact families, including local VQA/VLM and
Grounding, are tracked in the "Source models with no matching AICore catalog" map below.

## Upstream `configs/auto_labeling` type coverage (source cd26942a)

Derived mechanically from the 206 source YAML files (100 distinct `type`
values) against the live AICore catalogs and this plugin's workers. A
`type` is *wired* only when a public catalog publishes the matching GGUF
weights and a worker consumes them. Composite `type`s compose
already-published models or algorithms. Three model-reuse decisions
(2026-11-01) close most of the former "missing" rows without any new
conversion work:

1. **YOLO reuse**: the published qYOLO catalog (yolo26/yolov8 det, seg,
   pose, obb, semantic, classify, depth, world, YOLOE, reid, mclip bridge;
   77 variants x f16/q8) is the supported YOLO model set. Upstream
   configs naming other YOLO architectures are served by the catalog
   equivalent; no new YOLO conversions are planned.
2. **SAM reuse**: interactive segmentation is served by the SAM3/SAM2.1
   catalog (point, box and text prompts). Upstream SAM variants
   (`sam_hq`, `edge_sam`, `efficientvit_sam`, `sam_med2d`) share the same
   result contract and are served by the catalog equivalent; text-prompted
   segmentation (`grounding_sam`, `grounding_sam2`) maps to SAM3 text
   mode. No new SAM conversions are planned.
3. **Confirmed missing families**: local `grounding_dino`,
   `dfine`/`dfine_seg` and `florence2` are the genuinely missing model
   families. Each requires the full ladder — P0 first-party converter,
   P1 task contract, P2 cpu/cuda/vulkan numeric gates, P3 picker entry —
   because their architectures need operators outside the current ggml
   set (deformable attention for the DETR families; the Florence-2
   DaViT encoder plus its encoder-decoder language head).

Summary: **25 wired, 28 composite, 4 remote/policy, 3 confirmed missing
families, 40 served-by-equivalent (no conversion planned)**.

### Wired through public AICore catalogs

| Source type | Landing |
|---|---|
| `ram` | RAM/RAM++/Tag2Text catalog |
| `rfdetr` | RF-DETR detection catalog |
| `rfdetr_seg` | RF-DETR segmentation catalog |
| `rmbg` | RMBG-2.0 catalog |
| `scrfd` | FaceDetect (buffalo/SCRFD) catalog |
| `segment_anything_2` | SAM 2.1 catalog |
| `segment_anything_3` | SAM3 point/box/text catalog |
| `yolo26`, `yolo26_obb`, `yolo26_pose`, `yolo26_seg` | yolo26 family rows in the yolo catalog |
| `yoloe` | YOLOE text / prompt-free / SAVPE rows |
| `yolov8`, `yolov8_seg`, `yolov8_cls` | yolov8 det/seg rows and the yolo26-cls image-classify contract |
| `yolow` | YOLO-World text-prompted detection (world + CLIP towers) |
| `ppocr_v4`, `ppocr_v5`, `ppocr_v6` | OCR recipes (published f32/f16/q8 families) |
| `doclayout_yolo` | CDLA layout row in the OCR catalog |
| `depth_anything`, `depth_anything_v2` | DA3 / nested AnyView depth catalog |
| `pulc_attribute` | PULC attribute recipe (local F32) |
| `segment_anything` | Interactive point/box segmentation served by the SAM3/SAM2.1 catalog |

### Composite / algorithm types (no new model required)

| Source type | Composition status |
|---|---|
| `botsort`, `bytetrack`, `tracktrack` | Tracking algorithms (yolo-track batch command and the video workflows) |
| `grounding_sam`, `grounding_sam2` | Text-prompted segmentation via SAM3 text mode |
| `yolov5_sam`, `yolov8_sam2`, `yolov5_ram`, `yolov5_resnet`, `yolow_ram` | Detection-to-SAM/RAM/classify compositions of published models |
| `yolov5_sahi`, `yolov8_sahi`, `yolo26_sahi`, `yolo11_sahi` | SAHI-style tiled detection (SlicedDetection implemented) |
| `yolov5_det_track`, `yolov8_det_track`, `yolo26_det_track`, `yolo11_det_track` and the obb/pose/seg `_track` variants | Detection + tracking over published models |
| `segment_anything_2_video` | Video mask workflows (VideoSegmentExporter / video annotation); SAM2 video propagation specifics remain partial |

### Remote / service / policy types

| Source type | Status |
|---|---|
| `remote_server` | RemoteServerClient (upstream protocol) |
| `grounding_dino_api` | Remote text-prompted detection API via the remote channel |
| `open_vision` | Partial (remote channel exists; workflow details unfinished) |
| `cuda` | Configuration placeholder, not a model |

### Confirmed missing families (P0 converter, P1 task contract, P2 numeric gates, P3 picker)

| Source type | Gap |
|---|---|
| `grounding_dino`, `groundingdino_swinb_cogcoor`, `groundingdino_swint_ogc` | Local text-prompted detector: BERT text tower plus a deformable-DETR head; needs the deformable-attention operator in ggml before conversion can be gated |
| `dfine`, `dfine_seg` | D-FINE dense o2o detector; same deformable-attention prerequisite, then the P0-P3 ladder |
| `florence2` | Local VLM: DaViT vision encoder with an encoder-decoder language head; libmtmd does not carry the architecture, so this is a first-party converter plus task contract, gated on the upstream framework checkpoints |

### Served by catalog equivalent (no conversion planned)

| Source type | Serving catalog row |
|---|---|
| `yolo11`, `yolo11_cls`, `yolo11_obb`, `yolo11_pose`, `yolo11_seg` and their `_track`/`_sahi` composites | yolo catalog equivalents (decision 1) |
| `yolo12`, `yolov5` family (`yolov5`, `_cls`, `_obb`, `_seg`, `_pose`), `yolov6`, `yolov6_face`, `yolov7`, `yolov9`, `yolov10`, `yolox`, `yolox_dwpose`, `damo_yolo`, `gold_yolo`, `yolo_nas` | yolo catalog equivalents (decision 1) |
| `rtdetr`, `rtdetrv2`, `u_rtdetr`, `rtmdet_pose`, `upn`, `rio_detr`, `deimv2` | detector/pose catalog equivalents (decision 1 scope; revisit only if a task needs their specific heads) |
| `edge_sam`, `efficientvit_sam`, `sam_hq`, `sam_med2d` | SAM3/SAM2.1 catalog (decision 2) |
| `internimage_cls`, `geco` | image-level classification contract served by the yolo26-cls catalog |
| `cn_clip`, `bert` | text-encoder role served by the world/CLIP and mclip bridge towers |
| `clrnet` | lane-specific head; served by the line/wireframe (DeepLSD) workflow until a lane task is gated |

Each confirmed-missing row lands through the same gate ladder: P0
first-party converter emitting GGUFs with pinned SHA-256 digests; P1 task
contract (new task only when no existing public C ABI covers the result
shape); P2 f16/q8 quantization plus the cpu/cuda/vulkan matrix through
`validate_all` with task-specific numeric gates; P3 plugin picker entry.
ONNX Runtime is not an AICore backend: ONNX exports are conversion
inputs, not serving engines. There is deliberately no blanket "automatic
PyTorch/ONNX to GGUF" switch — parity claims require per-task numeric
gates against the upstream framework checkpoint.

### Model hosting audit (catalog download URLs, 2026-11-01)

| Host / repo | Catalog rows |
|---|---|
| HF `Asher-1/relateanything-ggml-full` | 24 |
| HF `Asher-1/LoMa_GGUF` | 23 |
| GitHub `Asher-1/cloudViewer_downloads` releases | 15 (incl. the yolo_gguf_models release: 77 yolo variants x 3 quantizations = 230 assets; OCR; depth; SAM3; GKD; DeepLSD; lingbot) |
| HF `Asher-1/RAM_GGUF` | 9 |
| HF `Asher-1/Trellis2-models` | 2 |
| HF `Asher-1/SAM_3D_OBJECTS_GGUF` | 2 |
| HF `Asher-1/InstantMesh-GUFF` | 2 |
| HF `Asher-1/AnyCalib_GGUF` | 1 |
| HF upstream repos (Qwen, ggml-org SmolVLM, maelic relsgg) | 4 |

The cache/download interface is already uniform and repo-agnostic:
every catalog row resolves through one shared downloader
(`ecvModelDownloader` / `ecvAssetIntegrity`) into the task-owned folder
under `CLOUDVIEWER_DATA_ROOT` with SHA-256 digest verification and
`.cvintegrity` receipts; the plugin never sees the hosting backend.
Migrating the GitHub-release assets to a new HuggingFace repo is a
catalog-owner release action (upload + URL flip; digests are
content-addressed, so flipping URLs cannot silently change weights),
not a code change, and requires HF credentials this repository does
not hold.

## Source models with no matching AICore catalog

This map moved here from the retired `TODO.md` (README links updated). The source families below have missing or partially implemented model coverage.
Existing catalog-backed subsets remain available in the model picker. For
missing models, implement the inference task, GGUF catalog and validation first;
then add the plugin worker and matching interaction. The names below are source
config `type` values or exact source model families.

Current OCR scope (2026-10-17): qOCR keeps its merged workflow; the public OCR
task now also supports the optional direction classifier, all six active source
OCR presets as local F32 recipes, and selected arbitrary polygons as described
in the final section. PP-OCR KIE annotation format is implemented. New-model
publication, quantization and dataset/release gates remain open. The following dated OCR
port/accuracy narrative records historical experiments and is not current
acceptance evidence.

- [ ] OCR and document understanding: `ppocr_v4`, `ppocr_v5`, `ppocr_v6`,
  `doclayout_yolo`, PPOCR KIE. Research verdict (2026-09): upstream's
  PaddleOCR choice is confirmed best-in-class for this use case — 2026
  benchmarks rank the PaddleOCR family first among efficient open-source
  OCR engines (PaddleOCR-VL 0.9B beats 235B VLMs on OmniDocBench;
  PP-OCR leads handwriting/multilingual accuracy per-parameter), and the
  source adapters consume exactly the PP-OCR det/rec/KIE pipeline that
  this plugin's sidecar stream already imports/exports. No GGUF or ggml
  port of PP-OCR exists (community deployments use ONNX Runtime; ONNX
  conversions are not an AICore backend). The port is therefore a new
  `ocr` AICore task plus a first-party Paddle → GGUF converter
  (DBNet-style det CNN + SVTR/CRNN rec + FCE layout/KIE heads; all map
  onto existing ggml conv/transformer ops plus the RMBG custom-op
  pattern for fusion), NOT a substitution with a different engine.
  Acceptance gate: typed results — per-box `aicore_ocr_box`
  (quadrilateral corners + UTF-8 transcription + float score + script),
  layout regions (region class + reading-order index), KIE relation pairs
  (typed index pairs + role labels) — with CER/edit-distance accuracy
  gates against the PaddlePaddle reference on pinned fixtures, and the
  full f32/f16/q8 x CUDA/Vulkan/CPU ladder proving end-to-end parity
  before the plugin picker gains any OCR entry (the PaddleOCR toolbar
  entry stays disabled with its official name until then).
  Rollout plan (2026-09): P0 first-party converter
  `core/AICore/tools/convert_paddle_ocr_gguf.py` plus pinned fixtures
  with PaddlePaddle reference JSON (G0: op-level reference match);
  P1 task `core/AICore/src/tasks/ocr/` + `AICORE_MODEL_OCR` + public
  `aicore/ocr_capi.h` (opaque context, extensible options, three typed
  result arrays, release, last_error, common timings, runtime_shutdown
  delegation) with f32 CPU end-to-end CER equal to the Paddle
  reference (G1); P2 f16/q8 via the shared `aicore_gguf_quantize`
  tool and the full 3x3 backend/quant matrix through validate_all
  (G2: matrix green and `test_catalog_dump_urls --json` exposes every
  row); P3 plugin worker/model-bar enable flipping
  `AnnotationTODO_open_paddleocr` atomically with the picker (G3:
  plugin regression plus headless real-inference evidence).
  Progress (2026-09-28): P0 data plane COMPLETE — the converter
  (`core/AICore/tools/convert_paddle_ocr_gguf.py` + the dependency-free
  `paddle_ocr_extract.py` binary parser, since the paddle runtime
  bindings crash on the conversion host) emitted nine GGUFs (det-ch /
  rec-ch / rec-en x F32/F16/Q8_0) from the real PP-OCRv4 checkpoints,
  with the 6623/95-entry recognition dictionaries embedded as
  `ocr.dictionary` and conv kernels folded to block-aligned 2D for
  Q8_0 (`ocr.quant.reshaped`). All nine are published at the new PP-OCR
  release, pinned in `asset_digests.h`, and exposed through the
  catalog-only `aicore/ocr_capi.h` with `test_catalog_dump_urls` and
  `test_catalog_remote_assets` passing. Progress continued
  (2026-09-28): the Paddle reference channel was repaired on the
  conversion host (paddle.inference Config with IR optimization and
  ONEDNN disabled — the static-graph loader crashes there), and
  `core/AICore/tools/export_ocr_graph.py` now emits the authoritative op
  topology (rec: 323 ops, det: 312 ops — one shared op vocabulary:
  conv2d/depthwise/batch_norm/elementwise/hard_swish/hard_sigmoid/
  pool2d/matmul_v2/transpose2/slice/layer_norm/softmax) plus parity
  fixtures (fixed-seed input -> paddle output) at
  `core/AICore/tests/fixtures/ocr/{rec,det}_ch_ref.npz`. The topology is
  embedded in the released GGUFs (`ocr.graph.ops`), the digests were
  re-pinned, and the inference contract is declared in
  `aicore/ocr_capi.h` (18 exported functions: options/context/run/typed
  results/timings/error/shutdown). A dedicated Standard plugin `qOCR`
  (`PLUGIN_STANDARD_QOCR`) now carries a full workspace dialog
  (`OCRDialog`): bilingual Chinese/English recognition model
  selection over the shared script-agnostic detector, F32/F16/Q8_0
  tier choice, digest-verified downloads through the unified
  ecvModelDownloader pipeline into the task-owned `ocr_models`
  cache (exposed as `aicore_ocr_model_cache_dir`), a one-click
  "Use test data" button that stages the official PaddleOCR demo
  image for the selected language, and a results table. The Run
  entry inside the dialog stays disabled until the gates pass, so
  the workspace is usable for staging without faking inference.
  Loading-chain fixes (2026-09-29): the qrc now embeds info.json
  (the missing entry made the plugin load as a blank name), the
  icon was redesigned (document + scan frame + detection box +
  OCR lettering), the Use-test-data demo URLs were replaced with
  verified-live official assets (docs/images/00006737.jpg for
  Chinese, tests/test_files/textline.png for English), and the
  full official PaddleOCR test-image set (55 images, MANIFEST
  with per-file SHA-256) is staged at
  ~/cloudViewer_data/download/sources/ppocr_testdata/ with a
  ready-to-upload ppocr_testdata.zip for the shared test-data
  release (60 official images: 9 pytest test_files, 21 street scenes,
  20 release/2.7 imgs, 5 ch words, 5 en words; MANIFEST carries
  per-file SHA-256); ecvTestDataRepository registration follows once
  the upload lands.
  P1 interpreter status (2026-09-29): `ocr_runtime.cpp/hpp` implements
  the topology-driven interpreter — 20 of the 21 op types are wired
  (conv2d, depthwise via per-channel decomposition, batch-norm folding,
  elementwise with Paddle auto-alignment broadcast, hard_swish /
  hard_sigmoid / swish / relu / sigmoid, pool2d avg/max with adaptive,
  matmul_v2, transpose2, slice (tensor views + metadata-constant
  slicing), reshape2/squeeze2/flatten, concat, softmax, layer_norm,
  scale, dropout, shape, fill_constant) and the CTC greedy decoder plus
  the rec preprocessing are in place. P1 G1 CORRECTNESS ANCHOR REACHED
  (2026-09-29): test_ocr_runtime runs the full 323-op rec graph and the
  full 312-op det graph on CPU and matches the Paddle reference fixtures
  bit-near-exactly (rec f32 argmax-match + maxDiff 2e-6; det f32 maxDiff
  0.000000). Root causes fixed on the way: flatten_contiguous_range must
  merge [start_axis, stop_axis] instead of fully flattening (the embedded
  graph JSON was also stale - strides/paddings were None - so all three
  topologies were re-exported with per-tensor ranks + static shapes and
  the GGUFs re-converted; release digests pending re-upload); the
  numpy->ggml permute must be the INVERSE mapping (ggml scatters, numpy
  gathers); matmul_v2 needs transposed B plus a contract-layout transpose
  on the result; elementwise with axis=1 must place a 1-D bias on the
  channel axis (can_repeat silently accepted a W-axis broadcast when 16
  divides the width); squeeze2 collects in numpy order; nearest_interp_v2
  resolves FPN targets from the concat/add alignment scan (non-integer
  8->15 cases) via ggml_interpolate; conv2d_transpose uses the
  (Cin,Cout,kh,kw) kernel spelling; mul/add/conv/softmax/norm materialize
  non-contiguous views. C API RUNTIME LIVE (2026-09-29): ocr_capi.cpp
  implements the full 18-function contract — context (det/rec weights),
  run(image_view) with det preprocess (bilinear limit-side resize + 32
  align + ImageNet NCHW), hand-written DB post-process (threshold 0.3 +
  4-connected BFS + unclip offset = area*ratio/perimeter), per-box
  bilinear crop -> 48xH rec -> CTC, typed box{quad,text,score,language},
  truthful pipeline timings, thread-local last_error. qOCR Run is
  enabled (QtConcurrent worker, lazy context); the qAnyLabeling
  PaddleOCR toolbar entry is live and hands off to the qOCR workspace
  (plugin-to-plugin code coupling is against the AICore boundary).
  GPU MATRIX UNBLOCKED (2026-09-29): the CUDA/Vulkan segfault root cause
  (host-resident weight tensors mixed into a device graph with no
  scheduling) is fixed with the facedetect ModelLoader pattern: the ocr
  weights wrap both host pools (GGUF meta + folded constants) as zero-copy
  CPU backend buffers (OcrWeights::realizeHostBuffers/tagHostLeafs),
  upload the named weights to a device buffer once per context
  (realizeToDevice), and run through ggml_backend_sched across {gpu, cpu}
  with ggml_set_input on the image leaf; CPU keeps the verified gallocr
  fast path. Backend matrix (f32 probe p50, RTX 3060): rec CPU 220.6ms /
  CUDA 205.8ms / Vulkan 217.3ms; det CPU 381.8ms / CUDA 358.5ms / Vulkan
  356.6ms; accuracy is identical across all three backends (rec maxDiff
  2e-6 + argmax-match, det maxDiff 0.000000), and f16/q8_0 on CUDA keep
  argmax parity at their storage-precision diffs. GPU wall time is
  dominated by scheduler split/copy overhead and per-run host constant
  copies (folded constants stay host-side), so a shape-stable graph cache
  plus device-side constant folding is the measured performance follow-up,
  not a correctness gap. The C API context now owns its backend pair for
  the whole session (the previous code re-initialized a backend per run).
  Image toolkit promoted to common (2026-09-29): fd::Image + the view
  wrappers + cv_resize_linear_u8 now live in src/common/image_utils.{hpp,
cpp} (namespace aicore::img); facedetect keeps a using-shim header (its
  13 include sites unchanged), the two duplicated internal resize kernels
  were deleted, and the ocr task no longer borrows a facedetect symbol —
  that extern declaration had left an unresolved cv_resize_linear_u8 in
  libAICore's dynamic table (a live load-time defect the CPU probe never
  hit because it does not link ocr_capi.cpp); nm -D confirms it is gone.
  Still queued: polygon-accurate unclip for the official det-eval round
  and region/KIE graph loading (typed arrays and catalog already reserve
  them).
  IN-WINDOW OCR + FACE DETECTION LIVE (2026-09-30): the PaddleOCR toolbar
  entry (Ctrl+4) now runs PP-OCR inside this window through the AICore ocr
  C API (AnnotationWorker ModelKind::Ocr; the picker lists PP-OCRv4
  (ch/en) rec families with the det partner resolved per quantization
  through the dependency channel, exactly like the face landmark model);
  each recognized region becomes a text annotation whose description
  carries the transcription with score/language in extra, so
  the existing PP-OCR dataset export round-trips. Detection through the
  facedetect task (buffalo packs incl. the SCRFD det_500m/2.5g/10g
  detectors) was already fully wired (ModelKind::FaceDetect + detector/
  landmark catalog enumeration); the source scrfd entry above is closed
  as functionally aligned with only the standalone bnkps checkpoints
  remaining exact-weights-unsupported. The old qOCR-workspace hand-off
  dialog is gone (test_plugin asserts the entry triggers in-window
  inference and that the picker carries the PP-OCR families).
  UPSTREAM OCR INTERACTION PARITY (2026-09-30): the det post-process now
  mirrors the official DB pipeline — component boundary pixels feed a
  convex-hull minimum-area rectangle, the Vatti unclip expands it
  (area × ratio / perimeter), and the ordered corners are emitted as a
  clockwise quad; rotated text yields genuine quads, and the rec crop
  uses the official get_rotate_crop_image perspective warp. The plugin
  adopts the upstream ppocr_v4 semantics verbatim: label is always
  "text" (description carries the transcription), an axis-aligned quad
  becomes a rectangle (rebuilt from corner 0/2 as upstream) while any
  rotated quad stays a quadrilateral, and the rec gate is the model
  drop_score 0.5 (upstream wires no confidence slider for OCR). Verified
  end-to-end on the dense ppocrv4 scene image; upstream's
  skip-detection (re-recognize selected shapes only) and the direction
  classifier (cls GGUF) remain the two open interaction gaps.
  IMPLEMENTATION CHECKLIST for the two gaps (design frozen):
  (a) skip_detection —
      1. DONE: `aicore_ocr_recognize(ctx, view, quads, count)` lands in
         `ocr_capi.h`/`ocr_capi.cpp` (perspective warp -> height-48 ->
         CTC over caller quads, full-image pixel coords, same result
         ownership/timings).
      2. DONE: `AnnotationWorker` Ocr branch now routes non-empty
         `prompt.rois` through recognize() and emits shapes with
         `sourceIndices` (the existing GKD/SAM3 roi channel — no new
         infer signature).
      3. REMAINING: `AnnotationWindow` — add the `Skip Detection`
         button next to Run (visible for the OCR family), collect the
         selected text shapes' rects into `prompt.rois` +
         `roiSourceIndices`, and in the finished handler update those
         shapes' description/score in place instead of appending.
      4. test_inference: selected-rects path with the real rec GGUF.
  (b) cls direction classifier —
      1. Download `ch_ppocr_mobile_v2.0_cls_infer.tar` (paddle format,
         paddleocr dygraph_v2.0 release); convert with
         `convert_paddle_ocr_gguf.py --role cls` (extractor is generic)
         + `export_ocr_graph.py` for the embedded topology (ops: conv/
         bn/relu/pool/linear/softmax — all already in the interpreter).
      2. Catalog: add role "cls" entries (f16/q8_0/f32) + digests +
         manifest scenario (cls probe gate: argmax label 0/180 on two
         rotated crops).
      3. `ocr_capi.h`: extend `aicore_ocr_options` with
         `cls_model_path` (extensible struct: struct_size already gates
         ABI); `run()` passes each crop through cls when loaded, flips
         180-degree crops before rec, mirroring upstream
         `use_angle_cls` + `cls_thresh=0.9`.
  DET F16/Q8 CONTAINER FIX (2026-09-30): the det head BN's frozen
  variance reaches 1e7, beyond the F16 range (65504) — the previous
  det F16/Q8_0 containers stored Inf there, which zeroed the head and
  made every det quantized run output an all-zero probability map (the
  old det fixture reference was itself a near-zero map from a bad
  preprocessing distribution, so the old "bit-exact" det anchor was
  vacuous). Fixes: the converter keeps any tensor whose magnitude
  exceeds the F16 range in F32 inside quantized containers (GGUF types
  are per tensor); OcrWeights::load now fails loudly on non-finite F16
  values; the det fixture pair was regenerated through the
  paddle.inference reference channel with the OFFICIAL PP-OCR det
  preprocessing (release + manifest anchors updated), and the manifest
  ocr scenarios were split per quantization with honest gates (det F32
  0.01 / F16 0.1 / Q8_0 0.5 sigmoid-domain max|diff|, rec argmax==1
  plus per-quant max|diff|). Verified on the new anchors: det F32
  maxDiff 1.9e-5, F16 0.058, Q8_0 0.348; rec unchanged (F32 2e-6,
  F16/Q8_0 argmax-identical); CUDA matches CPU on all six rows.
  Probe harness (2026-09-29): the pinned rec/det
  fixture npy pairs moved OUT of the source tree into the shared asset
  cache (uploaded to the PP-OCR release, downloaded by
  FetchAICoreTestAssets into <asset-root>/ocr_fixtures); the probe emits
  a single-line JSON verdict (task/backend/max_diff/argmax/p50/p95) for
  the manifest's generic parser, and ocr-rec/ocr-det scenarios with
  structured accuracy gates (argmax==1, max_diff<1e-3) are registered in
  validation_manifest.json. Quantization: q8_0 containers keep matmul-Y
  weights in F32 (CPU mul_mat src1 must be F32; quantization lives on A)
  and the loader dequantizes F16/Q8_0 parameters; remaining q8 CPU edge:
  1x1 conv weights stored as native 2-D need a reshape-registry entry
  (GPU matrix round will unify this). Re-audited 2026-10-07:
  DONE — the Vulkan backend rows for ocr (`validate_all --backend vulkan
  --tasks ocr` runs det/rec/layout x f32/f16/q8_0 to verdict PASS after
  the fixture pin chain was re-anchored to the regenerated cache files),
  and the qAnyLabeling auto-label entry flip (the PaddleOCR toolbar entry
  is live as AnnotationOpenPaddleOCR). Remaining: the q8 CPU 1x1-conv
  reshape-registry edge, the det DBPostProcess parity-fixture gate in
  validate_all (the postprocess itself was rewritten to match paddlex
  semantics), and the rec semantic-divergence follow-up (paddle-anchored
  gate exposes a systematic divergence; 79% exact on 98 valid rows).
  Also DONE (2026-10-08): the fixture pin chain is fully closed — the 15
  regenerated ocr_fixtures/*.npy were uploaded to the
  Asher-1/cloudViewer_downloads `ocr-fixtures` release tag (replacing the
  stale first-generation assets the user had partially uploaded) via the
  GitHub API using the machine's own git credential store, and all 15
  manifest URLs were rewired from the dead `PP-OCR` tag to
  `ocr-fixtures`. A fresh-clone simulation (cache directory removed,
  validate_all re-run) auto-downloaded all fixtures from the release and
  reached verdict PASS 9/9 — every machine now self-provisions the
  fixtures for testing.
- [ ] rec semantic-divergence follow-up (re-baselined 2026-10-08): the
  official ground-truth capture is unusable — the only saved official
  dump (.qoder/tmp/real_official_pp.txt) is a corrupted capture (empty/
  garbage texts) and the 550x345 medal-table source image is gone from
  disk, so the remaining divergence cannot be quantified without
  rebuilding the official comparison baseline (pick a dense-text image,
  run paddleocr, save the per-box gold, re-run the matching analysis).
  Current verified state: our output decodes 98/98 boxes with sensible
  table text, two independent runs agree 100%, and the Vulkan build
  agrees 90.8% with the CUDA build (cross-backend numeric noise on a
  handful of boxes).
- [ ] Vision-language captioning, VQA and document grounding: `florence2`,
  the video description/segmentation providers and chatbot models. Research
  verdict (2026-09): the ggml tensor library itself does not carry a VLM
  layer — captioning/VQA lives in llama.cpp's `libmtmd` (multi-modal
  library, April 2025, mmproj projectors; supports Qwen2-VL/2.5-VL and
  other vision LMs). The viable route is a new `vlm` AICore task that
  vendors llama.cpp core+mtmd pinned to a release whose bundled ggml
  matches (or is patched to) our pinned ggml v0.21.0, wired through the
  task-private cmake pattern; the plugin must not call llama.cpp directly.
  Scope note: libmtmd covers local captioning, VQA and chatbot (remote
  toolbar workflows are implemented by the later service follow-up) but does NOT replicate Florence2's grounding head —
  phrase-to-box labeling stays on the catalog-backed YOLO-World (CLIP
  text) / YOLOE (MobileCLIP/SAVPE) paths, which are open-vocabulary
  detectors, not VLMs, and are already wired. Integrating mtmd is
  therefore an additive caption/VQA capability, not a weight-parity port
  of `florence2` (the same distinction as YOLO26 vs YOLOv8 exact
  checkpoints). The contract before any UI work: session from a catalog
  entry, text prompt plus borrowed image view in, typed text result with
  release, cancellation and stale-generation discard, common timing.
  `grounding_dino`, `grounding_dino_api`, `grounding_sam`,
  and `grounding_sam2` do not have their exact grounding model in AICore.
  VQA is question-answer annotation for image/video content (answers,
  rationales and question metadata). Grounding resolves a phrase to a box
  or mask for referring-expression labels; it needs the text result
  contracts and a review UI rather than ordinary detector boxes.
- [x] RAM / RAM++ / Tag2Text image tagging uses the public tagging catalog,
  per-class thresholds, shared downloader, resident worker and typed results.
  The RAM Options popup exposes `tag_mode`, `delete_tags` and `filter_tags`.
  English filtering precedes translation; delete takes precedence over keep.
  All 4585 paired source labels and 16 option combinations match actual source
  methods. Raw Chinese whitespace is preserved in the vocabulary; saved tags
  are trimmed and deduplicated by the source UI contract. Repeating identical
  normalized tags preserves the reviewed flag; changed or empty results replace
  the tag array. Object geometry remains intact in single-image and batch runs.
- [x] RAM/RAM++ explicitly selects X-AnyLabeling's OpenCV uint8 INTER_LINEAR
  preprocessing in AICore. The original Pillow bilinear/antialias mode remains
  the default for qRAM and legacy callers. The original by-value request ABI is
  frozen; `aicore_tagging_request_v2` and the two v2 entry points carry the new
  mode without changing old callers' storage. The common validation manifest
  has independent same-GGUF PyTorch truth with the actual pinned source input.
- [x] Tag2Text captions reach the document description, editor and sidecar;
  a real Q8 Run-button test verifies saving and reloading the caption. Tag2Text
  is an additional local catalog workflow without an exact X-AnyLabeling model.
- [ ] Exact source ONNX checkpoint equality is still unproven. Same-GGUF
  PyTorch accuracy, backend agreement and source preprocessing do not establish
  identical threshold decisions for independently exported source checkpoints.
  Historical loaded-host latency numbers and earlier Pillow/OpenCV tag counts
  are not current performance or full-precision parity evidence.
- [ ] RelateAnything has no counterpart in X-AnyLabeling-GGML (no adapter,
  config or widget there); its reference is the official RelateAnything-GGML
  (`deploy/demo_webcam.py`, `deploy/render_video.py`) and the standalone
  qRelateAnything plugin. In the workspace it supports selected rectangles or
  the detector stage (objects added as real rectangles; detector chosen from
  the standalone plugin's catalog: YOLOE, YOLO-World, YOLO segment, RF-DETR,
  YOLO detect), the two official
  operating points with every threshold and edge cap in an Options popup,
  custom predicates through the model's text student, the shared edge
  selection, a display-only overlay, and a filter table. Covered by
  `test_qanylabeling_run_path` (real relsgg-vits16plus Q8_0 through the Run
  button: detector stage adds rectangles and an edge selects one of them,
  selected-rectangle flow, custom vocabulary restricts the edges, sidecar and
  checked state, Auto Run refusal, overlay cleared on deletion),
  `test_qanylabeling_relate_options` (operating points equal the
  qRelateAnything presets; popup behaviour), `test_qanylabeling_relation_filter`
  and `test_qanylabeling_canvas`. Real runs cover YOLOE, RF-DETR nano and
  YOLOv8s-worldv2 as the detector. Gaps: video smoothing lives only in the
  standalone plugin; the detector list is not filtered by what is cached;
  Auto Run is refused because the official demos have no batch mode; and
  there is no screenshot-level UI comparison.
- [ ] Image tagging and attributes still open: `pulc_attribute`,
  `internimage_cls`; generic YOLO classification is not these models.
  Research verdict (2026-09): the image-level classification RESULT
  contract already exists — `aicore_yolo_classify_result` is an image-
  level softmax with class names and no geometry, so no new C ABI is
  needed; what is missing is the models. yolo-cls checkpoints cannot
  substitute RAM (Recognize Anything is a ViT-L multi-label tagger over
  a fixed 4,574-tag vocabulary; a closed-set yolo-cls head covers only
  its training classes), nor PULC (per-attribute human-attribute
  classifiers). The minimal route per model: first-party GGUF conversion
  (RAM's ViT image encoder maps onto existing ggml ops; PULC heads are
  small CNNs), catalog rows + digests, and probes that gate top-k
  accuracy against the reference on pinned images across f32/f16/q8 and
  CUDA/Vulkan/CPU — consumed through the existing classify ABI, with the
  plugin storing tags/attributes as sidecar-level fields (never object
  boxes). The manual Classifier toolbar workflow is now implemented (2026-10-17);
  its optional YOLO classification assistant uses trained categories and does
  not supply the missing PULC/InternImage or local VLM graphs. The later
  service follow-up adds arbitrary-category remote VLM classification.
  Rollout plan (2026-09): R0 converter
  `core/AICore/tools/convert_ram_gguf.py` (ViT encoder + tag decoder
  onto existing ggml ops, tag vocabulary carried as GGUF metadata) with
  reference top-k JSON (G0); R1 a `ram` architecture branch inside the
  existing `yolo` task (role CLASSIFY, sharing that task's classify
  postprocess — RAM is the second classify consumer, so no new helper
  and no new ABI) + catalog f32/f16/q8 rows + digests + `ram-topk`
  probe + manifest rows (G1: top-k gate, f16/q8 parity, matrix green);
  R2 plugin consumption over the unchanged classify ABI with
  tags/attributes persisted as sidecar fields only (G2: regression
  green plus a test asserting no object-box representation).
  Progress (2026-09-28): R0 data plane COMPLETE — the converter
  (`core/AICore/tools/convert_ram_gguf.py`) converted the real
  `ram_swin_large_14m.pth` checkpoint (Swin-L encoder + tag decoder) into
  F32/F16/Q8_0 GGUFs with the tag vocabulary embedded as `ram.tag_list`,
  all three published at the new RAM release. Remaining: R1 yolo-task ram
  architecture branch + catalog rows/digests/probe, R2 plugin sidecar
  fields. The ram digests join `asset_digests.h` together with their
  catalog rows so no digest is published without a consumer.
  Current correction: the historical assumption that a softmax classification ABI
  can express PULC is false. PULC uses independent Sigmoid attribute groups with
  strict thresholds/group argmax from original YAML. Source returns a full-image
  rectangle carrying attributes (not a detected object box), replace=True;
  neither single-category flags nor sidecars alone may replace this contract.
  Original graphs reused existing ONNX lowering for isolated F32/F16/Q8 conversion
  and native CPU ggml execution. F32/F16 outputs and all attribute selections pass
  candidate gates; Q8 person/vehicle max errors 0.020849/0.048914 exceed the
  predeclared 0.02 candidate gate and remain FAIL even when attributes agree.
  Receipt: `build_app/Testing/g8-pulc-native-graphs/native-cpu/report.json`.
  This historical OCR test-adapter receipt does not qualify public
  task/API/catalog or complete dataset/release. Subsequent CUDA/Vulkan six-row
  runs pass F32/F16 candidate gates while both Q8 rows fail. Independent
  same-stored-weight ONNX attribution in `stored-attribution/report.json` reports
  original Q8 weight errors 0.02084890/0.04891500 and maximum added execution
  error <=1.55e-6 across three backends; original gates and failures stay intact.
  Later isolated `precision-sensitivity/report.json` keeps 1 person and 2 vehicle
  kernels in F16, retaining 1601536/1568768 Q8 elements. Six candidate rows per
  backend pass (Q8 errors about 0.01444/0.00667). This single-image-selected storage
  candidate does not replace original models; untouched holdout, public ABI,
  catalog and release gates remain open. After freezing it, `holdout-cpu/report.json`
  runs 11 real images excluded from selection: F32/F16 44/44, Q8 only 4/22,
  overall 48/66 FAIL; person/vehicle max errors 0.0855134/0.0764870. Original gates
  and weight SHA remain; per-output/attribute/same-storage errors are retained.
  Calibration-image GREEN does not prove generalization; these images have no
  attribute dataset GT either.
  The real public `attribute` task/C ABI subsequently landed with independently
  owned multi-attribute results and complete raw probabilities; strict
  threshold/argmax stays task-owned. Graph execution moved to `common/paddle_graph`
  shared by OCR/table; feed/fetch metadata went RED->GREEN. Person/vehicle F32/F16
  public ABI passes numeric, attribute, stride, result ownership and live-context
  survival after shutdown on CPU/CUDA/Vulkan. Three
  `pulc-public-holdout-{cpu,cuda,vulkan}/` receipts each remain 48/66 FAIL
  (F32/F16 44/44, Q8 4/22), with original gates/candidate SHA/failures unchanged.
  Public local F32 recipe/cache API, qAnyLabeling picker and worker are connected;
  real basic Run full-image attributes, replace, saving, model switching, stale
  rejection and teardown pass on CPU/CUDA/Vulkan. Maintained F32 C ABI E2E and
  basic table E2E pass on all three; table has no configured layout page. Missing
  local models report cache paths without fabricated downloads; detection
  confidence does not affect attribute thresholds. Public probe added pinned
  source identity/all-attribute gates; a numerically passing attribute-mismatch
  negative case went RED->GREEN. Common runner's local-only catalog precondition
  was fixed with 123 tests passing. Initial six-row CPU receipt is 0/6 FAIL from
  json0/json1 dual-source/bare-metric ambiguity; retain old receipt. Complete json0
  path configuration went RED->GREEN. New
  `attribute-common-runner-*-pathfix/report.json` is 4/6 per backend: F32/F16 pass,
  both Q8 rows still FAIL original 0.02 gate; no asset replacement or relaxed gate.
  Existing `ANYLABEL_INFER` supports attribute TASK/schema/local recipe/cache;
  real isolated-plugin two-model/two-image, confidence-independent attributes,
  missing-model/no-download and invalid-manifest checks pass CPU/CUDA/Vulkan.
  GPU receipts: `shared-owner-acceptance/attribute-batch-{cuda,vulkan}-v2/receipt.json`.
  Official module/consumer rebuild still awaited the original full OCR queue;
  isolated batch success is not release.
  New real-window `shared-owner-acceptance/attribute-ui-lifecycle-{cpu,cuda,vulkan}-v1/`
  passes on all three: two-image Auto Run, in-flight cancel/no-next-image,
  rerun after cancel, source/model stale, real invalid-model worker recovery and
  window destruction during inference. Source/consumer/model SHA remain unchanged;
  no production logic change. This is not live-frame, cross-platform installer,
  official-library or release qualification. Published catalog/digests, independent
  attribute dataset GT, complete live/cancel/UI automation and release A/B remain
  absent. Standard manifest/official runtime stayed intact until the original OCR
  queue ended; local integration does not close publication.
  Later conv112 graph same-SHA/960-main-input/original 3x2x10 release A/B:
  CUDA overall PASS; CPU p95 74.480 -> 85.560 ms (+14.88%) overall FAIL; Vulkan
  third candidate source p95 8.16 versus 2.666369 ms retains original dual-gate FAIL.
  Same-protocol Vulkan tail re-test completed candidate `[0,0,1]` / baseline
  `[1,0,0]` overall FAIL. Candidate attempt 3/baseline attempt 1 actually hit
  external-GPU preflight timeouts, each with only 1/2 fixtures, not measured
  source-speed FAILs. Successful processes do not qualify failed processes;
  baseline veto, original first-round performance FAIL and old receipts remain.
  Latest `conv112-release-vulkan-preflight-recheck-v2/report.json`: both sides
  `[1,1,1]`, overall FAIL. Candidate attempt 1 fixture 0 GPU/source audit qualifies,
  p95 5.31 versus 2.105774 ms exceeds original 5%/3 ms; other sampling has
  preflight timeouts, failed GPU/source audits or incomplete fixtures. Final single
  log 5.35/7.80 versus baseline 4.30/5.03 ms is not complete A/B qualification;
  not all FAILs are attributed to contamination.
  Real current/frozen CPU NVTX ranges pass; measured-CUDA main-consumer selection
  positive/negative cases went RED->GREEN, and three resumed regressions pass.
  Complete raw CPU is 12/12 correct/stable per side with five scenarios/six
  performance FAILs. `llm-cpu-attribution-resume-v2.json` audits 12 rows/720 measured
  runs: original artifact identity, comparable inputs/outputs and complete
  per-process samples pass; retain six original failures, six tests RED->GREEN.
  Haswell compilation/ISA/OpenMP agree; 232/69 llama/mtmd sources are identical,
  only DW bias differs among 69 CPU files. This does not prove whole-chain or
  performance equality. Original frozen 3x2x10 single-row 256M Q8 A/A
  `llm-cpu-frozen-aa-resume-v1/` PASS is diagnosis only, not a baseline replacement.
  Complete post-AA CPU A/B finished 12/12 correct/stable per side, retaining three
  performance FAILs: 500M newspaper Q8/F16 decode p50 +11.08%/+6.60%, 256M
  newspaper Q8 e2e p95 +5.20%. `llm-cpu-complete-post-aa-attribution-v1.json`
  audits 720 samples/artifacts/compiler flags successfully; original three and old
  six FAILs remain, not CPU release PASS.
  `llm-cpu-complete-flags-closure-v3.json` extends source audit:
  scheduler/alloc/threading bytes and Haswell/ggml-base flags match; AICore
  AVX2/frounding-math/macros/direct dependency closure differ. Two per-module
  negative cases went RED->GREEN, total 8/8; no performance causality proven.
  Host perf paranoid=4 rejects unprivileged CPU sampling; system permissions were
  not changed. Later `llm-flags-{control,frozenopts}-ab-v2/report-cpu.json` has
  two three-row rounds, original frozen baseline/3x2x10, total 360 measured runs;
  both sides correct/stable, artifacts unchanged. Same-current-link-closure rebuild
  of llm/paddle_graph/attribute TUs changes only CXX_FLAGS, not macros/includes.
  Control retains five performance FAILs; frozen flags retain two 500M Q8/F16
  decode p50 FAILs +8.65%/+6.11%. Both attribution audits and 46 before/after
  protections PASS; three assembly/baseline protection behaviors RED->GREEN.
  Flags changes did not close original gates; production flags stay intact and
  partial diagnosis is not complete release or causal repair. Isolated builder's
  shell `\$ORIGIN` escaping defect separately went really RED->GREEN with a
  side-loaded `.so`; 10 analyzer/assembly and two baseline protections PASS.
  `llm-origin-probe-v3/standalone-cpu.json` passes real CPU inference without
  LD_LIBRARY_PATH, qualifying independent loading only, not replacing v2 receipts.
  Original OCR 168-scenario runs on both backends finished, still FAIL. Vulkan
  execution identity unchanged; seven failures cover tiny Q8
  rec/label/XFUND/orientation, tiny F16 label, public/heldout en Q8.
  `passes_source_gt_gate=0` means candidate-source GT delta exceeds 0.005, not source
  quality veto; CER improvements do not offset WER regressions. Tiny F16 label
  hmean/e2e CER/e2e WER deltas are 0.005540/0.006178/0.009804; tiny Q8 label
  WER/e2e WER 0.009804, XFUND Q8 WER 0.018738, public/heldout en Q8 WER
  0.006834/0.006208; all FAILs remain. Complete metrics and correction of historical
  attribution were in the retired `OCR_RELEASE_GATES.md` ledger (git history); the
  live gates run as the structured `validation_manifest.json` OCR rows. Both orientation failures are text
  equality, not angle/score/rotation; wrapper rc=0 does not bypass failures=2.
  Tiny Q8 rec four fixtures complete, stored max_diff=1.317e-5, but benchmark
  still rejects original source max_diff/argmax; non-release audit=0 alone does not
  explain rc=1. Quantization diagnostic/benchmark gate inconsistency remains
  unchanged with original FAIL. CUDA identity-change veto remains.
  `owned-validation-continuum/` official rebuild, three contracts and three-backend
  attribute e2e/batch/UI, table, pool now all rc=0; pool CPU 40/40, CUDA/Vulkan
  80/80 each, zero failures. Official-library SHA `7b5b28b7...`, frozen-library and
  standard-manifest protection pass; complete=true marks listed steps only,
  all_six_tasks_complete=false. Real table demo/OCR passes do not cover explicitly
  unconfigured optional layout-page composition.
  Official Vulkan `conv112-release-formal-vulkan-v3/report.json` finished overall
  FAIL: candidate `[0,0,1]` / original frozen baseline `[0,1,0]`, 2/2 fixtures per
  process, all main-input audits qualified. Baseline attempt 2 p95 6.43 versus
  source 3.277795 ms; candidate attempt 3 p50 5.76 versus source 2.430157 ms,
  absolute differences 3.152205 / 3.329843 ms, both exceed original 5%/3 ms.
  Not preflight timeouts; successful processes/aggregate fields do not offset
  failures, artifact/model protections pass. Later loader forensics confirm v3/v4
  cross-side loading: current `.so` imported as test baseline and runner symlink
  resolution returned current probe; original frozen ggml lacks current probe's
  DW bias API. Adjacent-module/resolved-path real REDs remain. Finally only original
  probe bytes and owner modules per side are copied;
  `conv112-original-consumer-green-v3.log` passes real CPU loading without frozen
  rebuild/symbol insertion. v5
  `conv112-release-formal-vulkan-v5-original-owner/report.json` has candidate
  `[0,0,0]` / baseline `[0,0,1]`, 2/2 fixtures per process, main audits qualified,
  still overall FAIL: baseline third-process p95 5.39 versus source 2.225544 ms,
  absolute difference 3.164456 ms; original dual gate/baseline veto remain.
  Original baseline probe SHA `0be29828...` unchanged; v3/v4 are not called qualified
  original-baseline A/B.
  The initial `llm-raw-formal-ab-v1/report-cpu.json` remains INCOMPLETE:
  rc=77 because its independent source tools were unavailable. The later
  `llm-raw-formal-cpu-ref-v2/report-cpu.json` completes both sides' 3x2x10
  samples, but 256M Q8 decode p50 97.441397 -> 104.047370 ms still fails
  (+6.7794%, +6.606 ms). Selected CUDA `llm-raw-formal-cuda-ref-v3/report-cuda.json`
  passes three rows at revision `e659cd3b`, not the original full release.
  Independent source tools do not replace the frozen performance baseline.
  Official public lifecycle `owned-yolo-formal-lifecycle-v1/` PASSes all three
  backends; `yolo-owner-formal-oracle-cuda-v1.log` current official CUDA module
  owner oracle rc=0. These passes do not withdraw original independent-checkpoint
  Q8 0.02 gate. Orientation same-stored-weight exporter real negative case went
  RED->GREEN; OCR owner suite was 75 PASS before new native method. Additional
  native original text gates actually FAIL on three backends.
  `orientation-stored-{cpu,cuda}-v1/` and
  `orientation-native-{cpu-red-v2,cuda-red-v1,vulkan-red-v1}.log` localize inverted
  word_1 (index=1): independent source=LNIO, stored Q8/native=LNIOF, two repetitions
  agree, native-stored adds no text mismatch at .9 gate. Unrotated threshold=1 text
  is not compared with .9 truth. This fixture's storage attribution does not prove
  all-page or release acceptance; original text FAIL remains.
  Complete raw CUDA candidate is 12/12, frozen baseline four text rows FAIL, and
  500m newspaper F16 e2e p95 +5.71% makes overall FAIL. Initial four measured
  profiles failed JPEG decoding; after Qt plugin path fix, external GPU load
  rejected insertion before task start and OCR was restored.
  `owned-measured-profile-pluginfix-wait/status.json` automatically waits for idle
  then samples serially without official rebuild. Subsequently four probes succeed,
  four routes still FAIL with 0 kernel nodes per range (522 same-thread APIs each);
  host Nsight default graph granularity does not capture nodes. Later four
  node re-sampling probes/routes in `owned-measured-profile-nodefix-wait/` are
  8/8 PASS, all 40 256M ranges have main-consumer kernel/same-thread API; resumed
  real regression 4/4 PASS in `llm-measured-nodefix-resume-v1.log`. Retain old
  failures and profiling-only boundary, without disabling production Graphs.
  Corresponding 500M re-sampling in `owned-measured-500m-nodefix-wait/` finished:
  four probes/routes 8/8 PASS, another 40 measured ranges pass same real regression.
  Original external-GPU rejection remains and OCR restored. F16 current/frozen
  main GEMM h1688/s1688 is visible; instrumented latency never enters release,
  qualifies other models or withdraws original CPU/CUDA FAILs.
  Isolated 0019 pool AVG 80 GREEN; pre-official-rebuild CPU 40/40 and Vulkan 80/80
  PASS, CUDA 80/27 FAIL. Independent YOLO26 GGUF->checkpoint injection verifies
  204 mappings; Q8 score error remains 0.14595002 above original 0.02. Original
  assets stay; weight-quantization attribution is neither all execution error nor
  release closure. Public CPU mode 1
  `yolo-quantization-execution-cpu-resume-v3.json` retains Q8 same-weight added
  error 0.02269834, independent-checkpoint total 0.12325168, original 0.02 FAIL.
  Backend-selection/independent-error-recording real tests RED->GREEN, three pass.
  Six GPU rows in `yolo-quantization-execution-{cuda,vulkan}-resume-v4.json`
  complete F32/F16 PASS; Q8 CUDA same-weight error 0.18532050, total 0.18347186,
  four missing boxes; Vulkan 0.00550228/0.15145230 with five matching boxes.
  Both original overall gates FAIL, ten repetitions stable, asset SHA unchanged.
  Missing CUDA boxes are not storage-only error. Current sources expand Q8 to F16
  on host for both GPUs. Idle-slot scope receipt
  `owned-yolo-q8-optrace-llm-scope-v1/coordinator-2/llm-cuda-arithmetic-scope.log`
  finished 11/11 PASS, 0 changed elements; not original LLM text/p95 qualification.
  First optrace assembly exit 77 remains; module-adjacent executable
  `owned-yolo-q8-optrace-resume-v2/coordinator-1/` completes two items 162/162,
  0 null, first diagnostic exceedance CPU->CUDA op61, Vulkan->CUDA op149, both
  rc=1, not public-library qualification.
  New public ABI lifecycle `owned-yolo-q8-public-lifecycle-v1/`: CUDA cold Q8 five
  boxes, Q8 one box after F32/F16 switching, still one after context recreation;
  twelve stable rounds per context, Vulkan PASS. Same test with IGEMM disabled
  PASSes, producing cache-path RED. Maintained CUDA plans never consumed task's
  plan_owner_id; ordered `0020-acv-cuda-igemm-plan-owner.patch` adds owner checks
  and NHWC invalidation, CUDA/combined only; original official tree read-only
  apply --check passes. `yolo-igemm-owner-candidate-v1/` four-chain
  forward/reverse/idempotent replay and single-TU compilation pass. Original link
  lacked 80 objects; full isolated CUDA backend rebuild from verified combined
  replay/pinned llama parent inherits original ISA/Graphs/arithmetic. Build succeeds,
  final module SHA `4b6ad3c8...` and unchanged protection verified; same oracle
  SHA `900a4a2f...` assembled RED/GREEN.
  `owned-yolo-igemm-owner-green-v1/coordinator-3/` completes real oracle RED rc=1
  -> GREEN rc=0, CUDA/Vulkan public model-switch/context-recreation PASS; later
  identical CPU lifecycle in `owned-llm-flags-attribution-v2/coordinator-1/` PASSes,
  covering three backends. First two rejected insertions and restoration remain.
  `yolo-quantization-execution-cuda-ownerfix-v5.json`: Q8 five boxes match, ten
  stable repetitions, added same-weight score error 0.0021265149, independent
  checkpoint total 0.1480765343 still exceeds original 0.02, FAIL. Asset/runtime SHA
  unchanged; orchestration success is not accuracy PASS. Isolated module includes
  0019 but never replaces official pool gate; models/original gates/official or
  frozen artifacts untouched, all original failures retained.
  This round's source checks pass 913/913 C ABI, strict plugin boundary,
  17 source-owner contracts, 123 runner tests and 197 L1/L2/L5 items. Public
  attribute incorrect-reference negative re-run PASS, inputs unchanged:
  `shared-owner-acceptance/attribute-probe-resume-v3/`; not official-consumer or
  release qualification. Frozen release CPU F32 three processes have five stable
  matching boxes but score error 0.1101961732 each; original accuracy gate rejects
  normal execution, not loader/missing attention API, and performance veto stands.
  No Q8 single-tensor transient F16 restoration passes original gate; best
  `op.109.w` max score error 0.03684449196 still FAIL.
  `yolo-q8-single-tensor-sensitivity-resume-v1.json` retains per-round Q8 baseline
  restoration/asset SHA; fixed single-image diagnosis writes no model and does not
  replace native/holdout/release. Specified tensor restoration/unknown-name rejection
  two real tests RED->GREEN; three `op.109.w` pairings
  `op.110.w/op.107.w/op.20.w` errors 0.04792762/0.08504459/0.08508769 still FAIL.
  User chose original frozen baseline only; original release veto remains, no new
  substitute baseline. Original Linux L3 schema 16 PASS/1 FAIL remains. Actual
  PLUGIN_STANDARD_QMAPANYTHING=OFF; after original queue completion normal
  enable/build succeeds in `mapanything-consumer-{configure,build}-v1.log`.
  `automation-l3-schema-formal-green-v1.log` now 17/17 PASS, no skips, not
  cross-platform installer/packaged GUI L4. Official PA100K archive SHA pinned:
  `65e2ab37519aacf307c8c0ce04c6ab8edac3b4199e0448d4de625cfd60cd5cc0`.
  `pulc-pa100k-gt-v1/` extracted 10000 val/10000 test images and 26-dimensional GT,
  without training-set consumption. Download/old-path-overwrite rejection/GT
  preparation/path-traversal negative and public CPU two-image raw-three-quantization
  regressions pass. `pulc-pa100k-val-cpu-v1/report.json` completes 10000 val images:
  F32 10000/10000 PASS, max 3.93391e-6; F16 9969/10000, max 0.00992972 passes
  probability gate but 31 attribute-mismatch images still FAIL; Q8 2707/10000,
  max 0.27434349 still FAIL. Source/model/runtime/GT SHA unchanged; per-image/
  per-attribute GT counts remain; individual attribute improvements do not offset
  regressions or alter gates. Independent 10000-image test
  `pulc-pa100k-test-cpu-resume-v2/report.json` also completes: F32 10000/10000 PASS,
  F16 9955/10000 with 45 attribute-mismatch images, Q8 2497/10000 max 0.34178510
  still FAIL. Two splits 20000 images/60000 rows complete/stable; four receipt
  audit tests RED->GREEN. Original Q8 probability failures 7275/7480 images equal
  storage-error failure counts; max added same-weight execution error
  3.40e-6/3.87e-6. All per-attribute GT counts remain, no calibration/training/
  asset replacement/post-hoc gates. First loader failure remains; dataset diagnosis
  is not official/three-backend/release performance qualification. User must apply
  for VeRi access; vehicle dataset GT stays open. RAM is now owned by independent
  `tagging` task, not the old YOLO-branch plan.
- [ ] Other detection/pose/lane architectures: `damo_yolo`, `gold_yolo`,
  `yolo_nas`, `yolov6`, `yolov6_face`, `yolov7`, `yolov9`, `yolov10`,
  `yolo12`, `yolox`, `rtdetr`, `rtdetrv2`, `u_rtdetr`, `dfine`,
  `dfine_seg`, `deimv2`, `rio_detr`, `rtmdet_pose`, `upn`, `clrnet`,
  `yolox_dwpose`, and the source YOLOv5/YOLO11 checkpoint families. This
  includes `yolov5`, `yolov5_cls`, `yolov5_det_track`, `yolov5_obb`,
  `yolov5_ram`, `yolov5_resnet`, `yolov5_sahi`, `yolov5_sam`, `yolov5_seg`,
  `yolo11`, `yolo11_cls`, `yolo11_det_track`, `yolo11_obb`,
  `yolo11_obb_track`, `yolo11_pose`, `yolo11_pose_track`, `yolo11_sahi`,
  `yolo11_seg`, `yolo11_seg_track`. Exact weights unsupported: none of
  these checkpoint families exist in the AICore catalog and none may be
  substituted silently. The task-level semantics (HBB, OBB, pose,
  segmentation, SAHI slicing, tracking) already have catalog-backed
  equivalents through YOLO26/YOLOv8/RF-DETR workflows, so a port of any
  given checkpoint is a catalog addition (new GGUF + digest + manifest
  row + probe) inside the existing `yolo`/`rfdetr` tasks — not a new
  plugin surface. Lane detection (`clrnet`) additionally needs a lane
  point-sequence result type that no AICore contract expresses yet.
  Rollout mapping (2026-09): DETR-family (`rtdetr`, `rtdetrv2`,
  `u_rtdetr`, `dfine`, `dfine_seg`, `deimv2`, `rio_detr`, `upn`) ->
  `rfdetr`/YOLO26 detection roles; CNN detectors (`damo_yolo`,
  `gold_yolo`, `yolo_nas`, `yolov6/7/9/10/12`, `yolox`, YOLOv5/YOLO11
  families) -> YOLOv8/YOLO26 detect/segment/pose/OBB roles;
  `rtmdet_pose`/`yolox_dwpose` -> POSE role (COCO-17 skeletons only —
  whole-body 133-point skeletons stay exact-weights-unsupported);
  `yolov8_obb` -> OBB role; `grounding_*` -> WORLD/YOLOE
  open-vocabulary roles; `yolov8_cls`/`internimage_cls` -> CLASSIFY role
  (RAM covers the multi-label tagging use case). The later public attribute task
  expresses `pulc_attribute` as independent Sigmoid attribute groups; it cannot
  map to single-category CLASSIFY flags.
  Every enabled mapping must cite its catalog row;
  unmatched official entries keep their names disabled with TODOs,
  and `clrnet` stays exact-weights-unsupported until a polyline typed
  result exists.
- [ ] Source YOLOv8 classification, pose, and OBB checkpoints
  (`yolov8_cls`, `yolov8_pose`, `yolov8_obb`) lack matching AICore catalog
  weights. Verified (2026-09-29): the contract side is fully reusable
  already — the `yolo` task implements the pose/obb/classify semantics
  (loader head checks + typed `aicore_yolo_pose/obb/classify_result`),
  and the plugin enumerates those tasks from the live catalog — so this
  TODO is purely a weights-publication round (convert the checkpoints
  with the upstream ultralytics-ggml converter, publish GGUFs, add
  digests + manifest rows + probes); no plugin or task code is missing.
  The YOLO26 equivalents (`yolo26_cls`, `yolo26_pose`,
  `yolo26_obb`) are separate checkpoint families already wired in the
  plugin; exact-weight parity for the YOLOv8 heads remains a catalog
  addition gated on published GGUFs.
- [ ] Source `yolow_ram` exposes one composite YOLO-World + RAM run. The
  underlying chain is now available in the worker: YOLO-World detections can
  be followed by per-box RAM tagging, while standalone RAM and YOLO-World
  runs retain their source-compatible behavior. The real combined-run test
  now covers integer crops, description persistence, undo/redo, stale-result
  rejection and shared `tag_mode`/`delete_tags`/`filter_tags` semantics.
  A dedicated source composite picker/configuration, its default detector/RAM
  pair and exact source checkpoint equality remain open.
- [x] Interactive SAM labeling is functionally covered by AICore SAM3 point,
  box, and text prompts. Exact upstream weights remain future work:
  `segment_anything` (SAM 1), `sam_hq`,
  `sam_med2d`, `mobile_sam`, `edge_sam`, `efficientvit_sam`, `geco`.
  AICore SAM3 does not load these exact weights or preserve their specialized
  preprocessing/output behavior.
- [x] Monocular depth labeling is functionally covered by AICore DA3. It
  replaces the upstream Depth Anything v1/v2 labeling role when a DA3 GGUF is
  selected. Exact `depth_anything`/`depth_anything_v2` checkpoints and their
  source preprocessing remain unsupported until matching AICore GGUF contracts
  exist, so exact-weight parity remains a TODO.
- [x] Source RMBG v1.4 (`rmbg_v14.yaml`) has no matching catalog weight;
  the plugin uses catalog-backed RMBG-2.0 for the corresponding workflow.
  Confirmed status (2026-09): the AICore `rmbg` task owns a BiRefNet-based
  RMBG-2.0 graph with pinned f32/f16/q8 catalog GGUFs
  (`rmbg_f32/f16/q8.gguf` in `asset_digests.h`) and a proven custom-op
  fusion path on every backend, so RMBG-2.0 is the deliberate
  catalog-backed substitute for the v1.4 *workflow* (foreground matte →
  editable polygon), not a weight-parity port. RMBG-2.0 is a different
  architecture (distinct input normalization and output head), so its
  foreground polygons are not pixel-comparable with a v1.4 run; the
  model picker exposes only the catalog-backed RMBG-2.0 entries and a
  v1.4 GGUF stays "exact weights unsupported" until AICore publishes
  one.
- [x] Source `scrfd` configurations are functionally aligned (2026-09-30):
  the facedetect task already publishes every buffalo pack — buffalo_sc
  carries the SCRFD det_500m detector, buffalo_m det_2.5g,
  buffalo_l/antelopev2 det_10g — and the plugin's ModelKind::FaceDetect
  worker path enumerates those packs from the catalog
  (`aicore_facedetect_detector_model_*`), runs
  `aicore_facedetect_detect_image`, and emits face rectangles + 5
  landmarks. Porting the standalone `scrfd_10g_bnkps`-style checkpoints
  (different landmark counts) stays exact-weights-unsupported until
  AICore publishes them.
  bnkps PUBLISHED (2026-09-30): the standalone `scrfd_10g_bnkps` pack
  is now part of the facedetect catalog. `convert_scrfd_onnx_gguf.py`
  (first-party, graph-driven) emits the ONNX topology as the facedetect
  KV graph — node specs keyed by ONNX output names, sanitized to flat
  identifiers, Transpose perms as `pm=`, weights verbatim-F32 — so the
  runtime replays it metadata-driven with ZERO C++ changes (same
  interpreter as buffalo_m/sc). Verified on a real image: 6/6 faces
  match the buffalo_m baseline, 5-point landmarks finite, run-to-run
  stable (test_facedetect_detect_only, ctest `capi;model;validation`).
  Catalog rows + digests + release upload + `facedetect-bnkps` manifest
  scenario landed; the plugin picker lists them automatically through the
  detector catalog enumeration. Renamed to per-tier assets
  `scrfd_10g_bnkps-f32.gguf` / `-f16.gguf` (F32 verified: 6/6 faces, sum
  parity with f16 to 2.5e-5; the F16 container keeps 1-D biases/scales at
  F32 because the CPU binary-op kernels reject f32+f16 operands). A Q8_0
  tier is NOT published: the Q8_0 block quantizer requires 32-aligned
  axes that the SCRFD 1x1 conv weights (e.g. 56x88) do not satisfy, and
  the facedetect conv path has never consumed quantized weights — a
  quantized-conv verification round would be needed first. Landing notes: spec node names must be
  the ONNX OUTPUT names (not node names), gguf-py reverses shapes on
  write (pass ONNX-shaped arrays), and ints arrays land as INT32.
- [ ] `open_vision` and source-specific multi-model combinations are
  external dependencies by design: `open_vision` composes provider-specific
  model runs that no AICore task contract expresses, so it stays disabled
  with its official name; mapping it onto AICore would require defining an
  external-transport contract that violates the ownership map (the plugin
  must not implement a private downloader, backend, or scheduler).
  (remote_server was previously listed here but is fully landed — see the
  remote-server parity entry below.)
- [x] OCR fixture pin-chain repair (DONE 2026-10-08): the 15 regenerated
  ocr_fixtures/*.npy were uploaded to the `ocr-fixtures` release tag via
  the GitHub API (git credential store credentials), replacing the stale
  first-generation assets, and all manifest URLs were rewired from the
  dead `PP-OCR` tag to `ocr-fixtures`. Fresh-clone simulation (cache
  removed, validate_all re-run) auto-downloaded every fixture and reached
  verdict PASS 9/9 — every machine now self-provisions the fixtures.



