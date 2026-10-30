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
attribution of the four exact-file rows has been corrected in
`OCR_RELEASE_GATES.md`.
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
Grounding, are tracked in [TODO.md](TODO.md).
