# qAnyLabeling upstream parity audit

Audit baseline: the ACloudViewer worktree and local upstream
`/home/ludahai/develop/code/github/dl/X-AnyLabeling-GGML`, commit
`745f6c355f94cddb06f6965b3a4118f82033e935` (v4.1.3). Feature parity, model
publication, numerical/performance acceptance and packaging are separate
contracts. A focused pass does not establish complete upstream parity.

- **Aligned within scope**: implementation, source comparison and reproducible evidence exist.
- **Partial**: the main workflow exists, with remaining model, visual, quantization or platform coverage.
- **Open**: no equivalent implementation or qualification is complete.
- **Matches upstream absence**: upstream has no registered workflow; do not invent one and claim parity.

## Current assessment (2026-10-28)

本轮实际比较的上游为 `cd26942a007b6ce7bd521de801a6cfd455d7fd9b`。
开头的 `745f6c35` 是历史 audit baseline，并非当前全部模块已对齐的证明。
ACloudViewer HEAD 为 `a624587d7af8f6b11c44339de55cdc0f8b85d860`，工作树 dirty。

`build_app/Testing/upstream-cd26942-config-red-v2.json` 实跑 FAIL：缺
`clear_annotation_checked`、`fail_annotation_checked`，defaults 与当前 YAML
不同，九项 palette 和四项 panel 样式不同。已核实 source 的 `checked` 支持
`false/true/"failed"`，review/checked 属性搜索、rectangle wheel 默认禁用、
image-tags 默认展示及新 canvas 模式仍需对齐；本地文件搜索现在仅做名称子串。
用户本轮选择先复验证据，未实施新增三态或以复制 defaults 掩盖行为缺口。

OCR picker 发布全集回归已 RED→GREEN：按公共 preset 的 rec filename 集合
逐项比较，保留 catalog F32/F16/Q8_0 与下载元数据、无重复，缺 catalog 才 local
F32 fallback。`ocr-picker-catalog-red-v1.log` 复现被白名单隐藏的量化；修后
`ocr-picker-catalog-green-v2.log` 两项通过，原 attribute F32 资产环境下
`ocr-picker-catalog-suite-v3.log` 38/38。旧单 F32 测试过期，旧 38/38 不证明
发布全集覆盖。发布事实、picker 投影与独立精度/release 资格仍是三个契约。

对同一 `cd26942a` 的额外实测：`upstream-cd26942-document-v1.json` 的
LabelFile 路径/像素优先级 8 例通过，`upstream-cd26942-tagging-v1.json`
的 4585 bilingual labels / 16 展示与过滤例通过，不覆盖审核三态或推理质量。
`upstream-cd26942-render-v1/report.json` 的 36 绘制例总 FAIL，19 项错误：
包括三项 polygon geometry-only 像素不等，以及 16 项文本/颜色/位置超原界。
原一像素字体容差和数值界未改；不得把 geometry、字体、新 canvas 模式混为全窗口通过。
`ocr-picker-sibling-qocr-v1.log` 两项通过；`evidence-continuation-capi-contract-v2.log`
40/40 模型无关契约通过。未注入完整资产的广泛 capi 运行保留为诊断：
80 项中 16 skipped、attribute E2E 一项缺资产失败；补齐原 source reference 的
`evidence-continuation-attribute-v3.log` 单项通过，不追写整套为 80/80。

`complete_upstream_parity` remains **false**. The public local LLM/VLM task,
optional projector, catalog and worker exist; PULC person/vehicle also have a
public attribute task. Their implementation must not be described as absent,
but neither implementation proves full independent accuracy/release acceptance.

Real Linux Qt5.10.1 and Qt6.5.3 focused builds exercise shared helpers,
canvas/navigation and OCR document/table/rich-text consumers. These checks do
not certify complete window layouts, IME/DPI, old-Qt Markdown import or native
Windows/macOS packages. Runtime-copy/install tests cover DLL/alias naming and
failure propagation, not those platforms' loader/SDK/device behavior.

The original LLM CPU performance, full CUDA text, YOLO Q8 checkpoint, OCR GT
and Vulkan original-baseline veto remain open. See the latest assessment in
[OCR_RELEASE_GATES.md](OCR_RELEASE_GATES.md) and the llama evidence index.
Keep both ledgers until their original acceptance surfaces are closed.

## What the original twelve F32 files represent

以下 local/unpublished 表是发布前的历史来源记录。当前公开 OCR catalog 共
48 行，2026-10-25 的 36 新文件与一 replacement 发布事实见 release ledger。
本轮不重新上传或替换；当前 picker 已覆盖发布 rec 集合，原数值/速度 FAIL 不变。

These are local source-ONNX conversion outputs, separate from the twelve already
published OCR text/layout files:

| File | Upstream role | State |
| --- | --- | --- |
| `ppocr-cls-f32.gguf` | PP-OCR 0/180-degree orientation classifier | Local; unpublished |
| `ppocrv4-source-det-f32.gguf` | Chinese PP-OCRv4 detector | Local; unpublished |
| `ppocrv4-source-rec-f32.gguf` | Chinese PP-OCRv4 recognizer | Local; unpublished |
| `ppocrv3-japan-rec-f32.gguf` | Japanese PP-OCRv3 recognizer | Local; unpublished |
| `ppocrv5-mobile-det-f32.gguf` | PP-OCRv5 mobile detector | Local; unpublished |
| `ppocrv5-mobile-rec-f32.gguf` | PP-OCRv5 mobile recognizer | Local; unpublished |
| `ppocrv6-tiny-det-f32.gguf` | PP-OCRv6 Tiny detector | Local; unpublished |
| `ppocrv6-tiny-rec-f32.gguf` | PP-OCRv6 Tiny recognizer | Local; unpublished |
| `ppocrv6-small-det-f32.gguf` | PP-OCRv6 Small detector | Local; unpublished |
| `ppocrv6-small-rec-f32.gguf` | PP-OCRv6 Small recognizer | Local; unpublished |
| `ppocrv6-medium-det-f32.gguf` | PP-OCRv6 Medium detector | Local; unpublished |
| `ppocrv6-medium-rec-f32.gguf` | PP-OCRv6 Medium recognizer | Local; unpublished |

All twelve stems now have F32/F16/Q8_0 conversions: **36 new GGUFs**, plus two
separate public-Q8 repair candidates. All 38 are staged persistently at
`build_app/Testing/ocr-storage-reference/ocr-release-candidates-20261019/`.
`inventory.json` records original/resolved paths, size, SHA and real dtype
composition; `SHA256SUMS` covers all files. Latest blockers are tracked in
[OCR_RELEASE_GATES.md](OCR_RELEASE_GATES.md). Historical v7/v17/v19 experiments
are not complete release acceptance.

Source ONNX URL/size/SHA/revision are pinned in
[`ocr_onnx_sources.json`](../../../../core/AICore/tools/ocr_onnx_sources.json).
Orientation, six source presets, selected polygons and source-page formal
consumers passed on CPU/CUDA/Vulkan: 18 worker/window return codes were zero in
`build_app/Testing/qanylabeling-ocr-formal-{cpu,cuda,vulkan}` and
`check_ocr_formal_consumers.py`. This covers local F32 wiring only.

Conversion supports explicit source-SHA-pinned mixed precision.
`core/AICore/tools/ocr_quantization_policies.json` remains experimental, outside
the product download catalog. The 24 v7 quantized conversions replayed identical
SHAs; each backend passed 22/24 multi-input graphs and 33/36 including orientation.
Later Tiny F16, v4 Q8, v5 F16/Q8 and v19 Tiny Q8 repairs have separate identities
and evidence below. Original failed files do not inherit their results.

Published `ppocr-det-ch`, `ppocr-rec-ch`, `ppocr-rec-en`, `pplcnet-layout` each
have F32/F16/Q8_0 files, twelve in total, at:

```text
https://github.com/Asher-1/cloudViewer_downloads/releases/download/PP-OCR/<file>.gguf
```

The user authorized uploading all individually qualified files to this existing
tag. First pass all required numerical/page/performance gates, then upload,
verify release-asset SHA and update `asset_digests.h`, catalog dump, manifest and
cold-cache download. A local ZIP or candidate directory is not a release URL.
The pre-publication snapshot recorded zero newly qualified uploads. The later
2026-10-25 Plan B publication record (36 new files plus one replacement) is
preserved in the release ledger; it does not close current stricter revalidation.

## Complete parity surface

| Scope | Upstream contract | Current implementation/evidence | State | Remaining work |
| --- | --- | --- | --- | --- |
| Published OCR weights | Four roles, three dtypes under `PP-OCR` | Catalog/SHA/cache and historical matrix exist | Published; stricter revalidation open | Recheck exact existing files under current gates |
| Twelve source F32 roles | v4/v5/v6/Japanese/orientation YAMLs and source code | Three backends 12/12 source graphs; older CUDA developer 42/42; local-only files | Partial | Current-build pages, controlled performance, upload/SHA/catalog/download |
| Source F16/Q8_0 | Original ONNX/Paddle truth; no upstream GGML quantization contract | Failed old candidates retained; v17 Small F16/v19 Tiny Q8 have separate graph/page evidence | Partial | Remaining page combinations/XFUND/performance; disclose mixed storage |
| Orientation | Source crops, BGR, `use_angle_cls`, strict `score > 0.9` rotation | Optional public cls ABI, batch/threshold/angle/selected-region results | Aligned within scope | Quantized release and full-window visuals |
| Dedicated OCR window | `views/labeling/ppocr/*`, `ppocr_dialog.py`, PDF/layout/table/formula/API/remote service | Separate 1560x920 window, minimum 1200x760, sidebar 240; async API, JSON/JSONL, favorites/search/status, upload, discovery, remote image cache, block editing, JSON/Markdown/ZIP, table editor; Qt5/mock/plugin tests | Partial | Native PDF, formula/rich-text details, complete visual/interaction states |
| Composite presets | Source detector/recognizer/cls/dictionary/defaults | Eight AICore presets；published rec picker 集合回归及 38 项 suite 通过 | Partial | 全 det/rec/cls 量化组合、source defaults、独立精度与完整窗口验收 |
| KIE | Entity/link import/export; no active automatic SER/RE registration | Empty IDs, directional links and coordinates aligned; ABI retains KIE fields and rejects nonempty model option | Matches upstream absence | Add automatic flow only if upstream registers one |
| OCR accuracy | Independent original framework and real pages | Original and stored-weight source comparison, 50 receipts; failures retained | Partial | All pages/backends/controlled A/B; graph pass cannot cover page failure |
| Point-cloud 3D data | Cuboid/segmentation sidecars, KITTI/DATUMARO/Sly | C++ sidecars, class workspace, three ZIP formats, transactional preview/rollback, 87 gestures | Aligned within scope | Full windows and native cross-platform events |
| Point-cloud cameras | Camera panel, Z-up, distortion/depth order/preview | 15/15 scene pixels, calibration/projection/crop/live preview | Partial | Full-window fonts/DPI/icons/panel overlap |
| Orthographic views | Top/Side/Front handles, crop depth/colors/point size/expand | Borrowed visible indices, Side/Front +0.1m crop, Top/create bypass, semantic/instance/RGB/intensity, device-pixel sizing, expand/restore | Aligned within scope | Complete focus/overlap/keyboard/window states |
| Point-cloud streaming | Upstream asynchronously loads complete frames, no chunk protocol | Complete-frame loading | Matches upstream absence | Chunking is an optional optimization, not missing source parity |
| Chatbot | Streaming history/stop/images/providers/messages/UI | Qt Network/SSE, cancellation/timeouts, edit/delete/copy/resend/regenerate, bounded folder AutoRun, ShareGPT, eight providers, model search/favorites/discovery | Partial | Full Markdown/code/math and complete layouts; upstream has no Alpaca export |
| VQA | Templates/component migration/AI preview/apply/references/folder JSONL | Four component kinds, manual/AI values, disk references, field selection/renaming, import/export/sort/migration, eleven system templates/user templates, source-style dialogs/config | Partial | Complete visuals and Qt6/platform acceptance |
| VLM-R1 OVD | JSONL and dialog-annotation format | `AnnotationVlm` import/export/actions | Aligned within scope | Format support is not local VLM inference |
| Local LLM/VLM | Text model plus optional multimodal projector | Public `llm` C ABI, catalog/cache, local worker and model-backed probes | Partial | Original CPU/CUDA release, complete UI and native platform qualification; not a Florence-2 substitute |
| PULC attributes | Person/vehicle classification | Public attribute ABI/catalog and real plugin batch/UI consumers; PA100K val/test F32 pass, original F16/Q8 fail | Partial | Original quantized independent-checkpoint gates, vehicle licensed GT and complete release/UI |
| Other exact local families | `florence2_large_ft`, `grounding_dino_api`, four Grounding-DINO/SAM combinations, InternImage | No complete qualification for these exact registered checkpoints | Open | Exact model task/ABI/catalog/digest/defaults/numeric/performance/UI |
| DEIMv2/DINOv3 | S/M DINOv3 and HGNetv2-N official checkpoints | Other detection semantics exist, not these exact checkpoints | Open | Conversion/publication/catalog/manifest/backend gates |
| Other exact families | DAMO/Gold/YOLO-NAS, YOLO6/7/9/10/12/YOLOX, RT-DETR/v2/u_rtdetr/DFINE/RIO, RTMDet pose, UPN, CLRNet, DWPose and YOLO5/11 combinations | Some semantics/different checkpoints only; no exact consumer for all lanes/whole-body/tracking/SAHI/SAM/RAM combinations | Open | Record exact checkpoints vs substitutes; never silently substitute |
| Defaults/theme/resources | Shortcuts/palette/panels/themes/icons/RAM presentation | 旧 anchor 的局部证据；当前 cd26942 config comparator 实际 FAIL | Partial | 审核三态、搜索、defaults/palette/panel 新差异及 full-window font/DPI/Qt6/platform packaging |
| RAM resources | GGUF embeds English tags; Chinese mapping remains separate | `resources/ram/*` is presentation mapping embedded in RCC; no source-path lookup | Aligned within scope | Use a public API if future GGUFs provide Chinese mapping, then remove fallback |
| Linux relocation | Plugin/RCC/RPATH/Unicode prefix | `qanylabeling-resource-install.json` passed actual install block/RPATH removal/relocation | Partial | Complete current package inspection/clean-machine/release profile |
| Windows/macOS packages | Native installer/DMG/update workflows | No current native qAnyLabeling package/build/relocation receipt | Open | Native configure/build/install/relocate/start/resource checks |
| Qt5/Qt6 visuals | Upstream PyQt6, host Qt5.10–Qt6 | Real Linux Qt5.10.1/Qt6.5.3 focused consumer builds/tests; Qt5.15 app regression | Partial | Complete fonts/IME/DPI/resources/screenshots and native Windows/macOS acceptance |

## 2026-10-19: Per-model historical snapshot

The table retains v7 identities. F32 covers multi-input source graphs on all
three backends. F16/Q8 columns show CPU maximum probability errors, whose
PASS/FAIL was rechecked on both GPUs; none establishes full release eligibility.
Detectors use two real pages, recognizers four Chinese/English word images,
classifiers five word images plus 180-degree counterparts, ten inputs total.
Recognizers/classifiers require every timestep argmax; detectors also gate mean
error. Stored-weight ONNX comparisons distinguish runtime from quantization.

| Stem | F32 CPU/CUDA/Vulkan | F16 CPU max error | Q8 CPU max error | Release |
| --- | --- | --- | --- | --- |
| `ppocr-cls` | PASS/PASS/PASS | PASS 0.000174344 | PASS 0.000956574 | Blocked |
| `ppocrv4-source-det` | PASS/PASS/PASS | PASS 0.020687461 | PASS 0.106714340 | Blocked |
| `ppocrv4-source-rec` | PASS/PASS/PASS | PASS 0.000802502 | PASS 0.002008081 | Blocked |
| `ppocrv3-japan-rec` | PASS/PASS/PASS | PASS 0.000351965 | PASS 0.002486408 | Blocked |
| `ppocrv5-mobile-det` | PASS/PASS/PASS | PASS 0.020731091 | PASS 0.035158217 | Blocked |
| `ppocrv5-mobile-rec` | PASS/PASS/PASS | PASS 0.000558376 | PASS 0.003126979 | Blocked |
| `ppocrv6-tiny-det` | PASS/PASS/PASS | PASS 0.013615757 | PASS 0.075072199 | Blocked |
| `ppocrv6-tiny-rec` | PASS/PASS/PASS | FAIL 0.006315589 | FAIL 0.890249828; argmax 0.975 | Blocked |
| `ppocrv6-small-det` | PASS/PASS/PASS | PASS 0.014678866 | PASS 0.064754665 | Blocked |
| `ppocrv6-small-rec` | PASS/PASS/PASS | PASS 0.000339448 | PASS 0.002089202 | Blocked |
| `ppocrv6-medium-det` | PASS/PASS/PASS | PASS 0.003221691 | PASS 0.019202054 | Blocked |
| `ppocrv6-medium-rec` | PASS/PASS/PASS | PASS 0.000140548 | PASS 0.002930522 | Blocked |

Detector budgets: F16 max <0.1 and mean <0.0001; Q8 max <0.6 and mean <0.0005.
Recognizer/classifier max budgets: F16 <0.001, Q8 <0.005. Stored-source and
native-CPU diagnostics require max <0.001 and relevant argmax=1. These are
role-specific budgets, not interchangeable max-error rankings.

Reports under `build_app/Testing/ocr-storage-reference/`:

| Report | Historical result and limit |
| --- | --- |
| `f32-cpu-current-graphs/matrix.json` | 12/12 source graphs |
| `f32-vulkan-current-graphs/matrix.json` | 12/12 source graphs |
| `f32-cuda-current/matrix.json` | 42/42 before detector-filter repair; pages not current-build release evidence |
| `quant-v7-{cpu,cuda,vulkan}/matrix.json` | Each 33 PASS / 3 FAIL; Tiny F16/Q8 graphs and Tiny Q8 orientation failed |
| `quant-policy-replay-v7.json` | All 24 conversion SHAs replayed |
| `full-native-regressions.log` | 35/35, no skips |
| `source-detector-filter-native-tests.log` | 12/12 affected OCR/table/plugin/validation, no skips |
| `registered-gate-sroie-tests.log` | Runner 96, GT seven, probability seven, GPU audit seven tests |

Scene GT consists of 201 nonignored word boxes on 50 official validation images.
Detection Hmean, fixed-GT-region CER/WER and complete-pipeline CER/WER each gate
source degradation <=0.005. Word-GT vs line-detection granularity limits absolute
Hmean; these are not official leaderboard scores. XFUND Chinese/Japanese each
use 50 semantic-field pages for region recognition, not detection/automatic KIE.
SROIE contributor receipts have 50 full scans / 2,779 annotations / 18 composite
scenarios with 100 pinned image/CSV SHAs and annotation-directory digest.
See [DATA_CARD.md](../../../../core/AICore/tests/ocr/DATA_CARD.md) for selection,
IoU and provenance. This is not the official held-out/DetEval competition set.
Pinned input/archive/dictionary downloads do not establish model downloads.

Source `TextDetector.filter_tag_det_res` also reorders by first extrema, clips
to width-1/height-1 and removes short edges. The local source path had omitted
that. Equal extrema in a diamond can repeat corners; the singular OpenCV warp
samples the top-left pixel. The local implementation now reproduces that actual
behavior. Thirteen source-generated filter fixtures, singular crops and geometry
goldens passed. The graph is unchanged but pages require a new runtime identity.
Older CPU/receipt/v9 cohorts were stopped with interrupted receipts. New
`filter-v7-cuda` ran 36 scene/receipt cases; `.progress` files are excluded from
final metrics. v8/v9 results must retain their build identity.

The following use production runtime
`154a0b62460edbe0273deeb17f766bd0d949aefc63a4782b0d9c3251665105cf`
and page probe
`30c2bca6387f9f13b22e0ababf1716c4886c8ad4454cd0870211380fc3f81bee`:

| Candidate SHA prefix | Source graph | Page evidence | Limit |
| --- | --- | --- | --- |
| v8 v4-det Q8 `a62a8e2b86cb` | PASS; CPU max/mean 0.102772355/0.000298230 | Chinese/Japanese scenes CPU/Vulkan; CUDA scenes/receipts four checks | `filter-v8-{cpu,cuda,vulkan}`, CUDA 5/5; v8 replay identical |
| v12 v5-rec F16 `a5ce8f19f598` | Five inputs PASS, max 0.000957310, argmax 1 | CUDA scenes/receipts | Actual F16 21.23%, F32 78.77% |
| v14 v5-rec Q8 `72d723b5c8e2` | Five inputs PASS, max 0.004976571, argmax 1 | CUDA scenes/receipts | Q8 2.10%, F16 19.13%, F32 78.77%; near budget; v12/v14 replay identical |
| v15 Tiny-rec F16 `b5a7a723e105` | Four inputs PASS, max 0.000872791, argmax 1 | CUDA orientation/receipts; scenes with old F16 detector failed | Hmean -0.005540166; e2e WER +0.009803922 |
| v16 Tiny-det F16 `804a564ad2db` | Two inputs, all backends PASS | CUDA scenes/receipts with v15 recognizer | `conv2d_112.w_0` retained F32; v15/v16 replay identical; other pages/XFUND open at this snapshot |

The v5 recognizer added the real perspective crop `val_img_7.jpg`, increasing
inputs four to five. Old v7 Q8 max 0.042550772 / argmax 0.975 failed on all
backends (`v5-hard-crop-f32-q8-*`), so old four-input PASS cannot close the new
gate. F32 passed all five inputs on all backends. The 36-file current inventory
was `ocr-candidate-inventory-v16.json`; no public asset was automatically added.

Historical detector graphs used max-side 320, while default pages use 960.
All 15 detector graph scenarios were moved explicitly to 960; release probes
require 960 and reject 320/640 performance substitution. Old graph PASS does not
cover new input sizes. CPU 960 first pass was 14 PASS / 1 FAIL: Small F16 mean
0.000101013 exceeded 0.0001, while stored-source max was only 0.000008106,
identifying storage error. v17 keeps `conv2d_122.w_0` F32, SHA
`73efc76d8b6427363b19c5a88ecf7075febfd5bbbef0b5b7ff67ac4d8f5330c1`.
CPU graph max/mean was 0.010192215/0.000082143, all five scene deltas zero,
receipt maximum degradation 0.000733810: `filter-v17-cpu` 3/3; replay identical.
The 36-case default-size matrix later gave 35 PASS / 1 FAIL on every backend,
only old Tiny Q8 failing. New Small F16 GPU scenes/receipts each passed 2/2
(`filter-v17-pages-{cuda,vulkan}`). Public-current graph/layout rows passed
15/15 per backend under their historical budgets, not the stricter new release
gates. Twelve public files passed cold download/SHA/offline preflight at
`public-cold-download-0zhcaspv/report.json`, excluding new weights.

Reacquired original Paddle v4 det/Chinese-rec/English-rec parameters matched
public-GGUF source SHAs. Regenerated probabilities differed by at most
0.000016630/0.000000119/0.000000358; recognizer argmax matched exactly.

| Recipe | Fifty-page set | Hmean | Fixed-region CER/WER | E2E CER/WER | Maximum source degradation |
| --- | --- | ---: | --- | --- | ---: |
| v4 Q8 v8 | Scenes | 0.402332 | 0.157105 / 0.348039 | 1.240071 / 1.313725 | 0.004902 |
| v4 Q8 v8 | Receipts | 0.888528 | 0.270009 / 0.619240 | 0.363424 / 0.676187 | 0.000526 |
| v3-Japanese Q8 v8 | Scenes | 0.402332 | 0.659312 / 0.901961 | 1.235658 / 1.357843 | 0.004644 |
| v3-Japanese Q8 v8 | Receipts | 0.888528 | 0.579618 / 0.923953 | 0.587812 / 0.945681 | 0.000702 |
| v5 F16 v12 | Scenes | 0.377465 | 0.184466 / 0.406863 | 1.246249 / 1.362745 | 0.004902 |
| v5 F16 v12 | Receipts | 0.837987 | 0.274992 / 0.631155 | 0.403291 / 0.714211 | 0.000000 |
| v5 Q8 v14 | Scenes | 0.376404 | 0.183583 / 0.406863 | 1.245366 / 1.357843 | 0.001060 |
| v5 Q8 v14 | Receipts | 0.837038 | 0.275026 / 0.630979 | 0.405197 / 0.715437 | 0.001605 |
| v6 Tiny F16 v15/v16 | Scenes | 0.349030 | 0.176523 / 0.397059 | 1.338041 / 1.470588 | 0.004902 |
| v6 Tiny F16 v15/v16 | Receipts | 0.856725 | 0.245393 / 0.489224 | 0.374628 / 0.609077 | 0.000350 |

CER/WER is lower-is-better and can exceed one under mixed word/line granularity.
Maximum degradation is the maximum of five independently gated deltas; averages
never hide failure. These are not leaderboard accuracies.

Tiny Q8 scaling/block least-squares/block weight-MSE/single-kernel and transposed
linear-storage diagnostics failed the four-input budgets without changing the
formal quantizer/gates. v19 instead used explicit conversion-time activation
feedback from 256 pinned annotated train crops from PaddleX, observing original
ONNX activations and emitting ordinary Q8 blocks. Detector SHA
`fa5f6f6efdc79c4a2363c7719d6dd5d7ca999e14cbc71132a65d30ce834c4988`,
recognizer SHA
`f9e6c3793c95fa7e84b10d5fa277f56e22b059c4e1517ba93a223f83fa41d960`.
`filter-v19-cpu` passed four graph/scene/receipt checks; scene degradations were
nonpositive, receipt max 0.000208960, conversion replay byte-identical.
CUDA/Vulkan passed the same four each in `filter-v19-pages-{cuda,vulkan}`.
Detector Q8/F16/F32 ratios are 0.23903%/98.50118%/1.25979%; recognizer ratios
0.57945%/14.95273%/84.46782%. These are not pure Q8 or evidence of speedup.
Experimental policy SHA was
`3f8700eb6ab5672f396ff3c93b73c649ca480de4ec59cb0d2c14b3369d22ccc4`;
product defaults/public catalog/cache were unchanged. Full page combinations,
controlled performance and publication remained open.

GPU performance records native/original-CUDA graph latency, p50/p95, sampled
process VRAM and external activity. Sampled high water is not exact allocator
peak. Three old/new classifier attempts were activity-rejected. Original CUDA
I/O-binding smoke differed from CPU by max 0.000016362; shared-host time is not
controlled evidence. `controlled-cls-f32-current-cuda/matrix.json` also failed
source-phase activity despite some valid native samples. Own page queues resumed.
Audits now retain unique attempt IDs/raw JSON even on idle timeout; repeated
runs no longer overwrite telemetry. Fast tests passed runner 96, GT seven,
probability eleven, GPU audit eight.

Runner identity tracking later fixed executable/Python, runtime libraries,
CMake and parsed-manifest byte SHAs before execution and checks changed/added/
removed files afterwards. It is not a complete loader trace. Runner tests then
passed 98/98; `artifact-provenance-cpu` passed with
`execution_artifacts.unchanged=true`. Old processes do not inherit new identity
gates. New uploads remained zero.

## 2026-10-19: Published weights against original Paddle

Program/parameters/dictionary-YAML SHAs are separately pinned. A config archive
supplies YAML only; re-exported archive weights cannot replace published-file
source weights. `ocr_paddle_sources.json`/manifest register inputs; cold download
and offline verification passed. `run_ocr_paddle_probe.py` runs Paddle 3.0.0
in a normal-exiting independent process before native comparison.

`public-paddle-graphs-v4-{cpu,cuda,vulkan}/matrix.json` passed 9/9 each, 27/27 total.
Recognizer timestep argmax matched exactly. Historical CPU max probabilities:

| Public role | F32 | F16 | Q8 | Qualification limit |
| --- | ---: | ---: | ---: | --- |
| `ppocr-det-ch` | 0.000022560 | 0.061641514 | 0.490858376 | Original CPU graph, max and mean budgets |
| `ppocr-rec-ch` | 0.000006616 | 0.000072360 | 0.000119507 | Original CPU graph, exact argmax |
| `ppocr-rec-en` | 0.000000358 | 0.000573993 | 0.003898740 | Original CPU graph, exact argmax |
| `pplcnet-layout` | Original graph passes | Strict argmax FAIL | Strict argmax FAIL | Eight-head gate below supersedes old budget PASS |

Public pages use pinned PaddleOCR/PaddleX 3.0.1 and Paddle 3.0.0. Fixed GT quads
use Paddle's public quad recognizer, not X-AnyLabeling's selected-region bounding
rectangle. Six difficult crops passed at
`public-paddle-crop-contract-54p0tts4/result.json`; Chinese F32 50-scene pages
passed all five zero-delta checks under
`public-paddle-pages-v2-cpu-0/probes/ocr/ppocr-public-ch-labelled-scenes-f32/`.
A later source-CPU heap-corruption exit stopped v2; keep
`public-paddle-pages-v2-aborted.json`, not an all-page PASS.

GPU-wheel exit double free was traced to inherited
`GLOG_log_dir=/sandbox/logfiles/`; Valgrind/GDB receipts are
`paddle-import-memcheck-tiy9rwq9/` and `paddle-gpu-exit-gdb-w6q4uuuu/`.
Removing GLOG/MLOG only in children restored normal inference/cleanup/exit,
without skipping destructors or ignoring status. Paddle 3.0.0 ignored the old
`FLAGS_allow_tf32_*` names. `NVIDIA_TF32_OVERRIDE=0` now precedes GPU library init.
`public-paddle-gpu-no-tf32-_ux2mkyf/result.json` max CPU-fixture differences were
0.000017583/0.000006557/0.000000238, recognizer argmax one, all exits normal.
This shared-GPU smoke is not performance qualification. Source benchmark,
libpaddle, dependencies and math mode are recorded, including in-run code checks.

`requirements-ocr-paddle-gpu-reference.txt` pins wheel SHA and dependencies.
Fresh `pip check`/three graph checks exited normally with elementwise-identical
outputs in `ocr-paddle-gpu-minimal-4s9ttxya/validation.json`.
Graph/pages share `ocr_paddle_environment.py`; page success is written by the
parent only after normal source exit. Cached source references are reusable only
after rechecking code/library/dependencies/language/threads/evaluation/config/
GT/full coverage/per-image SHAs. Every native dtype/backend still runs itself.
Seven completed original-source families eventually included held-out SROIE.

Dependency enumeration had let vendored `typing_extensions 4.12.2` overwrite the
actual 4.16.0 selection. `ocr_paddle_metadata.py` records selected normalized
versions, every candidate and its own SHA. v3 evidence retains its old identity;
it is not retroactively upgraded. Source-reference tests passed 27/27.

### Published Q8 page failures

| Composition | Hmean degradation | Region CER degradation | E2E CER degradation | Verdict |
| --- | ---: | ---: | ---: | --- |
| Chinese Q8 | 0.002077406 | 0.000882613 | 0.009708738 | FAIL, limit 0.005 |
| English Q8 | 0.002077406 | -0.003530450 | 0.005295675 | FAIL, limit 0.005 |

`public-paddle-pages-v3-cpu-ppocr-public-{ch,en}-labelled-scenes/matrix.json`
consumed exact release SHAs. F32/F16 passed their 50-page groups. Source deltas
are not competition scores. `public-q8-stage-attribution-194al707/completed.json`
reran all 50 pages with one changed stage: Q8 detector/F32 recognizer e2e CER
increased 0.011473963 for both languages; F32 detector/Q8 recognizer passed.
That identifies detector storage for this scene failure, not every dataset.

`controlled-classifiers-current-v2-cuda` still failed baseline idle after own
queues were drained: GNOME Shell used about 27% SM through 265 wait samples.
The cohort was interrupted with raw audit and `interrupted.json` retained; own
queues resumed in finally. External jobs and idle thresholds were untouched.

### Multi-output layout gate

`run_ocr_paddle_layout_probe.py` pairs four class/four regression heads by unique
shape and exports all outputs from the same forward. Program/parameters/config
are pinned; cold downloads verified twelve public models and ten layout inputs,
then offline reuse. `public-layout-original-zfzv7xec/{cpu,cuda,vulkan}-v2/matrix.json`
had F32 PASS and F16/Q8 FAIL on every backend:

| File | Max error | Mean error | Worst-head argmax | Strict gate |
| --- | ---: | ---: | ---: | --- |
| F32 | 0.000015497 | 0.000000417 | 1 | PASS |
| F16 | 0.030360460 | 0.000918314 | 0.992307692 | FAIL |
| Q8 | 0.499403417 | 0.010272793 | 0.969230769 | FAIL |

Native GPU/CPU differences were below 0.000021 with identical argmax, so backend
agreement does not replace original truth. Original Paddle-GPU smoke max was
0.000020504, argmax one, normal exit (`public-layout-source-gpu-mn0d0bza/result.json`).
Injected real-graph NaN failed as required at `layout-nonfinite-mzdt15vc/result.json`.
Only isolated probes were relinked; active binaries/cache stayed frozen.
`layout-rounding-attribution-wd1mprzf/report.json` tested eight interventions
from bit-identical source F32 weights: norm-only F16 already flipped winners;
retaining norm F32 still had worst agreement 0.997894737. Keeping depthwise or
output heads alone also failed. These are fixed-fixture diagnostics, not repairs.

## Current verifiable conclusions and subsequent evidence

| Scope | Evidence | Still required |
| --- | --- | --- |
| Public det/ch-rec/en-rec, three dtypes | `public-paddle-graphs-v5-lf69oukv`: 9/9 per backend with selected/candidate metadata | Full pages/release stability/controlled A/B |
| Layout | Four independent eight-bin DFL edge argmax/softmax checks plus channel argmax; F32 PASS, quantized FAIL | Precision repair/GT/performance |
| 36 new files | `current-v19-graphs-6qbolt_5/cpu/matrix.json` 36/36; first CUDA 16 plus retry 20, Vulkan retry 36 | Initial VRAM-skipped INCOMPLETE retained; developer is not release |
| SLANet graphs | `current-table-graphs-aenvh0zm` 3/3 per backend | Separate TEDS/TEDS-S and controlled performance |
| Public detector single-layer attribution | `public-det-single-layer-pages-cmchzcrs/completed.json`, 14 layers/two languages, four F16-retained candidates pass scenes | Selected on existing regression pages, not independent validation |
| Public Q8 receipts | `public-det-layer-sroie-07gsu_x2`: Chinese repairs pass; English region/e2e WER +0.006833713/+0.008936394 fail | English recognizer repair |
| English F32/F16 controls | `public-sroie-en-cuda-controls-aernoeca/matrix.json` 2/2; F32 deltas zero; F16 WER +0.000175223/+0.001051340 | v3 metadata limits and performance remain |
| English-rec single-layer attribution | `public-rec-layer-sroie-_7968rya/completed.json`: only extra F16 `conv2d_161.w_0` among eight passed, WER -0.005431926/-0.000350447 | With det `conv2d_397.w_0`, old-identity GPU six pages/CPU scene pass; independent receipts/new identity/performance required |
| Initial GPU retries | `current-v19-graphs-retry-ovrm7450`: CUDA 20/20, Vulkan 36/36 | First CUDA 20 skips/Vulkan 36 unexecuted retained; no VRAM/audit bypass |
| CPU Medium pages | Some XFUND runs timed out at 3600s, rc124; single-page 8/4/2/1 threads about 57/20/19/25s, 44 boxes identical | CPU queues interrupted with receipts, not a GPU gate |
| Tests/publication | Source identity/head/failure tests 31/31; runner 98/98 | New uploads zero |

Page metadata now records selected versions, duplicate candidates and helper SHA,
rejects old replay identity and requires unchanged code/model/environment before/
after plus integer exit zero. `public-pages-v4-18iwva6p` source families passed.
Its CUDA and Vulkan playback each had 24 PASS / 4 FAIL, unchanged execution IDs:

| Published Q8 flow | Failing delta |
| --- | --- |
| Chinese scene | E2E CER +0.009708738 |
| English scene | E2E CER +0.005295675 |
| English SROIE 000–049 | Region/e2e WER +0.006833713/+0.009286841 |
| English SROIE 050–099 | Region/e2e WER +0.006207675/+0.006395786 |

Two repair SHAs passed all seven combined-page scenarios on each GPU without
held-out retuning. Held-out Chinese region/e2e CER deltas were
-0.000279779/+0.001259005, English -0.000384696/-0.000104917; Hmean delta
-0.000751945 both. The SROIE 050–099 split pins 100 image/GT files before
inference, disjoint by filename from 000–049. GT digest:
`8f085fedd59d4421352b4e1b37839f55515b74f75a89eae183fd810b03ecdc47`.
This contributor split is not an official competition test and does not prove
absence of model-training overlap. Common-runner cold preflight downloaded it.
Extension Vulkan pages later passed 48/48 in
`current-v19-pages-vulkan-0/matrix.json`, unchanged build identity. CPU timeout/
interruption records are not upgraded.

The 51-file index plus two repairs separates exact SHA, storage ratios, backend
pages, missing gates and weight-matched historical results with incomplete build
identity. See [OCR_RELEASE_GATES.md](OCR_RELEASE_GATES.md). v3 metadata gaps and
older library results cannot be upgraded to current/release evidence.

Native OCR/qAnyLabeling regression passed 35/35, then 12/12 after crop repairs,
without skips, only within implemented scope. Source-F32 formal consumers are
six presets plus three window paths per backend. Point-cloud comparisons are
15/15 scene pixels and 87/87 gestures, excluding full chrome/fonts/DPI.
Overall `complete_upstream_parity=false` remains: release, quantization, remaining
OCR/Chat/VQA/VLM workflows, full visuals and native packages are incomplete.

## Remaining sequence

1. Finish original probability/argmax, independent page metrics and controlled
   CUDA/Vulkan latency/throughput/VRAM/stability for each exact required file.
   CPU remains diagnostic under the later user instruction.
2. Upload only qualified files to `PP-OCR`, verify actual release SHA, register
   catalog/manifest/digest and test cold download. Upload authorization exists.
3. Finish dedicated OCR/VQA/Chat source workflows and screenshot/interaction/
   migration comparisons.
4. Complete exact local-model consumers and native package acceptance.
   Later user-authorized large-model waivers apply to llama compatibility scope,
   not to OCR release precision/performance gates.

### 2026-10-19: OCR document window

`ppocr-window-build-182bxflk` tests real Qt multipart submission, nested job IDs,
JSONL results, API error/cancel/timeout/restart, duplicate block IDs, unknown
fields, atomic save, favorites/reopen and actual menu/Ctrl+4 window reuse.
No paid external API is invoked; mocks are not official-service/full-visual tests.
`QT_USE_QSTRINGBUILDER` was enabled to match production Qt5 and fix temporary
string/byte-array expression lifetimes. Window/service, plugin entry, video UI
and dataset-export groups passed.

Document ZIP export preserves page/block order and revised text, generates
source-compatible `doc_0.md`, `imgs/` crops and `layout_det_res_N.jpg`, and reuses
the streaming ZIP writer without changing video top-level rules. Independent
Python checked CRC/JPEG dimensions/content; a two-page revised-Chinese/repeated-
image fixture matched actual upstream renderer text/names. Layout-font/JPEG
pixels are not yet equivalent. Custom API URL, image/PDF filters and sort order
are saved and consumed by subsequent submissions; real settings/HTTP mocks
cover in-flight record switches, cancellation and reparse. Precise deadline/poll
timers avoid early extra polling. Four groups in
`native-integration-results.json` returned zero. Native PDF/formula/remaining
visuals and real-service/native-package acceptance remain separate.

### 2026-10-19: VQA export, migration and templates

Native plugin tests cover component JSON import/export/sort, two-image title
rename, positional radio rename, checkbox option deletion/reset, component
delete and preservation of unrelated fields. Types remain immutable in edit;
text components disable option input. Every pending sidecar is parsed before
modification. Corrupt later sidecars prevent current changes; saves are atomic
per file but do not promise directory-wide rollback on disk failure.
Folder JSONL handles selected/renamed fields, unlabeled defaults and duplicate
columns in source overwrite order. Real `VQADialog.export_labels` and four
migration methods match two-page fixtures, with file UI substituted only.
Evidence: `vqa-export-reference-fpckst1b/{comparison.json,migration-comparison.json}`
and `ppocr-window-build-182bxflk/chat-integration-{result,commands}.json`.
VQA source SHA: `31189935ffcfcba8b6b3209db59560769bdee4f5492fff4ddae86d9247a36749`.

Eleven system templates are read-only; user add/double-click-edit/delete/exclusive
selection and all default names/prompts match `vqa/config.py`, SHA
`02a95ae49e2623f51fe8c95b78438c1dfb515a5c3841525a5fcfdb06f5305081`.
Components/prompts use upstream `xanylabeling_data/vqa/{components,prompts}.json`
under the home directory, ahead of old QSettings. Reopen/corrupt-preserve/explicit-
import recovery tests pass. ShareGPT export uses the shared 64KiB streaming ZIP
writer; four turns, `images/second.png` and image SHA match the actual nested
source exporter. Corrupt annotations fail and preserve the prior ZIP rather
than silently skip. Only local image basenames are supported; subpaths/remote
images report errors. Upstream has no Alpaca export.
Evidence: `vqa-export-reference-wemgqolp/`, including ShareGPT comparison/ZIP;
Chat source SHA `541936f68f9d0f0a218f81320cd10d9819f5412abd1089cc2eb4d3295c19578c`,
plugin SHA `a702b8dbba2844a47435c25672e8b4bbd80469c01654bd5d1a48171749179aca`.

CLS F32 CUDA release retry `cls-release-cuda-retry-4it5kq8o/matrix.json` failed:
first two candidate idle audits rejected; third native samples passed but source
GPU samples were contaminated. Vulkan `cls-release-vulkan-rnby3lq0` also failed
baseline/candidate idle. Partial cohorts cannot be combined into release PASS.

### 2026-10-19: VQA references/cancellation and Chat data

VQA reads actual current-task disk sidecars, including custom output roots,
ordered label/vqaData fields, nested Python str/repr, floats and Unicode14
escapes. It substitutes `@text`, `@widget`, `@label`, `@image` in source order.
Corrupt JSON blocks the request without overwriting; memory state is not disk
truth. Source-like prompt minimum width 600, margins24, input height120–160,
reference highlight/templates, 420x180 cancellable modal loading and preview/
apply were added. Image changes/cancel/close/Esc invalidate prior requests.

Bounded folder AutoRun keeps per-image history/error counts/cancellation epochs.
ShareGPT import preserves system/user/assistant and uses the first existing image
for sidecars; export preserves original images/multiturn messages. New sidecars
use source version 4.1.3. `vqa-export-reference-e_9wn16g/` records seven comparisons
from three source scripts in `upstream-comparison-results.json`; native plugin
SHA `5a45728107989e13ce0b2bade761c96fab76b0e83eef83d74b24cfaa436bb835`.
Four OCR/plugin/video regressions returned zero; strict plugin ABI boundary passed.
`vqa-python-repr-97_a3mxr/comparison.json` compared CPython3.11 across all
0x110000 code points/random binary64/nested values: 1,124,117 cases, zero deltas.
This establishes serialization, not inference/visual quality.

Layout diagnostics `layout-rounding-527jk1k0` (six) and
`layout-rounding-stage-u7tge9i8` (twelve) remain failed. Head-only F16 with
98.809% F32 parameters still had argmax 0.998421 and zero Q8; it cannot be called
a Q8 repair. No raw-head threshold was relaxed.

### 2026-10-19: Inline editors and provider configuration

Thirty-eight candidates were copied by exact SHA into persistent staging.
At 13:39 UTC extension page snapshot was CPU21/54, CUDA42/54, Vulkan54/54;
CPU/CUDA were still running. Two repair SHAs had all-backend graph/page developer
evidence; the English recognizer participates in only three of seven shared
page scenarios, not all seven. A 13:28 performance drain encountered a returning
external calibration job, so timing did not start, numerical queues resumed and
`gpu-queue-performance-pause.json` was kept. No valid controlled cohort/upload.

The inline table editor supports token/HTML/TSV, text/font, merge/split,
row/column edits, default3x3, clipboard and 120-state history. Formula editor
precedence, Python splitlines, and separate empty-table/paste undo steps match
source. All 2,231 HTML5 entities/numeric replacements are compiled into RCC/code,
without runtime external resources. Actual upstream parser/token serializer
comparisons cover 5,407 cases and native text/style/spans/two-round parsing;
Qt rendering objects were stand-ins, not full-window validation.
Evidence: `ppocr-table-inline-reference-hiodxzlw`. Native operations cover
paste undo/font/merge/split/rows/columns/branch redo; document Save/Cancel,
atomic save and unknown fields pass. Six source geometry methods, two theme
styles, icon toolbar, Cancel/Save order,16px margins, selection/right-click/
shortcut state and inherited palette match without changing host theme.
The result-block QFrame uses viewport/horizontal-scroll/20px bottom allowance;
six source height methods pass. Document/generation guards reject stale saves.

`vqa-export-reference-uy_4sw1l` covers eight provider defaults, nullable/unknown
fields, import/export/corruption, selected/favorite,45 capability cases and
1,060 searches using the actual SequenceMatcher source. Five source scripts
cover messages/ShareGPT/VQA. Delayed VQA destruction overwriting new Chat config
and Anthropic discovery headers were repaired. Four native integration groups
and Chat/VQA passed after relink, plugin SHA
`2714c02dd171715ae2a876928a0c2ee47156cd4c15086c73f8140eaefcbcd8a4`.

The 18-row SLANet page matrix covers three dtypes/two page paths/three backends
on100 real tables. CPU crop F32 TEDS/TEDS-S was0.9645975/0.9959180. A signal during
combined pages left INCOMPLETE; frozen SHAs were rechecked before resuming,
without claiming a confirmed signal cause. See `current-table-pages-4r6llgbn/`.
Remaining PDF/formula/rich-text/Markdown/math/local exact models/Qt6/native
packages are not closed by focused editor tests.

## 2026-10-20: Updated gaps and evidence

Later user direction makes CUDA/Vulkan the release backends; native CPU remains
diagnostic. Independent original frameworks, not backend agreement, define truth.

| Area | Current evidence | Limits |
| --- | --- | --- |
| Remote synchronization | Branch `5a6a40c06`, mainline `7bc6520bd`, HEAD `8db589b2d`; no conflict entries;289 unique scenarios | Merges/unrelated user work not pushed; later remote commits outside receipt |
| New real pages | Old runtime `154a0b62`54/54 each GPU, CPU diagnostic | Merged `9f2548ef` required new checks/stability/performance |
| SLANet | Old GPU12/12, CPU6; crop0.9645975/0.9959180, combined0.5431670/0.5917242 | New identity/release stability/performance separate |
| Original CUDA graphs | New36/36 per GPU; public10PASS/2FAIL per GPU | Layout quantized fails; developer time unqualified |
| CPU policy | Removed native-CPU accuracy requirement from48 graph rows;31 probe/100 runner tests pass | Original-source and GPU audit thresholds unchanged |
| Controlled performance | `gpu-release-20261020-py6cqztd` rejected three attempts/backend before timing | Clean controlled source/baseline/candidate sampling required |
| Rich text | Inline Markdown/HTML,H1-H6,format/save/cancel;13 serialization,6 heading,12 paste,3 crop,6 height,two themes,popup/icons/selection/scroll/delete | Full visual/formula/native PDF incomplete |
| Testing cleanup | Removed four obsolete environments/wheel caches/accepted temp payloads,16.87GiB freed; kept live envs/GGUF/logs/digests/deletion inventory | Never delete active inputs; cleanup is not model qualification |
| Packaging reassessment | Actual remote focal build/package/post-install passed;21 existing files restored and five new removed | Historical CI does not cover current uncommitted plugin changes |
| Publication | No new uploads | Full gates/catalog/SHA/manifest/cold download/local VLM/native packages |

Records: `cleanup-20261020/`, `gpu-release-20261020-py6cqztd/`,
`gpu-official-truth-20261020-5mjkehld/`, `gpu-paddle-truth-fixed-20261020-67514g3z/`
and `acceptance-20261020.json` under the OCR reference directory.
A Japanese XFUND F32 run had zero CER/WER delta but failed overall when formal
configure changed CMakeCache identity; preserve that failure. Fresh serial queues
must run against frozen builds. `merged-gpu-followup-path.txt` identifies them.
`merged-clean-installer-path.txt` points to a cancelled installer queue, not a
pending package. Native rich-text exports do not prove Qt5/Qt6 pixel identity.

### SLANet original graph and storage diagnostics

Official `SLANet_plus_infer.tar` SHA:
`68526c5c6573a8a17cdc14805e026f11c2545e42d89a1317bb58d3e1e4b58318`;
parameter SHA `012986b0e2bfe90410618bfb4b7cf4fcb6c978caefd84c63c2934f8387b09ab8`
matches published GGUF provenance. Original JSON/PIR outputs are locations and
structure probabilities; retain EOS, discard later padding. CPU first-input
max probability/location errors were1.847744e-6/3.576279e-7, exact structure.
`run_table_paddle_probe.py` uses independent original CUDA for GPU rows without
native CPU comparison, exports raw probabilities/locations and verifies sampling,
GPU audit/source timing and unchanged input/program/parameters/probe identities.
Twenty-seven tests passed; three existing `table-structure-*` consumers changed
without adding duplicate scenarios. See `table-original-graph-path.txt`.

Two predeclared held-out images exposed fixed-input layout overfit. F32 controls
matched argmax1; F16 candidate was0.9923076923/0.9994736842, another was
0.9923076923/1. The latter had only F32/F16, no Q8 tensors. Neither was promoted.
See `layout-tensor-rounding-path.txt` and `layout-rounding-holdout-path.txt`.

Isolated SLANet compilation with production definitions passed three CPU graph
inputs: max probability1.33365393e-6/location1.31130219e-6, argmax1. Earlier mixed
object/definition failures were retained. Formal CMake build later passed with
the same definitions. See `table-probe-isolated-path.txt`/`closure-report.json`.
Common-runner archive extraction also fixed leading `./` members: only pinned
members normalize, rejecting absolute/parent/symlink/normalized-duplicate paths.
102 runner tests and a real7,997,440-byte cold download passed, with three source
SHAs. See `slanet-common-archive-path.txt`.

Six GPU original-graph cases passed with three inputs each: F32/F16/Q8 CUDA
max probabilities2.32458e-6/0.00122018/0.0168662; Vulkan
0.000192106/0.00113231/0.0167495; all structure argmax1.
See `slanet-common-gpu-path.txt` and its two `matrix.json` files. This is
developer precision, not complete pages/stability/performance/release.

### Remote CI reassessment and packaging restoration

The previous inference from a stale local installation to a focal release defect
was unsupported and withdrawn after checking actual jobs/logs.

| Actual evidence | Conclusion and scope |
| --- | --- |
| [Ubuntu39273926284](https://github.com/Asher-1/ACloudViewer/actions/runs/39273926284), commit `5d14860893` | focal/jammy/noble Docker build/package/upload/test passed; focal Python845PASS/14skip |
| [focal installed artifact](https://github.com/Asher-1/ACloudViewer/actions/runs/39273926284/job/118215154142) | Actual .run installation at `/home/runner/ACloudViewer`;headless126PASS/1skip,JSON-RPC138PASS; host ubuntu-latest, not a pure minimal focal runtime |
| [CUDA39273926321](https://github.com/Asher-1/ACloudViewer/actions/runs/39273926321) | Three CUDA distro build/package/Docker/post-install jobs passed; not model GPU precision/performance qualification |
| Local Oct14 installation | Zero-byte executable/current plugin missing only proves stale local output is unusable |
| Additional local DESTDIR/space-path/dependency injections | Different conditions from CI `/root/install`/ordinary home; cannot establish same-condition focal failure |

Twenty-one existing paths were restored to synchronized HEAD: root CMake/
PostInstall/Linux packaging/CUDA bundler/launchers/IFW/platform helpers/plugin
toggles/CodeQL/macOS container/config/docs. New InstallStaging plus four packaging
tests were removed. Both index and worktree matched HEAD there; committed remote
updates were kept. Before-images/index/patches/job metadata/raw logs/excerpts are
under `packaging-reassessment-20261020/`, including `restoration.json`.
New local installer/container queues were cancelled. AnyLabeling uses AICore,
not independent qRAM/qOCR/qRelateAnything plugins; release-toggle expansion was
also reverted. Twelve old local packaging tests remain historical, not current
acceptance. Future worktree changes still need the existing CI.

### Shared OCR cache and integrity

`ocr_capi.h` and `src/tasks/ocr/model_catalog.cpp` own URLs/catalog/cache.
qOCR/AnyLabeling use public catalog plus shared `ecvModelDownloader` and
`ecvAssetIntegrity`; no private cache state machine is needed. Unpublished local
conversions cannot be registered as downloadable before qualified upload.
A fresh space-containing `CLOUDVIEWER_DATA_ROOT` returned the same OCR/table C-ABI
path, downloaded public F32 detector4,857,440bytes with SHA `aaad25f6…e06b11b9fd`,
reused offline without download and rejected offline tampering. See
`ocr-shared-cache-cold-path.txt`; this covers one public file, not unpublished
weights or plugin download interaction.

Real Qt cold-cache testing then found a bug: a modified previously SHA-verified
file invalidated its ledger, but deleting the ledger let header/size-only checks
accept it next time. Shared integrity now retains stale verification evidence
and repeatedly rejects quick hits until explicit successful `DeepVerify` refresh.
Warm valid caches still use stat, not a full hash each query. qOCR `modelPresent()`
uses shared integrity instead of size only; AnyLabeling already did. Legacy
files with no evidence follow the existing miss policy, not a false SHA claim.

The old failing receipt is preserved. New Qt tests passed cold public-catalog
transport/SHA/ledger, warm no-transport reuse, tamper rejection and explicit
recovery, with unchanged executable/libs. See `ocr-qt-shared-cache-before-path.txt`
and `ocr-qt-shared-cache-path.txt`. Regression covers repeated quick query,
empty-pin requests, persistent rejection after failed deep verification and pin
updates. Shared/TRELLIS compatibility plus seven AnyLabeling CTests passed9/9;
AICore/OCR probe SHAs unchanged. `merged-clean-package-path.txt` identifies
`build-result.json`, `tests-result.json`, `tests.log`; its historical directory
name does not mean a package was built.

### Chat/VQA automatic model discovery

Existing config and `ChatServiceClient` now cover initialization/provider switch/
menu opening. Shared in-process provider counters match actual source behavior:
Ollama queries each time, others use source `get_models_data` cache/timer rules;
manual refresh bypasses. A separate instance of the same client avoids cancelling
chat, writes existing `models.json` and uses source5s timeout without new caches.
`EventTracker.reset(provider)` sets timer0, so after the first300s interval,
later calls keep refreshing rather than implementing recurring cooldown.
Fifteen controlled time/cache/case/provider comparisons matched real Python;
native HTTP mocks cover forced/automatic/cache/switch/concurrent chat and async
empty-menu fill. Source workflow plus formal2/2 passed, probe SHAs unchanged.
See `chat-model-refresh-path.txt`. Popup upward anchor/available-height matches
actual source placement; both themes/formal2/2 passed at
`chat-model-position-path.txt`. This does not close whole settings/message visuals.

### Layout flips and independent human page evaluation

Read-only original arrays gave CUDA/Vulkan identical F32zero, F16class22/DFL40,
Q8class253/DFL419 flips. Original max class scores at flipped positions were
at most0.00922464/0.0403007. Low scores and intermediate distributions are still
in the user's strict gate. See `layout-flip-margins-path.txt`; no weights/gates/
catalog were changed.

CDLA pins all1,000 official validation pages/8,711 human regions and the
2,165,389,258-byte archive in `ocr_cdla_validation.json`. Official conversion/
rasterization matches every category/bbox/area; positive COCO IDs avoid sentinel
zero. Source score/NMS remain0.5/0.5. Native typed borrowed-image/region ABI is
used; DB text threshold1 forces zero text boxes only to isolate layout on the
original whole image, not to claim full-OCR quality/performance. Seventeen
evaluator tests passed. CPU source1,000/native3-page checks were diagnostic;
common cold download2,000 file hashes/offline zero-download passed.
See `layout-cdla-source-path.txt`.

Original CUDA and six GPU/dtype exports later completed every page normally.
Source mAP50:95=0.8546702421; native F32 CUDA/Vulkan deltas
-3.951446e-7/-4.016570e-7, F16+6.319819e-5, Q8-0.0009133202.
Formal CDLA CTest passed. These close page measurements, not the strict argmax
failure, reviewed release tolerances/public consumers/performance. Original-CUDA
selection repairs passed184 production-script tests; GPU rows run independently,
never inheriting CPU source qualification.

### 2026-10-20: Chat layout, messages and VQA interactions

Chat adds three-column widths/stretch, preview/settings splitter, Backend/
Generation, temperature slider/model popup and21 source light/dark QSS pairs.
Actual theme/provider colors, dynamic QTextEdit height,Enter/Shift+Enter,
@image highlight, shared send/stop, empty-input disable and default-No clear
confirmation pass. Native workflow,nine source construction/theme comparisons,
six formal CTests passed; frozen OCR identities unchanged.
See `chat-three-columns-path.txt`. Markdown/code/math/font/DPI/full visuals open.

Message source styles expanded to33 pairs: user bubble, assistant icon, actions
and editor. Context-menu order,one-second copy feedback and existing persistence
paths match. Theme colors, image-switch teardown while menu open,copy/edit/timer
pass native tests plus12 source comparisons/six formal tests;28 OCR dependency
SHAs unchanged. See `chat-message-source-path.txt`.

VQA adds1-based page navigation, disabled end controls,empty-input restoration,
upper-bound clipping, persistent sidebar collapse,default-No current clear and
left/right refresh. Existing image-load code rereads external JSON while keeping
unknown fields. Native tests match actual source page/clear/value/save methods;
six formal tests pass,28 OCR SHAs unchanged. See `vqa-navigation-source-path.txt`
and `vqa-refresh-source-path.txt`. Full component/popup/font/Qt-version visuals open.

Fourteen additional light/dark QSS pairs bring Chat/VQA total47, applied to file
labels,image containers,preview,pagination,splitters,navigation/actions with
source spacing. A palette-before-clearing-QSS ordering bug restored stale dark
colors; a native minimal reproducer confirmed it and the order was corrected.
Chat reads workspace theme. Actual fourteen-widget QSS/source execution,
selected colors/workflows/six formal tests passed,28 OCR SHAs unchanged.
See `vqa-styles-source-path.txt`; full Qt6/window equivalence is not claimed.

The latest original-CUDA graph/orientation snapshot was128/132, with the four
layout-quantized failures retained. `layout-backend-attribution.json` confirms
identical two-GPU flip positions and max delta<0.000021;
`layout-decision-confidence.json` records1/2 threshold crossings.
Eight controlled retry preflights retained source CUDA,old baseline,3/2/10
sampling/audit and separate repair SHAs; matching old SLANet probe was still
missing at that historical point. See `gpu-controlled-retry-path.txt`.
Subsequent work and exact current release state are in
[OCR_RELEASE_GATES.md](OCR_RELEASE_GATES.md); queued work is never PASS.

## Historical evidence locators

These exact paths supplement the abbreviated references above. They identify
historical local evidence and do not imply current release qualification.

- `build_app/Testing/ocr-storage-reference/acceptance-20261020.json`
- `build_app/Testing/ocr-storage-reference/layout-backend-attribution.json`
- `build_app/Testing/ocr-storage-reference/layout-decision-confidence.json`
- `build_app/Testing/ocr-storage-reference/packaging-reassessment-20261020/`
- `build_app/Testing/ocr-storage-reference/ppocr-window-build-182bxflk/`
- `build_app/Testing/ocr-storage-reference/vqa-export-reference-wemgqolp/`
- `core/AICore/tools/ocr_cdla_validation.json`
- `pointcloud/camera.py`
- `xanylabeling_data/vqa/components.json`
