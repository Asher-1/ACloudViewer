# OCR per-file release gates

Latest assessment: 2026-10-28. **Complete current qualification remains open.**
The 2026-10-25 Plan B record below reports 36 published files plus one
replacement; the pre-publication zero-upload snapshot is historical, not the
current publication inventory. This continuation performs no upload or model
replacement and does not revoke or overwrite any historical receipt.

The current OCR Vulkan matrix still has seven original failures; the original
CUDA identity veto and Vulkan conv112 original-owner baseline veto remain.
Shared pool/table success does not qualify all OCR pages or source-speed gates.

The later LLM CPU `build_app/Testing/llm-raw-formal-cpu-ref-v2/report-cpu.json`
contains real 3x2x10 samples for three scenarios on both original-baseline A/B
sides. Correctness/stability pass, but its verdict is FAIL: 256M Q8 decode p50
97.441397 -> 104.047370 ms (+6.7794%, +6.606 ms). Selected CUDA
`llm-raw-formal-cuda-ref-v3/report-cuda.json` passes three newspaper scenarios
only, at revision `e659cd3b`; it does not clear the original full release.
The initial exit-77 receipts remain INCOMPLETE and are preserved.

Focused real Qt5.10/Qt6 consumers and runtime-copy/install contracts are separate
from model accuracy. Native Windows/macOS packages, Apple Metal execution,
complete Markdown/code/math and full-window UI qualification are still open.
Keep this ledger and `UPSTREAM_PARITY_AUDIT.md`: neither is a completion report
that can be removed while its release/GT/platform failures remain unresolved.

## 2026-10-29: current execution observation

### 原完整矩阵最终结果

`build_ocr_validation_snapshot_20261028_v4/Testing/full-vulkan-v2/report.md`
最终为 FAIL：168 场景，161 PASS、7 FAIL、0 SKIP、0 UNSTABLE；原 48 模型、
241 输入和 developer 2×1×2/threads=0 协议不变。运行后 521 个执行名称身份
验证和旧收据保护通过。原 v3 的 FAIL/发现路径缺陷收据不改写。

失败场景为 tiny-rec-onnx-q8_0、tiny-labelled-scenes-f16/q8_0、
tiny-xfund-zh-recognition-q8_0、tiny-orientation-q8_0、public-en-SROIE-q8_0、
heldout-en-SROIE-050-099-q8_0。orientation 两次 rc=0，但 failures=2 违反
原 failures==0 结构化门，其余六场景两次 rc=1。完整执行不等于精度门通过；
七项算术根因仍待原始 probe 数据调查，不放宽 GT、阈值或移除模型。

### 原七项 FAIL 的指标归因补充

以下直接读取同一冻结 v4 `report.json` 与其 `probes/ocr/` 原输出，不改历史
报告。所有 labelled/region 的退化上界仍为 0.005；指标现象不是已证实的算子
根因，不能用改善的 CER 遮蔽 WER FAIL。

| 原失败场景 | 已核对的失败量 | 已排除/仍开放的边界 |
|---|---|---|
| tiny-rec-onnx-Q8 | 原 source max_diff=0.890249522、argmax=0.975；四 fixture 仅 1 被 probe 判通过 | stored_max_diff=1.31726265e-5、stored_argmax=1、4/4 完整；不是缺输出或 backend 缺席。冻结 wrapper 的独立 CUDA benchmark 分支仍以原 source <0.001/exact argmax 拒绝，不能静默取消该原门 |
| tiny-labelled-F16 | Hmean delta=0.005540166、e2e CER delta=0.006178288、e2e WER delta=0.009803922 | 固定 GT 区域 CER/WER delta=0；须追踪检测/完整流水线差异 |
| tiny-labelled-Q8 | e2e WER delta=0.009803922、固定区域 WER delta=0.009803922 | Hmean delta=0、CER delta=-0.003530450，改善不清除 WER 失败 |
| tiny-XFUND-zh-Q8 | WER delta=0.018737944 | CER delta=0.001646330 在门内；固定字段识别仍须定位具体 token 差异 |
| tiny-orientation-Q8 | failures=2，原 `test_ocr_orientation.cpp:95` 两次 text equality FAIL | 原 `.log` 没有 classifier 概率/旋转阈值 FAIL；不能把它误归为两个方向决策失败。wrapper rc=0 仍被结构化 failures==0 门拒绝 |
| public-en-SROIE-000–049-Q8 | 固定区域 WER delta=0.006833713、e2e WER delta=0.008936394 | CER delta=0.001270946 在门内；模型/GT 不改 |
| heldout-en-SROIE-050–099-Q8 | 固定区域 WER delta=0.006207675、e2e WER delta=0.006583898 | CER delta=0.000734420、Hmean delta=-0.000751945；仍为独立 held-out FAIL，不能用于调参 |

图级原输出为 `probes/ocr/ppocrv6-tiny-rec-onnx-q8_0/ocr-ppocrv6-tiny-rec-onnx-q8_0.json`；
方向失败日志为 `probes/ocr/ppocrv6-tiny-orientation-q8_0/ocr-ppocrv6-tiny-orientation-q8_0.log`。
冻结 `core/AICore/scripts/run_ocr_onnx_probe.py` 的量化诊断与 benchmark source 检查
是两个实际分支：同存储权重一致性通过不清除后者的原独立精度失败。此次没有
修改 wrapper、模型、阈值、source pin 或用 developer timing 作为性能资格。

Relate 后续独立候选已完成 CUDA 编译/链接、三后端各 256 数值和七路由 GREEN；
host 独立链接及 28 个同权重消费者字节对照完成，但四项 Vulkan p95 原门 FAIL、
large-F32 缺失和三源/冻结 release 门仍开放，详见 Relate owning record。

LLM scoped native schema/HELP/MCP/CPU text、repeat、two-image 与拒绝/错误路径
已有通过收据，仍为 release_qualified=false；live RPC 和原冻结 baseline/
性能 FAIL 均不由这些局部结果清除。既有 Canvas v12 报告 PASS，原 GL v13
启动失败于 Qt xcb 依赖，未执行 framebuffer 门；完整原生窗口验收仍开放。

以下保留较早的执行中观察，旧待机进度已被上述最终结果取代。

The corresponding-module continuation re-verifies current main config and
three-state review in `qal-config-current-v14.json` and
`qal-review-current-v14.log`; the model-free workspace test also exits zero.
Fresh `qal-chat-current-v15/` native fixtures pass six existing source comparators
and eight provider-comparator behavior tests. Its five recorded artifact hashes
are unchanged; local generation is explicitly skipped in an isolated empty data
root, not counted as model qualification. The v14 directory-launch failure stays.
`qal-ppocr-current-v14/` passes three native document/rich/table tests and three
source comparators with all three recorded probe hashes unchanged. None clears
independent OCR truth, source-speed, baseline or complete native-window gates.

Native logs retain inotify watch ENOSPC, independently reproduced by a successful
`inotify_init1` followed by a failed `inotify_add_watch` (errno 28). Disk has
244 GiB available; limits are 65536 watches and 128 instances. No sysctl/process
change is made. Native file-monitoring acceptance remains open.
Original OCR v3 is now at 133/168 (Medium SROIE F32); 130-132 complete both
repeats with rc=0, but final metrics and after identity remain pending. The
existing v4 snapshot-preparation queue is followed by the syntax-checked
`run_ocr_v4_after_prepare_v1.sh`: same lock, RAM/GPU admission, snapshot/protected
receipt verification, original real F32 consumer and then the existing complete
2x1x2/threads=0 launcher. It has not executed the preflight or matrix yet. LLM
native build/model/RPC and original GL/canvas queues still wait. Original models,
source pins, thresholds, frozen baselines and every historical FAIL remain.

The earlier observation below retains its original execution boundary.
Original OCR v3 reaches 128/168 (tiny SROIE F16); the preceding F32 scenario
completes both repeats with rc=0. No final verdict or after-run snapshot
verification is available. Existing GT, source-speed and baseline vetoes remain.
MemAvailable is about 34 GiB, swap used 5.9 GiB, and total OCR group RSS about
1.05 GiB. An external build is observed concurrently; the frozen matrix is
not stopped, but this does not establish quiet controlled performance evidence.
The session's -j1 RED-host build and -j2 scoped LLM build/models remain waiting.
Current autogen metadata still reports 16; the pending configure requests 2,
which must be checked after configuration rather than claimed already effective.

Fresh narrow checks pass: Python backend/CLI/MCP 209 tests, integration contracts
22 tests, OCR GT/thread-count and relation policy 40 tests plus 67 subtests, and
strict plugin boundaries. None establishes native/model/release closure.
RelateAnything completeness was checked separately: current upstream 537f48f
and historical 14c9744 have identical trees; DW7 modules exist, but relation host
hints/direct-bias and CUDA ADD/GELU fusion remain unported. The owning record is
[RELATEANYTHING_SYNC.md](../../../../3rdparty/llama/RELATEANYTHING_SYNC.md).
This observation changes no source pin, model, threshold, baseline or old FAIL.

The user subsequently approved the bounded RelateAnything host/CUDA-fusion
follow-up. Only test code is written: seven isolated real CUDA trace modes,
256 numeric cases and a matching-header/runtime test-only compile queue. It
waits on the original lock with 18000 MiB RAM/2048 MiB free GPU admission;
no real route RED or new production host/patch implementation has run yet.
Original OCR now reaches 130/168; tiny SROIE F16/Q8 both repeats return rc=0.
RAM remains about 34 GiB available, OCR group RSS about 1.10 GiB; this is still
not a final matrix/identity-after or original performance-release verdict.

## 2026-10-28: latest authorized scoped continuation

The main current-upstream config comparison now passes in
`build_app/Testing/upstream-cd26942-config-green-v8.json`; the original 15-error
receipts are historical and unchanged. Review three-state persistence and
Run/OCR review invalidation use the document owner. The real Chat theme RED in
`qal-chat-theme-diagnostic-v3.log` localized stale palette restoration when
clearing old QSS; clearing that old stylesheet before applying the next palette
fixes the workspace and dialogs without changing original color assertions.

The user separately approved strict source-provider equality plus explicit
`local` extension validation, and new-sidecar version 4.2.1 through the existing
single format owner. Eight comparator behavior tests pass after two real REDs;
the real Chat import regression also went RED to GREEN and verifies existing
explicit 4.1.3 metadata is retained. All plugin/test consumers rebuild at `-j4`
in `qal-version-consumers-build-v2.log`, and `qal-suite-version-v9.log` passes
39/39 with the original person/vehicle F32 assets and input digests verified.
Fresh native fixtures/screenshots in `qal-chat-upstream-current-v10/` pass all
six existing Chat/VQA source comparators. These are scoped workflow/QSS checks,
not a full-window pixel/font/IME/DPI/Markdown/code/math qualification.

The additional v4 `Testing/cpu-capital-isolated-v3/report.json` preserves the
original baseline/assets and release 3x2x10/threads=4. Correctness/stability is
2 PASS, but overall verdict remains FAIL: Q8 e2e p95 187.389 -> 211.106 ms
(+12.66%, +23.717 ms). All 451 snapshot names pass before/after SHA checks.
The observer records compiler/probe activity; a separately observed concurrent
OpenPCDet benchmark was not included in that observer's busy-process filter,
so this run is not fully quiet performance evidence. It neither diagnoses the
original F16 +8.77% as a code cause nor withdraws it. All complete three-backend
LLM failures, the seven original OCR accuracy failures, CUDA identity veto,
Vulkan conv112 original-owner baseline veto and native-platform gates remain.

### Later original-matrix continuation

The complete original CPU replay in the v4 snapshot's
`Testing/cpu-full-quiet-v4/report.json` is 12/12 correctness/stability, zero
skips, but overall FAIL: 500M F16 describe decode p50 3992.051 -> 4215.381 ms
(+5.59%, +223.330 ms). It retains 3x2x10/threads=4, the original baseline/assets,
and before/after 451-name SHA checks. A busy clangd process appears in its host
trace; this is not a proven fully quiet host or a code-cause diagnosis.
The prior capital +8.77%/+12.66% and all complete backend FAILs are retained.

Existing cuboid/document/shape_batch/tagging current-source comparisons pass
in `qal-v11-*.json`. Chat v10's four recorded artifact checksums match the
before receipt. These checks do not qualify full-window, font, IME, DPI, code
or math rendering. The user selected original matrices first and deferred
new rendering test expansion.

`build_app/Testing/run_ocr_original_matrix_v2.py` freezes current execution
files into a new `build_ocr_validation_snapshot_20261028_v2/` and invokes the
existing validator only: original full 168 scenarios, developer 2x1x2,
threads=0 and all four original source references. It freezes 503 names and
protects both old complete reports; it changes no model, threshold or baseline.
Early private-probe rows crash before backend selection. GDB in
`ocr-snapshot-crash-{backtrace-v1,layout-v2}.log` identifies
`OcrWeights::foldAffineChains` -> `memset(nullptr,0,64)`:
compiled probe reads tensor data at 0xf8, while the loaded library's
`ggml_get_data` reads 0x258. The stale private-ABI consumer is distinct from
the public-ABI layout rows that execute. The v2 matrix has finished: 39 PASS,
129 FAIL, zero skips; all 503 names verify afterward and protected original
reports are unchanged. The existing `test_ocr_runtime` consumer was rebuilt
at -j4 in `ocr-probe-abi-refresh-build-v1.log` (with deprecated ggml precision
API warnings). Its DW bias contract passes; `ocr-real-model-abi-green-v2.log`
also runs the real F32 rec graph: max_diff=0.000007, argmax match and slot reuse.
These consumer corrections are not full independent accuracy verdicts.
No quantized raw/source-benchmark veto, original text truth, source-speed gate
or frozen baseline is relaxed.

The v3 replay uses the clean original 745f6c35 reference clone rather than the
current cd26942a UI checkout. All 48 models, 241 input assets and 503 execution
names verify before sampling. The rebuilt private consumer passes model load
but its first twelve rows skip backend discovery: `load_backends_once()` resolves
the executable's `bin/aicore_tests/` directory, where this execution snapshot
omitted module aliases. The ONNX cls fixture log reproduces this exact boundary;
shared-library layout consumers still execute. V3 remains intact. The build-local
preparer now places internal aliases to the same frozen bin modules in a new v4
snapshot; real-model preflight is required before a further complete replay.

All three existing PPOCR comparisons pass in `qal-ppocr-upstream-current-v11/`
(pipeline, rich editor, table); the three recorded native probe hashes match
before/after. Five pointcloud comparators pass in v11, and correcting the
build-local wrong-target launcher yields v12 view/colors/mesh PASS with
214/32864/96 cases. The old launch failures remain. Pointcloud render v12 fails
before framebuffer capture: the installed swrast driver cannot resolve
`ffi_type_sint32@LIBFFI_BASE_7.0` through libLLVM-12. The process-local preload
of the existing system libffi is queued; there is no package installation,
Qt change or pixel-gate relaxation. These results do not close inference GT,
source-speed, original-baseline, full-window or native-platform gates.

### Resource-bounded continuation and approved LLM automation

The user approved an independent `LLM_GENERATE` adapter in qAnyLabeling using
only the existing public LLM C ABI and shared batch/schema/CLI/RPC/MCP chain.
`llm-agent-registry-red-v1.log` records six real Python discovery failures;
`llm-agent-registry-green-v2.log` passes the same six cases after adding the
single registry row. `llm-harness-regression-v3.log` subsequently passes all
209 tests in the backend/CLI/MCP files. This is Python-layer evidence only.
The later real L3 `llm-native-help-red-v1.log` and real MCP
`llm-mcp-native-schema-red-v1.log` both fail because the selected binary lacks
LLM_GENERATE, without loading a model. The L3 process max-RSS observation is
411512 KiB (not simultaneous process-tree RSS). `LLMBatchCommand.h/.cpp`,
plugin/CMake registration, the RPC whitelist and the 19th integration row are
now written; source contract checks pass 22/22 and the strict plugin boundary
check passes. None proves compilation or model execution. The -j2 scoped
plugin build and CPU real text/two-image/error tests wait in the memory-checked
`run_llm_batch_checks_v1.sh` queue; native/RPC/platform qualification is open.
The new real-plugin schema test is queued in `llm-native-schema-red-v1.log`;
it is not a completed RED or GREEN. It selects the real registered command,
checks model-free HELP and pre-load validation, and preserves attribute tests.

The original v3 OCR process is still running; the latest observed log reaches
119/168 (SROIE F16, first repeat rc=0). There is no final verdict or after-run
identity verification yet. Recent observations show 22-23 GiB MemAvailable,
6.3-6.4 GiB used swap and about 1.16 GiB summed RSS for the entire OCR process
group, not just its parent. New compilation/inference waits on the same lock;
the schema-test-only build uses -j1 and rechecks at least 12000 MiB available
memory plus absence of compiler processes after taking the lock. These are
admission checks, not a proof that an OOM is impossible. The existing
QANYLABELING autogen metadata also reports PARALLEL=16, outside make's jobserver.
The pending local configure now requests CMAKE_AUTOGEN_PARALLEL=2 as well as
-j2 compilation; actual generated metadata still needs post-configure checking.
Only this session's still-waiting build script was replaced before it performed
any configure/build/model work. Original OCR and other sessions remain running;
original outputs are neither stopped nor overwritten.

Direct report inspection narrows the tiny original GT failures. For labelled
F16, `val_img_59.jpg` has source/native detector matches 3/2 at unchanged IoU
0.5: source's `NOW` polygon matches while the native polygon does not, despite
both recognizing `NOW`. The page contributes two extra e2e word edits; the
aggregate is 301 versus 299 over 204 reference words. Detector totals are
62 versus 63 matches over the same 201 GT/160 predicted boxes, reproducing
hmean delta +0.005540166. For labelled Q8, fixed-GT `val_img_337.jpg` returns
`SpoOn`/`SOup` instead of source/GT `Spoon`/`Soup`, two case-sensitive word edits;
the aggregate is 83 versus 81 over 204. This recognition failure is distinct
from detector geometry and from Q8's separate e2e excess. Shared source/native
errors such as `MASA` -> `MKS` are not evidence of an added native regression.
No case folding, IoU, CER/WER threshold, model or baseline is changed; the
underlying arithmetic cause and complete original gates remain open.

## 2026-10-28: this round's re-verification and claim corrections

The first re-verification had ACloudViewer HEAD
`a624587d7af8f6b11c44339de55cdc0f8b85d860` with a dirty work tree; the later v4
snapshot and this round's render fix are at `660a39c2...`, and the corresponding
identities are retained separately below.
The user first chose evidence re-verification and then authorized only the fix of
the existing render differences; the new three-state review and the new canvas
mode remain deferred. No model, original gate or frozen baseline was changed, and
nothing was uploaded, committed, published or cleaned from the historical
receipts.

- **Complete published-set picker: fixed and re-verified.** The public catalog
  already has 48 rows; the plugin's two preset allowlists previously hid the
  source/v5/v6 F16/Q8 entries, and the single-F32 regression was a stale
  pre-release contract.
  `ocr-picker-catalog-red-v1.log` really reproduces the unequal source-rec set;
  after the fix `ocr-picker-catalog-green-v2.log` passes both items, checking all
  rec files per public preset with no duplicates. With the original F32
  person/vehicle assets present, `ocr-picker-catalog-suite-v3.log` is 38/38 with
  no skips. Family/URL/quantization come from the task catalog/presets, and the
  local F32 fallback applies only when the catalog entry is missing. This pass
  does not prove all det/rec/cls combinations or independent accuracy/release.
- **Complete upstream alignment: still no.** Running
  `upstream-cd26942-config-red-v2.json` against the actual
  `cd26942a007b6ce7bd521de801a6cfd455d7fd9b` FAILs overall: two review actions are
  missing, and the defaults, nine palette items and four panel styles are unequal.
  The three-state review, review/checked search, wheel default and the new
  tags/canvas behavior are not yet implemented; the old `745f6c35` alignment
  evidence cannot be promoted to complete current alignment.
- **Current upstream follow-up re-verification.** The 8 path/pixel priority cases
  of `upstream-cd26942-document-v1.json` and the 16 RAM display/filter cases of
  `upstream-cd26942-tagging-v1.json` pass; the original
  `upstream-cd26942-render-v1/report.json` FAILs overall with 19 original-gate
  errors, three of which are polygon geometry-only pixel differences; the later
  authorized fix and the GREEN results on three SDKs appear below, while the
  original FAIL is retained. These drawing comparisons are not OCR inference
  accuracy or full-window acceptance.
- **Current owner contracts and siblings.** `ocr-picker-sibling-qocr-v1.log`
  passes both items; the model-independent
  `evidence-continuation-capi-contract-v2.log` is 40/40. The broad capi log
  `evidence-continuation-capi-v1.log` retains 16 skipped items and one FAIL for
  the missing attribute asset/reference; after supplying the original F32/source
  reference, `evidence-continuation-attribute-v3.log` passes that single item, and
  the original 80-item diagnosis is not written up as a complete PASS.
- **Complete llama integration: still no.** The original frozen key runtime/probe
  SHA values agree with the old receipts; the current official probe is built, and
  `llm-current-full-preflight.log` only verifies 10 weights/projectors and one
  input offline and expands the 12 original rows of 3x2x10 with threads=4 -- it is
  not yet an inference result.
  The `build_llama_upstream_ref` of the old complete report is currently missing;
  the loader/device verification of the existing pinned pristine reference does not
  replace the original release receipts. External Bazel/PCL compilation polluted
  sampling, and the wait is recorded in `llm-external-build-wait-v1.log`: there
  were three consecutive compiler-free checks, but a new round of external
  Bazel/compilation appeared on re-check before sampling, so the complete A/B was
  not started then; the later actual execution results appear below.
- **Original OCR gates: still not all closed.** The seven FAILs of the original
  Vulkan complete 168 scenarios, the CUDA execution-identity veto and the conv112
  original-owner baseline veto below are all retained. A pool/table consumer PASS
  does not cover the layout-page composition (explicitly skipped in the original
  log) or all source GT.
- **Platform and UI: still open.** Linux build/test/schema/partial relocation does
  not equal complete native qualification; a complete
  installer/clean-machine/L4/live run, the Qt full window and macOS/Windows/Metal
  still have no complete native acceptance. The Markdown/code/math, native
  PDF/formula and exact model family gaps are listed in the upstream audit and are
  not turned green by this round's 38-item suite.

### 2026-10-28: the complete CUDA run executed, identity boundaries re-checked

- `build_app/Testing/llm-current-full-cuda-v1/report.{json,md}` has completed all
  12 original release rows at 3x2x10 with threads=4, and the SHA values of the 10
  models/projectors and the single input were all verified offline.
  The candidate's row-level correctness/stability is 12 PASS with no skips; the
  overall verdict is **FAIL**, not a complete release pass.
- All four original frozen baseline rows are rc=[1,1,1]: 500M Q8 describe, 256M
  Q8/F16 describe and Qwen Q8 poem. Their upstream text gate failures are retained;
  the baseline must not be replaced and the candidate's passes must not withdraw
  the original failures. There is also an independent `CMakeCache.txt` execution
  identity veto with `unchanged=false`.
- The reported pre-execution candidate `libAICore.so` SHA is
  `2c7cee1bbcf6f548e97f57beb4ab6c418557b56c0ddc56f794b6d290a933f0fe` and the probe
  is `04e2731ffadebd4469c7b26e66d0a59cc6c4dbaba75916aa0e83874b50d4e8c6`.
  The runner persists only the before state and the changed paths, not an after
  state or a transitive loader trace; the only changed path is the cache, so the
  difference between the earlier `7b5b28b7...` and this library must not be called
  a library swap during sampling.
- `llm-current-full-cuda-v1-host-v2.log` records the external Bazel and observed
  four additional compiler PIDs at 12:59:43; it is not proof of quiet sampling
  throughout. The library was updated again after the report ended: mtime 13:08:11,
  current SHA `d11d2dcc97e0f990a53ddf95b91f55decba386eba32c0b54c2f1075cba7ca3e6`.
  The current cache SHA is
  `555d5be5c807ef6d396b2a48b68bb162b42cbabd0e118d12161bb6d612695000`, and before
  the CUDA run it was
  `be22a5be800826eaed07a5ab5721384450df47d64e0eb527fdb85be468dee39e`.
  Later results must record their candidate identity separately and must not be
  merged into a single three-backend conclusion about one static build.
- The continued CPU/Vulkan `--list` runs each completed the same offline
  verification of 10 models/projectors and one input; this is preflight only and
  produces no inference report. After the external compiler exited and the current
  hashes matched twice, `llm-current-full-cpu-v2/` started but was aborted because
  the shared candidate changed again, and Vulkan was never started.
  The CPU log retains rc=0 three times per side for the first three rows and the
  KeyboardInterrupt after the fourth row's first baseline rc=0; no complete report
  was produced, so it is marked neither 12 PASS nor a complete runner FAIL, and no
  speed regression is claimed from it.
- During the CPU run this side's continuous hashing really measured the library
  change from `d11d2dcc...` before launch to
  `9e23f7fb287dedfb1765a6e7f53df097c7761377a13fb095e38f64bab52f2935` and then to
  `922a68a53d02f7e46a6f7f5c091c8ca2c487adef48bd6cfafcfdc4cb33aa0701`
  (mtime 13:26:24). Unlike the CUDA cache veto, this is a real library change
  during a CPU run.
  The current runner SHA also changed from CUDA's `44273d3c...` to
  `20702c6f223b8e86f6f0c5d73b84bc8e078183c6b545c57d2c77641738bee252`;
  the current sources must not be mixed with the identity of the earlier report.
- This side sent SIGINT only to the verified CPU runner PID 804498; the
  runner/observer have exited and the original `.log`, `-host.log` and probe
  outputs are all retained. Another process was then really found running
  `cmake --build .../build_app --target AICore bench_openpcdet_perf -j6`, and it
  was neither terminated nor overwritten.
  A brief compiler-free window cannot freeze shared artifacts; a later run needs an
  isolated candidate or coordination with the writer, with the original baseline,
  models, 3x2x10 protocol and original gates unchanged. This side did not build,
  change inference sources/patches/thresholds, commit, upload or clean any original
  receipt; those additional concurrent modifications are not attributed to this
  side's changes this round.

### 2026-10-28: user-authorized independent validation snapshot

The user chose the "independent validation snapshot", authorizing only a
build-local environment freeze and re-verification under the original protocol,
without changing the original
baseline, models, accuracy/performance gates or the UI behavior deferred earlier.
The snapshot is not a new baseline or release qualification.

- The preparer `build_app/Testing/prepare_llm_snapshot_20261028.py` only copies
  and verifies SHA values, reusing the official `CopyGgmlBackends.cmake` without
  changing ELF/RPATH or inference code; the source/reference libraries and the
  imageformats dependencies are all copied and the aliases point only inside the
  snapshot. The earlier preparation-failure logs and partial copies of v1 (Qt
  mixed loading) and v2 (system CUDA 11.1 missing `cudaLaunchKernelExC`) are
  retained. v3 completed the copy, but the catalog preflight failed because of 33
  empty OpenPCDet digests in a stale binary, and that gate was not bypassed.
- The current header already has the corresponding SHA values at 14:04 while the
  catalog binary was still from 13:26. This side rebuilt
  `test_catalog_dump_urls` with `-j4` in a check window free of external
  compilers, and both the build and the real `--json` returned 0; see
  `llm-snapshot-catalog-rebuild-v1.log` and
  `llm-snapshot-catalog-rebuilt-v1.json`.
  Those models/digests were not changed, only the stale consumer was refreshed.
  This is not OpenPCDet complete accuracy or release acceptance.
- The new snapshot `build_llm_validation_snapshot_20261028_v4/` was frozen at
  14:22:26 with 262 sources, 451 copies/aliases and 249 unique files, about
  3.2 GB. The complete copy SHA re-check passes, `ldd -r` on the main
  probe/reference shows no unresolved symbols and no fallback to the original
  build / Qt SDK, and the snapshot catalog `--json` returns 0.
  `snapshot-receipt.json` retains the source/copy SHA values and the loader text.
  It does not prove kernel/driver state, future host load or a complete installer;
  the models and the original baseline were not copied.
- The v4 library SHA is
  `54d0db4db26d17571a8a0c7326c7e94969982229f3a27e179b6af3d9c943aa16`, with runner
  `20702c6f...` and validation manifest `c0d03d46...`, and the observed HEAD was
  already `660a39c2d3610ec7ed2aeaea27ee7c1cf4772cde`; that external commit was not
  an action performed by this side.
  These identities must not be applied back to the earlier a624587d/CUDA or to the
  aborted CPU receipts.
- `run_llm_snapshot_20261028.sh` only launches the same original validator
  serially, with 3 processes, 2 warmups, 10 forwards and threads=4 per side; it
  also verifies all copy SHA values before and after each backend. The queue order
  is CUDA, Vulkan, CPU; the original FAILs are unchanged and the outputs and load
  records are in `llm-snapshot-full-three-backends-v1{,-host}.log`.
  The wait for the re-entering external Bazel at launch is a historical state; all
  three backends were subsequently really executed, with results below.

### 2026-10-28: the complete v4 three-backend receipts are collected, release still all FAIL

This round continued the original queue without restarting or overwriting the
complete matrix. CPU produced its final report at 16:04:44;
`build_llm_validation_snapshot_20261028_v4/Testing/{cuda,vulkan,cpu}-full-v1/report.{json,md}`
each contain the 12 original release rows at 3x2x10 with threads=4. The candidate
correctness/stability of all three backends is 12 PASS with 0 skips, 36 rows in
total; **row-level passes do not equal an overall release pass**.

| Backend | Complete report overall verdict | Original gates still blocking |
|---|---|---|
| CUDA | FAIL, runner exit=1 | The four original frozen baseline rows are still rc=[1,1,1]; additionally Qwen F16 capital e2e p95 +33.03% (15.653 -> 20.822 ms, +5.170 ms) |
| Vulkan | FAIL, runner exit=1 | Eight performance alarms covering 500M Q8 describe, 256M Q8 describe and Qwen F16 poem; 500M Q8 decode p50 +18.35% (601.330 -> 711.674 ms) and e2e p95 +30.13% (781.679 -> 1017.229 ms) |
| CPU | FAIL, runner exit=1 | Five performance alarms: 500M F16 newspaper decode p50 +8.49%, 256M Q8 describe decode p50 +5.69%, Qwen F16 capital decode p50 +8.13%, Qwen F16/Q8 poem e2e p95 +5.22%/+11.67% |

- The execution identity of all three reports is `unchanged=true`. The original
  queue re-checked all copies before and after each backend; after this round the
  preparer `--verify` was run again and the 451 names / 249 unique files pass.
  They reuse the v4 runtime `54d0db4d...`, manifest `c0d03d46...` and report
  revision `660a39c2...`, and do not inherit the identity of the earlier shared
  build or of the cache identity veto.
- The original scope of identity is still only the selected command files, the
  runner, the AICore/ggml libraries and the cache --
  **not a transitive loader trace and not a host load lock**. The host log records
  additional compiler/make/nvcc activity in sampling windows such as CUDA 14:42:56
  and CPU 15:26:18, so quiet sampling throughout cannot be claimed.
  The table above retains the original alarms; the CPU gate exceedances that have
  not been re-tested in isolation are not written up as confirmed code
  regressions, and load interference is not used to withdraw a FAIL. The next step
  is still an isolated re-test with the same snapshot and the same original
  baseline/assets/protocol.
- The current source HEAD has been confirmed as still `660a39c2...` with upstream
  `cd26942a...`; another session's OpenPCDet modifications/builds are not part of
  this round's changes. This round changed no model, original gate, patch or frozen
  baseline.
- The existing shared rich-text fidelity regression was check-built with `-j4`
  against the real `rich` target on both Qt5.10.1 and Qt6.5.3 and re-run with each
  SDK's pinned runtime: 1/1 each with no skips, logs
  `qt510-rich-continuation-v4.log` and `qt6-rich-continuation-v3.log`. It covers
  images, links, underline and list formatting on shared export/reopen, and does
  not prove complete Markdown/code/math or full-window alignment.

### 2026-10-28: user-authorized render fix, original drawing gate RED->GREEN

The user chose "fix the render differences first", scoped only to the polygon
first point, caption and anchor/leader, without extending to the three-state
review, the new canvas mode or the Markdown architecture. The shared
`AnnotationCanvas` owner now makes export and interaction use the same
caption/device-pixel layout; all leaders are drawn first, then the rounded label
background and the text. A closed polygon no longer draws its first point
unconditionally, while the in-creation / first-point-hover / selected paths are
preserved.

- The original 36 cases really ran as
  `upstream-cd26942-render-red-v2/report.json`, reproducing the original 19
  errors; after adding 8 cases for edge avoidance, anchor, display toggles, prompt
  and multi-label order,
  `upstream-cd26942-render-expanded-red-v3/report.json` has 27 errors.
- `upstream-cd26942-render-{qt510,qt515,qt6}-green-v6/report.json` are now each
  **44/44 PASS**, the 768 palette indices pass, and the original 18 geometry-only
  cases are all exactly pixel-equal;
  the one-pixel font space tolerance, cropped MAE<=1.0 and whole-image MAE<=0.25
  were not relaxed.
  All 44 Qt6.5.3 cases have MAE=0; Qt5.10.1/5.15.2 satisfy the original font
  tolerance.
  The reference's actual Qt runtime is 6.9.2 and the upstream revision is still
  `cd26942a...`; each report records the reference source SHA, the native Qt/font
  metrics and the probe SHA.
- After the first Qt5.10 rebuild six attributes items still FAILed, see the
  `qt510-green-v4` report.
  `render-qt510-metrics-v5/qt-runtime.json` really measured offscreen logical
  DPI=100 with 10pt Bold=14px/height16/width104, while the original 96-DPI
  reference is 13px/height15/width93.
  The comparator already set `QT_FONT_DPI=96`, but the old SDK did not size fonts
  by it. Only the cross-version `AA_Use96Dpi` is set before the test probe
  constructs its QApplication, restoring the original test protocol; the production
  fonts, the attributes implementation, the models and the original gates were not
  changed. The old six failures and the metrics receipts are all retained.
- The real Qt5.10/Qt6.5 canvas interaction regressions are 1/1 each and the root
  build's single Qt5.15 canvas item also passes;
  logs `render-qt510-canvas-v4.log`, `render-qt6-canvas-v4.log` and
  `render-qt515-canvas-v6.log`. The official `QANYLABELING_PLUGIN`, the render
  probe and the canvas target were all rebuilt in a single `-j4` queue; the strict
  plugin boundary check passes.
- After rebuilding the plugin, `upstream-cd26942-config-after-render-v6.json`
  still **FAILs** with 15 errors: defaults, the two insufficient review actions,
  nine palette items and four panel QSS items. Only the drawing sub-gate above is
  closed; the config, all-module, full-window, OCR/LLM accuracy-performance and
  native-platform gates are not.

### 2026-10-28: the isolated CPU capital subset still fails the complete performance gate

After the official plugin build exited, the F16/Q8 rows of `llm-chat-capital`
were re-tested with the same v4 snapshot, original baseline/assets/gates and
release 3x2x10 protocol with threads=4.
`Testing/cpu-capital-isolated-v2/report.{json,md}` has candidate
correctness/stability 2 PASS with no skips and execution identity unchanged, but
an overall **FAIL**: F16 e2e p95 154.422 -> 167.962 ms (+8.77%, +13.540 ms).
The original complete report's capital decode p50 +8.13% did not trigger again
in this subset; this neither confirms that decode regression nor withdraws the
old complete FAIL.
The 17:21:18-17:21:42 observation window in
`llm-snapshot-cpu-capital-isolated-v2-host.log` contains no additional compiler;
the observer attached after the runner launched and is not a full-run load lock
or an nsys trace. All 451 names passed another SHA check after completion.
These are only two diagnostic rows, not a substitute for re-testing the remaining
CPU/Vulkan/CUDA failures.

The later phrase "qAnyLabeling zero changes needed" is a historical assessment
from 2026-10-25, overturned by this round's set-comparison RED; "LLM skeleton is
next" also describes only the state at that time. The public llm
ABI/catalog/worker now exist, but complete release remains open. Historical
metrics and original FAIL/INCOMPLETE results are neither rewritten nor withdrawn.

## Historical status (2026-10-23 evening)

The CUDA detector source-speed recheck now
passes for published F16 and repaired Q8 after exact-gather scheduling and copy
elision. The Vulkan sibling scheduling patch preserves outputs bit-for-bit
(15/15 model/batch rows), but the Vulkan detector release-profile recheck still
fails the source-speed gate (p50 12.1 ms vs Paddle CUDA ~6 ms); Vulkan
qualification and complete exact-file page/stability receipts remain open.
These graph results alone do not qualify a release upload.
The release decision follows task-level page accuracy
and controlled timing against the official implementation. Text-graph quantized
raw-output differences are diagnostic; the separately requested layout gate
still requires 100% original class-head and per-edge DFL argmax agreement.
The published Q8 files still
exceed the declared page-error budget, but they are different SHA identities
from two new Q8 repair candidates. Those repair candidates now pass the selected
CUDA and Vulkan developer accuracy matrix (nine rows per backend, one repeat).
This is not release qualification: controlled official-speed/VRAM and stability
gates, the complete per-file matrix, production catalog/digest registration, and
cold-download verification remain open. Newly qualified uploads: **0**.

This is a local worktree evidence ledger. Results belong to exact model, input,
script and build identities. The 51 main files comprise 36 extension candidates,
12 published text/layout files and 3 published SLANet files. Two Q8 repair
candidates have separate SHA identities and never inherit the published files'
results. The 2026-10-20 GitHub API snapshot found 16 assets under `PP-OCR`
(15 GGUF files and one ZIP); push/admin access was available. Upload permission
is not the blocker.

The merged runtime SHA was
`9f2548ef28f6edadbcdbb0658d6fbf67e18dbd73ca2afb28183c2b2e9af59f21`.
Its 96 graph checks produced 46 PASS / 2 FAIL per GPU, with layout F16/Q8_0 the
only failures. Its 120 page/SLANet checks produced Vulkan 60/60 PASS and CUDA
59/60 PASS initially; the CUDA row rejected for a changed CMakeCache passed an
independent retry under the original conditions. Keep the first failure receipt.
These pages used original CPU references and remain developer evidence. They do
not establish original-CUDA page parity or release stability/performance.
See `merged-gpu-pages-20261020-v5k89ifx/` and `merged-gpu-followup-path.txt`.

The older runtime `154a0b62` completed 54/54 extension page scenarios on each GPU;
its CPU 54/54 is diagnostic only. CPU is not a release gate. Initial skips,
timeouts and interrupted runs remain INCOMPLETE even when a retry succeeds.
The 36 new files have single-run source probability/argmax evidence on each GPU.
Original-CUDA graph/orientation reruns completed 132 rows: 128 PASS / 4 FAIL,
with layout F16/Q8_0 failing on both GPUs. Two distinct public-Q8 repair SHAs
subsequently passed source graphs and combined page scenarios. The newer
persistent report at `public-q8-gpu-truth-vpx8osmk/{cuda,vulkan}/matrix.json`
records nine selected accuracy scenarios per backend: five public page sets,
two source graphs, and two held-out receipt sets. Both reports are developer
profile, one repeat, and have no controlled latency or VRAM claim.

Published text files retained 24 PASS / 4 FAIL per GPU. Layout F16/Q8_0 retained
their strict source failures. CDLA's full 1,000-page validation set and 8,711
human regions were independently pinned and evaluated; page tolerance approval
and public consumers remain open. SLANet original graphs passed 6/6 GPU rows;
the older crop-table/combined-page matrix passed 12/12 GPU rows, with six extra
CPU diagnostic rows. F32 crop TEDS/TEDS-S was 0.9645975/0.9959180 and combined-page
TEDS/TEDS-S was 0.5431670/0.5917242. Source agreement does not mean absolute quality
is 1. Table TEDS is not replaced by text CER/WER.

Controlled same-host baseline/candidate/source-CUDA A/B is incomplete.
`gpu-release-20261020-py6cqztd/{cuda,vulkan}/matrix.json` records three attempts per
backend rejected before timing because Xorg/desktop GPU activity exceeded the
idle gate. Extending idle waits to 120 seconds did not produce a qualified CUDA
baseline/candidate sample. Two rejection receipts were retained and remaining
waits cancelled so numerical work could continue; that later Vulkan retry did
not run. See `controlled-release-cancelled-after-audit.json` under the directory
identified by `merged-gpu-followup-path.txt`. Activity rejection is not evidence
that model inference is too slow. No external process was stopped and no audit
threshold was relaxed.

Original CUDA reference comparisons are independent of performance audits:
36/36 new files passed per GPU; public Paddle graphs were 10 PASS / 2 FAIL per
GPU. Worst layout-head argmax agreement was 0.992307692 for F16 and 0.969230769
for Q8_0. Initial interpreter-resolution and missing eight-head export failures
occurred before numeric comparison, were retained, and were rerun after repair.

OCR table editing, basic rich text, and Chat/VQA configuration/message workflows
have native/source-method evidence. Table content comparisons cover 5,407 cases,
six geometry groups, six viewport-height groups and two theme QSS groups.
Rich text covers 13 serializations, six heading metrics, 12 paste classes, three
crop previews, six height cases, light/dark styles, heading popup, source-drawn
icons and image selection/scroll/delete behavior. Four plugin-entry/video groups
and five Chat/VQA source-method groups passed. Mini-probes do not replace a
formal plugin rebuild, full-window visual comparison or model gates.
Local VLM, remaining exact catalog entries and Linux clean-package remain open.

Remote focal package generation, installation and headless/JSON-RPC tests had
succeeded. A stale local installation and additional local conditions did not
prove a remote packaging defect. Packaging code, toggles and build-document
changes were restored to the synchronized HEAD; the local installer queue was
cancelled. Historical packaging tests do not certify the restored code or future
uncommitted changes. See the CI reassessment in
[UPSTREAM_PARITY_AUDIT.md](UPSTREAM_PARITY_AUDIT.md),
[window acceptance](../../../../build_app/Testing/ocr-storage-reference/window-alignment-acceptance-20261019.json)
and [2026-10-20 acceptance](../../../../build_app/Testing/ocr-storage-reference/acceptance-20261020.json).

Raw reports, commands, input provenance and build identities are under
`build_app/Testing/ocr-storage-reference/`. `ocr-per-file-evidence.json` indexes
completed reports; it is not an automatic release qualification engine.
The following per-file index is historical developer evidence, not a claim that
the latest candidate has passed release gates.

# OCR per-file evidence snapshot

Developer evidence index only: 48 text/layout/classification files plus 3 SLANet table files. No unexecuted test becomes PASS. Missing page gates and controlled performance remain open. Table TEDS requires its own task matrix and is not replaced by CER/WER. Existing v3 page identity has known vendored metadata overwrite; only fresh v4 receipts use the selected-distribution/helper digest identity. Neither is a complete release gate.

Model SHA is the exact file consumed. Storage percentages count parameters rather than file bytes; names do not imply pure quantization. P/F refer only to completed common-runner rows; page counts are completed / manifest required. H is a weight-matched historical PASS with incomplete build identity; it does not qualify a release and may overlap a newer P.

| Model | SHA-256 | Storage | Source graph CPU/CUDA/Vulkan | Pages CPU (diagnostic) | Pages CUDA | Pages Vulkan | Remote/catalog SHA | Release |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pplcnet-layout-f16.gguf | fda1343642abc04475271438f29b58036528be5f3f6a422c10ffdde3d7bcda1a | F16 100.000% | pending/F/F | 0/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | match | BLOCKED |
| pplcnet-layout-f32.gguf | 7e2785606f99bc6d1c7ffa6df6b15e42bd89da6c7a98ce7f1de1961c4cb2fcf1 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | match | BLOCKED |
| pplcnet-layout-q8_0.gguf | 18056b5559f3da420ab6cb5e3dd14a5c9f18f5fb8da91837ba8fa24288b07e42 | F16 10.996%; Q8_0 89.004% | pending/F/F | 0/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | match | BLOCKED |
| ppocr-cls-f16.gguf | 323d117aeb7bd4340f09d92d0ba53675207e761dd3e2c4883a3e9cdc84b8e2eb | F32 7.960%; F16 92.040% | historical P; rebuilding evidence/P/P | 0/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | unpublished | BLOCKED |
| ppocr-cls-f32.gguf | 0ef8558a8de46bcb499f89c7313d7a53707061d452e5ebd2aa89512aa1b84e62 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | unpublished | BLOCKED |
| ppocr-cls-q8_0.gguf | 8f6e0fa64205da94f5fd8a70400f838b1cb0ccb836eaf44b279553a324d136b3 | F32 7.972%; F16 87.240%; Q8_0 4.788% | historical P; rebuilding evidence/P/P | 0/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | unpublished | BLOCKED |
| ppocr-det-ch-f16.gguf | 82517e16120a186cc85123272286a5f35e9ede4b4d272fc2e7d5b7148879550d | F16 99.998%; F32 0.002% | historical P; rebuilding evidence/P/P | 0/9 P; 0 F; 5 H | 2/9 P; 0 F; 9 H | 2/9 P; 0 F; 9 H | match | BLOCKED |
| ppocr-det-ch-f32.gguf | aaad25f65055ecf0c34a9050b249ded1f099e8d91beb7ad4749b24e06b11b9fd | F32 100.000% | historical P; rebuilding evidence/P/P | 0/9 P; 0 F; 7 H | 2/9 P; 0 F; 9 H | 2/9 P; 0 F; 9 H | match | BLOCKED |
| ppocr-det-ch-q8_0.gguf | ede6dd9620f482224ec10cf9bfde58957eb8815c461b755173031779568ac207 | F16 33.798%; F32 0.002%; Q8_0 66.200% | historical P; rebuilding evidence/P/P | 0/9 P; 0 F; 2 H | 2/9 P; 0 F; 5 H | 2/9 P; 0 F; 5 H | match | BLOCKED |
| ppocr-rec-ch-f16.gguf | 2507cedcb718b3b98af896d2c633d42a162d3119ad6ea776de808c2dd0e4213c | F16 61.884%; F32 38.116% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 3 H | 2/6 P; 0 F; 6 H | 2/6 P; 0 F; 6 H | match | BLOCKED |
| ppocr-rec-ch-f32.gguf | a29b0d53026ab18e9d2a0e537eea90a140521b1eecfb8c0578bd1d6b4187e60a | F32 100.000% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 5 H | 2/6 P; 0 F; 6 H | 2/6 P; 0 F; 6 H | match | BLOCKED |
| ppocr-rec-ch-q8_0.gguf | 7d85184357ae6ffec60633b1db8c033000940f585852b4f0af3a2b8bcc34bae2 | F16 22.664%; Q8_0 39.220%; F32 38.116% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 2 H | 2/6 P; 0 F; 5 H | 2/6 P; 0 F; 5 H | match | BLOCKED |
| ppocr-rec-en-f16.gguf | f05e08cbc81cd9863cddfc17621e788eb896278e3a36448cccecd5829c5729bf | F16 87.263%; F32 12.737% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 2 H | 0/3 P; 0 F; 3 H | 0/3 P; 0 F; 3 H | match | BLOCKED |
| ppocr-rec-en-f32.gguf | f38155dc785cbfc23f5fe37af2d6403b42cfb8fb73af130699a425834c2441dc | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 2 H | 0/3 P; 0 F; 3 H | 0/3 P; 0 F; 3 H | match | BLOCKED |
| ppocr-rec-en-q8_0.gguf | d336e48f4b53b756058fd4015270a8ae6ed10d03810fa976a558c9fd98f5bb42 | F16 31.741%; Q8_0 55.522%; F32 12.737% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 0 H | 0/3 P; 0 F; 0 H | 0/3 P; 0 F; 0 H | match | BLOCKED |
| ppocrv3-japan-rec-f16.gguf | 4cf0971793cadfe242b7a370546de4886543d96262019395848244bf5b0f654e | F32 64.884%; F16 35.116% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv3-japan-rec-f32.gguf | ec3eeb364bbbeecd365010e0e61f9f6594dd84260296e3b43b645e922503fa5d | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv3-japan-rec-q8_0.gguf | 893a813c27382d81e98aafa309653e77a679cad1e75a4294d91b730569cc0865 | F32 45.475%; F16 53.218%; Q8_0 1.307% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv4-source-det-f16.gguf | 07000b35b34fb24755fdd91987e04a927e1357aebee28c8ca38a61809739b7f6 | F32 0.636%; F16 99.364% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | unpublished | BLOCKED |
| ppocrv4-source-det-f32.gguf | b96f01f423cebca891b095da11f24b69225257bddc17a86283e3707e9e9a17ab | F32 100.000% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | unpublished | BLOCKED |
| ppocrv4-source-det-q8_0.gguf | a62a8e2b86cb730bbf95e61071c14a8f148cc36f1b7b55f366b8baf36b4e327e | F32 0.648%; F16 23.799%; Q8_0 75.553% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | unpublished | BLOCKED |
| ppocrv4-source-rec-f16.gguf | b38a649949a551898da9f86ab88570b63bcb46401ae390a7ddb8fa0ce48e5c98 | F32 55.179%; F16 44.821% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv4-source-rec-f32.gguf | c9bd1ed2f82e0f01a50cbcae2eff871ca12856393d03b407df895fa4d2925d0f | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv4-source-rec-q8_0.gguf | 8ab6f58145f5af99887be7a34d2aaa9aad39a1bd348a93acc6eb95498f10fcfa | F32 55.179%; F16 41.610%; Q8_0 3.212% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-det-f16.gguf | f4c539a72c5b9f41e96d2f83d4ce510a814606f6c1b824c35df2ade24cb637a7 | F32 12.207%; F16 87.793% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-det-f32.gguf | d565d9fa39e8e32416a469938e0b5b7b3d251f4b22c4239c1ab9aee94197fb61 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-det-q8_0.gguf | 5400362f3062b1d6636b5638a8200490e398f016e449216fd68ee9c4e6681f43 | F32 12.207%; F16 76.388%; Q8_0 11.405% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-rec-f16.gguf | a5ce8f19f5983dea315056a8d6857cb15b26b47107bfad0752b027adab2330d4 | F16 21.232%; F32 78.768% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-rec-f32.gguf | 3e2cb8e43687a549acdf489b0abbeb07989751804d5dec6501fa508398d26b8d | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-rec-q8_0.gguf | 72d723b5c8e21ffd57fb2e91f97a8e3714ae29e9c332925236b2ad4493130c1b | F16 19.132%; F32 78.768%; Q8_0 2.101% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-det-f16.gguf | 57d3dd71018dd535bbdd3c52cf93d8699192662618be4933584da4ea6fddf98e | F32 0.954%; F16 99.046% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-det-f32.gguf | ce8f37dbbde68ba1ac8fca40a076757a676ef48cae07d1b423cf996a1c92e692 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-det-q8_0.gguf | 951d8011139fb2d3f0ee2d1982fef6c97acd78cba998065b453ce1e62610e8f3 | F32 0.954%; F16 27.938%; Q8_0 71.107% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-rec-f16.gguf | e6a40f225b5d15de9a6e06b4a16075d510df164c3e41fbbf824481c6079501ad | F16 76.289%; F32 23.711% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-rec-f32.gguf | 5f5b315e151f1fab29cf2c4d0ad0fd9aa0694ce23929a2076e41cea2ffb4bd8b | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-rec-q8_0.gguf | 553d210c0f8d9765d04db7e0f0b7527ed5894502da1bc0e1827db86752513b0d | F16 7.977%; F32 23.711%; Q8_0 68.312% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-det-f16.gguf | 73efc76d8b6427363b19c5a88ecf7075febfd5bbbef0b5b7ff67ac4d8f5330c1 | F16 99.523%; F32 0.477% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-det-f32.gguf | 66a29730f76956678c5f2b8d72b7b79a5ac12a4d160d246a9a0f9eb3194445ce | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-det-q8_0.gguf | 8c6ee6c483d290ea534c1c8deb2b6f701ecdcfc5a337616408f9400a8d7787b1 | F16 13.563%; F32 0.507%; Q8_0 85.930% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-rec-f16.gguf | 48543319c045e7b4d05b2ed77008715bb750030bd95bbdc8ba8024c8871e40b0 | F16 52.341%; F32 47.659% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-rec-f32.gguf | 159df7215c842276cf288675b421d8a2d73c52b0816bdb724a862c04bf3f9a4e | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-rec-q8_0.gguf | 803a8053b03f1e7c7b5d1399909d8cb8bbf786613d24164f89add899371f17ec | F16 23.605%; F32 47.659%; Q8_0 28.736% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-det-f16.gguf | 804a564ad2db0fbd2b95325c167b854a4cba18c36889aab16e51c6ae2a7c0c83 | F16 98.740%; F32 1.260% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-det-f32.gguf | 9f0f264028796c6c61be5f79394367b0f9a2c5a46340b6c0041d0e8c81bfe5bb | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-det-q8_0.gguf | fa5f6f6efdc79c4a2363c7719d6dd5d7ca999e14cbc71132a65d30ce834c4988 | F16 98.501%; F32 1.260%; Q8_0 0.239% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-rec-f16.gguf | b5a7a723e105e19d7b01d1b0cc3656c4456931eba9e7e7363d85ef92a736fb18 | F32 84.468%; F16 15.532% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-rec-f32.gguf | 0e8848426d046549f926f057dd1d7ae587f50d8e327be1dbcce418dad3006406 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-rec-q8_0.gguf | f9e6c3793c95fa7e84b10d5fa277f56e22b059c4e1517ba93a223f83fa41d960 | F32 84.468%; F16 14.953%; Q8_0 0.579% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| slanet-plus-table-f16.gguf | 0eb25bd9a67d9eafc830df5400dd4b8bef8a5a506c067ce384f925a3a3abdf63 | F16 100.000% | historical P; rebuilding evidence/P/P | 0/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | match | BLOCKED |
| slanet-plus-table-f32.gguf | c93a3a24e39c30bd67294a8abb543e3e0ed7ef1f2c6d67505349dee77aa87bf4 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | match | BLOCKED |
| slanet-plus-table-q8_0.gguf | cf93b69e918d7202f9779ca376d25063d2fe15ffdd26ae209d4600ae7b56d1f0 | F16 35.537%; Q8_0 64.463% | historical P; rebuilding evidence/P/P | 0/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | match | BLOCKED |

## Separate unpublished repair candidates

These are new file identities and do not inherit the preceding published files' results. Their unchanged source weights/dictionary and reproducible conversions do not qualify them for upload.

| Model | SHA-256 | Q8_0 parameters | Gate state |
| --- | --- | --- | --- |
| ppocr-det-ch-q8_0.gguf | cab74270c09bd584869840ac11e60ee9dafab7b43ed317922c1d2215ea9ae56d | 66.069% | CUDA/Vulkan source graph and nine-scenario developer accuracy matrix PASS (one repeat); controlled official A/B, VRAM, stability, complete per-file matrix, catalog/digest and cold-download pending |
| ppocr-rec-en-q8_0.gguf | 695bdcbbce6707fb388441ed651060b6175b2580be8860942c3766d7bbac5217 | 43.398% | CUDA/Vulkan source graph and nine-scenario developer accuracy matrix PASS (one repeat); controlled official A/B, VRAM, stability, complete per-file matrix, catalog/digest and cold-download pending |

Common runner report paths, hashes, exact input models and failure metrics are retained in [ocr-per-file-evidence.json](../../../../build_app/Testing/ocr-storage-reference/ocr-per-file-evidence.json).
All new uploads: 0. Dedicated OCR workspace implementation has begun; remaining OCR/VQA/Chat workflows, catalog and Linux clean-package are open.

## 2026-10-19: Persistent paths and actual blockers

The 38 candidates were copied without changing their SHAs into:

```text
/home/ludahai/develop/code/github/ACloudViewer/build_app/Testing/ocr-storage-reference/ocr-release-candidates-20261019
```

`inventory.json` records original/resolved paths, bytes, SHA and actual storage
composition; `SHA256SUMS` covers all 38 files. Copying into staging is not release
qualification. Each of the following 12 stems has `{f32,f16,q8_0}.gguf` files:

| Stem | Files | Historical numerical evidence | Remaining release work |
| --- | ---: | --- | --- |
| `ppocr-cls` | 3 | Source graphs and associated extension page matrix | Controlled CUDA/Vulkan A/B and publication/download verification |
| `ppocrv3-japan-rec` | 3 | Same matrix; file-specific consumers in the evidence index | Same |
| `ppocrv4-source-det` | 3 | Same | Same |
| `ppocrv4-source-rec` | 3 | Same | Same |
| `ppocrv5-mobile-det` | 3 | Same | Same |
| `ppocrv5-mobile-rec` | 3 | Same | Same |
| `ppocrv6-medium-det` | 3 | Same | Same |
| `ppocrv6-medium-rec` | 3 | Same | Same |
| `ppocrv6-small-det` | 3 | Same | Same |
| `ppocrv6-small-rec` | 3 | Same | Same |
| `ppocrv6-tiny-det` | 3 | Same | Same |
| `ppocrv6-tiny-rec` | 3 | Same | Same |
| Separate public Q8 repairs | 2 | Seven combined page scenarios plus source graphs, recorded by new SHA | Controlled A/B and publication/download verification; never overwrite old evidence |

Failures and missing evidence are distinct:

- Published layout F16/Q8_0 failed original Paddle CUDA argmax, while backend
  agreement passed. CDLA page measurements do not waive this accuracy failure.
- Published text Q8 Chinese/English scene e2e CER increased by 0.009708738 /
  0.005295675, exceeding 0.005; English receipt fixed-region WER increased by
  0.006833713. Repair SHAs require independent evaluation.
- GPU idle audits rejected A/B before valid measurement. Desktop activity and
  external `dynamic_svm_online_calib_node` / `semantic_segmentation_eval_probe`
  workloads were recorded, not stopped. The 13:28 drain attempt was cancelled;
  another external workload at 14:18 prevented a new A/B start.
- All 36 extension files include F32/F16/Q8_0 variants. Q8 files are mixed
  storage: Tiny detector/recognizer contain only 0.239% / 0.579% Q8 parameters.
  Names do not establish pure quantization or speedup.
- A historical public-CPU page queue stopped at 20/28 without a final matrix.
  Its log is retained; per-scenario runner retries do not retroactively pass it.
- Upload, release SHA, catalog, manifest and cold-cache download remain separate
  steps after numerical/page/performance qualification. Being unpublished is
  not itself a numerical test failure.

## 2026-10-20: Independent CDLA evaluation

`core/AICore/tools/ocr_cdla_validation.json` pins all 1,000 official validation
images/annotations, the archive's size/SHA and source revision. No training pages
or prediction-derived labels are used for validation. Executing the official
LabelMe-to-COCO converter and rasterizer reproduced categories, bbox and area for
all 8,711 regions. Positive COCO annotation IDs avoid the zero unmatched sentinel
without changing geometry. Evaluation is class-aware bbox AP50:95/AP50/AP75 and
AR100, including zero-prediction pages. Original Paddle postprocessing keeps
score=0.5 and NMS=0.5; finite out-of-bounds boxes follow COCO semantics.

The evaluator, source exporter and public typed-C-ABI region exporter have 17
unit checks and CTest registration. Initial 1,000-page source-CPU and three-page
native-CPU runs were plumbing diagnostics. Common-runner cold-cache download
verified 2,000 consumed files; offline reuse performed zero downloads.
See `layout-cdla-source-path.txt`.

Original CUDA and all six native backend/dtype exports later completed all
1,000 pages normally; the formal CDLA CTest passed:

| Execution | mAP50:95 | Delta from original CUDA |
| --- | ---: | ---: |
| Original Paddle CUDA | 0.8546702421 | 0 |
| Native CUDA F32 | 0.8546698470 | -0.0000003951 |
| Native Vulkan F32 | 0.8546698405 | -0.0000004017 |
| Native CUDA/Vulkan F16 | 0.8547334403 | +0.0000631982 |
| Native CUDA/Vulkan Q8_0 | 0.8537569219 | -0.0009133202 |

Page release tolerance and public manifest consumers still require review.
Strict original-head failure and controlled performance remain independent
blockers. Do not infer a release tolerance from these observed AP deltas.

## 2026-10-20: Original CUDA reference selection

Older GPU graph probes first applied an extra original-CPU numeric gate, so a
subsequent CUDA match could not undo that failure. The corrected probes select
original CUDA before native GPU execution. Both original ONNX and reconstructed
stored-weight graphs use CUDA; additional images, orientation and CTC retain
CUDA truth. CPU fixtures verify provenance, not GPU numeric qualification.
CUDA-provider fallback fails. Probability, stored-weight and strict argmax
thresholds were retained. The isolated graph checks passed 37 tests.

Pages had the same issue. Explicit CUDA references were added to 66 ONNX page,
21 Paddle page and 18 orientation consumers. Paddle rejects replay of CPU
exports and disables TF32; ONNX verifies its actual CUDA provider. Source success
is written only after a normal subprocess exit. Human GT and Hmean/CER/WER
thresholds were unchanged. Isolated tests passed 167; after the frozen queue
ended, seven production-script groups passed 184 tests including 17 CDLA tests.
Native runtime/probe binaries were unchanged. See `gpu-truth-selection-path.txt`.

The resulting full graph/orientation run was 128 PASS / 4 FAIL out of 132:
36 ONNX graphs and 18 orientation flows passed on each GPU; only layout F16/Q8_0
failed among the 12 Paddle rows per GPU. Read-only saved-tensor comparison found
identical CUDA/Vulkan failure positions: F16 22 class / 40 DFL-edge flips and
Q8_0 253 / 419, with backend max difference below 0.000021. See
`layout-backend-attribution.json` and `layout-decision-confidence.json` in the
raw-report directory. Although flipped class winners were below score .5,
1 F16 and 2 Q8 anchors crossed .5. This does not prove identical final boxes.

Eight offline controlled-retry preflights passed. Planned release sampling was
three processes, two warmups, ten timed forwards and a 120-second idle audit,
with original CUDA truth and the frozen same-host older runtime. A failed
classifier sentinel stops that backend's full performance cohort. Queued work
and preflight never count as PASS. See `gpu-controlled-retry-path.txt` and
`public-q8-gpu-truth-path.txt`.

## 2026-10-20: Stored-weight attribution and controlled sentinels

`export_paddle_ocr_stored_weights.py` checks names, shapes, order, original SHA
and serialized payload, replacing only an independent Paddle parameter copy with
actual dequantized GGUF values. F32 round-trips byte-for-byte; all 486 parameters
were reconstructed for each dtype, with six tests. The unchanged Paddle CPU
graph then matched every native CUDA/Vulkan head's argmax; max differences were
below 0.000025 / 0.000041. This is storage-attribution evidence, not a GPU release
gate. Original F32 truth and strict failures remain. See `layout-stored-paddle-path.txt`.

Classifier F32 controlled sentinels kept the original audit and sampling rules.
Desktop activity rejected both baseline and candidate; active probes exited
before remaining attempts were cancelled with -15. Neither cohort has a complete
final matrix. The page scheduler was restored. See `gpu-controlled-first-path.txt`.
Pausing a scheduler during one CUDA XFUND F16 scenario caused outer timeout 124;
its partial output is not success. The original-condition retry was queued under
`gpu-page-pause-retry-path.txt`. Do not pause a runner during a timed scenario.
`gpu-controlled-after-pages-path.txt` records a later serial retry plan, not PASS.

The 2026-10-22 policy required **100% argmax agreement for every original
head**, including low-confidence classes and intermediate DFL distributions.
On 2026-10-23 the user clarified that release acceptance should follow task-level
accuracy and inference time relative to the official implementation. Keep raw
probability, argmax and DFL differences as diagnostics, and use independently
labelled pages plus the predeclared metric budgets for acceptance.
`layout-quantization-review.json` also records a Q8 Equation-class AP drop of
0.359033 percentage points; overall mAP alone omits that difference.

The supposed older `test_table_runtime` was a symlink to the current build, not
a historical baseline. A common new public-C-ABI `bench_table` was instead run
against independently retained old/current `libAICore.so` files. Its two CPU
plumbing checks verified actual library paths, stable output and 24 tokens;
they do not establish GPU qualification or make the new probe historical.

`run_table_capi_probe.py` checks three lossless PNGs and exact original NCHW
preprocessing, independent Paddle CUDA, strict GPU audit, repeated tokens,
truthful timings, source SHAs and both sides' sampled VRAM. Its
`decoded_token_match` is explicitly not all-raw-head argmax. Quantiles use raw
samples consistently. Seven rejection tests and two offline preflights passed;
the six backend/dtype release cohorts remained queued. See `table-capi-baseline-path.txt`.

### Layout accuracy root cause and remaining evidence

`layout-attribution-reverified.json` rechecked 27 identities and recomputed both
comparisons for the fixed public `layout_input.npy`:

| Storage | Worst head vs original Paddle CUDA | Class / DFL-edge flips | All-head agreement vs stored-weight Paddle CPU |
| --- | ---: | ---: | ---: |
| F32 | 1.0 | 0 / 0 | 1.0 |
| F16 | 0.9923076923 | 22 / 40 | 1.0 |
| Q8_0 | 0.9692307692 | 253 / 419 | 1.0 |

Counts were independently computed for each GPU and were identical. The original
486-parameter Paddle graph reproduces native decisions after replacing weights
with actual storage values. This attributes these fixed-input failures to weight
rounding and propagation, not an unknown decoder problem. It is not a proof for
all OCR models, inputs or conversion policies, and CPU remains diagnostic.

The Q8 file has 59 actual Q8_0 tensors and 427 F16 tensors. Separate interventions
kept all other parameters original F32:

| Replaced parameter group | Tensors | Class flips | DFL-edge flips |
| --- | ---: | ---: | ---: |
| Actual Q8_0 tensors only | 59 | 244 | 411 |
| F16 fallback tensors only | 427 | 22 | 48 |

Both groups independently cause flips. Nonlinear propagation means their counts
cannot be added. See `layout-storage-factorial-path.txt` / `factorial-attribution.json`.
Small probability errors, backend agreement and similar mAP do not imply every
winner is unchanged. Do not rename F32 as F16/Q8, fit a single validation input,
replace original truth with stored-weight truth, or hide low-confidence flips.

### Quantization budgets: head-level diagnostics and task acceptance

For a vector with original top-two gap `margin` and component error bounded by
`epsilon`, `margin > 2 * epsilon` guarantees the winner is unchanged. Near ties
can flip under tiny rounding. Recovering every winner on a finite acceptance set
does not guarantee all future inputs under lossy storage. DFL boxes use weighted
softmax distances in `ocr_capi.cpp:gflDecode`; bin argmax is not box accuracy.

A read-only review of 24 report/output identities reproduced all original flips;
every flipped vector failed this sufficient stability condition. At score .5,
1 F16 / 2 Q8 candidates crossed the threshold. These are fixed-input anchors,
not final NMS boxes or page acceptance. See `layout-margin-review-jj9e632k/report.json`.

| Acceptance layer | Mandatory evidence | Optimization boundary |
| --- | --- | --- |
| Independent original framework | Per-head probability/logit errors, argmax, DFL and per-class results | Diagnostic evidence; head-level mismatch alone does not reject a model |
| Added backend error | Independent stored-weight graph vs CUDA/Vulkan | Separate from weight rounding and task-level page accuracy |
| Real pages | Hmean/mAP, per-class degradation, CER/WER and table structure | Current page manifests allow up to 0.005 degradation per declared metric; include independent held-out inputs |
| Performance/release | Controlled same-host latency/throughput/VRAM against official source, stability, exact SHA and download | Reject a timing row only when it exceeds both the 5% relative and 3 ms absolute noise thresholds; require a clean GPU audit |

Sensitive tensors/channels, explicit F32/F16 exceptions, calibration and offline
QAT can be investigated on independent training inputs. Retaining only final
heads cannot guarantee exactness when the backbone/F16 fallback also changes.
Each candidate records actual tensor dtypes, size and SHA before independent
page/backend checks. The current user acceptance supersedes the previous raw-head
100% veto. Keep the automated page-budget result visible; a close task-level
result may be accepted under the user's direction with its exact deviation
recorded. Current Q8 page accuracy is close by that criterion, but release is
still pending qualified official timing and the remaining per-file provenance,
SHA and download checks. Raw-head mismatches alone do not block it.

### 2026-10-21: Repair SHAs, precision policies and SLANet queue

The two public Q8 repairs passed two original Paddle-CUDA graphs and seven shared
combined-page scenarios on each GPU, including scenes, SROIE, XFUND and held-out
SROIE 050–099. Old published-SHA failures remain.

| F16 role-based diagnostic policy | F32 parameters | Class flips | DFL-edge flips | Verdict |
| --- | ---: | ---: | ---: | --- |
| Norm and bias F32 | 2.867968% | 13 | 16 | FAIL |
| Also retain four final convolutions F32 | 4.049439% | 10 | 15 | FAIL |

These are fixed-input CPU diagnostics, not page/GPU acceptance. The SLANet queue
had waited for an OCR worker that had already exited after a build-identity
change. Its ERROR/INCOMPLETE receipt was completed without rewriting the error.
After frozen inputs were reverified, four CUDA idle audits each waited 120s and
rejected before sampling. Minimum GPU utilization was 11%, 15%, 13%, 12% against
the 10% limit. Only this queue's redundant waits were stopped. Vulkan did not run
and is neither PASS nor FAIL. Overall status remains INCOMPLETE. See
[follow-up evidence](../../../../build_app/Testing/ocr-storage-reference/ocr-release-followup-20261021.json).

### 2026-10-21: llama CUDA migration

Isolated probes were rebuilt against installed llama/ggml 0.27 headers/libraries
without reconfiguring or replacing `build_app`. Loader traces confirmed only new
base/CPU/CUDA modules, with no old 0.21 contamination.

| Scope | Exact model SHAs | Input checks | Elementwise equality to retained pre-migration native output |
| --- | ---: | ---: | ---: |
| Public Paddle text/layout plus two Q8 repairs | 14 | 14 | 14/14 |
| 12 extension models, three dtypes | 36 | 135 | 135/135 |
| SLANet, three dtypes | 3 | 9 | 9/9 |
| Total | 53 | 158 | 158/158 |

Independent original-CUDA arrays covered 14 Paddle, nine SLANet and 36 of the
135 ONNX checks. The other 99 checks replayed native output, not fresh framework
truth. Pages and controlled performance were not rerun. Original layout flips
remain 22/40 F16 and 253/419 Q8. See `build_llama_ocr_cuda_2m6kcjxu/` and
[resource migration evidence](../../../../3rdparty/llama/validation/resource-migration-evidence.json).

### 2026-10-21: Ordered Vulkan patches and installed libraries

Thirty-three source/shader files were ported to embedded ggml 0.27 and registered
in llama's manifest. A real `ExternalProject_Add(ext_llama)` extraction/patch/
build/install produced llama, mtmd and one embedded ggml. Three CTests passed,
including actual CPU/Vulkan ops and 11 CMake contracts. Reapplying patches was
idempotent; 225 Vulkan files matched the reviewed tree. Loader traces verified
the new installation without old or scratch modules.

| Vulkan scope | Models / inputs | Max probability delta vs old native output | Added applicable-head flips |
| --- | ---: | ---: | ---: |
| Public Paddle plus Q8 repairs | 14 / 14 | 1.47820e-5 | 0 |
| Extension ONNX, three dtypes | 36 / 135 | 1.49608e-5 | 0 |
| SLANet, three dtypes | 3 / 9 | 2.29537e-4 | 0 |

SLANet location max delta was 2.84731e-4. Original-CUDA coverage matches the
previous section; frameworks were not rerun here. See
[Vulkan evidence](../../../../3rdparty/llama/validation/vulkan-rebase-evidence.json).
A combined CUDA/Vulkan install then shared its base runtime, passed four entry
CTests, relinked monolithic AICore using 240 retained candidate objects and
passed 36 public-contract tests without skips. This is link/execution evidence,
not a full application rebuild or all-task acceptance.

Both GPUs reran all 158 checks. An initial attempt selected old CUDA 11.1 and
skipped CUDA execution; it was retained. A separate correct CUDA-11.8 run checked
all native return codes were zero. See
[dual-GPU evidence](../../../../3rdparty/llama/validation/dual-gpu-evidence.json).
Metal, remaining task regression and controlled performance were not completed;
the final legacy production entry was retained.

### 2026-10-21: Rounding calibration

The unchanged Paddle graph with actual GGUF weights reproduced 62 F16 and 672
Q8 original-head flips on CPU. The corresponding GPU migration added no applicable
head flips. This isolates weight storage/propagation for this input, not decoding.

Eight independent official training images supplied 64 layer activations with
damping 0.01. F16 activation feedback without structural exceptions produced
18/44 class/DFL flips and was worse than nearest rounding. Keeping one-dimensional
structural parameters F32 gave nearest 13/16 and feedback 7/12; Q8 improved from
243/413 to 122/226. Training-error selection of 1/2/4/8/16 F32 exceptions still
left best F16 7/11 and Q8 100/201. None reached strict 100%.

`ocr_q8_calibration.py` and `test_ocr_f16_calibration.py` are offline tools, not
default conversion policy. Three F16 tests passed using:

```bash
OPENBLAS_NUM_THREADS=2 python3 -m unittest discover -s core/AICore/tools -p test_ocr_f16_calibration.py -v
```

Receipts are under `build_llama_dual_gpu_3nl01j4j/ocr-calibration/`;
`promoted-v2-results.json` is authoritative for the exception experiment. The
first Q8 attempt used a different parent-selection rule and was retained;
v2 read the actual parent parameter bytes. Do not merge those attempts.
[PaddleOCR compression](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/slim/auto_compression/README.md)
and [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html) motivate experiments,
not identical raw-head guarantees or current-model release claims.

### 2026-10-21: Serialized candidates and in-domain calibration

Six policies were written into actual GGUFs and all 486 decoded parameters
matched their Paddle copies. Fresh original-F32 CUDA, stored-weight CUDA and both
native GPUs reproduced the remaining failures. An affine-compensated F16 policy
still had 8/8 flips.

A separate calibration used the first 16 fixed-order **train** images from CDLA,
with archive CRC, file and original-model SHAs checked. No validation images,
labels or outputs were fitted. The same 64 activations/damping were retained.

| Stored policy | CUDA class / DFL flips vs original | Vulkan class / DFL flips vs original | Both GPUs vs stored Paddle |
| --- | ---: | ---: | ---: |
| Original F32 | 0 / 0 | 0 / 0 | 0 / 0 |
| F16 nearest, structural F32 | 13 / 16 | 13 / 16 | 0 / 0 |
| F16 feedback | 5 / 15 | 5 / 15 | 0 / 0 |
| F16 feedback, two training-selected exceptions | 5 / 15 | 5 / 15 | 0 / 0 |
| F16 feedback, affine compensation | 6 / 14 | 6 / 14 | 0 / 0 |
| Q8 feedback | 107 / 198 | 107 / 198 | 0 / 0 |
| Q8 feedback, 16 training-selected exceptions | 97 / 171 | 97 / 171 | 0 / 0 |

Receipts, real dtype ratios and arrays are in
`ocr-calibration/{serialized-candidates.json,gpu-serialized/results.json}` and
`cdla-train/` under the isolated dual-GPU build. Twenty-eight native GPU checks
include an F32 control per round. They establish only fixed-input diagnostics.
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/Quantization_intro.md)
provides calibration/bias/reconstruction approaches;
[CDLA](https://github.com/buptlihang/CDLA) separates 5,000 train / 1,000 validation
pages. Official INT8 is not GGML Q8_0, and task metrics do not prove raw-head identity.

### 2026-10-21: Margins and final projection reconstruction

Saved arrays contain original class top-two gaps as small as `7.0664e-8`.
For the affine-F16 first class head, median flipped-vector gap was about
`1.1858e-6` while median max perturbation was `2.8471e-6`. Low-confidence positions
remain gated. See `ocr-calibration/head-margin-diagnostics.json`.

Four final joint class/DFL projection layers were reconstructed on the same
16 train pages with relative ridge 0.01. Their kernels remained F32 and other
stored parameters were frozen. Training reconstruction MSE decreased, but
independent checks still found F16 9/16 and Q8 105/174 flips. Actual files
round-tripped all 486 parameters: F16 has 388 F32 / 98 F16 tensors; Q8 has
388 F32 / 59 Q8 / 39 F16 tensors. Independent original CUDA and both native
GPUs reproduced those counts, with zero flips vs stored-weight Paddle and
zero for F32 controls. See `ocr-calibration/cdla-train/block-reconstruction/`
and `block-reconstruction/gpu-serialized/results.json`. Both remain FAIL.

A separate SAM3 F16 CUDA output-stride defect was found during full-task
migration and fixed by an ordered patch/new-old module test. It is unrelated to
layout quantization; see
[stride evidence](../../../../3rdparty/llama/validation/cuda-native-stride-evidence.json).

### 2026-10-21: Full-model preflight and timing repairs

RF-DETR's initial 44-model run reproduced old output on both GPUs but reported
five CUDA / 29 Vulkan timing failures. The probe lacked warmup; Vulkan base Q4_K
had forwards of 307.676924 / 49.645335 ms in one process. Warmup, cold-call/raw
samples and nonzero errors for plugin-path/nonfinite failures were added to
`bench_rfdetr_perf`. The same new probe was linked against frozen old/new static
libraries without rebuilding `build_app`. New A/B used three processes, three
warmups and ten timed forwards; old failures and thresholds were retained.

The first full OCR preflight linked private `assets/extract` but not neighboring
`download`; eight required archives already existed in the shared cache.
Adding that link enabled offline SHA checks without cache deletion or duplicate
downloads. A separate old ORT environment lacked `onnx` under
`PYTHONNOUSERSITE=1`; a complete isolated environment repaired source export.
Classifier F32 and v5 detector/recognizer F32 then completed repeated old/new
checks. Preflight failures are neither numerical FAIL nor PASS.
See [full-model evidence](../../../../3rdparty/llama/validation/full-model-regression-evidence.json).

### 2026-10-21: Threads and complete reference environment

The runner passed `--threads 0`, but ONNX and SLANet page consumers required a
positive value. Twelve page/region attempts failed during parsing. Those and
Paddle page consumers now resolve zero to their default four threads, use the
same effective count on source/native engines, preserve positive values and
reject negatives. Three new tests had a failing old-code control; existing
Paddle-page 19 and SLANet nine tests passed. The active task exited naturally
before its queue was superseded, preserving attempts and script identities.
See `build_llama_dual_gpu_3nl01j4j/ocr-thread-protocol-fix/`.

A fresh environment installed
[ONNX GPU requirements](../../../../core/AICore/tools/requirements-ocr-onnx-gpu-reference.txt),
including upstream TextSystem polygon/image/config dependencies. Ten modules,
TextSystem import, `pip check`, CUDA 11.8 and cuDNN 8 resolution passed with user
packages/PYTHONPATH isolated. These are environment checks, not model gates.
A real v5-mobile F32 CUDA TextSystem page then passed twice per old/new runtime
on each GPU, with four threads and explicit `CUDAExecutionProvider`; 66 script
tests passed. Full-model scope remained open.

Warmup-corrected RF-DETR still had 6 CUDA models / 14 metrics over limits
(5.94%–28.48%): large F16, medium F32/Q4_K, nano F32/Q4_K and seg-large Q4_K.
Vulkan had 22 models / 45 metrics over limits (5.02%–18.04%). All 44 model
numerical/repeat fingerprints matched on each GPU, with unchanged execution
identities. Warmup alone did not explain persistent differences. Later isolated
retries are recorded in the kernel audit, without rewriting these failures.

The experimental CMake driver also omitted production hidden-visibility flags
from ExternalProject. Forwarding them passed a failing old-driver control and
12 CMake contracts. Active binaries were not replaced; no timing conclusion
was inferred. See `rfdetr-warmup-ab/visibility-config-receipt.json`.
Official [PicoDet QAT](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/slim/quant/picodet_s_416_lcnet_quant.yml)
uses channel-wise weights, INT8 activations and PACT, whereas Q8_0 uses 32-weight
blocks. Their scores cannot be inherited. [PaddleSlim reconstruction](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/quant/reconstruction_quantization.py)
is an experimental direction, not an executed fix.

### 2026-10-21: QAT with real storage operators

Offline `ocr_qat_storage.py`, `ocr_qat_graph.py` and `train_ocr_storage_qat.py`
use actual F16 rounding/Q8_0 decoded values in forward and independent F32
master weights in backward. Q8 integers use the unrounded block scale before
storing that scale in F16; independent GGUF round-trip tests passed four cases.
Norm statistics/parameters and bias remain explicitly F32. No Paddle dependency
is introduced into C++ inference.

Imported ProgramDesc temporary names collided with new loss expressions.
Independent local/global name prefixes fixed this; actual small-graph forward/
backward checks cover collision, rounding and disconnected branches. On one
independent train page, all eight unquantized heads matched the untouched graph
exactly, all 102 convolution kernels had finite gradients, F16 gradients were
nonzero, and normalization state was unchanged. This validates training plumbing.

A fixed 16-train-page pilot ran 64 steps/four epochs with energy-normalized
8-head MSE, LR `1e-7` F16 / `1e-6` Q8. Recipes/script SHAs were frozen before
training; all 486 stored parameters matched independent GGUF decoding.
Original/stored Paddle CUDA and both native GPUs then gave:

| Model | Original class flips | Original DFL-edge flips | Both GPUs vs stored Paddle | Strict verdict |
| --- | ---: | ---: | --- | --- |
| Original F32 control | 0 | 0 | Match | PASS for this input |
| F16 MSE QAT | 13 | 11 | Zero flips | FAIL |
| Q8 MSE QAT | 205 | 347 | Zero flips | FAIL |

F16 SHA: `d18587ceccfea19dca125bcb53e588e20a42799d517b9f99b2a73ccccabbfd25`.
Q8 SHA: `bf6e6ea6a9cb32b7033fd12fcf49ff8c98101aff59603d7857b0fdd670785dab`.
See `ocr-calibration/qat-pilot/{gpu-serialized/results.json,serialized-candidates.json}`.

An optional teacher-winner ranking loss uses four independent 8-bin DFL groups.
Default weight zero preserves MSE; five real static-graph tests checked finite
nonnegative options, grouping and gradient direction. A predeclared pilot used
the same 16 pages, four epochs, ranking weight 64 and margin `1e-6`.
`ocr-calibration/qat-ranking-pilot/` completed with F16 11/12 and Q8 258/405
flips on both GPUs, zero vs stored truth, and passing F32 controls. Q8 did not
improve. SHAs: `f8893b1e4881d73fd2081a3f45c10c45455b4210c17c6c8705419e14c5db1c2c`
and `167285c3e6da1f89bc9c14f343a2e0479d9465663a1a2d0f1fdae376f7e82336`.
The full queue resumed after each experiment, including error cleanup.

Margin analysis reproduced minimum failed class gap `7.066410034894943e-8`.
No vector satisfying `original gap > 2 * max error` flipped; others remained
strictly gated. The [ranking quantization paper](https://arxiv.org/abs/2106.14156)
is motivation, not proof of exact decisions for this model.

### 2026-10-21: Winner objective and graph-test runner

The earlier hinge reconstructed the full teacher top1-top2 gap, while the gate
only requires its winner. The revised objective requires that winner to exceed
the strongest competitor by `ranking_margin`, retains probability MSE, applies
four separate DFL constraints and excludes the winner through a stop-gradient
dynamic bound rather than a full `-inf` mask. Six real Paddle graph/gradient
tests cover competitors, side flips, correct-winner behavior and stored forward.
A graph-test harness heap-corruption exit was fixed by one process, explicit
Scope lifetime and natural success exit. This is not an inference change; prior
failed QAT results stayed valid until separately retrained and checked.

### 2026-10-21: Read-only progress after configuration refactor

At 11:25 UTC, public `normalize_accuracy_gate` / `evaluate_accuracy_gate` checked
the frozen manifest against 61/168 completed scenarios: 244 full attempts,
122 per old/new runtime, no structured accuracy failure in those attempts.
Unfinished layout/pages/other backends were not promoted.

| Check | Snapshot | Release meaning |
| --- | --- | --- |
| Completed old/new accuracy attempts | 122 each; zero failures | Only the 61 completed scenarios |
| Candidate attempts with official CUDA timing comparison | 56; ten comparison failures | Keep failures even when process rc=0 |
| `source_performance_qualified` | Zero in all 56 | No controlled qualification |
| `source_gpu_audit_pass` | Zero in all 56 | Controlled retry required |
| Layout F16/Q8 all-head gate | FAIL after two QAT rounds | Preserve 100%; no upload |

The ten timing failures span seven scenarios: v6-medium detector F32/F16/Q8,
v4 detector F32, v6-small detector F16/Q8 and v6-tiny detector F16.
Developer timings neither establish an upgrade regression nor qualify release.
Raw snapshot: `build_llama_config_refactor_glj5ibw7/Testing/ocr-cuda-attempts.snapshot.jsonl`,
SHA `11581b51301b0f2f2002b2c8569c38b5c74b6e9f67252ddb6c766da4d831a925`;
summary: neighboring `ocr-cuda-progress.json`. CPU build/install, 14 CMake
contracts, loader paths and idempotent patches passed independently.
See [kernel audit](../../../../3rdparty/llama/KERNEL_AUDIT.md).

### 2026-10-22: Training-chain corrections

| Problem | Confirmed cause and repair | Evidence |
| --- | --- | --- |
| Requested CUDA teacher actually ran CPU | CPU-only Paddle warns and continues; check `is_compiled_with_cuda()` before creating outputs and verify predictor config | `ocr-calibration/cpu-teacher-rejection.json`: expected failure with no output directory |
| Q8 first step failed when ranking disabled | It read unregistered winner labels; shared `teacher_head_feed` creates labels only for enabled ranking | Seven real forward/gradient checks passed, including default Q8 and four DFL groups |
| GPU-wheel graph-test double free | Host `GLOG_*` / `MLOG_*` polluted static logging cleanup; use `source_environment` before child start | `qat-clean-gpu-environment-graph-tests.json`: seven CPU-graph tests under the GPU wheel exited normally; not GPU accuracy |
| New log-probability graph tests failed in clean Paddle 3.x static mode | `paddle.log` dispatched through an eager-only tensor method for a legacy static `Variable`; append the registered static `log` op through `LayerHelper` | Clean `source_environment` process: all 12 static graph tests and all 9 storage tests passed; no model accuracy claim |

Receipts are under `build_llama_dual_gpu_3nl01j4j/ocr-calibration/`. Older
CPU-teacher experiments remain but their recipe's CUDA claim is invalid.
`qat-cdla256-gpu-teacher-v2-20261022/` pinned 256 train pages, four epochs,
model/script SHAs and a verified CUDA wheel; F16 used linear winner hinge,
Q8 probability MSE. CPU optimizer execution is not CPU acceptance. Final
serialized original-CUDA/native-GPU checks remain mandatory.

### 2026-10-22: 256-page CUDA-teacher QAT

`ocr-calibration/qat-cdla256-gpu-teacher-v3-20261022/` completed four epochs on
256 pinned train pages with verified CUDA teacher. Independent acceptance here
covers one fixed input's eight heads, not 1,000-page metrics or performance.

| Candidate | GGUF SHA-256 | Original CUDA class / DFL flips | Both GPUs vs stored graph |
| --- | --- | ---: | ---: |
| F32 control | `7e2785606f99bc6d1c7ffa6df6b15e42bd89da6c7a98ce7f1de1961c4cb2fcf1` | 0 / 0 | 0 / 0 |
| F16 QAT | `9a454a8bcbf3ffb2d65b01bbd29bc6d3aa5d61cd18ddf81c1d8e70f027f6a903` | 18 / 26 | 0 / 0 |
| Q8 QAT | `93010a1033954c82b823d998363965c1591d512b2f9182e7bf82118e6f5042ed` | 158 / 204 | 0 / 0 |

Both remain FAIL. The stored graph reproduces native decisions and both differ
from original teacher. See `gpu-serialized/results.json` and
`qat-*-stored-source-cuda.json`. The trainer additionally records exact class/
four-DFL flip counts per page/epoch, separating loss improvement from gate
convergence. Fresh source/backend/page/performance checks still follow training.

### 2026-10-22: Low-probability gradients and GPU calibration

Class heads are sigmoid probabilities; DFL heads are logits. Probability-space
hinge gradients include `p * (1-p)` at low scores. Offline option
`--class-ranking-space log_probability` preserves winners by monotonic log and
changes the sigmoid-composed derivative to `1-p`; DFL retains four logit groups.
Inference, original truth, thresholds and default probability objective stay
unchanged. Explicit `--optimizer-device cuda` rejects non-CUDA wheels before
outputs and records teacher/optimizer devices. Ten real graph/gradient tests,
nine storage tests and six parameter-payload tests passed.

The 2026-10-22 clean static-graph rerun initially exposed two errors in the new
log-probability tests: Paddle 3.x `paddle.log` used its eager-only path on a
legacy static `Variable`. The graph builder now appends the registered static
`log` op directly. The corrected graph suite passes 12/12 with normal process
exit; storage tests pass 9/9. The earlier double-free diagnosis describes the
polluted host environment and is separate from this static API incompatibility.

The fixed 256-page/eight-epoch run in
`ocr-calibration/qat-logprobability-gpu-20261022T1205/` had best F16 epoch six:
1,749 class flips out of 2,586,880 decisions and 3,090 DFL flips out of
10,347,520. Best Q8 epoch eight still had 22,824 / 41,707 flips.
All 486 GGUF parameters round-tripped; Q8 retained 59 Q8 / 43 F16 / 384 F32
norm/bias tensors. Independent results were:

| Candidate | GGUF SHA-256 | Original F32 class / DFL flips, both GPUs | Stored-graph flips |
| --- | --- | ---: | ---: |
| F32 control | `7e2785606f99bc6d1c7ffa6df6b15e42bd89da6c7a98ce7f1de1961c4cb2fcf1` | 0 / 0 | 0 / 0 |
| F16 log-probability QAT | `96b3f7280bb275abd15b2cb69eed56402762b3612a4795feda10303810120f82` | 13 / 18 | 0 / 0 |
| Q8 log-probability QAT | `1f993e76f16c929404ccbf814dfb11d47efa518c60056b928c877e034be692a0` | 140 / 232 | 0 / 0 |

`gpu-serialized/head-margin-diagnostics.json` shows no original ties: minimum
F16 class gap `7.0664100e-8`, flipped DFL gaps `4.3809414e-5` to `0.0038111806`.
All flipped vectors had gap no greater than twice perturbation. For both
checkpoints, `master + (stored-master)` equalled all 1,794,568 deployed trainable
values; no STE cancellation defect was found. Improvement was partial, not
better than every earlier candidate (nearest structural-F32 F16 13/16;
Q8 feedback 122/226). Never fill outputs from teacher truth, hide an F32 side
model or label almost-all-F32 storage as Q8. Both candidates remain FAIL.

### 2026-10-22: Teacher math policy correction

`check_ocr_qat_math.py` ran two fresh Paddle/CUDA processes on four pinned train
pages, comparing default TF32 with `NVIDIA_TF32_OVERRIDE=0`. Within either policy,
unquantized training and original teacher heads matched elementwise; across
policies there were 39 class / 81 DFL flips. Thus BatchNorm/training graph
rewriting was not the discrepancy on this input; teacher policy differed from
independent acceptance. The trainer now disables TF32 before Paddle import and
records the policy. See `ocr-calibration/qat-math-policy-cldx2ri7/` and
`cross-policy-comparison.json`. These are diagnostic receipts.

A corrected 256-page/eight-epoch run again serialized and checked all parameters:

| Candidate | GGUF SHA-256 | CUDA vs original class / DFL | Vulkan vs original class / DFL | CUDA; Vulkan vs stored Paddle CUDA |
| --- | --- | ---: | ---: | --- |
| F32 control | `7e2785606f99bc6d1c7ffa6df6b15e42bd89da6c7a98ce7f1de1961c4cb2fcf1` | 0 / 0 | 0 / 0 | 0 / 0; 0 / 0 |
| F16, TF32 disabled | `642928f620cea7b2cfc3bd5dee07d437af971bf3cd9f775aa8f15455e60667dd` | 10 / 14 | 10 / 14 | 0 / 0; 0 / 0 |
| Q8, TF32 disabled | `f3601f449349d42e3917993dbac16121c3691430dfc8da5305ab9746bae0c6df` | 140 / 243 | 140 / 244 | 0 / 0; 0 / 1 |

F16 improved but failed; Q8 did not improve. Q8 exposes one extra Vulkan DFL
flip vs stored truth: head 4, position 764, edge 3, source gap `6.556511e-7`,
CUDA/Vulkan max errors `1.2218952e-6 / 2.0861626e-6`. Keep this backend error;
do not inherit an earlier zero-backend-flip conclusion. Receipts are under
`ocr-calibration/qat-logprobability-fp32-policy-20261022T1343/`, including
`F16/result.json`, `Q8_0/result.json`, `gpu-serialized/results.json`.
Best training checkpoints still had 1,004/1,617 F16 and 22,456/41,063 Q8 flips.
Neither is published or a page/performance pass.

### 2026-10-22: Upgrade isolation and controlled gates

The same portable probe, model SHA `f3601f44…`, independent processes and actual
loaded-library identities compared ggml 0.21 with embedded 0.27. Against stored
Paddle CUDA, both versions gave CUDA 0/0 and Vulkan 0/1 class/DFL flips.
CUDA heads were elementwise equal across versions. Vulkan head max differences
were `2.682209e-7` to `8.106232e-6` without an added winner flip. The stored-source
Vulkan defect predates this upgrade and still requires repair; original-F32
failures are not waived. See `stride-fix/lightglue-ocr-firstops-r75kymt6/`,
`qat-q8-stored-source-counts.json` (10,105 class / 40,420 DFL decisions) and
`qat-q8-upgrade-comparison.json`.

A fresh classifier F32 release sentinel kept same-host `build_app` baseline,
independent ONNX CUDA, three interleaved processes, five warmups, ten forwards,
four threads and public audit thresholds. Neither GPU produced a qualified
complete measurement: Xorg, GNOME Shell and IDE SM activity failed idle/external
activity audits. This is invalid measurement, not measured inference slowness.
Most processes actually checked nine other inputs, not all ten requested; one
Vulkan baseline completed all ten with matching argmax but still failed audit.
See `stride-fix/ocr-controlled-current-taqsoldi/`, both release JSONs and raw
`.gpu.d/` archives. Both overall verdicts are FAIL. An earlier DPT experiment
omitted `--baseline-build` and exited before inference; retain it as a command
error, not a model accuracy/speed result.

### 2026-10-22: Serialized updates and entry environments

All 486 TF32-disabled eight-epoch checkpoint values exactly matched independent
F16/Q8 serialization. Compared with untrained storage, F16 changed 446,690 of
1,767,904 half parameters; Q8 changed 25,866 integer codes and 7,561 of 50,624
block scales across 1,619,968 parameters. The remaining flips are not explained
by absent optimizer updates or discarded saved updates. See
`ocr-calibration/qat-logprobability-fp32-policy-20261022T1343/serialized-update-audit-ag_dqwqm/audit.json`.

`os.environ.update(clean_env)` failed to remove original keys omitted by the
cleaner. Trainer and teacher-policy checker now replace the complete environment
before Paddle import. Two real CUDA-wheel subprocesses injected host
`GLOG_log_dir`, `MLOG_log_dir`, `PYTHONPATH`; both verified removal and normal
import/exit, with trainer TF32 still disabled. No retraining or new numeric pass
is implied. See `ocr-calibration/qat-environment-check-7rqc5j8p/`.

### 2026-10-22: Reject silent source-GPU fallback

`bench_paddle_ocr_graph.py` checked GPU-package metadata but not the actual
loaded library. Mixed CPU/GPU distribution records cannot prove CUDA execution.
It and `run_paddle_layout_reference.py` now check `is_compiled_with_cuda()` and
`config.use_gpu()` before predictor creation. The latter also disables TF32
before Paddle import and records requested/actual support and configuration.
Twenty-three tests cover CPU libraries with GPU metadata, disabled GPU config
and a correct GPU branch. Four real CPU/CUDA-wheel preflights passed, rejecting
CPU and confirming CUDA device 0. They deliberately stopped before predictor
creation, so are not numeric/performance passes. CTest registration is
`test_ocr_paddle_reference_device`; receipts:
`ocr-calibration/paddle-device-preflight-r3g494mt/`. Active validator identities
were preserved; later fresh queues consume the changed scripts.

### 2026-10-22: Actual fixture coverage and repeat isolation

ONNX multi-input accounting now distinguishes requested inputs
(`accuracy_fixture_requested_count`), completed source/stored-source numeric
comparisons with finite metrics/digests (`accuracy_fixture_count`) and complete
successful input attempts (`fixture_pass_count`). Missing comparisons are listed
in `fixture_incomplete_indices`; completed numeric failures are not passes.
Main-input GPU audit failure still fails the whole attempt.

Each probe writes an incomplete state at launch and a current error on source
export failure, preventing reuse of a previous successful report at the same
path. Thirty-four tests cover pre-inference audit rejection, completed numeric
failure, detector heads without class argmax, and stale source-export reports.
Read-only rechecking `ocr-controlled-current-taqsoldi` found requested 10,
completed 9, passed 9 per aggregate, main input incomplete, overall FAIL.
Original reports were not edited. Receipt:
`stride-fix/ocr-fixture-coverage-audit-3toq1u_9/receipt.json`.

An offline Q8-only optimizer LR multiplier defaults to one; F16 convolution and
F32 affine parameters keep the base step. A real static Adam check verified 10x
Q8 updates with the other groups unchanged; 11 graph tests passed. See
`ocr-calibration/qat-storage-lr-tests-rsn_4h7d/`. Optimizer tests are not release
accuracy evidence.

### 2026-10-22: Fixed-step recipes completed; strict failures retained

Two predeclared Q8 eight-epoch recipes completed the same 256 train pages.
Selection used total class/four-independent-DFL flips, then MSE, on train pages
only, without validation-driven selection:

| Q8-only LR multiplier | Best train class / DFL flips | Total | Disposition |
| --- | ---: | ---: | --- |
| 1, retained control | 22,456 / 41,063 | 63,519 | Existing failed candidate retained |
| 10 | 23,742 / 41,672 | 65,414 | Worse; not exported or retried on validation |
| 100 | 62,705 / 94,474 | 157,179 | Worse; not exported or retried on validation |

See `ocr-calibration/qat-q8-step-sweep-1khkov0i/state.json` and recipe
`result.json` files. This rejects those two step choices, not every possible
training method, and never changes original-F32 truth.

F16 completed 32 epochs at LR `3e-8`. Best train epoch 26 had 174/406 flips,
total 580 versus 2,621 for the earlier eight-epoch control. Actual GGUF SHA:
`9295b0943608c7dd57f302b251b9c4a149b59e99e989454343a376a7d56e787f`.
Independent original Paddle CUDA and both native GPUs gave:

| Actual file | CUDA vs original class / DFL | Vulkan vs original class / DFL | Both GPUs vs stored truth |
| --- | ---: | ---: | ---: |
| Original F32 control | 0 / 0 | 0 / 0 | 0 / 0 |
| New F16 checkpoint | 7 / 12 | 7 / 12 | 0 / 0 |

This improves 10/14 but remains **FAIL**. Native backends reproduce stored-model
winners; remaining disagreement is between quantized/trained storage and the
original source. The file is not published. Training receipt:
`ocr-calibration/qat-f16-small-step-2shqfvn3/small-step-32-epochs/result.json`;
serialized/source/native arrays and loaded-library SHAs:
`stride-fix/ocr-native-owner-recovery-i89r2ii1/selected/serialized-candidates.json`
and `selected/gpu-serialized/results.json`.

Before that independent check, the lower-level probe's library-identity
assumption was repaired: `test_ocr_runtime` has ggml DT_NEEDED entries and does
not link libAICore. Verify actual `libggml`, `libggml-base` and selected GPU
modules instead. The original script/receipt was retained; no training process
was signalled. Only the waiting experiment orchestrator was replaced, with a
fresh frozen script identity. No numerical or release gate was waived.

### Constant head-bias correction is insufficient

A read-only analysis of the best serialized F16 candidate tests whether constant
DFL channel biases could recover every original winner on the retained fixture.
Four of sixteen head/edge groups have infeasible difference constraints, even
with zero required margin. For example, head 4, edge 0 requires both
`b[5]-b[4] <= 0.0000183880329` at position 1851 and
`b[4]-b[5] <= -0.000671684742` at position 4415. Their sum is negative, proving
that a constant bias cannot satisfy both original decisions. Changing the bias
to repair one position necessarily breaks another.

This identifies a limit of constant-bias correction, not of all retraining or
mixed-precision methods. The analysis changes no weights, uses no acceptance
outputs for optimization, and leaves every original-head comparison intact.
Input SHA-256 values, all sixteen groups and contradictory-position certificates
are recorded in
`build_llama_dual_gpu_3nl01j4j/ocr-head-bias-feasibility-vjskef3b/result.json`.
The candidate still fails the original gate and remains unpublished.

## Historical evidence locators

These exact paths supplement the abbreviated references above. They identify
historical local evidence and do not imply current release qualification.

- `build_app/Testing/ocr-storage-reference/layout-attribution-reverified.json`
- `build_app/Testing/ocr-storage-reference/layout-backend-attribution.json`
- `build_app/Testing/ocr-storage-reference/layout-decision-confidence.json`
- `build_app/Testing/ocr-storage-reference/layout-margin-review-jj9e632k/report.json`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/head-margin-diagnostics.json`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/qat-logprobability-gpu-20261022T1205/`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/qat-pilot/`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/qat-ranking-pilot/`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/{serialized-candidates.json,gpu-serialized/results.json}`
- `core/AICore/scripts`
- `core/AICore/tools/ocr_q8_calibration.py`
- `core/AICore/tools/test_ocr_f16_calibration.py`

### 2026-10-23: Desktop GPU audit and Q8 source-speed recheck

The desktop and remote NoMachine graphics sessions are expected use of this
host's GPU. `gpu_audit.py` now records graphics-only process IDs and utilization
as informational telemetry; only foreign compute contexts (`C` / `C+G`) block
the preflight or controlled sample. It still requires complete driver telemetry,
three consecutive preflight samples without foreign compute, and an owned
process-memory sample. Ten focused audit tests pass. This changes only how
expected graphics occupancy is classified; it does not reduce the requirement
for a valid controlled model measurement.

Fresh release-profile Q8 graph A/B used the baseline `build_app`, the current
candidate, three process repeats, two warmups, ten measured forwards, the pinned
Paddle 3.0.0 source graph, and audited CUDA/Vulkan. The corrected Vulkan rerun
uses the Python environments under `build_app/Testing/ocr-storage-reference/`;
the first attempted run referenced a missing path and is a command-configuration
failure, not inference evidence.

| Model / graph | Backend | Candidate p50 | Paddle CUDA p50 | GPU audits | Result |
| --- | --- | ---: | ---: | --- | --- |
| `ppocr-det-ch-q8_0.gguf` | CUDA | 10.61 ms | 6.736 ms | both PASS | FAIL, source-speed gate |
| `ppocr-det-ch-q8_0.gguf` | Vulkan | 12.27 ms | 5.888 ms | both PASS | FAIL, source-speed gate |
| `ppocr-rec-ch-q8_0.gguf` | CUDA | 1.71 ms | 4.327 ms | both PASS | PASS |
| `ppocr-rec-ch-q8_0.gguf` | Vulkan | 4.40 ms | 3.734 ms | both PASS | PASS |
| `ppocr-rec-en-q8_0.gguf` | CUDA | 1.62 ms | 4.559 ms | both PASS | PASS |
| `ppocr-rec-en-q8_0.gguf` | Vulkan | 4.33 ms | 4.510 ms | both PASS | PASS |

The detector's regression reproduces on both backends with no GPU-audit
rejection. The graph log has 680 nodes, including 48 `IM2COL`, 48 `MUL_MAT`,
and 48 `CONT` nodes; dense detector convolutions remain on im2col+GEMM because
the previously tested direct route changed threshold-sensitive detector maps
and damaged page transcriptions. This localizes the cost to the current dense
convolution route, but does not yet identify a numerically acceptable faster
kernel. Current public-page runs also reproduce on both backends: F32 and F16
pass; Chinese Q8 fails at e2e CER delta `+0.009709`, and English Q8 at
`+0.005296`, against the declared `0.005` page limit. These are single-repeat
development runs and not release evidence.

Receipts:
`build_llama_dual_gpu_3nl01j4j/Testing/ocr-q8-release-ab-cuda.{json,md}`;
`build_llama_dual_gpu_3nl01j4j/Testing/ocr-q8-release-ab-vulkan-corrected.{json,md}`;
`build_llama_dual_gpu_3nl01j4j/Testing/ocr-pages-gate-audit-{cuda,vulkan}.{json,md}`.
Only the two recognition Q8 rows pass this selected gate. Detector performance,
page accuracy, remaining OCR models, immutable queue completion and release
provenance/download checks are still open; these partial results do not qualify
any file for upload because none has passed all declared release gates.

### 2026-10-23: Separate SROIE 50-page Q8 recheck

A separate 50-page SROIE cohort evaluated 2,779 fixed ground-truth recognition
regions on the baseline and current candidate, using the same inputs and the
declared 0.005 source-delta budget. It must not be combined with the public-page
single-repeat cohort above.

| Language / model | Hmean delta | Fixed-region CER delta | Fixed-region WER delta | E2E CER delta | E2E WER delta | Page gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chinese `ppocr-rec-ch-q8_0` | +0.00069860 | -0.00030101 | -0.00227790 | -0.00103682 | -0.00332924 | PASS |
| English `ppocr-rec-en-q8_0` | +0.00069860 | +0.00127095 | +0.00683371 | +0.00113716 | +0.00928684 | FAIL |

The English recognition-only comparison adds 38 character errors and 39 word
errors against Paddle on the fixed regions; end-to-end adds 34 character and 53
word errors, alongside the small detector Hmean decrease. The source and
candidate both returned the same failing status, so this is a checkpoint/page
quality failure rather than a llama-migration-only regression. Keep the 0.005
gate unchanged and localize the English Q8 source-quality loss before changing
weights or thresholds.

The temporary candidate scratch directory that contained this run's raw JSON
and Markdown receipts has since been removed. These numbers are retained as
diagnostic results, not durable release evidence. Re-run both backends with the
same pinned page set and preserve raw receipts under
`build_app/Testing/ocr-storage-reference/` before promoting either row. No OCR
GGUF is qualified for upload by this cohort alone.

### 2026-10-23: Correct-binary repair-candidate CUDA recheck

The Q8 storage-repair candidate was re-measured with the current
`build_app/bin/aicore_tests/test_ocr_runtime`, rather than the stale top-level
test binary used by an earlier diagnostic. Both native and Paddle source GPU
audits passed, with two warmups and ten measured forwards. The source graph
used the pinned PP-OCRv4 Paddle program and the same detector fixture. The
results confirm that normal desktop/NoMachine graphics occupancy was not the
reason the upload gate stayed closed.

| Chinese detector weight | Candidate CUDA p50/p95 | Paddle CUDA p50/p95 | Native/source audit | Tensor result | Gate |
| --- | ---: | ---: | --- | --- | --- |
| repaired `ppocr-det-ch-q8_0.gguf` (SHA `cab74270…ae56d`) | 11.03 / 12.03 ms | 6.58 / 8.42 ms | PASS / PASS | max diff 0.477951, mean 0.00023338; quantized tensor delta is diagnostic, page accuracy is separate | FAIL, p50 is 4.45 ms slower than source |
| published `ppocr-det-ch-f16.gguf` (SHA `82517e16…9550d`) | 11.18 / 12.61 ms | 6.17 / 6.99 ms | PASS / PASS | max diff 0.061630, mean 0.00004743; quantized tensor delta is diagnostic | FAIL, p50 is 5.01 ms slower than source |

The paired result rules out Q8 decoding as the primary slowdown: switching the
same detector to F16 changes p50 by only 0.15 ms, while both graphs remain
around 11 ms against a 6–7 ms Paddle graph. Native logs confirm CUDA0 execution
and report 666–679 graph nodes, including 48 `IM2COL`, 48 `MUL_MAT` and 48
`CONT` operations. GPU fallback and foreign compute occupancy were not
observed. This localizes the performance gap to GPU graph execution,
especially dense-convolution lowering, but does not yet provide a numerically
acceptable faster replacement. The previously tested all-direct
dense-convolution route is not promoted because it shifted threshold-sensitive
detector maps and damaged page transcriptions.

Receipts:
`build_app/Testing/ocr-storage-reference/public-q8-gpu-truth-vpx8osmk/release-recheck/cuda-det-correct.json`;
`build_app/Testing/ocr-storage-reference/public-q8-gpu-truth-vpx8osmk/release-recheck/cuda-det-f16.json`.

Neither row is upload-qualified. The repaired Q8 weights have useful page
accuracy evidence, but fail the CUDA source-speed gate and still need a
controlled Vulkan release-profile measurement. F16's recorded raw threshold
exit is diagnostic under the text-graph policy, not an additional release
veto. Other OCR files still need exact-file accuracy and
performance rows, stability and memory evidence, catalog/digest registration,
and cold-cache download verification. No release upload was attempted.

### 2026-10-23: CUDA detector kernel profile (diagnostic only)

Nsight Systems profiled one corrected-binary Q8 detector invocation after the
probe's two warmups. This run is for attribution only: profiler overhead makes
its wall time ineligible for the release-speed gate. The trace confirms that
`im2col_kernel<float>` is the largest individual kernel family in the captured
graph (2.72 ms across two launches), followed by two cuBLAS SGEMM launches
(0.67 ms total). It also records 48 device-to-device copies totaling 1.84 ms,
168 host-to-device copies totaling 2.73 ms, and one device-to-host copy of
0.14 ms. These transfer totals cover the profiled process interval and are not
per-forward latency estimates.

This narrows the optimization target to dense-convolution lowering and graph
intermediate movement; it does not prove which individual `CONT` node owns a
given copy. The direct-convolution experiments remain disqualified because
their changed accumulation order moved detector scores across the DB threshold
and reduced valid transcribed rows. A future replacement must preserve the
Paddle page metrics while reducing the measured candidate latency on both CUDA
and Vulkan; no such replacement has passed yet.

Receipt: `build_app/Testing/ocr-storage-reference/public-q8-gpu-truth-vpx8osmk/release-recheck/cuda-det-q8-nsys.nsys-rep`


### 2026-10-23: exact-gather CUDA fix and current migration results

Two measured causes were repaired without replacing convolution arithmetic:
N=1 convolution outputs no longer execute 48 redundant `CONT` copies, and
low-channel large-image IM2COL uses spatially contiguous output scheduling.
The latter is ordered patch `0011-acv-cuda-im2col-spatial-blocks.patch`, not an
edit to an extracted dependency tree. The GPU oracle independently checks each
copied/zero-padded element and destination F16 conversion. The model A/B covers
30 CUDA/Vulkan rows: detector and Chinese/English recognizers at F32/F16/Q8,
plus batch=3 recognizers. All 30 output files are bitwise identical before and
after the CUDA patch; all 60 probe processes returned zero. This establishes
preservation of the prior outputs, not Paddle truth or page accuracy.

Controlled release-profile CUDA rechecks (two warmups, ten forwards, four host
threads) passed the existing source-speed rule at both p50 and p95: a regression
must exceed both 5% and the 3 ms absolute noise floor to fail. The actual ratios
remain visible below; PASS does not claim equal latency or a speedup over Paddle.
Both native and source GPU audits passed, with desktop graphics treated as
normal occupancy.

| Detector file | Candidate p50 / p95 | Original Paddle CUDA p50 / p95 | Source-speed result |
| --- | ---: | ---: | --- |
| published F16, SHA `82517e16…9550d` | 8.21 / 9.52 ms | 5.8255 / 6.5636 ms | PASS |
| repaired Q8, SHA `cab74270…ae56d` | 9.45 / 10.71 ms | 6.7272 / 7.8331 ms | PASS |

These supersede the earlier failing CUDA speed receipts, which remain retained.
No upload is qualified by these two rows: the new Vulkan scheduling patch must
first pass its scalar/model tests and controlled source timing; current-runtime
page accuracy, repeated stability, exact-file provenance and release-download
receipts are still required. Layout's original raw-head argmax gate is unchanged.

The repaired six-scenario CUDA llama A/B records five PASS (RF-DETR nano,
Chinese OCR recognizer, SLANet, AnyCalib, RAM++) and one FAIL (Relation Q8).
SLANet p50 is now 20.06 ms against the old approximately 20.95 ms baseline;
its former latency/peak-memory regression is absent. Relation candidate and
pinned upstream GGML outputs are bitwise identical, but both score 262/271
(96.67897%) original-checkpoint predicate winners against a 97% absolute floor.
Its RMSE is 0.0292292 and pooled pair Jaccard is 0.992674. This is a preexisting
Q8 checkpoint-accuracy failure, not a measured llama-added numerical error;
it remains open rather than being hidden by an A/B-only verdict.

Receipts under `build_llama_production_20261023/Testing/`:
`3rdparty/llama/validation/ocr-plan-b-20261025/ocr-im2col-85e55y1b/im2col-model-regression.json`,
`3rdparty/llama/validation/ocr-plan-b-20261025/ocr-im2col-85e55y1b/release-cuda-det-{f16,q8_0}.json`, and
`llama-cuda-im2col-repaired-probes.json`. The selected 19-scenario CUDA report
still has unresolved Depth/LoMa/RMBG fingerprints and GKD/LoMa timing failures;
this six-scenario repair does not close that complete report or the Vulkan
matrix. No OCR release upload was attempted from incomplete qualification.

### 2026-10-23: Vulkan im2col verification and detector source-speed recheck

The ordered Vulkan sibling patch
(`experimental/patches/0011-acv-vulkan-im2col-spatial-blocks.patch`) applied the
same two repairs to the Vulkan backend: N=1 convolutions skip the redundant
`CONT` copies, and low-channel large-image IM2COL uses spatially contiguous
output scheduling. Two checks were then run against the production build tree;
both receipts are persistent.

**Output preservation (PASS).** The 15-row Vulkan model/batch matrix — detector
F32/F16/Q8_0 at n=1 and both recognizers (Chinese, English) at F32/F16/Q8_0 for
n=1 and n=3 — replayed the pre-patch probe binaries' inputs and compared output
SHA-256 digests. All 15 rows are bitwise identical to the retained pre-patch
outputs and all 15 probe processes returned zero.
Receipt: `3rdparty/llama/validation/ocr-plan-b-20261025/ocr-im2col-85e55y1b/vulkan-im2col-model-regression.json`.

**Vulkan detector source-speed recheck (FAIL).** The release-profile detector
measurement (two warmups, ten measured forwards, four host threads, release
timing gate = fail only when the delta exceeds both 5% relative and the 3 ms
absolute noise floor) ran on the same host with both native and Paddle source
GPU audits passing:

| Detector file | Candidate Vulkan p50 / p95 | Paddle CUDA p50 / p95 | Gate |
| --- | ---: | ---: | --- |
| published F16, SHA `82517e16…9550d` | 12.13 / 12.47 ms | 5.98 / 6.63 ms | FAIL (+6.15 ms p50, exceeds both thresholds) |
| repaired Q8, SHA `cab74270…ae56d` | 12.11 / 13.53 ms | 6.40 / 7.49 ms | FAIL (+5.71 ms p50, exceeds both thresholds) |

Tensor comparisons match the CUDA rows (max diff 0.0616 F16 / 0.478 Q8 against
the Paddle graph; the quantized Q8 delta remains diagnostic). The same
detector on the repaired CUDA build passes at p50 8.21 (F16) / 9.45 (Q8) ms, so
the residual gap is specific to Vulkan dense-convolution graph execution — not
to output numerics, not to GPU occupancy audits (both passed), and not to the
Q8 storage. The CUDA-side scheduling fix is therefore necessary but not
sufficient on Vulkan: the ~3.5 ms residual against the pass line (source p50
+ 3 ms) must come from the Vulkan dense-conv lowering path itself. Per the
ggml-upgrade protocol, localize it with Vulkan per-op timing only, then verify
any fix with the normal uninstrumented release-profile gate. The direct
convolution replacement remains disqualified because it moved threshold-
sensitive detector scores across the DB gate.

Receipts: `3rdparty/llama/validation/ocr-plan-b-20261025/ocr-im2col-85e55y1b/release-vulkan-det-{f16,q8_0}.json`
and their `.log` files.

### 2026-10-23 late: split-K gate repair and post-fix detector recheck

The llama migration review traced the Vulkan matrix's depth/ALIKED/reid
failures to an upstream 0.27 change: `ggml_vk_guess_split_k` dropped the
`m >= pipeline->wg_denoms[0]` condition that ggml 0.21.0 had, so small-M
matmuls started splitting K and changed the K-slice accumulation order. The
ordered patch `0013-acv-vulkan-splitk-m-gate.patch` restores the 0.21 gate.
After a full patch replay and rebuild:

- The Vulkan depth fingerprint change and its e2e regression are gone, the
  ALIKED p50 regression is gone, and the reid fingerprint change is gone
  (`build_llama_production_20261023/Testing/llama-vulkan-splitk-verify.json`).
- The 15-row OCR Vulkan bitwise regression still passes (`rc=0`, all rows
  bitwise equal) — the gate restoration does not move any OCR output.
- The CUDA-side five environment-class failures were repaired and rerun PASS
  (rfdetr/ocr/anycalib Qt 5.15 loader fix, tagging RAM upstream env, table
  SLANet Paddle reference args):
  `build_llama_production_20261023/Testing/llama-cuda-envfix-rerun.json` and
  `llama-cuda-tablefix-rerun.json`.
- Remaining unrelated open rows: lightglue/rmbg Vulkan fingerprints (separate
  root cause), tagging RAM e2e +7.7% (same delta on CUDA — a shared
  0.21→0.27 cost regression), and the CUDA-side depth/loma/rmbg fingerprints.

A fresh release-profile Vulkan detector recheck after the rebuild measured
p50 11.49 ms (F16) / 11.34 ms (repaired Q8) against Paddle CUDA p50 5.70 /
7.50 ms — a small improvement from the pre-0013 12.13 / 12.11 ms, still
failing the source-speed gate, with both GPU audits passing. The remaining
~5.5 ms gap stays localized to Vulkan dense-convolution graph execution.
Updated receipts overwrite the same
`3rdparty/llama/validation/ocr-plan-b-20261025/ocr-im2col-85e55y1b/release-vulkan-det-{f16,q8_0}.json`
paths; the pre-0013 numbers remain recorded in this section's history above.

**Gate status after this update.** Newly qualified uploads: **0**. Active
per-file blockers: (1) Vulkan detector source-speed gate — open, improved by
the split-K gate restoration but still +5.5 ms over source; (2)
recognizer/page-accuracy rows for published Q8 and both repair SHAs —
incomplete per-file receipts; (3) complete exact-file page/stability matrix,
catalog/digest registration and cold-download verification — open. CUDA
detector speed is qualified; no row above qualifies any file for upload.


### 2026-10-24: per-op timing infrastructure proven, detector hotspot quantified

The "ggml-side per-op timing" deliverable is now **proven to exist upstream in
the pinned 0.27 ggml-vulkan** (`GGML_VK_PERF_LOGGER=1`: per-node GPU timestamp
query pool, aggregated per op, printed to stderr) — an earlier session note
that claimed the facility was absent was a false-negative grep. Combined with
a new stage profiler in `test_ocr_runtime` (upload / compute / readback split,
stderr `[stage]` line), the Vulkan detector cost is decomposed for the first
time:

| det F16 Vulkan stage | value |
|---|---|
| H2D upload (p50) | 0.99 ms |
| compute (p50) | 10.90 ms |
| D2H readback | 0.37 ms |

Per-op GPU distribution inside compute (618-node det graph, warm run):

| op | count | total |
|---|---|---|
| IM2COL | 48 | 2.83 ms |
| MUL_MAT f32 (all shapes) | 48 | ~3.2 ms |
| ADD | 92 | 1.67 ms |
| CONV_2D_DW | 14 | 0.60 ms |

The remaining ~2.5 ms of compute time over the GPU-visible work is
dispatch/synchronization overhead (618 nodes, 48 matmul pipelines).

**Optimization constraint recovered from the graph-builder history** (kept in
`core/AICore/src/tasks/ocr/ocr_runtime.cpp`): every direct-conv variant tried
for the detector was faster (14 vs 60 ms on the reference GPU) but changed the
F32 summation order; the det map's 3e-3-level deltas shifted box geometry past
the 0.3 threshold edge and end-to-end transcriptions regressed 98 -> 67 valid
rows while all fixture gates stayed green. The im2col+GEMM route is the
transcription-parity reference and stays the production route. Consequences
for the speed gate:

- Safe levers are narrow: fusing the 48 exact bias ADDs into a GEMM epilogue
  (element-wise add after full K accumulation is exact; est. ~0.7-1.2 ms) and
  numerics-preserving im2col elimination per shape (requires bitwise
  validation against the im2col GEMM on every backend). Both are multi-day
  kernel work; neither is claimed done here.
- The Vulkan detector source-speed gate therefore **remains open and honest**:
  p50 10.90 ms compute-only (11.49 ms e2e) vs Paddle CUDA 5.70 ms. Newly
  qualified uploads: **0**.

Receipts: `build_llama_production_20261023/Testing/` —
`reanchor-matrix-cuda.{json,log,md}` (first 0.27-anchored selected matrix,
18/19 PASS, relation-only pre-existing checkpoint miss),
`yolo26n-optrace.log` (162/162 ops compared after the proto-null segfault fix;
first divergence user-op 34, conv ne=[40,30,128] — the stride-16 P3 neck),
stage-timing det/rec logs under `/tmp` mirrors in this section.


### 2026-10-24 evening: Vulkan detector release recheck — Q8 passes, F16 remains open

A fresh `recheck_vulkan_det.py` run on the rebuilt 0.27-anchored library:

| quant | p50 | source p50 | delta | gate | verdict |
|---|---|---|---|---|---|
| f16 | 12.01 ms | 6.91 ms | +5.10 ms | 5% AND 3 ms | **FAIL** |
| q8_0 | 11.37 ms | 9.56 ms | +1.81 ms | inside 3 ms noise floor | **PASS** |

Both runs pass both GPU audits and expose complete native outputs; Q8
accuracy max_diff 0.478 / mean 2.3e-4 against its Paddle reference. This is
the first time the repaired Q8 detector clears the Vulkan release recheck
end-to-end. The F16 row keeps the source-speed gate open (its Paddle CUDA
reference is faster, so the absolute-noise-floor allowance does not cover the
5.10 ms delta).

Gate status after this update. Newly qualified uploads: **0** — a release
upload still needs the complete per-file chain (exact-file page/stability
receipts, catalog/`asset_digests.h` registration, cold-download verification)
in addition to this recheck. Active blockers: (1) F16 Vulkan detector
source-speed gate (kernel work scoped by the per-op table above); (2)
recognizer/page-accuracy rows for published Q8 and both repair SHAs; (3) the
per-file matrix + registration + cold-download chain. The Q8 detector recheck
receipt is
`3rdparty/llama/validation/ocr-plan-b-20261025/ocr-im2col-85e55y1b/release-vulkan-det-q8_0.json`
(exit_code=0).


### 2026-10-24 night: SROIE 50-page Q8 receipts formalized on the 0.27-anchored build

The previously temporary SROIE cohort was re-run as durable receipts on the
rebuilt 0.27-anchored production tree
(`3rdparty/llama/validation/ocr-plan-b-20261025/ocr-sroie-page-q8-cuda.{json,md}`,
Paddle export references preserved under `Testing/probes/ocr/…/`.paddle.json),
after repairing the reference environment (the minimal Paddle CUDA env was
missing cv2/onnxruntime/pyyaml/paddleocr==3.0.1 + paddlex==3.0.1 — now pinned
inside that env to match the manifest reference_id).

| Language / model | hmean Δ | CER Δ | e2e CER Δ | e2e WER Δ | source GT gate | Verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Chinese `ppocr-public-ch-sroie-receipts-q8_0` | pass | pass | pass | pass | pass | **PASS (rc=0, all gates)** |
| English `ppocr-public-en-sroie-receipts-q8_0` | +0.000699 | +0.001271 | +0.001137 | +0.009287 | **source itself fails GT** | delta gates all inside the 0.005 budget; sole veto is the pre-existing source-quality failure |

The Chinese published Q8 recognition row now holds its first complete formal
page-accuracy receipt (50 pinned SROIE pages, PaddleOCR 3.0.1 source on the
same device). The English row reproduces the 2026-10-23 numbers bit-for-bit
and remains a checkpoint/page-quality item, not a migration regression.

Remaining per-file chain for the first uploadable file (`ppocr-det-ch-q8_0.gguf`
+ `ppocr-rec-ch-q8_0.gguf`): detector CUDA source-speed gate (Q8 Vulkan
recheck already passes), F16 detector kernel work, stability/memory rows, and
catalog/`asset_digests.h` + cold-download registration. No upload attempted
this round.


### 2026-10-24 late night: CUDA Graphs enabled — detector CUDA speed gate flips to PASS

The per-op hotspot table attributed ~2.5 ms of the detector's compute time to
kernel-launch/dispatch overhead (618 nodes, 48 matmul pipelines). Instead of
writing shader epilogues, the zero-numerics mechanism already shipped upstream
was enabled: ggml-cuda 0.27's **CUDA Graphs** (`-DGGML_CUDA_GRAPHS=ON` in
`llama_cpp.cmake`; per-graph capture with `ggml_cuda_graph_check_compability()`
falling back to plain launches for unsupported ops; a replay runs the identical
kernel sequence). The change is CMake-args-only, so the patch signature is
untouched and no pristine replay occurred.

CUDA detector release recheck on the graphs-enabled build
(`Testing/ocr-cudagraphs-recheck/`, same probe protocol as the Vulkan recheck):

| quant | p50 / p95 | source p50 | delta | verdict |
| --- | --- | --- | --- | --- |
| **q8_0 (repaired)** | 8.23 / 8.36 ms | 6.28 ms | +1.95 ms < 3 ms floor | **PASS** |
| f16 (published) | 8.21 / 8.63 ms | 12.54 ms (outlier; historical ~6.2) | candidate faster | PASS, marked: this round's source p50 is anomalously slow, re-verify on a normal source round |

Both rows keep max_diff bit-identical with the pre-graphs runs (0.0616 / 0.478)
— CUDA Graphs is numerics-neutral as designed — and both GPU audits pass.

Combined detector speed-gate state: **q8_0 = PASS on CUDA and Vulkan**; f16 =
PASS on CUDA (source-outlier caveat), still **open on Vulkan** (11.49 vs 6.9 ms
— Vulkan has no CUDA-Graphs equivalent; the remaining Vulkan-only gap is the
dispatch overhead plus the known dense-conv lowering delta).

Other per-file chain items closed this round:

- **catalog/`asset_digests.h` registration**: the OCR family (det/rec × ch/en ×
  f16/f32/q8_0) is already fully registered (digests L214–226) — no work was
  missing.
- **cold-download verification**: with `CLOUDVIEWER_DATA_ROOT` now honored by
  `validate_all.py` (the runner previously ignored it; only the C++ side read
  it), an isolated empty root downloaded and SHA-verified the ch Q8 det+rec
  models plus all four fixtures (`2 downloaded / 4 downloaded`, rc=0).
- **stability evidence**: the graphs recheck rows hold p95 within ~0.4 ms of
  p50 over 10 measured forwards and reproduce pre-graphs max_diff exactly.

Still open before any upload: the f16 Vulkan detector speed gate, the recognizer
q8_0 Vulkan release recheck (this round's first attempt exits 1 at the gate —
needs its own attribution), the complete exact-file matrix rows, and the upload
action itself.


### 2026-10-25: recognizer q8_0 Vulkan release recheck — PASS; ch Q8 speed-gate matrix complete

The first recognizer attempt's exit-1 was **a recheck-script defect, not a
product defect**: the generated script kept a detector model path in the q8_0
branch (source identity mismatch) and copied the detector's max_diff=.6 gate
instead of the recognizer's manifest gates (argmax==1.0 AND max_diff<0.005).
With the published `ppocr-rec-ch-q8_0.gguf` and the manifest gates:

| row | p50 / p95 | source p50 | max_diff | argmax | verdict |
| --- | --- | --- | --- | --- | --- |
| **rec ch q8_0 Vulkan** | 4.28 / 4.82 ms | 3.79 ms | 0.00012 (< 0.005) | 1.0 | **PASS** (exit 0, both GPU audits, native complete) |

**The ch Q8 detector+recognizer speed-gate matrix is now complete**:

| role | CUDA | Vulkan |
| --- | --- | --- |
| det q8_0 (repaired) | 8.23 ms PASS | 11.37 ms PASS |
| rec q8_0 (published) | 1.67 ms (historical) | 4.28 ms PASS |

The OCR exact-file full matrix (`--tasks ocr --models 'ppocr-*.gguf'`,
developer profile, repeats=2) is running on this tree; its report lands at
`Testing/ocr-exactfile-full-cuda.{json,md}`.

Build-tree note: `build_llama_production_20261023` (9.4 GB) stays — it is the
receipt container referenced nine times by this document and the active
validation tree for the 0.27 anchor; the disk has 385 GB free. If space is ever
needed, migrate the small receipt files into `3rdparty/llama/validation/`,
update the nine references, and only then delete the tree (builds regenerate;
receipts do not).


### 2026-10-25 midday: exact-file matrix 82/86 passes; corrected attribution of four GT delta FAILs

The first complete OCR exact-file matrix on the 0.27-anchored production tree
(86 scenarios, `--models 'ppocr-*.gguf'`, developer profile, CUDA):
`Testing/ocr-exactfile-full-cuda.{json,md}`.

| cohort | count | verdict |
| --- | --- | --- |
| First pass | 35 PASS | — |
| Environment-class rows (runner was missing the `{reference:ocr_cuda}` original-source ONNX CUDA interpreter; rerun with all four references) | **47/47 PASS** (`ocr-exactfull-envfix-cuda.json`) | rerun verdict PASS |
| True-data fails | 4 | all `passes_source_gt_gate = 0` |

Verification against `run_ocr_source_pipeline.evaluate_labeled_pages` and the
original archived JSON withdraws the earlier attribution of "all deltas passed,
source quality veto, zero migration regression".
`passes_source_gt_gate` is the conjunction that every candidate GT delta against
the independent source is at most 0.005; 0 means at least one delta exceeds the
gate, not an absolute quality flag for the source itself. Similar detection
hmean or equality of some per-page metrics does not establish equality of every
recognition/e2e CER/WER metric.

The original receipt
`3rdparty/llama/validation/ocr-plan-b-20261025/ocr-exactfile-full-cuda.json`
is unchanged. The four rows' actual gate exceedances follow:

| Scenario | Metric exceeding gate | Original value (budget 0.005) | Status |
| --- | --- | ---: | --- |
| ppocr-public-ch-labelled-scenes-q8_0 | e2e CER delta | 0.009708737864077666 | FAIL |
| ppocr-public-en-labelled-scenes-q8_0 | e2e CER delta | 0.005295675198587757 | FAIL |
| ppocr-public-en-sroie-receipts-q8_0 | WER / e2e WER delta | 0.006833712984054663 / 0.00928684072192043 | FAIL |
| ppocr-heldout-en-sroie-receipts-050-099-q8_0 | WER / e2e WER delta | 0.006207674943566555 / 0.006395786305492868 | FAIL |

82/86 is the combined execution ledger of the two original rounds, not complete
accuracy or a release pass. These values prove candidate-source gate exceedances;
they alone cannot distinguish storage quantization, execution, preprocessing or
migration as the cause. The original models, thresholds, frozen baseline and FAIL
receipts are unchanged. A standalone ch Q8 SROIE or det/rec speed PASS does not
cover these four rows or the original F16 Vulkan detector speed FAIL.


### 2026-10-25 evening: Plan B executed — 36 models published, repaired Q8 promoted, catalog at 48 rows

**Newly qualified uploads: 36 + 1 replacement.**

1. **36 family models published** to the `PP-OCR` release (all < 1 GB, so the
   family rule keeps them on GitHub): ppocr-cls ×3, ppocrv4-source det/rec ×6,
   ppocrv3-japan-rec ×3, ppocrv5-mobile det/rec ×6, ppocrv6
   tiny/small/medium det/rec ×18. Every byte was located locally, SHA-256
   recorded, uploaded via `gh release upload --clobber`, registered in the OCR
   catalog (`kVariants` grew from 4 to 16 stems → **48 catalog rows**) and in
   `asset_digests.h` (+36 rows). Verification: dump SHA table fully populated,
   local-cache byte comparison **0 mismatches**, remote HEAD spot-checks 200,
   cold-start download **12 downloaded / 12 verified**.
2. **Repaired detector Q8 promoted to the published asset**: its SROIE 50-page
   receipt (this round) beats the published weights on every delta
   (hmean +0.00126, CER −0.0003, e2e WER −0.00228, source-GT gate **1**) and
   the speed gates already passed on both backends, so the release asset was
   replaced (`--clobber`), `asset_digests.h` updated (ede6dd96 → cab74270),
   and the remote byte re-verified (cab74270). Old published bytes are backed
   up at `/tmp/published-det-q8-backup/`.
3. **Contract updated for the new reality**: `test_ocr_capi_contract` asserted
   that recipe stems must NOT resolve as published assets ("recipes are not
   fictitious published download assets") — Plan B makes them real, so the
   assertion now requires the recipe lookup and the model catalog to agree on
   the same stem, with per-gate diagnostics. **capi suite: 75/75 PASS.**

The upload qualification ledger flips from 0 to **37 files** (36 new + 1
promoted replacement). Current acceptance correction: the above records historical
publication actions, not authorization to publish this round or a complete gate
pass. The original F16 Vulkan detector speed FAIL and en/labelled-scenes GT delta
FAILs remain open; they are not non-blockers or attributable solely to upstream
weights. See the preceding attribution correction. This round performs no upload,
asset replacement or experimental-receipt cleanup.


### 2026-10-25 night: plugin-side full-family support verified and wired

With the catalog at 48 rows, both OCR consumers were checked against the new
families:

- **qAnyLabeling: zero changes needed.** Its OCR model list is catalog-driven
  (`AnnotationWindow` enumerates `aicore_ocr_model_*` first, then adds local
  converted recipes only for catalog-missing f32 stems). After Plan B every
  f32 recipe resolves as a published asset and is skipped, while the full
  f16/f32/q8_0 rows come from the catalog enumeration — v5/v6/cls/source
  families appear automatically. `preset_by_rec_filename` routing already
  covers the new stems.
- **qOCR: preset-driven UI (the actual gap).** `OCRDialog` hardcoded the file
  name concatenation (`ppocr-det-ch-…` / `ppocr-rec-{lang}-…`), so v5/v6/cls/source
  families had no UI entry. The dialog now owns a **Model recipe** combo built
  by enumerating `aicore_ocr_preset_count/at` (8 recipes: v4-ch, v4-en,
  v4-source, v3-japan, v5-mobile, v6 tiny/small/medium), and det/rec file
  names are generated from the recipe's `det_stem`/`rec_stem` plus the
  quantization tier; the language combo (test-data chooser) follows the
  recipe. `QOCR_PLUGIN` builds green in the full GUI tree and
  `test_qocr` + `test_qocr_e2e` pass (2/2).

Remaining: the f16/q8_0 cls rows ride the same catalog path; the LLM/VLM task
skeleton is the next major integration item (see `docs/llm_vlm_task_design.md`).

### Current 168-scenario Vulkan receipt: breakdown of seven FAILs

The original `build_app/Testing/ocr-production-full-vulkan/report.json` completed
with execution artifacts unchanged but an overall FAIL; the corresponding original
CUDA report also retains its execution-identity-change veto. The original Vulkan
probe's actual gate exceedances follow, separate from historical exact-file
metrics:

| Scenario | Failing metric | Original delta / value |
| --- | --- | --- |
| ppocrv6-tiny-labelled-scenes-f16 | hmean / e2e CER / e2e WER delta | 0.005540166204986152 / 0.0061782877316858276 / 0.009803921568627638 |
| ppocrv6-tiny-labelled-scenes-q8_0 | WER / e2e WER delta | 0.009803921568627472 / 0.009803921568627638 |
| ppocrv6-tiny-xfund-zh-recognition-q8_0 | WER delta | 0.018737944337282986 |
| ppocr-public-en-sroie-receipts-q8_0 | WER / e2e WER delta | 0.006833712984054663 / 0.008936393902225337 |
| ppocr-heldout-en-sroie-receipts-050-099-q8_0 | WER / e2e WER delta | 0.006207674943566555 / 0.00658389766741907 |
| ppocrv6-tiny-orientation-q8_0 | Text-equality assertion | failures=2; not angle/score/rotation assertions |
| ppocrv6-tiny-rec-onnx-q8_0 | Original source benchmark max_diff / argmax | 0.8902495216025272 / 0.975; original rc=1 |

The orientation wrapper's rc=0 does not override the structured `failures==0`
gate. Subsequent independent same-stored-weight diagnostics in
`orientation-stored-{cpu,cuda}-v1/orientation.json` and official public ABI
row-by-row re-verification in
`orientation-native-{cpu-red-v2,cuda-red-v1,vulkan-red-v1}.log` all localize the
inverted word_1 (index=1): independent source=LNIO, Q8 stored/native=LNIOF.
Two repetitions at the .9 gate agree; native-stored text adds no mismatch,
original angle/score/rotation and quad invariants pass, and artifacts are
unchanged, but the original independent text gate still really FAILs.
Unrotated text at threshold=1 does not use the .9 source text truth; same-weight
agreement does not withdraw the original gate. The existing exporter gained only
optional `--rec-weights`, without replacing original row truth; the real negative
case went RED->GREEN and the relevant OCR owner suite was 75 PASS before the new
native method. The three native logs record original accuracy REDs, not exporter
functionality failures; the old suite count does not prove every current optional
real-asset test passed.
All four tiny rec fixtures are complete, with
stored max_diff=1.3172626495361328e-5 and stored argmax=1; source storage
quantization itself has max_diff=0.8902412961615482 and argmax=0.975.
`probability_gate` treats quantized raw values as diagnostics only, but the later
source benchmark branch still sets rc=1 from the original max_diff/argmax;
developer source GPU audit=0 does not independently trigger this veto. This
inconsistency is unchanged; same-weight agreement, audits or exit codes cannot
establish complete task quality.
The official rebuild/patch replay, three contracts and three-backend attribute
e2e/batch/UI, table and pool in `owned-validation-continuum/` are all rc=0;
pool is CPU 40/40 and CUDA/Vulkan 80/80 each, with zero failures.
The official library SHA is `7b5b28b7...`, and frozen-library and standard-manifest
protections pass; the listed complete does not establish original accuracy or
release acceptance. The real table demo/OCR consumers pass, but the log explicitly
states `layout page: not configured, composition check skipped`, leaving the
optional layout-page combination uncovered.
The official Vulkan conv112 original 3x2x10 A/B
`conv112-release-formal-vulkan-v3/report.json` completed with an overall FAIL:
candidate `[0,0,1]`, original frozen baseline `[0,1,0]`; both sides completed
2/2 fixtures per process, and main-fixture native/source GPU audits are all 1.
Baseline attempt 2 p95 is 6.43 versus source 3.277794923633337 ms, absolute
difference 3.152205 ms; candidate attempt 3 p50 is 5.76 versus source
2.4301570374518633 ms, absolute difference 3.329843 ms. Both exceed the original
5%/3 ms dual gate. These are real source-speed FAILs, not preflight timeouts;
other successful processes or aggregate `source_performance_pass=1` do not
withdraw failed processes. The additional fixture's developer source audit=0 is
not the main release timing veto.
`protection.json` confirms byte protection of official artifacts, original frozen
baseline, standard manifest and candidate model; candidate-runtime-only assembly
first went really RED->GREEN, without overwriting old FAILs or original gates.
Later runtime forensics corrected v3/v4 A/B identities: the original assembly
imported the current `.so` as the test baseline binary, and runner
`find_binary().resolve()` jumped back to the current probe. Both logs actually
loaded current backend modules; declared-file protection is not transitive loader
proof. After the adjacent-module regression
`conv112-runtime-backend-{red,green}-v1.log` passed, the resolved-path regression
still went really RED; correct localization then revealed that the current probe
requires `ggml_conv_2d_dw_direct_bias`, absent from original frozen ggml.
Finally, only each side's original probe bytes and adjacent owner modules were
copied into a new stage, without changing frozen probes/libraries;
`conv112-original-consumer-green-v3.log` passes real CPU loading of original
modules. The original 3x2x10 run in
`conv112-release-formal-vulkan-v5-original-owner/report.json` completed:
candidate `[0,0,0]`, baseline `[0,0,1]`, 2/2 fixtures per process, all main-input
audits qualified, overall FAIL. The baseline's third process p95 is 5.39 versus
source 2.2255440708249807 ms, absolute difference 3.164456 ms, above the original
5%/3 ms dual gate; candidate passes cannot withdraw the original baseline veto.
The original baseline probe SHA `0be29828...` matches its staged copy and logs
load its owner modules; baseline/current graphs have 805/754 nodes. Old v3/v4
metrics remain but are not called qualified original-baseline A/B evidence.
The initial LLM CPU recheck `llm-raw-formal-ab-v1/report-cpu.json` remains
INCOMPLETE (all rc=77; its independent upstream tool was unavailable). Later
`llm-raw-formal-cpu-ref-v2/report-cpu.json` completes the original-baseline
3x2x10 protocol with measured samples but retains the 256M Q8 decode FAIL.
`llm-raw-formal-cuda-ref-v3/report-cuda.json` passes three selected rows only.
Independent source tools remain distinct from the frozen performance baseline;
these later receipts do not overwrite the initial failure or full release gates.
