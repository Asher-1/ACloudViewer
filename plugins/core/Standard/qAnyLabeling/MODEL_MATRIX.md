# qAnyLabeling AICore model matrix

The model picker is driven directly by the public AICore catalogs at runtime.
未发布的 OCR/attribute 模型另由公共 task-owned recipe 描述；本地文件支持不产生
下载 catalog，也不证明量化精度或 release 资格。
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
| Source Chinese PP-OCRv4 / Japanese PP-OCRv3 / PP-OCRv5 / PP-OCRv6 Tiny, Small, Medium | Source detector/recognizer recipes with shared 0/180 text classifier | Editable text quads/descriptions; typed orientation results | 公共 OCR catalog 已发布 F32/F16/Q8_0 家族；原 source defaults 与内嵌字典；各文件精度门单独记录 |
| RelateAnything | Relationships between selected/detected rectangles | Relations referencing existing objects | Public relation catalog and optional catalog-backed detector |
| 本地 LLM / SmolVLM | `aicore://llm` Chat/VQA provider | 流式 UTF-8 文本与独立持有的结果 | 公共 `llm` catalog；视觉模型自动配对 projector |
| PULC person/vehicle（本地 F32 recipe） | 全图多属性标注；本地 cache 文件缺失时提示路径 | 公共 `attribute` typed 概率/属性 → 全图 rectangle、attributes、replace | owning task 的 source 身份与严格阈值/argmax；无已发布 URL；Q8/release 未通过 |

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
by the common runner, separately from graph gates. 当时 F16/Q8 候选因数值门失败而
未进入六个新配方选择器；当前 catalog/preset 已枚举发布家族，完整 48 文件/168 场景
双 GPU 复验现已执行结束、两份总门仍 FAIL，详情见下方当前收据。这是历史状态变更，
不代表失败的精度门已关闭。A one-page
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
service's `/models` endpoint or a user-entered ID. 远程 provider 不增加本地模型条目；
共享 ChatServiceClient 的 `aicore://llm` provider 则从公共 `llm` catalog 枚举模型，
通过 worker 调用 `aicore_llm_generate`，由任务持有模型及 projector。 Image Classifier AI Assistant/AutoRun use the source
category prompt and JSON flags, while its Local model menu retains the YOLO
classification path above. 本地 provider 不提供 source 的 exact Florence2 图，
也不建立与任意远程模型的输出等价性。正式 `build_app` 的 LLM CPU/CUDA/Vulkan
各 12/12 developer 场景已通过，收据为
`build_app/Testing/llm-production-full-{cpu,cuda,vulkan}.json`。正式受控 release A/B
收据位于 `build_app/Testing/formal-gates-resume/llm-release-*/report.json`：Vulkan
12/12 与总门 PASS；CUDA candidate 12/12，但冻结 baseline 四行文本失配及两个
p95 回归使总门 FAIL；CPU 最终正确性/稳定性 12/12，但 6 个场景的 10 项性能
超门、总 FAIL。两行受控复验
`build_app/Testing/llm-cuda-tail-controlled-recheck/report.json` 保留 Q8 p95 +8.28%
的总 FAIL，F16 原回归未重现；不替换完整 CUDA 原失败。
后续同一 raw consumer 的 `llm-raw-tail-pathfix/report.json` 两行 CUDA 原协议
3×2×10 受控 A/B 总 PASS，30 个 measured 样本/模型/侧完整保留；不撤销原完整
CUDA/CPU FAIL，不作性能因果结论。四份 loaderfix Nsight 收据区分主 consumer/reference child，
F16 主 GEMM 路由差异已实测；profile 含 load/warmup/recreate/cancel，不是 measured-only 资格。
后续 NVTX 诊断修正负返回值导致漏 pop：`llm-nvtx-paired-{current,frozen}-v6/` 的真实 CPU
区间数量/持续时间通过，主 measured CUDA 选择器正/负例 RED→GREEN，恢复执行后 3 项再次通过。
`llm-raw-full-cpu-pathfix/report.json` 两侧正确性/稳定性各 12/12，5 个场景的 6 项性能超门、
总 FAIL；`llm-raw-full-cuda-pathfix/report.json` candidate 12/12，但冻结 baseline 四行文本
失配及 500m newspaper F16 e2e p95 +5.71%（248.876 → 263.095 ms）使总门 FAIL。
四份 measured profile 首轮 JPEG 解码失败；补齐 Qt plugin 路径后的插入在任务开始前因外部 GPU
consumer 被拒绝，OCR 已恢复，无新归因收据。`owned-measured-profile-pluginfix-wait/status.json`
记录自动等待/串行采集状态；仅窄任务、不触发正式重建。该轮随后四份 probe 均成功，
四份 routes 因每范围 0 个 kernel 节点仍 FAIL；本机 Nsight 默认 graph 粒度不采集 graph
节点，真实每范围已有 522 个同线程 API。后续 `owned-measured-profile-nodefix-wait/`
的 node 重采已完成：四份 probe/routes 8/8 PASS，256M 的 40 个 measured 范围都有主
consumer kernel/同线程 API；`llm-measured-nodefix-resume-v1.log` 真实回归 4/4 PASS。
profiling_only/release_qualified=false 保持；不撤销原 FAIL，不关闭生产 CUDA Graphs。
500M 对应补采现已结束：`owned-measured-500m-nodefix-wait/status.json` 的四份
probe/routes 8/8 PASS，额外 40 measured 范围通过同一真实回归；首次外部 GPU 拒绝
保留、OCR 已恢复。500M F16 主 GEMM current/frozen 为 h1688/s1688；profile p95
172.421877/183.116541 ms，不作无插桩 release 或原 p95 失败已闭合的声明。
SmolVLM 不等同于 source Florence2。

本地未发布的 source YOLO11s 候选已完成 F32/F16/Q8_0 转换：source ONNX 与官方
checkpoint 在固定输入上的坐标/分数最大差异为 `0.00201416` / `7.39098e-6`。
隔离和正式 runtime 均使用原官方逐框门，经 common runner 验证
CPU/CUDA/Vulkan 各 3/3 通过。正式收据见
`build_app/Testing/formal-gates-resume/yolo11s-{cpu,cuda,vulkan}/report.json`；
隔离收据保留于 `build_app/Testing/g8-yolo11-exact/local-official-catalog-fixture-*.json`。
这是单一 checkpoint/输入的 developer 证据，不是生产 catalog、完整 YOLO11 家族
或发布性能证据；候选及其 SHA 位于同目录 `source-identity.json`。

PULC person/vehicle 的 exact source ONNX 已下载到 build-local 目录，直接执行
原 source 的预处理/属性解码取得独立输出；图含 32 个 Conv、27 个 BN、HardSigmoid
和多标签 Sigmoid。收据为 `build_app/Testing/g8-pulc-source/source-graphs.json`。
原 source 返回携带 attributes 的全图 rectangle、replace=True，不是单类 flags。
复用现有 ONNX lowering 的 OCR test adapter 数据面已执行 CPU/CUDA/Vulkan 各六行：
F32/F16 的原概率和属性选择通过候选门，Q8 person/vehicle 三后端均失败。
`build_app/Testing/g8-pulc-native-graphs/stored-attribution/report.json` 的独立同权重
ONNX 归因确认 Q8 权重量化最大误差 0.02084890/0.04891500，原生附加误差
≤1.55e-6；保留原 0.02 候选门和全部失败。这份 OCR adapter 收据不是公共
API/catalog、完整数据集或 release A/B 支持。
后续隔离 Q8 候选保留 person 1 个、vehicle 2 个 kernel 为 F16，分别仍有
1601536/1568768 个 Q8 元素；三后端 `native-*-q8-candidate/report.json` 各六行
候选门通过，Q8 最大误差约 0.01444/0.00667。选择依据、完整单 tensor 试验和
候选 SHA 在 `precision-sensitivity/report.json`。它仍是固定单图的存储候选，
没有替换旧失败、发布资产或证明 untouched holdout/release。
后续固定候选的 `holdout-cpu/report.json` 覆盖 11 张未参与选择的真实图片：
F32/F16 44/44 通过，Q8 仅 4/22 通过（总 48/66、FAIL）；person/vehicle 最大误差
为 0.0855134/0.0764870。原阈值和候选 SHA 不变，同存储附加误差及属性差异均记录。
这批图片无属性 dataset GT；单图混合候选不能提升为已支持/已发布模型。
后续公共 `attribute` task/C ABI 已实现，共享图解释器 owner 提升到
`common/paddle_graph`，OCR/table 复用该 owner。person/vehicle F32/F16 的三后端
公共输出、属性、stride、独立结果所有权及 live-context shutdown 已通过。
`pulc-public-holdout-{cpu,cuda,vulkan}/report.json` 各为 48/66、FAIL，Q8 原门、
候选 SHA、逐输出和属性失配保留。公共本地 F32 recipe/cache API 已供 UI picker
消费，不提供虚构下载 URL；qAnyLabeling CPU/CUDA/Vulkan 的真实基础 Run 验证
全图 attributes、replace、保存、模型切换、stale 拒收和 teardown 通过。
维护中的 F32 C ABI E2E、基础 table E2E 三后端通过；table 未配置 layout page。
公共 probe 的 pinned source identity/全部属性门真实负例 RED→GREEN。
初次六行 common runner CPU 收据为 0/6 FAIL（json0/json1 双来源与裸指标歧义），
原收据保留；隔离配置改完整 json0 路径后
`attribute-common-runner-{cpu,cuda,vulkan}-pathfix/report.json` 各 4/6：
F32/F16 通过，Q8 原 0.02 门仍 FAIL，不改变权重或阈值。
现有 `ANYLABEL_INFER` 原生命令增加 attribute TASK；public recipe/cache 提供本地
列表和默认模型，DOWNLOAD 对无 URL 的缺失模型明确失败。真实隔离插件 CPU/CUDA/Vulkan 的
HELP schema、两模型/两图、confidence 不改变属性、错误 manifest 测试通过；GPU 收据为
`shared-owner-acceptance/attribute-batch-{cuda,vulkan}-v2/receipt.json`，不是正式重建或 release。
`shared-owner-acceptance/attribute-ui-lifecycle-{cpu,cuda,vulkan}-v1/` 新增真实窗口验收三后端
通过两图 Auto Run、在途取消/不调度下一图、取消后重跑、源图/模型 stale、无效模型 worker
错误恢复和推理中销毁窗口。源码/消费者/模型前后 SHA 不变，未改生产逻辑；不是 live frame、
跨平台 installer、正式库或 release 精度/性能资格。GPU 空档验收结束后原 OCR 已恢复。
已发布 catalog/digest、独立属性 dataset GT、完整 live/cancel/UI 自动化与 release
仍待闭合。原 Linux L3 schema 16 PASS/1 FAIL 保留于 `automation-l3-schema-resume.log`。
实际配置 PLUGIN_STANDARD_QMAPANYTHING=OFF；原正式队列结束后启用并构建该消费者，
`mapanything-consumer-{configure,build}-v1.log` 成功，
`automation-l3-schema-formal-green-v1.log` 的全部 17 schema PASS、无 skip。
这不是跨平台 installer 或 packaged GUI L4 资格。
官方 PA100K 539463680-byte 归档经独立预记录 MD5 校验，SHA-256 为
`65e2ab37519aacf307c8c0ce04c6ab8edac3b4199e0448d4de625cfd60cd5cc0`。
`pulc-pa100k-gt-v1/dataset-identity.json` 固定 val/test 各 10000 张及 26 维 GT/每图 SHA，
未消费训练集。下载/覆盖拒绝/GT 准备/路径穿越负例回归通过；公共原始三量化两图 GT
回归 RED→GREEN。`pulc-pa100k-val-cpu-v1/report.json` 的 CPU val 10000 图三量化已完成：
F32 10000/10000 PASS（max 3.93391e-6）；F16 9969/10000，max 0.00992972 未超
概率门，但 31 图属性失配仍 FAIL；Q8 2707/10000、max 0.27434349，仍 FAIL。
source/模型/runtime/GT 前后 SHA unchanged；逐图和逐属性 GT 计数保留，不用局部
计数改善掩盖回退。`pulc-pa100k-test-cpu-resume-v2/report.json` 的 test 10000 图已结束：
F32 10000/10000 PASS、max 4.64916e-6；F16 9955/10000、45 图属性失配；Q8
2497/10000、max 0.34178510，仍 FAIL。两 split 共 20000 图/60000 行均完整且稳定；
收据审计四测试 RED→GREEN，Q8 同权重执行附加误差最大为 3.40e-6/3.87e-6，
存储概率失败分别 7275/7480 图。原模型/门不变，不设事后 release 下限，不是性能证据。
VeRi 官方流程要求申请，vehicle GT 仍缺，不代替用户接受访问条款。

YOLO26n 的公共 opaque options 新增 upstream/F32 PSA 累加请求，默认仍 upstream。
隔离 ABI/真实图传播测试通过。`build_app/Testing/yolo-attention-official-matrix/`
保留两模式三后端三量化的 18 行完整模型收据：Vulkan F16 原模式最大分数误差
0.0238298 FAIL，F32 累加模式 0.0122409 通过原 0.02 门；Q8 三后端两模式仍 FAIL。
这是单进程、两 warmups、十 forwards 的原 checkpoint 正确性证据，不替代冻结
release 性能 A/B；插件未改变默认，正式 runtime 未替换。独立 CPU 权重注入复验收据
`yolo-quantization-attribution-resume-v2.json` 核验 204 个真实 GGUF/checkpoint 参数映射：
F32 导出逐位相等、F16 权重量化分数误差 0.00776911、Q8 权重量化误差 0.14595002，
原 Q8 0.02 门仍 FAIL；这不排除 task 执行另有附加误差，也不授权替换原资产。
`yolo-quantization-execution-cpu-resume-v3.json` 用公共 CPU ABI/显式 mode 1 进一步测得
Q8 同权重执行附加分数误差 0.02269834、独立 checkpoint 总误差 0.12325168，仍 FAIL。
三量化逐框误差、原始十次样本与模型/runtime 前后 SHA 均保存；不将两个最大误差相减
归因，不把同权重一致性当独立真值或冻结 release 通过。
后端选择/独立误差记录真实测试 RED→GREEN，恢复参数套件 3/3 通过；GPU 六行
`yolo-quantization-execution-{cuda,vulkan}-resume-v4.json` 的 F32/F16 PASS。Q8 CUDA 同权重
分数误差 0.18532050、总误差 0.18347186、缺四框；Vulkan 0.00550228/0.15145230、五框匹配，
原总门均 FAIL，均十次稳定、资产前后 SHA 不变。CUDA 缺框不归为只有存储误差；当前源码
两 GPU 都 host 展开 Q8 为 F16。LLM CUDA arithmetic scope 11/11 PASS（全部 0 changed），
不替代原文本/p95 门；初次 optrace 装配退出 77 保留，模块旁重编译后两项 162/162、0 null，
CPU→CUDA 首超诊断 op61、Vulkan→CUDA op149，均 rc=1；见 `owned-yolo-q8-optrace-resume-v2/`。
新 `owned-yolo-q8-public-lifecycle-v1/` 公共 ABI RED：CUDA 冷 Q8 五框，经 F32/F16 切换后
Q8 一框，再建 context 仍一框；每 context 十二轮稳定，Vulkan PASS。禁用 IGEMM 同测试
PASS，定位到缓存路径；旧计划未消费 task 已写入的单调 plan_owner_id。有序 `0020` 仅
CUDA/combined，增加身份检查和 NHWC 失效处理，原正式树只读 apply --check 通过。
`yolo-igemm-owner-candidate-v1/` 四链正反向/幂等重放与单 TU 编译通过；原链接缺 80 个对象，
正从 verified combined replay 和同一 pinned llama parent 完整重建隔离 CUDA backend。
继承原 CUDA ISA/Graphs/算术选项，构建成功；最终 module SHA `4b6ad3c8...`、保护 unchanged，
同二进制 oracle SHA `900a4a2f...` 已装配 RED/GREEN。`owned-yolo-igemm-owner-green-v1/coordinator-3/`
已完成真实 oracle RED rc=1→GREEN rc=0，CUDA/Vulkan 公共模型切换/再建 context 生命周期 PASS。
后续 `owned-llm-flags-attribution-v2/coordinator-1/` 同一 CPU 生命周期亦 PASS，三后端齐全。
前两次插入拒绝与恢复收据保留；`yolo-quantization-execution-cuda-ownerfix-v5.json` 的 Q8 五框
匹配、十次稳定，同权重执行附加分数误差 0.0021265149，但独立 checkpoint 总误差
0.1480765343 仍超原 0.02 门、FAIL。资产/runtime 前后 SHA unchanged，编排成功不是精度 PASS。
隔离模块含 0019 但不替代 pool 正式门，不改模型/原门或正式/冻结产物。
`yolo-q8-single-tensor-sensitivity-resume-v1.json` 的全部临时内存单 tensor F16
恢复方案均未过原 0.02 门；最佳 `op.109.w` 最大分数误差 0.03684449196 仍 FAIL。
逐轮 Q8 baseline 恢复与资产前后 SHA 通过；不写模型、不称为原生/留出或 release。
指定 tensor 恢复/非法名称拒绝两真实测试 RED→GREEN；三个 `op.109.w` 配对
`op.110.w/op.107.w/op.20.w` 的误差 0.04792762/0.08504459/0.08508769 均 FAIL，
改善不具可加性。用户选择仅用原冻结基线，其 release veto 保持，不另建替代基线。
`yolo-attention-release/{cpu,cuda,vulkan}/report.json` 已完整结束：candidate 各 2/3
（F32/F16 PASS、Q8 FAIL），冻结 baseline 各 0/3，性能比较被 baseline status veto。
冻结 Vulkan F32 最大分数误差 0.0908023715 超原 0.005；总 release FAIL。
冻结 CPU F32 三进程均稳定且五框匹配，最大分数误差同为 0.1101961732；正常执行后
被原精度门拒绝，不是缺少 attention API 或 loader 错误。原 baseline/数值门保持，
不把候选稳定或 ABI 通过写成 release 性能通过。

conv112 存储候选的原页与独立 SROIE 精度门三后端通过；同 SHA、960 主输入、原 3×2×10
release A/B 的 CUDA 总 PASS，CPU p95 74.480 → 85.560 ms（+14.88%）总 FAIL。
Vulkan 首轮候选第三进程 source p95 超门；`conv112-release-vulkan-tail-recheck/report.json`
复验 candidate `[0,0,1]` / baseline `[1,0,0]` 总 FAIL：candidate attempt 3、baseline attempt 1
被外部 GPU 活动导致的 preflight timeout 拒绝，各只有 1/2 fixture，不是测得的 source-speed
FAIL；成功进程不补齐其 GPU 资格。baseline veto、原首轮性能 FAIL 和旧收据均保留。
最新 `conv112-release-vulkan-preflight-recheck-v2/report.json` candidate/baseline 各 `[1,1,1]`，
总 FAIL。candidate attempt 1 fixture 0 的 GPU/source 审计合格，p95 5.31 对 2.105774 ms
超原 5%/3 ms 双门；其余采样另有 preflight timeout、native/source 审计失败或 fixture 不全。
末次单 log 5.35/7.80 对 4.30/5.03 ms 不具完整 A/B 资格；不能把全部失败归为污染。
`0019` pool AVG 隔离 80 case GREEN；正式重建前独立 oracle 直接加载 `build_app/bin` 的
CPU 40/40、Vulkan 80/80 PASS，CUDA 80 case / 27 FAIL，真实消费者与 release 仍开放。
LLM CPU `llm-cpu-attribution-resume-v2.json` 审计原 12 行/720 次 measured、产物身份与输入/
输出可比性，六项原性能 FAIL 保留；六个分析器行为测试 RED→GREEN。Haswell 编译/ISA/
OpenMP 一致；llama/mtmd 232/69 个源文件全相同，CPU 69 文件仅 DW bias 分支不同，不能
据此断言整个 CPU 调用链或性能相同。冻结原 3×2×10 A/A 256M Q8 单行 PASS，仅为诊断；
完整 `llm-cpu-complete-post-aa-resume-v1/report.json` A/B 已结束，两侧各 12/12 正确且稳定；
500M newspaper Q8/F16 decode p50 +11.08%/+6.60%、256M newspaper Q8 e2e p95 +5.20%，
三项原性能 FAIL。`llm-cpu-complete-post-aa-attribution-v1.json` 的 720 measured 样本、产物
身份与编译参数审计通过，三项及旧六项 FAIL 保留，尚非 CPU release PASS，不替换原 baseline。
`llm-cpu-complete-flags-closure-v3.json` 扩展底层源码审计：scheduler/alloc/threading 字节相同；
Haswell/ggml-base flags 相同，AICore 的 AVX2/frounding-math/宏及直接依赖闭包不同。新增分模块
flags 两项负例 RED→GREEN、分析器共 8/8；不是性能因果证明。非特权 CPU 采样被本机 perf
paranoid=4 拒绝，未调整系统设置。恢复的源码检查：913/913 C ABI、strict plugin boundary、
17 source-owner 契约、123 runner 测试、L1/L2/L5 197 项与公共属性错误参照负例复跑均通过；
属性输入 unchanged，收据 `shared-owner-acceptance/attribute-probe-resume-v3/`，不替代正式消费者或 release。
`llm-flags-{control,frozenopts}-ab-v2/report-cpu.json` 两轮三行、原冻结 baseline/3×2×10，
共 360 measured；两侧正确稳定、产物 unchanged。相同 current 链接闭包补编 llm/paddle_graph/
attribute 三 TU，仅改 CXX_FLAGS、不改宏/includes。control 五项性能 FAIL，冻结 flags 仍有
500M Q8/F16 decode p50 +8.65%/+6.11% 两项 FAIL；两份 attribution 审计及 46 项前后保护 PASS。
装配/原 baseline 保护三行为 RED→GREEN，不改生产 flags；不是完整 release 或因果关闭。
隔离 builder 的 shell `\$ORIGIN` 转义缺陷另经真实旁置 `.so` RED→GREEN，分析器/装配
10 项及 baseline 保护两项 PASS；`llm-origin-probe-v3/standalone-cpu.json` 在无
LD_LIBRARY_PATH 时真实 CPU 推理 PASS，仅为独立加载资格，原 v2 收据不覆盖。
原 OCR 两后端 168 场景均已结束、仍总 FAIL；Vulkan 执行身份 unchanged，七个失败场景
涉及 tiny Q8 rec/label/XFUND/orientation、tiny F16 label 及 public/heldout en Q8。
`passes_source_gt_gate=0` 表示 candidate-source 各 GT 指标 delta 的至少一项超过 0.005，
不是 source 自身质量门。tiny F16 label 的 hmean/e2e CER/e2e WER delta 为
0.005540/0.006178/0.009804；tiny Q8 label WER/e2e WER 为 0.009804，XFUND Q8
WER 为 0.018738，public/heldout en Q8 WER 为 0.006834/0.006208，均 FAIL；
CER 改善不抵消 WER 回退。历史 exact-file 四行的错误归因已在 `OCR_RELEASE_GATES.md` 纠正。
orientation 两失败是文本断言，不是 angle/score/rotation 断言；wrapper rc=0 不绕过
failures=2。tiny Q8 rec 四 fixture 完整、stored max_diff=1.317e-5，但 source benchmark
分支仍对原 max_diff/argmax 置 rc=1；developer 的 source audit=0 不单独触发该 veto。
未更改量化诊断/benchmark gate 的不一致或原 FAIL。原 CUDA 身份变化 veto 保留；
`owned-validation-continuum/` 正式重建/补丁重放、三项契约和三后端 attribute e2e/batch/UI、
table、pool 已全部 rc=0；pool CPU 40/40、CUDA/Vulkan 各 80/80、零失败。
正式库 SHA `7b5b28b7...`，冻结库/标准 manifest 保护通过；所列 complete 不代表 release。
`owned-yolo-formal-lifecycle-v1/` 的正式公共生命周期三后端 PASS；
`yolo-owner-formal-oracle-cuda-v1.log` 的正式 CUDA module owner oracle rc=0。
同存储权重 orientation exporter 真实 RED→GREEN，新增 native 方法前的 OCR owner 套件 75 PASS；
`orientation-stored-{cpu,cuda}-v1/` 与 native 三后端诊断均定位到倒置 word_1：
独立 source=LNIO，Q8 存储/native=LNIOF，原两次文本断言仍 FAIL。
.9 门下 native-stored 文本无附加失配、产物 unchanged；不能据此撤销独立文本门、
其他页面准确性门或原 release FAIL。threshold=1 不使用 .9 的文本真值。
正式 Vulkan `conv112-release-formal-vulkan-v3/report.json` 已结束、总 FAIL：candidate
`[0,0,1]` / 原冻结 baseline `[0,1,0]`，两侧每进程 2/2 fixture，主输入 audits 全合格。
baseline attempt 2 p95 6.43 对 source 3.277795 ms；candidate attempt 3 p50 5.76 对
source 2.430157 ms，绝对差分别 3.152205 / 3.329843 ms，均超原 5%/3 ms 双门。
不是 preflight timeout；成功进程和聚合值不抵消失败进程，产物/模型/标准 manifest 保护通过。
后续实际加载取证确认 v3/v4 仍使用当前 backend modules：`.so` 被当作测试导入 baseline，
runner 又 resolve probe symlink。原冻结 ggml 不含当前 probe 的 DW bias API，不能混装。
相邻 module 与 resolved 路径的真实加载 RED 保留；最终只复制各侧原 probe 字节与 owner modules，
不重建冻结库。`conv112-original-consumer-green-v3.log` 的真实 CPU 加载通过；v5
`conv112-release-formal-vulkan-v5-original-owner/report.json` candidate `[0,0,0]` / baseline
`[0,0,1]`，完整 2/2 fixture、主输入 audits 合格，仍总 FAIL：baseline 第三进程 p95
5.39 对 source 2.225544 ms，绝对差 3.164456 ms，原双门与 baseline veto 保留。
v3/v4 不再称为合格原 baseline A/B，v5 原 baseline probe SHA `0be29828...` 未变。
table 的正式 demo/OCR 消费者通过不覆盖日志明确未配置的可选 layout 页组合。
新正式 LLM CPU 三行 `llm-raw-formal-ab-v1/report-cpu.json` 两侧全部 rc=77，独立
upstream 工具不可用，无 measured 推理或新性能比较资格；原性能 FAIL 保留。

Source configurations without a matching local AICore contract are intentionally
not added to this table. Their exact families, including local VQA/VLM and
Grounding, are tracked in [TODO.md](TODO.md).
