# qAnyLabeling upstream parity audit

审计基线：ACloudViewer 当前工作树，qAnyLabeling 的上游仓库
`/home/ludahai/develop/code/github/dl/X-AnyLabeling-GGML`，提交
`745f6c355f94cddb06f6965b3a4118f82033e935`（v4.1.3）。本报告把功能
对齐、模型发布、精度/性能验收和安装包验收分开统计；局部测试通过不等于
完整上游 parity 通过。

状态含义：

- **已对齐（范围内）**：当前范围有代码、上游对照和可复现证据。
- **部分对齐**：主流程可用，但仍缺少上游范围、视觉状态、量化矩阵或平台证据。
- **待完成**：当前没有等价实现，或发布/验收条件尚未满足。
- **按上游缺失对齐**：上游本身没有注册该流程，因此保持同样的缺失，不虚构实现。

## “12 个 F32”到底是什么

新增的 12 个文件是源 ONNX 适配链的本地 F32 GGUF，不是当前公共 OCR
catalog 中的 12 个已发布权重。组成如下：

| 文件 | 上游角色 | 当前状态 |
| --- | --- | --- |
| `ppocr-cls-f32.gguf` | PP-OCR 0/180 度方向分类器 | 本地 F32；未发布 |
| `ppocrv4-source-det-f32.gguf` | 中文 PP-OCRv4 detector | 本地 F32；未发布 |
| `ppocrv4-source-rec-f32.gguf` | 中文 PP-OCRv4 recognizer | 本地 F32；未发布 |
| `ppocrv3-japan-rec-f32.gguf` | 日文 PP-OCRv3 recognizer | 本地 F32；未发布 |
| `ppocrv5-mobile-det-f32.gguf` | PP-OCRv5 mobile detector | 本地 F32；未发布 |
| `ppocrv5-mobile-rec-f32.gguf` | PP-OCRv5 mobile recognizer | 本地 F32；未发布 |
| `ppocrv6-tiny-det-f32.gguf` | PP-OCRv6 Tiny detector | 本地 F32；未发布 |
| `ppocrv6-tiny-rec-f32.gguf` | PP-OCRv6 Tiny recognizer | 本地 F32；未发布 |
| `ppocrv6-small-det-f32.gguf` | PP-OCRv6 Small detector | 本地 F32；未发布 |
| `ppocrv6-small-rec-f32.gguf` | PP-OCRv6 Small recognizer | 本地 F32；未发布 |
| `ppocrv6-medium-det-f32.gguf` | PP-OCRv6 Medium detector | 本地 F32；未发布 |
| `ppocrv6-medium-rec-f32.gguf` | PP-OCRv6 Medium recognizer | 本地 F32；未发布 |

这里“12 个 F32”描述最初的转换批次。当前已生成 **12 个 stem × F32/F16/Q8_0，
共 36 个新 GGUF**，另有两个公共 Q8 修复候选。38 个文件的统一持久目录为
build_app/Testing/ocr-storage-reference/ocr-release-candidates-20261019/；
其中 inventory.json 逐文件记录原路径、解析路径、大小、SHA 和真实存储比例，
SHA256SUMS 覆盖所有文件。最新发布阻塞以 [OCR_RELEASE_GATES.md](OCR_RELEASE_GATES.md)
为准；历史 v7/v17/v19 表保留实验过程，不代表完整发布验收。

这些文件的源 ONNX URL、字节数、SHA-256 和来源提交固定在
[`core/AICore/tools/ocr_onnx_sources.json`](../../../../core/AICore/tools/ocr_onnx_sources.json)。
方向分类、六组 source preset、选中多边形和 source-page 的 CPU/CUDA/Vulkan
formal consumer 检查均已通过；证据是
`build_app/Testing/qanylabeling-ocr-formal-{cpu,cuda,vulkan}` 和
`check_ocr_formal_consumers.py` 的 18 个 worker/window 返回码为 0。

当前转换入口已支持 F32/F16/Q8_0，并支持按源 SHA 固定的混合精度策略。
v7 的 24 个 F16/Q8_0 候选均已验证转换重放 SHA 相同；其 CPU/CUDA/Vulkan 多输入概率门
各为 22/24 通过，加上方向流程各为 33/36。后续 Tiny F16、v4 Q8 和 v5 F16/Q8
修复及对应新文件证据见增量表；旧 Tiny Q8 失败，后续 v19 的三后端图与街景/收据已通过。
策略保存在
`core/AICore/tools/ocr_quantization_policies.json`，仍为实验候选，未进入产品下载目录。
每个文件还必须完成真实页面、CPU/CUDA/Vulkan 一致性和受控性能门。

现有公共 OCR catalog 是另一组资产：`ppocr-det-ch`、`ppocr-rec-ch`、
`ppocr-rec-en`、`pplcnet-layout` 各有 `f32/f16/q8_0`，共 12 个已发布权重，
地址模式为：

```text
https://github.com/Asher-1/cloudViewer_downloads/releases/download/PP-OCR/<file>.gguf
```

该发布页当前仍是已有 OCR 权重的来源；新文件只有逐文件通过全部发布门，真实上传并取得
release-asset SHA-256 后，才能使用同一 `PP-OCR` tag、补充
`asset_digests.h`、catalog dump 和 validation manifest。当前不能把本地
`ocr-extension-release-candidate/` ZIP 写成可下载 URL，也不能把它们标记为
已发布。用户已授权将全部通过逐文件门控的新模型上传到上述 `PP-OCR` release；
当前没有满足所有条件的新文件，尚未上传。

## 全量对齐审计表

| 范围 | 上游契约/代码 | ACloudViewer 当前实现与证据 | 状态 | 剩余工作 |
| --- | --- | --- | --- | --- |
| 公共 OCR 权重 | `PP-OCR` release 的 4 角色 × F32/F16/Q8_0 | 已有 catalog、SHA、缓存及历史 OCR 矩阵；不等于本次逐文件全部发布门已复核 | 已发布；新门待复核 | 在当前树按新增验收要求复核已有权重，并与新增 source 权重区分 |
| 新增 12 个 source F32 | 上游 `ppocr_v4.py`、`ppocr_v5.py`、`ppocr_v6.py`、方向分类和六个 YAML | 源图三后端各 12/12；旧 CUDA 开发矩阵 42/42。新增源检测框过滤修复后页面矩阵重跑；均为 local-only | 部分对齐 | 按新生产库 SHA 重验页面和受控性能，再补 URL/SHA/catalog/冷缓存下载闭环 |
| 新增 source F16/Q8_0 | 上游使用 ONNX/Paddle 权重；不存在 GGML 量化约定 | 旧候选失败保留；v17 Small F16 和 v19 Tiny Q8 的新 SHA 已通过三后端源图及街景/收据，完整页面组合仍在补验 | 部分对齐 | 完成其余页面、XFUND 和全部受控性能门；混合精度不得标成纯量化 |
| OCR 方向分类 | `use_angle_cls`，source crop、BGR、严格 `score > 0.9` 旋转 | 公共 C ABI 追加可选 cls，批次/阈值/角度结果和 selected-region 路径已覆盖 | 已对齐（范围内） | 量化分类器和完整窗口视觉仍随模型发布工作关闭 |
| OCR 专用窗口 | 上游 `views/labeling/ppocr/*`、`ppocr_dialog.py`：布局、表格、公式、PDF、官方 PaddleOCR API、远程服务 | 新增独立 PaddleOCR 文档窗口（1560×920，最小 1200×760、侧栏 240）、官方 API 异步任务、JSON/JSONL、收藏/搜索/状态过滤、图像/PDF 上传、远程服务探测/模型发现、远程结果图片缓存、文档块联动与修订、JSON/Markdown/ZIP 导出和表格 token/HTML/TSV 专用编辑器；隔离 Qt 5 编译、服务 mock 和真实插件窗口测试通过 | 部分对齐 | 本地 PDF 渲染、专用公式/富文本编辑器、表格完整视觉/交互状态、其余窗口视觉仍未对齐 |
| PPOCR 组合预设 | 上游 v4/v5/v6/japan YAML 的 detector/recognizer/cls/字典/阈值组合 | AICore 有 8 个 task preset，新增六组 source preset 可用；新权重未进入公共下载 picker | 部分对齐 | 发布后补 catalog-backed composite rows 和完整默认配置回归 |
| KIE | 上游有 PPOCR 标注的 entity/link 导入导出；当前提交没有活动 SER/RE 自动模型注册 | 本地导入/导出、空 ID/方向 link/坐标规则已对齐；AICore 保留 KIE 字段并拒绝非空 KIE model option | 按上游缺失对齐 | 不应凭空添加自动 KIE；若上游注册 SER/RE，再新增模型、ABI、矩阵和 UI |
| OCR 量化精度 | 上游 ONNX/Paddle 输出是参考；GGML 必须有独立图门和真实数据门 | 同时测原始源输出及相同存储权重 ONNX 输出；新增 50 张完整收据；v7 部分组合存在街景 Hmean/完整流程 CER/WER 失败 | 部分对齐 | 完成全部真实页面、三后端和受控 A/B；图门通过不能覆盖页面失败；历史公共模型证据需按新门复核 |
| 点云 3D 数据模型 | 上游 cuboid/segmentation sidecar、KITTI/DATUMARO/Sly 导入导出 | C++ sidecar、检测/分割 class workspace、三类 ZIP 导入导出、事务预览/回滚和 87 个 gesture 对照已通过 | 已对齐（范围内） | 继续补窗口级视觉和跨平台事件验证 |
| 点云相机与投影 | `pointcloud/camera.py`、相机 panel、Z-up 视图、畸变/深度排序/预览 | 15/15 scene pixel 对照、相机标定/投影/裁剪、camera panel 和 live preview 已通过 | 部分对齐 | 完成全窗口截图、字体/DPI、handle icon、panel/dialog overlap 状态 |
| 点云正交视图 | `cuboid_viewport.py`：Top/Side/Front、handle 顺序、crop 深度、颜色/点大小、展开恢复 | 可见点索引借用、Side/Front `+0.1m` crop、Top/create bypass、语义/实例/RGB/intensity、device-pixel point size、expand/restore 已实现并测试 | 已对齐（范围内） | 仍缺完整窗口视觉和全部焦点/重叠/键盘状态 |
| 点云流式处理 | 当前上游异步加载整帧，未注册 chunk/stream 协议 | 本地同样按整帧加载；没有额外假冒的 streaming API | 按上游缺失对齐 | chunked streaming 是可选优化，不应作为当前 parity 缺口误报 |
| Chatbot | 上游流式多轮消息、停止、图片引用、历史、模型/提供商设置和对话 UI | Qt Network provider transport、SSE、取消/超时、消息编辑/删除/复制/重发/重生成、有界并发 folder AutoRun、ShareGPT 多模态导入/导出、8 个提供商默认配置及模型选择/收藏/搜索、自动模型发现/刷新决策已通过；数据、配置与消息变更均对照上游实际方法 | 部分对齐 | 完整 Markdown/code/math 渲染、精确对话/弹出列表布局和完整视觉状态；当前上游没有 Alpaca 导出入口 |
| VQA | 上游模板、组件编辑/排序、AI preview/apply、reference、folder JSONL、迁移 | 本地支持四类组件、手工值、AI preview/apply、reference、全目录选择/重命名字段 JSONL、组件配置导入/导出、排序及标题/选项/删除迁移、11 个默认模板及用户模板管理、自动配置目录、按 Python 语义的磁盘 reference 与专用提示/加载/预览窗口；与 Chat 共享模型刷新决策和配置 | 部分对齐 | 完整窗口视觉、Qt6/跨平台验收 |
| VLM-R1 OVD | 上游 JSONL 导入/导出和对话标注格式 | `AnnotationVlm` 已支持格式解析/导出和插件 action | 已对齐（范围内） | 这不是本地 VLM 推理；不能作为 VLM 模型完成的证据 |
| 本地 VLM/属性/剩余模型 | 上游配置包含 `florence2_large_ft`、`grounding_dino_api`、四组 Grounding-DINO/SAM 组合、`pulc_person_attribute`、`pulc_vehicle_attribute`、`internimage_l_22kto1k_384` | 当前 AICore/qAnyLabeling 没有对应本地图和完整 catalog consumer；远程分类只依赖用户服务 | 待完成 | 为每个模型补 AICore task、catalog/digest、默认配置、精度/性能矩阵和 UI |
| DEIMv2/DINOv3 检测 | 上游 `deimv2_dinov3_s_coco.yaml`、`deimv2_dinov3_m_coco.yaml` 及 `deimv2_hgnetv2_n_coco.yaml` | AICore 有 RF-DETR/YOLO 的检测语义，但没有这些 DEIMv2 官方 checkpoint 的同名 GGUF/catalog consumer | 待完成 | 按精确 checkpoint 转换、发布、digest、manifest、CPU/CUDA/Vulkan 精度与性能门；语义等价检测不能关闭此项 |
| 其他精确模型族 | 上游还注册 `damo_yolo`、`gold_yolo`、`yolo_nas`、YOLOv6/7/9/10/12、YOLOX、RT-DETR/RT-DETRv2、`u_rtdetr`/DFINE、RIO-DETR、RTMDet pose、UPN、CLRNet、`yolox_dwpose`，以及 YOLOv5/YOLO11 的 cls/det-track/OBB/pose/seg/SAHI/SAM/RAM 组合 | 当前 AICore 只覆盖部分语义和不同 checkpoint；未注册的官方权重、lane polyline、whole-body pose、跟踪/SAHI 组合没有同名 consumer | 待完成 | 逐族建立“精确权重 vs 语义替代”矩阵；新增类型时补 public ABI、catalog/digest、manifest、真实数据和跨后端门，禁止静默替代 |
| 默认配置/主题/资源 | 上游快捷键、palette、panel style、dark/light theme、图标和 RAM presentation lookup | 配置 comparator、8 组 panel style、主题像素、快捷键、Qt RCC、翻译和 RAM relocation 测试已通过 | 部分对齐 | 全窗口截图仍有布局/字体/DPI 差异；Qt6/平台包未验证 |
| RAM 资源 | 上游 RAM GGUF 内含英文 tag list；中文展示映射仍是资源文件 | `resources/ram/*` 仅用于英文到中文展示映射，已通过 RCC 嵌入 plugin；运行时不查 source path | 已对齐（范围内） | 若 GGUF 将来提供中文映射，应改用公共 API 后再移除 fallback |
| Linux 安装/搬迁 | 插件、RCC、RPATH、Unicode 安装目录可运行 | `qanylabeling-resource-install.json`：实际 install block、RPATH removal、Unicode prefix relocation PASS | 部分对齐 | 完整 Linux package inspection、干净机器依赖和 release profile 尚未完成 |
| Windows/macOS 安装包 | 上游提供 Windows Inno Setup、Linux installer、macOS DMG/更新流程 | 没有当前分支的原生 Windows/macOS qAnyLabeling package/build/relocation 报告 | 待完成 | 三平台原生 configure/build/install/relocation/启动和资源验证 |
| Qt5/Qt6 视觉 | 上游 PyQt6；ACloudViewer 支持 Qt5/Qt6 | 当前验证为 Qt5；本机没有可用 Qt6，不能声称 Qt6 视觉等价 | 待完成 | 在 Qt6 环境重跑窗口/输入法/DPI/资源/截图回归 |

## 2026-10-19：逐模型验收快照

下表保留 v7 候选快照；后续不同 SHA 的修复见下方增量表。F32 是 CPU/CUDA/Vulkan 多输入源图门，F16/Q8_0 的数字是 CPU 源图误差，
其 PASS/FAIL 已在 CUDA/Vulkan 复核；**不是完整发布资格表**。检测使用两张真实页面，
识别使用四张中英文词图，分类使用五张词图及对应 180 度图，共十个输入。
识别/分类还要求每个时间步 argmax 完全相同；检测同时检查 mean 误差。
所有源图都额外检查相同存储权重的独立 ONNX，避免把量化损失误判为运行时误差。

| 模型 stem | F32 CPU/CUDA/Vulkan 源图 | F16 CPU 最大概率误差 | Q8_0 CPU 最大概率误差 | 发布 |
| --- | --- | --- | --- | --- |
| `ppocr-cls` | PASS/PASS/PASS | PASS `0.000174344` | PASS `0.000956574` | 未满足全部门 |
| `ppocrv4-source-det` | PASS/PASS/PASS | PASS `0.020687461` | PASS `0.106714340` | 未满足全部门 |
| `ppocrv4-source-rec` | PASS/PASS/PASS | PASS `0.000802502` | PASS `0.002008081` | 未满足全部门 |
| `ppocrv3-japan-rec` | PASS/PASS/PASS | PASS `0.000351965` | PASS `0.002486408` | 未满足全部门 |
| `ppocrv5-mobile-det` | PASS/PASS/PASS | PASS `0.020731091` | PASS `0.035158217` | 未满足全部门 |
| `ppocrv5-mobile-rec` | PASS/PASS/PASS | PASS `0.000558376` | PASS `0.003126979` | 未满足全部门 |
| `ppocrv6-tiny-det` | PASS/PASS/PASS | PASS `0.013615757` | PASS `0.075072199` | 未满足全部门 |
| `ppocrv6-tiny-rec` | PASS/PASS/PASS | **FAIL `0.006315589`** | **FAIL `0.890249828`，argmax `0.975`** | 阻断 |
| `ppocrv6-small-det` | PASS/PASS/PASS | PASS `0.014678866` | PASS `0.064754665` | 未满足全部门 |
| `ppocrv6-small-rec` | PASS/PASS/PASS | PASS `0.000339448` | PASS `0.002089202` | 未满足全部门 |
| `ppocrv6-medium-det` | PASS/PASS/PASS | PASS `0.003221691` | PASS `0.019202054` | 未满足全部门 |
| `ppocrv6-medium-rec` | PASS/PASS/PASS | PASS `0.000140548` | PASS `0.002930522` | 未满足全部门 |

检测 F16 门为 max `<0.1` 且 mean `<0.0001`，Q8_0 为 max `<0.6` 且
mean `<0.0005`；识别/分类分别为 max `<0.001` / `<0.005`。
相同存储权重参考与直接 CPU 后端比较均要求 max `<0.001`，识别/分类 argmax 为 `1`。
这些是清单明示的存储精度预算，不能只比较表中的 max 数字推断角色间的精度。

可复核报告位于 `build_app/Testing/ocr-storage-reference/`：

- `f32-cpu-current-graphs/matrix.json`：当前 CPU 源图 **12/12 PASS**。
- `f32-vulkan-current-graphs/matrix.json`：当前 Vulkan 源图 **12/12 PASS**。
- `f32-cuda-current/matrix.json`：源检测框过滤修复前 CUDA F32 **42/42 PASS**，含 12 个源图和 12 个真实标注数据场景；页面行不能直接算作新生产库发布证据。
- `quant-v7-{cpu,cuda,vulkan}/matrix.json`：各 **33 PASS / 3 FAIL**，包含 24 个量化源图与 12 个方向流程；失败为 Tiny F16/Q8 源图和 Tiny Q8 方向。
- `quant-policy-replay-v7.json`：24 个转换产物的 SHA **全部重放相同**。
- `full-native-regressions.log`：受影响原生回归 **35/35 PASS，零跳过**。
- `source-detector-filter-native-tests.log`：补充源检测框过滤/裁剪修复后，OCR/table/plugin/validation **12/12 PASS，零跳过**。
- `registered-gate-sroie-tests.log`：四组门控 CTest 全部通过（runner 96 项、GT 指标 7 项、概率 7 项、GPU 审计 7 项）。

标注场景是官方 50 张验证图的 201 个非忽略词框；分别报告原始检测 Hmean、固定 GT
区域 CER/WER 和完整流程 CER/WER，与未修改源 TextSystem 的五项退化均要求 `<=0.005`。
词级 GT 与模型行级检测的粒度差异限制绝对 Hmean 的解释，不能冒充官方榜单结果。
XFUND 中/日文各 50 页是语义字段，只检查固定区域识别，不提供检测或自动 KIE 证据。
SROIE contributor corpus 的 50 张完整扫描收据、2,779 个原始标注另有 18 个
组合/量化场景，固定 100 个图片/CSV 输入 SHA 和标注目录摘要；数据来源、选择方式和
IoU 协议见 [`DATA_CARD.md`](../../../../core/AICore/tests/ocr/DATA_CARD.md)。
这是 contributor 子集，不能声称官方 held-out/DetEval 比赛分数。
数据归档、每个消费文件和字典均固定 SHA，公共 runner 验证下载/安全解包；新模型下载闭环
仍须等待真实 release 和 catalog 注册，输入数据下载通过不等于模型下载已完成。

整页核对发现源 `TextDetector.filter_tag_det_res` 还会按首个极值重排顶点、裁至
`width-1/height-1` 并剔除短边，本地 source ONNX 检测路径此前漏了这一步。
真实菱形框的极值相同时，上游会重复角点，OpenCV 的奇异透视变换采样左上角像素；
本地现已匹配该行为。13 组源生成的过滤 fixture、真实奇异裁剪及既有几何 golden 均通过。
此修复不改变原始模型图；页面/工作流必须按新库 SHA 重新验收。未完成的旧 CPU、收据和
v9 页面 cohort 已停止并保留中断记录；`filter-v7-cuda` 重新执行 36 个街景/收据场景，
逐页 `.progress` 文件不会混入 runner 的最终 JSON 指标。现有 v8/v9 候选结果也须区分库版本。

以下增量均使用检测框过滤修复后的生产库
`154a0b62460edbe0273deeb17f766bd0d949aefc63a4782b0d9c3251665105cf`；
页面探针为 `30c2bca6387f9f13b22e0ababf1716c4886c8ad4454cd0870211380fc3f81bee`。
均为开发精度证据，尚未获得发布资格。

| 候选文件 SHA 前缀 | 源图 CPU/CUDA/Vulkan | 页面结果 | 限制/证据 |
| --- | --- | --- | --- |
| v8 v4-det Q8 `a62a8e2b86cb` | PASS，CPU max `0.102772355`，mean `0.000298230` | 中/日文街景 CPU/Vulkan 通过；CUDA 街景及收据四项通过 | `filter-v8-{cpu,cuda,vulkan}`，CUDA 5/5；`quant-policy-replay-v8.json` 重放相同 |
| v12 v5-rec F16 `a5ce8f19f598` | 五输入 PASS，CPU max `0.000957310`，argmax `1` | CUDA 街景、收据通过 | `filter-v12-*`；实际 21.23% F16、78.77% F32 |
| v14 v5-rec Q8 `72d723b5c8e2` | 五输入 PASS，CPU max `0.004976571`，argmax `1` | CUDA 街景、收据通过 | `filter-v14-*`；实际 2.10% Q8_0、19.13% F16、78.77% F32，接近概率门限；v12/v14 重放 SHA 相同 |
| v15 Tiny-rec F16 `b5a7a723e105` | 四输入 PASS，CPU max `0.000872791`，argmax `1` | CUDA 方向、收据通过；搭配原 F16 detector 的街景失败 | `filter-v15-*`；检测 Hmean 下降 `0.005540166`，完整流程 WER 增加 `0.009803922`，不能作为组合 PASS |
| v16 Tiny-det F16 `804a564ad2db` | 两输入三后端 PASS | 搭配 v15 recognizer 的 CUDA 街景/收据通过 | `filter-v16-*`；敏感核 `conv2d_112.w_0` 保留 F32，v15/v16 转换重放 SHA 相同；其他后端页面及 XFUND 未完成 |

v5 recognizer 的三个量化场景已加入真实 `val_img_7.jpg` 的源透视裁剪，输入数量
从四增至五，数据来源/选择范围见 DATA_CARD。旧 v7 Q8 在该输入 max 为
`0.042550772`、argmax 为 `0.975`，CPU/CUDA/Vulkan 均失败
（`v5-hard-crop-f32-q8-*`）；不能继续引用旧四输入 PASS 关闭新门。
原 F32 的新五输入三后端通过。当前策略保留 v8/v12/v14 和 Tiny F16 的 v15/v16，
全部 36 个当前本地产物的 SHA/实际存储组成保存在 `ocr-candidate-inventory-v16.json`。
Tiny Q8 仍失败，窄范围 PASS 未自动注册为公共资产。

检测概率报告还须区分输入尺寸：上述历史图级矩阵为 max-side 320，页面流程为源默认
960。当前 15 个 detector 源图清单已显式切换至 960，`default-size-accuracy-*`
正在按新输入重验，旧图 PASS 不能覆盖新门。release 探针强制 960，拒绝小图性能
替代；额外检测输入也保持 960。新增回归防止 320/640 测量冒充默认配置。

960 CPU 首轮 `default-size-accuracy-cpu` 为 **14 PASS / 1 FAIL**：Small detector
F16 mean `0.000101013` 超过 `<0.0001`，相同权重 native/ONNX max 仅
`0.000008106`，属于存储误差。v17 保留 `conv2d_122.w_0` 为 F32，新 SHA
`73efc76d8b6427363b19c5a88ecf7075febfd5bbbef0b5b7ff67ac4d8f5330c1`
在 CPU 960 源图 max/mean 为 `0.010192215` / `0.000082143`，街景五项源退化均为 0；
CPU 收据亦通过，五项最大退化为 `0.000733810`；`filter-v17-cpu` 共 **3/3 PASS**。
转换重放相同。该文件不能继承旧 Small F16 的页面状态。
当前默认尺寸 CPU 的完整 36 项概率矩阵为 **35 PASS / 1 FAIL**，唯一失败是旧
Tiny Q8 recognizer；不能把该失败从矩阵中删除。
CUDA/Vulkan 的同一 36 项矩阵亦各为 **35 PASS / 1 FAIL**，失败项相同。
新 Small F16 的 CUDA/Vulkan 街景/收据各为 **2/2 PASS**
（`filter-v17-pages-{cuda,vulkan}`）。
已有公共 catalog 的当前 CPU/CUDA/Vulkan 图/布局回归各为 **15/15 PASS**
（`public-current-{cpu,cuda,vulkan}`）；
它保留历史模型各自的门限，不能替代本次更严格的逐文件发布验收。
已有 12 个发布文件的空缓存下载、逐文件 SHA 及离线预检通过，记录为
`public-cold-download-0zhcaspv/report.json`；不包含任何新权重。
重新取得原始 Paddle v4 检测/中文识别/英文识别参数，均与公共 GGUF 的源参数
SHA 一致；重新生成三份概率真值，最大差分别为 `0.000016630`、
`0.000000119`、`0.000000358`，两项识别 argmax 完全一致。

上述新候选的 CUDA 页面原始分数如下；CER/WER 越低越好。街景为词级 GT 与行级
检测的混合粒度，完整流程 CER/WER 可超过 1，不能解释为官方榜单准确率。
“最大退化”取检测 Hmean、区域 CER/WER、完整流程 CER/WER 的五项源对照退化最大值，
每项仍单独检查 `<=0.005`，不会用平均值抵消失败。

| 配方/量化 | 50 页数据 | 检测 Hmean | 固定区域 CER/WER | 完整流程 CER/WER | 最大退化 |
| --- | --- | --- | --- | --- | --- |
| v4-source Q8 v8 | 街景 | 0.402332 | 0.157105 / 0.348039 | 1.240071 / 1.313725 | 0.004902 |
| v4-source Q8 v8 | 收据 | 0.888528 | 0.270009 / 0.619240 | 0.363424 / 0.676187 | 0.000526 |
| v3-japan Q8 v8 | 街景 | 0.402332 | 0.659312 / 0.901961 | 1.235658 / 1.357843 | 0.004644 |
| v3-japan Q8 v8 | 收据 | 0.888528 | 0.579618 / 0.923953 | 0.587812 / 0.945681 | 0.000702 |
| v5-mobile F16 v12 | 街景 | 0.377465 | 0.184466 / 0.406863 | 1.246249 / 1.362745 | 0.004902 |
| v5-mobile F16 v12 | 收据 | 0.837987 | 0.274992 / 0.631155 | 0.403291 / 0.714211 | 0.000000 |
| v5-mobile Q8 v14 | 街景 | 0.376404 | 0.183583 / 0.406863 | 1.245366 / 1.357843 | 0.001060 |
| v5-mobile Q8 v14 | 收据 | 0.837038 | 0.275026 / 0.630979 | 0.405197 / 0.715437 | 0.001605 |
| v6-tiny F16 v15/v16 | 街景 | 0.349030 | 0.176523 / 0.397059 | 1.338041 / 1.470588 | 0.004902 |
| v6-tiny F16 v15/v16 | 收据 | 0.856725 | 0.245393 / 0.489224 | 0.374628 / 0.609077 | 0.000350 |

Tiny Q8 的四输入源诊断另试了比例缩放、按块最小二乘和按块权重 MSE，仍没有
通过预算的单核替换；线性层转置存储的独立诊断也失败。这些只用于定位存储敏感性，
这些诊断没有修改正式量化算法或放宽门限。

后续 v19 采用显式的转换期激活误差反馈：从固定 PaddleX 训练集选 256 个原始
标注裁剪，固定图片 SHA/quad，在原始 ONNX 中观察激活，输出普通 Q8_0 块。
数据与选择范围见 `core/AICore/tests/ocr/DATA_CARD.md`。新 Tiny detector SHA
`fa5f6f6efdc79c4a2363c7719d6dd5d7ca999e14cbc71132a65d30ce834c4988`、
recognizer SHA `f9e6c3793c95fa7e84b10d5fa277f56e22b059c4e1517ba93a223f83fa41d960`
在 `filter-v19-cpu` 的检测概率、识别概率、50 张街景和 50 张收据 **4/4 PASS**。
街景五项退化均不大于零，收据最大退化 `0.000208960`；转换重放均字节相同。
detector 的 Q8/F16/F32 参数占比为 0.23903% / 98.50118% / 1.25979%，
recognizer 为 0.57945% / 14.95273% / 84.46782%，不得宣称纯 Q8 或量化加速。
CUDA/Vulkan 的同四项检查也各 **4/4 PASS**
（`filter-v19-pages-{cuda,vulkan}`），未放宽门限。转换策略已同步到仓库的显式
实验配置，SHA 为 `3f8700eb6ab5672f396ff3c93b73c649ca480de4ec59cb0d2c14b3369d22ccc4`；
模型文件仍在 build 目录，未改变产品默认配置、公共 catalog 或共享模型缓存。
全部页面组合、受控性能和发布闭环仍未完成。

受控 GPU 门同时记录 native 与原始源 CUDA provider 的图延迟、p50/p95、驱动进程显存
采样和外部进程活动；显存是采样高水位，不能声称精确分配器峰值。分类器受控矩阵的
三次 baseline 和三次 candidate 测量全部被桌面/GPU 活动拒绝，未计作性能通过。
原生 baseline 已保留，测量 harness 对齐并记录二进制 SHA。原始源 CUDA I/O-binding
smoke 已运行，CPU/CUDA max 差为 `0.000016362`；共享 GPU 上的 smoke 耗时不是受控基线。
新生产库的 `controlled-cls-f32-current-cuda/matrix.json` 同样失败：部分 native
采样通过，但源端阶段检测到桌面或另一 RelateAnything 基准进程。测量后自有页面
队列已恢复。GPU 审计现在为每次尝试保留唯一 ID 和独立原始 JSON，包含空闲等待失败；
四组 CTest 通过（runner 96、GT 7、概率 11、GPU 审计 8），重复运行不再覆盖旧遥测。
公共 runner 后续新增执行身份记录：推理前固定所选 executable/Python、构建目录
AICore/ggml 库及 CMake 配置 SHA，推理后检测新增、删除或改写；manifest 记录实际
解析的字节 SHA，并检测运行中变更。该记录不是完整动态链接器加载轨迹。
新增测试后 runner **98/98 PASS**，四组 fast CTest 通过；实际 CPU 分类器检查
`artifact-provenance-cpu` 为 PASS 且 `execution_artifacts.unchanged=true`。
此前启动的旧进程不会自动得到这份新记录，不能追认旧报告具备该身份门。
本次未上传任何新文件，专用 OCR/VQA/Chat 窗口、其余本地模型和 Linux clean-package
仍未完成；按照用户要求，在模型门之后继续。

## 2026-10-19：公共权重的原始 Paddle 复核

公共 detector/ch-rec/en-rec 的原始 program、parameters 及字典 YAML 已分别固定
SHA，配置归档只消费 YAML，不能把归档中的重导出权重替换成发布文件的源权重。
`ocr_paddle_sources.json` 和 manifest 注册这些输入，空缓存下载及离线复核通过。
新的 `run_ocr_paddle_probe.py` 在独立进程中运行原始 Paddle 3.0.0，并要求该进程
正常退出，随后才比较固定 fixture、原生概率和 CPU 后端。

`public-paddle-graphs-v4-{cpu,cuda,vulkan}/matrix.json` 各为 **9/9 PASS**，
共 **27/27 PASS**；识别器逐时间步 argmax 均为 1，GPU 同时逐元素比较原生 CPU。
以下为当前 CPU 图的最大绝对概率误差，仍沿用各角色既定预算。

| 发布模型 | F32 | F16 | Q8_0 | 本次验收范围 |
| --- | --- | --- | --- | --- |
| `ppocr-det-ch` | 0.000022560 | 0.061641514 | 0.490858376 | 原始 Paddle CPU；检测还检查 mean 误差 |
| `ppocr-rec-ch` | 0.000006616 | 0.000072360 | 0.000119507 | 原始 Paddle CPU；argmax 全同 |
| `ppocr-rec-en` | 0.000000358 | 0.000573993 | 0.003898740 | 原始 Paddle CPU；argmax 全同 |
| `pplcnet-layout` | 三后端原始源图通过 | argmax 门失败 | argmax 门失败 | 新增 8 个输出头的严格检查，见下表 |

公共 Paddle 页级参考使用 PaddleOCR/PaddleX 3.0.1 和 Paddle 3.0.0 的固定配置。
固定 GT quad 通过公开的 Paddle quad 识别 API 传入；不混用 X-AnyLabeling 的
选区最小外接矩形规则。六个难例裁剪比较 **6/6 PASS**；中文 F32 的 50 张街景
完整页面检查通过，Hmean、固定区域 CER/WER 和完整流程 CER/WER 五项差值均为 0。
对应产物为 `public-paddle-crop-contract-54p0tts4/result.json` 和
`public-paddle-pages-v2-cpu-0/probes/ocr/ppocr-public-ch-labelled-scenes-f32/`；
旧 v2 页面队列随后出现 CPU 源进程退出堆损坏，已中止并保留
`public-paddle-pages-v2-aborted.json`。完整三后端队列按新隔离配置重跑，不能据旧
部分结果记为全部页面通过。

公共 GPU 源参考的退出异常已定位到宿主继承的 `GLOG_log_dir=/sandbox/logfiles/`：
官方 Paddle GPU 3.0.0 的退出析构重复释放该静态字符串。Valgrind 与 GDB 诊断
分别保存于 `paddle-import-memcheck-tiy9rwq9/` 和
`paddle-gpu-exit-gdb-w6q4uuuu/`；移除子进程继承的 GLOG/MLOG 配置后，同一版本
可正常推理、清理和退出，不改动宿主环境，也不跳过析构或忽略退出码。

数学模式复核另发现 Paddle 3.0.0 不识别此前的两个 `FLAGS_allow_tf32_*` 名称。
改用 NVIDIA 文档的 `NVIDIA_TF32_OVERRIDE=0`，在初始化 GPU 库之前禁用 TF32。
`public-paddle-gpu-no-tf32-_ux2mkyf/result.json` 中原始检测、中文识别、英文识别
GPU 对 CPU fixture 的最大差分别为 `0.000017583`、`0.000006557`、`0.000000238`，
识别 argmax 均为 1，三进程正常退出。该共享 GPU smoke 不是受控性能结果。
GPU 参考继续固定 Paddle 3.0.0，无需升级版本。探针同时记录源 benchmark SHA、
源 libpaddle SHA、依赖版本与实际数学模式，检测 benchmark 在执行期间被修改。

GPU 图参考环境增加固定 wheel SHA 和全部依赖版本清单
`requirements-ocr-paddle-gpu-reference.txt`。新建环境的 `pip check` 及三个图的正常
退出/数值复核通过，与此前 GPU 环境输出逐元素相同；记录在
`ocr-paddle-gpu-minimal-4s9ttxya/validation.json`。

页面与图参考共享 `ocr_paddle_environment.py`，统一隔离宿主日志和动态库配置。
页面参考只在源子进程正常退出后由父进程写入成功凭据。显式固定 SHA 的参考可以
复用，但必须重新检查源库/依赖/代码、语言、线程、评价方式、配置、GT、完整页面
覆盖与逐图 SHA；各量化/后端仍独立执行原生推理。新 v3 中英街景的源参考各 50 页
已正常退出，其余页面仍在运行。不能复用崩溃前写出的预测。

公共源参考测试现为 **27/27 PASS**。这不关闭受控性能、发布资格或高级窗口任务。

依赖身份记录还修复了一个隐藏问题：Paddle 引入 setuptools 的 vendored 元数据，
遍历字典会用 vendored `typing_extensions 4.12.2` 覆盖实际选择的 `4.16.0`。
图参考现通过 `ocr_paddle_metadata.py` 记录规范化的实际选择版本与全部候选版本，
并固定该 helper 的 SHA。现行页面 v3 队列仍使用冻结的旧版本枚举；它的数值结果
可以保留，但不能追认已获得更强的依赖身份验收。

### 公共 Q8_0 在完整街景流程的新失败

当前 v3 的中英 F32/F16 各 50 页通过；两组 Q8_0 均失败：

| 组合 | 检测 Hmean 退化 | 固定区域 CER 退化 | 完整流程 CER 退化 | 门限/结果 |
| --- | --- | --- | --- | --- |
| 中文 Q8_0 | 0.002077406 | 0.000882613 | 0.009708738 | <=0.005，FAIL |
| 英文 Q8_0 | 0.002077406 | -0.003530450 | 0.005295675 | <=0.005，FAIL |

证据是 `public-paddle-pages-v3-cpu-ppocr-public-{ch,en}-labelled-scenes/matrix.json`，
实际文件 SHA 与现有 release 相符。图门通过不能掩盖完整流程 CER 失败。
GT 的词级/行级粒度限制仍然适用；这里是相同 GT 下的源差值，而非官方比赛分数。

`public-q8-stage-attribution-194al707/completed.json` 只替换一个阶段，在 CUDA
重跑相同全部 50 页：Q8 detector + F32 recognizer 的中英 e2e CER 退化都为
`0.011473963`；F32 detector + Q8 recognizer 两组均通过。因此先修复检测器的
存储精度，不能通过替换识别器或调整 0.005 门限关闭失败。

`controlled-classifiers-current-v2-cuda` 再次排空并暂停自有页面任务后，baseline
空闲门仍失败：265 个等待采样中，桌面 `gnome-shell` 持续约 27% SM。
保留原始审计及 `interrupted.json` 后中断该 cohort，finally 已恢复自有 CPU/GPU
页面队列。该记录没有完整 matrix，也没有 CUDA/Vulkan 性能 PASS；不能将中断
记录补写成完整三次 A/B。外部桌面进程和既定空闲阈值均未修改。

### 多输出 layout 的新验收缺口

`run_ocr_paddle_layout_probe.py` 比较原始 PicoDet 的 4 个分类头及 4 个回归头，
按唯一形状配对，导出同次原生 forward 的全部输出，并直接比较原生 CPU。
源 program/parameters/config 进入独立 SHA 清单及自动下载输入；空缓存下载
12 个已有公共模型、10 个 layout 输入并完成后续离线复核。

`public-layout-original-zfzv7xec/{cpu,cuda,vulkan}-v2/matrix.json` 的实际结果是
**F32 三后端通过，F16/Q8_0 三后端各失败**，不能沿用旧概率预算下的 PASS。
下表为 CPU 对原始 Paddle CPU 的数值；argmax 列表示最差输出头的匹配比例。

| layout 文件 | 最大误差 | 平均误差 | 最差头 argmax | 新门控 |
| --- | --- | --- | --- | --- |
| F32 | 0.000015497 | 0.000000417 | 1 | PASS |
| F16 | 0.030360460 | 0.000918314 | 0.992307692 | FAIL |
| Q8_0 | 0.499403417 | 0.010272793 | 0.969230769 | FAIL |

三个 dtype 的 CUDA/Vulkan 对原生 CPU 的最大误差均小于 `0.000021`、argmax 全同。
因此本次不一致来自量化文件对源输出的偏差，不能用后端互相一致代替源真值。
原始 Paddle GPU 的 8 头正常退出 smoke 对 CPU 最大误差 `0.000020504`、argmax 全同，
记录为 `public-layout-source-gpu-mn0d0bza/result.json`，不作受控性能证据。

原生探针新增逐头导出和非有限值拒绝；真实图注入 NaN 的回归退出 1，见
`layout-nonfinite-mzdt15vc/result.json`。本次仅隔离重链接测试探针，现有运行队列的
二进制和模型缓存未替换；构建命令、源 SHA、探针 SHA 与后端库 SHA 均单独归档。

`layout-rounding-attribution-wd1mprzf/report.json` 进一步从与 F32 GGUF 逐位相同的
源参数生成八组隔离候选：仅归一化参数 F16（其余 F32）已产生 argmax 翻转；
归一化参数保留 F32 仍只有 `0.997894737` 最差头匹配。单独保留深度卷积或输出头
也未消除全部差异。这是单 fixture 的舍入归因，不是发布修复或真实页面验收。

## 当前可复核结论

### 本轮逐文件门控补充（未达到发布条件）

| 范围 | 本轮结果 | 尚缺的证据 |
| --- | --- | --- |
| 公共 det/ch-rec/en-rec × F32/F16/Q8_0 | `public-paddle-graphs-v5-lf69oukv`：CPU/CUDA/Vulkan 源图各 9/9 PASS；记录实际选中的依赖版本和重复 metadata 候选 | 全部页面集、release 重复和受控同机 A/B |
| 公共 PicoDet layout × 三种存储 | 新增四边各自 8-bin 的 DFL argmax 与 softmax 概率误差，保留原有通道 argmax；F32 通过，F16/Q8_0 仍失败 | 精度修复、布局 GT、受控性能 |
| 新增 36 个文件的当前构建身份 | `current-v19-graphs-6qbolt_5/cpu/matrix.json`：36/36 PASS；首轮 CUDA 16 PASS，加上 retry 的 20 PASS，覆盖全部 36 个文件；retry Vulkan 36/36 PASS | 首轮 INCOMPLETE 和显存保护未执行项原样保留；这些单轮 developer 行不代替 release |
| SLANet × 三种存储 | `current-table-graphs-aenvh0zm`：CPU/CUDA/Vulkan 各 3/3 PASS | TEDS/TEDS-S 和受控 release 性能须单独完成 |
| 公共 Q8_0 检测器逐层诊断 | `public-det-single-layer-pages-cmchzcrs/completed.json`：14 层 × 中英文共 28 项，4 个单层保留 F16 的候选同时通过 50 页街景回归 | 属于用现有回归集选择候选；尚无发布修复，不能作为未参与选择的测试集成绩 |
| 公共 Q8_0 收据流程 | `public-det-layer-sroie-07gsu_x2`：两种候选的中文收据通过；英文固定区域 WER 退化 `0.006833713`、完整流程 WER 退化 `0.008936394`，超过 `0.005` | 英文识别器仍须修复；街景的检测器归因不能外推为所有页面只有检测器问题 |
| 英文收据 F32/F16 对照 | `public-sroie-en-cuda-controls-aernoeca/matrix.json`：2/2 PASS；F32 五项退化均为 0，F16 固定/完整流程 WER 退化为 `0.000175223` / `0.001051340` | 页面参考源 v3 的 metadata 身份缺口仍保留，且不是 release 性能结果 |
| 英文 Q8_0 识别器逐层诊断 | `public-rec-layer-sroie-_7968rya/completed.json`：八个候选中只有额外保留 `conv2d_161.w_0` 为 F16 的候选通过；固定/完整流程 WER 退化为 `-0.005431926` / `-0.000350447` | 额外保留检测器 `conv2d_397.w_0`、英文识别器 `conv2d_161.w_0` 的组合已通过英文 CPU 街景；旧身份 CUDA/Vulkan 的六组页面也通过。候选选择用了原回归集，独立收据集、新身份完整页面与受控性能仍待验 |
| 新增文件的 GPU 重跑 | 首轮 CUDA 为 16 PASS / 20 显存保护未执行，Vulkan 为 36 项未执行；均保留 INCOMPLETE。`current-v19-graphs-retry-ovrm7450` 只重跑未执行项：CUDA 20/20、Vulkan 36/36 PASS | 不将首轮改写为 PASS；显存保护和外部 GPU 进程均未绕过 |
| 大模型 CPU 页面 | 当前队列部分 PP-OCRv6 Medium XFUND 项触及默认 `3600 s` 超时（退出 124） | 失败保留；单页 8/4/2/1 线程诊断耗时约 57/20/19/25 秒，44 个框的文本、坐标、分数完全一致。本会话三个 CPU 队列已记录中断，待显式线程和更长超时重跑；共享负载耗时不是受控速度结论 |
| 测试与发布 | 参考源隔离、身份、输出头和失败门控单元测试 **31/31 PASS**；公共 runner **98/98 PASS**；本轮新上传 **0** | 无文件已通过用户指定的完整发布门控 |

公共页面参考也已修复 metadata 身份：分别记录实际选择的版本、全部重复候选和
metadata helper SHA，并拒绝回放旧身份文件。进程初始化后与结束前的参考代码、模型、
源环境身份均须一致，正常退出码必须为整数 0。`public-pages-v4-18iwva6p` 的中英文
50 页街景 F32 seed 已通过；其余源导出及三后端回放仍在运行，旧 v3 不追溯升级。

增加未用于量化层选择的 SROIE 收据 `050–099`：先于新推理固定 100 个图像/GT 文件
的 Git revision、字节数和 SHA-256，再声明中文/英文 × 三种存储的六个强制消费者。
与 `000–049` 回归集文件名不重叠，整套 GT 摘要为
`8f085fedd59d4421352b4e1b37839f55515b74f75a89eae183fd810b03ecdc47`。
新队列由公共 runner 从空缓存预检并下载；这不是官方竞赛 test split，尚无完成结果，
也不声称已证明模型训练集无重叠。

逐文件索引覆盖 48 个文本/方向分类/布局文件及 3 个 SLANet 文件，共 **51** 个。
文件 SHA、实际存储比例及逐后端页面缺口见 [OCR_RELEASE_GATES.md](OCR_RELEASE_GATES.md)；
另列两个尚未发布的修复版本，不与公共文件的证据合并。
索引单独记录旧报告中模型 SHA 相同但构建身份不完整的结果；它们不升级为当前构建
或 release 通过。公共页面 v3 仍使用旧 metadata 记录方式，依赖身份缺口不能追溯
改写为已通过新检查。所有原始报告位于本机 `build_app/Testing/ocr-storage-reference/`。

- 新增专用 OCR 窗口测试已覆盖真实 Qt Network 的 multipart 提交、嵌套 job ID、结果 JSONL、API 错误、取消、超时和重启；另覆盖重复 block ID 编辑、未知字段保留、原子保存、收藏与重开。隔离插件测试验证菜单/`Ctrl+4` 打开且复用独立窗口。证据在 `ppocr-window-build-182bxflk`；这不是完整视觉对照或真实官方 API 服务验收，未依赖 API key 发起外部解析。
- 完整 OCR/qAnyLabeling 原生回归曾为 **35/35 PASS，零跳过**；补充检测框过滤/奇异裁剪修复后的受影响回归为 **12/12 PASS，零跳过**，其中包含四组门控测试。这证明已实现范围的回归，不证明本表所有行完成。
- 新增 source OCR 的正式消费者检查为 CPU/CUDA/Vulkan 各 6 个 preset 加 3 个窗口路径，返回码全部为 0；它只覆盖 F32 local-only 产物。
- 点云上游对照为 **15/15 场景像素相同、87/87 手势相同**；比较范围明确排除完整窗口 chrome、字体和 DPI。
- 当前整体验收仍为 `complete_upstream_parity=false`；主要原因是新权重未发布、F16/Q8 未闭环、OCR 专用窗口剩余流程/VLM/高级 Chat/VQA、完整窗口视觉以及 Windows/macOS 安装包均未完成。

## 下一阶段顺序

1. 每个 F32/F16/Q8_0 文件分别完成源 probability/argmax、真实页面 Hmean/CER/WER、三后端数值一致性及 CUDA/Vulkan 受控吞吐/显存和同机 baseline；已有公共模型也需按这些要求复核。
2. 仅将全部门通过的文件上传既有 `PP-OCR` tag，再注册真实 release SHA/catalog/manifest，验证冷缓存自动下载。用户已授权上传，无需再次询问；目前没有新文件满足所有门。
3. 移植上游 PPOCR 专用窗口和 VQA/Chat 高级工作流，增加截图/交互/迁移测试。
4. 完成剩余本地模型 catalog 和 AICore consumer 后，再做 Linux clean-package、Windows、macOS 三平台 release 验收。

### 2026-10-19 补充：强身份页面与专用 OCR 文档导出

`public-pages-v4-18iwva6p` 的七个 F32 源参考族均已正常退出并通过，包括
SROIE `050–099` 留出集。CUDA 回放 28 项为 **24 PASS / 4 FAIL**，运行前后
构建产物身份未变；这是 developer 单轮结果。四个失败属于旧公共 Q8 文件：

| 流程 | 失败差值 | 结果 |
| --- | --- | --- |
| 中文街景 Q8 | 完整流程 CER `+0.009708738` | FAIL |
| 英文街景 Q8 | 完整流程 CER `+0.005295675` | FAIL |
| 英文 SROIE `000–049` Q8 | 固定/完整流程 WER `+0.006833713 / +0.009286841` | FAIL |
| 英文 SROIE `050–099` Q8 | 固定/完整流程 WER `+0.006207675 / +0.006395786` | FAIL |

两个未发布修复 SHA 的七组 CUDA 页面流程全部通过，包含中英留出集，未按留出集
重新调参。留出集中文固定/完整流程 CER 差值为 `-0.000279779 / +0.001259005`，
英文为 `-0.000384696 / -0.000104917`；两组 Hmean 差值均为 `-0.000751945`。
Vulkan 新身份回放同为 **24 PASS / 4 FAIL**，失败指标与 CUDA 相同，
两个修复 SHA 的七组 Vulkan 页面也全部通过；CPU 回放仍在运行。
受控性能仍未通过；本轮新上传仍为 **0**。

新增 ONNX 源模型的 Vulkan 页面补测现已完成：
`current-v19-pages-vulkan-0/matrix.json` **48/48 PASS**，执行前后构建身份未变。
这是先前缺失的页面行；CPU Medium 超时及中断记录不因此升级为通过。

专用 OCR 窗口新增文档 ZIP：读取页级 block 顺序、保留修订文本，生成源兼容的
`doc_0.md`、`imgs/` 裁剪图及 `layout_det_res_N.jpg`。复用流式 ZIP 写入器，
视频数据集保留原来的顶层目录规则。Python 独立解码检查 CRC、JPEG 尺寸及成员内容；
两页修订中文/重复图片夹具的 Markdown 和图名与实际 upstream renderer 相同。
布局图的字体/JPEG 像素尚未对齐，不将结构相同写成完整视觉一致。

设置窗口保存的自定义 API URL 现用于后续提交；新增图片/PDF 类型和新旧排序。
实际 Qt 设置交互及本地 HTTP 模拟服务覆盖 URL 使用、在途切换记录、取消/超时后
重新解析。两只 deadline/poll timer 使用精确定时器，避免提前触发造成额外轮询。
`ppocr-window-build-182bxflk/native-integration-results.json` 的 OCR 窗口、实际
插件入口、视频窗口及视频数据集导出四组均返回 0。测试不调用外部付费 API，
不等同于真实官方服务或完整原生包验收。原生 PDF、专用表格/公式编辑、远程图片
缓存、剩余视觉状态及高级 VQA/Chat 仍待完成。



### 2026-10-19 补充：VQA 目录导出与组件迁移

- 实际插件的原生测试通过组件配置 JSON 导入/导出、排序、双图标题重命名、
  radio 选项按位置改名、checkbox 删除选项重置和组件删除；保留无关字段。
  组件编辑时类型不可变，文本组件不启用选项输入，与上游编辑约束一致。
- 对损坏的后续侧车文件先报错，当前标注、既有配置与输出不改写。
  所有待写侧车先解析再修改，使用逐文件原子保存；磁盘写入中途失败会报告，
  这里没有承诺整目录回滚。
- 全目录 JSONL 覆盖选择字段、重命名、未标注图默认值和重复导出列按行覆盖顺序。
  双页 fixture 的 JSON 对象与实际执行的上游 `VQADialog.export_labels` 一致；
  标题/选项/删除迁移 fixture 与上游对应的四个方法及其 helper 一致。
  上游参考仅替换文件选择等 UI，不是完整源 GUI 截图验收。
- 原始证据：`build_app/Testing/ocr-storage-reference/ppocr-window-build-182bxflk/`
  的 `chat-integration-result.json`、`chat-integration-commands.json`，以及
  `vqa-export-reference-fpckst1b/comparison.json`、`migration-comparison.json`。
  源 VQA 文件 SHA-256：`31189935ffcfcba8b6b3209db59560769bdee4f5492fff4ddae86d9247a36749`。
- Qt 5 正式编译使用 `QT_USE_QSTRINGBUILDER`。隔离 OCR 窗口目标已开启同一设置，
  修复临时字符串/字节数组的延迟表达式生命周期；窗口/服务、实际菜单、视频 UI
  及视频数据集四组测试通过。新增功能仍缺完整视觉和原生包。

### 2026-10-19 补充：VQA 模板与 Chat ShareGPT 导出

- VQA 模板库支持新增、双击修改、删除和互斥选择；11 个系统模板只读。
  全部默认名称和提示词逐项匹配上游 `vqa/config.py`，其 SHA-256 为
  `02a95ae49e2623f51fe8c95b78438c1dfb515a5c3841525a5fcfdb06f5305081`。
- 组件和模板读写上游相同的 `xanylabeling_data/vqa/components.json` 与
  `prompts.json` 格式；默认根目录为用户家目录。组件文件优先于旧 QSettings
  缓存；原生插件测试验证窗口重开恢复、损坏文件保留和明确导入后的恢复。
- Chat 新增当前目录 ShareGPT ZIP 导出，使用已共享的 64 KiB 流式 ZIP 写入器。
  四轮消息、`images/second.png` 和原图 SHA 与实际执行的上游嵌套导出方法一致。
  目录中损坏的标注会使本次导出失败，既有 ZIP 保留；这比源程序跳过错误文件
  更容易发现遗漏。当前支持图片 basename；非本地或子目录图片会明确报错。
  数据集导入、AutoRun、完整消息控制及视觉仍待完成。当前上游没有 Alpaca 导出入口。
- 最新证据目录 `build_app/Testing/ocr-storage-reference/vqa-export-reference-wemgqolp/`
  包含 `comparison.json`、`migration-comparison.json`、`sharegpt-comparison.json`
  与原生 ZIP。Chat 源文件 SHA-256 为
  `541936f68f9d0f0a218f81320cd10d9819f5412abd1089cc2eb4d3295c19578c`。
  原生插件 SHA-256 为 `a702b8dbba2844a47435c25672e8b4bbd80469c01654bd5d1a48171749179aca`。
  这些是有限夹具对照和 Qt 5 插件测试，不是完整窗口或原生安装包验收。

最新 CLS F32 CUDA release 三轮交错 A/B (`cls-release-cuda-retry-4it5kq8o/matrix.json`)
仍为 FAIL：候选前两轮未达到 GPU 空闲门；第三轮原生采样通过，但对应官方源 GPU
采样被干扰，不能计为受控性能通过。没有新上传；原始失败和采样档案保留。


### 2026-10-19 补充：VQA 引用、取消与 Chat 数据流程

- VQA 从当前任务真实 sidecar（含自定义输出目录）读取固定 label 字段与
  `vqaData`，保留键顺序、嵌套 Python `str`/`repr`、浮点数和 Unicode 14
  转义；依上游顺序替换 `@text`、`@widget`、`@label` 与 `@image`。
  损坏 JSON 明确阻止请求，不覆盖文件；当前内存状态不冒充磁盘引用。
- 专用提示窗口使用上游 600 像素最小宽、24 像素边距、120–160 像素输入区、
  引用高亮与模板入口；420×180 模态加载窗可取消，结果窗先预览再应用。
  图像切换、取消、关闭和 Esc 均取消旧请求。原生测试通过；尚未完成完整
  PyQt6/Qt5 字体、DPI、平台 window chrome 和所有视觉状态验收。
- Chat 有界 folder AutoRun 保留各图像独立 history、错误计数和取消代次；
  ShareGPT 导入保留 system/user/assistant 角色、按第一个存在图像建 sidecar，
  导出保留多轮消息和原图。新 sidecar 默认版本统一为源 `4.1.3`。
- 最新原生证据为 `build_app/Testing/ocr-storage-reference/
  vqa-export-reference-e_9wn16g/`。三个上游对照脚本共七项对照全部通过，
  `upstream-comparison-results.json` 记录返回码；插件 SHA-256 为
  `5a45728107989e13ce0b2bade761c96fab76b0e83eef83d74b24cfaa436bb835`。
  另四组 OCR 文档、真实插件入口、视频 UI、视频数据集回归返回码均为 0；
  严格 AICore 插件边界检查通过。
- 独立 formatter 探针 `vqa-python-repr-97_a3mxr/comparison.json` 对照
  CPython 3.11：覆盖全 0x110000 码点及随机 binary64/嵌套值，共
  **1,124,117 项、0 差异**。这验证字符串序列化，不是模型精度或完整视觉。
- 布局量化诊断保留 `layout-rounding-527jk1k0` 的六项和
  `layout-rounding-stage-u7tge9i8` 的十二项失败结果。即使仅检测头使用
  F16、其余 98.809% 参数保留 F32，argmax 仍为 0.998421，未达到 1.0；
  该文件含 **0% Q8**，不得称为 Q8 修复或降低门限后发布。
- CLS F32 Vulkan release 三轮交错 A/B `cls-release-vulkan-rnby3lq0`
  同样 FAIL：baseline/candidate 的 GPU 空闲审计未通过。CUDA 第三轮部分
  采样通过也不能合成为完整 cohort PASS。新上传仍为 **0**。CPU 新身份
  页面队列仍在执行，逐文件索引只统计已完成、构建身份未变化的报告。

## 2026-10-19 本轮窗口与发布证据补齐

- 38 个发布候选已按原 SHA 复制到持久 staging，并复核大小/摘要一致。
  其中 36 个是 12 stem × F32/F16/Q8_0，另两个是不同 SHA 的公共 Q8 修复。
  逐文件路径、实际存储比例和失败/缺失门见 OCR_RELEASE_GATES.md。
- 当前 36 个新文件三后端源图均有单轮通过证据。13:39 UTC 的源页面
  快照为 CPU 21/54、CUDA 42/54、Vulkan 54/54 PASS；CPU/CUDA 仍在执行。
  两个修复候选的三后端组合页面和源图均已通过开发验收；英文 rec 只参与七组中
  的三组，不把未参与的场景计入该文件覆盖率。
- 公共 layout F16/Q8 仍有严格源 argmax 失败，旧公共文本 Q8 原文件的失败保留。
  GPU idle audit 拒绝的 A/B 不能说明模型速度不达标。13:28 UTC 在暂停自己的调度器
  后，外部校准进程重新出现，性能测试未启动，已恢复数值队列；原始记录为
  gpu-queue-performance-pause.json。完整受控性能和发布下载闭环仍未完成，新增上传 0。
- 新增专用 OCR 表格编辑器，支持 token/HTML/TSV、文本与字体、合并/拆分、
  行列增删、默认 3x3、剪贴板和 120 状态历史。修正公式标签的编辑器优先级、
  Python splitlines 的额外换行符，以及创建空表/粘贴分成两个撤销步骤的源行为。
  2,231 个 HTML5 实体及数字替换规则编译进插件，无运行时外置文件。
- 表格参考证据：/home/ludahai/develop/code/github/ACloudViewer/build_app/Testing/ocr-storage-reference/ppocr-table-inline-reference-hiodxzlw。5,407 项对照执行上游实际 token/HTML/TSV
  解析与 token 序列化方法，比较原生文本、样式、跨度和两次解析结果；
  其中 Qt 呈现对象使用站位对象，这不验证完整上游窗口视觉。
  原生操作回归覆盖默认表创建、粘贴撤销、字体、merge/split、行列增删及分支重做。
  OCR 文档窗口另验证表格编辑 Save/Cancel、原子保存、未知 JSON 字段保留。
- 表格新增 6 组列宽/行高方法对照和 2 组浅/深色 QSS 对照，通过上游实际方法。
  顶部图标工具栏、Cancel/Save 顺序、16 像素内容边距、选区按钮状态、右键
  多选恢复及快捷键已补齐。编辑器显式继承父窗口调色板；不修改宿主全局主题。
  样式对照包含 editors.py、style.py、theme.py、config.py 的 SHA；
  字体测量取原生运行值，不能用这些对照宣称 Qt5/Qt6 全窗口像素相同。
- 表格已改为结果块内嵌 QFrame，替换模态容器；按外层滚动视口、横向滚动条和
  20 像素底部余量约束高度，新增 6 项实际源高度方法对照。文档/代次检查阻止
  已失效编辑器的延迟 Save 写回；实际原子保存与取消路径均通过。
- Chat/VQA 原生证据：/home/ludahai/develop/code/github/ACloudViewer/build_app/Testing/ocr-storage-reference/vqa-export-reference-uy_4sw1l。8 个默认提供商、nullable 参数和未知字段、
  配置导入/导出、损坏配置保留、选中/收藏、45 项视觉能力判定及 1,060 项
  模型搜索对照通过；模型搜索执行上游实际 SequenceMatcher 方法。
  消息编辑/删除/重发/重生成及 ShareGPT/VQA 五组脚本均通过上游方法对照。
  修正延后销毁 VQA 窗口覆盖新 Chat 配置的问题，以及 Anthropic 模型发现请求头。
- 实际插件隔离重链接后 OCR 文档、插件入口、视频 UI、视频数据集四组回归通过；
  同一插件 Chat/VQA 原生测试通过，插件 SHA-256：
  2714c02dd171715ae2a876928a0c2ee47156cd4c15086c73f8140eaefcbcd8a4。
  上游全窗口布局、完整视觉和剩余编辑器仍需对齐。
- 已启动当前库 SLANet 的 18 项 developer TEDS/TEDS-S 矩阵：三种存储 ×
  裁剪表格/组合页面 × CPU/CUDA/Vulkan。输入为固定 SHA 的 100 张真实表格；
  GPU 部分等待当前 OCR 页面队列完成后串行运行。证据目录为
  build_app/Testing/ocr-storage-reference/current-table-pages-4r6llgbn/；
  当前运行不等于通过，更不构成受控性能验收。
  CPU F32 裁剪表格的 TEDS/TEDS-S 为 0.9645975/0.9959180，已通过；队列在
  组合页面阶段收到终止信号后，保留未完成记录并核对冻结 SHA 后恢复，原因未确认。
- 未关闭：原生 PDF 渲染、公式/富文本编辑器、完整 Markdown/code/math、
  模型自动刷新节流/提供商顺序/弹出布局、本地 VLM/剩余精确模型 catalog、
  Qt6/三平台原生包及 Linux clean-package。没有把局部窗口测试当成发布证据。


## 2026-10-20 当前差距及新证据

本节更新前面的历史队列快照。发布只以 CUDA/Vulkan 验收，原生 CPU 结果仅作诊断；
独立原始框架输出仍是精度真值，不能用后端彼此一致替代。

| 项目 | 当前结果/证据 | 尚未完成 |
| --- | --- | --- |
| 远程同步 | 已合并当时分支远程 `5a6a40c06` 和主线 `7bc6520bd`；HEAD `8db589b2d`；无冲突项；manifest 289 个唯一场景；随后按用户要求恢复打包相关暂存与工作区改动 | 没有推送这些合并或用户原有改动；之后远程新提交不在此同步收据范围 |
| 新模型真实页面 | 旧库 `154a0b62` 的 CUDA 54/54、Vulkan 54/54 PASS；CPU 54/54 为诊断 | 合并后新库 `9f2548ef` 的页面复核正在运行；release 重复稳定性、受控吞吐/显存 |
| SLANet | 旧库 GPU 12/12 PASS，另 CPU 6 项诊断通过；F32 裁剪 TEDS/TEDS-S 0.9645975/0.9959180，组合页面 0.5431670/0.5917242 | 新库复核已排队；完整 release 稳定性、受控性能 |
| 原始 CUDA 图真值 | 新增模型 CUDA 36/36、Vulkan 36/36 PASS；公共 Paddle 两后端各 10 PASS / 2 FAIL，失败为布局 F16/Q8；比较实际原始 CUDA probability/argmax | 最终矩阵以原始报告为准，developer 延迟不具备性能资格 |
| CPU 门控策略 | 48 个图场景移除原生 CPU 指标要求；31 个探针测试和 100 个公共 runner 测试通过；保留独立源精度门 | 不放宽原始模型的精度或 GPU 审计门限 |
| 受控性能 | gpu-release-20261020-py6cqztd 的两后端三轮均在计时前因 Xorg/桌面 SM 活动被拒绝，未产生合格延迟 | 实际空闲 GPU 下完整 baseline/candidate/source CUDA 交错 A/B |
| 量化精度 | 新 36 文件页面已通过；公共 layout F16/Q8 严格源 argmax 仍失败，旧公共文本 Q8 失败不被修复候选的结果覆盖 | 布局精度修复、独立 GT；每个新 SHA 单独验收 |
| OCR 富文本 | 新增块内编辑、Markdown/HTML、H1-H6、格式和保存/取消；13 序列化、6 标题指标、12 类粘贴、3 组裁剪预览、6 组高度、浅/深两主题 QSS 和标题弹层对照；新增图标及图片选择框，两组滚动/删除测试通过；实际窗口及正式插件回归通过 | 完整窗口；公式、PDF 原生流程仍缺 |
| Testing 清理 | 删除四个过期环境/轮子缓存及验收后的临时依赖载荷，累计释放 16.87 GiB；保留当前环境、GGUF、日志、摘要和删除清单 | 继续运行所需文件不得删；清理不是模型发布证据 |
| Linux 打包复核 | 实际远程 focal 构建、打包、安装后测试成功；撤回将本地问题扩大成远程打包缺陷的结论；21 个现有文件及 5 个新增文件已恢复/移除，保留同步后 HEAD | 既有绿色 CI 不覆盖当前未提交插件改动；本地安装器队列已取消，后续使用原有 CI 验证 |
| 发布与安装 | 尚无新上传；发布授权已有，但没有文件通过完整门控 | catalog/SHA/manifest/冷缓存下载、本地 VLM、Linux clean-package、Qt6/跨平台原生验证 |

原始记录位于 `build_app/Testing/ocr-storage-reference/`：
`cleanup-20261020/`、`gpu-release-20261020-py6cqztd/`、
`gpu-official-truth-20261020-5mjkehld/`、
`gpu-paddle-truth-fixed-20261020-67514g3z/`、`current-table-pages-4r6llgbn/`。
富文本序列化对照明确只执行源决策与原生 Qt 导出，不证明 Qt5/Qt6 像素等价。

Linux 安装调查：既有 Oct 14 安装树的主程序为 0 字节，缺少当前 AnyLabeling
插件，只说明该本地旧产物不可复用，未证明远程 focal 打包失败。此前增加的
DESTDIR、依赖探测和启动器改动已撤销；12 项测试和隔离载荷记录仅保留作历史
调查，不计为恢复后代码的验收。远程实际安装包已通过安装后测试；当前未提交
插件改动仍需原有 CI 验证。最终命令、退出码和构建 SHA 见
`build_app/Testing/ocr-storage-reference/acceptance-20261020.json`。

新库日文 XFUND F32 首轮的 CER/WER 增量均为 0，但正式构建重新生成
`CMakeCache.txt`，公共 runner 因构建身份改变给出 FAIL。保留该失败报告，
未将场景内数值通过升级为整体验收通过；串行后续队列将在构建稳定后重跑，
并补验 SLANet 图与受控 release sentinel。状态入口为
`merged-gpu-followup-path.txt`。`merged-clean-installer-path.txt` 所指本地
安装器队列已按用户要求取消，不再等待该产物。

### SLANet 原始图与布局存储诊断增量

- 重新获取官方 Paddle 3.0 `SLANet_plus_infer.tar`，归档 SHA 为
  `68526c5c6573a8a17cdc14805e026f11c2545e42d89a1317bb58d3e1e4b58318`；
  参数 SHA `012986b0e2bfe90410618bfb4b7cf4fcb6c978caefd84c63c2934f8387b09ab8`
  与现有 GGUF 来源一致。原始 JSON/PIR 图的两个输出为位置和结构概率，必须保留
  EOS，丢弃其后的导出填充行。首个官方输入的 CPU 诊断相对既有夹具：概率最大误差
  `1.847744e-6`、位置最大误差 `3.576279e-7`，结构 argmax 完全一致。
- 新增 `run_table_paddle_probe.py`：GPU 行直接调用独立官方 CUDA 图，不运行原生
  CPU 比较；支持原始概率/位置导出、采样协议检查、进程级 GPU 审计和源 CUDA
  性能比较。重复进程的原始输出保存在不同目录，并核验输入、程序、参数和探针
  在运行期间未变化。27 项表格/Paddle 探针单元测试通过。公共 manifest 中原有的
  三个 `table-structure-*` 消费者已改为执行该原始图对照，唯一场景总数仍为 289；
  正式 CMake 探针的构建及复核仍在队列中，没有发布资格。
  入口为 `table-original-graph-path.txt`。
- 布局模型按张量降精度的单图诊断已结束，另外预先选定两张图片作保留集检查。
  F32 控制组两图 argmax 均为 `1`；F16 候选分别为 `0.9923076923`、
  `0.9994736842`，另一个候选分别为 `0.9923076923`、`1`。后者实际上只有
  F32/F16 张量，**没有 Q8_0 张量，不能称为合格 Q8 模型**。不推广这些候选、
  不修改发布 catalog，不把这些 CPU 诊断作为 GPU 发布门控。完整存储组成与
  保留集记录见 `layout-tensor-rounding-path.txt`、`layout-rounding-holdout-path.txt`。

### 远程 CI 复核及打包改动恢复（2026-10-20）

用户质疑后重新获取实际 GitHub job/step 和完整日志。此前将本机陈旧安装树及
额外验收条件的问题推广为 focal 发行打包缺陷，证据范围错误，撤回该结论。

| 实际证据 | 结论及范围 |
| --- | --- |
| [Ubuntu run 39273926284](https://github.com/Asher-1/ACloudViewer/actions/runs/39273926284)，提交 `5d14860893` | focal/jammy/noble 的 Docker build、安装包上传和 Docker test 均成功；focal Docker 中 Python 测试 845 PASS / 14 skip |
| 同一 run 的 [focal 安装产物测试](https://github.com/Asher-1/ACloudViewer/actions/runs/39273926284/job/118215154142) | 实际安装 `.run` 到 `/home/runner/ACloudViewer`，headless 126 PASS / 1 skip，JSON-RPC 138 PASS；该 job 在 `ubuntu-latest` runner 安装 focal 产物，不能改称纯 focal 最小环境验收 |
| [CUDA run 39273926321](https://github.com/Asher-1/ACloudViewer/actions/runs/39273926321) | cuda-focal/jammy/noble 的构建、打包、Docker test 和安装产物测试均成功；这不代表真实 CUDA/Vulkan 模型精度或受控性能已验收 |
| 本机 Oct 14 旧安装树 | 主程序 0 字节且缺当前插件，只证明该本机旧产物不可复用；未从本机记录证明远程同条件故障 |
| 本轮额外 DESTDIR、含空格目录和依赖失败注入 | 与 CI 使用 `/root/install` 构建、普通 home 目录安装的条件不同；局部复现不等于 focal CI 失败，不应据此扩大本次发行流程改动 |

已将 21 个现有文件恢复为同步后的 HEAD：根 CMake/PostInstall、Linux 打包与
CUDA bundler、三个启动器/Qt IFW 脚本、跨平台 CI helper/插件开关、CodeQL、
macOS 容器配置及对应构建文档。另移除本轮新增的 InstallStaging 和四个相关
打包测试文件。工作区和暂存区在这些路径上均与 HEAD 一致；远程已提交更新保留。
修改前的内容、索引和补丁逐文件保存在
`build_app/Testing/ocr-storage-reference/packaging-reassessment-20261020/`，
包含 `restoration.json`、远程 job 元数据、原始日志和行号摘录。

本地新安装器生成和依赖它的容器队列已取消，不再使用被撤销的代码出包；只继续
正式插件/共享缓存回归和 OCR GPU 队列。AnyLabeling 通过 AICore 推理，不依赖
独立 qRAM/qOCR/qRelateAnything 插件，所以本次也撤销发行开关扩展。旧的 12 项
本地打包测试及容器载荷记录仅作历史调查，不计为当前代码的完成项。尚未发布的
工作树改动仍需既有 CI 正常构建/安装验收；不宣称它们已被历史绿色 run 覆盖。

模型缓存也已有明确所有者：OCR URL/catalog/cache 由 `ocr_capi.h` 和
`src/tasks/ocr/model_catalog.cpp` 提供；qOCR 与 AnyLabeling 均使用公开 catalog、
共享 `ecvModelDownloader` 和 `ecvAssetIntegrity`。未发布的新模型只允许本地转换
文件，不能先登记为可下载模型。通过全部 GPU 门控后才上传到原 PP-OCR release，
并一起更新 catalog、SHA、manifest 消费者和空缓存下载验收；不再增加私有缓存接口。
现已用全新且含空格的 `CLOUDVIEWER_DATA_ROOT` 验证既有公开
`ppocr-det-ch-f32.gguf`：OCR/table C ABI 返回相同缓存路径，从 catalog URL
下载 4,857,440 字节并匹配 SHA `aaad25f6…e06b11b9fd`；第二次离线复用不下载，
篡改后离线请求因 SHA 不符失败。入口 `ocr-shared-cache-cold-path.txt`；只证明
这一公开文件的共享缓存链，不覆盖未发布文件和插件 UI 下载交互。

SLANet 新探针已用统一生产编译定义完成独立编译，三个官方原始输入的 F32 CPU
诊断通过：概率最大误差 `1.33365393e-6`、位置最大误差 `1.31130219e-6`、
结构 argmax `1`。这是探针接线证据，不是 GPU 发布门控。报告入口：
`table-probe-isolated-path.txt` 所指目录的 `closure-report.json`。保留此前隔离编译
参数缺失及混合对象定义导致的失败记录；正式 CMake 目标补齐相同编译定义并
构建通过，原始图 CPU 诊断通过，GPU 验证在串行队列中。

公共下载/解包链已补上 SLANet 官方归档的实测闭环：归档成员带 `./` 前缀，
旧逻辑按字符串查找会误报缺失。现在仅对已固定 SHA/大小的目标成员规范化路径，
拒绝绝对路径、父目录跳转、符号链接和规范化后重名；仍由公共 runner 负责下载、
校验和原子写入。102 项公共 runner 测试通过，实际重新下载官方 7,997,440 字节
归档并解出三个匹配 SHA 的源文件。证据入口为 `slanet-common-archive-path.txt`。

使用与生产目标相同编译定义的独立探针，公共 manifest 的六个 GPU 原始图场景
全部通过：CUDA/Vulkan × F32/F16/Q8_0，各场景比较三个固定输入，结构 argmax
均为 `1`。CUDA 概率最大误差分别为 `2.32458e-6`、`0.00122018`、`0.0168662`；
Vulkan 分别为 `0.000192106`、`0.00113231`、`0.0167495`。结果记录在
`slanet-common-gpu-path.txt` 所指目录的两个 `matrix.json`。这是 developer 精度
证据，正式构建、页面 TEDS/TEDS-S、重复稳定性和受控性能仍分别验收；未将其
升级为完整 OCR 发布通过，也没有上传新文件。

共享 Qt 下载服务的冷缓存补测发现并复现另一个真实缺口：已下载并记录 SHA 的
文件被修改后，默认快速检查会删除过期 ledger，随即仅凭文件头/大小接受该文件。
现在共享 `ecvAssetIntegrity` 保留这份已失效证据并持续拒绝快速命中；显式
`DeepVerify` 成功后才刷新证据。普通暖缓存仍只检查 stat，不增加逐次全文件哈希。
qOCR 的 `modelPresent()` 也从单纯大小判断改用这项共享检查，AnyLabeling 已在
使用该服务。无既有证据的历史本地文件仍遵循原有 miss 策略，不将它称为 SHA 验证。

保留旧实现的失败报告，新库的真实 Qt 下载补测通过：同一公开 OCR F32 文件，
全新含空格数据根目录，公开 catalog URL 冷下载、SHA 与 ledger、暖命中不再调用
transport、篡改拒绝和恢复后的显式校验均 PASS；程序及相关库身份前后不变。
证据入口 `ocr-qt-shared-cache-before-path.txt`、`ocr-qt-shared-cache-path.txt`。
共享服务的回归还覆盖重复快速查询、空 pin 查询、失败的深校验之后持续拒绝和
pin 更新；正式 CMake 的共享服务/TRELLIS 兼容性与七项 AnyLabeling 回归
共 9/9 PASS，构建前后 AICore 和 OCR 探针 SHA 不变。收据为
`merged-clean-package-path.txt` 所指目录的 `build-result.json`、`tests-result.json`
和 `tests.log`，该目录名沿用旧队列名，实际未生成安装包。该补测不代替插件窗口交互、
未发布候选或 GPU 发布验收。

### Chat/VQA 模型自动刷新

复用现有配置文件与 `ChatServiceClient`，补上初始化、提供商切换和菜单打开的
自动发现。Chat/VQA 共用进程内提供商调用计数，Ollama 每次查询；其他提供商
按上游 `get_models_data` 的缓存/计时决策，手动刷新可绕过。模型发现另持有一个
相同客户端实例，不再占用或取消对话请求，仍写原有 `models.json`，没有新增模型
下载/缓存服务。发现超时与上游一致为 5 秒。

上游 `EventTracker.reset(provider)` 将计时器设为 0，因此首个 300 秒区间结束后，
后续调用继续触发刷新，而非持续的五分钟冷却；这里保留实际源行为并明确记录。
15 组受控时间/缓存/大小写/提供商条件与执行原始 Python 函数的结果相同；真实
插件 HTTP mock 覆盖自动/强制刷新、缓存命中、切换和对话并行，且首次空缓存菜单
在异步列表返回后自动填充，完整原生工作流
及源比较均 PASS，随后正式 Chat/plugin 测试 2/2 PASS。AICore/OCR 探针 SHA
不变。入口为 `chat-model-refresh-path.txt`。另修正模型菜单向上展开的位置和
可用高度，真实插件坐标与执行上游定位方法一致；浅/深主题工作流及正式 2/2
回归通过，入口 `chat-model-position-path.txt`。这只覆盖相对定位规则，完整
对话/设置面板布局与 Markdown/code/math 尚未完成；Qt6 原生视觉未验证。

### 布局量化失败的原始输出定位

对现有失败报告中的八个原始头只读统计，没有再使用保留集调参：CUDA/Vulkan
的 F32 均无翻转；F16 各有 22 个类别 argmax 和 40 个 DFL edge-bin 翻转，Q8_0
各有 253 和 419 个。发生类别翻转的位置，其原始最大分数分别不超过
0.00922464、0.0403007。低分位置也属于用户要求的严格原始输出门，故不能
改成仅比较后处理框来宣告通过；DFL 分布变化也不能仅凭分数推断无影响。
证据入口 `layout-flip-margins-path.txt`，未修改权重、阈值或发布 catalog。

### 独立人工布局标注与页面评估

补入 CDLA 官方验证集全部 1,000 页的独立人工 GT，共 8,711 个区域。
`core/AICore/tools/ocr_cdla_validation.json` 记录每页图像/标注 SHA、完整
2,165,389,258 字节归档的 SHA 及来源提交。不是从现有预测框生成标注，也未
使用验证页选择量化参数。实际执行官方转换器和 LabelMe 栅格化函数，与新
GT 消费者的全部类别、bbox 和 area 相同。COCO annotation ID 使用正整数，
避免零值未匹配哨兵；按类别评价 AP50:95/AP50/AP75/AR100，验证全页面覆盖。

原始 Paddle 导出器执行固定 SHA 的官方 PicoDet 后处理，保留 score=0.5、
NMS=0.5；原生导出器仅使用公开借用图像/typed-region C ABI。现有 API 仍需
det/rec，因此这项布局精度隔离测试将 DB 阈值设为 1 并强制文本框数量为零，
布局继续读取完整原图；不据此声称整套 OCR 流水线性能或质量。17 项评估器
手动测试通过。原始 CPU 图正常完成 1,000 页，原生 CPU 三页接线通过，均仅
诊断。正式 CMake 测试已登记，但为保持活动 GPU 的构建身份尚未重新配置。

公共 runner 的空缓存归档下载/2,000 个文件 SHA/零下载离线复用全部通过；独立原始 CUDA
及 CUDA/Vulkan × F32/F16/Q8_0 全页面导出已排在现有串行 GPU 队列之后。
证据入口 `layout-cdla-source-path.txt`。GPU 页面评价、独立审核的页面容差和
公共 manifest 消费者尚未完成，严格原始 argmax 失败继续阻止发布；不会用
后处理 AP 或 CPU 诊断替换该门。


CDLA GPU 补验现已完成：原始 CUDA 加两后端三种存储均正常退出，覆盖全部
1,000 页。原始 CUDA mAP50:95=0.8546702421；CUDA/Vulkan F32 差值分别为
-3.951446e-7/-4.016570e-7，F16 均为 +6.319819e-5，Q8_0 均为 -0.0009133202。
正式 CDLA CTest 通过。页面数值接近并不解除严格原始 argmax 失败；页面发布
容差、公共 manifest 接入与受控性能仍待完成。当前 GPU 原始图/方向/页面队列
已应用原始 CUDA 参考选择修复，184 项生产脚本测试通过，实际 GPU 行逐项
运行，不继承旧 CPU 参考结果。


2026-10-20 Chat 增量：三栏宽度/伸缩、预览设置分割、Backend/Generation、温度滑条、
模型弹窗锚点及 21 组上游浅/深 QSS 已集成。实际工作区主题切换的背景和提供商前景色
对照通过；输入区支持原始 QTextEdit 动态高度、Enter/Shift+Enter、@image 高亮、
发送/停止复用按钮、空输入禁用及默认 No 的清空确认。插件真实工作流、九项源构造/
主题比较和六项正式 CTest 通过；OCR 在测程序/运行库身份未改变。证据入口
`chat-three-columns-path.txt`。完整消息 Markdown/code/math、字体/DPI 和窗口视觉仍未关闭。


2026-10-20 消息显示增量：静态源样式扩大为 33 对，补齐用户气泡、助手角色图标、
动作和编辑器样式；右键菜单动作顺序、复制反馈 1 秒与上游一致，继续调用现有
消息写入路径。浅/深色实际颜色、菜单打开期间切图销毁、复制/编辑与计时器均通过
实际插件测试；12 项源结构/主题/菜单对照及 6 项正式 CTest 通过。OCR 队列的
28 个执行依赖 SHA 未变。记录在 `chat-message-source-path.txt`；这仍不表示
Markdown/Pygments/公式或完整窗口视觉已经对齐。

原始 CUDA graph/orientation 最新结果为 128/132 通过；layout F16/Q8_0 各在两后端
失败。对已保存八头张量逐项比较：CUDA/Vulkan argmax 无分歧，且相对原图的失败
位置完全一致（F16 类别 22、DFL 边 40；Q8_0 类别 253、边 419）。两后端张量最大
差异均小于 0.000021；这是排查后端差异的证据，不能取消量化精度门。
详见 `build_app/Testing/ocr-storage-reference/layout-backend-attribution.json`。


2026-10-20 VQA 增量：补齐一基页码跳转、首尾禁用、空输入恢复、上界裁剪、
侧栏折叠及切图后保持、默认 No 的当前图清空、左右刷新入口。刷新复用工作区
已有当前图片加载路径，实际外部 JSON 修改能读回，未知 VQA 和 sidecar 字段保留。
原生插件测试和独立执行的上游页码/清空/取值/保存方法一致；正式六项 CTest
通过，OCR 的 28 个冻结执行依赖 SHA 不变。收据入口为
`vqa-navigation-source-path.txt`、`vqa-refresh-source-path.txt`。这关闭上述交互，
未关闭完整 VQA 窗口布局、所有组件/弹窗样式、字体/DPI 或跨 Qt 版本视觉。

受控 CUDA/Vulkan 重试的八组 release 预检通过，包含两份独立 SHA 的公共 Q8
修复，按既有原始 CUDA 和同机旧版
baseline、3/2/10 采样及空闲审核，串行等待当前页面和公共 Q8 修复队列。
匹配的旧版 SLANet probe 尚缺；未测行及预检不计性能通过。见
`gpu-controlled-retry-path.txt`；当前新增上传仍为零。


2026-10-20 VQA 样式增量：新增 14 对原始亮/暗 QSS，Chat/VQA 共 47 对，
替换重复的提示词/加载/按钮样式并应用到文件名、图片容器、预览、页码、分割、
导航和动作控件；图片容器留白与源代码一致。修复工作区先设置 palette 再清空
QSS 时恢复旧暗色的顺序错误，原生最小复现已确认；Chat 读取工作区的主题。
独立执行上游 QSS 函数与两种主题下 14 个实际控件逐字一致，选定渲染颜色及
工作流通过；正式 6 项 CTest 通过，28 个 OCR 执行依赖 SHA 未变。证据入口
`vqa-styles-source-path.txt`。尚不代表整个组件/弹窗、字体/DPI 或 Qt6 全窗口视觉对齐。

布局失败追加诊断：保留的单个原始 CUDA 输入上，两种量化的类别/DFL argmax
翻转均在 score .5 以下，但 F16/Q8_0 各有 1/2 个锚点跨越该阈值。不能由背景
翻转推导为任务输出无变化，更不能取消严格精度门。详见
`build_app/Testing/ocr-storage-reference/layout-decision-confidence.json`。
