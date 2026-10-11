# AICore OCR：模型、正确性与端到端证据

本目录实现 AICore `ocr` 任务的模型加载、图构建与 C ABI。Qt 工作流和下载 UI 属于 [qOCR](../../../../../plugins/core/Standard/qOCR/README.md)。

## 数据流与所有权

```text
caller-owned aicore_image_view
  -> det preprocess / DB postprocess
  -> quad crop / rec preprocess
  -> rec graph / CTC decode
  -> caller-released aicore_ocr_result
```

模型、图缓存、设备 lease、阶段计时和错误均由任务 context 持有；调用方只借出图像内存，并通过公开 C ABI 获得带文本和 quad 的结果。实现不得读取任务私有环境变量，也不得把 ggml 类型泄漏到 ABI。

## 数值门与外部对照是两件事

`validation_manifest.json` 的 OCR 行是**图级数值门**：固定输入下，F32 GGUF 的 det/rec/layout 输出必须满足 Paddle 语义 numpy fixture 的 `max_diff` 门，并要求 rec CTC argmax 序列相同。它验证转换、图和后端计算，不等价于完整 OCR pipeline 的语义真值。

端到端对照以 `core/AICore/scripts/run_ocr_official_alignment.py`、`bench_ocr` 与 `bench_paddle_official.py` 运行。它按中心距不超过 30 px 一对一匹配框，报告空间匹配、逐字符串相同率和 p50/p95。逐字符串相同率是**与 Paddle 输出的差异指标**，不是无标签图像上的准确率：Paddle 自身也会错字、丢标点或切分单词。

### 已确认的归因边界（2026-10）

- f32 GGUF 的图级 argmax fixture 门通过，CUDA/Vulkan 的同一 F32 图输出无后端特异性文本漂移；因此“Paddle TF32”不能解释端到端文字差异，更不是充分根因。
- det/rec 量化会改变少数候选框或字符，但三种量化的字符串相同率处于同一范围；它不是 75.3% / 68.5% 数值的主导因素。
- 端到端差异发生在图级门覆盖不到的区域：DB 后处理的框几何、旋转裁剪/resize 语义、以及两个 OCR pipeline 对小字和标点的不同贪心 CTC 结果。
- `layout_doc_test` 的大量差异是 Paddle 缺少句尾 `.`、把 `Document` 切为 `Docume nt`、把 `placeholder` 切为 `place holder`；AICore 也有 `Bodly`、`0o/Oo` 等真实错字。没有逐框人工标签时，不能把任一侧字符串当作真值。
- 为检验插值假设，曾把 AICore 透视 crop 临时替换为 PaddleX 的 cubic + replicate 语义；F32 字符串相同数分别从 `90→91`（ppocrv4）、`70→69`（medal）、`37→36`（layout），没有净收益，改动已撤销。该候选不是可接受的根因修复。

下一步若要声明语义准确率，必须为每张发布图提供逐框人工标签（文本和 quad），再分别计算 CER/WER 与空间 precision/recall；不得把 Paddle 的预测字符串包装成 ground truth。

## 官方 GT 语义评测

`core/AICore/scripts/ocr_gt_eval.py` 消费 PaddleOCR 官方标注格式（`image<TAB>json`，每个
标注有 `points`、`transcription`，`###` 为 ignore）或等价 JSONL，并以四边形 IoU 进行
一对一匹配。它输出 detection precision/recall/Hmean、端到端 word accuracy、CER/WER，
并把漏检记作删除、额外框记作插入，因此不能通过少报文本来抬高识别结果。

`run_ocr_gt_matrix.py` 在固定的 GT SHA-256 和数据集 revision 下，执行完整矩阵：中文和
英文 rec 各自覆盖 det/rec `f32`、`f16`、`q8_0` 的 3×3 组合、CUDA 与 Vulkan，并将每行
Hmean/CER 与同一 GT 上 Paddle 的结果比较。它还记录 CPU governor、频率和 GPU compute
进程；`--require-performance-governor` 只拒绝不受控的性能运行，绝不修改主机状态。

已用公开 IIIT5K 转换仓库的 2 个随附英文样本验证了识别路径：GT JSONL 的
SHA-256 为 `2f15e354a4d42f19059cc7817b708373409a0cd67f8692090cf1ef3991589636`，
来源 revision 为 `5711d139a51a411c7097327ddeb47d7540d0e1d4`。英文 F32/F16/Q8_0
3×3 组合在 CUDA/Vulkan 共 18 行均为 Hmean=1、CER/WER=0，且未低于同输入 Paddle
GT 门。这只是两个 cropped-word 样本的 recognition smoke，不覆盖 det，也不构成
发布准确率或受控性能结论（宿主仍为 powersave）。原始报告保存在
`/tmp/ocr_check/iiit5k_sample/ocr_gt_matrix.json`。

Paddle 的公开文档列出了 ICDAR2015 检测 GT 以及英文 recognition benchmark；其中
ICDAR2015 下载要求官方站点注册。当前仓库和测试缓存尚未包含已许可下载且 SHA-256
固定的官方包，故完整 GT 矩阵尚不可执行，也不能声称已有真实端到端语义准确率。拿到许可数据后：

```bash
python3 core/AICore/scripts/run_ocr_gt_matrix.py \
  --ground-truth GT_LABELS.jsonl --ground-truth-sha256 GT_SHA256 \
  --dataset-revision icdar2015-<official-revision> --image-root GT_IMAGES \
  --bench build_app/bin/bench_ocr --models-dir "$HOME/cloudViewer_data/extract/ocr_models" \
  --languages ch en --backends cuda vulkan --require-performance-governor \
  --output build_app/Testing/ocr_gt
```

Layout 的 `pplcnet-layout` 是区域类别模型，不应被上述文本指标误评；它需要拥有类别
标签和 region quad 的 CDLA GT 合约后再单独登记。

## 端到端性能矩阵

同一 RTX 4090、同一 PNG、PaddleOCR 3.0.1 / paddlepaddle-gpu 3.0.0、3 warmups + 10 timed runs；每行是 det `{f32,f16,q8_0}` × rec `{f32,f16,q8_0}` 的九种组合范围。全部 54 个 AICore 组合均低于 Paddle 的 p50。

| 图像 | Paddle GPU p50 | CUDA p50（9 组） | Vulkan p50（9 组） |
|---|---:|---:|---:|
| ppocrv4，2566×1312 | 175.40 ms | 62.43–65.35 ms（×2.68–2.81） | 71.83–75.77 ms（×2.31–2.44） |
| medal_table | 99.87 ms | 34.19–35.07 ms（×2.85–2.92） | 43.59–45.47 ms（×2.20–2.29） |
| layout_doc_test | 125.96 ms | 53.44–55.02 ms（×2.29–2.36） | 68.28–92.35 ms（×1.36–1.84） |

![全量端到端延迟对比](images/latency_comparison.png)

## 文本与可视化证据

下图列出每种 backend × det quant × rec quant 的“精确字符串相同 / 空间匹配框”比例。它用于发现回归，而不是准确率发布门；对应的逐对差异见 [mismatch_pairs.md](images/mismatch_pairs.md)。

![文本一致率热图](images/text_agreement_heatmap.png)

绿色框表示与 Paddle 字符串相同，橙色表示同一空间框内字符串不同，红色表示 30 px 内无法配对。三栏依次为 Paddle、AICore CUDA q8_0、AICore Vulkan q8_0。

| ppocrv4 | medal_table | layout_doc_test |
|---|---|---|
| ![ppocrv4 效果](images/effect_ppocrv4.png) | ![medal 效果](images/effect_medal_table.png) | ![layout 效果](images/effect_layout_doc_test.png) |

## Vulkan 延迟离散性

`layout_doc_test` 的 Vulkan 68–92 ms 双峰已重复观测到，且在慢样本中 det/rec GPU 图时间仍约为 6.2 / 48 ms，放大的是 CPU pre/post/upload。`intel_pstate` powersave 主机允许核心在 `ggml_vk_wait_for_fence()` 的长 `waitForFences` 内降频；CUDA 的同步路径更积极自旋，掩盖了该宿主效应。4 核负载能把 Vulkan 收敛到约 68–76 ms，但同时让 CUDA 从约 54 ms 退化到约 61 ms，说明它不是可接受的运行时修复。

因此当前结论是：这是**基准宿主的频率状态依赖**，不是已证实的 OCR 或 Vulkan 数值缺陷。不会加入常驻保温线程或修改全局 ggml 等待策略；它们会改变其它任务的功耗/延迟。性能报告必须记录 governor、GPU 共用状态与 p50/p95。若产品需要低方差，应在部署层将指定 CPU governor 固定为 performance，并以独占主机重新做 A/B，而非在任务内隐式改变系统策略。

## 复现

```bash
# 图级数值门
ctest --test-dir build_app -R "probe_ocr|ocr" --output-on-failure

# 单图与 Paddle 的端到端差异报告
LD_LIBRARY_PATH="build_app/external/lib:build_app/bin:$LD_LIBRARY_PATH" \
GGML_VK_DISABLE_COOPMAT2=1 \
python3 core/AICore/scripts/run_ocr_official_alignment.py \
  build_app/bin/bench_ocr INPUT.png \
  "$HOME/cloudViewer_data/extract/ocr_models/ppocr-det-ch-f32.gguf" \
  "$HOME/cloudViewer_data/extract/ocr_models/ppocr-rec-ch-f32.gguf" \
  --devices cuda vulkan --warmups 3 --iters 10 --details
```

发布级证据还需要 `validate_all.py --profile release`、固定模型/输入 digest、受控的候选/基线 A/B，以及有标签的端到端语义门。
