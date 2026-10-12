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

det/rec fixture 是官方测试图（det `imgs/00006737.jpg`，rec `imgs_words_ch/word_1.jpg`、`words_en/word_1.png`）经官方预处理后送入 GGUF 源归档的 Paddle Inference 预测器得到的输出（`core/AICore/tools/export_ocr_fixture.py`，发布为 `ocr-fixtures` release 的 `*_page_*.npy`）。旧的均匀随机输入 fixture 是空门：DB 概率图最大值仅 0.0036（门限 0.01，全零输出也能通过），rec 参考 40/40 步全部是 blank。真实 fixture 首先暴露了一个移植缺陷：稠密卷积的 im2col 固定为 F16，f32 模型的激活在 GEMM 前被舍入（第一个 conv 相对误差已 5e-4，det 图 max\|diff\| 0.025）；F32 权重改为 F32 im2col + `GGML_PREC_F32` 后 CPU 降到 2e-5、CUDA 0.0102（剩余来自 2x2/stride-2 conv_transpose 快路径的 TF32）、Vulkan 0.0267（coopmat2 设备把 F32 GEMM 操作数转为 F16）。各档门限按 cpu/cuda/vulkan 实测最大值设定，并同时门控 `mean_diff`。

端到端对照以 `core/AICore/scripts/run_ocr_official_alignment.py`、`bench_ocr` 与 `bench_paddle_official.py` 运行。它按中心距不超过 30 px 一对一匹配框，报告空间匹配、逐字符串相同率和 p50/p95。逐字符串相同率是**与 Paddle 输出的差异指标**，不是无标签图像上的准确率：Paddle 自身也会错字、丢标点或切分单词。

### 已确认的归因边界（2026-10）

- f32 GGUF 的图级 argmax fixture 门通过，CUDA/Vulkan 的同一 F32 图输出无后端特异性文本漂移；因此“Paddle TF32”不能解释端到端文字差异，更不是充分根因。
- det/rec 量化会改变少数候选框或字符，但三种量化的字符串相同率处于同一范围；它不是 75.3% / 68.5% 数值的主导因素。
- 端到端差异发生在图级门覆盖不到的区域：DB 后处理的框几何、旋转裁剪/resize 语义、以及两个 OCR pipeline 对小字和标点的不同贪心 CTC 结果。
- `layout_doc_test` 的大量差异是 Paddle 缺少句尾 `.`、把 `Document` 切为 `Docume nt`、把 `placeholder` 切为 `place holder`；AICore 也有 `Bodly`、`0o/Oo` 等真实错字。没有逐框人工标签时，不能把任一侧字符串当作真值。
- 为检验插值假设，曾把 AICore 透视 crop 临时替换为 PaddleX 的 cubic + replicate 语义；F32 字符串相同数分别从 `90→91`（ppocrv4）、`70→69`（medal）、`37→36`（layout），没有净收益，改动已撤销。该候选不是可接受的根因修复。
- 2026-10-12 同权重分解（7 张 `ppocr_testdata/imgs`，CPU，诊断脚本未入库）：Paddle GPU 与 Paddle CPU 143/143 字符串相同；AICore cpu/cuda/vulkan 三者互相一致，因此差异不是数值问题。用 Python 复刻 PaddleX 的 `get_minarea_rect_crop` + cubic 透视 + 6 张一批的 rec，在 Paddle 自己的框上复现 286/286；把输入换成 **AICore 的框**、其余全部用 Paddle 语义，只有 252/286（88.1%），与 AICore 自身的 251/286 相当。主导因素是 det 后处理的框几何：其一，PaddleOCR 3.0.1 OCR 流水线实际使用 `limit_side_len=64, limit_type=min`（`text_det_params` 实测），而 AICore 默认 736/min（det 模块 `inference.yml` 的默认），短边 < 736 的图分辨率不同、连通域会合并或拆分；其二，即使两侧都用 736，框角点仍差 p50 0.64 px / max 2.3 px（Paddle 取整为 int，pyclipper 在整数坐标上做 unclip，AICore 全程浮点），亚像素裁剪差即可翻转小字。修复需要逐位移植 `boxes_from_bitmap` + `get_minarea_rect_crop` 语义并决定默认 det limit，属下一步工作。

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

Layout 的 `pplcnet-layout` 是区域类别模型，不应被上述文本指标误评。它的解码后区域由
`ocr-layout-regions-{f32,f16,q8_0}` 单独门控：`check_layout_regions.py` 把 `aicore_ocr_run`
返回的区域按同类别贪心 IoU 与官方 PP-Structure 输出（`run_paddle_layout_reference.py`
运行上游 PicoDet CDLA 模型与上游 `picodet_postprocess.py`，固定于
`ocr_fixtures/layout_regions.json`）匹配，要求 recall ≥ 1、F1 ≥ 1/1/0.95、mean IoU ≥ 0.99。
参考里 `table_page_1.png` 的一个 `table_caption` 正好压在阈值上：上游 f32 得分 0.50031（阈值
0.5），把同一 Paddle 图的权重舍入为 float16 后为 0.4998，即 Paddle 自己的 f16 权重也会丢掉它。
因此 f16/q8_0 行用 `--optional-region` 豁免这一个区域（找到仍计入，漏掉记 `optional_missed`、
移出 recall 分母），其余 26 个区域仍必须全部找到；f32 行不豁免。实测（cpu/cuda/vulkan）f32 27/27、
F1 = 1；f16 在 cpu/cuda 上漏掉该区域（`optional_missed = 1`）、vulkan 找到，其余 26/26、F1 = 1；
q8_0 recall = 1、F1 0.964。表格还原的语义门（TEDS）见 [`../table/README.md`](../table/README.md)。

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

根因（2026-10-12 更正）：不是 CPU 降频。RTX 4090 宿主未开启 Resizable BAR，Vulkan 的 DEVICE_LOCAL|HOST_VISIBLE heap 只有 246 MiB（nvidia-smi BAR1 256 MiB），而 ggml-vulkan 会把它当 ReBAR 优先用于设备缓冲。OCR + SLANet 同进程时 BAR1 占用涨到 238 MiB，窗口接近饱和后 CPU 向映射缓冲的 memmove 坍塌约 100 倍（upload p50 113 ms，perf 显示 73% 进程时间在 `ggml_backend_tensor_set` 的 memmove）；页错误、futex/poll/ioctl 耗时与 fence 等待在快慢两组间相同。`3rdparty/ggml/patches/vulkan_small_bar/0001-vulkan-no-rebar-small-bar-window.patch` 在设备初始化时检测「最大 host-visible VRAM heap < 最大 device-local heap 的一半」，自动改用 device-local 缓冲 + staging 上传（等价于上游 `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`，但由内存属性决定、不读环境变量）。同主机交替 3 轮 A/B：wall p50 134.6 → 22.0 ms，p95 约 325 → 49 ms，upload 113 → 1.47 ms，BAR1 238 → 10 MiB；1200 个输出哈希逐位一致，精度不受影响。在用户报告的完整表格 pipeline（det + rec + SLANet，100 张 `table_rec`，f16，每臂 300 样本，交替 2 轮，只替换 `libggml-vulkan.so`）上：p50 150.3/148.1 → 35.5/37.3 ms，p95 355.6/356.8 → 72.0/77.0 ms，BAR1 峰值 226 → 11–14 MiB，两臂输出逐字节相同（详见 [`../table/README.md`](../table/README.md)）。早先 `layout_doc_test` 68–92 ms 双峰与 powersave 归因均出自该问题之前的测量，不再作为结论。

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
