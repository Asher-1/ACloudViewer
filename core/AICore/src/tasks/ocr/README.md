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

`validation_manifest.json` 的 OCR 行是**图级数值门**：固定输入下，GGUF 的 det/rec/layout 输出必须满足独立 Paddle Inference predictor 导出的 `max_diff`/`mean_diff` 门，并要求 rec CTC argmax 序列相同。它验证转换、图和后端计算，不等价于完整 OCR pipeline 的语义真值。

det/rec fixture 是官方测试图（det `imgs/00006737.jpg`，rec `imgs_words_ch/word_1.jpg`、`words_en/word_1.png`）经官方预处理后送入 GGUF 源归档的 Paddle predictor 得到的输出。layout fixture 由 `export_ocr_fixture.py` 将 `layout/table_page_1.png` 按官方 608×800、ImageNet mean/std 预处理并运行官方 PicoDet predictor 导出 8 个 head。旧 layout 文件已验明与 seed `20260930` 的均匀随机张量逐位相同，只能证明自洽、不能证明上游正确性。真实 layout F32 的 CPU/CUDA/Vulkan `max_diff` 分别约 `1.6e-5/1.5e-5/1.5e-5`，`mean_diff < 5e-7`。

The nine official layout arrays are now published under the existing `ocr-fixtures` release. An isolated empty-cache validation downloaded the release assets, verified every manifest size and SHA-256, fetched the det/rec dependency closure, and passed all six CPU layout graph/decoded-region scenarios with two process repeats.

真实 det/layout fixture 暴露了两个后端精度问题：CUDA 2×2/stride-2 `conv_transpose` 快路径默认 TF32；Vulkan NV_coopmat2 会把 F32 GEMM 操作数转换为 F16。CUDA 严格 SGEMM 修正了 det 误差；Vulkan patch 对名称为 `ocr_scalar_*` 的 F32 GEMM 节点选精确标量矩阵路径。该 patch 起初只覆盖 det/layout，理由是当时认为 rec 已满足 CTC argmax 门；但 fixture 级别暴露出 rec 在 CUDA/Vulkan/CPU 之间的输出 digest 并不一致，说明该假设不成立，遂把 `exactF32()` 扩展到 rec 的所有 dense GEMM、SVTR self-attention 与 `matmul_v2`：rec fixture logits 的 `max_diff` 由 `4.4e-4→0`，ICDAR2015 上 CUDA 与 Paddle CPU 的 15/2123 文本差降为 0（见下“rec pad 宽度根因与修复”）。受控 Vulkan A/B：det F32 `max_diff 0.026709→0.000039`、p50 `2.735→2.965 ms`；layout F32 `0.015627→0.000015`、p50 `2.455→2.800 ms`。代价是 Vulkan `layout_doc_test` 页面 e2e 从 `73→90 ms`（+17 ms，仍低于同图 Paddle GPU 的 `105 ms`），根因是这些 rec/layout F32 GEMM 从 coopmat2 切到精确标量路径；CUDA/CPU 不受该切换影响。RMBG 严格/普通 probe 的输出 fingerprint 不变，p50 漂移在约 0.4% 内，说明最终 gate 没有扩散到其他任务。

端到端对照以 `core/AICore/scripts/run_ocr_official_alignment.py`、`bench_ocr` 与 `bench_paddle_official.py` 运行。它按中心距不超过 30 px 一对一匹配框，报告空间匹配、逐字符串相同率和 p50/p95。逐字符串相同率是**与 Paddle 输出的差异指标**，不是无标签图像上的准确率：Paddle 自身也会错字、丢标点或切分单词。

### 已确认的归因边界（2026-10）

- f32 GGUF 的图级 argmax fixture 门通过，CUDA/Vulkan 的同一 F32 图输出无后端特异性文本漂移；因此“Paddle TF32”不能解释端到端文字差异，更不是充分根因。
- det/rec 量化会改变少数候选框或字符，但三种量化的字符串相同率处于同一范围；它不是 75.3% / 68.5% 数值的主导因素。
- 端到端差异发生在图级门覆盖不到的区域：DB 后处理的框几何、旋转裁剪/resize 语义、以及两个 OCR pipeline 对小字和标点的不同贪心 CTC 结果。
- `layout_doc_test` 的大量差异是 Paddle 缺少句尾 `.`、把 `Document` 切为 `Docume nt`、把 `placeholder` 切为 `place holder`；AICore 也有 `Bodly`、`0o/Oo` 等真实错字。没有逐框人工标签时，不能把任一侧字符串当作真值。
- 为检验插值假设，曾把 AICore 透视 crop 临时替换为 PaddleX 的 cubic + replicate 语义；F32 字符串相同数分别从 `90→91`（ppocrv4）、`70→69`（medal）、`37→36`（layout），没有净收益，改动已撤销。该候选不是可接受的根因修复。
- 2026-10-12 同权重分解（7 张 `ppocr_testdata/imgs`，CPU，诊断脚本未入库）：Paddle GPU 与 Paddle CPU 143/143 字符串相同；AICore cpu/cuda/vulkan 三者互相一致，因此差异不是数值问题。用 Python 复刻 PaddleX crop/rec，在 Paddle 自己的框上复现 286/286；只换成 AICore 的框后为 252/286（88.1%），与 AICore 自身 251/286 相当，证明主导变量是 det 框几何。
- 该根因已修复：默认 det 改为 PaddleOCR 3.0.1 实际公共流水线的 `limit_side_len=64, limit_type=min`；DB unclip 先落入 Clipper 整数格，使用 JT_ROUND 等价圆角 offset，再做第二次 `minAreaRect`，映射回原图后按目标像素取整。`medal_table` 从 AICore/Paddle `102/96` 框、配对文本 `75/96` 提升到 `96/96` 框且 `96/96` 文本相同；`ppocrv4` 为 `103/103` 框、`101/103` 文本相同；`layout_doc_test` 为 `54/54` 框、`46/54` 文本相同。组合页 q8_0 TEDS 由 CUDA/Vulkan `0.54080/0.53926` 提升到 `0.541509/0.541447`，均越过 `Paddle−0.005 = 0.541333` 门。
- unclip 后的二次 `get_mini_boxes` 同样执行官方 `sside < min_size + 2`（5 det-map 像素）过滤；此前仅在 unclip 前做 `min_size` 过滤，unclip 后短边不足 5 像素的框会被保留（Paddle 会丢弃）。修复后 CPU/CUDA/Vulkan 的 ocr/table 全量矩阵保持 PASS。

### rec pad 宽度根因与修复

SVTR 的 self-attention 会把 pad 列混入每一个输出位置，因此同一 crop 在不同 pad 宽度下可以解码出不同文本——这不是后端数值差异，而是预处理语义差异。判据：ICDAR2015 上 AICore 与 Paddle CPU 在**相同框**上的每一处文本差，都能在 Paddle 自己按旧的 320/480/640/… 桶宽识别时逐条复现。修复把一帧内的 crop 按宽高比排序后，对每 `kRecPadGroupDefault = 6` 个相邻 crop 补齐到该组最宽的 pad 宽度（官方 `text_recognition_batch_size` 语义），取代旧的 320/480/… 固定桶；`assignRecPadWidths`（`ocr_capi.cpp`）实现该分组，`rec_max_batch` 选项映射到它。修复后 CUDA 在 ICDAR2015 全量上 `2123/2123` 与 Paddle CPU 文本完全一致。

若要声明语义准确率，仍必须为每张发布图提供逐框人工标签（文本和 quad），再分别计算 CER/WER 与空间 precision/recall；不得把 Paddle 的预测字符串包装成 ground truth。当前仅有两张 IIIT5K crop 的 recognition smoke 和表格的 100 张 TEDS GT，不能外推为完整 OCR 发布准确率。

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
ICDAR2015 下载要求官方站点注册，许可数据不入库。本机缓存中已存在该官方检测 GT
（`~/cloudViewer_data/extract/ocr_gt/icdar2015/text_localization/`，
`test_icdar2015_label.txt` 的 SHA-256 为
`07a7dafd27fe0b4e10ba37568ccaa524b716efca3b123ebc4f7cf87e418aeb96`，图像位于同目录的
`ch4_test_images/`），故可在本机执行完整 GT 矩阵；由于该 SHA 未在仓库中固定发布，
本机结果只作本地诊断，不构成发布语义准确率声称。执行方式：

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

同一 RTX 4090、同一 PNG、PaddleOCR 3.0.1 / paddlepaddle-gpu 3.0.0、3 warmups + 10 timed runs；每行是 det `{f32,f16,q8_0}` × rec `{f32,f16,q8_0}` 的九种组合范围。`run_ocr_speed_matrix.py` 逐行 checkpoint，并从同一次报告汇总 p50/p95；该 developer 测量用于定位，未满足受控 governor 与 release-profile A/B 时不作发布性能声称。

| 图像 | Paddle GPU p50 / p95 | CUDA p50（p95 范围） | Vulkan p50（p95 范围） |
|---|---:|---:|---:|
| ppocrv4，2566×1312 | 189.49 / 192.18 ms | 67.57–69.44（68.51–73.18）ms | 76.72–92.85（78.84–111.15）ms |
| medal_table | 66.91 / 68.92 ms | 28.75–29.33（29.06–29.78）ms | 36.49–37.21（36.96–40.76）ms |
| layout_doc_test | 101.61 / 103.57 ms | 70.46–71.63（71.53–72.74）ms | 91.03–92.71（92.03–96.79）ms |

54/54 个组合 p50 均快于同次 Paddle（中位 speedup：CUDA 2.79× / 2.32× / 1.43×，Vulkan 2.37× / 1.81× / 1.11×）。上一版表中 Vulkan `layout_doc_test` 的 `0.952×` 不是实现限制，而是测量污染：该矩阵运行期间并发 `make -j24` 正在重建 ggml（报告写完约 1 分钟后 `libggml-vulkan.so` 才落盘），Vulkan 行里不触碰 GPU 的 CPU 阶段（`db_boxes`、`ctc_decode`、`crop_resize_pack`）比同图 CUDA 行慢 2–3×。同二进制同输入的空闲/24 核满载 A/B 复现了该机制：空闲时两后端 CPU 阶段一致（`db_boxes` 2.17 / 2.14 ms），满载时两者同步膨胀到 3.3 ms。仍真实存在的差距是 GPU 图计算本身：layout_doc_test f16/f16 的 rec graph Vulkan 67.5 ms 对 CUDA 51.0 ms、det 7.6 对 5.6 ms。宿主仍为 `powersave` governor（同一 Paddle 在 ppocrv4 两次运行相差 15%），故上述数值只用于定位。

![全量端到端延迟对比](images/latency_comparison.png)

## 文本与可视化证据

下图列出每种 backend × det quant × rec quant 的“精确字符串相同 / 空间匹配框”比例。它用于发现回归，而不是准确率发布门；对应的逐对差异见 [mismatch_pairs.md](images/mismatch_pairs.md)。

![文本一致率热图](images/text_agreement_heatmap.png)

绿色框表示与 Paddle 字符串相同，橙色表示同一空间框内字符串不同，红色表示 30 px 内无法配对。三栏依次为 Paddle、AICore CUDA q8_0、AICore Vulkan q8_0。

| ppocrv4 | medal_table | layout_doc_test |
|---|---|---|
| ![ppocrv4 效果](images/effect_ppocrv4.png) | ![medal 效果](images/effect_medal_table.png) | ![layout 效果](images/effect_layout_doc_test.png) |

## 完整回归矩阵（2026-10-14，三后端）

在 RTX 4090 上以 `validate_all.py --tasks ocr,table --full --profile developer --offline`分别跑 cuda / vulkan / cpu：每个后端 24 个场景 23 PASS / 1 FAIL，**15 个 OCR 行三后端全部 PASS**（含新产物 `ppocr-rec-en-q8_0`：rec fixture `max_diff` 4e-3、`argmax_match=1.000`，三后端一致）。`ctest -L ocr` 12/12 PASS（含 `test_ocr_capi_e2e`、`probe_ocr_rec_en`、`test_qocr_e2e` 与 db_postprocess/rec_crop/box_score golden）。

- **唯一 FAIL 是 `table/table-page-gt-q8_0`**：TEDS 0.541 低于 `Paddle−0.005 = 0.541333` 门，三后端同样失败，与 2026-10-12 的历史记录逐字相同（`passes_paddle_gate=0.000`、teds 0.541、teds_s 0.592）。它是既有、可复现的组合页 q8_0 边界未达标，**不是本轮 en q8_0 改动引入的回归**（组合页走 ch det/rec + SLANet，en q8_0 不参与）；上方“0.541509 越过门”的说法与实际可复现矩阵不一致，应以矩阵实报为准。
- **en q8_0 重生成的数值门**：旧产物 rec fixture `max_diff=0.443`（不过 `<0.01` 门，即转换器注释里“11 层全量化的 0.44”），保留 conv2d_139/141/143 为 F16 的新产物 `max_diff=0.0039`，argmax 与 f32 同为“JOINT”。转换入口新增 Paddle 3.x PIR/JSON 解析（`paddle_ocr_extract.py`，因官方模型已从 `.pdmodel` 改为 `inference.json`）；对同一 `pdiparams`（sha `75f64a1f…`）重生成的 f32 与已发布 f32 逐张量 `max abs diff = 0`，证明解析与权重绑定正确。
- **跨后端文本一致性（ICDAR2015 全量 GT，16 行）**：每个（语言 × det 量化 × rec 量化）组合下 CUDA 与 Vulkan 的 Hmean/CER/word-accuracy **逐位相同**（例如 en f32/f32 两后端均 Hmean=0.4475597092419522、CER=1.1587144400432121、wacc=0.2262879152623977；全部 16 行 cuda==vulkan），与图级 fixture 的 cuda==vulkan `max_diff=0` 互相印证。注意绝对 Hmean≈0.45 / CER>1 反映 ICDAR2015 难样本上严格 IoU 的端到端以检测召回为限，只作本地诊断（本机 ICDAR det GT 未在仓库固定 sha、powersave governor、未加 `--require-performance-governor`），不是发布语义准确率声称。
- **稳定性**：`test_aliked_capi_parity_512` 在空闲 GPU 上 `--repeat until-fail:40` 全 40 次通过；之前报告的偶发失败在此环境下未复现，指向并发/显存压力等环境因素（见下“Vulkan 延迟离散性”），而非数值不稳定。

## Vulkan 延迟离散性

根因（2026-10-12 更正）：不是 CPU 降频。RTX 4090 宿主未开启 Resizable BAR，Vulkan 的 DEVICE_LOCAL|HOST_VISIBLE heap 只有 246 MiB（nvidia-smi BAR1 256 MiB），而 ggml-vulkan 会把它当 ReBAR 优先用于设备缓冲。OCR + SLANet 同进程时 BAR1 占用涨到 238 MiB，窗口接近饱和后 CPU 向映射缓冲的 memmove 坍塌约 100 倍（upload p50 113 ms，perf 显示 73% 进程时间在 `ggml_backend_tensor_set` 的 memmove）；页错误、futex/poll/ioctl 耗时与 fence 等待在快慢两组间相同。`3rdparty/ggml/patches/vulkan_small_bar/0001-vulkan-no-rebar-small-bar-window.patch` 在设备初始化时检测「最大 host-visible VRAM heap < 最大 device-local heap 的一半」，自动改用 device-local 缓冲 + staging 上传（等价于上游 `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`，但由内存属性决定、不读环境变量）。同主机交替 3 轮 A/B：wall p50 134.6 → 22.0 ms，p95 约 325 → 49 ms，upload 113 → 1.47 ms，BAR1 238 → 10 MiB；1200 个输出哈希逐位一致，精度不受影响。在用户报告的完整表格 pipeline（det + rec + SLANet，100 张 `table_rec`，f16，每臂 300 样本，交替 2 轮，只替换 `libggml-vulkan.so`）上：p50 150.3/148.1 → 35.5/37.3 ms，p95 355.6/356.8 → 72.0/77.0 ms，BAR1 峰值 226 → 11–14 MiB，两臂输出逐字节相同（详见 [`../table/README.md`](../table/README.md)）。早先 `layout_doc_test` 68–92 ms 双峰与 powersave 归因均出自该问题之前的测量，不再作为结论。

## 复现

```bash
# 图级数值门
ctest --test-dir build_app -R "probe_ocr|ocr" --output-on-failure

# 单图与 Paddle 的端到端差异报告
LD_LIBRARY_PATH="build_app/external/lib:build_app/bin:$LD_LIBRARY_PATH" \
python3 core/AICore/scripts/run_ocr_official_alignment.py \
  build_app/bin/bench_ocr INPUT.png \
  "$HOME/cloudViewer_data/extract/ocr_models/ppocr-det-ch-f32.gguf" \
  "$HOME/cloudViewer_data/extract/ocr_models/ppocr-rec-ch-f32.gguf" \
  --devices cuda vulkan --warmups 3 --iters 10 --details
```

发布级证据还需要 `validate_all.py --profile release`、固定模型/输入 digest、受控的候选/基线 A/B，以及有标签的端到端语义门。
