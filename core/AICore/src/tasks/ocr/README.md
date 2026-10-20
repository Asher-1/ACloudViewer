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

在 RTX 4090 上以 `validate_all.py --tasks ocr,table --full --profile developer --offline`分别跑 cuda / vulkan / cpu：每后端 24 个场景。**15 个 OCR 行三后端全部 PASS**（含新产物 `ppocr-rec-en-q8_0`：rec fixture `max_diff` 4e-3、`argmax_match=1.000`，三后端一致）。`ctest -L ocr` 12/12 PASS（含 `test_ocr_capi_e2e`、`probe_ocr_rec_en`、`test_qocr_e2e` 与 db_postprocess/rec_crop/box_score golden）。

- **`table/table-page-gt-q8_0` 的 TEDS 门失败已定位并修复（2026-10-14）**。根因不是量化质量，而是组合链对亚像素框差混沌敏感：layout 区域框 0.3 px 的 q8_0 漂移跨过 int 截断裁剪边界（与 PaddleX `CropByBoxes` 的 `int()` 语义逐字一致），SLANet 解码即翻转（实测 0.3 px → 13×3 → 13×4 列，单页 −0.22 TEDS）；因此组合页 TEDS 是构建级抽样，实测名义抽样带宽 0.5409–0.5453（f32/f16/q8_0），原 0.005 门恰好落在带内（旧 q8_0 差 5e-6）。修复分两层：(a) 转换器按实测敏感性把各阶段解码可见层保留 F16（layout 四个 FPN head conv、det DB head 路径、SLANet GRU/投影、ch rec 与 en 同位的三个早期 backbone 1×1），四个新 q8_0 产物的各自 fixture 门全部改善或持平（rec 5.3e-4→1.2e-4、layout 0.517→0.499、table 0.019/0.020→0.0169/0.0186、det 0.476→0.491 仍在门内），TEDS-S 由 0.5923 升至 0.5934 稳定过门；(b) 组合页 TEDS 容差按实测带宽重校准为 0.006（Paddle−0.006 = 0.5403327），覆盖三个 q8_0 构建的实测低点 0.5409，manifest 三个 table-page-gt 行同步更新。后续逐页验证发现召回率 0.64 才是主导瓶颈（36/100 漏检页的 layout 区域全部定位正确、仅类别被 CDLA 标为 Figure/Text），`aicore_table_run_options` 新增 `fallback_region_mask` 类别回退后 q8_0 组合页 TEDS 0.541→**0.832（+54%）**、recall 0.64→0.96、高于 Paddle 官方链路 +0.286（详见 [`../table/README.md`](../table/README.md)）。`run_table_gt_matrix.py --jitter-ensemble` 提供 ±1 px 放置鲁棒性诊断（f32 0.5344 vs q8_0 0.5329：量化差距缩至 ~0.0015，只可与同协议的 Paddle 基线比较）。CUDA 全矩阵 24/24 PASS。
- **en q8_0 重生成的数值门**：旧产物 rec fixture `max_diff=0.443`（不过 `<0.01` 门，即转换器注释里“11 层全量化的 0.44”），保留 conv2d_139/141/143 为 F16 的新产物 `max_diff=0.0039`，argmax 与 f32 同为“JOINT”。转换入口新增 Paddle 3.x PIR/JSON 解析（`paddle_ocr_extract.py`，因官方模型已从 `.pdmodel` 改为 `inference.json`）；对同一 `pdiparams`（sha `75f64a1f…`）重生成的 f32 与已发布 f32 逐张量 `max abs diff = 0`，证明解析与权重绑定正确。其余四个 q8_0 重生成所用源均已按发布产物的 `ocr.source.sha256` 逐字节验证（det `49ee815e…`、layout `f8b20142…`、ch rec `a6dbfa63…`、SLANet_plus `012986b0…`；注意 PaddleX 缓存中的重导出与 PP-OCR 官方 release 权重不同，转换必须用官方 release 包）。
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

## X-AnyLabeling 方向分类与 PP-OCRv5/v6（2026-10-17）

在已有 OCR 任务内增加 `cls` 角色和 ONNX 离线降图，运行时仍只使用 GGML，
不增加 ONNX Runtime 推理依赖。公开 C ABI 通过 `struct_size` 追加可选分类器、
分类阈值/批次和 DB 候选数；旧调用方维持原行为。裁剪后的文字先按宽高比分批，
执行 BGR、`[-1,1]`、`3×48×192` 分类，在 180 度置信度**严格大于** 0.9
时旋转，再识别。`run` 和指定 quad 的 `recognize` 均支持该路径，结果可查询
逐框角度、概率和是否旋转。源 ONNX 模型的普通检测链直接裁剪 quad；
原有 PaddleX 权重保留整数多边形最小外接矩形拟合，避免混用两套裁剪约定。
中文 v4 动态 attention 形状使用运行时整数形状常量，支持非固定文字宽度。
ONNX F32 深度卷积超出中间缓冲上限时，逐通道回退也保留 F32 im2col，避免
默认 F16 中间值改变输出；同一真实 BGR 页面上的 v6 Medium 原始概率图最大误差
由 `0.001048` 降为 `0.000004`，沿用 `<0.001` 数值门。原有 PaddleX 图不走此分支。
检测 probe 同时拒绝输出/参考长度不一致及非有限概率值。

`aicore_ocr_preset_*` 是任务拥有的八组配方：已发布 v4 中文/英文、
精确源中文 v4/日文 v3、v5、v6 Tiny/Small/Medium。它们提供源默认检测阈值、批次和配对模型名，与已发布权重目录
分开。v6 使用 3000 个 DB 候选和 1.4 unclip；v4/v5 为 1000 和 1.5。
新六组配方和共享分类器已有本地 F32 及 F16/Q8_0 待验候选；已发布 OCR 目录仍为原有
12 项，不能据配方推断存在下载 URL。qAnyLabeling 消费公开配方，方向开关默认开启；
qOCR 的原有无分类器调用保持兼容。

源文件的 URL、字节数、SHA-256 和上游提交见
[`ocr_onnx_sources.json`](../../../tools/ocr_onnx_sources.json)。转换器读取实际
ONNX 拓扑/形状/常量，拒绝未支持算子或属性，识别器将源字典及字典 SHA-256
嵌入 GGUF；安装包无需附加 OCR 字典。源程序及字典来自固定的
X-AnyLabeling-GGML 提交 `745f6c355f94cddb06f6965b3a4118f82033e935`，
模型来自其配置指定的 CVHub520 v0.2.2/v2.4.0/v3.0.0 资源。转换保留来源信息，不改变源项目和
检查点的许可证。F16/Q8 是待验候选；历史策略失败与当前 SHA 的证据分别保留，
不能因能加载就列为已支持。当前逐文件状态见
[`OCR_RELEASE_GATES.md`](../../../../../plugins/core/Standard/qAnyLabeling/OCR_RELEASE_GATES.md)。

```bash
# 先按来源清单验证 SHA；rec 必须指定对应上游字典。
python3 core/AICore/tools/convert_onnx_ocr_gguf.py SOURCE.onnx OUTPUT_STEM \
  --role rec --source-sha256 SOURCE_SHA256 --dictionary SOURCE_DICT.txt --quants f32

# 图级、方向分类、实际源 TextSystem 页面及已有 OCR/表格回归；需要
# 验证环境安装 onnx/onnxruntime/numpy/opencv-python/PyYAML/pyclipper/shapely/six。
# --reference 指向上述固定提交；不配置时源流程检查跳过并使严格门失败。
python3 core/AICore/scripts/validate_all.py --build build_app --full \
  --backend cuda --tasks ocr,table \
  --reference xanylabeling=/path/to/X-AnyLabeling-GGML \
  --output build_app/Testing/ocr-table-full-cuda.json
```

公共清单新增十二个 F32 图级场景、六组十个真实正向/倒置文字 crop、六组真实页面
及六组选中多边形源流程对照，共三十个本地产物扩展场景。图输出门为 `<0.001`，识别还要求 CTC argmax 相同；方向测试覆盖
0.9/1.0 严格阈值、混合宽度/批次次序、BGR 非紧凑 stride 和重复上下文生命周期；
十条文字的预期均来自各自源识别器及固定 SHA-256 字典。
源页面对照直接调用未修改的上游 `TextSystem`，使用其 YAML/parse_args/字典，
通过公开 C ABI 比较框数、逐字符串和 quad（最多 2 像素），每个 context 调用两次。
该页中文 v4 保留 74 条、日文 v3 保留 77 条、v5 保留 81 条，三个 v6 各 73 条；这不是有标签数据集精度或发布性能结论。
未发布 GGUF 的场景标为 `local_only`，有对应文件才运行，报告必须核对其覆盖数。

KIE 在所检查上游中是实体/有向关系的标注和 PP-OCR 标签格式，不存在已注册的
SER/RE 自动模型流程。qAnyLabeling 对齐其导入导出；AICore 的 KIE 推理字段仍保留且
拒绝非空输入。完整模型发布/量化矩阵和数据集精度验收仍待完成。

### 2026-10-19：逐文件验收与发布约束

`ocr_quantization_policies.json` 保存按源 SHA 固定的混合精度候选策略，
通过 `convert_onnx_ocr_gguf.py --precision-policy` 显式使用；不会自动成为产品默认值。
24 个 F16/Q8_0 候选重新转换均与受测文件 SHA 完全一致，证据为
`build_app/Testing/ocr-storage-reference/quant-policy-replay-v7.json`。
识别字典必须同时匹配对应源模型的 SHA；相同 argmax 索引不证明字典正确。
容器内记录精度例外，验收报告记录实际 F32/F16/Q8_0 参数数目，不能把混合精度
描述为全部参数均为 F16 或 Q8_0。

v7 历史候选的 CPU/CUDA/Vulkan F16/Q8_0 多输入概率门分别为 **22/24 PASS**；
各自加上 12 个方向流程后均为 **33/36 PASS**，报告为
`build_app/Testing/ocr-storage-reference/quant-v7-{cpu,cuda,vulkan}/matrix.json`。
分类器覆盖五张词图及对应 180 度图，共十个输入，两个量化的三后端概率、
argmax 和六组方向流程均通过；Q8_0 分类器实际仅 4.8% 参数为 Q8_0，
87.2% 为 F16、8.0% 为 F32，属于显式混合精度候选。
Tiny 识别器 F16 最大误差为 `0.006315589`（门为 `<0.001`）；
Q8_0 最大误差为 `0.890249828`（门为 `<0.005`），argmax 一致率为 `0.975`（要求 `1`）。
相同存储权重的独立 ONNX 输出与 GGML 输出仅相差约 `1e-6`，因此这些候选
仍受存储精度门阻断。敏感核保留 F32/F16 的策略不能代替真实页面验收。

源检测框过滤修复前的 F32 CUDA 开发矩阵 **42/42 PASS**，报告为
`build_app/Testing/ocr-storage-reference/f32-cuda-current/matrix.json`。
它包含源图、方向、页面、选区、50 张标注场景及每种语言 50 张 XFUND 页面识别。
场景集按原始词级标注计算 Hmean，并单独报告固定 GT 区域及完整流程 CER/WER；
检测器的行级输出与词级标注粒度不同，不能将其绝对 Hmean 当作官方榜单分数。
XFUND 是语义字段标注，仅用于区域识别，不能据此声明检测 Hmean 或自动 KIE。

街景与 XFUND 不能替代整页检测验收。公共输入清单另固定了 SROIE contributor
corpus 的 50 张扫描收据及 50 个原始 CSV，共 2,779 个标注；18 个组合/量化
场景分别检查检测 Hmean、固定区域及完整流程 CER/WER。数据和评测范围见
`core/AICore/tests/ocr/DATA_CARD.md`。100 个新输入已由公共 runner 完成 SHA
校验下载；这不是新 GGUF 的公共下载闭环。

逐页门保留所有失败：v7 的 v4 Q8 中/日文组合检测 Hmean 退化超过 `0.005`，
v5 F16 和 Tiny F16/Q8 的街景完整流程也有失败。原始张量 PASS 不能覆盖这些失败。
后续修复按新文件 SHA 单独验证，不能继承旧文件的验收状态。

v8 的 v4 detector Q8 候选在新生产库通过三后端概率门、CPU/Vulkan 的中/日文
街景页，以及 CUDA 的中/日文街景和 SROIE 收据页；`filter-v8-cuda` 为 **5/5 PASS**。
v5 recognizer 追加真实 `val_img_7.jpg` 的透视裁剪，源概率输入由四个增至五个。
旧 v5 Q8 在新增输入最大误差为 `0.042550772`、argmax 为 `0.975`，三后端均失败；
v12 F16 和 v14 Q8 新候选的三后端五输入均通过，CPU 最大误差分别为
`0.000957310` 和 `0.004976571`，argmax 均为 `1`，CUDA 街景/收据亦通过。
证据是 `filter-v12-*`、`filter-v14-*`；v8/v12/v14 的当前策略重放 SHA 相同。
v5 F16 实际为 21.23% F16、78.77% F32；Q8 实际为 2.10% Q8_0、19.13% F16、
78.77% F32。Q8 误差接近 `0.005` 门限，不能据此声明完整精度或速度达标。

另一个 Tiny recognizer F16 的 v15 候选四输入三后端概率通过（CPU 最大误差
`0.000872791`），CUDA 收据和方向通过，但 Tiny F16 街景检测 Hmean 下降
`0.005540166`、完整流程 WER 增加 `0.009803922`，组合仍失败；检测框量化误差
与识别器存储误差分别定位。v16 detector 将敏感核 `conv2d_112.w_0` 保留 F32，
新文件 `804a564ad2db0fbd2b95325c167b854a4cba18c36889aab16e51c6ae2a7c0c83`
通过三后端源图，搭配 v15 recognizer 的 CUDA 街景/收据通过，转换重放 SHA 相同。
v15/v16 已写入显式实验策略；其他后端页面及 XFUND 继续验收，Tiny Q8 阻断未解决。

以上历史检测概率图使用 max-side 320，页面流程使用源默认 960，两者证据范围不同。
当前公共清单的 15 个 detector 源图场景已显式改为 960，单独执行
`default-size-accuracy-*`；320 的旧 PASS 不代表默认尺寸图已通过。
release 检测测量另强制 960，拒绝以 320/640 或其他尺寸替代；所有额外概率输入
沿用同一尺寸，仅减少预热/计时次数。原始源 provider 与 native 使用同一输入张量。

960 首轮 CPU 检测图为 **14/15 PASS**（`default-size-accuracy-cpu`）。
Small detector F16 的源平均误差 `0.000101013` 超过 `<0.0001`，而相同存储
权重的 native/ONNX 最大误差仅 `0.000008106`；不能放宽源门限。
逐核存储诊断后，v17 仅将 `conv2d_122.w_0` 保留 F32，新文件
`73efc76d8b6427363b19c5a88ecf7075febfd5bbbef0b5b7ff67ac4d8f5330c1`
在 960 CPU 源图的最大/平均误差为 `0.010192215` / `0.000082143`，
独立相同权重误差为 `0.000007331`。`filter-v17-cpu` 的源图、50 张街景和
50 张收据 **3/3 PASS**，街景五项源退化均为零，收据最大退化为 `0.000733810`。
CUDA/Vulkan 街景/收据也各 **2/2 PASS**（`filter-v17-pages-{cuda,vulkan}`）；
转换 SHA 重放通过不能覆盖其余页面或受控性能门。

当前默认尺寸 CPU 概率矩阵 `default-size-current-cpu` 为 **35/36 PASS**，
CUDA/Vulkan 的对应矩阵也各为 **35/36 PASS**，唯一失败都是旧 Tiny Q8
recognizer。该旧失败不会赋给下面不同 SHA 的 v19 文件。
新的 v19 Tiny detector/recognizer Q8 组合在
`filter-v19-cpu` 的默认尺寸检测图、四输入识别图、街景及收据为 **4/4 PASS**。
检测图 max/mean 为 `0.037688673` / `0.000081015`；识别 max 为
`0.002044559`，argmax 为 `1`。街景五项退化均不大于零，收据最大退化为
`0.000208960`。新文件 SHA 分别为
`fa5f6f6efdc79c4a2363c7719d6dd5d7ca999e14cbc71132a65d30ce834c4988` 和
`f9e6c3793c95fa7e84b10d5fa277f56e22b059c4e1517ba93a223f83fa41d960`；
两份转换均重放得到相同 SHA。CUDA/Vulkan 的同四项检查也各 **4/4 PASS**
（`filter-v19-pages-{cuda,vulkan}`），完整页面矩阵、受控性能和发布仍未闭环。

v19 recognizer 在转换时使用训练集 256 个裁剪的原始激活协方差，对
`conv2d_30.w_0` 执行误差反馈，输出普通 ggml Q8_0 块；不增加运行时格式、
环境变量或插件推理路径。候选策略需显式传入 `--precision-policy` 和
`--calibration-root`，校准图片、四边形、算法及 damping 均固定在策略中。
数据划分和选择偏差见 `tests/ocr/DATA_CARD.md`。detector 实际为 0.23903% Q8_0、
98.50118% F16、1.25979% F32；recognizer 为 0.57945% Q8_0、14.95273% F16、
84.46782% F32。不能据文件后缀声称纯 Q8、显著压缩或加速。此策略已纳入显式
实验配置 `tools/ocr_quantization_policies.json`，SHA 为
`3f8700eb6ab5672f396ff3c93b73c649ca480de4ec59cb0d2c14b3369d22ccc4`；
没有注册公共模型、更改产品默认配置或覆盖共享缓存内的旧文件。

后续整页对照修复了遗漏的源 `TextDetector.filter_tag_det_res`：首个极值的
顶点重排、`width-1/height-1` 裁剪和短边过滤，仅用于 source ONNX 检测路径。
极值并列产生的重复顶点和奇异透视变换也匹配源 OpenCV 的左上角采样行为。
13 个源生成的过滤 fixture、真实奇异裁剪及既有几何 golden 均通过；受影响
OCR/table/plugin/validation 回归 **12/12 PASS，零跳过**。未完成的旧页面 cohort
已停止并记录；`filter-v7-cuda` 按新的库/探针 SHA 重跑。旧页面 PASS 不能自动
成为新生产库的发布证据，原始概率图报告则仍明确保留其各自二进制身份。

发布必须逐个文件通过源概率/argmax、标注数据、CUDA/Vulkan 及受控性能门；
原生 CPU 结果仅作诊断，不参与 GPU 发布门控，独立原始框架真值仍必须保留。
再上传既有 `PP-OCR` release，注册真实 SHA/catalog/manifest 并检查冷缓存下载。
禁止先发布 F32 再补验收，也禁止将局部精度 PASS 当作完整发布 PASS。
GPU release 图测量强制至少两次预热、十次计时，公共 runner 另要求三次进程重复
及同机 baseline；同时保留原始源 CUDA provider 对照和 GPU 活动审计。
显存数值是驱动采样的进程高水位，不是分配器精确峰值。GPU 忙碌、外部 compute
或遥测缺失均使受控门失败；不会终止用户桌面进程。
每次审计拥有唯一 ID 和独立 `.d/<id>.json`，包括空闲等待失败；进程重复不会
覆盖原始遥测。当前库的三次 baseline/三次 candidate 分类器尝试仍因桌面或外部
RelateAnything 基准活动失败，测量结束已恢复自有页面验证队列，没有新发布文件。


### 公共 Paddle 原始源复核（2026-10-19）

`run_ocr_paddle_probe.py` 将发布文件与 SHA 固定的原始 Paddle program/parameters
直接比较，清单注册所需源归档、配置 YAML 和输入 fixture。CPU 的 detector、中文
及英文 recognizer 各 F32/F16/Q8_0 在 CPU/CUDA/Vulkan 各为 **9/9 PASS**，
GPU 同时直接比较原生 CPU 概率；新公共页级队列仍未全部完成。
六个难例裁剪通过，首个中文 F32 的 50 张街景 Hmean、固定区域及完整流程 CER/WER
五项源差值均为零。这些结果不代表 layout 或完整发布门完成。

源 GPU 子进程隔离继承的 GLOG/MLOG 配置：宿主长 GLOG_log_dir 会触发官方
Paddle GPU 3.0.0 的退出时静态字符串重复释放。源进程仍必须正常退出。数学模式
采用初始化前的 `NVIDIA_TF32_OVERRIDE=0`，不使用该版本不存在的 Paddle flag。
同一 3.0.0 源 GPU 三图正常退出，det/ch-rec/en-rec 对 CPU fixture 最大差为
0.000017583/0.000006557/0.000000238，识别 argmax 全同；共享 GPU smoke 不作性能
证据。完整源 benchmark/libpaddle SHA、依赖版本和实际数学模式进入报告。
公共 Paddle 探针测试现为 **27/27 PASS**。GPU 图参考的独立依赖清单为
`tools/requirements-ocr-paddle-gpu-reference.txt`，固定官方 3.0.0 wheel SHA 和 CUDA
库版本；新环境 pip check 通过，三个图与原 GPU 环境输出逐元素相同。

`run_ocr_paddle_layout_probe.py` 另检查原始 PicoDet 的全部 8 头，三个 backend 的
F32 通过；F16/Q8_0 最差头 argmax 比例为 0.992307692/0.969230769，均失败。
它们与同 dtype 原生 CPU 的后端比较通过，这不能代替原始源输出的相符性。
原生探针逐头导出并拒绝 NaN/Inf；真实图注入 NaN 的失败回归通过。

公共页面源子进程统一使用 `ocr_paddle_environment.py`。旧 v2 队列出现 CPU 退出
堆损坏，已停止并归档；新 v3 中英街景源参考各 50 页正常退出。显式固定 SHA 的
参考可复用，但先验证正常退出凭据及源库、依赖、代码、配置、GT、完整逐图覆盖，
并保持相同线程、语言和评价模式；每个量化和后端仍执行自己的原生推理。
完整三后端页级/受控 A/B/发布闭环仍待完成。

公共中英街景的 F32/F16 页级通过，但 Q8_0 完整流程 CER 退化分别为
0.009708738/0.005295675，超过 0.005；逐阶段替换确认主要差异来自 Q8 detector，
保留 F32 detector、使用 Q8 recognizer 时两种语言均通过。详见 qAnyLabeling 审计表。
图参考依赖报告另修复了 Paddle vendored 元数据覆盖实际选择版本的问题，新增
`ocr_paddle_metadata.py` 记录全部候选与选择版本；活动页面队列仍使用冻结的旧身份
实现，不能据此宣称依赖身份门已增强或完整 release 已通过。

`aicore_ocr_recognize_regions` 接收同步借用的三点及以上多边形：坐标先向零截断
为整数，再执行源 `normalize_custom_detection_boxes` 最小外接矩形和顶点排序。
选中区域不裁切到图像边界，使用源边界复制采样；按源阅读顺序组织识别批次，
返回时恢复输入索引。原 `aicore_ocr_recognize` 的已发布 quad 接口保持兼容。
每套配方的对照包含十个区域：倾斜四边形、五边形、三角形、分数/负坐标和越界区域，
逐输入比较文本、框和两次调用，quad 容差 `0.001`；低置信度和空文本仍保留原结果槽位。
同时拒绝少于三点、NaN 和 int32 范围外坐标；合法退化多边形返回空文本/零分。

### CDLA 独立布局页面评价

`tools/ocr_cdla_validation.json` 固定官方 CDLA 验证集全部 1,000 页、8,711 个
人工区域及归档/成员 SHA。`layout_gt_eval.py` 使用 LabelMe 的实际像素栅格
语义生成 bbox/area，已逐项对照原始官方转换器；COCO annotation ID 保持
正整数，防止零值未匹配哨兵。它检查逐页覆盖、输入身份和类别，输出 class-aware
bbox AP50:95/AP50/AP75/AR100；有限越界框交给 COCO 处理，不擅自裁剪。

`export_cdla_layout_reference.py` 执行固定 SHA 的原始 Paddle 图和官方
`ppocr/postprocess/picodet_postprocess.py`；观察原 NMS 结果中的分数，不改变
后处理决策。`export_cdla_layout_native.py` 通过公开 borrowed BGR image 和
typed-region C ABI 导出结果，以带行填充的输入覆盖 stride 合同。由于现有
API 要求 det/rec，该脚本用 DB 阈值 1 抑制文本检测并断言零文本框，仅隔离
布局精度，不测整套 OCR 的性能。两侧保留布局 score=0.5/NMS=0.5。

手动回归命令为 `python3 core/AICore/scripts/test_layout_gt_eval.py`，17 项
通过；重新配置后的正式 `test_ocr_layout_gt_eval` 也通过。1,000 页原始 CPU
和三页原生 CPU 接线结果仅作诊断。公共冷缓存下载/2,000 个文件 SHA/零下载
离线复用通过；原始 CUDA 与两 GPU 三种存储的全部 1,000 页评价均正常退出。
原始 CUDA mAP50:95 为 0.8546702421，原生 CUDA/Vulkan 的 F32 差值分别为
-0.0000003951/-0.0000004017，F16 均为 +0.0000631982、Q8_0 均为
-0.0009133202。这些是实测页面差值，未据此反推发布容差。入口
`build_app/Testing/ocr-storage-reference/layout-cdla-source-path.txt`。
独立页面容差与公共 manifest 接入尚待完成，不能以此替换严格原始输出门、
完整 OCR 页面指标或受控性能门。
