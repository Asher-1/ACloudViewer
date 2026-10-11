# AICore table：表格结构识别与表格还原

本目录实现 AICore `table` 任务：SLANet_plus 表格结构模型（HTML 结构 token + 每个
cell 的四点多边形）以及把它与调用方 OCR 结果组合成带 rowspan/colspan 的表格。公开
ABI 为 [`aicore/table_capi.h`](../../../include/aicore/table_capi.h)；Qt 视图、导出与批处理属于
[qOCR](../../../../../plugins/core/Standard/qOCR/README.md)。

## 为什么是独立任务

det/rec/layout 只输出文本框、文本与区域类别，信息量里没有“哪几个框属于同一个
cell”“cell 跨几行几列”。这两个事实只能来自表格结构模型，因此它是新的推理合约，
而不是 OCR 的后处理参数。组合关系通过公开 OCR 访问器完成（`aicore_ocr_result` +
`aicore_ocr_ctx` 的 layout 类别名），两个任务之间没有私有耦合。

```text
caller-owned aicore_image_view (+ 可选 aicore_ocr_result: 文本框 + layout 区域)
  -> 表格区域：layout "table" 类区域，或整图
  -> 每个区域：ResizeByLong(488) + 归一化 + pad 488 -> SLANet 编码器
  -> 自回归结构解码器：每步 softmax(token) + sigmoid(8 维 cell 多边形)
  -> token -> 网格（td 计数、rowspan/colspan 占位）
  -> OCR 框分配到 cell（官方 TableMatch：IoU 优先、L1 距离次之）
  -> typed tables/cells + HTML + RFC 4180 CSV，归 aicore_table_result 所有
```

- `table_runtime.*`：GGUF 加载、编码器/解码器 ggml 图、会话缓存与设备 lease。
- `table_postprocess.*`：纯函数（无 ggml）：token 解码、网格与 span、OCR 匹配、HTML、CSV；
  由 `test_table_postprocess` 直接编译测试。
- `table_capi.cpp` / `table_catalog.cpp`：C ABI、`struct_size`/`abi_version` 校验、错误、
  计时、目录（`slanet-plus-table-{f32,f16,q8_0}.gguf`，与 OCR 共享 `ocr_models` 缓存目录）。

## 模型来源与转换

- 上游：PaddleX 官方 `SLANet_plus` 推理包（`SLANet_plus_infer.tar`，SHA-256
  `1c6ac146c894a37455dccc4a0ff1e3dd9a9ebef6ef8031340856619e6dcce971`；其中
  `inference.pdiparams` 为 `012986b0e2bfe90410618bfb4b7cf4fcb6c978caefd84c63c2934f8387b09ab8`，
  即 GGUF `ocr.source.sha256` 记录的值，与 PaddleX 自动下载的 `official_models/SLANet_plus` 相同）。
- 转换：`core/AICore/tools/convert_paddle_ocr_gguf.py <model_dir> <out> --role table
  --arch slanet-plus --source-name SLANet_plus_infer.tar --graph-json <export_ocr_graph.py 输出>`
  （同时写出 `-f16`/`-q8_0` 兄弟文件）。
  结构词表、`max_text_length + 1` 步数上限、`loc_reg_num`、`max_len`、`pad_size` 直接从
  `inference.yml` 写入 `ocr.table.*` 元数据，运行时不另存手抄常量。
- 发布：[PP-OCR release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/PP-OCR)，
  digest 固定在 `aicore/asset_digests.h`，并由 `test_catalog_dump_urls --json` /
  `test_catalog_remote_assets` 覆盖。

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| slanet-plus-table-f32.gguf | 7 875 968 | `c93a3a24e39c30bd67294a8abb543e3e0ed7ef1f2c6d67505349dee77aa87bf4` |
| slanet-plus-table-f16.gguf | 4 047 968 | `0eb25bd9a67d9eafc830df5400dd4b8bef8a5a506c067ce384f925a3a3abdf63` |
| slanet-plus-table-q8_0.gguf | 2 500 544 | `7f8b52cd9852aea71774094914b53b51368ca24d1449ff3355d354244415dc25` |

## 两类门：图级数值门 ≠ 语义真值

**图级数值门**（`table-structure-{f32,f16,q8_0}`，`test_table_runtime`）：三张官方表格
crop 由 `core/AICore/tools/export_table_fixture.py` 以 PaddleX 预处理 + Paddle 推理导出
（`table_<k>_{input,probs,loc}.npy`，发布于 `ocr-fixtures` release）。门要求结构 token
序列与 Paddle **完全相同**（`token_match == 1`），每步 softmax `max_diff` 与 cell 多边形
`loc_max_diff` 分别 < 0.01/0.01/0.05（f32/f16/q8_0）与 < 0.005/0.01/0.05。实测
cpu/cuda/vulkan 9 组全部 `token_match = 1`；`max_diff` f32 ≈ 0.0022、f16 ≈ 0.0014、
q8_0 ≈ 0.019；`loc` f32 ≤ 0.0011、f16 ≤ 0.0024、q8_0 ≤ 0.0196。

**语义门**（标签真值，TEDS / TEDS-S，PubTabNet 指标）：`core/AICore/scripts/table_teds.py`
复现 PubTabNet `metric.py` 的代价模型（Zhang-Shasha 与 APTED 同为精确最优编辑距离）。
GT 为 PaddleX `table_rec_dataset_examples` 的 `val.txt`（100 张标注表格，SHA-256
`10e200ec714af4041f883cd5f07cb00cabd9c68d2c2eb93d6c6868ec3f3500ea`），打包在
`ppocr_testdata.zip` 的 `table_rec/` 下。参照是同一上游模型的官方 PaddleX
`table_recognition` 流水线（`run_paddle_table_baseline.py`，det `limit_type=max`/960），
每行必须满足 `TEDS ≥ Paddle − 0.005` 且 `TEDS-S ≥ Paddle − 0.002`。

| 场景 | Paddle | AICore（det/rec/layout/table 同一量化档，cpu/cuda/vulkan） |
|---|---|---|
| 裁剪表 `table-gt-*` | TEDS 0.96388 / TEDS-S 0.99592 | TEDS 0.96331–0.96512，TEDS-S 0.99592–0.99599，结构完全一致 94/100；全部通过 |
| 组合页 `table-page-gt-*` | TEDS 0.54271 / TEDS-S 0.59263 / recall 0.64 | TEDS 0.54081–0.54592，TEDS-S 0.59205–0.59419，recall 0.64；全部通过 |

组合页由 `make_table_layout_pages.py` 确定性生成：每页按原分辨率嵌入一张 GT 表格，
周围是标题、段落、图注等 CDLA 类别文本块，并记录粘贴框；评测选取与粘贴框 IoU 最大
（≥ 0.5）的预测表格，否则记为漏检。组合页 TEDS 偏低的原因是上游：CDLA layout 模型
把一部分带合成边框的表格判为 `figure`（Paddle 官方链路 recall 同为 0.64），AICore 与
官方链路逐页一致；这是模型行为，不是移植缺陷，也不应通过降低 layout 阈值“修好”。

layout 区域本身另有区域级门 `ocr-layout-regions-*`（`check_layout_regions.py` 对照
`run_paddle_layout_reference.py` 固定的官方 PP-Structure 输出）：f32/f16 F1 = 1，
q8_0 recall = 1、F1 0.964（门限 0.95），mean IoU ≥ 0.99。

## 复现

```bash
# 契约、后处理、e2e 与图级门
ctest --test-dir build_app -R "table|probe_table|probe_ocr_layout" --output-on-failure

# 完整矩阵（含 TEDS 门），每个 backend 各跑一次
python3 core/AICore/scripts/validate_all.py --build build_app \
  --backend cuda --tasks ocr,table --full --output /tmp/table_cuda.json

# 单独的 GT 矩阵
python3 core/AICore/scripts/run_table_gt_matrix.py \
  --ground-truth "$HOME/cloudViewer_data/extract/ppocr_testdata/table_rec/val.txt" \
  --ground-truth-sha256 10e200ec714af4041f883cd5f07cb00cabd9c68d2c2eb93d6c6868ec3f3500ea \
  --dataset-revision paddlex-table_rec_dataset_examples-val \
  --image-root "$HOME/cloudViewer_data/extract/ppocr_testdata/table_rec" \
  --bench build_app/bin/bench_table --models-dir "$HOME/cloudViewer_data/extract/ocr_models" \
  --quants f16 --ocr-quant f16 --backends cuda vulkan \
  --paddle-teds 0.9638835 --paddle-teds-s 0.9959180 --output /tmp/table_gt
```

当前 `validate_all --tasks ocr,table --full` 在 cuda、vulkan、cpu 上均为 24/24 PASS
（developer profile，2 次进程重复）。

## fixture 人工核验

`core/AICore/tools/inspect_table_fixture.py` 把已发布的 `table_<k>_probs.npy` 用 Paddle
推理包自带的结构词表解码成 HTML，与 `val.txt` 标注计算 TEDS-S，并把 `loc` 映射回原图
（`TableLabelDecode._get_bbox_scales` 的缩放）画出 cell 多边形叠加图与 `report.html`；
给定 `--bench` 时同时列出 AICore 的结构。用原始 `SLANet_plus_infer.tar` 重新执行
`export_table_fixture.py` 得到的 9 个 npy 与 `table_fixtures.json` 与发布文件逐字节相同；三张
crop 的 Paddle 结构与标注完全一致（TEDS-S = 1：25 步/9 cell、78 步/38 cell、185 步/136 cell）。

```bash
python3 core/AICore/tools/inspect_table_fixture.py \
  --fixtures ~/cloudViewer_data/extract/ocr_fixtures \
  --images ~/cloudViewer_data/extract/ppocr_testdata/table_rec/images \
  --model-dir <解压后的 SLANet_plus_infer> \
  --gt ~/cloudViewer_data/extract/ppocr_testdata/table_rec/val.txt \
  --bench build_app/bin/bench_table --table <slanet-plus-table-f32.gguf> --device cuda \
  --out /tmp/fixture_check/report
```

## 与 Paddle GPU 的速度 A/B

`core/AICore/scripts/run_table_speed_ab.py`：两侧都从已解码像素开始计时（Paddle 传
ndarray，`bench_table` 计时前解码），warmup 3，3 遍 × 100 张 `table_rec` 图，同主机同输入。
`structure` = 仅 SLANet；`pipeline` = det + rec + SLANet（det `limit_type=max`/960，三阶段
同一量化档）。Paddle 为 paddlepaddle-gpu 官方 `SLANet_plus` / `table_recognition`。
一致性列为 AICore 与 Paddle 逐图结构比较。RTX 4090，CPU governor `powersave`。

| 口径 | Paddle GPU p50 / p95 (ms) | cuda f32 / f16 / q8_0 p50 | vulkan f32 / f16 / q8_0 p50 |
|---|---|---|---|
| structure | 29.98 / 61.29 | 7.97 / 7.04 / 7.06（3.8–4.3×） | 10.97 / 12.06 / 12.07（2.5–2.7×） |
| pipeline | 174.33 / 634.8 | 30.94 / 30.91 / 35.07（5.0–5.6×） | 65.36 / 53.97 / 68.22（2.6–3.2×） |

- 与 Paddle 的结构一致性：f32、f16 逐图 100% 相同；q8_0 96% 相同，TEDS-S 0.9968
  （q8_0 的标签 TEDS 门见上节，已通过）。
- 第二轮（测量期间另有外部 GPU 进程）结论方向一致：structure cuda 7.3–8.0 ms、vulkan
  12.4–14.1 ms；pipeline cuda 30.6–31.9 ms，Paddle 205 ms。
- cuda 的 p95 全部低于 Paddle。vulkan pipeline 的 p95 为 120–354 ms，低于 Paddle pipeline
  p95（575–635 ms），但尾部明显：拆分后来自 OCR det/rec 阶段（table 阶段 p95 27–33 ms），
  与文本框数相关性低（r = 0.16，cuda 为 0.66），关闭 coopmat2 更差（p95 307 ms）。这与
  powersave 主机上 Vulkan fence 等待期间 CPU 降频的已知现象一致；vulkan 尾部数字须在
  `performance` governor、GPU 空闲时复测后才能作为发布证据。

```bash
D=$HOME/cloudViewer_data/extract/ppocr_testdata/table_rec
python3 -c "import json;[print(json.loads(l)['filename']) for l in open('$D/val.txt')]" \
  > /tmp/table_ab_list.txt
python3 core/AICore/scripts/run_table_speed_ab.py --images "$D" --list /tmp/table_ab_list.txt \
  --bench build_app/bin/bench_table --models-dir "$HOME/cloudViewer_data/extract/ocr_models" \
  --backends cuda vulkan --quants f32 f16 q8_0 --passes 3 --output /tmp/table_ab
```
