# OpenPCDet CUDA 性能专项 — 变更说明

记录 2026-10 CUDA 性能专项（0023/0024 ggml patch + 任务侧 CUDA 扩展 + 0025/0026
收尾）的全部改动，并逐项标注**上游移植**与**有意偏离**。上游参考实现：
[Asher-1/OpenPCDet-GGML](https://github.com/Asher-1/OpenPCDet-GGML)
`cpp_ggml/src/extensions/` 与 `cpp_ggml/patches/0001-native-sparse-gather.patch`。

## 结果概览

| 指标 | 专项前 | 专项后 | 上游同机 |
|------|--------|--------|----------|
| centerpoint CUDA e2e | 255.6ms | **83–91ms**（与桌面负载相关） | 85.8ms 实测（早期基线 79ms 在当前负载下不可复现） |
| kitti_pointpillar CUDA e2e | 95.8ms | **46–50ms** | ~40.5ms |
| centerpoint 精度 (307/307) | 0.99859 | **0.99908** | — |
| kitti 精度 (24/24) | 0.99976 | **0.99982** | — |

已知残留：与上游的 ~5ms 差距集中在 dense 图的通用 graph replay gap（∝节点数，
kitti 交叉对照证据链完整），定位需要 CUPTI/nsys 基建，本机不可用，列为后续工作。
稀疏端（23ms）与 GPU rulebook（4ms）已与上游持平。

## ggml patch 清单（均在 cuda-manifest.yaml / cuda-vulkan-manifest.yaml 登记）

| Patch | 内容 | 来源 |
|-------|------|------|
| 0021 | `opcpd_conv` F32 conv 纳入 TF32 igemm gate（0022 后仅剩 f16/q8 conv 路由） | 本仓库自有 gate 机制 |
| 0022 | 上游 TF32 conv2d kernel 链逐字节移植（winograd/TF32 mma/FMA fallback，`opcpd_exact` namespace） | **上游移植**（`0001-native-sparse-gather.patch` 的 conv2d.cu 区间） |
| 0023 | `ggml_sparse_gather` / `ggml_sparse_scatter_add(_n)` 三算子（op 枚举尾部追加，`GGML_MAX_SRC` 10→30、`GGML_MAX_OP_PARAMS` 64→256；CPU 实现 + CUDA kernel + 分派） | **上游移植**（上游同款算子语义）+ CPU 实现为本仓库补齐（上游 CPU 走原生分支） |
| 0024 | `ggml_bn_fold_relu` 融合算子（CUDA float4 快路径 + CPU 回退） | **上游移植**（bn-fold-relu.cu） |
| 0025 | per-op CUDA event 计时门（`GGML_CUDA_PERF_OP=1`，默认关，纯诊断） | **上游移植** |
| 0026 | SECONDFPN deblock 的相位分解 GEMM 路径（gather + cublasSgemm + scatter，含调试门） | **上游移植**（conv2d-transpose.cu 的 `conv2d_transpose_gemm_path`），gate 采用本仓库节点名 opt-in 先例 |

## 任务侧 CUDA 扩展（`extensions/`，`AICORE_OPENPCDET_CUDA_EXT` 编译期宏）

| 组件 | 来源 | 说明 |
|------|------|------|
| `sparse_rulebook_cuda.{h,cu}` | **上游移植** | rulebook 生成上 GPU（30→4ms） |
| `vfe_scatter_cuda.{h,cu}` | **上游移植 + 有意偏离**（见下） | pillar VFE CUDA kernel |
| `height_compression_cuda.{h,cu}` | **上游移植** | BEV 高度压缩 D2D |
| `openpcdet_cuda_ext.cu` | **本仓库自有** | 三组件合译为单一编译单元（见偏离 #4） |
| `remap_dense_indices_cuda`（内含于 ext） | **本仓库自有** | dense 图 D2D -1→零行重映射 + padding 清零（修复容量重建后未初始化索引隐患） |
| `sparse_ops` compact lowering | **上游移植 + 有意偏离**（见下） | 融合 sparse gather 图内接入 |
| 图拓扑（`voxel_backbone_runner` / `pointpillar` / `sparse_backbone` / `second`） | **本仓库自有** | 设备直通（D2D 链、`spatial_device_ready`）、容量张量、VFE/BN 融合接线 |

## 有意偏离清单（相对上游）

0. **focal_multimodal 投影矩阵求逆修复（本仓库 bug 修复，非移植也非偏离）**：
   `voxelrcnn.cpp` 的 `invert_4x4`（4×4 伴随矩阵求逆）`inv[10]` 项抄写错误
   （`-m[12]*m[2]*m[5]` 应为 `-m[12]*m[3]*m[5]`）。标定矩阵含平移时逆矩阵
   错 → `lidar_to_camera` 投影错位 → `conv_focal_multimodal` 图像融合失效 →
   focal backbone score 塌陷至 2 检（0.39/0.31）vs 上游 C++ 8 检。1 行修复后
   f32 CPU 恢复 **8/8 全匹配 mean_iou=1.000000**（上游 focal 基线为 scene
   模式），f16/q8_0 恢复 8/8 且与上游自身量化差距持平或更优。残余 f32 CUDA
   0.9978 为 ggml mul_mat 求和噪声（~2e-7）× mask_kernel 阈值敏感位点的拓扑
   漂移（CPU taps 114 层逐位一致证明 openpcdet 层无缺陷），容差定性，依据见
   [UPSTREAM_ALIGNMENT.md](UPSTREAM_ALIGNMENT.md)。

1. **VFE BEV 画布清零**：上游 `gpu_pillar_vfe_scatter` 的持久 BEV 画布从不清零，
   逐帧 pillar 布局变化时会散射到陈旧数据上（同帧重复基准下被掩盖的跨帧残影
   缺陷）。移植版在 scatter 前 `cudaMemsetAsync` 清零，并注明为有意偏离。
2. **量化稀疏权重回退 dense lowering**：q8 稀疏权重无法按 kernel 切片重排，
   量化模型自动保持 dense lowering + 设备端重映射上传（上游无此回退）。f16
   稀疏权重走 compact 路径时保持严格 fp32 GEMM 精度策略。
3. **`is_cuda()` 设备名判断修复**：原 `device.find("cuda")` 对 lease 设备串
   `CUDA0` 永远 miss，导致持久稀疏图缓存从未生效（每帧重建图）。改为
   `ggml_backend_name` 前缀判断——这是本仓库 bug 修复（恢复上游既有的图缓存
   行为），非算术偏离。
4. **三扩展合译单编译单元**：分文件编入 libAICore 时 HeightCompression/VFE
   kernel 启动持续报 `cudaErrorInvalidDeviceFunction`（多 fatbin 注册隐患；
   注册数据经 nm/cuobjdump 核验一致、同对象独立链接却正常）。合译为
   `openpcdet_cuda_ext.cu` 后消除，并配合目标级 `CUDA_SEPARABLE_COMPILATION
   OFF`（rdc=true 下 CUB 设备级 kernel 启动失败，aliked 时代遗留设置一并清理）。
5. **deconv gate 节点名 opt-in**：上游 `conv2d_transpose_gemm_path` 按全局
   dispatch；本仓库沿用 0021/0022 的节点名 opt-in 先例（`opcpd_deconv`），仅
   OpenPCDet F32 deblock 走该路径，f16（SAM3）、mapany pristine 与其他任务
   路由不变。
6. **`ggml_conv_2d_direct_packed` 未移植（证据否决）**：上游 `bev_backbone.cpp`
   注明 packed kernel 仅支持量化权重（converter 只打包 quantized weights），
   f32 centerpoint 的 50 个 conv 节点经 `OPCPD_CONVDBG` 实测全部走
   wino/f16-s2/fast-b4（与 0022 移植链逐条一致），不经 direct_packed。剩余
   差距的真实根因是 CT2D 1x1/s1 走 naive kernel（0026 修复）与 graph replay
   gap（未解决，见残留）。
7. **CUDA 输出 hash 变化**：0026 使 1x1/s1 deblock 从 naive 循环归约改为 GEMM
   归约序（与上游一致），output_hash 相应变化（`e1e7…→6e52…`），属预期数值
   变化；逐框 IoU 门禁实测达标，CPU/Vulkan 未走该路径、输出逐位不变。

## 终审结论与遗留清单（2026-10-29，与对齐审计同步）

**集成面终审（六项全过）**：54 行模型全部集成且 `asset_digests.h` SHA256
逐个实测一致；manifest 六场景 glob 并集覆盖全部 54 行且带结构化 accuracy
对象；catalog-dump/level1-2-5（111 passed）/契约/coverage 100%/边界
`--strict` 全绿。详见
[UPSTREAM_ALIGNMENT.md](UPSTREAM_ALIGNMENT.md)「集成面核对」。

**遗留清单（全部定性，无未处理项）**：

| 项 | 定性 | 处置依据 |
|---|------|---------|
| focal f32 CUDA 0.99816（方案 C 实施后） | focal 链已逐位一致（memcmp 单测 0 字节差） | 残余差异首分歧在 focal 链**上游** semseg 分支（后端固有噪声喂给阈值敏感位点），方案 C 范围外；见 DETERMINISTIC_FOCAL_DESIGN.md 验证记录 |
| pointrcnn_iou 0.9947 | 继承上游未决项 | 上游 REPORT 自declared "strict parity pending" 0.9944 |
| dynpillar/second_multihead/transfusion/pointrcnn f32 CUDA 0.997–0.999 | 容差 | 检测数全一致，处上游自declared floor（0.9981–0.9988） |
| q8_0 nuScenes center-head 类 | 无判定手段 | 上游无量化基线；f16 全 aligned + KITTI anchor q8_0 对齐佐证实现正确 |
| graph replay ~5–10ms | 性能遗留 | 上游 Voxel CenterPoint 最优 76ms vs 我方 83–91ms 同机；需 CUPTI/nsys 基建 |
| bevfusion GPU 后端 ~10% | 性能遗留（可选） | 上游 hybrid routing decoder/head scalar GEMM 未移植 |
| 上游 baseline/cuda 8/18 旧基线 | 上游树侧问题 | 我方只读未代改；对比须用 fresh 参照 |

## focal f32 CUDA 残差来源分析与可行 patch 方案

**噪声链条**：任务侧图算子分解与上游不同（parta2 对照：上游
SPARSE_GATHER/SCATTER_ADD+MUL_MAT×732 vs 我方 GET_ROWS+MUL_MAT+CONCAT 分解）
→ 每个 MUL_MAT 的 tile 形状/split-K 归约树不同 → ~2e-7 量级的逐元素噪声
（双方各自与 CPU 自洽，即两边都不是"错"，只是浮点归约序不同）→ focal 的
`conv_imp` 重要性分支做 sigmoid 后过 **0.5 硬阈值**选 voxel 子集，该集合中
2945 个位点距阈值 <3e-5（最近 8e-9）→ 噪声使个别位点翻转选中态 → 下游特征
拓扑漂移 → 框坐标微差。本质是**硬阈值对连续噪声的放大**，权重分布决定的
结构敏感度，上游 CUDA 自身相对其 CPU 也漂 0.9997。

**方案评估**：

| 方案 | 评估 |
|------|------|
| A. 对齐 ggml mul_mat 归约序/算子分解 | 否决：影响全部任务的分派面，且上游自身也未消除噪声（上游 cuda vs cpu = 0.9997），"修"=复现对方噪声模式，方向错误 |
| B. mask 阈值加迟滞/双路 | 否决：改变算术语义，偏离上游定义 |
| C. focal 链全确定性化（**可行，范围有限**）：噪声逐层传播——每层 main conv 的 mul_mat 输出就是下一层 imp_features 的输入，仅 conv_imp 确定化无效；需把 `run_sparse_conv` 的 mul_mat 路径替换为确定性稀疏点积 kernel（K 维顺序归约，无 split-K/tile 重排），focal 模型 opt-in（仅 3 层 FocalSparseConv，计算量占该模型 ~85% 但绝对值秒级；其他 17 模型不受影响）。收益：focal CUDA ≈ CPU 逐位一致（超越上游 0.9997）。代价：一个确定性稀疏 conv kernel 实现 + 三后端回归，工作量中等 |
| D. 维持现状 0.9978 | 当前定性；检测级 8/8 全匹配已达标 |

推荐：若需要发布级"CUDA 与 CPU 逐位一致"证据，实施 C；否则维持 D。详细
设计（kernel 伪代码、fmaf/expf 舍入陷阱、文件清单、验证门禁、性能预算）见
[DETERMINISTIC_FOCAL_DESIGN.md](DETERMINISTIC_FOCAL_DESIGN.md)。2026-10-29
检测级复核：focal f32 CUDA 三次重复运行均 **8/8 matched、mean_iou=0.9978088
（逐次一致，运行间确定）**，per-box IoU 最差 0.99602、无漏检/多检，分数分
布与上游平行——**检测级完全达标**，残差仅在框坐标第 3–4 位小数。

## 验证矩阵（2026-10-29，focal 修复后）

- `validate_all.py --tasks openpcdet`：**CPU 54/54 / CUDA 54/54 / Vulkan
  54/54 全部 PASS**；focal_multimodal 由 `openpcdet-focal-scene` 场景覆盖
  （scene 模式带标定，三量化 × 三后端九格全过，运行于修复后的 multimodal
  路径）。
- 全量对齐审计（18 checkpoints × 3 量化 × 3 后端 = 162 格）：
  [UPSTREAM_ALIGNMENT.md](UPSTREAM_ALIGNMENT.md)。核心结论：CPU f32 15/15
  满额匹配（IoU≈1.0）；上游 baseline/cuda 的 8/18 为旧构建离群产物，fresh
  参照下 parta2/pp_multihead/centerpoint 转 aligned；容差格全部检测数一致且
  处于上游自declared跨实现 floor；唯一代码修复即本节 focal 投影矩阵求逆。
- pointrcnn CUDA 精度（对上游同输入基线）：`kitti_pointrcnn` 11/11 matched
  mean_iou_bev=0.99924；`kitti_pointrcnn_iou` 12/12 matched 0.99862（检测级全匹配）。
- 契约测试 `test_openpcdet_capi_contract`、插件 `test_qopenpcdet_helpers`、
  C-API coverage 946/946（100%）、`check_plugin_aicore_boundaries.py --strict`
  全部通过。

## 配套 manifest 变更（同专项）

- `openpcdet-kitti` 场景排除 `kitti_pointrcnn*.gguf`（该模型要求固定
  16384 点预采样输入，通用场景输入不适用）。
- 新增 `openpcdet-pointrcnn` 场景：输入
  `openpcdet_inputs/kitti_000008_sampled16384.bin`（seeded 采样自 000008，
  fixture `openpcdet_input_pointrcnn.zip` 已发布至
  cloudViewer_downloads `OpenPCDet-GGUF` release 并在 input_assets 登记 SHA）。

## 方案 C 落地记录（2026-10-29，focal 链确定性化）

按 [DETERMINISTIC_FOCAL_DESIGN.md](DETERMINISTIC_FOCAL_DESIGN.md) 实施：focal
模型（`openpcdet.voxelrcnn` + `focal.multimodal`，GGUF 键
`focal.deterministic` 可覆盖，缺省=开启）的每层 FocalSparseConv 的 conv_imp
与 main conv 共 8 次稀疏卷积改走共享确定性核，绕过 ggml mul_mat 图；其余层
与其余 17 个模型路径零改动。ggml patch 不动、无 public ABI 变更。

**改动文件**：`extensions/deterministic_sparse_conv.{h,cpp}`（共享
det_sparse_conv_cell/rows 模板 + det_expf/det_sigmoid + CPU 邻居表展开与
std::fma 实例化，OpenMP 仅按输出 cell 划分）；`extensions/openpcdet_cuda_ext.cu`
追加 `det_sparse_conv_kernel` + host 包装（保持单编译单元；grid=(n/128,Cout/8)，
线程内 K 外层/C 内层串行 fmaf，无 shared/warp/原子归约）；
`models/voxelrcnn.cpp` `run_sparse_conv` 增 deterministic 分支（f16/q8 权重
回退原图作为对照）；`gguf_loader.{h,cpp}` 增 `hparams::focal_deterministic`；
`extensions/focal_sparse_conv.cpp` 分割 sigmoid 改用共享 det_sigmoid；
AICore/tests 增白盒门禁测试。

**实现要点（设计文档未展开、实施期确定）**：
1. 权重布局是 ggml 原生 o-major `[C_out, K*C_in]`（flat = o*(K*Cin)+k*Cin+c），
   det 核按该布局取数，c 内层天然连续访问。实施初期按数学行列序误读，被单测
   的 ggml 对照腿当场抓出（差 O(1) 量级）。
2. det_expf 用显式 det_fma 的 6 阶 Taylor（1 + r + r²P(r), P = 1/2 + r·q），
   两段 ln2 范围规约；|x|>87 饱和钳制（2^k·(1+e) 不溢出 FLT_MAX 的上界）。
   常量用 hex-float 字面量精确落盘（23-bit 尾数与 24-bit hex 字形差一位 nibble，
   手写易错，需 float.hex() 往返校验）。
3. CUDA 路径每层从 host 上传特征/规则表/权重（thread_local 常驻显存缓冲），
   结果拷回 host 后按位做 det_relu；与图路径的 backend 端 relu 数值等价。

**验证结果**：
- 单测 `test_openpcdet_det_sparse_conv`（whitebox；AICore_test 以同一
  `openpcdet_cuda_ext.cu` 编译单元链接 CUDA 核）：5 个形状
  （C_in 16/64 × C_out 16/27/32/64，K=27，n_out=1024，含 -1 空邻居与整行空）
  CPU(std::fma, OpenMP) 与 CUDA kernel 输出 **memcmp=0 字节差**；与 ggml
  dense lowering 对照差 ≤3.2e-5（按每 cell Σ|term| 缩放的 2e-6 相对门内，
  绝对差随 K·Cin 归约长度增长属正常序噪声）；det_expf 对 double exp 最大
  相对误差 2.2e-7。`ctest -R test_openpcdet_det_sparse_conv` 通过。
- 端到端（focal f32 scene，KITTI 000008）：vs 上游 cpp_cpu_f32 三后端
  **8/8 全 matched**——cpu mean_iou=0.9999998（中心 L2 2.3e-7 m，确定性
  归约序与上游 C++ CPU 循环对齐，较实施前 1.000000 的完全一致仅差末位），
  cuda 0.99816（实施前 0.99781），vulkan 0.97442。
- 回归：kitti_pointpillar f32 三后端 output_hash 逐字节不变
  （5ba9…/fece…/025d…）；kitti_voxelrcnn_car f32 cuda 检测 JSON 逐字节不变；
  `test_openpcdet_capi_contract` 通过。
- 延迟（e2e 中位数，同机同载）：CUDA 1412→~1409ms（Δ≈0，上传 ≈ kernel 节省，
  预算 +15–25ms 内）；CPU 2318→3748ms（+1.4s，确定性核为标量 fma 顺序归约，
  非实时模型可接受）；Vulkan 1133→3304ms（确定性核同样回退 CPU 执行）。

**门禁偏差（如实记录）：三后端 e2e output_hash 一致未达成，且在方案 C 范围内
不可达成**。114 层 taps 逐层二分（cpu vs cuda）证据链：
1. 首个分歧在 focal 链**上游**——semseg 图像分支 `voxelrcnn_image_features`
   cpu/cuda 差 0.0285（ggml 图多后端浮点差，含双线性插值）；plain 稀疏层
   conv_input/conv1_0 mul_mat 差 2.4e-6/3.8e-6，经 imp 累积放大到 2.3e-5，
   而阈值最近位点 margin 仅 8e-9 → conv1.1 / conv_focal_multimodal 的
   fore/back 选择集仍有翻转（`focal_mainconv_in_coords` 行数 577544 vs
   577516）。
2. 对照实验：**无 focal 链的 kitti_voxelrcnn_car 本身三后端检测 JSON 就不是
   逐位一致**（BEV backbone/dense head/RoI 的多后端浮点尾数差直达输出）。
   因此"检测框原始比特构成的 output_hash 三后端一致"对任何 VoxelRCNN 模型
   都需要全图逐算子确定性（semseg/BEV/RoI 全链），超出方案 C 范围。
   方案 C 的可达收益已交付：确定性核三后端逐位一致（单测）、focal 卷积自身
   的归约噪声从翻转预算中剔除、cpu 与上游对齐到 1 ULP 量级。
后续如需闭合（可选，未实施）：把确定性核扩展到 focal backbone 的全部
run_sparse_conv 调用（conv_input/conv1_0/conv2_pre/conv3_pre/conv4/conv_out）
可消除 LiDAR-only 路径的阈值翻转（scene 路径仍受 semseg 分支支配）；再往
上游/下游的全图确定化是独立立项。

**顺带修复（taps 专用路径）**：`run_focal_backbone` 的
`focal_mm_mainconv_{bias,weight}` tap 原直接解引用 `tensor->data`——GPU 后端
下权重在设备内存，cuda+ taps 必 segfault（CPU taps 不触发，属既有潜伏缺陷）。
改为 `ggml_backend_tensor_get` 读入宿主缓冲后再交给 sink。
