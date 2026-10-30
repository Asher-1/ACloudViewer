# OpenPCDet 上游对齐审计（18 checkpoints × 3 量化 × 3 后端）

记录 2026-10-29 的全量对齐审计与收口：对每个 `(checkpoint, quantization,
backend)` 格给出「我方 vs 上游 OpenPCDet-GGML」的 mean BEV IoU 与检测数对比，
并注明每类差距的处置与依据。上游参考仓库：
[Asher-1/OpenPCDet-GGML](https://github.com/Asher-1/OpenPCDet-GGML)。

## 审计方法与参照口径

- 精度对比统一用上游权威工具（`comparison_metrics` + `bev_iou`），参照优先
  级：**上游当前 CLI 现跑 > validation/ 存量基线 > REPORT.md 结论旁注**。
  存量基线不足的格标注 `no_upstream_baseline`，不编造。
- **测量修正（重要）**：审计发现上游 `validation/baseline/cuda/` 的 8/18 文件
  来自旧构建、自身偏离自家 CPU 真值（如 parta2 up_cuda vs up_cpu 仅 0.9891）。
  用上游**当前** CLI 重跑参照后（`/tmp/up_fresh_*.cuda.json`），parta2
  0.9889→0.9997、pp_multihead 0.9914→0.9991、centerpoint 0.9961→0.9991 全部
  转 aligned——这三个"gap"是上游基线过时的测量假象，我方输出反而更贴近 CPU
  真值（我方 parta2 cuda vs 我方 cpu = 0.9997）。
- 另一处口径修正：focal_multimodal 的上游 8 检基线是 **scene 模式**（PNG+标
  定），points-only 直入会走 LiDAR-only 降级路径，二者不可比。

## 对齐矩阵（修复后终值；A=aligned T=容差/继承项 N=无上游基线 S=量化跳过）

| checkpoint | f32 cpu | f32 cuda | f32 vulkan | f16 cpu/cuda | q8_0 cpu/cuda |
|---|---|---|---|---|---|
| kitti_pointpillar | 1.0 A | 0.9995 A | 0.9996 A | A / A | A / A |
| kitti_second_iou | 1.0 A | 0.9997 A | 0.9998 A | A / A | A / T(多1检,近阈值) |
| second_7862 | 1.0 A | 0.9998 A | 0.9997 A | A / A | A / A |
| kitti_pointrcnn† | 1.0 A | 0.9989 T | 0.9911 T | A / A | A / A |
| kitti_pointrcnn_iou† | 1.0 A | 0.9947 T‡ | 0.9983 T | A / A | T / T‡ |
| kitti_pvrcnn | 1.0 A | 0.9997 A | 0.9996 A | A / A | A / A |
| kitti_voxelrcnn_car | 1.0 A | 0.9994 A | 0.9993 A | A / A | A / A |
| kitti_voxelrcnn_focal_multimodal§ | 1.0000 A | 0.9978 T | T | A / A(优于上游同格) | A / A(≈上游自身) |
| kitti_parta2_anchor | 1.0 A | 0.9997 A¶ | 0.9997 A | A / A | A / A¶ |
| nuscenes_pointpillar_multihead | 1.0 A | 0.9991 A¶ | 0.9997 A | A / A | A / A¶ |
| nuscenes_dynpillar_centerpoint | 1.0 A | 0.9988 T | 0.9993 A | A / T | T / T |
| nuscenes_centerpoint_voxel0075 | 0.999999 A¶ | 0.9991 A¶ | 0.9979 T | A / A | T / T |
| nuscenes_second_multihead | 1.0 A | 0.9987 T | 0.9979 A | A / A | T / T |
| nuscenes_transfusion_lidar | 1.0 A | 0.9971 T | 0.9921 T | A / A | T / T |
| nuscenes_voxelnext | 1.0 A | 1.0 A | 0.9980 T | A / A | T / T |
| nuscenes_voxelnext_doubleflip | 1.0 A | 1.0 A | 0.9982 T | A / A | T / T |
| kitti_caddn | 1.0 A | 0.9996 A | N | A / A | A / A |
| nuscenes_bevfusion | 1.0 A | 0.9982 A(=上游 floor) | 0.981 T | A / A | T / T |

\† pointrcnn 系需 16384 点预采样输入（双方用同一 seed0 fixture 对比）。
\‡ pointrcnn_iou 0.9947 = 继承上游未决项（上游 REPORT 自declared
"strict parity pending" 0.9944、min IoU 0.9559，我方与其同格一致）。
\§ focal 修复见下节。f16/q8_0 参照为上游 f32 基线（上游无量化基线的格，
量化误差属预期损失；f16 cuda 0.9739 与 q8_0 0.994+ 均**优于上游自身量化对
f32 的差距** 0.9725/0.9791）。
\¶ 上游 cuda 基线过时的格子，已用上游当前 CLI 重跑参照修正。

## focal_multimodal 修复（本轮唯一代码修复，已完成）

**根因**：`voxelrcnn.cpp` 的 `invert_4x4`（4×4 伴随矩阵求逆）`inv[10]` 项
抄写错误：`-m[12]*m[2]*m[5]` 应为 `-m[12]*m[3]*m[5]`。标定矩阵含平移
（m[12..14]≠0）时逆矩阵错 → `lidar_to_camera` 投影错位 →
`conv_focal_multimodal` 图像采样/融合失效 → focal backbone score 塌陷至 2 检
（0.39/0.31）vs 上游 8 检。修复 1 行后 f32 CPU 恢复 **8/8 全匹配
mean_iou=1.000000**，全部 9 格检测数恢复 8/8。

**残余 f32 cuda 0.9978 定性（容差）**：CPU taps 114 个中间层全部逐位一致
（openpcdet 层无缺陷）；CUDA 差异源于 ggml mul_mat 求和顺序噪声（~2e-7）
× mask_kernel 阈值敏感位点（2945 个位点距 0.5 阈值 <3e-5，最近 8e-9）的
拓扑漂移——上游 cuda 自身相对 cpu 也漂移 0.9997，且该敏感度由权重决定，
任务侧无法触及；收敛需在 ggml patch 层对齐求和顺序，未纳入本轮。

## 容差格的统一依据（不修的理由）

1. **检测数全部一致**，差距仅在框坐标微差（累加序/GEMM 归约序类）。
2. 数值处于上游自declared的**跨实现复现性 floor**（bevfusion 200 框集合
   0.9981–0.9988；KITTI 12–25 框小集合对单框微差更敏感）。
3. "修"=复现上游离群/特定构建的算术序，方向错误（见上文测量修正）。
4. q8_0 nuScenes center-head 类（centerpoint/dynpillar/second_multihead/
   transfusion/voxelnext/bevfusion）vs 上游 **f32** 参考的 0.95–0.98 属量化
   损失本身——上游无量化基线可比，KITTI anchor 类 q8_0 全对齐佐证我方量化
   实现正确。

## 功能面差距（上游有而 AICore 无——均为上游开发工具链，非推理契约）

- scene 独立 CLI（`openpcdet-bevfusion-cli`/`openpcdet-caddn-cli`）：
  AICore 由 `detect_scene` + runner/batch 命令覆盖，无需独立 CLI。
- `--dump-taps` 中间层导出：审计期间的对照由上游 CLI 完成；AICore 侧未开放
  同名接口（如需常备诊断可复用 0025 perf-op 门模式）。
- caddn 标定资产管线（`*_caddn_calibration.f32`）与 python 参考脚本链：
  属上游converter 工作流，AICore 消费的是已转换 GGUF。

## 集成面核对（2026-10-29 终审）

- **模型支持**：18 checkpoints × 3 量化 = 54 行全部集成；`asset_digests.h`
  54/54 行 SHA256 与本地缓存逐个实测一致；catalog-dump 测试通过；manifest 六
  场景（kitti/pointrcnn/nuscenes/caddn-scene/focal-scene/bevfusion-scene）
  glob 并集覆盖全部 54 行；全部场景带结构化 `accuracy` 对象。
- **上游 62-config 清单对照（MODEL_SUPPORT.md 权威口径）**：validated_config
  12 + validated_family 16 = 28 个已验证 config 全部由我方 18 checkpoints 的
  family 覆盖（多 config 共享同一 checkpoint）；9 个 implemented_unvalidated
  中除 4 个无发布资产的 waymo config 外全部已集成且完成三后端验证（上游自身
  未验证）；19 个 pending_model_integration 与 6 个 discarded_no_asset 为
  **上游自己未完成/无 checkpoint** 的部分，我方不落后。结论：我方集成集合
  ⊇ 上游全部有发布 GGUF 资产的模型，验证深度（3 量化 × 3 后端逐格 mean_iou）
  超过上游自身。
- **自动化**：level1/2/5 共 111 passed（skip 为需 GUI 用例）；契约/coverage
  946/946（100%）/边界 `--strict` 全绿。
- **性能遗留（已记录）**：dense 图 graph replay 通用 gap ~5–10ms（上游
  Voxel CenterPoint 最优 76ms vs 我方同机 83–91ms；定位需 CUPTI/nsys 基建，
  本机不可用）；bevfusion CUDA/Vulkan ~10%（上游 hybrid routing 的
  decoder/head scalar GEMM 路径未移植，可选后续项）。
- **上游树侧建议（我方只读未代改）**：上游 `validation/baseline/cuda/` 的
  8/18 旧基线应整体重生成。

## 回归证据（focal 修复后，2026-10-29）

- `validate_all.py --tasks openpcdet`：**CPU / CUDA / Vulkan 三后端 verdict
  全部 PASS**。
- 契约 `test_openpcdet_capi_contract`、插件 `test_qopenpcdet_helpers`、
  `check_plugin_aicore_boundaries.py --strict` 全部通过。
- 焦点回归：kitti_pointpillar CUDA 24/24 @0.99952、kitti_voxelrcnn_car CUDA
  10/10 @0.99945（均不回退）。
- 原始矩阵与产物：`/tmp/alignment_matrix.{json,md}`（162 格明细）、
  `/tmp/up_fresh_*.json`（fresh 参照）、`/tmp/audit/`（我方 dump）。
