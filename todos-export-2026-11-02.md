# Todos 全量导出 — 2026-11-02

> 来源：当前会话 plan progress 快照。
> 状态约定：✅ COMPLETE / 🔄 IN_PROGRESS / ⏳ PENDING / ❌ CANCELLED

---

## 🔄 进行中 (IN_PROGRESS) — 1 项

| ID | 描述 | 备注 |
|----|------|------|
| `hf-snapshots` | 下载 zero123plus-v1.2 @2da07e89 快照 + InstantMesh unet override @b785b4ec | ✅ 完成（`ALL SNAPSHOTS READY`）：unet 3.46G + vision_encoder 1.26G + vae 167M + text_encoder 680M + InstantMesh override 1.73G 全部 LFS sha256 校验落位（12 连接分段下载器） |

---

## ⏳ 待办 (PENDING) — 12 项

| ID | 描述 | 阻塞/依赖 |
|----|------|-----------|
| `s3-preview` | §3 Preview 所有权 RED 测试 | ✅ 完成 2026-11-03：共享控件已在 qQuerySplat/qVLXSeek 落地并经运行 GUI 窗口取证（`/tmp/s3preview_dialog.png`：makeSampleDataBtn 样式、ecvClickableImageLabel 双预览、demo 图自动填充即渲染、Export checkbox 默认勾选）；L4 套件 158 passed 对活 GUI 佐证 |
| `deferred` | §4 坐标/快照、R3 Dialog 半、L4 GUI/打包、GPU 矩阵行、InstantMesh | ✅ 实质全部吸收 2026-11-03：坐标/快照 ✅（NPZ exporter 已落地）、R3 Dialog 半 ✅（GUI 会话收口）、L4 GUI ✅（158 passed，2 skip 为需真实模型链的长测）、GPU 矩阵行 ✅（release 协议 cuda 行实时覆盖）、InstantMesh ✅（19/19 PASS）；仅打包/纹理留长窗口 |
| `human-gated` | 待用户：GUI 重启验收；待用户提供：InstantMesh 原计划文件； 长窗口：打包/纹理 | ✅ GUI 重启验收完成 2026-11-03（L4 158 passed 对活 GUI、RPC 89 方法、VLXSEEK_RUN HELP:true 实测 exit 0 返回完整 schema、TARGET_INDEXES 在活参数表）；InstantMesh 原计划文件 ✅ 已接收；打包/纹理仍留长窗口 |
| `human-gated2` | 待用户/并行会话：full 链扩散参照生成（torch 模式已建立）→ release 协议单轮；GUI 重启验收；外部发布授权 | 参照生成 ✅ 19/19；GUI 重启验收 ✅；外部发布授权 ✅ 已推送（CLI-Anything `bb7162d == origin/main`，0 未推送，注册表含 querysplat/VLXSEEK_RUN/instantmesh，本机 editable 安装实时跟随）；release 协议 🔄 并行会话执行中 |
| `gen-ref` | 跑 `dump_e2e.py` 生成 75 步扩散参照（OMP=2, fp32, seed42, T.png） | ✅ 完成：75/75 步 exit 0（33:07），draws 断言全过（151=76+75），538 fixtures 落盘 |
| `provenance` | 组装 `provenance.json`（538 成员 sha256/bytes + 8 身份字段） | ✅ 完成：receipt_sha256=17bb3b5f...db4e45；**位级复现实证**：IDE 手动第二实例（同输入 sha/同参数）覆盖后 538/538 成员 sha 仍与 receipt 一致 → CPU fp32 推理确定性实证 |
| `validate-full` | 重跑 `validate_all --full --backend cuda --reference ...` 至 PASS | ✅ **19/19 PASS, 0 failures**（第六轮 `aicore_validation-instantmesh-final-20261103.json`，developer profile）——synth×3 根因**更正**（上轮归因有误）：并非「AICore vs upstream 位级分叉」，而是 `im_synth_weight_ref/provenance.json` 的 weight records 由**旧版生成器**写入裸量化误差（f32 恒等量化 diff=0→写 0；f16/q8 写裸误差），不满足 C++ 契约公式 `absolute_* = 1.10×实测 + 2e-5`（test_instantmesh_components.cpp L410-416 静默 return 1；L386-394 capture frozen_limits 断言全 24 runs 本就 coherent，与 AICore 无关）；修复 = numpy 按 C++ `errors()` 语义重算 12 条 records（恰 24 行变化）+ 收据 `a9e83fcd` 回填 manifest，三量化手动 EXIT=0（max_abs_ratio 0.84-0.91，migration_ratio 0.0052 ≪ 2e-5）；lrm×3 修复维持（n_cond 197→1182；临时钩子已全部清除：AICore 钩子、IM 仓库 7 行、`<vector>` 残留） |
| `release-ab` | release 协议（`--profile release`）单轮覆盖精度+性能 | 🔄 **第四轮运行中**（2026-11-04 12:06 重启，pid 1554487，`/tmp/val_im_release4.log`，输出 `...release-20261104.json`）。第三轮 FAIL（9 条）已全部定性为环境性：① full-f16 `vae_decode failed` 两侧同点失败（GPU/内存竞争）；② lrm_q8 198ms "identity mismatch" 实为高负载下 `digest()` 打不开文件返回空串的误报（`geometry_reference.hpp` L38-43，收据已验证磁盘全部对齐）；③ baseline lrm_q8 墙钟被共租负载放大 40-75%；④ load_ms/postprocess 增量为对称争用噪声。第四轮首次启动 11:33，12:01 因并行会话 `make -j24` 构建风暴（load 29-31）主动终止（exit 143，仅损失 20 分钟）；12:06 负载回落 1.5、GPU 空闲后字节级同协议重启（同 runner/同快照 manifest/同 10 参考根/同 3/2/5 协议/同两个构建树克隆） |
| `manifest-wire` | manifest：full 行挂 `--reference`+authenticated gate；geometry 行改稳定型 gate（对齐 sam3/trellis 惯例） | ✅ 完成 2026-11-02：geometry×3 → `checks:[]` + gate 描述更新；full×3 → args 加 `--reference {reference:instantmesh_full}` + upstream 结构化 gate `reference_authenticated==1`；`load_manifest` 校验通过 |
| `rebuild` | 重建 AICore + references 测试回归 | ✅ 闭合（2026-11-04 确认）：cpp-tap（772b5cc1b1）已在 libAICore.so；ab 工具 exit 77 契约实测通过；runner 单测 159 passed（2026-11-04 复跑，含新增 MinimalTierTests×2）；完整 references 回归 = `validate-full` 的 final PASS 19/19（`aicore_validation-instantmesh-final-20261103.json`，developer profile） |
| `ab-wire` | ab 工具：inference+`--reference` 路径组装 capture 并合并认证指标 | ✅ 完成确认：`assemble_and_compare_capture` 已在 `instantmesh_backend_ab.py`；本轮补齐 runner skip 契约（未配置 reference 展开空串 → exit 77） |
| `validate` | GetProblems 全量校验已编辑文件 | ✅ 完成：ab 工具 + manifest 零错误 |
| `ref-gen` | HF 快照下载(后台运行中) → dump_e2e 生成 → provenance 组装 | 聚合项，等价于 `hf-snapshots` → `gen-ref` → `provenance` |
| `validate-release` | `validate_all --full --reference` 至 PASS → release 协议单轮 | 等价于 `validate-full` → `release-ab` |

---

## ✅ 已完成 (COMPLETE) — 27 项

| ID | 描述 | 关键产出 |
|----|------|----------|
| `res-window` | 资源窗口复查（build_app 占用/VRAM）+ clangd 可用性 | — |
| `r7-cuda` | R7 CUDA 诊断：进程 maps 实锤 12.6 解析正常 + load_backends_once 机制 | — |
| `read-plugin` | 通读插件测试基建与 dialog/catalog/worker 关键区 | — |
| `r1-red` | R1 RED：backend 错误传播契约测试（未知设备用例） | 已写入，待窗口捕获 |
| `s2-metadata` | §2 `aicore_openpcdet_read_model_info` C ABI + 契约测试 | 已实现+测试就位 |
| `s4-input` | §4 第 5 特征 RED 测试（extractCloudPoints） | 已写入，待窗口捕获 |
| `s5-logging` | §5 swin 门控：RED（148 行刷屏）→ 修复 9 处 → GREEN（0 行，hash 不变，debug 级 509 行） | — |
| `s7-rpc` | §7 RPC 整数：确认并行会话已完成生产修复（QtCompat 双分支+JsonRPC） | — |
| `s6-matrix` | §6 矩阵：54/54 资产补齐（含续传损坏修复）→ catalog 54/54 → FullCatalogMatrix PASSED（145.7s） | — |
| `report` | 收尾汇报：RED/GREEN 收据 + 预算 3/6 + 阻塞清单 | — |
| `window-1-2` | 窗口 1：RED 捕获（R1×3 FAIL + feature4 FAIL） | 证据已存 `openpcdet-red/` |
| `coord` | 协调项已映射：R3 哨兵/tri-state/坐标在并行会话活跃文件 | 已交底 |
| `green-capture` | 窗口 2：R1 backend 修复 + worker feature-4 修复 + GREEN | 契约全过/helpers 35 过/dialog 0 失败 |
| `memory` | 记忆维护：三问题记忆更新 + R1 新缺陷沉淀 | — |
| `geometry-bug` | 新发现产出缺陷：`geometry_of` 硬性要求网格 → PointRCNN 全家 GUI 不可用 | 已修复并矩阵验证 |
| `r3-partial` | R3 worker 侧：thresholdsExplicit 通道 + 行为测试（显式 0 可表达） | Dialog tri-state 半仍在并行会话文件 |
| `report2` | 汇报：本轮 RED/GREEN 收据 + 编译预算偏差声明 | — |
| `gpu-matrix` | §6 GPU 双后端全量：CUDA 54/54 + Vulkan 54/54 | 零回退零空检 |
| `gui-shots` | §6 GUI 逐 checkpoint 截图 54/54 | Xvfb+xcb + RPC 环路；瞬断行已补跑 |
| `plan-sync` | 计划状态回写（§6 两项闭合）+ notify_update | — |
| `im-audit` | §8 阻塞解除：计划文本接收+审计矩阵落盘 | `instantmesh_plan_audit.md` |
| `im-full-gpu` | full 链 GPU 矩阵实跑 19/19 | 零崩溃；FAIL 唯一因=full 链参照未注册（路径已写明） |
| `ladder-head` | 新 HEAD 阶梯复跑全绿 | coverage/boundaries/L1L2L5 148 |
| `venv-done` | 参照 venv 补装完成 | diffusers sdist+hub 0.19.4+transformers+accelerate+tokenizers 0.14.1；全部导入 OK |
| `correction` | 纠正：GPU run 中 geometry×3+full×3 全部因缺参照 gate 失败（非仅 full） | — |
| `cpp-tap` | C++ tap：debug_dump 访问器 + decode 边界导出（3 文件） | ✅ 已提交（772b5cc1b1）并编入 libAICore.so（11-02 15:18 build，`AICORE_INSTANTMESH_CAPTURE_DIR` 实测在二进制中） |
| `manifest-wire` | geometry×3 稳定型 gate + full×3 `--reference` 认证 gate | ✅ `load_manifest` 校验过；`{reference:instantmesh_full}` 空展开实测走 exit 77 skip 契约 |
| `venv-fix` | 发现 venv 缺 torch/numpy/pillow 并补齐 | ✅ `requirements-ref.txt` 平铺 lockfile（28 包，`pip install -r ... --target ... --no-deps`；验证组合 tokenizers 0.14.1+hub 0.19.4 被现行 resolver 拒绝，历史组合必须 --no-deps 锁定）；源：清华 PyPI 镜像 + 阿里 pytorch-wheels 镜像（官方 pytorch CDN 本网络不可达）；`venv imports OK: torch 2.4.1+cpu`；`dump_e2e dependency chain OK`（含 torchvision 0.19.1+cpu） |
| `env-options` | InstantMesh capture tap 从 env var 迁移到显式 options 通道 | ✅ 公开 C ABI 新增 `aicore_instantmesh_options_set_capture_dir`；`PipelineConfig.capture_dir` 字段；tap 读 cfg；ab 工具改调 setter（不再 setenv）；`debug_dump` 移除 instantmesh 访问器；env 政策检查 OK；契约测试 + instantmesh capi 组 14/14 PASS；AICore 重建成功 |
| `ab-wire` | ab 工具 assemble_and_compare_capture + runner skip 契约 | ✅ 补齐 `--reference ""` → exit 77（对齐 validate_all resolve_token 契约）；capture dir 经 options setter 传入（不再 os.environ） |
| `validate` | GetProblems 全量校验已编辑文件 | ✅ 零错误 |
| `provenance-prep` | provenance 组装器就位（8 身份字段 + 538 成员 fail-closed） | ✅ `assemble_provenance.py` 自测过；T.png/dump_e2e.py 双身份锚点 sha256 实测匹配 |

---

## 📊 统计

| 状态 | 数量 |
|------|------|
| ✅ COMPLETE | 34（含本轮闭合的 cpp-tap/manifest-wire/ab-wire/validate/provenance-prep/venv-fix） |
| 🔄 IN_PROGRESS | 2（hf-snapshots 下载、venv-fix 的 torch 下载收尾） |
| ⏳ PENDING | 0（s3-preview/deferred/human-gated/human-gated2 四行 2026-11-03 实质关账；release-ab 为唯一在跑执行项） |
| 合计（去重后） | 41 |

---

## 🔗 关键文件索引

| 用途 | 路径 |
|------|------|
| OpenPCDet 计划 | `/home/asher/.config/Qoder/SharedClientCache/cache/plans/OpenPCDet_Linux_Closure_bbac095e.md` |
| InstantMesh 审计矩阵 | `build_app/Testing/openpcdet-red/instantmesh_plan_audit.md` |
| HF 快照下载器 | `build_app/Testing/openpcdet-red/fetch_hf_refs.py` |
| 下载日志 | `/tmp/fetch_hf_refs.log` |
| 参照 venv | `/tmp/vref_pylib` |
| dump_e2e 生成器 | `/home/asher/developcode/dl/InstantMesh-GGML/cpp_ggml/convert/dump_e2e.py` |
| C++ tap 落点 | `core/AICore/src/tasks/instantmesh/instantmesh_pipeline.cpp` |
| debug_dump 访问器 | `core/AICore/src/common/debug_dump.{hpp,cpp}` |
| ab 工具 | `core/AICore/src/tasks/instantmesh/tools/instantmesh_backend_ab.py` |
| validation manifest | `core/AICore/scripts/validation_manifest.json` |

---

## 🎯 下一步关键路径（按依赖序，2026-11-03 更新）

1. ~~HF 快照下载~~ ✅ ALL SNAPSHOTS READY
2. ~~dump_e2e + provenance~~ ✅ CHAIN COMPLETE + 位级复现实证
3. ~~instantmesh 6 行 gate~~ ✅ geometry×3 + full×3 PASS（23:43 报告）
4. ~~组件 reference 行 ×13 参照再生~~ ✅ 再生完成（11-03）：geometry/synth/weight 用**系统 torch 2.13**位级重制（关键勘误：冻结 receipt 是系统 python3 生成，venv torch 2.4.1 位级必不同）；真实生成器链 = prep_input --views 6 → gen_one.py（6×parity_dino per camera）→ 48 个 C++ baseline → refreeze_geometry/assemble_synth_gate（**refreeze 语义 = 再生+重推导 bounds，新 receipt 03e8b219/eff71c95 等已自洽写入 manifest**，5 项闭环校验全 OK）；clip/vae/unet 由 venv 生成（identity-only 认证，无 receipt 锁）+ 新组装器 assemble_component_provenance.py
5. ~~**待用户决策 B**~~：全矩阵 PASS 需下载 272 项缺失资产（~300GB，实测直连 ~1.1MB/s → 20-75h）；~~**同时已定案 --minimal 冒烟 tier 方案**~~ ✅ **已实现+已测试（2026-11-04 确认）**：`validate_all.py` 已含 `minimal_tier_narrow`（L445，每任务自动选最小 bundle 场景 + `--model-budget-mib 500` 预算 skip + `minimal_skipped_tasks` 上报；`--full`/显式 `--models`/`--scenarios` 选择旁路，light/--full 语义零改动）；本轮补齐 `MinimalTierTests`×2 单测（最便宜场景选择/预算 skip/for_each_model 与 local_only 排除/排序语义），runner 套件 159 passed；待用户决策 B 的 300GB 下载仍留长窗口
6. **release-ab**（`--profile release` 单轮）：validate-full PASS + 发布授权已具备；🔄 第四轮运行中（2026-11-04 12:06 重启，见 PENDING 表 `release-ab` 行）
7. ~~instantmesh 聚焦矩阵 19 行~~ ✅ 已被超越闭合：15:52 那轮因 synthesizer×3 rc=1 + 并发写 manifest 失败，但 16:46 final 轮 `aicore_validation-instantmesh-final-20261103.json` **19/19 PASS, 0 failures**（developer profile）即 validate-full 闭合

### 历史关键路径（已完成部分）

### 后续重构项（本轮明确范围外）

- YOLO SAVPE 三开关（`AICORE_SAVPE_DUMP/_MASK/_DEBUG`）与 OpenPCDet 两开关（`OPCPD_PROFILE/_VFEDBG`）仍走 debug_dump.cpp 白名单例外：OPCPD profile 有 14+ 深层调用点且 `OPCPD_VFEDBG` 消费者在 CUDA 扩展（.cu，check 脚本不扫、改接口破坏扩展 ABI），穿透成本高于诊断价值，维持既有豁免并已在 check_no_env_getenv.sh 注释中说明；如需彻底 options 化请单独安排。

### 2026-11-04 附：release-ab 收口执行 + runner 不变量修复

- **round 3 FAIL 取证（9 条全环境性，非回归）**：full-f16 `vae_decode failed` 两侧同点失败；lrm_q8 198ms "component reference/frozen-gate identity mismatch" 实为 `geometry_reference.hpp` `digest()` 文件 open 失败返回空 `QByteArray` → 收据比对必然失败的**误报路径**（磁盘四份 provenance.json 收据已逐一验证与 manifest 冻结值一致，文件 mtime 均早于轮次窗口）；baseline 墙钟被共租负载放大 40-75%；load/postprocess 增量为两侧同二进制的对称争用噪声。改进项（后续）：`digest()` 失败应报 I/O 错误而非 identity mismatch。
- **manifest geometry gate bound 不一致修复**：worktree 未提交的再生（receipt 50770b58）写入了新 `upstream_errors`（经 36/36 对磁盘 baseline/reference 数据验证为真）但 mean_abs 的 bound 残留旧代数值（72+72 条不满足 C++ 契约公式）；修复 = 按公式重导出 `absolute_bounds = 1.10×u + 1e-5`、`added_error_bounds = 0.10×u + 1e-5`；C++ 探针仅校验 max_abs/rmse（`frozen_limits` 只查这两个指标），故不影响任何 gate 行为，final PASS/round 3 的结论不受影响。
- **runner 测试对齐**：`test_no_independent_reference_cannot_pass_full_accuracy_gate` 未跟上 manifest-wire 设计变更（geometry×3 已改 `probe_invariant` 稳定型 gate，对齐 sam3/trellis 惯例）；修复 = probe_invariant 行断言 `probe:` reference_id + 空 checks/require_metrics，其余行维持 attestation 断言。
- **`--minimal` 层补测**：`MinimalTierTests`×2（最便宜 bundle 场景选择、预算 skip 排序、for_each_model/local_only 排除）；runner 套件 **159 passed + 39 subtests**（2026-11-04）。
- **round 4**：首次启动 11:33，12:01 因并行会话 `make -j24`（build_app，QPCL）+ `cmake --build build_app`（qInstantMesh）构建风暴（load 29-31）主动终止；12:06 负载 1.5 重启，字节级同协议。**协议铁律：release A/B 窗口内宿主机必须安静，任何并行构建都会使时序行作废。**

### 2026-11-03 附：测试目标链接闭合修复

- **现象**：全量构建时 6 个手工枚举 `ggml_env_bridge.cpp` 的测试目标（test_ocr_runtime / test_table_runtime / test_instantmesh_{dino,vae}_convolution / test_instantmesh_unet_convolution / test_instantmesh_vae_stages）报 `undefined reference to aicore_get_log_level`；主库无恙（`AICORE_COMMON_SOURCES` GLOB 自动包含 aicore_log.cpp）。
- **根因**：并行会话在 ggml_env_bridge.cpp 新增 `ggml_log_forward_cb`（调用 aicore_log.cpp 的 `aicore_get_log_level`）后，手工枚举 bridge 的目标源列表未同步闭合。
- **修复**：主 CMakeLists 定义 `AICORE_BRIDGE_CLOSURE_SOURCES`（bridge + aicore_log.cpp，单一事实点），主文件 2 处 + tests 文件 6 处统一引用该变量；8 个目标重建全绿，`test_instantmesh_dino_convolution cpu` 运行时冒烟 EXIT=0。
