# OCR 逐文件发布门控快照

这是当前工作树的本机验证快照，尚无新文件通过完整发布门控，新增上传数量为 **0**。
51 个主文件（36 个新增候选、12 个已发布文本/布局文件、3 个已发布 SLANet）
与另外两个 Q8 修复候选按精确 SHA 分开记录；同名文件不继承旧文件验收。
2026-10-20 的 GitHub API 复查确认 PP-OCR release 仍有 16 个资产
（15 个 GGUF 和 1 个 ZIP），当前凭据具备 push/admin 权限；没有上传权限卡点。

合并后运行库 SHA 为 `9f2548ef28f6edadbcdbb0658d6fbf67e18dbd73ca2afb28183c2b2e9af59f21`。
96 项图门已在这个新库完整复核，两 GPU 各 46 PASS / 2 FAIL；失败只有布局 F16/Q8。
新库的 120 项页面/SLANet 复核已完成：Vulkan 60/60 PASS，CUDA 首轮 59/60 PASS；
唯一因 CMakeCache 身份变化失败的 CUDA 项按原条件独立重试 PASS，原失败收据保留。
这批页面仍使用原始 CPU 源参考，只计 developer 证据，原始 CUDA 页面参考复核另列。
逐项状态见 `merged-gpu-pages-20261020-v5k89ifx/` 和 `merged-gpu-followup-path.txt`。
下述旧库 `154a0b62` 的结果继续保留为历史证据，不能跨库 SHA 转记为新库 PASS。
36 个新增文件的源 probability/argmax 图门均已有单轮通过证据。
旧库新增源页面 54 个场景已全部完成：CUDA **54/54 PASS**、Vulkan **54/54 PASS**。
CPU 的 54/54 PASS 仅保留为诊断，不参与 GPU 发布资格；没有等待 CPU 的发布卡点。
三后端源图首轮的跳过和旧 CPU 超时记录保留，重跑不会将旧 INCOMPLETE 改写为 PASS。
这些都是 developer 数值/页面证据，不是完整 release 稳定性或性能验收。

公共文本 CUDA/Vulkan 原文件回放各为 24 PASS / 4 FAIL。旧 Q8 修复候选的七组
组合页面和源图结果保留为历史证据；当前另两个修复候选 SHA 必须重新验收，
不继承旧文件的通过状态。
公共 layout F16/Q8 的严格源 argmax 门仍失败。已固定 CDLA 全部 1,000 页
人工验证标注和 8,711 个区域，官方转换等价检查通过；独立原始 CUDA 与
两 GPU、三种存储均已完成全部 1,000 页导出，实际 AP 差值见文末；
页面发布容差及公共消费者仍待完成，不能据此记为完整页面门通过；
SLANet 三种存储的正式源图消费者在两 GPU 上 6/6 PASS；旧库裁剪表格/组合页面 × F32/F16/Q8_0 的
CUDA/Vulkan **12/12 PASS**，CPU 另外 6 项诊断也通过。
GPU F32 裁剪表格 TEDS/TEDS-S 为 0.9645975/0.9959180，
组合页面为 0.5431670/0.5917242；这是源对照通过，不代表页面绝对质量达到 1。
表格 TEDS 不以文字 CER/WER 替代。

受控性能仍缺有效的完整同机 baseline/candidate A/B。
2026-10-20 再次运行 CUDA/Vulkan 各三轮 release：每轮在计时前被
“GPU did not become idle before measurement” 拒绝。Xorg/桌面活动有实际
SM 占用，报告为 gpu-release-20261020-py6cqztd/{cuda,vulkan}/matrix.json。
没有停止外部任务或放宽门限；这次拒绝不能证明候选速度不达标。
后续将等待窗口延长至 120 秒，CUDA baseline/candidate 首轮仍在计时前拒绝；
等待样本同时记录外部计算进程和桌面图形活动。已保留两次完整拒绝收据并取消
剩余重复等待，让独立精度队列继续。Vulkan 本次后续复测未执行，不伪记失败或通过。
本次中止记录见 `merged-gpu-followup-path.txt` 所指目录中的
`controlled-release-cancelled-after-audit.json`，性能资格仍为未通过。
原始 CUDA 输出对照独立于性能审计：36 个新文件在 CUDA、Vulkan 各 36/36 PASS。
公共 Paddle 图在两后端各 10 PASS / 2 FAIL；失败均为 layout F16/Q8_0。
它们相对原始 Paddle CUDA 的最差头 argmax 分别为 0.992307692 / 0.969230769，
两后端一致；developer 延迟不具备受控性能资格。
公共首轮失败是虚拟环境解释器路径解引用和探针缺少八个布局头导出，
发生在精度比较之前；修复后完整重跑，旧失败记录保留。

已完成本轮 OCR 文档窗口的表格编辑、富文本基础行为及 Chat/VQA 配置/消息工作；
模型发布资格、完整视觉、本地 VLM/剩余 catalog 和 Linux clean-package 仍未完成。
表格 5,407 项原生/上游内容对照、6 组几何、6 组视口高度和 2 组主题 QSS 对照，
以及实际插件四组入口/视频回归及 Chat/VQA 五组上游方法对照通过；
富文本另有 13 项序列化、6 项标题指标、12 类粘贴、3 组裁剪预览、6 组高度
上游方法对照，另有浅/深两主题 QSS/自定义标题弹层、源绘制图标及两组图片
选择/滚动/删除对照，包含图片过滤、标题恢复、图片块后的 Enter 和元数据预览
载入；最新 mini-probe 结果不替代正式插件重建或模型验收。远程 focal 的实际
安装包生成、安装和 headless/JSON-RPC 测试成功，详见审计文档的 CI 复核。
本机旧安装树及额外测试不证明远程打包失败；相关打包代码、开关和构建文档已
全部恢复到同步后的 HEAD，本地新安装器队列已取消。历史打包测试不再计为
当前代码的修复完成项；现有 CI 仍须验证未来提交的工作树改动。
命令、退出码和实际插件/运行库 SHA 汇总在
[窗口验收记录](../../../../build_app/Testing/ocr-storage-reference/window-alignment-acceptance-20261019.json)。
本轮最终结果另见 [2026-10-20 验收记录](../../../../build_app/Testing/ocr-storage-reference/acceptance-20261020.json)。

状态更新：2026-10-20；逐文件证据索引的生成时间记录在 JSON 中。
原始报告、命令、输入来源和构建身份保存在 build_app/Testing/ocr-storage-reference/，
机器可读索引为 ocr-per-file-evidence.json；本索引只整理已完成报告，
不是自动发布资格判定器。逐项工程缺口见 [UPSTREAM_PARITY_AUDIT.md](UPSTREAM_PARITY_AUDIT.md)。

# OCR per-file evidence snapshot

Developer evidence index only: 48 text/layout/classification files plus 3 SLANet table files. No unexecuted test becomes PASS. Missing page gates and controlled performance remain open. Table TEDS requires its own task matrix and is not replaced by CER/WER. Existing v3 page identity has known vendored metadata overwrite; only fresh v4 receipts use the selected-distribution/helper digest identity. Neither is a complete release gate.

Model SHA is the exact file consumed. Storage percentages count parameters rather than file bytes; names do not imply pure quantization. P/F refer only to completed common-runner rows; page counts are completed / manifest required. H is a weight-matched historical PASS with incomplete build identity; it does not qualify a release and may overlap a newer P.

| Model | SHA-256 | Storage | Source graph CPU/CUDA/Vulkan | Pages CPU (diagnostic) | Pages CUDA | Pages Vulkan | Remote/catalog SHA | Release |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pplcnet-layout-f16.gguf | fda1343642abc04475271438f29b58036528be5f3f6a422c10ffdde3d7bcda1a | F16 100.000% | pending/F/F | 0/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | match | BLOCKED |
| pplcnet-layout-f32.gguf | 7e2785606f99bc6d1c7ffa6df6b15e42bd89da6c7a98ce7f1de1961c4cb2fcf1 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | match | BLOCKED |
| pplcnet-layout-q8_0.gguf | 18056b5559f3da420ab6cb5e3dd14a5c9f18f5fb8da91837ba8fa24288b07e42 | F16 10.996%; Q8_0 89.004% | pending/F/F | 0/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | 1/1 P; 0 F; 1 H | match | BLOCKED |
| ppocr-cls-f16.gguf | 323d117aeb7bd4340f09d92d0ba53675207e761dd3e2c4883a3e9cdc84b8e2eb | F32 7.960%; F16 92.040% | historical P; rebuilding evidence/P/P | 0/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | unpublished | BLOCKED |
| ppocr-cls-f32.gguf | 0ef8558a8de46bcb499f89c7313d7a53707061d452e5ebd2aa89512aa1b84e62 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | unpublished | BLOCKED |
| ppocr-cls-q8_0.gguf | 8f6e0fa64205da94f5fd8a70400f838b1cb0ccb836eaf44b279553a324d136b3 | F32 7.972%; F16 87.240%; Q8_0 4.788% | historical P; rebuilding evidence/P/P | 0/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | 18/18 P; 0 F; 18 H | unpublished | BLOCKED |
| ppocr-det-ch-f16.gguf | 82517e16120a186cc85123272286a5f35e9ede4b4d272fc2e7d5b7148879550d | F16 99.998%; F32 0.002% | historical P; rebuilding evidence/P/P | 0/9 P; 0 F; 5 H | 2/9 P; 0 F; 9 H | 2/9 P; 0 F; 9 H | match | BLOCKED |
| ppocr-det-ch-f32.gguf | aaad25f65055ecf0c34a9050b249ded1f099e8d91beb7ad4749b24e06b11b9fd | F32 100.000% | historical P; rebuilding evidence/P/P | 0/9 P; 0 F; 7 H | 2/9 P; 0 F; 9 H | 2/9 P; 0 F; 9 H | match | BLOCKED |
| ppocr-det-ch-q8_0.gguf | ede6dd9620f482224ec10cf9bfde58957eb8815c461b755173031779568ac207 | F16 33.798%; F32 0.002%; Q8_0 66.200% | historical P; rebuilding evidence/P/P | 0/9 P; 0 F; 2 H | 2/9 P; 0 F; 5 H | 2/9 P; 0 F; 5 H | match | BLOCKED |
| ppocr-rec-ch-f16.gguf | 2507cedcb718b3b98af896d2c633d42a162d3119ad6ea776de808c2dd0e4213c | F16 61.884%; F32 38.116% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 3 H | 2/6 P; 0 F; 6 H | 2/6 P; 0 F; 6 H | match | BLOCKED |
| ppocr-rec-ch-f32.gguf | a29b0d53026ab18e9d2a0e537eea90a140521b1eecfb8c0578bd1d6b4187e60a | F32 100.000% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 5 H | 2/6 P; 0 F; 6 H | 2/6 P; 0 F; 6 H | match | BLOCKED |
| ppocr-rec-ch-q8_0.gguf | 7d85184357ae6ffec60633b1db8c033000940f585852b4f0af3a2b8bcc34bae2 | F16 22.664%; Q8_0 39.220%; F32 38.116% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 2 H | 2/6 P; 0 F; 5 H | 2/6 P; 0 F; 5 H | match | BLOCKED |
| ppocr-rec-en-f16.gguf | f05e08cbc81cd9863cddfc17621e788eb896278e3a36448cccecd5829c5729bf | F16 87.263%; F32 12.737% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 2 H | 0/3 P; 0 F; 3 H | 0/3 P; 0 F; 3 H | match | BLOCKED |
| ppocr-rec-en-f32.gguf | f38155dc785cbfc23f5fe37af2d6403b42cfb8fb73af130699a425834c2441dc | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 2 H | 0/3 P; 0 F; 3 H | 0/3 P; 0 F; 3 H | match | BLOCKED |
| ppocr-rec-en-q8_0.gguf | d336e48f4b53b756058fd4015270a8ae6ed10d03810fa976a558c9fd98f5bb42 | F16 31.741%; Q8_0 55.522%; F32 12.737% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 0 H | 0/3 P; 0 F; 0 H | 0/3 P; 0 F; 0 H | match | BLOCKED |
| ppocrv3-japan-rec-f16.gguf | 4cf0971793cadfe242b7a370546de4886543d96262019395848244bf5b0f654e | F32 64.884%; F16 35.116% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv3-japan-rec-f32.gguf | ec3eeb364bbbeecd365010e0e61f9f6594dd84260296e3b43b645e922503fa5d | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv3-japan-rec-q8_0.gguf | 893a813c27382d81e98aafa309653e77a679cad1e75a4294d91b730569cc0865 | F32 45.475%; F16 53.218%; Q8_0 1.307% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv4-source-det-f16.gguf | 07000b35b34fb24755fdd91987e04a927e1357aebee28c8ca38a61809739b7f6 | F32 0.636%; F16 99.364% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | unpublished | BLOCKED |
| ppocrv4-source-det-f32.gguf | b96f01f423cebca891b095da11f24b69225257bddc17a86283e3707e9e9a17ab | F32 100.000% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | unpublished | BLOCKED |
| ppocrv4-source-det-q8_0.gguf | a62a8e2b86cb730bbf95e61071c14a8f148cc36f1b7b55f366b8baf36b4e327e | F32 0.648%; F16 23.799%; Q8_0 75.553% | historical P; rebuilding evidence/P/P | 0/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | 6/6 P; 0 F; 6 H | unpublished | BLOCKED |
| ppocrv4-source-rec-f16.gguf | b38a649949a551898da9f86ab88570b63bcb46401ae390a7ddb8fa0ce48e5c98 | F32 55.179%; F16 44.821% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv4-source-rec-f32.gguf | c9bd1ed2f82e0f01a50cbcae2eff871ca12856393d03b407df895fa4d2925d0f | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv4-source-rec-q8_0.gguf | 8ab6f58145f5af99887be7a34d2aaa9aad39a1bd348a93acc6eb95498f10fcfa | F32 55.179%; F16 41.610%; Q8_0 3.212% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-det-f16.gguf | f4c539a72c5b9f41e96d2f83d4ce510a814606f6c1b824c35df2ade24cb637a7 | F32 12.207%; F16 87.793% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-det-f32.gguf | d565d9fa39e8e32416a469938e0b5b7b3d251f4b22c4239c1ab9aee94197fb61 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-det-q8_0.gguf | 5400362f3062b1d6636b5638a8200490e398f016e449216fd68ee9c4e6681f43 | F32 12.207%; F16 76.388%; Q8_0 11.405% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-rec-f16.gguf | a5ce8f19f5983dea315056a8d6857cb15b26b47107bfad0752b027adab2330d4 | F16 21.232%; F32 78.768% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-rec-f32.gguf | 3e2cb8e43687a549acdf489b0abbeb07989751804d5dec6501fa508398d26b8d | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv5-mobile-rec-q8_0.gguf | 72d723b5c8e21ffd57fb2e91f97a8e3714ae29e9c332925236b2ad4493130c1b | F16 19.132%; F32 78.768%; Q8_0 2.101% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-det-f16.gguf | 57d3dd71018dd535bbdd3c52cf93d8699192662618be4933584da4ea6fddf98e | F32 0.954%; F16 99.046% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-det-f32.gguf | ce8f37dbbde68ba1ac8fca40a076757a676ef48cae07d1b423cf996a1c92e692 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-det-q8_0.gguf | 951d8011139fb2d3f0ee2d1982fef6c97acd78cba998065b453ce1e62610e8f3 | F32 0.954%; F16 27.938%; Q8_0 71.107% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-rec-f16.gguf | e6a40f225b5d15de9a6e06b4a16075d510df164c3e41fbbf824481c6079501ad | F16 76.289%; F32 23.711% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-rec-f32.gguf | 5f5b315e151f1fab29cf2c4d0ad0fd9aa0694ce23929a2076e41cea2ffb4bd8b | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-medium-rec-q8_0.gguf | 553d210c0f8d9765d04db7e0f0b7527ed5894502da1bc0e1827db86752513b0d | F16 7.977%; F32 23.711%; Q8_0 68.312% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-det-f16.gguf | 73efc76d8b6427363b19c5a88ecf7075febfd5bbbef0b5b7ff67ac4d8f5330c1 | F16 99.523%; F32 0.477% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-det-f32.gguf | 66a29730f76956678c5f2b8d72b7b79a5ac12a4d160d246a9a0f9eb3194445ce | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-det-q8_0.gguf | 8c6ee6c483d290ea534c1c8deb2b6f701ecdcfc5a337616408f9400a8d7787b1 | F16 13.563%; F32 0.507%; Q8_0 85.930% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-rec-f16.gguf | 48543319c045e7b4d05b2ed77008715bb750030bd95bbdc8ba8024c8871e40b0 | F16 52.341%; F32 47.659% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-rec-f32.gguf | 159df7215c842276cf288675b421d8a2d73c52b0816bdb724a862c04bf3f9a4e | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-small-rec-q8_0.gguf | 803a8053b03f1e7c7b5d1399909d8cb8bbf786613d24164f89add899371f17ec | F16 23.605%; F32 47.659%; Q8_0 28.736% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-det-f16.gguf | 804a564ad2db0fbd2b95325c167b854a4cba18c36889aab16e51c6ae2a7c0c83 | F16 98.740%; F32 1.260% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-det-f32.gguf | 9f0f264028796c6c61be5f79394367b0f9a2c5a46340b6c0041d0e8c81bfe5bb | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-det-q8_0.gguf | fa5f6f6efdc79c4a2363c7719d6dd5d7ca999e14cbc71132a65d30ce834c4988 | F16 98.501%; F32 1.260%; Q8_0 0.239% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-rec-f16.gguf | b5a7a723e105e19d7b01d1b0cc3656c4456931eba9e7e7363d85ef92a736fb18 | F32 84.468%; F16 15.532% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-rec-f32.gguf | 0e8848426d046549f926f057dd1d7ae587f50d8e327be1dbcce418dad3006406 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| ppocrv6-tiny-rec-q8_0.gguf | f9e6c3793c95fa7e84b10d5fa277f56e22b059c4e1517ba93a223f83fa41d960 | F32 84.468%; F16 14.953%; Q8_0 0.579% | historical P; rebuilding evidence/P/P | 0/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | 3/3 P; 0 F; 3 H | unpublished | BLOCKED |
| slanet-plus-table-f16.gguf | 0eb25bd9a67d9eafc830df5400dd4b8bef8a5a506c067ce384f925a3a3abdf63 | F16 100.000% | historical P; rebuilding evidence/P/P | 0/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | match | BLOCKED |
| slanet-plus-table-f32.gguf | c93a3a24e39c30bd67294a8abb543e3e0ed7ef1f2c6d67505349dee77aa87bf4 | F32 100.000% | historical P; rebuilding evidence/P/P | 0/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | match | BLOCKED |
| slanet-plus-table-q8_0.gguf | cf93b69e918d7202f9779ca376d25063d2fe15ffdd26ae209d4600ae7b56d1f0 | F16 35.537%; Q8_0 64.463% | historical P; rebuilding evidence/P/P | 0/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | 2/2 P; 0 F; 2 H | match | BLOCKED |

## Separate unpublished repair candidates

These are new file identities and do not inherit the preceding published files' results. Their unchanged source weights/dictionary and reproducible conversions do not qualify them for upload.

| Model | SHA-256 | Q8_0 parameters | Gate state |
| --- | --- | --- | --- |
| ppocr-det-ch-q8_0.gguf | cab74270c09bd584869840ac11e60ee9dafab7b43ed317922c1d2215ea9ae56d | 66.069% | cpu: graph pending; repair pages 0 P / 0 F; cuda: graph pending; repair pages 0 P / 0 F; vulkan: graph pending; repair pages 0 P / 0 F; controlled release A/B and cold-download pending |
| ppocr-rec-en-q8_0.gguf | 695bdcbbce6707fb388441ed651060b6175b2580be8860942c3766d7bbac5217 | 43.398% | cpu: graph pending; repair pages 0 P / 0 F; cuda: graph pending; repair pages 0 P / 0 F; vulkan: graph pending; repair pages 0 P / 0 F; controlled release A/B and cold-download pending |

Common runner report paths, hashes, exact input models and failure metrics are retained in [ocr-per-file-evidence.json](../../../../build_app/Testing/ocr-storage-reference/ocr-per-file-evidence.json).
All new uploads: 0. Dedicated OCR workspace implementation has begun; remaining OCR/VQA/Chat workflows, catalog and Linux clean-package are open.

## 2026-10-19 逐文件路径与实际阻塞

下表保留 2026-10-19 的路径及历史 developer 测量；原始 CUDA 参考重跑和发布状态以本文最新逐文件表及 2026-10-20 收据为准。

2026-10-19 本轮 API 快照读取 GitHub PP-OCR Release：16 个资产，其中 15 个 GGUF 和一个测试数据 ZIP；36 个新增候选和两个修复候选均未上传。
已把 38 个文件按原 SHA 复制到持久 staging，避免历史临时目录或实验目录被清理；复制不等于发布合格。

统一绝对目录：`/home/ludahai/develop/code/github/ACloudViewer/build_app/Testing/ocr-storage-reference/ocr-release-candidates-20261019`。该目录的 inventory.json 记录原路径、解析路径、字节数和 SHA；SHA256SUMS 覆盖全部 38 个文件。

| 模型 stem | staging 中的文件 | 数量 | 当前阻塞 |
| --- | --- | --- | --- |
| `ppocr-cls` | `ppocr-cls-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv3-japan-rec` | `ppocrv3-japan-rec-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv4-source-det` | `ppocrv4-source-det-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv4-source-rec` | `ppocrv4-source-rec-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv5-mobile-det` | `ppocrv5-mobile-det-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv5-mobile-rec` | `ppocrv5-mobile-rec-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv6-medium-det` | `ppocrv6-medium-det-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv6-medium-rec` | `ppocrv6-medium-rec-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv6-small-det` | `ppocrv6-small-det-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv6-small-rec` | `ppocrv6-small-rec-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv6-tiny-det` | `ppocrv6-tiny-det-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| `ppocrv6-tiny-rec` | `ppocrv6-tiny-rec-{f32,f16,q8_0}.gguf` | 3 | 源图三后端单轮已通过；CUDA/Vulkan 源页面各 54/54 已通过；受控 CUDA/Vulkan A/B 未齐；发布下载闭环待完成 |
| 公共 Q8 修复候选 | ppocr-det-ch-q8_0.gguf、ppocr-rec-en-q8_0.gguf | 2 | 新 SHA 的七组组合页面在 CPU/CUDA/Vulkan 均通过，源图亦已补验；受控性能与发布闭环待验；不得覆盖旧发布文件 |

具体故障与缺证据须分开：

- 公共布局 pplcnet-layout-f16/q8_0：原始 Paddle CUDA 真值的最差头 argmax 一致率为 0.992307692 / 0.969230769，要求 1；CUDA/Vulkan 均失败，后端间一致仍通过。这是精度阻塞。原生 CPU 仅作诊断；CDLA 独立人工 GT 已固定并完成两 GPU 三种存储的 1,000 页测量，公共发布消费者仍待接入。
- 原公共文本 Q8：中/英文街景 e2e CER 增量 0.009708738 / 0.005295675，大于 0.005；英文收据 fixed-region WER 增量 0.006833713。修复文件必须独立验收。
- 性能：受控 baseline/candidate 被 GPU idle audit 拒绝，尚无有效完整吞吐或采样显存 A/B；不能据此宣称 CUDA/Vulkan 速度不达标。除桌面活动，外部 dynamic_svm_online_calib_node 计算进程也反复出现；13:28 的性能排空尝试已中止并留证。14:18 再查时外部 semantic_segmentation_eval_probe 已返回，因此未启动新 A/B，未停止外部任务或放宽门限。
- 覆盖：36 个新候选各有 F32/F16/Q8_0，不是只生成 F32；Q8 是混合存储，Tiny det/rec 的真实 Q8 参数占比仅 0.239% / 0.579%，不能宣称纯 Q8 或推断加速。
- CPU 公共页面旧队列在第 20/28 项被中断且无最终 matrix；保留其完整日志，未把进程退出前结果升级为完整通过。改以公共 runner 的逐场景矩阵恢复。
- 发布 SHA/catalog/manifest、正式 URL 冷缓存下载仍未闭环；已有 SHA 和 local-only 验证场景不等于下载支持。先通过数值/页面/性能门，再上传和复核下载完整性，不把尚未上传本身作为数值验收失败。

机器可读证据是 `build_app/Testing/ocr-storage-reference/ocr-per-file-evidence.json`。

## 2026-10-20 独立 CDLA 页面评价接线

`core/AICore/tools/ocr_cdla_validation.json` 固定官方验证集全部 1,000 页的
图像/标注 SHA、原始归档大小/SHA 和来源版本，不使用训练集或调参挑选页面。
实际执行 CDLA 的官方 LabelMe→COCO 转换与原始 LabelMe 栅格化逻辑，8,711 个
区域的类别、bbox 和 area 全部相同；COCO annotation ID 改为正整数以避开
COCOeval 的零值未匹配哨兵，不改几何。评价使用 class-aware bbox AP50:95、
AP50/AP75、AR100，完整逐页覆盖包含零预测页面。原始 Paddle 后处理保留
score=0.5、NMS=0.5，有限的越界浮点框按 COCO 原语义评价，不额外裁切。

新增评估器/参考导出器/公开 C ABI 区域导出器；17 项手动单元测试通过，
已登记正式 CMake 测试，尚未为其重新配置活动 GPU 队列的构建树。原始 CPU
图的 1,000 页及原生 CPU 的三页接线诊断正常退出；CPU 指标不是 GPU 门控。
公共下载器已从空缓存下载完整官方归档，验证 2,000 个固定文件；离线复用
零下载且全部通过。独立原始 CUDA 及 CUDA/Vulkan × F32/F16/Q8_0 页面导出在已有
GPU 队列之后串行运行，不并发抢占性能测试。证据入口为
`layout-cdla-source-path.txt`；未完成的行不计 PASS，也未放宽严格原始输出门。
页面指标的发布容差与公共 manifest 接入仍待独立审核，不以诊断 AP 反推门限。


## 2026-10-20 GPU 参考选择复核

确认旧图探针仍有额外的原始 CPU 数值门控：先把 CPU 数组交给原生探针，
随后原始 CUDA 输出匹配也不能撤销前面的失败。隔离修复先选择原始 CUDA
参考，再运行原生 GPU；ONNX 原始图和 GGUF 存储权重重建图都使用 CUDA，
额外图像、方向分类及其后的 CTC 同样保留 CUDA 参考。CPU 固定样本继续用于
来源/身份核对，不再提供 GPU 数值合格结论。所有概率、存储权重误差和严格
argmax 门限保持不变，CUDA provider 回退会失败。

37 项隔离探针测试通过，另有原始/存储权重图和方向分类/CTC 的 CPU 接线检查。
生产文件仍等待既有冻结队列结束后切换，随后串行重跑公共 GPU consumers；
这些准备结果不代表 GPU 门控通过。收据：`gpu-truth-selection-path.txt`。
已有布局 F16/Q8 对原始 CUDA 的严格 argmax 失败仍然有效，尚无新增合格上传。


页面参考也存在相同限制：当前旧队列的人工 GT 指标有效，但上游比较分支是
CPU，不能据此声称已完成原始 CUDA 页面差值门控。隔离修复已为 66 个 ONNX
页面流程、21 个官方 Paddle 页面流程和 18 个方向流程传入明确 CUDA 参考。
Paddle GPU 行拒绝重放 CPU 导出，关闭 TF32；ONNX 核验 CUDA provider，正常
子进程退出后才记录成功来源。人工 GT、Hmean/CER/WER 阈值均保持原值。
167 项隔离探针/runner 测试通过；尚待真实 GPU 重跑，不产生发布资格。

CUDA 参考源修复已在旧冻结队列结束后应用到当前工作树，七组生产脚本回归
共 184 项通过（含 CDLA 17 项）。原生运行库和探针二进制未变；原始 CUDA
全图/方向/页面复核等待当前 CDLA GPU 导出完成，不能把单元测试转记成 GPU PASS。


## 2026-10-20 完整 CDLA GPU 页面结果

原始 Paddle CUDA 与六份 CUDA/Vulkan × F32/F16/Q8_0 导出均正常退出，
每项评价官方验证集全部 1,000 页、8,711 个人工区域。正式 CDLA CTest 也已通过。

| 执行 | mAP50:95 | 对原始 CUDA 差值 |
| --- | --- | --- |
| 原始 Paddle CUDA | 0.8546702421 | 0 |
| 原生 CUDA F32 | 0.8546698470 | -0.0000003951 |
| 原生 Vulkan F32 | 0.8546698405 | -0.0000004017 |
| 原生 CUDA/Vulkan F16 | 0.8547334403 | +0.0000631982 |
| 原生 CUDA/Vulkan Q8_0 | 0.8537569219 | -0.0009133202 |

两后端 F16/Q8_0 的上述页面指标相同；该结果仅补齐独立人工页面测量，
严格逐输出 argmax 失败、未审核的页面发布容差、公共 manifest 消费者和
受控性能仍各自阻止发布。原始 CUDA 参考选择修复已投入公共全图/方向/页面
串行重跑，不能将排队行记为通过。新公共 Q8 修复按新 SHA 单独排队，离线
输入预检已通过。详见 `layout-cdla-source-path.txt`、`gpu-truth-selection-path.txt`
和 `public-q8-gpu-truth-path.txt` 所指收据；当前仍无新增上传。


## 2026-10-20 原始 CUDA 图/方向复核与性能复测安排

原始 CUDA 参考的 132 个 graph/orientation 公共 runner 行已全部完成，128 PASS、
4 FAIL。36 个 ONNX 图/18 个方向流程在两后端全部通过；每后端 Paddle 的 12 行
中仅 layout F16/Q8_0 失败。逐文件表已更新，表中进行中的页面结果仍是部分行，
不能记为整个 cohort 或发布通过。

对八头留存张量的只读对比确认，CUDA/Vulkan 的所有 argmax 相同；相对原图的
失败位置也完全一致：F16 类别 22、DFL 边 40，Q8_0 类别 253、边 419。
两后端张量最大差异小于 0.000021；支持先排查权重转换/精度策略，未取消严格
原图门，也未将单样本混合精度拟合当作修复。证据为
`build_app/Testing/ocr-storage-reference/layout-backend-attribution.json`。

受控复测已通过八组离线公共 runner 预检，并串行排在当前页面与新 SHA 的 Q8
修复队列之后：CUDA/Vulkan 分别先运行方向分类 F32 哨兵，满足既有 release 门后
再运行 36 个 ONNX 图、12 个 Paddle 图及两份独立 SHA 的公共 Q8 修复图。
保持原始 CUDA 真值、旧版同机运行库
baseline、3 次进程重复、2 次预热、10 次计时及 120 秒空闲审核；哨兵失败会保留
日志并停止该后端的完整性能组，不放宽审核。现无匹配的旧版 table probe，SLANet
受控 A/B 仍未接齐。排队和预检都不产生性能 PASS 或发布资格。收据入口
`gpu-controlled-retry-path.txt`；新增合格/上传数量仍为 0。

只读布局定位补充：当前保留输入中的 F16/Q8_0 argmax 翻转都在 score .5 以下，
但分别有 1/2 个锚点跨越 .5，不能声称任务输出完全一致。该单输入置信度分析
仅帮助定位存储精度/阈值敏感性，不改变原始 probability/argmax、人工页面、
受控 CUDA/Vulkan 性能或发布门。记录在
`build_app/Testing/ocr-storage-reference/layout-decision-confidence.json`。
