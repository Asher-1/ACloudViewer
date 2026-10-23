# OCR per-file release gates

Latest update: 2026-10-23 (evening). The CUDA detector source-speed recheck now
passes for published F16 and repaired Q8 after exact-gather scheduling and copy
elision. The Vulkan sibling scheduling patch preserves outputs bit-for-bit
(15/15 model/batch rows), but the Vulkan detector release-profile recheck still
fails the source-speed gate (p50 12.1 ms vs Paddle CUDA ~6 ms); Vulkan
qualification and complete exact-file page/stability receipts remain open.
These graph results alone do not qualify a release upload.
The release decision follows task-level page accuracy
and controlled timing against the official implementation. Text-graph quantized
raw-output differences are diagnostic; the separately requested layout gate
still requires 100% original class-head and per-edge DFL argmax agreement.
The published Q8 files still
exceed the declared page-error budget, but they are different SHA identities
from two new Q8 repair candidates. Those repair candidates now pass the selected
CUDA and Vulkan developer accuracy matrix (nine rows per backend, one repeat).
This is not release qualification: controlled official-speed/VRAM and stability
gates, the complete per-file matrix, production catalog/digest registration, and
cold-download verification remain open. Newly qualified uploads: **0**.

This is a local worktree evidence ledger. Results belong to exact model, input,
script and build identities. The 51 main files comprise 36 extension candidates,
12 published text/layout files and 3 published SLANet files. Two Q8 repair
candidates have separate SHA identities and never inherit the published files'
results. The 2026-10-20 GitHub API snapshot found 16 assets under `PP-OCR`
(15 GGUF files and one ZIP); push/admin access was available. Upload permission
is not the blocker.

The merged runtime SHA was
`9f2548ef28f6edadbcdbb0658d6fbf67e18dbd73ca2afb28183c2b2e9af59f21`.
Its 96 graph checks produced 46 PASS / 2 FAIL per GPU, with layout F16/Q8_0 the
only failures. Its 120 page/SLANet checks produced Vulkan 60/60 PASS and CUDA
59/60 PASS initially; the CUDA row rejected for a changed CMakeCache passed an
independent retry under the original conditions. Keep the first failure receipt.
These pages used original CPU references and remain developer evidence. They do
not establish original-CUDA page parity or release stability/performance.
See `merged-gpu-pages-20261020-v5k89ifx/` and `merged-gpu-followup-path.txt`.

The older runtime `154a0b62` completed 54/54 extension page scenarios on each GPU;
its CPU 54/54 is diagnostic only. CPU is not a release gate. Initial skips,
timeouts and interrupted runs remain INCOMPLETE even when a retry succeeds.
The 36 new files have single-run source probability/argmax evidence on each GPU.
Original-CUDA graph/orientation reruns completed 132 rows: 128 PASS / 4 FAIL,
with layout F16/Q8_0 failing on both GPUs. Two distinct public-Q8 repair SHAs
subsequently passed source graphs and combined page scenarios. The newer
persistent report at `public-q8-gpu-truth-vpx8osmk/{cuda,vulkan}/matrix.json`
records nine selected accuracy scenarios per backend: five public page sets,
two source graphs, and two held-out receipt sets. Both reports are developer
profile, one repeat, and have no controlled latency or VRAM claim.

Published text files retained 24 PASS / 4 FAIL per GPU. Layout F16/Q8_0 retained
their strict source failures. CDLA's full 1,000-page validation set and 8,711
human regions were independently pinned and evaluated; page tolerance approval
and public consumers remain open. SLANet original graphs passed 6/6 GPU rows;
the older crop-table/combined-page matrix passed 12/12 GPU rows, with six extra
CPU diagnostic rows. F32 crop TEDS/TEDS-S was 0.9645975/0.9959180 and combined-page
TEDS/TEDS-S was 0.5431670/0.5917242. Source agreement does not mean absolute quality
is 1. Table TEDS is not replaced by text CER/WER.

Controlled same-host baseline/candidate/source-CUDA A/B is incomplete.
`gpu-release-20261020-py6cqztd/{cuda,vulkan}/matrix.json` records three attempts per
backend rejected before timing because Xorg/desktop GPU activity exceeded the
idle gate. Extending idle waits to 120 seconds did not produce a qualified CUDA
baseline/candidate sample. Two rejection receipts were retained and remaining
waits cancelled so numerical work could continue; that later Vulkan retry did
not run. See `controlled-release-cancelled-after-audit.json` under the directory
identified by `merged-gpu-followup-path.txt`. Activity rejection is not evidence
that model inference is too slow. No external process was stopped and no audit
threshold was relaxed.

Original CUDA reference comparisons are independent of performance audits:
36/36 new files passed per GPU; public Paddle graphs were 10 PASS / 2 FAIL per
GPU. Worst layout-head argmax agreement was 0.992307692 for F16 and 0.969230769
for Q8_0. Initial interpreter-resolution and missing eight-head export failures
occurred before numeric comparison, were retained, and were rerun after repair.

OCR table editing, basic rich text, and Chat/VQA configuration/message workflows
have native/source-method evidence. Table content comparisons cover 5,407 cases,
six geometry groups, six viewport-height groups and two theme QSS groups.
Rich text covers 13 serializations, six heading metrics, 12 paste classes, three
crop previews, six height cases, light/dark styles, heading popup, source-drawn
icons and image selection/scroll/delete behavior. Four plugin-entry/video groups
and five Chat/VQA source-method groups passed. Mini-probes do not replace a
formal plugin rebuild, full-window visual comparison or model gates.
Local VLM, remaining exact catalog entries and Linux clean-package remain open.

Remote focal package generation, installation and headless/JSON-RPC tests had
succeeded. A stale local installation and additional local conditions did not
prove a remote packaging defect. Packaging code, toggles and build-document
changes were restored to the synchronized HEAD; the local installer queue was
cancelled. Historical packaging tests do not certify the restored code or future
uncommitted changes. See the CI reassessment in
[UPSTREAM_PARITY_AUDIT.md](UPSTREAM_PARITY_AUDIT.md),
[window acceptance](../../../../build_app/Testing/ocr-storage-reference/window-alignment-acceptance-20261019.json)
and [2026-10-20 acceptance](../../../../build_app/Testing/ocr-storage-reference/acceptance-20261020.json).

Raw reports, commands, input provenance and build identities are under
`build_app/Testing/ocr-storage-reference/`. `ocr-per-file-evidence.json` indexes
completed reports; it is not an automatic release qualification engine.
The following per-file index is historical developer evidence, not a claim that
the latest candidate has passed release gates.

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
| ppocr-det-ch-q8_0.gguf | cab74270c09bd584869840ac11e60ee9dafab7b43ed317922c1d2215ea9ae56d | 66.069% | CUDA/Vulkan source graph and nine-scenario developer accuracy matrix PASS (one repeat); controlled official A/B, VRAM, stability, complete per-file matrix, catalog/digest and cold-download pending |
| ppocr-rec-en-q8_0.gguf | 695bdcbbce6707fb388441ed651060b6175b2580be8860942c3766d7bbac5217 | 43.398% | CUDA/Vulkan source graph and nine-scenario developer accuracy matrix PASS (one repeat); controlled official A/B, VRAM, stability, complete per-file matrix, catalog/digest and cold-download pending |

Common runner report paths, hashes, exact input models and failure metrics are retained in [ocr-per-file-evidence.json](../../../../build_app/Testing/ocr-storage-reference/ocr-per-file-evidence.json).
All new uploads: 0. Dedicated OCR workspace implementation has begun; remaining OCR/VQA/Chat workflows, catalog and Linux clean-package are open.

## 2026-10-19: Persistent paths and actual blockers

The 38 candidates were copied without changing their SHAs into:

```text
/home/ludahai/develop/code/github/ACloudViewer/build_app/Testing/ocr-storage-reference/ocr-release-candidates-20261019
```

`inventory.json` records original/resolved paths, bytes, SHA and actual storage
composition; `SHA256SUMS` covers all 38 files. Copying into staging is not release
qualification. Each of the following 12 stems has `{f32,f16,q8_0}.gguf` files:

| Stem | Files | Historical numerical evidence | Remaining release work |
| --- | ---: | --- | --- |
| `ppocr-cls` | 3 | Source graphs and associated extension page matrix | Controlled CUDA/Vulkan A/B and publication/download verification |
| `ppocrv3-japan-rec` | 3 | Same matrix; file-specific consumers in the evidence index | Same |
| `ppocrv4-source-det` | 3 | Same | Same |
| `ppocrv4-source-rec` | 3 | Same | Same |
| `ppocrv5-mobile-det` | 3 | Same | Same |
| `ppocrv5-mobile-rec` | 3 | Same | Same |
| `ppocrv6-medium-det` | 3 | Same | Same |
| `ppocrv6-medium-rec` | 3 | Same | Same |
| `ppocrv6-small-det` | 3 | Same | Same |
| `ppocrv6-small-rec` | 3 | Same | Same |
| `ppocrv6-tiny-det` | 3 | Same | Same |
| `ppocrv6-tiny-rec` | 3 | Same | Same |
| Separate public Q8 repairs | 2 | Seven combined page scenarios plus source graphs, recorded by new SHA | Controlled A/B and publication/download verification; never overwrite old evidence |

Failures and missing evidence are distinct:

- Published layout F16/Q8_0 failed original Paddle CUDA argmax, while backend
  agreement passed. CDLA page measurements do not waive this accuracy failure.
- Published text Q8 Chinese/English scene e2e CER increased by 0.009708738 /
  0.005295675, exceeding 0.005; English receipt fixed-region WER increased by
  0.006833713. Repair SHAs require independent evaluation.
- GPU idle audits rejected A/B before valid measurement. Desktop activity and
  external `dynamic_svm_online_calib_node` / `semantic_segmentation_eval_probe`
  workloads were recorded, not stopped. The 13:28 drain attempt was cancelled;
  another external workload at 14:18 prevented a new A/B start.
- All 36 extension files include F32/F16/Q8_0 variants. Q8 files are mixed
  storage: Tiny detector/recognizer contain only 0.239% / 0.579% Q8 parameters.
  Names do not establish pure quantization or speedup.
- A historical public-CPU page queue stopped at 20/28 without a final matrix.
  Its log is retained; per-scenario runner retries do not retroactively pass it.
- Upload, release SHA, catalog, manifest and cold-cache download remain separate
  steps after numerical/page/performance qualification. Being unpublished is
  not itself a numerical test failure.

## 2026-10-20: Independent CDLA evaluation

`core/AICore/tools/ocr_cdla_validation.json` pins all 1,000 official validation
images/annotations, the archive's size/SHA and source revision. No training pages
or prediction-derived labels are used for validation. Executing the official
LabelMe-to-COCO converter and rasterizer reproduced categories, bbox and area for
all 8,711 regions. Positive COCO annotation IDs avoid the zero unmatched sentinel
without changing geometry. Evaluation is class-aware bbox AP50:95/AP50/AP75 and
AR100, including zero-prediction pages. Original Paddle postprocessing keeps
score=0.5 and NMS=0.5; finite out-of-bounds boxes follow COCO semantics.

The evaluator, source exporter and public typed-C-ABI region exporter have 17
unit checks and CTest registration. Initial 1,000-page source-CPU and three-page
native-CPU runs were plumbing diagnostics. Common-runner cold-cache download
verified 2,000 consumed files; offline reuse performed zero downloads.
See `layout-cdla-source-path.txt`.

Original CUDA and all six native backend/dtype exports later completed all
1,000 pages normally; the formal CDLA CTest passed:

| Execution | mAP50:95 | Delta from original CUDA |
| --- | ---: | ---: |
| Original Paddle CUDA | 0.8546702421 | 0 |
| Native CUDA F32 | 0.8546698470 | -0.0000003951 |
| Native Vulkan F32 | 0.8546698405 | -0.0000004017 |
| Native CUDA/Vulkan F16 | 0.8547334403 | +0.0000631982 |
| Native CUDA/Vulkan Q8_0 | 0.8537569219 | -0.0009133202 |

Page release tolerance and public manifest consumers still require review.
Strict original-head failure and controlled performance remain independent
blockers. Do not infer a release tolerance from these observed AP deltas.

## 2026-10-20: Original CUDA reference selection

Older GPU graph probes first applied an extra original-CPU numeric gate, so a
subsequent CUDA match could not undo that failure. The corrected probes select
original CUDA before native GPU execution. Both original ONNX and reconstructed
stored-weight graphs use CUDA; additional images, orientation and CTC retain
CUDA truth. CPU fixtures verify provenance, not GPU numeric qualification.
CUDA-provider fallback fails. Probability, stored-weight and strict argmax
thresholds were retained. The isolated graph checks passed 37 tests.

Pages had the same issue. Explicit CUDA references were added to 66 ONNX page,
21 Paddle page and 18 orientation consumers. Paddle rejects replay of CPU
exports and disables TF32; ONNX verifies its actual CUDA provider. Source success
is written only after a normal subprocess exit. Human GT and Hmean/CER/WER
thresholds were unchanged. Isolated tests passed 167; after the frozen queue
ended, seven production-script groups passed 184 tests including 17 CDLA tests.
Native runtime/probe binaries were unchanged. See `gpu-truth-selection-path.txt`.

The resulting full graph/orientation run was 128 PASS / 4 FAIL out of 132:
36 ONNX graphs and 18 orientation flows passed on each GPU; only layout F16/Q8_0
failed among the 12 Paddle rows per GPU. Read-only saved-tensor comparison found
identical CUDA/Vulkan failure positions: F16 22 class / 40 DFL-edge flips and
Q8_0 253 / 419, with backend max difference below 0.000021. See
`layout-backend-attribution.json` and `layout-decision-confidence.json` in the
raw-report directory. Although flipped class winners were below score .5,
1 F16 and 2 Q8 anchors crossed .5. This does not prove identical final boxes.

Eight offline controlled-retry preflights passed. Planned release sampling was
three processes, two warmups, ten timed forwards and a 120-second idle audit,
with original CUDA truth and the frozen same-host older runtime. A failed
classifier sentinel stops that backend's full performance cohort. Queued work
and preflight never count as PASS. See `gpu-controlled-retry-path.txt` and
`public-q8-gpu-truth-path.txt`.

## 2026-10-20: Stored-weight attribution and controlled sentinels

`export_paddle_ocr_stored_weights.py` checks names, shapes, order, original SHA
and serialized payload, replacing only an independent Paddle parameter copy with
actual dequantized GGUF values. F32 round-trips byte-for-byte; all 486 parameters
were reconstructed for each dtype, with six tests. The unchanged Paddle CPU
graph then matched every native CUDA/Vulkan head's argmax; max differences were
below 0.000025 / 0.000041. This is storage-attribution evidence, not a GPU release
gate. Original F32 truth and strict failures remain. See `layout-stored-paddle-path.txt`.

Classifier F32 controlled sentinels kept the original audit and sampling rules.
Desktop activity rejected both baseline and candidate; active probes exited
before remaining attempts were cancelled with -15. Neither cohort has a complete
final matrix. The page scheduler was restored. See `gpu-controlled-first-path.txt`.
Pausing a scheduler during one CUDA XFUND F16 scenario caused outer timeout 124;
its partial output is not success. The original-condition retry was queued under
`gpu-page-pause-retry-path.txt`. Do not pause a runner during a timed scenario.
`gpu-controlled-after-pages-path.txt` records a later serial retry plan, not PASS.

The 2026-10-22 policy required **100% argmax agreement for every original
head**, including low-confidence classes and intermediate DFL distributions.
On 2026-10-23 the user clarified that release acceptance should follow task-level
accuracy and inference time relative to the official implementation. Keep raw
probability, argmax and DFL differences as diagnostics, and use independently
labelled pages plus the predeclared metric budgets for acceptance.
`layout-quantization-review.json` also records a Q8 Equation-class AP drop of
0.359033 percentage points; overall mAP alone omits that difference.

The supposed older `test_table_runtime` was a symlink to the current build, not
a historical baseline. A common new public-C-ABI `bench_table` was instead run
against independently retained old/current `libAICore.so` files. Its two CPU
plumbing checks verified actual library paths, stable output and 24 tokens;
they do not establish GPU qualification or make the new probe historical.

`run_table_capi_probe.py` checks three lossless PNGs and exact original NCHW
preprocessing, independent Paddle CUDA, strict GPU audit, repeated tokens,
truthful timings, source SHAs and both sides' sampled VRAM. Its
`decoded_token_match` is explicitly not all-raw-head argmax. Quantiles use raw
samples consistently. Seven rejection tests and two offline preflights passed;
the six backend/dtype release cohorts remained queued. See `table-capi-baseline-path.txt`.

### Layout accuracy root cause and remaining evidence

`layout-attribution-reverified.json` rechecked 27 identities and recomputed both
comparisons for the fixed public `layout_input.npy`:

| Storage | Worst head vs original Paddle CUDA | Class / DFL-edge flips | All-head agreement vs stored-weight Paddle CPU |
| --- | ---: | ---: | ---: |
| F32 | 1.0 | 0 / 0 | 1.0 |
| F16 | 0.9923076923 | 22 / 40 | 1.0 |
| Q8_0 | 0.9692307692 | 253 / 419 | 1.0 |

Counts were independently computed for each GPU and were identical. The original
486-parameter Paddle graph reproduces native decisions after replacing weights
with actual storage values. This attributes these fixed-input failures to weight
rounding and propagation, not an unknown decoder problem. It is not a proof for
all OCR models, inputs or conversion policies, and CPU remains diagnostic.

The Q8 file has 59 actual Q8_0 tensors and 427 F16 tensors. Separate interventions
kept all other parameters original F32:

| Replaced parameter group | Tensors | Class flips | DFL-edge flips |
| --- | ---: | ---: | ---: |
| Actual Q8_0 tensors only | 59 | 244 | 411 |
| F16 fallback tensors only | 427 | 22 | 48 |

Both groups independently cause flips. Nonlinear propagation means their counts
cannot be added. See `layout-storage-factorial-path.txt` / `factorial-attribution.json`.
Small probability errors, backend agreement and similar mAP do not imply every
winner is unchanged. Do not rename F32 as F16/Q8, fit a single validation input,
replace original truth with stored-weight truth, or hide low-confidence flips.

### Quantization budgets: head-level diagnostics and task acceptance

For a vector with original top-two gap `margin` and component error bounded by
`epsilon`, `margin > 2 * epsilon` guarantees the winner is unchanged. Near ties
can flip under tiny rounding. Recovering every winner on a finite acceptance set
does not guarantee all future inputs under lossy storage. DFL boxes use weighted
softmax distances in `ocr_capi.cpp:gflDecode`; bin argmax is not box accuracy.

A read-only review of 24 report/output identities reproduced all original flips;
every flipped vector failed this sufficient stability condition. At score .5,
1 F16 / 2 Q8 candidates crossed the threshold. These are fixed-input anchors,
not final NMS boxes or page acceptance. See `layout-margin-review-jj9e632k/report.json`.

| Acceptance layer | Mandatory evidence | Optimization boundary |
| --- | --- | --- |
| Independent original framework | Per-head probability/logit errors, argmax, DFL and per-class results | Diagnostic evidence; head-level mismatch alone does not reject a model |
| Added backend error | Independent stored-weight graph vs CUDA/Vulkan | Separate from weight rounding and task-level page accuracy |
| Real pages | Hmean/mAP, per-class degradation, CER/WER and table structure | Current page manifests allow up to 0.005 degradation per declared metric; include independent held-out inputs |
| Performance/release | Controlled same-host latency/throughput/VRAM against official source, stability, exact SHA and download | Reject a timing row only when it exceeds both the 5% relative and 3 ms absolute noise thresholds; require a clean GPU audit |

Sensitive tensors/channels, explicit F32/F16 exceptions, calibration and offline
QAT can be investigated on independent training inputs. Retaining only final
heads cannot guarantee exactness when the backbone/F16 fallback also changes.
Each candidate records actual tensor dtypes, size and SHA before independent
page/backend checks. The current user acceptance supersedes the previous raw-head
100% veto. Keep the automated page-budget result visible; a close task-level
result may be accepted under the user's direction with its exact deviation
recorded. Current Q8 page accuracy is close by that criterion, but release is
still pending qualified official timing and the remaining per-file provenance,
SHA and download checks. Raw-head mismatches alone do not block it.

### 2026-10-21: Repair SHAs, precision policies and SLANet queue

The two public Q8 repairs passed two original Paddle-CUDA graphs and seven shared
combined-page scenarios on each GPU, including scenes, SROIE, XFUND and held-out
SROIE 050–099. Old published-SHA failures remain.

| F16 role-based diagnostic policy | F32 parameters | Class flips | DFL-edge flips | Verdict |
| --- | ---: | ---: | ---: | --- |
| Norm and bias F32 | 2.867968% | 13 | 16 | FAIL |
| Also retain four final convolutions F32 | 4.049439% | 10 | 15 | FAIL |

These are fixed-input CPU diagnostics, not page/GPU acceptance. The SLANet queue
had waited for an OCR worker that had already exited after a build-identity
change. Its ERROR/INCOMPLETE receipt was completed without rewriting the error.
After frozen inputs were reverified, four CUDA idle audits each waited 120s and
rejected before sampling. Minimum GPU utilization was 11%, 15%, 13%, 12% against
the 10% limit. Only this queue's redundant waits were stopped. Vulkan did not run
and is neither PASS nor FAIL. Overall status remains INCOMPLETE. See
[follow-up evidence](../../../../build_app/Testing/ocr-storage-reference/ocr-release-followup-20261021.json).

### 2026-10-21: llama CUDA migration

Isolated probes were rebuilt against installed llama/ggml 0.27 headers/libraries
without reconfiguring or replacing `build_app`. Loader traces confirmed only new
base/CPU/CUDA modules, with no old 0.21 contamination.

| Scope | Exact model SHAs | Input checks | Elementwise equality to retained pre-migration native output |
| --- | ---: | ---: | ---: |
| Public Paddle text/layout plus two Q8 repairs | 14 | 14 | 14/14 |
| 12 extension models, three dtypes | 36 | 135 | 135/135 |
| SLANet, three dtypes | 3 | 9 | 9/9 |
| Total | 53 | 158 | 158/158 |

Independent original-CUDA arrays covered 14 Paddle, nine SLANet and 36 of the
135 ONNX checks. The other 99 checks replayed native output, not fresh framework
truth. Pages and controlled performance were not rerun. Original layout flips
remain 22/40 F16 and 253/419 Q8. See `build_llama_ocr_cuda_2m6kcjxu/` and
[resource migration evidence](../../../../3rdparty/llama/validation/resource-migration-evidence.json).

### 2026-10-21: Ordered Vulkan patches and installed libraries

Thirty-three source/shader files were ported to embedded ggml 0.27 and registered
in llama's manifest. A real `ExternalProject_Add(ext_llama)` extraction/patch/
build/install produced llama, mtmd and one embedded ggml. Three CTests passed,
including actual CPU/Vulkan ops and 11 CMake contracts. Reapplying patches was
idempotent; 225 Vulkan files matched the reviewed tree. Loader traces verified
the new installation without old or scratch modules.

| Vulkan scope | Models / inputs | Max probability delta vs old native output | Added applicable-head flips |
| --- | ---: | ---: | ---: |
| Public Paddle plus Q8 repairs | 14 / 14 | 1.47820e-5 | 0 |
| Extension ONNX, three dtypes | 36 / 135 | 1.49608e-5 | 0 |
| SLANet, three dtypes | 3 / 9 | 2.29537e-4 | 0 |

SLANet location max delta was 2.84731e-4. Original-CUDA coverage matches the
previous section; frameworks were not rerun here. See
[Vulkan evidence](../../../../3rdparty/llama/validation/vulkan-rebase-evidence.json).
A combined CUDA/Vulkan install then shared its base runtime, passed four entry
CTests, relinked monolithic AICore using 240 retained candidate objects and
passed 36 public-contract tests without skips. This is link/execution evidence,
not a full application rebuild or all-task acceptance.

Both GPUs reran all 158 checks. An initial attempt selected old CUDA 11.1 and
skipped CUDA execution; it was retained. A separate correct CUDA-11.8 run checked
all native return codes were zero. See
[dual-GPU evidence](../../../../3rdparty/llama/validation/dual-gpu-evidence.json).
Metal, remaining task regression and controlled performance were not completed;
the final legacy production entry was retained.

### 2026-10-21: Rounding calibration

The unchanged Paddle graph with actual GGUF weights reproduced 62 F16 and 672
Q8 original-head flips on CPU. The corresponding GPU migration added no applicable
head flips. This isolates weight storage/propagation for this input, not decoding.

Eight independent official training images supplied 64 layer activations with
damping 0.01. F16 activation feedback without structural exceptions produced
18/44 class/DFL flips and was worse than nearest rounding. Keeping one-dimensional
structural parameters F32 gave nearest 13/16 and feedback 7/12; Q8 improved from
243/413 to 122/226. Training-error selection of 1/2/4/8/16 F32 exceptions still
left best F16 7/11 and Q8 100/201. None reached strict 100%.

`ocr_q8_calibration.py` and `test_ocr_f16_calibration.py` are offline tools, not
default conversion policy. Three F16 tests passed using:

```bash
OPENBLAS_NUM_THREADS=2 python3 -m unittest discover -s core/AICore/tools -p test_ocr_f16_calibration.py -v
```

Receipts are under `build_llama_dual_gpu_3nl01j4j/ocr-calibration/`;
`promoted-v2-results.json` is authoritative for the exception experiment. The
first Q8 attempt used a different parent-selection rule and was retained;
v2 read the actual parent parameter bytes. Do not merge those attempts.
[PaddleOCR compression](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/slim/auto_compression/README.md)
and [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html) motivate experiments,
not identical raw-head guarantees or current-model release claims.

### 2026-10-21: Serialized candidates and in-domain calibration

Six policies were written into actual GGUFs and all 486 decoded parameters
matched their Paddle copies. Fresh original-F32 CUDA, stored-weight CUDA and both
native GPUs reproduced the remaining failures. An affine-compensated F16 policy
still had 8/8 flips.

A separate calibration used the first 16 fixed-order **train** images from CDLA,
with archive CRC, file and original-model SHAs checked. No validation images,
labels or outputs were fitted. The same 64 activations/damping were retained.

| Stored policy | CUDA class / DFL flips vs original | Vulkan class / DFL flips vs original | Both GPUs vs stored Paddle |
| --- | ---: | ---: | ---: |
| Original F32 | 0 / 0 | 0 / 0 | 0 / 0 |
| F16 nearest, structural F32 | 13 / 16 | 13 / 16 | 0 / 0 |
| F16 feedback | 5 / 15 | 5 / 15 | 0 / 0 |
| F16 feedback, two training-selected exceptions | 5 / 15 | 5 / 15 | 0 / 0 |
| F16 feedback, affine compensation | 6 / 14 | 6 / 14 | 0 / 0 |
| Q8 feedback | 107 / 198 | 107 / 198 | 0 / 0 |
| Q8 feedback, 16 training-selected exceptions | 97 / 171 | 97 / 171 | 0 / 0 |

Receipts, real dtype ratios and arrays are in
`ocr-calibration/{serialized-candidates.json,gpu-serialized/results.json}` and
`cdla-train/` under the isolated dual-GPU build. Twenty-eight native GPU checks
include an F32 control per round. They establish only fixed-input diagnostics.
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/Quantization_intro.md)
provides calibration/bias/reconstruction approaches;
[CDLA](https://github.com/buptlihang/CDLA) separates 5,000 train / 1,000 validation
pages. Official INT8 is not GGML Q8_0, and task metrics do not prove raw-head identity.

### 2026-10-21: Margins and final projection reconstruction

Saved arrays contain original class top-two gaps as small as `7.0664e-8`.
For the affine-F16 first class head, median flipped-vector gap was about
`1.1858e-6` while median max perturbation was `2.8471e-6`. Low-confidence positions
remain gated. See `ocr-calibration/head-margin-diagnostics.json`.

Four final joint class/DFL projection layers were reconstructed on the same
16 train pages with relative ridge 0.01. Their kernels remained F32 and other
stored parameters were frozen. Training reconstruction MSE decreased, but
independent checks still found F16 9/16 and Q8 105/174 flips. Actual files
round-tripped all 486 parameters: F16 has 388 F32 / 98 F16 tensors; Q8 has
388 F32 / 59 Q8 / 39 F16 tensors. Independent original CUDA and both native
GPUs reproduced those counts, with zero flips vs stored-weight Paddle and
zero for F32 controls. See `ocr-calibration/cdla-train/block-reconstruction/`
and `block-reconstruction/gpu-serialized/results.json`. Both remain FAIL.

A separate SAM3 F16 CUDA output-stride defect was found during full-task
migration and fixed by an ordered patch/new-old module test. It is unrelated to
layout quantization; see
[stride evidence](../../../../3rdparty/llama/validation/cuda-native-stride-evidence.json).

### 2026-10-21: Full-model preflight and timing repairs

RF-DETR's initial 44-model run reproduced old output on both GPUs but reported
five CUDA / 29 Vulkan timing failures. The probe lacked warmup; Vulkan base Q4_K
had forwards of 307.676924 / 49.645335 ms in one process. Warmup, cold-call/raw
samples and nonzero errors for plugin-path/nonfinite failures were added to
`bench_rfdetr_perf`. The same new probe was linked against frozen old/new static
libraries without rebuilding `build_app`. New A/B used three processes, three
warmups and ten timed forwards; old failures and thresholds were retained.

The first full OCR preflight linked private `assets/extract` but not neighboring
`download`; eight required archives already existed in the shared cache.
Adding that link enabled offline SHA checks without cache deletion or duplicate
downloads. A separate old ORT environment lacked `onnx` under
`PYTHONNOUSERSITE=1`; a complete isolated environment repaired source export.
Classifier F32 and v5 detector/recognizer F32 then completed repeated old/new
checks. Preflight failures are neither numerical FAIL nor PASS.
See [full-model evidence](../../../../3rdparty/llama/validation/full-model-regression-evidence.json).

### 2026-10-21: Threads and complete reference environment

The runner passed `--threads 0`, but ONNX and SLANet page consumers required a
positive value. Twelve page/region attempts failed during parsing. Those and
Paddle page consumers now resolve zero to their default four threads, use the
same effective count on source/native engines, preserve positive values and
reject negatives. Three new tests had a failing old-code control; existing
Paddle-page 19 and SLANet nine tests passed. The active task exited naturally
before its queue was superseded, preserving attempts and script identities.
See `build_llama_dual_gpu_3nl01j4j/ocr-thread-protocol-fix/`.

A fresh environment installed
[ONNX GPU requirements](../../../../core/AICore/tools/requirements-ocr-onnx-gpu-reference.txt),
including upstream TextSystem polygon/image/config dependencies. Ten modules,
TextSystem import, `pip check`, CUDA 11.8 and cuDNN 8 resolution passed with user
packages/PYTHONPATH isolated. These are environment checks, not model gates.
A real v5-mobile F32 CUDA TextSystem page then passed twice per old/new runtime
on each GPU, with four threads and explicit `CUDAExecutionProvider`; 66 script
tests passed. Full-model scope remained open.

Warmup-corrected RF-DETR still had 6 CUDA models / 14 metrics over limits
(5.94%–28.48%): large F16, medium F32/Q4_K, nano F32/Q4_K and seg-large Q4_K.
Vulkan had 22 models / 45 metrics over limits (5.02%–18.04%). All 44 model
numerical/repeat fingerprints matched on each GPU, with unchanged execution
identities. Warmup alone did not explain persistent differences. Later isolated
retries are recorded in the kernel audit, without rewriting these failures.

The experimental CMake driver also omitted production hidden-visibility flags
from ExternalProject. Forwarding them passed a failing old-driver control and
12 CMake contracts. Active binaries were not replaced; no timing conclusion
was inferred. See `rfdetr-warmup-ab/visibility-config-receipt.json`.
Official [PicoDet QAT](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/slim/quant/picodet_s_416_lcnet_quant.yml)
uses channel-wise weights, INT8 activations and PACT, whereas Q8_0 uses 32-weight
blocks. Their scores cannot be inherited. [PaddleSlim reconstruction](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/quant/reconstruction_quantization.py)
is an experimental direction, not an executed fix.

### 2026-10-21: QAT with real storage operators

Offline `ocr_qat_storage.py`, `ocr_qat_graph.py` and `train_ocr_storage_qat.py`
use actual F16 rounding/Q8_0 decoded values in forward and independent F32
master weights in backward. Q8 integers use the unrounded block scale before
storing that scale in F16; independent GGUF round-trip tests passed four cases.
Norm statistics/parameters and bias remain explicitly F32. No Paddle dependency
is introduced into C++ inference.

Imported ProgramDesc temporary names collided with new loss expressions.
Independent local/global name prefixes fixed this; actual small-graph forward/
backward checks cover collision, rounding and disconnected branches. On one
independent train page, all eight unquantized heads matched the untouched graph
exactly, all 102 convolution kernels had finite gradients, F16 gradients were
nonzero, and normalization state was unchanged. This validates training plumbing.

A fixed 16-train-page pilot ran 64 steps/four epochs with energy-normalized
8-head MSE, LR `1e-7` F16 / `1e-6` Q8. Recipes/script SHAs were frozen before
training; all 486 stored parameters matched independent GGUF decoding.
Original/stored Paddle CUDA and both native GPUs then gave:

| Model | Original class flips | Original DFL-edge flips | Both GPUs vs stored Paddle | Strict verdict |
| --- | ---: | ---: | --- | --- |
| Original F32 control | 0 | 0 | Match | PASS for this input |
| F16 MSE QAT | 13 | 11 | Zero flips | FAIL |
| Q8 MSE QAT | 205 | 347 | Zero flips | FAIL |

F16 SHA: `d18587ceccfea19dca125bcb53e588e20a42799d517b9f99b2a73ccccabbfd25`.
Q8 SHA: `bf6e6ea6a9cb32b7033fd12fcf49ff8c98101aff59603d7857b0fdd670785dab`.
See `ocr-calibration/qat-pilot/{gpu-serialized/results.json,serialized-candidates.json}`.

An optional teacher-winner ranking loss uses four independent 8-bin DFL groups.
Default weight zero preserves MSE; five real static-graph tests checked finite
nonnegative options, grouping and gradient direction. A predeclared pilot used
the same 16 pages, four epochs, ranking weight 64 and margin `1e-6`.
`ocr-calibration/qat-ranking-pilot/` completed with F16 11/12 and Q8 258/405
flips on both GPUs, zero vs stored truth, and passing F32 controls. Q8 did not
improve. SHAs: `f8893b1e4881d73fd2081a3f45c10c45455b4210c17c6c8705419e14c5db1c2c`
and `167285c3e6da1f89bc9c14f343a2e0479d9465663a1a2d0f1fdae376f7e82336`.
The full queue resumed after each experiment, including error cleanup.

Margin analysis reproduced minimum failed class gap `7.066410034894943e-8`.
No vector satisfying `original gap > 2 * max error` flipped; others remained
strictly gated. The [ranking quantization paper](https://arxiv.org/abs/2106.14156)
is motivation, not proof of exact decisions for this model.

### 2026-10-21: Winner objective and graph-test runner

The earlier hinge reconstructed the full teacher top1-top2 gap, while the gate
only requires its winner. The revised objective requires that winner to exceed
the strongest competitor by `ranking_margin`, retains probability MSE, applies
four separate DFL constraints and excludes the winner through a stop-gradient
dynamic bound rather than a full `-inf` mask. Six real Paddle graph/gradient
tests cover competitors, side flips, correct-winner behavior and stored forward.
A graph-test harness heap-corruption exit was fixed by one process, explicit
Scope lifetime and natural success exit. This is not an inference change; prior
failed QAT results stayed valid until separately retrained and checked.

### 2026-10-21: Read-only progress after configuration refactor

At 11:25 UTC, public `normalize_accuracy_gate` / `evaluate_accuracy_gate` checked
the frozen manifest against 61/168 completed scenarios: 244 full attempts,
122 per old/new runtime, no structured accuracy failure in those attempts.
Unfinished layout/pages/other backends were not promoted.

| Check | Snapshot | Release meaning |
| --- | --- | --- |
| Completed old/new accuracy attempts | 122 each; zero failures | Only the 61 completed scenarios |
| Candidate attempts with official CUDA timing comparison | 56; ten comparison failures | Keep failures even when process rc=0 |
| `source_performance_qualified` | Zero in all 56 | No controlled qualification |
| `source_gpu_audit_pass` | Zero in all 56 | Controlled retry required |
| Layout F16/Q8 all-head gate | FAIL after two QAT rounds | Preserve 100%; no upload |

The ten timing failures span seven scenarios: v6-medium detector F32/F16/Q8,
v4 detector F32, v6-small detector F16/Q8 and v6-tiny detector F16.
Developer timings neither establish an upgrade regression nor qualify release.
Raw snapshot: `build_llama_config_refactor_glj5ibw7/Testing/ocr-cuda-attempts.snapshot.jsonl`,
SHA `11581b51301b0f2f2002b2c8569c38b5c74b6e9f67252ddb6c766da4d831a925`;
summary: neighboring `ocr-cuda-progress.json`. CPU build/install, 14 CMake
contracts, loader paths and idempotent patches passed independently.
See [kernel audit](../../../../3rdparty/llama/KERNEL_AUDIT.md).

### 2026-10-22: Training-chain corrections

| Problem | Confirmed cause and repair | Evidence |
| --- | --- | --- |
| Requested CUDA teacher actually ran CPU | CPU-only Paddle warns and continues; check `is_compiled_with_cuda()` before creating outputs and verify predictor config | `ocr-calibration/cpu-teacher-rejection.json`: expected failure with no output directory |
| Q8 first step failed when ranking disabled | It read unregistered winner labels; shared `teacher_head_feed` creates labels only for enabled ranking | Seven real forward/gradient checks passed, including default Q8 and four DFL groups |
| GPU-wheel graph-test double free | Host `GLOG_*` / `MLOG_*` polluted static logging cleanup; use `source_environment` before child start | `qat-clean-gpu-environment-graph-tests.json`: seven CPU-graph tests under the GPU wheel exited normally; not GPU accuracy |
| New log-probability graph tests failed in clean Paddle 3.x static mode | `paddle.log` dispatched through an eager-only tensor method for a legacy static `Variable`; append the registered static `log` op through `LayerHelper` | Clean `source_environment` process: all 12 static graph tests and all 9 storage tests passed; no model accuracy claim |

Receipts are under `build_llama_dual_gpu_3nl01j4j/ocr-calibration/`. Older
CPU-teacher experiments remain but their recipe's CUDA claim is invalid.
`qat-cdla256-gpu-teacher-v2-20261022/` pinned 256 train pages, four epochs,
model/script SHAs and a verified CUDA wheel; F16 used linear winner hinge,
Q8 probability MSE. CPU optimizer execution is not CPU acceptance. Final
serialized original-CUDA/native-GPU checks remain mandatory.

### 2026-10-22: 256-page CUDA-teacher QAT

`ocr-calibration/qat-cdla256-gpu-teacher-v3-20261022/` completed four epochs on
256 pinned train pages with verified CUDA teacher. Independent acceptance here
covers one fixed input's eight heads, not 1,000-page metrics or performance.

| Candidate | GGUF SHA-256 | Original CUDA class / DFL flips | Both GPUs vs stored graph |
| --- | --- | ---: | ---: |
| F32 control | `7e2785606f99bc6d1c7ffa6df6b15e42bd89da6c7a98ce7f1de1961c4cb2fcf1` | 0 / 0 | 0 / 0 |
| F16 QAT | `9a454a8bcbf3ffb2d65b01bbd29bc6d3aa5d61cd18ddf81c1d8e70f027f6a903` | 18 / 26 | 0 / 0 |
| Q8 QAT | `93010a1033954c82b823d998363965c1591d512b2f9182e7bf82118e6f5042ed` | 158 / 204 | 0 / 0 |

Both remain FAIL. The stored graph reproduces native decisions and both differ
from original teacher. See `gpu-serialized/results.json` and
`qat-*-stored-source-cuda.json`. The trainer additionally records exact class/
four-DFL flip counts per page/epoch, separating loss improvement from gate
convergence. Fresh source/backend/page/performance checks still follow training.

### 2026-10-22: Low-probability gradients and GPU calibration

Class heads are sigmoid probabilities; DFL heads are logits. Probability-space
hinge gradients include `p * (1-p)` at low scores. Offline option
`--class-ranking-space log_probability` preserves winners by monotonic log and
changes the sigmoid-composed derivative to `1-p`; DFL retains four logit groups.
Inference, original truth, thresholds and default probability objective stay
unchanged. Explicit `--optimizer-device cuda` rejects non-CUDA wheels before
outputs and records teacher/optimizer devices. Ten real graph/gradient tests,
nine storage tests and six parameter-payload tests passed.

The 2026-10-22 clean static-graph rerun initially exposed two errors in the new
log-probability tests: Paddle 3.x `paddle.log` used its eager-only path on a
legacy static `Variable`. The graph builder now appends the registered static
`log` op directly. The corrected graph suite passes 12/12 with normal process
exit; storage tests pass 9/9. The earlier double-free diagnosis describes the
polluted host environment and is separate from this static API incompatibility.

The fixed 256-page/eight-epoch run in
`ocr-calibration/qat-logprobability-gpu-20261022T1205/` had best F16 epoch six:
1,749 class flips out of 2,586,880 decisions and 3,090 DFL flips out of
10,347,520. Best Q8 epoch eight still had 22,824 / 41,707 flips.
All 486 GGUF parameters round-tripped; Q8 retained 59 Q8 / 43 F16 / 384 F32
norm/bias tensors. Independent results were:

| Candidate | GGUF SHA-256 | Original F32 class / DFL flips, both GPUs | Stored-graph flips |
| --- | --- | ---: | ---: |
| F32 control | `7e2785606f99bc6d1c7ffa6df6b15e42bd89da6c7a98ce7f1de1961c4cb2fcf1` | 0 / 0 | 0 / 0 |
| F16 log-probability QAT | `96b3f7280bb275abd15b2cb69eed56402762b3612a4795feda10303810120f82` | 13 / 18 | 0 / 0 |
| Q8 log-probability QAT | `1f993e76f16c929404ccbf814dfb11d47efa518c60056b928c877e034be692a0` | 140 / 232 | 0 / 0 |

`gpu-serialized/head-margin-diagnostics.json` shows no original ties: minimum
F16 class gap `7.0664100e-8`, flipped DFL gaps `4.3809414e-5` to `0.0038111806`.
All flipped vectors had gap no greater than twice perturbation. For both
checkpoints, `master + (stored-master)` equalled all 1,794,568 deployed trainable
values; no STE cancellation defect was found. Improvement was partial, not
better than every earlier candidate (nearest structural-F32 F16 13/16;
Q8 feedback 122/226). Never fill outputs from teacher truth, hide an F32 side
model or label almost-all-F32 storage as Q8. Both candidates remain FAIL.

### 2026-10-22: Teacher math policy correction

`check_ocr_qat_math.py` ran two fresh Paddle/CUDA processes on four pinned train
pages, comparing default TF32 with `NVIDIA_TF32_OVERRIDE=0`. Within either policy,
unquantized training and original teacher heads matched elementwise; across
policies there were 39 class / 81 DFL flips. Thus BatchNorm/training graph
rewriting was not the discrepancy on this input; teacher policy differed from
independent acceptance. The trainer now disables TF32 before Paddle import and
records the policy. See `ocr-calibration/qat-math-policy-cldx2ri7/` and
`cross-policy-comparison.json`. These are diagnostic receipts.

A corrected 256-page/eight-epoch run again serialized and checked all parameters:

| Candidate | GGUF SHA-256 | CUDA vs original class / DFL | Vulkan vs original class / DFL | CUDA; Vulkan vs stored Paddle CUDA |
| --- | --- | ---: | ---: | --- |
| F32 control | `7e2785606f99bc6d1c7ffa6df6b15e42bd89da6c7a98ce7f1de1961c4cb2fcf1` | 0 / 0 | 0 / 0 | 0 / 0; 0 / 0 |
| F16, TF32 disabled | `642928f620cea7b2cfc3bd5dee07d437af971bf3cd9f775aa8f15455e60667dd` | 10 / 14 | 10 / 14 | 0 / 0; 0 / 0 |
| Q8, TF32 disabled | `f3601f449349d42e3917993dbac16121c3691430dfc8da5305ab9746bae0c6df` | 140 / 243 | 140 / 244 | 0 / 0; 0 / 1 |

F16 improved but failed; Q8 did not improve. Q8 exposes one extra Vulkan DFL
flip vs stored truth: head 4, position 764, edge 3, source gap `6.556511e-7`,
CUDA/Vulkan max errors `1.2218952e-6 / 2.0861626e-6`. Keep this backend error;
do not inherit an earlier zero-backend-flip conclusion. Receipts are under
`ocr-calibration/qat-logprobability-fp32-policy-20261022T1343/`, including
`F16/result.json`, `Q8_0/result.json`, `gpu-serialized/results.json`.
Best training checkpoints still had 1,004/1,617 F16 and 22,456/41,063 Q8 flips.
Neither is published or a page/performance pass.

### 2026-10-22: Upgrade isolation and controlled gates

The same portable probe, model SHA `f3601f44…`, independent processes and actual
loaded-library identities compared ggml 0.21 with embedded 0.27. Against stored
Paddle CUDA, both versions gave CUDA 0/0 and Vulkan 0/1 class/DFL flips.
CUDA heads were elementwise equal across versions. Vulkan head max differences
were `2.682209e-7` to `8.106232e-6` without an added winner flip. The stored-source
Vulkan defect predates this upgrade and still requires repair; original-F32
failures are not waived. See `stride-fix/lightglue-ocr-firstops-r75kymt6/`,
`qat-q8-stored-source-counts.json` (10,105 class / 40,420 DFL decisions) and
`qat-q8-upgrade-comparison.json`.

A fresh classifier F32 release sentinel kept same-host `build_app` baseline,
independent ONNX CUDA, three interleaved processes, five warmups, ten forwards,
four threads and public audit thresholds. Neither GPU produced a qualified
complete measurement: Xorg, GNOME Shell and IDE SM activity failed idle/external
activity audits. This is invalid measurement, not measured inference slowness.
Most processes actually checked nine other inputs, not all ten requested; one
Vulkan baseline completed all ten with matching argmax but still failed audit.
See `stride-fix/ocr-controlled-current-taqsoldi/`, both release JSONs and raw
`.gpu.d/` archives. Both overall verdicts are FAIL. An earlier DPT experiment
omitted `--baseline-build` and exited before inference; retain it as a command
error, not a model accuracy/speed result.

### 2026-10-22: Serialized updates and entry environments

All 486 TF32-disabled eight-epoch checkpoint values exactly matched independent
F16/Q8 serialization. Compared with untrained storage, F16 changed 446,690 of
1,767,904 half parameters; Q8 changed 25,866 integer codes and 7,561 of 50,624
block scales across 1,619,968 parameters. The remaining flips are not explained
by absent optimizer updates or discarded saved updates. See
`ocr-calibration/qat-logprobability-fp32-policy-20261022T1343/serialized-update-audit-ag_dqwqm/audit.json`.

`os.environ.update(clean_env)` failed to remove original keys omitted by the
cleaner. Trainer and teacher-policy checker now replace the complete environment
before Paddle import. Two real CUDA-wheel subprocesses injected host
`GLOG_log_dir`, `MLOG_log_dir`, `PYTHONPATH`; both verified removal and normal
import/exit, with trainer TF32 still disabled. No retraining or new numeric pass
is implied. See `ocr-calibration/qat-environment-check-7rqc5j8p/`.

### 2026-10-22: Reject silent source-GPU fallback

`bench_paddle_ocr_graph.py` checked GPU-package metadata but not the actual
loaded library. Mixed CPU/GPU distribution records cannot prove CUDA execution.
It and `run_paddle_layout_reference.py` now check `is_compiled_with_cuda()` and
`config.use_gpu()` before predictor creation. The latter also disables TF32
before Paddle import and records requested/actual support and configuration.
Twenty-three tests cover CPU libraries with GPU metadata, disabled GPU config
and a correct GPU branch. Four real CPU/CUDA-wheel preflights passed, rejecting
CPU and confirming CUDA device 0. They deliberately stopped before predictor
creation, so are not numeric/performance passes. CTest registration is
`test_ocr_paddle_reference_device`; receipts:
`ocr-calibration/paddle-device-preflight-r3g494mt/`. Active validator identities
were preserved; later fresh queues consume the changed scripts.

### 2026-10-22: Actual fixture coverage and repeat isolation

ONNX multi-input accounting now distinguishes requested inputs
(`accuracy_fixture_requested_count`), completed source/stored-source numeric
comparisons with finite metrics/digests (`accuracy_fixture_count`) and complete
successful input attempts (`fixture_pass_count`). Missing comparisons are listed
in `fixture_incomplete_indices`; completed numeric failures are not passes.
Main-input GPU audit failure still fails the whole attempt.

Each probe writes an incomplete state at launch and a current error on source
export failure, preventing reuse of a previous successful report at the same
path. Thirty-four tests cover pre-inference audit rejection, completed numeric
failure, detector heads without class argmax, and stale source-export reports.
Read-only rechecking `ocr-controlled-current-taqsoldi` found requested 10,
completed 9, passed 9 per aggregate, main input incomplete, overall FAIL.
Original reports were not edited. Receipt:
`stride-fix/ocr-fixture-coverage-audit-3toq1u_9/receipt.json`.

An offline Q8-only optimizer LR multiplier defaults to one; F16 convolution and
F32 affine parameters keep the base step. A real static Adam check verified 10x
Q8 updates with the other groups unchanged; 11 graph tests passed. See
`ocr-calibration/qat-storage-lr-tests-rsn_4h7d/`. Optimizer tests are not release
accuracy evidence.

### 2026-10-22: Fixed-step recipes completed; strict failures retained

Two predeclared Q8 eight-epoch recipes completed the same 256 train pages.
Selection used total class/four-independent-DFL flips, then MSE, on train pages
only, without validation-driven selection:

| Q8-only LR multiplier | Best train class / DFL flips | Total | Disposition |
| --- | ---: | ---: | --- |
| 1, retained control | 22,456 / 41,063 | 63,519 | Existing failed candidate retained |
| 10 | 23,742 / 41,672 | 65,414 | Worse; not exported or retried on validation |
| 100 | 62,705 / 94,474 | 157,179 | Worse; not exported or retried on validation |

See `ocr-calibration/qat-q8-step-sweep-1khkov0i/state.json` and recipe
`result.json` files. This rejects those two step choices, not every possible
training method, and never changes original-F32 truth.

F16 completed 32 epochs at LR `3e-8`. Best train epoch 26 had 174/406 flips,
total 580 versus 2,621 for the earlier eight-epoch control. Actual GGUF SHA:
`9295b0943608c7dd57f302b251b9c4a149b59e99e989454343a376a7d56e787f`.
Independent original Paddle CUDA and both native GPUs gave:

| Actual file | CUDA vs original class / DFL | Vulkan vs original class / DFL | Both GPUs vs stored truth |
| --- | ---: | ---: | ---: |
| Original F32 control | 0 / 0 | 0 / 0 | 0 / 0 |
| New F16 checkpoint | 7 / 12 | 7 / 12 | 0 / 0 |

This improves 10/14 but remains **FAIL**. Native backends reproduce stored-model
winners; remaining disagreement is between quantized/trained storage and the
original source. The file is not published. Training receipt:
`ocr-calibration/qat-f16-small-step-2shqfvn3/small-step-32-epochs/result.json`;
serialized/source/native arrays and loaded-library SHAs:
`stride-fix/ocr-native-owner-recovery-i89r2ii1/selected/serialized-candidates.json`
and `selected/gpu-serialized/results.json`.

Before that independent check, the lower-level probe's library-identity
assumption was repaired: `test_ocr_runtime` has ggml DT_NEEDED entries and does
not link libAICore. Verify actual `libggml`, `libggml-base` and selected GPU
modules instead. The original script/receipt was retained; no training process
was signalled. Only the waiting experiment orchestrator was replaced, with a
fresh frozen script identity. No numerical or release gate was waived.

### Constant head-bias correction is insufficient

A read-only analysis of the best serialized F16 candidate tests whether constant
DFL channel biases could recover every original winner on the retained fixture.
Four of sixteen head/edge groups have infeasible difference constraints, even
with zero required margin. For example, head 4, edge 0 requires both
`b[5]-b[4] <= 0.0000183880329` at position 1851 and
`b[4]-b[5] <= -0.000671684742` at position 4415. Their sum is negative, proving
that a constant bias cannot satisfy both original decisions. Changing the bias
to repair one position necessarily breaks another.

This identifies a limit of constant-bias correction, not of all retraining or
mixed-precision methods. The analysis changes no weights, uses no acceptance
outputs for optimization, and leaves every original-head comparison intact.
Input SHA-256 values, all sixteen groups and contradictory-position certificates
are recorded in
`build_llama_dual_gpu_3nl01j4j/ocr-head-bias-feasibility-vjskef3b/result.json`.
The candidate still fails the original gate and remains unpublished.

## Historical evidence locators

These exact paths supplement the abbreviated references above. They identify
historical local evidence and do not imply current release qualification.

- `build_app/Testing/ocr-storage-reference/layout-attribution-reverified.json`
- `build_app/Testing/ocr-storage-reference/layout-backend-attribution.json`
- `build_app/Testing/ocr-storage-reference/layout-decision-confidence.json`
- `build_app/Testing/ocr-storage-reference/layout-margin-review-jj9e632k/report.json`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/head-margin-diagnostics.json`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/qat-logprobability-gpu-20261022T1205/`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/qat-pilot/`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/qat-ranking-pilot/`
- `build_llama_dual_gpu_3nl01j4j/ocr-calibration/{serialized-candidates.json,gpu-serialized/results.json}`
- `core/AICore/scripts`
- `core/AICore/tools/ocr_q8_calibration.py`
- `core/AICore/tools/test_ocr_f16_calibration.py`

### 2026-10-23: Desktop GPU audit and Q8 source-speed recheck

The desktop and remote NoMachine graphics sessions are expected use of this
host's GPU. `gpu_audit.py` now records graphics-only process IDs and utilization
as informational telemetry; only foreign compute contexts (`C` / `C+G`) block
the preflight or controlled sample. It still requires complete driver telemetry,
three consecutive preflight samples without foreign compute, and an owned
process-memory sample. Ten focused audit tests pass. This changes only how
expected graphics occupancy is classified; it does not reduce the requirement
for a valid controlled model measurement.

Fresh release-profile Q8 graph A/B used the baseline `build_app`, the current
candidate, three process repeats, two warmups, ten measured forwards, the pinned
Paddle 3.0.0 source graph, and audited CUDA/Vulkan. The corrected Vulkan rerun
uses the Python environments under `build_app/Testing/ocr-storage-reference/`;
the first attempted run referenced a missing path and is a command-configuration
failure, not inference evidence.

| Model / graph | Backend | Candidate p50 | Paddle CUDA p50 | GPU audits | Result |
| --- | --- | ---: | ---: | --- | --- |
| `ppocr-det-ch-q8_0.gguf` | CUDA | 10.61 ms | 6.736 ms | both PASS | FAIL, source-speed gate |
| `ppocr-det-ch-q8_0.gguf` | Vulkan | 12.27 ms | 5.888 ms | both PASS | FAIL, source-speed gate |
| `ppocr-rec-ch-q8_0.gguf` | CUDA | 1.71 ms | 4.327 ms | both PASS | PASS |
| `ppocr-rec-ch-q8_0.gguf` | Vulkan | 4.40 ms | 3.734 ms | both PASS | PASS |
| `ppocr-rec-en-q8_0.gguf` | CUDA | 1.62 ms | 4.559 ms | both PASS | PASS |
| `ppocr-rec-en-q8_0.gguf` | Vulkan | 4.33 ms | 4.510 ms | both PASS | PASS |

The detector's regression reproduces on both backends with no GPU-audit
rejection. The graph log has 680 nodes, including 48 `IM2COL`, 48 `MUL_MAT`,
and 48 `CONT` nodes; dense detector convolutions remain on im2col+GEMM because
the previously tested direct route changed threshold-sensitive detector maps
and damaged page transcriptions. This localizes the cost to the current dense
convolution route, but does not yet identify a numerically acceptable faster
kernel. Current public-page runs also reproduce on both backends: F32 and F16
pass; Chinese Q8 fails at e2e CER delta `+0.009709`, and English Q8 at
`+0.005296`, against the declared `0.005` page limit. These are single-repeat
development runs and not release evidence.

Receipts:
`build_llama_dual_gpu_3nl01j4j/Testing/ocr-q8-release-ab-cuda.{json,md}`;
`build_llama_dual_gpu_3nl01j4j/Testing/ocr-q8-release-ab-vulkan-corrected.{json,md}`;
`build_llama_dual_gpu_3nl01j4j/Testing/ocr-pages-gate-audit-{cuda,vulkan}.{json,md}`.
Only the two recognition Q8 rows pass this selected gate. Detector performance,
page accuracy, remaining OCR models, immutable queue completion and release
provenance/download checks are still open; these partial results do not qualify
any file for upload because none has passed all declared release gates.

### 2026-10-23: Separate SROIE 50-page Q8 recheck

A separate 50-page SROIE cohort evaluated 2,779 fixed ground-truth recognition
regions on the baseline and current candidate, using the same inputs and the
declared 0.005 source-delta budget. It must not be combined with the public-page
single-repeat cohort above.

| Language / model | Hmean delta | Fixed-region CER delta | Fixed-region WER delta | E2E CER delta | E2E WER delta | Page gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chinese `ppocr-rec-ch-q8_0` | +0.00069860 | -0.00030101 | -0.00227790 | -0.00103682 | -0.00332924 | PASS |
| English `ppocr-rec-en-q8_0` | +0.00069860 | +0.00127095 | +0.00683371 | +0.00113716 | +0.00928684 | FAIL |

The English recognition-only comparison adds 38 character errors and 39 word
errors against Paddle on the fixed regions; end-to-end adds 34 character and 53
word errors, alongside the small detector Hmean decrease. The source and
candidate both returned the same failing status, so this is a checkpoint/page
quality failure rather than a llama-migration-only regression. Keep the 0.005
gate unchanged and localize the English Q8 source-quality loss before changing
weights or thresholds.

The temporary candidate scratch directory that contained this run's raw JSON
and Markdown receipts has since been removed. These numbers are retained as
diagnostic results, not durable release evidence. Re-run both backends with the
same pinned page set and preserve raw receipts under
`build_app/Testing/ocr-storage-reference/` before promoting either row. No OCR
GGUF is qualified for upload by this cohort alone.

### 2026-10-23: Correct-binary repair-candidate CUDA recheck

The Q8 storage-repair candidate was re-measured with the current
`build_app/bin/aicore_tests/test_ocr_runtime`, rather than the stale top-level
test binary used by an earlier diagnostic. Both native and Paddle source GPU
audits passed, with two warmups and ten measured forwards. The source graph
used the pinned PP-OCRv4 Paddle program and the same detector fixture. The
results confirm that normal desktop/NoMachine graphics occupancy was not the
reason the upload gate stayed closed.

| Chinese detector weight | Candidate CUDA p50/p95 | Paddle CUDA p50/p95 | Native/source audit | Tensor result | Gate |
| --- | ---: | ---: | --- | --- | --- |
| repaired `ppocr-det-ch-q8_0.gguf` (SHA `cab74270…ae56d`) | 11.03 / 12.03 ms | 6.58 / 8.42 ms | PASS / PASS | max diff 0.477951, mean 0.00023338; quantized tensor delta is diagnostic, page accuracy is separate | FAIL, p50 is 4.45 ms slower than source |
| published `ppocr-det-ch-f16.gguf` (SHA `82517e16…9550d`) | 11.18 / 12.61 ms | 6.17 / 6.99 ms | PASS / PASS | max diff 0.061630, mean 0.00004743; quantized tensor delta is diagnostic | FAIL, p50 is 5.01 ms slower than source |

The paired result rules out Q8 decoding as the primary slowdown: switching the
same detector to F16 changes p50 by only 0.15 ms, while both graphs remain
around 11 ms against a 6–7 ms Paddle graph. Native logs confirm CUDA0 execution
and report 666–679 graph nodes, including 48 `IM2COL`, 48 `MUL_MAT` and 48
`CONT` operations. GPU fallback and foreign compute occupancy were not
observed. This localizes the performance gap to GPU graph execution,
especially dense-convolution lowering, but does not yet provide a numerically
acceptable faster replacement. The previously tested all-direct
dense-convolution route is not promoted because it shifted threshold-sensitive
detector maps and damaged page transcriptions.

Receipts:
`build_app/Testing/ocr-storage-reference/public-q8-gpu-truth-vpx8osmk/release-recheck/cuda-det-correct.json`;
`build_app/Testing/ocr-storage-reference/public-q8-gpu-truth-vpx8osmk/release-recheck/cuda-det-f16.json`.

Neither row is upload-qualified. The repaired Q8 weights have useful page
accuracy evidence, but fail the CUDA source-speed gate and still need a
controlled Vulkan release-profile measurement. F16's recorded raw threshold
exit is diagnostic under the text-graph policy, not an additional release
veto. Other OCR files still need exact-file accuracy and
performance rows, stability and memory evidence, catalog/digest registration,
and cold-cache download verification. No release upload was attempted.

### 2026-10-23: CUDA detector kernel profile (diagnostic only)

Nsight Systems profiled one corrected-binary Q8 detector invocation after the
probe's two warmups. This run is for attribution only: profiler overhead makes
its wall time ineligible for the release-speed gate. The trace confirms that
`im2col_kernel<float>` is the largest individual kernel family in the captured
graph (2.72 ms across two launches), followed by two cuBLAS SGEMM launches
(0.67 ms total). It also records 48 device-to-device copies totaling 1.84 ms,
168 host-to-device copies totaling 2.73 ms, and one device-to-host copy of
0.14 ms. These transfer totals cover the profiled process interval and are not
per-forward latency estimates.

This narrows the optimization target to dense-convolution lowering and graph
intermediate movement; it does not prove which individual `CONT` node owns a
given copy. The direct-convolution experiments remain disqualified because
their changed accumulation order moved detector scores across the DB threshold
and reduced valid transcribed rows. A future replacement must preserve the
Paddle page metrics while reducing the measured candidate latency on both CUDA
and Vulkan; no such replacement has passed yet.

Receipt: `build_app/Testing/ocr-storage-reference/public-q8-gpu-truth-vpx8osmk/release-recheck/cuda-det-q8-nsys.nsys-rep`


### 2026-10-23: exact-gather CUDA fix and current migration results

Two measured causes were repaired without replacing convolution arithmetic:
N=1 convolution outputs no longer execute 48 redundant `CONT` copies, and
low-channel large-image IM2COL uses spatially contiguous output scheduling.
The latter is ordered patch `0011-acv-cuda-im2col-spatial-blocks.patch`, not an
edit to an extracted dependency tree. The GPU oracle independently checks each
copied/zero-padded element and destination F16 conversion. The model A/B covers
30 CUDA/Vulkan rows: detector and Chinese/English recognizers at F32/F16/Q8,
plus batch=3 recognizers. All 30 output files are bitwise identical before and
after the CUDA patch; all 60 probe processes returned zero. This establishes
preservation of the prior outputs, not Paddle truth or page accuracy.

Controlled release-profile CUDA rechecks (two warmups, ten forwards, four host
threads) passed the existing source-speed rule at both p50 and p95: a regression
must exceed both 5% and the 3 ms absolute noise floor to fail. The actual ratios
remain visible below; PASS does not claim equal latency or a speedup over Paddle.
Both native and source GPU audits passed, with desktop graphics treated as
normal occupancy.

| Detector file | Candidate p50 / p95 | Original Paddle CUDA p50 / p95 | Source-speed result |
| --- | ---: | ---: | --- |
| published F16, SHA `82517e16…9550d` | 8.21 / 9.52 ms | 5.8255 / 6.5636 ms | PASS |
| repaired Q8, SHA `cab74270…ae56d` | 9.45 / 10.71 ms | 6.7272 / 7.8331 ms | PASS |

These supersede the earlier failing CUDA speed receipts, which remain retained.
No upload is qualified by these two rows: the new Vulkan scheduling patch must
first pass its scalar/model tests and controlled source timing; current-runtime
page accuracy, repeated stability, exact-file provenance and release-download
receipts are still required. Layout's original raw-head argmax gate is unchanged.

The repaired six-scenario CUDA llama A/B records five PASS (RF-DETR nano,
Chinese OCR recognizer, SLANet, AnyCalib, RAM++) and one FAIL (Relation Q8).
SLANet p50 is now 20.06 ms against the old approximately 20.95 ms baseline;
its former latency/peak-memory regression is absent. Relation candidate and
pinned upstream GGML outputs are bitwise identical, but both score 262/271
(96.67897%) original-checkpoint predicate winners against a 97% absolute floor.
Its RMSE is 0.0292292 and pooled pair Jaccard is 0.992674. This is a preexisting
Q8 checkpoint-accuracy failure, not a measured llama-added numerical error;
it remains open rather than being hidden by an A/B-only verdict.

Receipts under `build_llama_production_20261023/Testing/`:
`ocr-im2col-85e55y1b/im2col-model-regression.json`,
`ocr-im2col-85e55y1b/release-cuda-det-{f16,q8_0}.json`, and
`llama-cuda-im2col-repaired-probes.json`. The selected 19-scenario CUDA report
still has unresolved Depth/LoMa/RMBG fingerprints and GKD/LoMa timing failures;
this six-scenario repair does not close that complete report or the Vulkan
matrix. No OCR release upload was attempted from incomplete qualification.

### 2026-10-23: Vulkan im2col verification and detector source-speed recheck

The ordered Vulkan sibling patch
(`experimental/patches/0011-acv-vulkan-im2col-spatial-blocks.patch`) applied the
same two repairs to the Vulkan backend: N=1 convolutions skip the redundant
`CONT` copies, and low-channel large-image IM2COL uses spatially contiguous
output scheduling. Two checks were then run against the production build tree;
both receipts are persistent.

**Output preservation (PASS).** The 15-row Vulkan model/batch matrix — detector
F32/F16/Q8_0 at n=1 and both recognizers (Chinese, English) at F32/F16/Q8_0 for
n=1 and n=3 — replayed the pre-patch probe binaries' inputs and compared output
SHA-256 digests. All 15 rows are bitwise identical to the retained pre-patch
outputs and all 15 probe processes returned zero.
Receipt: `build_llama_production_20261023/Testing/ocr-im2col-85e55y1b/vulkan-im2col-model-regression.json`.

**Vulkan detector source-speed recheck (FAIL).** The release-profile detector
measurement (two warmups, ten measured forwards, four host threads, release
timing gate = fail only when the delta exceeds both 5% relative and the 3 ms
absolute noise floor) ran on the same host with both native and Paddle source
GPU audits passing:

| Detector file | Candidate Vulkan p50 / p95 | Paddle CUDA p50 / p95 | Gate |
| --- | ---: | ---: | --- |
| published F16, SHA `82517e16…9550d` | 12.13 / 12.47 ms | 5.98 / 6.63 ms | FAIL (+6.15 ms p50, exceeds both thresholds) |
| repaired Q8, SHA `cab74270…ae56d` | 12.11 / 13.53 ms | 6.40 / 7.49 ms | FAIL (+5.71 ms p50, exceeds both thresholds) |

Tensor comparisons match the CUDA rows (max diff 0.0616 F16 / 0.478 Q8 against
the Paddle graph; the quantized Q8 delta remains diagnostic). The same
detector on the repaired CUDA build passes at p50 8.21 (F16) / 9.45 (Q8) ms, so
the residual gap is specific to Vulkan dense-convolution graph execution — not
to output numerics, not to GPU occupancy audits (both passed), and not to the
Q8 storage. The CUDA-side scheduling fix is therefore necessary but not
sufficient on Vulkan: the ~3.5 ms residual against the pass line (source p50
+ 3 ms) must come from the Vulkan dense-conv lowering path itself. Per the
ggml-upgrade protocol, localize it with Vulkan per-op timing only, then verify
any fix with the normal uninstrumented release-profile gate. The direct
convolution replacement remains disqualified because it moved threshold-
sensitive detector scores across the DB gate.

Receipts: `build_llama_production_20261023/Testing/ocr-im2col-85e55y1b/release-vulkan-det-{f16,q8_0}.json`
and their `.log` files.

**Gate status after this update.** Newly qualified uploads: **0**. Active
per-file blockers: (1) Vulkan detector source-speed gate — open, root cause
localized to dense-conv graph execution; (2) recognizer/page-accuracy rows for
published Q8 and both repair SHAs — incomplete per-file receipts; (3) complete
exact-file page/stability matrix, catalog/digest registration and cold-download
verification — open. CUDA detector speed is qualified; no row above qualifies
any file for upload.
