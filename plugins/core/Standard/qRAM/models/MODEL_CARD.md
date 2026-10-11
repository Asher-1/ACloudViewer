# qRAM model and sample data card

| Field | Value |
|---|---|
| Models | Recognize Anything (RAM), RAM++, Tag2Text |
| Input | RGB image, Pillow-compatible antialiased bilinear resize to 384×384, ImageNet normalization |
| Outputs | Every class logit, sigmoid score, per-class threshold, selected tags; Tag2Text caption and optional specified-tag conditioning |
| Label counts | RAM / RAM++: 4,585; Tag2Text: 3,429 |
| Formats | All nine published GGUFs: each family in F32, F16, Q8_0 |
| Published inventory | [Asher-1/RAM_GGUF](https://huggingface.co/Asher-1/RAM_GGUF/tree/main) |
| Source | [recognize-anything-ggml](https://github.com/Asher-1/recognize-anything-ggml), commit `136d09dff71802bc7b6abedb326785bca8b2fcb4` |
| License | Upstream implementation and published model repository: Apache-2.0; retained source license: `core/AICore/src/tasks/tagging/LICENSE.upstream` |

The authoritative filenames, exact sizes, download URLs and SHA-256 values are
in `core/AICore/src/tasks/tagging/model_catalog.cpp`. The plugin consumes that
catalog through the public C ABI. Weights are verified at ingestion and reused
from `~/cloudViewer_data/extract/ram_models`, using the existing shared downloader.

The official sample archive is
[`ram_datasets.zip`](https://huggingface.co/Asher-1/RAM_GGUF/blob/main/ram_datasets.zip),
21,529,289 bytes, SHA-256
`ee3060f41247895a7140162f8e217a51bf2dd46df8e16679157b8086f7490177`.
`ecvTestDataRepository::Dataset::RAM` owns download and extraction. The **Use test
data** action chooses `ram_datasets/demo/demo1.jpg`; that file is 5,701,095 bytes,
SHA-256 `1b2906f4058a69936df49cb6156ec4cd117a286b420e1eb14764033bf8f3c05f`. The
archive also provides `demo/ram_demo1_to_4.avi`, 262,706 bytes, SHA-256
`53c17ba6b53bde9b21660142a1685bfa8ce4a32d6ab9c0cc08f1c57ce944b202`, used by
**Load sample video**.
Its encoded content is PNG despite the upstream `.jpg` filename, so image I/O
detects the content. Cached sample corruption is repaired from the verified ZIP.

The archive also includes the upstream's selected 50 evaluation frames across
OpenImages common/rare, ImageNet multi-label, and HICO. These are evaluation
subsets, not the complete original training or benchmark datasets. Image rights
remain with the original datasets and image owners; the model license does not
relicense those images.

Numeric regression distinguishes two references: exact-GGUF dequantized PyTorch
logits on demo1, and original official PyTorch checkpoints on the 50 evaluation
frames. Dataset reports include mAP, CP/CR, micro and macro tag metrics, score
errors, original-GGML deltas, and latency samples. Caption reports compare all
50 generated sentences with both original GGML and official PyTorch.
Tag2Text has no supplied Tag2Text ground-truth tags in the rare, ImageNet and
HICO subsets; their zero-target scores are diagnostic values, as in upstream,
and are explicitly marked in the report. Caption scores describe agreement
with the reference decoders rather than human-caption ground truth.
