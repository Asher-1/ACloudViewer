# RelateAnything assets

Published inventory:
[Asher-1/relateanything-ggml-full/gguf](https://huggingface.co/Asher-1/relateanything-ggml-full/tree/main/gguf).
The public AICore catalog is the source of truth for URLs, exact sizes, SHA-256
values and architecture roles. The plugin reads that catalog.

| Family | Relation weights | Native predicate encoder |
|---|---|---|
| relsgg-vitb16 | F32, F16, Q8_0 | text-student F32 |
| relsgg-vits16 | F32, F16, Q8_0 | text-student F32 |
| relsgg-vits16plus | F32, F16, Q8_0 | text-student F32 |

The default is `relsgg-vits16plus-q8_0.gguf`. The twelve published GGUFs and
pinned official CLIP tokenizer (`relateanything-tokenizer.json`) use `extract/relation_models` in the shared data
root. Detector weights use the existing YOLO cache. The three student files
presently have identical SHA-256 values but retain their published names and
architecture associations.

The tokenizer is `maelic/relsgg-vits16` revision
`282652489584b156d533e704bdbf51bca66f3a90`, SHA-256
`6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35`.
It is shared by all three families. Checkpoint provenance follows the original
model cards. The copied upstream inference source license, AGPL-3.0, is retained
in `core/AICore/src/tasks/relation/LICENSE.upstream`.

The shared test archive is
[relate_datasets.zip](https://huggingface.co/Asher-1/relateanything-ggml-full/resolve/main/assets/relate_datasets.zip),
119359470 bytes, SHA-256
`e0cf31c540dfa4ac8b122e7bc73c7d3070a5d385d17a7017229344a2b7895568`.
It includes six demo JPEGs, five demo WebMs and 100 benchmark samples spanning
RA4M, PSG, IndoorVG, VG150 and the SpatialSense validation/test subsets. The archive stays under `download`, media and
annotations under `extract/relate_datasets`. Consumed member hashes are in the
validation manifest; demo media also use the shared Qt integrity checks. The
fixtures retain original provenance and rights information from upstream.

The published Q8_0 exports now retain `ra.vocab_head.W` in F16. On 2026-10-12,
all nine relation exports were updated in Hugging Face revision
`2377ca3539bf5c4f09161a886a38dd0fc7517bd0`; F16/F32 files also carry corrected
export-policy metadata. Catalog download URLs pin that revision, so a later
upload cannot change the bytes during verification or inference. The shared
downloader replaces outdated cache files after their size or digest fails.
The complete inventory was verified against the
[pinned Hugging Face tree API](https://huggingface.co/api/models/Asher-1/relateanything-ggml-full/tree/2377ca3539bf5c4f09161a886a38dd0fc7517bd0/gguf?recursive=false).
`UPSTREAM.json` records the current inventory, previous Q8 digests and
inference-source hashes. Reports made with the previous weights remain
historical evidence and do not validate the updated exports.

Relation predictions depend on object regions. Quantized models can change
candidate pairs and predicate scores compared with original F32 checkpoints;
validation requires the host to preserve the upstream framework error envelope.
