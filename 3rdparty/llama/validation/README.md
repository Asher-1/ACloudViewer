# Migration verification archive

These twelve JSON files are historical verification receipts, not dependency
configuration, model catalogs or release manifests. Their original bytes and
embedded hashes are preserved. Paths and source hashes inside a receipt refer
to the tree at the time of that experiment; moving the receipt does not make
an old run evidence for the current tree.

| Record | What it establishes |
|---|---|
| [migration-evidence.json](migration-evidence.json) | Original source/patch inventory and isolated CPU contracts |
| [cmake-entry-evidence.json](cmake-entry-evidence.json) | Initial ExternalProject entry and installed loader trace |
| [source-owner-guard-evidence.json](source-owner-guard-evidence.json) | One archive per build/install prefix and collision rejection |
| [cuda-rebase-evidence.json](cuda-rebase-evidence.json) | CUDA custom-operator port, build and linkage |
| [vulkan-rebase-evidence.json](vulkan-rebase-evidence.json) | Vulkan port and selected retained-output comparisons |
| [dual-gpu-evidence.json](dual-gpu-evidence.json) | Combined CUDA/Vulkan build and focused contracts |
| [cuda-native-stride-evidence.json](cuda-native-stride-evidence.json) | FP16 output-stride defect, negative control and focused fix |
| [all-model-regression-scope.json](all-model-regression-scope.json) | Full catalog scope, not a pass result |
| [full-model-regression-evidence.json](full-model-regression-evidence.json) | Running full matrix and unresolved accuracy/performance gates |
| [relocated-patch-replay-evidence.json](relocated-patch-replay-evidence.json) | Original patch bytes replayed after relocation |
| [resource-move-inventory.json](resource-move-inventory.json) | Inventory of moved patch/dependency resources |
| [resource-migration-evidence.json](resource-migration-evidence.json) | Source-owner/resource relocation verification |

Fresh raw probe output belongs under the selected build's `Testing/` directory.
Do not add machine-generated logits or more per-run receipts to the dependency
root. Record current conclusions and reproducible report locations in the
owning task's validation ledger. Build/linkage, developer-profile runs and
incomplete reports do not authorize production promotion or model publication.
