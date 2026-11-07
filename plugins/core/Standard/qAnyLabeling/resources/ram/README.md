# RAM display vocabulary provenance

These two unmodified vocabulary files are copied from X-AnyLabeling-GGML
`745f6c355f94cddb06f6965b3a4118f82033e935` (v4.1.3),
`anylabeling/services/auto_labeling/configs/ram/`, under GPL-3.0-only.
See the plugin's UPSTREAM-LICENSE and upstream attribution.

They map English inference labels to the source Chinese presentation labels;
model inference, label spaces and thresholds remain owned by AICore. Keep
line endings and whitespace intact to preserve the upstream strings.

Current catalog RAM/RAM++ GGUFs contain `ram.tag_list` (English) and
`ram.tag_thresholds`, but no Chinese vocabulary. These resources supply only
the upstream Chinese presentation mapping. The English file is its lookup
key table, not an inference vocabulary override. Matching by English name
also works when a model has a different ordering or a Tag2Text/custom label
space; unknown names remain English. Removing the pair would remove Chinese
output parity. If published GGUFs later include translations, the public
tagging API should expose those before this fallback is removed.

Both files are compiled by Qt RCC into the plugin via `qAnyLabeling.qrc`.
`AddPlugin` registers that binary with the central `InstallPlugins` rules;
installers and packages carry the resources inside the `.so`/`.dylib`/`.dll`.
No runtime path refers to this source folder, an install prefix, or the current
working directory. Do not add a second external resource-copy/install step.
`test_qanylabeling_plugin` copies the actual plugin to a temporary Unicode path,
changes its working directory, loads it without linking this resource bundle
into the test, and verifies both embedded SHA-256 digests. This exercises
Linux relocation; Windows/macOS packages still require native CI execution.

- `ram_tag_list.txt`: SHA-256 `5d96d2f7225527480ad773aaac76b6e5c08a18adf6eae9ca7d2ab923d5e7bd8b`
- `ram_tag_list_chinese.txt`: SHA-256 `c912657a0f30be7b2f5df6ccffa17e1d84561d0bea508528e38b6d0d711bcfbe`
