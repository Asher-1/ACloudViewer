---
name: ggml-upgrade
description: ACloudViewer ggml version upgrade verification pipeline. Use when upgrading the pinned ggml version (3rdparty/ggml), verifying upgrade regressions, running the one-click performance A/B, or diagnosing Vulkan DeviceLost / backend parity issues across ggml versions.
---

# ggml Version Upgrade Verification Skill

Single source of truth: [`docs/guides/ggml_upgrade_pipeline.md`](../../../docs/guides/ggml_upgrade_pipeline.md).
One-click tool: [`scripts/ggml_upgrade_verify.py`](../../../scripts/ggml_upgrade_verify.py).

## When to use

- Upgrading `GGML_VERSION` in `3rdparty/ggml/ggml.cmake`.
- Answering "is this ggml upgrade a gain or a regression?".
- Investigating Vulkan DeviceLost / numerical parity failures tied to a ggml version.

## Mandatory workflow (5 stages, fail-stop)

1. **Research** — upstream fixes/breaking changes, API/CMake compatibility, patch-chain dry run.
2. **Experimental tree + patch chain** — bump version in a repo copy; replay patches twice (idempotency).
3. **Contract tier** — `ctest -L capi -LE "model|gpu|e2e"` must be 100%.
4. **Parity/model tier** — CPU-vs-GPU parity gates with GGUF assets.
5. **Performance A/B** — run the one-click script against baseline and candidate build trees:

```bash
python3 scripts/ggml_upgrade_verify.py \
    --baseline <baseline-build-dir> \
    --candidate <candidate-build-dir> \
    --report /tmp/ggml_upgrade_report.md
```

Exit code 0 + `Verdict: NO REGRESSION` is the release gate; regression
threshold defaults to +5% with a 3% noise floor.

## Hard rules

- Never hand-edit ggml sources; every change is a patch in
  `3rdparty/ggml/patches/` registered in `manifest.yaml`
  (see `.agents/rules/acloudviewer-ggml-aicore.mdc`).
- Never declare "fixed"/"regression-free" without a measured number
  (same probe, same asset, same machine, both trees).
- Known pre-existing defects must not be misclassified as upgrade
  regressions — consult the "Known pre-existing issues" table in the pipeline doc
  (e.g. RMBG Vulkan conv2d descriptor assert exists on 0.18.1 AND 0.21;
  0.18.1 Vulkan DeviceLost is fixed by the upgrade to >= 0.19).
- Run the script in a clean shell: stale `AICORE_TEST_*` env vars
  (e.g. a leftover single-model subset dir) silently change asset
  selection; check the `[env]` echo the script prints at startup.
- Any single CPU regression call (>5%) must be re-verified with an
  isolated subset re-measurement before it is written into conclusions
  (background machine load can fake regressions).

## Quick references

| Need | Command |
|---|---|
| Contract tier only | `ctest -L capi -LE "model\|gpu\|e2e" -j4` |
| Model tier (asset-gated) | `ctest -L model -j1` |
| YOLO perf JSON | `AICORE_TEST_YOLO_MODELS_DIR=... test_yolo_capi_performance` |
| SAM3 acceptance | `bench_sam3_backend_acceptance <gguf> <img> [runs] [all\|cpu\|vulkan]` |
| Depth CPU-vs-GPU parity | `AICORE_TEST_DEPTH_GGUF=... AICORE_TEST_DEVICE=vulkan test_depth_capi_backend_parity` |
