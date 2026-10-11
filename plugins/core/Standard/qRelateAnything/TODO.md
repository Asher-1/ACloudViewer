# RelateAnything integration acceptance

Baseline: local upstream working tree recorded by
`core/AICore/src/tasks/relation/UPSTREAM.json`, commit
`62f8540a2453349b371a51154ac9b5ed69404bfa` plus 19 file hashes.
Implementation and acceptance are tracked separately. A failed, incomplete or
developer-profile report is never a release pass.

## Completed implementation and focused checks

- [x] One AICore runtime; public typed relation/text C ABI; borrowed decoded
  images preserve formats and row stride. Plugins have no private task/ggml
  includes. Strict boundary audit passes.
- [x] Upstream graph, preprocessing, tokenizer, default arithmetic and
  acceleration options are retained. ICU is the Unicode tokenizer dependency;
  production and white-box test archives use portable CMake imported targets.
  Canonical Qt 5/6 Conda environments and Ubuntu/CI build recipes explicitly
  install ICU; existing dependency bundlers handle the linked libraries.
- [x] RelateAnything has no PyTorch/ATen/CUTLASS/mem-efficient-attention
  inference dependency. A dependency-closure gate rejects their reintroduction.
- [x] All 12 published GGUFs (nine relation weights and three text students)
  plus the official tokenizer use shared model catalogs, verified downloads and
  task cache paths. Live published names, sizes and SHA-256 values match the
  built catalog: `build_app/Testing/relate/published-model-audit.json`.
- [x] One shared sample-data button verifies the official archive and fills
  six images and the video file together; extraction/cache repair is shared.
- [x] Icon includes `RelateAnything` and depicts directed scene relations.
- [x] Image/batch, DB handoff, typed exports, consumer-driven video, error
  recovery and close/reopen are implemented and have real Qt workflow probes.
- [x] CPU/CUDA arithmetic changes are owner-scoped; ggml changes are ordered
  manifest patches. Clean forward replay and second-run idempotence pass;
  1,122 source files match the built tree exactly:
  `build_app/Testing/relate/patch-replay.json`.
- [x] Public ABI/lifecycle/catalog contract subset: 25/25 tests pass. Shared
  validation runner: 67 tests pass. Relation upstream audit: 15 tests pass.
- [x] Every successful process repetition is numerically gated; reports keep
  per-attempt metrics and errors. Probe failures name sample/metric/limit and
  cannot pass with missing or nonfinite metrics. Individual timing samples,
  native/candidate p50/p95 and CPU affinity are recorded.

## Remaining acceptance gates

- [ ] CPU: complete 32 default/strict model and pipeline rows on the final
  build with controlled native before/candidate/after timing. The latest
  pinned, affinity-controlled full run completed all 32 rows and kept exact
  native outputs, but 17 rows failed only the per-attempt >5% timing gate
  (`build_app/Testing/relate/cpu-final/validation.json` and `.md`). Follow-up
  isolated runs for the vits text student and vits16-f16 image model recorded
  p50 ratios 0.937 and 1.037 respectively, so the remaining failure is a
  repeatable release-profile evidence gap rather than a missing inference
  path (`build_app/Testing/relate/text-diagnosis/vits-cpu.json`,
  `build_app/Testing/relate/image-diagnosis/vits16-f16-cpu.json`).
- [ ] CUDA: complete the same 32 rows on the final build. Prior model smokes
  are evidence for their tested inputs only; the audited single-sample CUDA
  comparison below does not close the 32-row matrix. A fresh strict attempt
  was stopped after the first rows because each official-reference repetition
  took roughly one minute; it produced no complete report and therefore adds
  no acceptance evidence.
- [ ] Vulkan: complete the same 32 rows on the final build. Prior strict matrix
  contains one failed repetition and is not a complete acceptance pass.
- [x] Real Qt workflow on the current final binary for CPU, CUDA and Vulkan:
  shared image/video sample autofill, real model inference, six-image batch,
  DB handoff, JSON/CSV/text exports, three video frames, invalid-image recovery
  and close/reopen all pass. Current test, plugin and AICore hashes are in
  `build_app/Testing/relate/workflow-final/summary.json`; screenshots, logs and
  exported typed results are in the three backend folders.
- [x] One audited CUDA single-sample upstream comparison passes on the pinned
  native build: exact raw output, typed edge scores within 5e-10, p50/p95
  timing ratios 0.880/0.654. The report records the active-graphics audit and
  retained logs under `build_app/Testing/relate/gpu-audit-final/`.
- [ ] Controlled release-profile regression evidence for affected shared
  arithmetic paths; developer or partial matrices cannot close this gate.
- [ ] Windows and macOS configure/build/runtime verification. ICU target
  portability is implemented; Linux-only execution does not prove these OSes.
- [ ] Physical camera workflow. This host currently exposes no `/dev/video*`
  camera; video-file frame pacing does not verify camera capture.

## Reproduction

Run one inference suite at a time. On the current 5950X Linux host, cores 8–15
are eight distinct physical cores sharing one L3 cache. Pin both candidate and
native children through the same parent; no production runtime defaults change.

```bash
export AICORE_TEST_RELATE_UPSTREAM=/path/to/pinned/RelateAnything-GGML
taskset -c 8-15 \
  python3 core/AICore/scripts/validate_all.py --build build_app \
  --tasks relation --backend cpu --full --threads 8 \
  --repeats 3 --warmup-runs 2 --inference-runs 10 \
  --output build_app/Testing/relate/cpu-final/validation.json
```

Use `--backend cuda` and `--backend vulkan` for the other supported backends.
Use the platform's equivalent affinity tooling on other hosts. For release
evidence, also use `--profile release --baseline-build <accepted-before-build>`
with an independently preserved, qualifying same-host baseline.
