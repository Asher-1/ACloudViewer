# qRelateAnything

Scene relation inference for images, image batches, DB images, video files and
cameras. The plugin detects objects through the public AICore YOLO task, then
uses the public relation task to produce subject–predicate–object edges. The
upstream relation model requires object regions; detector time is reported
separately from relation time.

## Build

```bash
cmake -S . -B build_app -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QRELATEANYTHING=ON -DAICore_BUILD_TESTS=ON
cmake --build build_app --target QRELATEANYTHING_PLUGIN test_relate_workflow -j4
```

ICU (`uc` and `i18n`) is required for the upstream CLIP tokenizer. The tokenizer
does Unicode NFC normalization, root-locale lowercasing, and Unicode regex
matching before BPE, so replacing it with ASCII-only helpers would change the
text embedding contract. CMake uses the portable `ICU::uc` and `ICU::i18n`
imported targets: Linux toolchains need their ICU development package,
macOS toolchains need ICU exposed through the package prefix, and Windows
toolchains need ICU provided by the selected package environment. No platform
uses a hard-coded ICU path. The static `AICore_test` archive links the same
targets because it contains the relation implementation used by white-box
tests.

All supported Qt 5/6 Conda manifests explicitly include `icu` on Linux, macOS
and Windows. The Ubuntu dependency installer, CPU CI job and system-package
build images include `libicu-dev`. Existing dependency bundlers discover ICU
through AICore's linked-library dependencies; no private library-copy path is
needed.

RelateAnything inference itself depends on AICore's ggml target and the three
ordered `relation_merged` custom-operator patches. It does not include
`3rdparty/ggml/relation/pytorch_mem_eff`, ATen, CUTLASS, libtorch, or a PyTorch
runtime. The PyTorch mem-efficient attention headers visible in `ggml.cmake`
belong to the independent SAM3D CUDA task and are not part of this plugin's
dependency closure.

The default arithmetic and acceleration options match upstream. **Use precise model
arithmetic** selects its strict F32 graph, including explicit CUDA attention.
PyTorch is not a runtime dependency of the plugin. CPU, CUDA and Vulkan are
supported by the task.

## Use

Open **Plugins → RelateAnything**. **Try sample data** uses the shared sample-data
button and repository, verifies the published archive, and fills the image list
with six images and the video path with the default WebM on the same click.
Select image rows to process a subset; with no selection all images are processed.
The video tab also accepts other video files and cameras. Playback advances after
inference completes, including cancellation and error paths.

Choose a relation model and detector, then run. All nine relation GGUFs are listed
by the public catalog. Model downloads use the shared verified downloader and
existing task cache directories. Custom predicates are separated by `|`; they
load the matching native text student and pinned official CLIP tokenizer into an
in-memory vocabulary. The three published student GGUFs are supported through
this pipeline and the public text embedding API. **Decompose** returns the
upstream spatial and semantic partitions.

The result table can be filtered. JSON preserves typed object coordinates,
relation indices, scores, model/device/source and stage timings; CSV and text
export readable triplets. Video results are spooled to disk rather than retained
without limit in memory. **Add to DB** stores the annotated image and full JSON
metadata. Closing, cancelling, changing input and reopening share the persistent
worker lifecycle used by qRAM.

## Validation

```bash
export AICORE_TEST_RELATE_UPSTREAM=/path/to/pinned/RelateAnything-GGML
python3 core/AICore/scripts/validate_all.py --build build_app \
  --tasks relation --backend cuda --full \
  --output build_app/Testing/relation-cuda.json
QT_QPA_PLATFORM=offscreen ./build_app/bin/plugin_test/test_relate_workflow \
  --download-sample --real --backend cuda \
  --output-dir build_app/Testing/relate/workflow
```

The native baseline is the local upstream source snapshot recorded in
`core/AICore/src/tasks/relation/UPSTREAM.json` (revision
`62f8540a2453349b371a51154ac9b5ed69404bfa` plus the listed working-tree hashes).
The probe rejects changed sources and stale builds.
The LIGHT tier runs 16 default-mode model/pipeline rows; `--full` adds the same
16 rows with strict arithmetic. The gate consumes every published model, compares raw outputs against that native
baseline on three pinned image/region profiles (RA4M, PSG and IndoorVG), plus
independently generated original-checkpoint PyTorch outputs, and
brackets the host forward latency with native runs. Repeated-output stability
checks run separately from the timing loop, keeping full-tensor validation scans
out of the measured forward sequence. Shared-backend locking,
output ownership, borrowed strides, cancellation, shutdown with live sessions,
repeated output stability and GUI/video workflows have focused probes. Reports
include model and binary hashes, options and raw artifacts. A gate invocation is
not evidence until its report passes.
Native and candidate iteration samples, p50/p95 and CPU affinity are retained
so scheduling noise can be distinguished from a repeatable regression. See
[the acceptance ledger](TODO.md) for completed checks and remaining gates.

## Provenance

Private graph/preprocessing/tokenizer code is adapted from
[RelateAnything-GGML](https://github.com/Asher-1/RelateAnything-GGML) at the revision
above. The copied upstream license is AGPL-3.0; see
`core/AICore/src/tasks/relation/LICENSE.upstream`. The tokenizer and Pillow
preprocessing notices are retained alongside the corresponding sources. See [model and data card](models/MODEL_CARD.md).
