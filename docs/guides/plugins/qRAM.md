# Recognize Anything (qRAM)

qRAM adds RAM, RAM++, and Tag2Text image tagging to ACloudViewer through the
shared AICore runtime. It can score a single image or a batch, process video
and camera frames, and export complete per-class results.

## Start with the sample

Open **Plugins → Recognize Anything (RAM++)**, select a device, then click
**Use test data**. The button uses ACloudViewer's shared download and extraction
cache for the official `ram_datasets.zip` archive and fills the input list with
`demo/demo1.jpg`. Choose a RAM-family model and click **Run**.

The first run may download the selected GGUF model. qRAM verifies its catalog
SHA-256 before inference. Later runs reuse the shared model cache.

## Models and output

The Model combo lists all nine published models: RAM++, RAM, and Tag2Text,
each in F32, F16, and Q8_0. RAM++ F16 is selected by default. Device choices
are the available Auto, CUDA, Vulkan, Metal, and CPU backends.

The results show each label's score, threshold, and raw logit. By default,
qRAM applies the per-class thresholds stored in the model. Disable **Use model
per-class thresholds** to set one score threshold for the whole request.
Tag2Text models can generate captions, with optional specified tags and beam
execution controls. Custom label lists and thresholds use text files; custom
embeddings use the upstream's headerless float32 format with matching dimensions.
The Tag2Text threshold margin defaults to zero and applies to both selected
tags and caption conditioning; result rows show the effective threshold.
For CPU RAM Q8, **CPU precision → Auto** protects the first two attention
stages from additional activation quantization. Select **Upstream arithmetic**
to reproduce the original CPU quantized results. This option is retained
when reopening the dialog and is identified in JSON exports.

Inputs can come from files, a directory, or selected database images. For video
and cameras, decoded frames advance after inference completes. Export results
as JSON, CSV, or text, or add the current labeled image to the database.

## Build and accuracy checks

Enable `AICore_ENABLED` and `PLUGIN_STANDARD_QRAM`; see the
[plugin developer README](../../../plugins/core/Standard/qRAM/README.md) for
the build command and the CUDA/Vulkan validation commands. The automatic light
validation runs all three Q8_0 families on the official demo image. The full
validation covers all nine models, full-class logits for the exact dequantized
GGUF weights, the upstream's 50 evaluation frames, and all three Tag2Text
caption variants. It records original-GGML timing comparisons and official
PyTorch accuracy independently; see the developer README for reference setup.
Tag2Text's rare, ImageNet, and HICO subsets have no supplied Tag2Text ground
truth; their zero-target scores are recorded as diagnostics. Caption agreement
checks decoder parity with both engines.

Run the documented validation commands after changing the tagging task, plugin
workflow, models, or supported backends.
