# RelateAnything upstream sync — epilogue/direct-bias patch track (design, not started)

Status on 2026-10-25: **design frozen, implementation queued**. This document
is the executable plan for syncing the RelateAnything-GGML upstream commits
(`0ca96f3` "G2 gate closure", HEAD `49207da`) into this repository.

## 1. What upstream changed

- `cpp_ggml/third_party/patches/0001-relateanything-ggml.patch` (+752 lines):
  a new **elementwise epilogue mechanism** — `ggml_elementwise_set_epilogue_hint(t,
  GGML_EPILOGUE_GELU)` marks an ADD so the backend applies `gelu_erf` at the
  add's write-out and skips the standalone gelu node; `ggml_vk_op_f32` gains a
  `dst_out` parameter for the fused write-out; plus acceptance-probe tooling
  and a cuda-f32/vulkan deviation whitelist.
- `cpp_ggml/src/full_model.cpp` (+31 lines): ConvNeXt blocks now (a) pass the
  depthwise bias to the convolution itself (`ggml_conv_2d_dw_direct_bias`)
  instead of a broadcast add, and (b) flag the pwconv1 bias-add with the GELU
  epilogue hint.

## 2. Why it maps onto our existing track

Our 0.27 chain already has the direct-bias precedent
(`ggml_conv_2d_direct_bias_silu` in `0001-acv-cpu-operators.patch` /
`0005-acv-vulkan-operators.patch`), and the F16 Vulkan OCR detector needs a
bias epilogue of the same family. The RelateAnything sync and the OCR F16
work therefore share ONE patch: `0013-acv-elementwise-epilogue.patch`
(working title) providing

1. `ggml_elementwise_set_epilogue_hint()` + `GGML_EPILOGUE_GELU` in ggml.h,
2. CPU / CUDA / Vulkan execution of the hinted epilogue (Vulkan: apply gelu_erf
   at the add's write-out via `dst_out`; CPU: fused elementwise pass; CUDA:
   same, or fall back to separate nodes under the graphs compatibility check),
3. `ggml_conv_2d_dw_direct_bias()` in the direct-conv family (all backends),
followed by the relation `full_model.cpp` host-side port (semantic diff
preserved at `/tmp/relate-sem.diff`, 807 normalized lines; the semantic core is
the two epilogue hints and the dw-bias call).

## 3. Verification gates

- relation exact-file scenarios re-run on CUDA and Vulkan: outputs must stay
  within the upstream G2 deviation whitelist; the checkpoint-accuracy row
  (96.68%) is unaffected by construction (same graph math, fused scheduling).
- OCR detector: the same epilogue patch enables the det bias fusion —
  the F16 Vulkan speed gate (11.49 vs 6.9 ms) is the acceptance metric.
- Full contract tests (17) + selected matrices re-run.

## 4. Sequencing

The patch is a prerequisite for both R3c (F16 Vulkan det gate) and R1
(RelateAnything sync); implement it once, verify both consumers.
