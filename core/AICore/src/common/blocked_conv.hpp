#pragma once
#include "ggml.h"

// Blocked-layout (nChw16c) AVX-512 register-tiled DIRECT convolution island
// ops, shared by every AICore task whose body profits from staying inside one
// blocked buffer (two real consumers: the facedetect ArcFace recognizer and
// the OCR PP-OCR det/rec/layout stacks).
//
// No im2col, no Winograd transform: the kernels stream the activation
// broadcast over a 16-OC weight vector (zmm), register-blocked over
// output-width, FMA over the IC*KH*KW window. Weights are pre-packed
// [OC/16][KH][KW][IC][16] and cached at first use. Runtime CPUID dispatch
// (AVX-512 16c / AVX2 8c / scalar), no global -mavx512f -- ship-safe on
// AVX2-only hosts (mirrors winograd.cpp's function-multiversioning approach).
//
// Layout (ggml ne, fastest dim first), identical contract to winograd_conv3x3:
//   x  : [W, H, IC, N]
//   w  : [3, 3, IC, OC]
//   out: [Wout, Hout, OC, N],  Wout = W + 2*pad - 2, Hout = H + 2*pad - 2
// Only KW==KH==3, stride==1, pad>=1, F32, N==1. Bias/ReLU applied by the caller
// (matching the im2col/direct path); the f32 accumulation order differs from
// im2col so values are within fp32 tolerance, not bitwise identical.
namespace aicore {
namespace blk {

ggml_tensor* directconv_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad);

// ---- blocked-layout (nChw16c) island ops -----------------------------------
// Keep a whole network body in the blocked buffer between ONE reorder-in
// (after the stem) and ONE reorder-out (before the head), amortizing the
// per-conv NCHW<->blocked round-trip. The blocked buffer has ggml ne =
// [16, W, H, CB] (CB = ceil(C/16)); flat index ((cb*H+h)*W+w)*16 + l,
// channel c = cb*16 + l. Padding lanes (c >= C) are zero. The conv kh,kw,ic
// accumulation order matches directconv_conv3x3, so blocked vs NCHW direct
// conv is bit-identical (layout-only change); reorder is an exact permute.
//
//   x  (NCHW)   : [W, H, C, 1]
//   xb (blocked): [16, W, H, CB]
//   w 3x3       : [3, 3, IC, OC]   (ggml weight layout)
//   w 1x1       : [1, 1, IC, OC]
// True iff the blocked island's AVX-512 fast path will run at runtime. On
// non-AVX512 hosts the blocked ops fall back to scalar (slower than the per-conv
// directconv's AVX2 path), so callers default the island OFF unless this is true.
bool blocked_available();

ggml_tensor* blocked_reorder_in (ggml_context* ctx, ggml_tensor* x_nchw);          // -> [16,W,H,CB]
ggml_tensor* blocked_reorder_out(ggml_context* ctx, ggml_tensor* xb, int C);       // -> [W,H,C,1]
// Conv ops optionally FUSE per-channel bias add (+ ReLU) into the register tile,
// removing the separate blocked-bias/blocked-relu passes. bias==nullptr -> raw conv.
ggml_tensor* blocked_conv3x3    (ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb, int pad, int stride,
                                 ggml_tensor* bias = nullptr, bool do_relu = false);
ggml_tensor* blocked_conv1x1    (ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb, int stride,
                                 ggml_tensor* bias = nullptr, bool do_relu = false);
ggml_tensor* blocked_bias       (ggml_context* ctx, ggml_tensor* xb, ggml_tensor* bias);
ggml_tensor* blocked_relu       (ggml_context* ctx, ggml_tensor* xb);
ggml_tensor* blocked_add        (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
ggml_tensor* blocked_add_relu   (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);  // max(0, a+b)
// Per-channel affine ops (broadcast a [C] parameter over the 16-lane layout;
// channel c = cb*16 + l, padding lanes c >= C are passthrough).
//   scale_shift: y = x*scale[c] + shift[c]   (host-folded BatchNorm)
//   prelu      : y = x>0 ? x : alpha[c]*x    (per-channel parametric ReLU)
ggml_tensor* blocked_scale_shift(ggml_context* ctx, ggml_tensor* xb, ggml_tensor* scale, ggml_tensor* shift);
ggml_tensor* blocked_prelu      (ggml_context* ctx, ggml_tensor* xb, ggml_tensor* alpha);

// ---- OCR (PP-OCR) island ops ------------------------------------------------
// The PP-OCR det/rec/layout bodies are LCNet/MobileNet stacks: 3x3/1x1 convs +
// per-channel affine + hard_swish + SE blocks (global avg pool -> 1x1 -> ReLU
// -> 1x1 -> hard_sigmoid -> channel scale). These ops keep that body inside the
// same single blocked island.
//   Depthwise 3x3: per-LANE kernels (channel c convolved by w[c]), per-axis
//     strides 1..2 (the LCNet downsample is e.g. [2,1]), pad 0/1, fused bias.
//     w layout: ggml depthwise kernel [kw, kh, 1, C].
//   hard_swish / hard_sigmoid / silu: whole-buffer elementwise activations
//     (the manual chains the stock path builds collapse to one pass each).
//   mul_channel: y = x * v[c] (SE excitation); v may be a [C] vector or an
//     already-blocked [16,1,1,CB] tensor.
//   gap: global average pool over W,H -> [16,1,1,CB] (SE squeeze).
ggml_tensor* blocked_conv3x3_dw (ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb, int pad,
                                 int stride0, int stride1,
                                 ggml_tensor* bias = nullptr);
ggml_tensor* blocked_hard_swish (ggml_context* ctx, ggml_tensor* xb);
// hard_sigmoid with the graph's slope/offset (MobileNetV3 uses 1/6 + 0.5).
ggml_tensor* blocked_hard_sigmoid(ggml_context* ctx, ggml_tensor* xb,
                                  float slope = 0.2f, float offset = 0.5f);
ggml_tensor* blocked_silu       (ggml_context* ctx, ggml_tensor* xb);
ggml_tensor* blocked_mul_channel(ggml_context* ctx, ggml_tensor* xb, ggml_tensor* v);
ggml_tensor* blocked_gap        (ggml_context* ctx, ggml_tensor* xb);

} // namespace blk
} // namespace aicore
