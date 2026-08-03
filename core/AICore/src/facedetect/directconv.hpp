#pragma once
#include "ggml.h"

// Blocked-layout (nChw16c) AVX-512 register-tiled DIRECT 3x3 stride-1 convolution
// for the ArcFace recognizer's deep-channel/small-spatial body (the regime where
// it beats Winograd at high thread counts; see fd::conv2d for the routing gate).
// Winograd stays the default for SCRFD's large-spatial detector convs.
// No im2col, no Winograd transform: it
// streams the NCHW activation broadcast over a 16-OC weight vector (zmm),
// register-blocked over output-width, FMA over the IC*KH*KW window. Weights are
// pre-packed [OC/16][KH][KW][IC][16] and cached at first use. Runtime CPUID
// dispatch (AVX-512 16c / AVX2 8c / scalar), no global -mavx512f -- ship-safe on
// AVX2-only hosts (mirrors winograd.cpp's function-multiversioning approach).
//
// Layout (ggml ne, fastest dim first), identical contract to winograd_conv3x3:
//   x  : [W, H, IC, N]
//   w  : [3, 3, IC, OC]
//   out: [Wout, Hout, OC, N],  Wout = W + 2*pad - 2, Hout = H + 2*pad - 2
// Only KW==KH==3, stride==1, pad>=1, F32, N==1. Bias/ReLU applied by the caller
// (matching the im2col/direct path); the f32 accumulation order differs from
// im2col so values are within fp32 tolerance, not bitwise identical.
namespace fd {

ggml_tensor* directconv_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad);

// ---- blocked-layout (nChw16c) island ops -----------------------------------
// Keep the whole ArcFace IResNet recognizer body in the blocked buffer between
// ONE reorder-in (after the stem) and ONE reorder-out (before the output head),
// amortizing the per-conv NCHW<->blocked round-trip the directconv path pays on
// every layer. The blocked buffer has ggml ne = [16, W, H, CB] (CB = ceil(C/16));
// flat index ((cb*H+h)*W+w)*16 + l, channel c = cb*16 + l. Padding lanes (c >= C)
// are zero (the IResNet stages are 64/128/256/512 ch, all multiples of 16, so for
// buffalo_l/m there are none). The conv kh,kw,ic accumulation order matches
// directconv_conv3x3, so blocked vs NCHW direct conv is bit-identical (layout-only
// change); reorder is an exact permute.
//
//   x  (NCHW)   : [W, H, C, 1]
//   xb (blocked): [16, W, H, CB]
//   w 3x3       : [3, 3, IC, OC]   (ggml weight layout)
//   w 1x1       : [1, 1, IC, OC]
// True iff the blocked island's AVX-512 fast path will run at runtime. On
// non-AVX512 hosts the blocked ops fall back to scalar (slower than the per-conv
// directconv's AVX2 path), so callers default the island OFF unless this is true.
bool directconv_blocked_available();

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
// ArcFace-specific blocked ops (NOT in the voice WeSpeaker island, which uses
// plain ReLU + BN-folded-into-bias). The IResNet body uses a STANDALONE
// pre-activation BatchNorm (per-channel affine) and PReLU (per-channel parametric
// slope), so the island needs blocked versions that broadcast a [C] parameter
// over the 16-lane (channel c = cb*16 + l; padding lanes c >= C are passthrough).
//   scale_shift: y = x*scale[c] + shift[c]   (bn1, host-folded gamma/var + beta/mean)
//   prelu      : y = x>0 ? x : alpha[c]*x     (per-channel parametric ReLU)
ggml_tensor* blocked_scale_shift(ggml_context* ctx, ggml_tensor* xb, ggml_tensor* scale, ggml_tensor* shift);
ggml_tensor* blocked_prelu      (ggml_context* ctx, ggml_tensor* xb, ggml_tensor* alpha);

} // namespace fd
