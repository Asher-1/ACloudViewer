#pragma once
#include "ggml.h"

// Winograd convolution for SCRFD's large early 3x3 stride-1 pad-1 convs.
//
// Implemented as a CPU custom op (ggml_custom_4d) with an AVX2 winograd-domain
// multiply (the production build sets GGML_NATIVE=OFF, so AVX-512 is unavailable
// and undesirable here; this file is arch-flagged -mavx2 -mfma in CMake, with a
// scalar fallback for non-AVX2 hosts). Ported from depth-anything.cpp's
// parity-tested src/winograd.cpp; the algorithm/inner kernel is selectable via
// the FACEDETECT_WINO env var:
//   "f2"  : F(2x2,3x3), per-tile GEMV
//   "f2b" : F(2x2,3x3), blocked GEMM over a block of tiles  <-- default
//           (parity-identical to f2, reuses each U-row across the block)
//   "f4"  : F(4x4,3x3), 4x fewer mults vs direct, blocked GEMM. Less accurate
//           (1/6,1/24 fractions).
//
// Tensor layout (ggml ne, fastest dim first):
//   x : [W, H, IC, N]    input feature map (F32)
//   w : [3, 3, IC, OC]   filter (torch (OC,IC,KH,KW) reversed)  (F32)
//   out: [Wout, Hout, OC, N]  with Wout = W + 2*pad - 2, Hout = H + 2*pad - 2
//
// Only valid for KW==KH==3, stride==1, F32 inputs. `pad` is arbitrary (SCRFD's
// 3x3 backbone/neck convs always use pad=1 -> same-size output).
//
// Bias + ReLU fusion: when `relu` is true the per-channel `bias` (OC floats, may
// be null) and max(0,x) are folded INTO the winograd inverse (output) transform
// store, so the value written to dst is already relu(y + bias[oc]). This removes
// a full bandwidth read+write pass over the [Wout,Hout,OC,N] output (the separate
// ggml_add + ggml_relu the caller would otherwise run). ReLU is exact and the
// bias add is a single f32 add per element, so the fused result is numerically
// identical to the unfused path. When `relu` is false nothing is fused and the
// caller adds bias afterwards via ggml_add (matching the direct/im2col path).
//
// `prefer_f4` requests the F(4x4,3x3) variant for THIS conv (4x fewer multiplies
// than direct vs F2's 2.25x, but worse f32 conditioning from its 1/6,1/24
// transform entries). It is a hint, not a force: the FACEDETECT_WINO env still
// overrides globally (for A/B). When the env is unset, prefer_f4 selects F4, else
// F2B. The caller (fd::conv2d) only sets it for SCRFD's large early maps
// (320/160/80), where the FLOPs concentrate and the per-tile transform overhead
// is best amortized; ArcFace and the small maps keep the parity-exact F(2x2) path.
namespace fd {

ggml_tensor* winograd_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x,
                              int pad, bool prefer_f4 = false,
                              ggml_tensor* bias = nullptr, bool relu = false);

} // namespace fd
