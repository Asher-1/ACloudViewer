#pragma once
#include "ggml.h"

// Winograd convolution for the DPT head's 3x3 stride-1 convs.
//
// Implemented as a CPU custom op (ggml_custom_4d) with an AVX-512 winograd-domain
// multiply. The algorithm/inner kernel is selectable via the DA_WINO env var:
//   "f2"  : F(2x2,3x3), per-tile GEMV (original CPU-opt #4 path)
//   "f2b" : F(2x2,3x3), blocked GEMM over a block of tiles  <-- default
//           (parity-identical to f2, but reuses each U-row across the block ->
//           ~5% faster head on BASE, ~22% on GIANT)
//   "f4"  : F(4x4,3x3), 4x fewer mults vs direct, blocked GEMM. Less accurate
//           (1/6,1/24 fractions); passes parity but is not faster than f2b.
//
// Tensor layout (ggml ne, fastest dim first):
//   x : [W, H, IC, N]    input feature map (F32)
//   w : [3, 3, IC, OC]   filter (torch (OC,IC,KH,KW) reversed)  (F32)
//   out: [Wout, Hout, OC, N]  with Wout = W + 2*pad - 2, Hout = H + 2*pad - 2
//
// Only valid for KW==KH==3, stride==1, F32 inputs. `pad` is arbitrary (the DPT
// head always uses pad=1 -> same-size output). Bias is NOT applied here; add it
// after with ggml_add (matching the direct-conv path).
namespace aicore {
namespace depth {

ggml_tensor* winograd_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad);

} // namespace depth
} // namespace aicore
