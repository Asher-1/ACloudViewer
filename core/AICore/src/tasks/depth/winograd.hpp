#pragma once
#include "ggml.h"

// Winograd convolution for the DPT head's 3x3 stride-1 convs.
//
// Implemented as a CPU custom op (ggml_custom_4d) with an AVX-512 winograd-domain
// multiply. Only the F(2x2,3x3) blocked-GEMM variant ships: it reuses each U-row
// across a block of tiles (~5% faster head on BASE, ~22% on GIANT) and is
// parity-identical to the per-tile GEMV. The historical DA_WINO env selector
// (f2/f2b/f4) was A/B scaffolding and is removed.
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
