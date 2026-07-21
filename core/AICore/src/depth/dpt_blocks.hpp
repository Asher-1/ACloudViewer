#pragma once
#include "ggml.h"

// DualDPT (DPT) building blocks as ggml graph helpers.
//
// Tensor layout conventions (ggml ne, fastest dim first):
//   feature maps : x      = [W, H, C, N]
//   conv2d kernel: w      = [KW, KH, IC, OC]  (torch (OC,IC,KH,KW) reversed)
//   convT kernel : w      = [KW, KH, OC, IC]  (torch (IC,OC,KH,KW) reversed)
//   bias         : b      = [OC] (broadcast over spatial via reshape [1,1,OC,1])
namespace aicore {
namespace depth {

// conv2d with optional bias; bias broadcast over spatial. x:[W,H,IC,N] kernel:[KW,KH,IC,OC]
ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x, int stride, int pad);

// conv-transpose, padding 0, given stride. x:[W,H,IC,N] kernel:[KW,KH,OC,IC]
ggml_tensor* conv_transpose2d_p0(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x, int stride);

// bilinear interpolate to (out_w,out_h) with align_corners=true
ggml_tensor* interp_bilinear_ac(ggml_context* ctx, ggml_tensor* x, int out_w, int out_h);

// residual conv unit (relu->conv1->relu->conv2->+x). conv1/conv2 = 3x3 pad1 stride1 WITH bias.
ggml_tensor* residual_conv_unit(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* c1w, ggml_tensor* c1b, ggml_tensor* c2w, ggml_tensor* c2b);

// feature fusion block; lateral may be null (refinenet4).
// out_w/out_h: target size (>0), else pass 0,0 for scale_factor 2.
ggml_tensor* feature_fusion(ggml_context* ctx, ggml_tensor* top, ggml_tensor* lateral,
                            ggml_tensor* rc1c1w, ggml_tensor* rc1c1b, ggml_tensor* rc1c2w, ggml_tensor* rc1c2b,
                            ggml_tensor* rc2c1w, ggml_tensor* rc2c1b, ggml_tensor* rc2c2w, ggml_tensor* rc2c2b,
                            ggml_tensor* outw, ggml_tensor* outb, int out_w, int out_h);

} // namespace depth
} // namespace aicore
