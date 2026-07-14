#include "dpt_blocks.hpp"
#include "winograd.hpp"
#include "compute_mode.hpp"
#include <cstdlib>
#include <cstring>

namespace da {

// Reshape an [OC] bias to [1,1,OC,1] so it broadcasts over spatial dims of a
// conv output [W,H,OC,N].
static ggml_tensor* bias_chw(ggml_context* ctx, ggml_tensor* b) {
    return ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
}

ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x,
                    int stride, int pad) {
    // w:[KW,KH,IC,OC] x:[W,H,IC,N] -> [W_out,H_out,OC,N]
    // Direct convolution (no im2col 9x expansion) wins for K>1 kernels — the DPT
    // head's 3x3 convs at full resolution. 1x1 convs are pure GEMM, so keep im2col
    // (llamafile sgemm). A/B toggle via DA_CONV: "direct"|"im2col"|"auto" (default auto).
    const char* mode = std::getenv("DA_CONV");
    const bool kgt1 = (w->ne[0] > 1 || w->ne[1] > 1);
    // Winograd is valid for 3x3 stride-1 F32 convs and is the AUTO default for
    // them: it beats ggml's direct conv on the warm DPT head, parity-exact
    // (max|d|~1.4e-5). The default kernel is the BLOCKED F(2x2) GEMM (DA_WINO=f2b,
    // ~193ms vs direct ~242ms @504 BASE/16t; ~490ms vs ~625ms on GIANT). DA_WINO
    // also selects f2 (per-tile GEMV) or f4 (F(4x4)). See benchmarks/BENCHMARK.md.
    // A/B via DA_CONV: winograd|direct|im2col|auto.
    const bool wino_ok = (w->ne[0] == 3 && w->ne[1] == 3 && stride == 1 &&
                          w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32);
    // GPU mode: route ALL convs through ggml_conv_2d (im2col + mul_mat). On CUDA the
    // mul_mat kernel (cuBLAS-class) is ~2x faster than the basic ggml_conv_2d_direct
    // CUDA kernel (measured on GB10: head 119ms vs 253ms), and the Winograd custom
    // op is CPU-only (would force GPU<->CPU round-trips). On CPU (gpu_mode() false)
    // Winograd stays the auto default for 3x3 — CPU path unchanged.
    const bool gpu = da::gpu_mode();
    bool use_wino = wino_ok && !gpu;     // winograd: CPU 3x3 only
    bool direct   = !gpu && kgt1;        // CPU: direct for K>1 non-winograd; GPU: never (im2col)
    if (mode) {                           // explicit DA_CONV override takes precedence
        if (!std::strcmp(mode,"winograd"))   { use_wino = wino_ok; direct = kgt1; }
        else if (!std::strcmp(mode,"direct")){ use_wino = false; direct = true; }
        else if (!std::strcmp(mode,"im2col")){ use_wino = false; direct = false; }
        // "auto" (or anything else) keeps the gpu/cpu auto default above.
    }
    ggml_tensor* y;
    if (use_wino)      y = winograd_conv3x3(ctx, w, x, pad);
    else if (direct)   y = ggml_conv_2d_direct(ctx, w, x, stride, stride, pad, pad, 1, 1);
    else               y = ggml_conv_2d(ctx, w, x, stride, stride, pad, pad, 1, 1);
    if (b) y = ggml_add(ctx, y, bias_chw(ctx, b));
    return y;
}

ggml_tensor* conv_transpose2d_p0(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                                 ggml_tensor* x, int stride) {
    // ggml_conv_transpose_2d_p0 expects kernel layout (KW,KH,Cout,Cin) == [KW,KH,OC,IC],
    // which is exactly the straight dim-reversal of torch ConvTranspose2d (IC,OC,KH,KW).
    // x:[W,H,IC,N] -> [W_out,H_out,OC,N]
    ggml_tensor* y = ggml_conv_transpose_2d_p0(ctx, w, x, stride);
    if (b) y = ggml_add(ctx, y, bias_chw(ctx, b));
    return y;
}

ggml_tensor* interp_bilinear_ac(ggml_context* ctx, ggml_tensor* x, int out_w, int out_h) {
    return ggml_interpolate(ctx, x, out_w, out_h, x->ne[2], x->ne[3],
                            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
}

ggml_tensor* residual_conv_unit(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* c1w, ggml_tensor* c1b,
                                ggml_tensor* c2w, ggml_tensor* c2b) {
    // out = relu(x); out = conv1(out); out = relu(out); out = conv2(out); return out + x
    ggml_tensor* out = ggml_relu(ctx, x);
    out = conv2d(ctx, c1w, c1b, out, 1, 1);     // 3x3 pad1 stride1
    out = ggml_relu(ctx, out);
    out = conv2d(ctx, c2w, c2b, out, 1, 1);     // 3x3 pad1 stride1
    return ggml_add(ctx, out, x);
}

ggml_tensor* feature_fusion(ggml_context* ctx, ggml_tensor* top, ggml_tensor* lateral,
                            ggml_tensor* rc1c1w, ggml_tensor* rc1c1b, ggml_tensor* rc1c2w, ggml_tensor* rc1c2b,
                            ggml_tensor* rc2c1w, ggml_tensor* rc2c1b, ggml_tensor* rc2c2w, ggml_tensor* rc2c2b,
                            ggml_tensor* outw, ggml_tensor* outb, int out_w, int out_h) {
    ggml_tensor* y = top;
    // Fuse lateral skip connection through resConfUnit1 (only when present).
    if (lateral && rc1c1w) {
        ggml_tensor* res = residual_conv_unit(ctx, lateral, rc1c1w, rc1c1b, rc1c2w, rc1c2b);
        y = ggml_add(ctx, y, res);
    }
    // resConfUnit2
    y = residual_conv_unit(ctx, y, rc2c1w, rc2c1b, rc2c2w, rc2c2b);
    // interpolate: explicit target size, else scale_factor 2.
    int ow = out_w > 0 ? out_w : (int)(y->ne[0] * 2);
    int oh = out_h > 0 ? out_h : (int)(y->ne[1] * 2);
    y = interp_bilinear_ac(ctx, y, ow, oh);
    // out_conv: 1x1 stride1 pad0 WITH bias
    y = conv2d(ctx, outw, outb, y, 1, 0);
    return y;
}

} // namespace da
