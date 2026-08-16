#include "birefnet_decoder.hpp"
#include "nn_ops.hpp"

namespace rmbg {

static bool mul_scl_cat(std::vector<float> & full, const std::vector<float> & half_up,
                        int Cf, int Ch, int H, int W, std::string & err) {
    if ((int) full.size() != Cf * H * W || (int) half_up.size() != Ch * H * W) {
        err = "mul_scl_cat shape mismatch";
        return false;
    }
    std::vector<float> merged;
    concat_nchw_channel(full, Cf, half_up, Ch, H, W, merged);
    full = std::move(merged);
    return true;
}

bool forward_encoder_4scale(const std::vector<float> & nchw_in, int H, int W,
                            SwinBackboneForward & bb,
                            Encoder4ScaleOutput & out,
                            std::string & err) {
    if (!bb.forward_bb_four_scales(nchw_in, H, W, out.x1, out.x2, out.x3, out.x4, err))
        return false;

    const int h4 = H / 32, w4 = W / 32;
    const int h1 = H / 4, w1 = W / 4;
    const int h2 = H / 8, w2 = W / 8;
    const int h3 = H / 16, w3 = W / 16;

    std::vector<float> half_in;
    bilinear_resize_nchw(nchw_in, 1, 3, H, W, H / 2, W / 2, half_in);
    Encoder4ScaleOutput half;
    if (!bb.forward_bb_four_scales(half_in, H / 2, W / 2, half.x1, half.x2, half.x3, half.x4, err))
        return false;

    std::vector<float> u1, u2, u3, u4;
    bilinear_resize_nchw(half.x1, 1, 192, h1 / 2, w1 / 2, h1, w1, u1);
    bilinear_resize_nchw(half.x2, 1, 384, h2 / 2, w2 / 2, h2, w2, u2);
    bilinear_resize_nchw(half.x3, 1, 768, h3 / 2, w3 / 2, h3, w3, u3);
    bilinear_resize_nchw(half.x4, 1, 1536, h4 / 2, w4 / 2, h4, w4, u4);

    if (!mul_scl_cat(out.x1, u1, 192, 192, h1, w1, err)) return false;
    if (!mul_scl_cat(out.x2, u2, 384, 384, h2, w2, err)) return false;
    if (!mul_scl_cat(out.x3, u3, 768, 768, h3, w3, err)) return false;
    if (!mul_scl_cat(out.x4, u4, 1536, 1536, h4, w4, err)) return false;

    std::vector<float> x1u, x2u, x3u;
    bilinear_resize_nchw(out.x1, 1, 384, h1, w1, h4, w4, x1u);
    bilinear_resize_nchw(out.x2, 1, 768, h2, w2, h4, w4, x2u);
    bilinear_resize_nchw(out.x3, 1, 1536, h3, w3, h4, w4, x3u);

    std::vector<float> cat13;
    concat_nchw_channel(x1u, 384, x2u, 768, h4, w4, cat13);
    std::vector<float> cat123;
    concat_nchw_channel(cat13, 384 + 768, x3u, 1536, h4, w4, cat123);
    std::vector<float> x4_ctx;
    concat_nchw_channel(cat123, 384 + 768 + 1536, out.x4, 3072, h4, w4, x4_ctx);
    out.x4 = std::move(x4_ctx);
    return true;
}

} // namespace rmbg
