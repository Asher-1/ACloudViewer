#include "nn_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rmbg {
namespace {

inline size_t nchw_idx(int n, int c, int h, int w, int C, int H, int W) {
    return (((size_t) n * C + c) * H + h) * W + w;
}

float bilinear_sample(const std::vector<float> & in, int C, int H, int W,
                      int c, float y, float x) {
    if (y <= -1.f || y >= (float) H || x <= -1.f || x >= (float) W) return 0.f;
    const float fy = y;
    const float fx = x;
    const int y0 = (int) std::floor(fy);
    const int x0 = (int) std::floor(fx);
    const int y1 = y0 + 1;
    const int x1 = x0 + 1;
    const float wy1 = fy - (float) y0;
    const float wx1 = fx - (float) x0;
    const float wy0 = 1.0f - wy1;
    const float wx0 = 1.0f - wx1;
    auto at = [&](int yy, int xx) -> float {
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) return 0.0f;
        return in[nchw_idx(0, c, yy, xx, C, H, W)];
    };
    return wy0 * (wx0 * at(y0, x0) + wx1 * at(y0, x1)) +
           wy1 * (wx0 * at(y1, x0) + wx1 * at(y1, x1));
}

} // namespace

void conv2d_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                 const Conv2dParams & p, int stride, int pad,
                 std::vector<float> & out) {
    const int OC = p.oc, KH = p.kh, KW = p.kw;
    const int OH = (H + 2 * pad - KH) / stride + 1;
    const int OW = (W + 2 * pad - KW) / stride + 1;
    out.assign((size_t) N * OC * OH * OW, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float s = (!p.bias.empty()) ? p.bias[oc] : 0.f;
                    for (int ic = 0; ic < C; ++ic) {
                        for (int kh = 0; kh < KH; ++kh) {
                            for (int kw = 0; kw < KW; ++kw) {
                                const int ih = oh * stride + kh - pad;
                                const int iw = ow * stride + kw - pad;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                const size_t wi = (((size_t) oc * C + ic) * KH + kh) * KW + kw;
                                s += in[nchw_idx(n, ic, ih, iw, C, H, W)] * p.weight[wi];
                            }
                        }
                    }
                    out[nchw_idx(n, oc, oh, ow, OC, OH, OW)] = s;
                }
            }
        }
    }
}

void batch_norm2d_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                       const BatchNorm2dParams & p, std::vector<float> & out) {
    out.resize(in.size());
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float scale = p.weight[c] / std::sqrt(p.running_var[c] + p.eps);
            const float shift = p.bias[c] - p.running_mean[c] * scale;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    const size_t i = nchw_idx(n, c, h, w, C, H, W);
                    out[i] = in[i] * scale + shift;
                }
            }
        }
    }
}

void relu_inplace(std::vector<float> & x) {
    for (float & v : x) v = v > 0.f ? v : 0.f;
}

void sigmoid_inplace(std::vector<float> & x) {
    for (float & v : x) v = 1.f / (1.f + std::exp(-v));
}

void bilinear_resize_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                          int OH, int OW, std::vector<float> & out) {
    out.resize((size_t) N * C * OH * OW);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                const float sy = (OH == 1 || H == 1) ? 0.f :
                    (float) oh * (float) (H - 1) / (float) (OH - 1);
                for (int ow = 0; ow < OW; ++ow) {
                    const float sx = (OW == 1 || W == 1) ? 0.f :
                        (float) ow * (float) (W - 1) / (float) (OW - 1);
                    out[nchw_idx(n, c, oh, ow, C, OH, OW)] =
                        bilinear_sample(in, C, H, W, c, sy, sx);
                }
            }
        }
    }
}

void concat_nchw_channel(const std::vector<float> & a, int Ca,
                         const std::vector<float> & b, int Cb,
                         int H, int W, std::vector<float> & out) {
    const int C = Ca + Cb;
    out.resize((size_t) C * H * W);
    for (int c = 0; c < Ca; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                out[nchw_idx(0, c, h, w, C, H, W)] = a[nchw_idx(0, c, h, w, Ca, H, W)];
    for (int c = 0; c < Cb; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                out[nchw_idx(0, Ca + c, h, w, C, H, W)] = b[nchw_idx(0, c, h, w, Cb, H, W)];
}

void deform_conv2d_nchw(const std::vector<float> & in, int N, int C, int H, int W,
                        const DeformConv2dParams & p,
                        std::vector<float> & out) {
    const Conv2dParams & rw = p.regular;
    const int OC = rw.oc, KH = rw.kh, KW = rw.kw;
    const int pad = p.padding;
    const int stride = 1;
    const int OH = (H + 2 * pad - KH) / stride + 1;
    const int OW = (W + 2 * pad - KW) / stride + 1;

    std::vector<float> offset, modulator;
    conv2d_nchw(in, N, C, H, W, p.offset, 1, pad, offset);
    conv2d_nchw(in, N, C, H, W, p.modulator, 1, pad, modulator);
    for (float & v : modulator) v = 2.f * (1.f / (1.f + std::exp(-v)));

    out.assign((size_t) N * OC * OH * OW, 0.f);
    const int off_c = 2 * KH * KW;
    const int mod_c = KH * KW;

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float s = (!rw.bias.empty()) ? rw.bias[oc] : 0.f;
                    for (int ic = 0; ic < C; ++ic) {
                        for (int kh = 0; kh < KH; ++kh) {
                            for (int kw = 0; kw < KW; ++kw) {
                                const int k = kh * KW + kw;
                                const float off_y = offset[nchw_idx(n, 2 * k + 0, oh, ow, off_c, OH, OW)];
                                const float off_x = offset[nchw_idx(n, 2 * k + 1, oh, ow, off_c, OH, OW)];
                                const float mask = modulator[nchw_idx(n, k, oh, ow, mod_c, OH, OW)];
                                const float ih = (float) (oh * stride + kh - pad) + off_y;
                                const float iw = (float) (ow * stride + kw - pad) + off_x;
                                const float v = bilinear_sample(in, C, H, W, ic, ih, iw);
                                const size_t wi = (((size_t) oc * C + ic) * KH + kh) * KW + kw;
                                s += v * rw.weight[wi] * mask;
                            }
                        }
                    }
                    out[nchw_idx(n, oc, oh, ow, OC, OH, OW)] = s;
                }
            }
        }
    }
}

bool load_conv2d(const WeightMap & w, const std::string & prefix, Conv2dParams & out,
                 std::string & err) {
    const std::vector<float> * wt = w.get_f32((prefix + "weight").c_str());
    if (!wt) { err = "missing " + prefix + "weight"; return false; }
    const std::vector<int64_t> * sh = w.get_shape((prefix + "weight").c_str());
    if (!sh || sh->empty()) { err = "bad conv shape " + prefix; return false; }
    if (sh->size() == 4) {
        out.oc = (int) (*sh)[0];
        out.ic = (int) (*sh)[1];
        out.kh = (int) (*sh)[2];
        out.kw = (int) (*sh)[3];
    } else if (sh->size() == 3) {
        out.ic = (int) (*sh)[0];
        out.kh = (int) (*sh)[1];
        out.kw = (int) (*sh)[2];
        const size_t denom = (size_t) out.ic * out.kh * out.kw;
        out.oc = denom ? (int) (wt->size() / denom) : 0;
    } else {
        err = "bad conv shape " + prefix;
        return false;
    }
    out.weight = *wt;
    out.bias.clear();
    if (const std::vector<float> * b = w.get_f32((prefix + "bias").c_str())) out.bias = *b;
    return true;
}

bool load_batch_norm2d(const WeightMap & w, const std::string & prefix, BatchNorm2dParams & out,
                       std::string & err) {
    const std::vector<float> * wt = w.get_f32((prefix + "weight").c_str());
    const std::vector<float> * bs = w.get_f32((prefix + "bias").c_str());
    const std::vector<float> * mn = w.get_f32((prefix + "running_mean").c_str());
    const std::vector<float> * vr = w.get_f32((prefix + "running_var").c_str());
    if (!wt) { err = "missing " + prefix + "weight"; return false; }
    if (!bs) { err = "missing " + prefix + "bias"; return false; }
    if (!mn) { err = "missing " + prefix + "running_mean"; return false; }
    if (!vr) { err = "missing " + prefix + "running_var"; return false; }
    out.weight = *wt;
    out.bias = *bs;
    out.running_mean = *mn;
    out.running_var = *vr;
    return true;
}

bool load_deform_conv2d(const WeightMap & w, const std::string & prefix, DeformConv2dParams & out,
                        std::string & err) {
    out.padding = 0;
    if (!load_conv2d(w, prefix + "atrous_conv_offset_conv_", out.offset, err)) return false;
    if (!load_conv2d(w, prefix + "atrous_conv_modulator_conv_", out.modulator, err)) return false;
    if (!load_conv2d(w, prefix + "atrous_conv_regular_conv_", out.regular, err)) return false;
    if (out.regular.kh > 1) out.padding = out.regular.kh / 2;
    return true;
}

static bool aspp_branch(const std::vector<float> & in, int N, int C, int H, int W,
                        const WeightMap & w, const std::string & prefix,
                        std::vector<float> & out, std::string & err) {
    DeformConv2dParams dc;
    BatchNorm2dParams bn;
    if (!load_deform_conv2d(w, prefix, dc, err)) return false;
    if (!load_batch_norm2d(w, prefix + "bn_", bn, err)) return false;
    std::vector<float> tmp;
    deform_conv2d_nchw(in, N, C, H, W, dc, tmp);
    batch_norm2d_nchw(tmp, N, dc.regular.oc, H, W, bn, out);
    relu_inplace(out);
    return true;
}

bool aspp_deform_branch_forward(const std::vector<float> & in, int N, int C, int H, int W,
                                const WeightMap & w, const std::string & prefix,
                                std::vector<float> & out, std::string & err) {
    return aspp_branch(in, N, C, H, W, w, prefix, out, err);
}

bool aspp_deformable_forward(const std::vector<float> & in, int N, int C, int H, int W,
                             const WeightMap & w, const std::string & prefix,
                             std::vector<float> & out, std::string & err) {
    std::vector<float> b1, b0, b1d, b2, gap;
    if (!aspp_branch(in, N, C, H, W, w, prefix + "aspp1_", b1, err)) return false;
    if (!aspp_branch(in, N, C, H, W, w, prefix + "aspp_deforms_0_", b0, err)) return false;
    if (!aspp_branch(in, N, C, H, W, w, prefix + "aspp_deforms_1_", b1d, err)) return false;
    if (!aspp_branch(in, N, C, H, W, w, prefix + "aspp_deforms_2_", b2, err)) return false;

    std::vector<float> pooled((size_t) N * C, 0.f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            double acc = 0.0;
            for (int h = 0; h < H; ++h)
                for (int w_ = 0; w_ < W; ++w_)
                    acc += in[nchw_idx(n, c, h, w_, C, H, W)];
            pooled[(size_t) n * C + c] = (float) (acc / (H * W));
        }
    }
    Conv2dParams gconv;
    BatchNorm2dParams gbn;
    if (!load_conv2d(w, prefix + "global_avg_pool_1_", gconv, err)) return false;
    if (!load_batch_norm2d(w, prefix + "global_avg_pool_2_", gbn, err)) return false;
    std::vector<float> g1;
    conv2d_nchw(pooled, N, C, 1, 1, gconv, 1, 0, g1);
    batch_norm2d_nchw(g1, N, gconv.oc, 1, 1, gbn, gap);
    relu_inplace(gap);
    std::vector<float> gap_up;
    bilinear_resize_nchw(gap, N, gconv.oc, 1, 1, H, W, gap_up);
    gap = std::move(gap_up);

    const int branch_c = (int) (b1.size() / ((size_t) std::max(1, N * H * W)));
    const int cat_c = branch_c * 5;
    std::vector<float> cat((size_t) N * cat_c * H * W, 0.f);
    const std::vector<float> * branches[5] = {&b1, &b0, &b1d, &b2, &gap};
    for (int bi = 0; bi < 5; ++bi) {
        const std::vector<float> & src = *branches[bi];
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < branch_c; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w_ = 0; w_ < W; ++w_) {
                        cat[nchw_idx(n, bi * branch_c + c, h, w_, cat_c, H, W)] =
                            src[nchw_idx(n, c, h, w_, branch_c, H, W)];
                    }
                }
            }
        }
    }

    Conv2dParams fuse;
    BatchNorm2dParams fuse_bn;
    if (!load_conv2d(w, prefix + "conv1_", fuse, err)) return false;
    if (!load_batch_norm2d(w, prefix + "bn1_", fuse_bn, err)) return false;
    std::vector<float> fused;
    conv2d_nchw(cat, N, cat_c, H, W, fuse, 1, 0, fused);
    batch_norm2d_nchw(fused, N, fuse.oc, H, W, fuse_bn, out);
    relu_inplace(out);
    return true;
}

bool basic_dec_blk_forward(const std::vector<float> & in, int N, int C, int H, int W,
                           const WeightMap & w, const std::string & prefix,
                           std::vector<float> & out, std::string & err) {
    Conv2dParams cin, cout;
    BatchNorm2dParams bn_in, bn_out;
    if (!load_conv2d(w, prefix + "conv_in_", cin, err)) return false;
    if (!load_batch_norm2d(w, prefix + "bn_in_", bn_in, err)) return false;
    if (!load_conv2d(w, prefix + "conv_out_", cout, err)) return false;
    if (!load_batch_norm2d(w, prefix + "bn_out_", bn_out, err)) return false;

    std::vector<float> x;
    conv2d_nchw(in, N, C, H, W, cin, 1, 1, x);
    batch_norm2d_nchw(x, N, cin.oc, H, W, bn_in, x);
    relu_inplace(x);

    std::vector<float> att;
    if (!aspp_deformable_forward(x, N, cin.oc, H, W, w, prefix + "dec_att_", att, err)) return false;

    conv2d_nchw(att, N, cin.oc, H, W, cout, 1, 1, out);
    batch_norm2d_nchw(out, N, cout.oc, H, W, bn_out, out);
    return true;
}

bool lateral_block_forward(const std::vector<float> & in, int N, int C, int H, int W,
                           const WeightMap & w, const std::string & prefix,
                           std::vector<float> & out, std::string & err) {
    Conv2dParams p;
    if (!load_conv2d(w, prefix + "conv_", p, err)) return false;
    conv2d_nchw(in, N, C, H, W, p, 1, 0, out);
    return true;
}

bool simple_convs_forward(const std::vector<float> & in, int N, int C, int H, int W,
                            const WeightMap & w, const std::string & prefix,
                            std::vector<float> & out, std::string & err) {
    Conv2dParams c1, c2;
    if (!load_conv2d(w, prefix + "conv1_", c1, err)) return false;
    if (!load_conv2d(w, prefix + "conv_out_", c2, err)) return false;
    std::vector<float> x;
    conv2d_nchw(in, N, C, H, W, c1, 1, 1, x);
    conv2d_nchw(x, N, c1.oc, H, W, c2, 1, 1, out);
    return true;
}

void image2patches_split_nchw(const std::vector<float> & image, int H, int W,
                              int patch_h, int patch_w,
                              std::vector<float> & patches) {
    const int gh = H / patch_h, gw = W / patch_w;
    const int ph = patch_h, pw = patch_w;
    const int oc = 3 * gh * gw;
    patches.assign((size_t) oc * ph * pw, 0.f);
    for (int hi = 0; hi < ph; ++hi) {
        for (int wi = 0; wi < pw; ++wi) {
            for (int hg = 0; hg < gh; ++hg) {
                for (int wg = 0; wg < gw; ++wg) {
                    for (int c = 0; c < 3; ++c) {
                        const int och = c * gh * gw + hg * gw + wg;
                        const int ih = hg * ph + hi, iw = wg * pw + wi;
                        patches[nchw_idx(0, och, hi, wi, oc, ph, pw)] =
                            image[nchw_idx(0, c, ih, iw, 3, H, W)];
                    }
                }
            }
        }
    }
}

void add_nchw(const std::vector<float> & a, const std::vector<float> & b,
              int C, int H, int W, std::vector<float> & out) {
    out.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
}

bool gdt_attn_forward(const std::vector<float> & in, int N, int C, int H, int W,
                      const WeightMap & w, const std::string & gdt_prefix,
                      const std::string & attn_prefix,
                      std::vector<float> & out, std::string & err) {
    Conv2dParams g0, attn;
    BatchNorm2dParams gbn;
    if (!load_conv2d(w, gdt_prefix + "0_", g0, err)) return false;
    if (!load_batch_norm2d(w, gdt_prefix + "1_", gbn, err)) return false;
    if (!load_conv2d(w, attn_prefix + "0_", attn, err)) return false;

    std::vector<float> g;
    conv2d_nchw(in, N, C, H, W, g0, 1, 1, g);
    batch_norm2d_nchw(g, N, g0.oc, H, W, gbn, g);
    relu_inplace(g);

    std::vector<float> attn_map;
    conv2d_nchw(g, N, g0.oc, H, W, attn, 1, 0, attn_map);
    sigmoid_inplace(attn_map);

    out.resize(in.size());
    const int out_c = attn.oc;
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w_ = 0; w_ < W; ++w_) {
                const size_t i = nchw_idx(0, c, h, w_, C, H, W);
                const float m = attn_map[nchw_idx(0, 0, h, w_, out_c, H, W)];
                out[i] = in[i] * m;
            }
        }
    }
    return true;
}

bool squeeze_module_forward(const std::vector<float> & in, int N, int C, int H, int W,
                            const WeightMap & w,
                            std::vector<float> & out, std::string & err) {
    return basic_dec_blk_forward(in, N, C, H, W, w, "sq0_", out, err);
}

} // namespace rmbg
