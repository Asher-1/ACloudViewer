// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// InstantMesh end-to-end pipeline implementation. See instantmesh_pipeline.hpp
// for the data-flow description. Model stages follow the upstream port
// (InstantMesh-GGML cpp_ggml/src/tools/{instantmesh,zero123pp}.cpp), with
// conditioning inputs checked against the independent framework processor.
// Backend leases, RMBG delegation and typed output/export use AICore owners.

#include "instantmesh_pipeline.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>

#include "aicore/rmbg_capi.h"
#include "aicore/runtime_capi.h"
#include "clip_vision.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/gguf_file_io.hpp"
#include "common/image_utils.hpp"
#include "dino.hpp"
#include "flexicubes.hpp"
#include "ggml-backend.h"
#include "gguf_io.hpp"
#include "lrm_transformer.hpp"
#include "scheduler.hpp"
#include "synthesizer.hpp"
#include "texture_map.hpp"
#include "unet.hpp"
#include "vae.hpp"

namespace instantmesh {

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

void set_error(std::string *error, const std::string &msg) {
    if (error) *error = msg;
}

// ---------------------------------------------------------------------------
// The shared lease retains the canonical device ID; backend display names are
// report-only, not task inputs.
// ---------------------------------------------------------------------------
aicore::runtime::BackendLease init_backend(std::string &name_out,
                                           const std::string &device,
                                           int threads) {
    std::string error;
    auto lease =
            aicore::runtime::acquire_backend_lease(device, threads, &error);
    name_out = lease ? ggml_backend_dev_name(
                               ggml_backend_get_device(lease.handle()))
                     : "unavailable";
    return lease;
}

// ---------------------------------------------------------------------------
// Deterministic CPU float32 noise: MT19937 low-24-bit uniforms and the
// independent framework's 16-value Box-Muller pairing. No framework runtime is
// required.
// ---------------------------------------------------------------------------
struct Rng {
    std::mt19937 g;
    explicit Rng(uint32_t seed) : g(seed) {}
    float uniform() { return float(g() & 0xffffffu) * (1.f / 16777216.f); }
    void randn(float *out, size_t n) {
        // All diffusion tensors (38400 or B*16384 floats) consist of complete
        // contiguous blocks; small/noncontiguous tensor sampling is not this
        // contract.
        assert(out && n >= 16 && n % 16 == 0);
        for (size_t i = 0; i < n; ++i) out[i] = uniform();
        const float two_pi = 6.28318530717958647692f;
        for (size_t k = 0; k < n; k += 16)
            for (size_t j = 0; j < 8; ++j) {
                const float r = std::sqrt(-2.f * std::log(1.f - out[k + j]));
                const float th = two_pi * out[k + j + 8];
                out[k + j] = r * std::cos(th);
                out[k + j + 8] = r * std::sin(th);
            }
    }
};

// ---------------------------------------------------------------------------
// Image preprocessing (mirrors zero123pp.cpp / transformers CLIPImageProcessor:
// bicubic shortest-edge resize -> center crop -> normalize).
// ---------------------------------------------------------------------------
float cubic_w(float x) {
    const float a = -0.5f;
    x = std::fabs(x);
    if (x < 1.f) return (a + 2.f) * x * x * x - (a + 3.f) * x * x + 1.f;
    if (x < 2.f) return a * x * x * x - 5.f * a * x * x + 8.f * a * x - 4.f * a;
    return 0.f;
}

void bicubic_resize(const std::vector<float> &src,
                    int sh,
                    int sw,
                    int sc,
                    std::vector<float> &dst,
                    int dh,
                    int dw) {
    dst.assign((size_t)dh * dw * sc, 0.f);
    const float sx = (float)sw / (float)dw;
    const float sy = (float)sh / (float)dh;
    for (int y = 0; y < dh; ++y) {
        const float fy = (y + 0.5f) * sy - 0.5f;
        int iy = (int)std::floor(fy);
        const float ty = fy - iy;
        float wy[4];
        for (int j = -1; j <= 2; ++j) wy[j + 1] = cubic_w(ty - j);
        for (int x = 0; x < dw; ++x) {
            const float fx = (x + 0.5f) * sx - 0.5f;
            int ix = (int)std::floor(fx);
            const float tx = fx - ix;
            float wx[4];
            for (int j = -1; j <= 2; ++j) wx[j + 1] = cubic_w(tx - j);
            for (int c = 0; c < sc; ++c) {
                float acc = 0.f, wn = 0.f;
                for (int j = 0; j < 4; ++j) {
                    const int yy = iy + j - 1;
                    if (yy < 0 || yy >= sh) continue;
                    for (int i = 0; i < 4; ++i) {
                        const int xx = ix + i - 1;
                        if (xx < 0 || xx >= sw) continue;
                        const float w = wy[j] * wx[i];
                        acc += w * src[((size_t)yy * sw + xx) * sc + c];
                        wn += w;
                    }
                }
                dst[((size_t)y * dw + x) * sc + c] =
                        (wn != 0.f) ? acc / wn : 0.f;
            }
        }
    }
}

// Zero123++ uint8 input semantics: scaled support, double weights, adaptive
// int16 coefficients and uint8 rounding after each separable pass (torch 2.6).
// Keep the legacy float resizer above for DINO and the compatibility RGB API.
enum class InputFilter { Bilinear, Bicubic };

struct InputAxis {
    struct Span {
        int first;
        std::vector<int16_t> weights;
    };
    unsigned precision = 0;
    std::vector<Span> spans;
};

InputAxis input_axis(int source, int destination, InputFilter filter) {
    const double scale = double(source) / destination;
    const double support = (filter == InputFilter::Bilinear ? 1.0 : 2.0) *
                           std::max(scale, 1.0);
    const double inverse = scale >= 1.0 ? 1.0 / scale : 1.0;
    std::vector<std::vector<double>> normalized(destination);
    InputAxis axis;
    axis.spans.resize(destination);
    double maximum = 0;
    for (int i = 0; i < destination; ++i) {
        const double center = scale * (i + 0.5);
        const int64_t first =
                std::max(int64_t(center - support + 0.5), int64_t(0));
        const int64_t last =
                std::min(int64_t(center + support + 0.5), int64_t(source));
        auto &weights = normalized[i];
        weights.resize(size_t(last - first));
        axis.spans[i].first = int(first);
        double total = 0;
        for (size_t j = 0; j < weights.size(); ++j) {
            const double x =
                    std::abs((double(j) + first - center + 0.5) * inverse);
            double weight = 0;
            if (filter == InputFilter::Bilinear) {
                if (x < 1.0) weight = 1.0 - x;
            } else if (x < 1.0) {
                weight = ((1.5 * x - 2.5) * x) * x + 1.0;
            } else if (x < 2.0) {
                weight = ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
            }
            weights[j] = weight;
            total += weight;
        }
        if (total != 0)
            for (double &weight : weights) {
                weight /= total;
                maximum = std::max(maximum, weight);
            }
    }
    for (; axis.precision < 22; ++axis.precision)
        if (int(0.5 + maximum * (1 << (axis.precision + 1))) >= (1 << 15))
            break;
    for (int i = 0; i < destination; ++i) {
        auto &weights = axis.spans[i].weights;
        weights.reserve(normalized[i].size());
        for (double weight : normalized[i]) {
            const double value = weight * (1 << axis.precision);
            weights.push_back(
                    int16_t(int(value < 0 ? value - 0.5 : value + 0.5)));
        }
    }
    return axis;
}

bool resize_input_aa(const uint8_t *rgb,
                     int sh,
                     int sw,
                     int dh,
                     int dw,
                     InputFilter filter,
                     std::vector<uint8_t> &out) {
    if (!rgb || sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0) return false;
    const size_t limit = std::numeric_limits<size_t>::max() / 3;
    if (size_t(sh) > limit / size_t(sw) || size_t(sh) > limit / size_t(dw) ||
        size_t(dh) > limit / size_t(dw))
        return false;
    if (sh == dh && sw == dw) {
        out.assign(rgb, rgb + size_t(sh) * sw * 3);
        return true;
    }
    auto pixel = [](int64_t sum, unsigned precision) {
        // Negative cubic lobes saturate to zero; division avoids signed-shift
        // portability assumptions.
        return uint8_t(sum <= 0 ? 0
                                : std::min(sum / (int64_t(1) << precision),
                                           int64_t(255)));
    };
    std::vector<uint8_t> horizontal;
    const uint8_t *rows = rgb;
    // O(sh*dw*support_x + dh*dw*support_y): downsampling must integrate the
    // expanded footprint. The intermediate is sh*dw*3 bytes (1.5 MiB at
    // 1024x512).
    if (sw != dw) {
        const auto axis = input_axis(sw, dw, filter);
        horizontal.resize(size_t(sh) * dw * 3);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < dw; ++x)
                for (int c = 0; c < 3; ++c) {
                    const auto &span = axis.spans[x];
                    int64_t sum = int64_t(1) << (axis.precision - 1);
                    for (size_t j = 0; j < span.weights.size(); ++j)
                        sum += rgb[(size_t(y) * sw + span.first + j) * 3 + c] *
                               span.weights[j];
                    horizontal[(size_t(y) * dw + x) * 3 + c] =
                            pixel(sum, axis.precision);
                }
        rows = horizontal.data();
    }
    if (sh == dh) {
        out.assign(rows, rows + size_t(sh) * dw * 3);
    } else {
        const auto axis = input_axis(sh, dh, filter);
        out.resize(size_t(dh) * dw * 3);
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x)
                for (int c = 0; c < 3; ++c) {
                    const auto &span = axis.spans[y];
                    int64_t sum = int64_t(1) << (axis.precision - 1);
                    for (size_t j = 0; j < span.weights.size(); ++j)
                        sum += rows[((size_t(span.first) + j) * dw + x) * 3 +
                                    c] *
                               span.weights[j];
                    out[(size_t(y) * dw + x) * 3 + c] =
                            pixel(sum, axis.precision);
                }
    }
    return true;
}

void center_crop(const std::vector<float> &src,
                 int sh,
                 int sw,
                 int sc,
                 int ch,
                 int cw,
                 std::vector<float> &dst) {
    const int y0 = (sh - ch) / 2, x0 = (sw - cw) / 2;
    dst.assign((size_t)ch * cw * sc, 0.f);
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x)
            for (int c = 0; c < sc; ++c)
                dst[((size_t)y * cw + x) * sc + c] =
                        src[((size_t)(y0 + y) * sw + (x0 + x)) * sc + c];
}

// Borrowed RGBA rows -> packed RGB with the independent pipeline's PIL paste
// rounding.
void composite_rgba_white(const uint8_t *rgba,
                          int h,
                          int w,
                          int stride,
                          std::vector<uint8_t> &rgb) {
    rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint8_t *px = rgba + (size_t)y * stride + (size_t)x * 4;
            const unsigned alpha = px[3];
            for (int c = 0; c < 3; ++c)
                rgb[((size_t)y * w + x) * 3 + c] = uint8_t(
                        (px[c] * alpha + 255u * (255u - alpha) + 127u) / 255u);
        }
}

// Resize shortest edge to `target`, center-crop target x target, then apply
// (x/255 - mean) / std with per-channel mean/std, producing NCHW for both
// encoders.
void preprocess_image(const uint8_t *rgb,
                      int H,
                      int W,
                      int target,
                      const float *mean_ch,
                      const float *std_ch,
                      std::vector<float> &out) {
    const int sc = 3;
    std::vector<float> src((size_t)H * W * sc);
    for (size_t k = 0; k < src.size(); ++k) src[k] = rgb[k];
    const float scale = (float)target / (float)(H < W ? H : W);
    const int nh = (int)std::lround(H * scale),
              nw = (int)std::lround(W * scale);
    std::vector<float> rs, cr;
    bicubic_resize(src, H, W, sc, rs, nh, nw);
    center_crop(rs, nh, nw, sc, target, target, cr);
    out.resize((size_t)target * target * sc);
    for (int y = 0; y < target; ++y)
        for (int x = 0; x < target; ++x)
            for (int c = 0; c < sc; ++c)
                // VAE and CLIP copy this buffer as NCHW; only resized/cropped
                // pixels are HWC.
                out[((size_t)c * target + y) * target + x] =
                        (cr[((size_t)y * target + x) * sc + c] / 255.f -
                         mean_ch[c]) /
                        std_ch[c];
}

// Published Zero123++ conditioning: bilinear VAE, bicubic CLIP, shortest-edge
// integer truncation, center crop and fused byte-domain normalization to NCHW.
bool condition_images(const uint8_t *rgb,
                      int h,
                      int w,
                      int vae_size,
                      int clip_size,
                      std::vector<float> &vae_image,
                      std::vector<float> &clip_image) {
    if (!rgb || h <= 0 || w <= 0 || vae_size <= 0 || clip_size <= 0)
        return false;
    auto prepare = [&](int target, InputFilter filter, const float *mean,
                       const float *deviation, std::vector<float> &image) {
        const int shortest = std::min(h, w);
        const int64_t height = int64_t(target) * h / shortest;
        const int64_t width = int64_t(target) * w / shortest;
        if (height > std::numeric_limits<int>::max() ||
            width > std::numeric_limits<int>::max())
            return false;
        std::vector<uint8_t> resized;
        if (!resize_input_aa(rgb, h, w, int(height), int(width), filter,
                             resized))
            return false;
        const size_t limit =
                std::numeric_limits<size_t>::max() / (3 * sizeof(float));
        if (size_t(target) > limit / size_t(target)) return false;
        image.resize(size_t(target) * target * 3);
        const int64_t top = (height - target) / 2, left = (width - target) / 2;
        for (int c = 0; c < 3; ++c) {
            const float byte_mean = mean[c] * 255.f;
            const float byte_deviation = deviation[c] * 255.f;
            for (int y = 0; y < target; ++y)
                for (int x = 0; x < target; ++x)
                    image[(size_t(c) * target + y) * target + x] =
                            (float(resized[((size_t(top + y) * width) + left +
                                            x) * 3 +
                                           c]) -
                             byte_mean) /
                            byte_deviation;
        }
        return true;
    };
    const float vae_mean[3] = {0.5f, 0.5f, 0.5f},
                vae_std[3] = {0.8f, 0.8f, 0.8f};
    const float clip_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float clip_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    return prepare(vae_size, InputFilter::Bilinear, vae_mean, vae_std,
                   vae_image) &&
           prepare(clip_size, InputFilter::Bicubic, clip_mean, clip_std,
                   clip_image);
}

// ---------------------------------------------------------------------------
// FlexiCubes voxel grid (mirrors fc.construct_voxel_grid in
// flexicubes_geometry.py).
// ---------------------------------------------------------------------------
void construct_voxel_grid(int res,
                          std::vector<float> &verts,
                          std::vector<int32_t> &cubes) {
    static const int corners[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                                      {1, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                      {0, 1, 1}, {1, 1, 1}};
    const long ncell = (long)res * res * res;

    std::vector<std::tuple<long, float, float, float>> items;
    items.reserve(ncell * 8);
    long idx = 0;
    for (int iz = 0; iz < res; ++iz)
        for (int iy = 0; iy < res; ++iy)
            for (int ix = 0; ix < res; ++ix) {
                float bx = (float)ix / res, by = (float)iy / res,
                      bz = (float)iz / res;
                for (int c = 0; c < 8; ++c) {
                    float x = std::roundf((bx + corners[c][0] / (float)res) *
                                          1e5f) /
                              1e5f;
                    float y = std::roundf((by + corners[c][1] / (float)res) *
                                          1e5f) /
                              1e5f;
                    float z = std::roundf((bz + corners[c][2] / (float)res) *
                                          1e5f) /
                              1e5f;
                    items.emplace_back(idx, x, y, z);
                    ++idx;
                }
            }

    auto before = [](const std::tuple<long, float, float, float> &a,
                     const std::tuple<long, float, float, float> &b) {
        if (std::get<1>(a) != std::get<1>(b))
            return std::get<1>(a) < std::get<1>(b);
        if (std::get<2>(a) != std::get<2>(b))
            return std::get<2>(a) < std::get<2>(b);
        return std::get<3>(a) < std::get<3>(b);
    };
    std::sort(items.begin(), items.end(), before);

    std::vector<int> remap(items.size());
    std::vector<float> ux, uy, uz;
    int uid = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const auto &it = items[i];
        if (i == 0 || std::get<1>(it) != std::get<1>(items[i - 1]) ||
            std::get<2>(it) != std::get<2>(items[i - 1]) ||
            std::get<3>(it) != std::get<3>(items[i - 1])) {
            ux.push_back(std::get<1>(it));
            uy.push_back(std::get<2>(it));
            uz.push_back(std::get<3>(it));
            ++uid;
        }
        remap[std::get<0>(it)] = uid - 1;
    }

    verts.resize((size_t)uid * 3);
    for (int i = 0; i < uid; ++i) {
        verts[i * 3] = ux[i] - 0.5f;
        verts[i * 3 + 1] = uy[i] - 0.5f;
        verts[i * 3 + 2] = uz[i] - 0.5f;
    }
    cubes.resize(ncell * 8);
    for (long ic = 0; ic < ncell; ++ic)
        for (int c = 0; c < 8; ++c)
            cubes[ic * 8 + c] = remap[(size_t)ic * 8 + c];
}

// Mirror get_center_boundary_index: indices into the (res+1)^3 vertex grid.
void center_boundary_indices(int res,
                             std::vector<int> &center,
                             std::vector<int> &boundary) {
    const int n = res + 1;
    const int c = res / 2 + 1;
    center.clear();
    center.push_back((c * n + c) * n + c);

    boundary.clear();
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (i < 2 || i > res - 1 || j < 2 || j > res - 1 || k < 2 ||
                    k > res - 1)
                    boundary.push_back((k * n + j) * n + i);
            }
}

struct Vec3 {
    float x, y, z;
};
static Vec3 cross3(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
static Vec3 norm3(Vec3 a) {
    float n = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    if (n > 0) {
        a.x /= n;
        a.y /= n;
        a.z /= n;
    }
    return a;
}

// Build the default zero123plus input cameras [6,16] (radius=4.0, fov=30).
std::vector<float> default_cameras() {
    const float radius = 4.0f;
    const float focal = 0.5f / std::tan((float)M_PI / 6.0f * 0.5f);
    const float az[6] = {30, 90, 150, 210, 270, 330};
    const float el[6] = {20, -10, 20, -10, 20, -10};
    const Vec3 up = {0, 0, 1};
    std::vector<float> cam(6 * 16);
    for (int v = 0; v < 6; ++v) {
        float a = az[v] * (float)M_PI / 180.0f;
        float e = el[v] * (float)M_PI / 180.0f;
        Vec3 pos = {radius * std::cos(e) * std::cos(a),
                    radius * std::cos(e) * std::sin(a), radius * std::sin(e)};
        Vec3 z = norm3(pos);
        Vec3 x = norm3(cross3(up, z));
        Vec3 y = cross3(z, x);
        float *c = &cam[v * 16];
        c[0] = x.x;
        c[1] = y.x;
        c[2] = z.x;
        c[3] = pos.x;
        c[4] = x.y;
        c[5] = y.y;
        c[6] = z.y;
        c[7] = pos.y;
        c[8] = x.z;
        c[9] = y.z;
        c[10] = z.z;
        c[11] = pos.z;
        c[12] = focal;
        c[13] = focal;
        c[14] = 0.5f;
        c[15] = 0.5f;
    }
    return cam;
}

}  // namespace

// ABI-shared preprocessing: bicubic shortest-edge resize + center crop of a
// packed RGB8 buffer into out_rgb (out_size*out_size*3 bytes, RGB8).
bool preprocess_rgb(const uint8_t *rgb,
                    int w,
                    int h,
                    int out_size,
                    unsigned char *out_rgb) {
    if (!rgb || !out_rgb || w <= 0 || h <= 0 || out_size <= 0) return false;
    const int sc = 3;
    std::vector<float> src((size_t)h * w * sc);
    for (size_t k = 0; k < src.size(); ++k) src[k] = rgb[k];
    const float scale = (float)out_size / (float)(h < w ? h : w);
    const int nh = (int)std::lround(h * scale),
              nw = (int)std::lround(w * scale);
    std::vector<float> rs, cr;
    bicubic_resize(src, h, w, sc, rs, nh, nw);
    center_crop(rs, nh, nw, sc, out_size, out_size, cr);
    const size_t n = (size_t)out_size * out_size * sc;
    for (size_t k = 0; k < n; ++k) {
        float v = cr[k];
        v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
        out_rgb[k] = (unsigned char)(v + 0.5f);
    }
    return true;
}

// ---------------------------------------------------------------------------
// PipelineContext::Impl
// ---------------------------------------------------------------------------
struct PipelineContext::Impl {
    aicore::runtime::BackendLease lease;
    aicore::runtime::BackendLease host_storage_lease;
    ggml_backend_t backend = nullptr;
    std::string backend_name = "CPU";

    ClipVisionModel cond;
    UnetModel unet;
    VaeModel vae;
    DinoModel dino;
    LrmTransformerModel trans;
    SynthesizerModel syn;

    aicore_rmbg_ctx *rmbg = nullptr;

    // Zero123++ diffusion availability (all three loaded).
    bool has_cond = false, has_unet = false, has_vae = false;

    // Zero123++ multi-view diffusion: rgb_in (tightly packed RGB8) -> 6 views
    // of [320,320,3] u8. Full diffusion correctness requires independent
    // numeric evidence.
    bool zero123pp_multiview(const PipelineConfig &cfg,
                             int steps,
                             uint64_t seed,
                             float guidance,
                             const uint8_t *rgb,
                             int h,
                             int w,
                             std::vector<uint8_t> &views,
                             ProgressFn progress,
                             void *user,
                             std::string *error);
};

// ---------------------------------------------------------------------------
// Zero123++ multi-view diffusion (upstream graph and independent conditioning
// inputs).
// ---------------------------------------------------------------------------

// rgb_in: tightly packed RGB8 [h,w,3]; output views: 6 x [320,320,3] u8.
bool PipelineContext::Impl::zero123pp_multiview(const PipelineConfig &cfg,
                                                int steps,
                                                uint64_t seed,
                                                float guidance,
                                                const uint8_t *rgb,
                                                int h,
                                                int w,
                                                std::vector<uint8_t> &views,
                                                ProgressFn progress,
                                                void *user,
                                                std::string *error) {
    ClipVisionModel &clip = cond;
    UnetModel &unet = this->unet;
    VaeModel &vae = this->vae;
    const auto canceled = [&]() {
        if (!aicore_cancel_requested()) return false;
        ggml_backend_synchronize(backend);
        set_error(error, "instantmesh: canceled");
        return true;
    };
    if (canceled()) return false;
    // A prior forward may have parked this context's inactive diffusion weights
    // in host storage. Restore them before constructing any graph; inference
    // still uses the requested backend.
    if (!clip.gguf.move_weights(backend, error) ||
        !unet.gguf.move_weights(backend, error) ||
        !vae.gguf.move_weights(backend, error))
        return false;

    // Pipeline constants from the cond GGUF.
    const int crop_vae = (int)kv_i32(clip.gguf.gguf, "fx_vae.crop", 512);
    const int crop_clip = (int)kv_i32(clip.gguf.gguf, "fx_clip.crop", 224);

    std::vector<float> text_emb, neg_lat, alphas;
    auto tt = clip.gguf.tensors.find("cond.text_emb");
    auto tn = clip.gguf.tensors.find("cond.negative_lat");
    auto ta = clip.gguf.tensors.find("scheduler.alphas_cumprod");
    if (tt == clip.gguf.tensors.end() || tn == clip.gguf.tensors.end() ||
        ta == clip.gguf.tensors.end()) {
        set_error(error, "instantmesh: cond GGUF missing pipeline constants");
        return false;
    }
    auto read_tensor = [&](ggml_tensor *t, std::vector<float> &out) {
        out.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    };
    read_tensor(tt->second, text_emb);
    read_tensor(tn->second, neg_lat);
    read_tensor(ta->second, alphas);
    const int L = 77;
    std::vector<float> ramp(L);
    {
        const char *rs = kv_str(clip.gguf.gguf, "cond.ramp", "");
        int n = 0;
        const char *p = rs;
        while (p && *p && n < L) {
            ramp[n++] = (float)std::atof(p);
            p = std::strchr(p, ',');
            if (p) ++p;
        }
        if (n != L) {
            set_error(error, "instantmesh: cond.ramp parse failed");
            return false;
        }
    }

    // Preprocess: VAE branch (crop_vae) and CLIP branch (224).
    std::vector<float> img_vae, img_clip;
    if (!condition_images(rgb, h, w, crop_vae, crop_clip, img_vae, img_clip)) {
        set_error(error, "instantmesh: invalid conditioning image dimensions");
        return false;
    }

    // Condition latent [B,4,64,64]: negative = zero-image latent (GGUF).
    const int B = guidance > 1.f ? 2 : 1;
    int lh = 0, lw = 0;
    float *pos_lat = vae_encode_mode(vae, img_vae.data(), 1, crop_vae, crop_vae,
                                     &lh, &lw);
    if (!pos_lat) {
        set_error(error, "instantmesh: vae_encode_mode failed");
        return false;
    }
    const size_t lat64 = (size_t)4 * 64 * 64;
    std::vector<float> cond_lat((size_t)B * lat64);
    if (B == 2)
        std::memcpy(cond_lat.data(), neg_lat.data(), lat64 * sizeof(float));
    std::memcpy(cond_lat.data() + (size_t)(B - 1) * lat64, pos_lat,
                lat64 * sizeof(float));
    std::free(pos_lat);

    // CLIP vision embeds + context [B,77,1024].
    int emb_dim = 0;
    float *glob = clip_vision_encode(clip, img_clip.data(), 1, &emb_dim);
    if (!glob || emb_dim != 1024) {
        std::free(glob);
        set_error(error, "instantmesh: clip_vision_encode failed");
        return false;
    }
    std::vector<float> context((size_t)B * L * 1024);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < L; ++t)
            for (int d = 0; d < 1024; ++d) {
                float v = text_emb[(size_t)t * 1024 + d];
                if (b == B - 1) v += glob[d] * ramp[t];
                context[((size_t)b * L + t) * 1024 + d] = v;
            }
    std::free(glob);
    // CLIP has no consumers after conditioning. Preserve its exact bytes in
    // this context rather than keeping 2.5 GB of inactive weights beside
    // UNet/decoder graph reservations. No model file is reloaded and no shared
    // backend cache or RMBG context is invalidated.
    if (!lease.is_cpu()) {
        if (!host_storage_lease)
            host_storage_lease = aicore::runtime::acquire_backend_lease(
                    "cpu", cfg.threads, error);
        if (!host_storage_lease ||
            !clip.gguf.move_weights(host_storage_lease.handle(), error))
            return false;
    }

    // Scheduler (config from cond GGUF KV).
    EulerAncestralScheduler::Config scfg;
    scfg.num_train_timesteps =
            (int)kv_i32(clip.gguf.gguf, "sched.num_train_timesteps", 1000);
    scfg.beta_start = kv_f32(clip.gguf.gguf, "sched.beta_start", 0.00085f);
    scfg.beta_end = kv_f32(clip.gguf.gguf, "sched.beta_end", 0.012f);
    scfg.beta_schedule =
            kv_str(clip.gguf.gguf, "sched.beta_schedule", "linear");
    scfg.prediction_type =
            kv_str(clip.gguf.gguf, "sched.prediction_type", "v_prediction");
    scfg.timestep_spacing =
            kv_str(clip.gguf.gguf, "sched.timestep_spacing", "trailing");
    EulerAncestralScheduler sched(scfg);
    sched.set_alphas_cumprod(alphas.data(), (int)alphas.size());
    sched.set_timesteps(steps);

    // Keep cond constants and weights in the context for subsequent forwards
    // and retries after cancellation.

    // Denoising loop.
    const int H = 120, W = 80;
    const int ref_h = 64, ref_w = 64;
    const size_t lat_n = (size_t)B * 4 * H * W / B;
    const size_t lat_total = (size_t)B * 4 * H * W;
    std::vector<float> latents(lat_n);
    Rng rng((uint32_t)seed);
    rng.randn(latents.data(), lat_n);
    const float init_sigma = sched.sigma(0);
    for (size_t k = 0; k < lat_n; ++k) latents[k] *= init_sigma;

    std::vector<float> scaled_in(lat_total), cond_noise((size_t)B * lat64),
            noisy_cond((size_t)B * lat64), scaled_cond((size_t)B * lat64),
            noise(lat_n), eps_buf(lat_total);

    for (int i = 0; i < steps; ++i) {
        if (canceled()) return false;
        const float t = sched.timesteps()[i];
        std::memcpy(scaled_in.data(), latents.data(), lat_n * sizeof(float));
        if (B == 2)
            std::memcpy(scaled_in.data() + lat_n, latents.data(),
                        lat_n * sizeof(float));
        sched.scale_model_input(scaled_in.data(), (int)lat_total,
                                scaled_in.data());

        rng.randn(cond_noise.data(), (size_t)B * lat64);
        sched.add_noise(cond_lat.data(), cond_noise.data(), (int)(B * lat64), t,
                        noisy_cond.data());
        sched.scale_model_input(noisy_cond.data(), (int)(B * lat64),
                                scaled_cond.data());

        int oh = 0, ow = 0;
        float *eps = unet_forward_refonly(unet, scaled_in.data(),
                                          scaled_cond.data(), context.data(), L,
                                          B, H, W, ref_h, ref_w, t, &oh, &ow);
        if (!eps) {
            set_error(error, "instantmesh: unet forward failed at step " +
                                     std::to_string(i));
            return false;
        }
        for (size_t k = 0; k < lat_n; ++k) {
            eps_buf[k] = B == 2 ? eps[k] + guidance * (eps[k + lat_n] - eps[k])
                                : eps[k];
        }
        std::free(eps);

        rng.randn(noise.data(), lat_n);
        sched.step(eps_buf.data(), latents.data(), noise.data(), (int)lat_n,
                   latents.data());

        if (progress) progress(user, 2 /* MULTIVIEW */, i + 1, steps);
        if (canceled()) return false;
    }

    // The context also owns UNet; do not invalidate it after one forward.

    for (size_t k = 0; k < lat_n; ++k) latents[k] = latents[k] / 0.75f + 0.22f;

    if (progress) progress(user, 3 /* VAE_DECODE */, 0, 0);
    if (canceled()) return false;
    int dh = 0, dw = 0;
    float *dec = vae_decode(vae, latents.data(), 1, H, W, &dh, &dw);
    if (!dec) {
        set_error(error, "instantmesh: vae_decode failed");
        return false;
    }
    const int OH = dh, OW = dw;  // 960 x 640
    std::vector<uint8_t> out_rgb((size_t)OH * OW * 3);
    // vae_decode returns torch [B,3,H,W] channel-planes; read with matching
    // strides, not interleaved HWC.
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < OH; ++y)
            for (int x = 0; x < OW; ++x) {
                float v = dec[((size_t)c * OH + y) * OW + x] / 0.5f * 0.8f;
                v = v / 2.f + 0.5f;
                v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                out_rgb[((size_t)y * OW + x) * 3 + c] =
                        (uint8_t)(v * 255.f + 0.5f);
            }
    std::free(dec);
    // Diffusion and decode have finished. Keep their exact weights in the
    // context, not beside the upcoming multi-GB geometry queries; repeated
    // forwards restore them above.
    if (host_storage_lease &&
        (!unet.gguf.move_weights(host_storage_lease.handle(), error) ||
         !vae.gguf.move_weights(host_storage_lease.handle(), error)))
        return false;

    // Split the 3x2 grid into 6 views (run.py rearrange 'c (n h) (m w)').
    const int V = 320, rows = 3, cols = 2;
    views.assign((size_t)rows * cols * V * V * 3, 0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            for (int y = 0; y < V; ++y)
                for (int x = 0; x < V; ++x)
                    for (int ch = 0; ch < 3; ++ch)
                        views[((size_t)(r * cols + c) * V * V + (size_t)y * V +
                               x) * 3 +
                              ch] = out_rgb[((size_t)(r * V + y) * OW +
                                             (size_t)(c * V + x)) *
                                                    3 +
                                            ch];
    return true;
}

PipelineContext::~PipelineContext() {
    if (impl_) {
        if (impl_->rmbg) aicore_rmbg_free(impl_->rmbg);
        delete impl_;
    }
}

bool PipelineContext::load(const PipelineConfig &cfg, std::string *error) {
    cfg_ = cfg;
    impl_ = new Impl();

    impl_->lease = init_backend(impl_->backend_name, cfg.device, cfg.threads);
    impl_->backend = impl_->lease.handle();
    backend_name_ = impl_->backend_name;
    if (!impl_->backend) {
        last_error_ =
                "instantmesh: requested backend unavailable: " + cfg.device;
        set_error(error, last_error_);
        return false;
    }
    ggml_backend_t be = impl_->backend;

    std::string err;
    if (!dino_load(cfg.dino_gguf, be, impl_->dino, &err) ||
        !lrm_transformer_load(cfg.transformer_gguf, be, impl_->trans, &err) ||
        !synthesizer_load(cfg.synthesizer_gguf, be, impl_->syn, &err)) {
        set_error(error, "instantmesh: model load failed: " + err);
        last_error_ = *error;
        return false;
    }

    has_multiview_ = false;
    if (!cfg.cond_gguf.empty() && !cfg.unet_gguf.empty() &&
        !cfg.vae_gguf.empty()) {
        if (!clip_vision_load(cfg.cond_gguf, be, impl_->cond, &err) ||
            !unet_load(cfg.unet_gguf, be, impl_->unet, &err) ||
            !vae_load(cfg.vae_gguf, be, impl_->vae, &err)) {
            set_error(error,
                      "instantmesh: zero123++ model load failed: " + err);
            last_error_ = *error;
            return false;
        }
        impl_->has_cond = impl_->has_unet = impl_->has_vae = true;
        has_multiview_ = true;
    }

    if (!cfg.rmbg_gguf.empty()) {
        aicore_rmbg_options *ropts = aicore_rmbg_options_new();
        if (ropts) {
            aicore_rmbg_options_set_device(ropts,
                                           impl_->lease.device().c_str());
            if (cfg.threads > 0)
                aicore_rmbg_options_set_threads(ropts, cfg.threads);
            impl_->rmbg = aicore_rmbg_load_opts(cfg.rmbg_gguf.c_str(), ropts);
            aicore_rmbg_options_free(ropts);
            if (!impl_->rmbg || !aicore_rmbg_is_ready(impl_->rmbg)) {
                const char *e = impl_->rmbg
                                        ? aicore_rmbg_last_error(impl_->rmbg)
                                        : "load failed";
                set_error(error,
                          std::string("instantmesh: RMBG load failed: ") +
                                  (e ? e : "?"));
                last_error_ = *error;
                return false;
            }
        }
    }

    ready_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// Zero123++ multi-view diffusion (upstream graph and independent conditioning
// inputs).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------
bool PipelineContext::generate(const uint8_t *rgba,
                               int w,
                               int h,
                               int row_stride_bytes,
                               const GenParams &params,
                               ProgressFn progress,
                               void *user,
                               PipelineMesh *out,
                               aicore_pipeline_timings *timings,
                               std::string *error) {
    if (!ready_ || !impl_ || !impl_->backend) {
        set_error(error, "instantmesh: pipeline not loaded");
        return false;
    }
    if (!rgba || w <= 0 || h <= 0) {
        set_error(error, "instantmesh: invalid image");
        return false;
    }
    const int stride = row_stride_bytes > 0 ? row_stride_bytes : w * 4;
    const auto canceled = [&]() {
        if (!aicore_cancel_requested()) return false;
        ggml_backend_synchronize(impl_->backend);
        last_error_ = "instantmesh: canceled";
        set_error(error, last_error_);
        return true;
    };
    if (canceled()) return false;

    GenParams p = params;
    if (p.steps <= 0) p.steps = cfg_.steps;
    if (p.guidance <= 0) p.guidance = cfg_.guidance;
    if (p.grid_res <= 0) p.grid_res = cfg_.grid_res;
    if (p.grid_scale <= 0) p.grid_scale = cfg_.grid_scale;
    if (p.texture_res <= 0) p.texture_res = cfg_.texture_res;
    if (p.bake_texture < 0) p.bake_texture = cfg_.bake_texture ? 1 : 0;
    if (p.steps < 2 || p.steps > 1000) {
        set_error(error, "instantmesh: steps out of range");
        return false;
    }

    const double t_e2e = now_ms();
    double t_pre = 0, t_inf = 0, t_post = 0;

    // ---- preprocess: borrowed RGBA -> (RMBG) -> white-composited RGB -----
    if (progress) progress(user, 0 /* PREPROCESS */, 0, 0);
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    std::vector<uint8_t> rmbg_hold;  // keeps the matting buffer alive
    {
        const uint8_t *src = rgba;
        if (impl_->rmbg) {
            if (progress) progress(user, 1 /* RMBG */, 0, 0);
            std::vector<uint8_t> in_rgb((size_t)w * h * 3);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const uint8_t *px =
                            rgba + (size_t)y * stride + (size_t)x * 4;
                    in_rgb[((size_t)y * w + x) * 3 + 0] = px[0];
                    in_rgb[((size_t)y * w + x) * 3 + 1] = px[1];
                    in_rgb[((size_t)y * w + x) * 3 + 2] = px[2];
                }
            uint8_t *m_rgba = nullptr;
            int32_t mw = 0, mh = 0, mlen = 0;
            const int rc = aicore_rmbg_remove_background_rgba(
                    impl_->rmbg, in_rgb.data(), w, h, &m_rgba, &mw, &mh, &mlen);
            if (rc != 0) {
                const char *e = aicore_rmbg_last_error(impl_->rmbg);
                set_error(error, std::string("instantmesh: RMBG failed: ") +
                                         (e ? e : "?"));
                last_error_ = *error;
                return false;
            }
            // Composite the fresh alpha over white now; keep the buffer alive
            // until the composite copy below is done (same scope).
            rmbg_hold.assign(m_rgba, m_rgba + (size_t)mlen);
            aicore_rmbg_free_buffer(m_rgba);
            src = rmbg_hold.data();
        }
        composite_rgba_white(src, h, w, stride, rgb);
    }
    t_pre = now_ms() - t_e2e;

    // ---- multiview: Zero123++ diffusion or replicate ----------------------
    const double t_mv0 = now_ms();
    constexpr int MV = 320;
    std::vector<float> image_mv;  // [6,3,224,224] ImageNet-normalized
    {
        std::vector<uint8_t> views;
        if (has_multiview_) {
            if (!impl_->zero123pp_multiview(cfg_, p.steps, p.seed, p.guidance,
                                            rgb.data(), h, w, views, progress,
                                            user, error)) {
                last_error_ = (error && !error->empty()) ? *error
                                                         : "multiview failed";
                return false;
            }
        } else {
            views.assign((size_t)6 * MV * MV * 3, 0);
            for (int v = 0; v < 6; ++v)
                std::memcpy(&views[(size_t)v * MV * MV * 3], rgb.data(),
                            (size_t)MV * MV * 3 * std::min(1, 0) + 0);
            // replicate: tile the input so each view shares the same RGB
            // (upstream benchmark path). Resize the input to 320 first.
            std::vector<float> fsrc;
            std::vector<uint8_t> src32((size_t)MV * MV * 3);
            {
                std::vector<float> in((size_t)w * h * 3);
                for (size_t k = 0; k < in.size(); ++k) in[k] = rgb[k];
                bicubic_resize(in, h, w, 3, fsrc, MV, MV);
                for (size_t k = 0; k < fsrc.size(); ++k)
                    src32[k] = (uint8_t)std::lround(
                            std::min(255.f, std::max(0.f, fsrc[k])));
            }
            for (int v = 0; v < 6; ++v)
                std::memcpy(&views[(size_t)v * MV * MV * 3], src32.data(),
                            (size_t)MV * MV * 3);
        }

        // views (320 RGB) -> 224 ImageNet-normalized DINO input.
        const int IMG = impl_->dino.hp.image_size;
        const float in_mean[3] = {0.485f, 0.456f, 0.406f};
        const float in_std[3] = {0.229f, 0.224f, 0.225f};
        image_mv.resize((size_t)6 * 3 * IMG * IMG);
        for (int v = 0; v < 6; ++v) {
            std::vector<float> norm;
            // zero123++ views are already RGB u8 in [0,255]; the ImageNet
            // normalize uses per-channel mean/std.
            preprocess_image(&views[(size_t)v * MV * MV * 3], MV, MV, IMG,
                             in_mean, in_std, norm);
            // The helper already produces channel planes; retain each view's
            // NCHW block.
            std::copy(norm.begin(), norm.end(),
                      image_mv.begin() + (size_t)v * 3 * IMG * IMG);
        }
    }
    t_inf = now_ms() - t_mv0;

    // ---- DINO encode -------------------------------------------------------
    const double t_dino0 = now_ms();
    if (progress) progress(user, 4 /* ENCODE */, 0, 0);
    if (canceled()) return false;
    const int IMG = impl_->dino.hp.image_size;
    std::vector<float> camera = default_cameras();
    int seq = 0, hidden = 0;
    float *image_feats = dino_encode(impl_->dino, image_mv.data(),
                                     camera.data(), 6, IMG, IMG, &seq, &hidden);
    std::unique_ptr<float, decltype(&std::free)> image_feats_owner(image_feats,
                                                                   std::free);
    if (!image_feats) {
        set_error(error, "instantmesh: dino encode failed");
        last_error_ = *error;
        return false;
    }
    // The context owns weights for subsequent warm forwards and retries after
    // cancellation.
    t_inf += now_ms() - t_dino0;

    // ---- TriplaneTransformer ----------------------------------------------
    const double t_tr0 = now_ms();
    if (progress) progress(user, 5 /* TRIPLANE */, 0, 0);
    if (canceled()) return false;
    int n_planes = 0, p_dim = 0, pH = 0, pW = 0;
    float *planes = lrm_transformer_forward(impl_->trans, image_feats,
                                            /*N=*/1, /*n_cond=*/6 * seq,
                                            &n_planes, &p_dim, &pH, &pW);
    std::unique_ptr<float, decltype(&std::free)> planes_owner(planes,
                                                              std::free);
    image_feats_owner.reset();
    if (!planes) {
        set_error(error, "instantmesh: triplane forward failed");
        last_error_ = *error;
        return false;
    }
    t_inf += now_ms() - t_tr0;

    // ---- voxel grid + OSGDecoder ------------------------------------------
    const double t_sy0 = now_ms();
    if (progress) progress(user, 6 /* SYNTH */, 0, 0);
    if (canceled()) return false;
    std::vector<float> verts;
    std::vector<int32_t> cubes;
    construct_voxel_grid(p.grid_res, verts, cubes);
    const int M = (int)verts.size() / 3;
    const int n_cubes = (int)cubes.size() / 8;
    for (auto &v : verts) v *= p.grid_scale;

    float *sdf = nullptr, *deformation = nullptr, *weight = nullptr;
    synthesizer_forward(impl_->syn, planes, /*N=*/1, pH, pW, verts.data(), M,
                        cubes.data(), n_cubes, &sdf, &deformation, &weight,
                        [&](int done, int total) {
                            if (progress)
                                progress(user, 6 /* SYNTH */, done, total);
                        });
    std::unique_ptr<float, decltype(&std::free)> sdf_owner(sdf, std::free);
    std::unique_ptr<float, decltype(&std::free)> deformation_owner(deformation,
                                                                   std::free);
    std::unique_ptr<float, decltype(&std::free)> weight_owner(weight,
                                                              std::free);
    if (!sdf || !deformation || !weight) {
        planes_owner.reset();
        set_error(error, "instantmesh: synthesizer forward failed");
        last_error_ = *error;
        return false;
    }
    t_inf += now_ms() - t_sy0;

    // ---- deformation normalization + empty-shape sdf fix (host) -----------
    const double t_mesh0 = now_ms();
    if (progress) progress(user, 7 /* MESH */, 0, 0);
    if (canceled()) return false;
    const float dscale = 1.0f / (p.grid_res * 4.0f);
    for (int i = 0; i < M * 3; ++i)
        deformation[i] = std::tanh(deformation[i]) * dscale;

    {
        long pos = 0, neg = 0;
        const int n1 = p.grid_res + 1;
        for (int k = 1; k <= p.grid_res - 1; ++k)
            for (int j = 1; j <= p.grid_res - 1; ++j)
                for (int i = 1; i <= p.grid_res - 1; ++i) {
                    float s = sdf[((size_t)k * n1 + j) * n1 + i];
                    if (s > 0)
                        ++pos;
                    else if (s < 0)
                        ++neg;
                }
        if (pos == 0 || neg == 0) {
            float mn = sdf[0], mx = sdf[0];
            for (int i = 1; i < M; ++i) {
                if (sdf[i] < mn) mn = sdf[i];
                if (sdf[i] > mx) mx = sdf[i];
            }
            std::vector<int> center, boundary;
            center_boundary_indices(p.grid_res, center, boundary);
            std::vector<float> update(M, 0.0f);
            for (int ci : center) update[ci] += (1.0f - mn);
            for (int bi : boundary) update[bi] += (-1.0f - mx);
            for (int i = 0; i < M; ++i)
                if (update[i] != 0.0f) sdf[i] = update[i];
        }
    }

    std::vector<float> vdef(M * 3);
    for (int i = 0; i < M * 3; ++i) vdef[i] = verts[i] + deformation[i];

    std::vector<float> beta((size_t)n_cubes * 12), alpha((size_t)n_cubes * 8),
            gamma((size_t)n_cubes);
    for (int c = 0; c < n_cubes; ++c) {
        for (int k = 0; k < 12; ++k) beta[c * 12 + k] = weight[c * 21 + k];
        for (int k = 0; k < 8; ++k) alpha[c * 8 + k] = weight[c * 21 + 12 + k];
        gamma[c] = weight[c * 21 + 20];
    }

    FlexiMesh mesh;
    flexicubes_extract(vdef.data(), sdf, cubes.data(), n_cubes, beta.data(),
                       alpha.data(), gamma.data(), p.grid_res, mesh);
    sdf_owner.reset();
    deformation_owner.reset();
    weight_owner.reset();

    const size_t nv = mesh.vertices.size() / 3;
    const size_t nf = mesh.faces.size() / 3;
    if (nv == 0 || nf == 0) {
        planes_owner.reset();
        set_error(error, "instantmesh: flexicubes produced an empty mesh");
        last_error_ = *error;
        return false;
    }
    t_post = now_ms() - t_mesh0;

    // ---- vertex colors + optional texture bake ----------------------------
    const double t_tex0 = now_ms();
    out->verts = std::move(mesh.vertices);
    out->faces = std::move(mesh.faces);
    out->vcolors.assign(nv * 3, 0.7f);
    out->tex_w = out->tex_h = 0;

    if (p.bake_texture) {
        if (progress) progress(user, 8 /* TEXTURE */, 0, 0);
        if (canceled()) return false;
        const int tr = p.texture_res;
        std::vector<float> cuvs;
        int charts = 0;
        if (uv_unwrap(out->verts.data(), out->faces.data(), (int)nv, (int)nf,
                      tr, cuvs, &charts) == 0) {
            std::vector<float> sv, suv, sn;
            std::vector<int32_t> sf;
            split_mesh(out->verts.data(), out->faces.data(), cuvs.data(),
                       (int)nf, sv, suv, sn, sf);
            std::vector<float> gb_pos, gb4;
            std::vector<uint8_t> mask;
            rasterize_texture(sv.data(), sf.data(), suv.data(), (int)sf.size(),
                              (int)nf, tr, gb_pos, gb4, mask);

            std::vector<int> covered;
            for (int px = 0; px < tr * tr; ++px)
                if (mask[px]) covered.push_back(px);

            std::vector<uint8_t> tex((size_t)tr * tr * 3, 0);
            const int BATCH = 16384;
            for (size_t off = 0; off < covered.size(); off += BATCH) {
                int m = (int)std::min((size_t)BATCH, covered.size() - off);
                std::vector<float> pts((size_t)m * 4 * 3);
                for (int i = 0; i < m; ++i) {
                    int px = covered[off + i];
                    for (int s = 0; s < 4; ++s)
                        for (int c = 0; c < 3; ++c)
                            pts[((size_t)i * 4 + s) * 3 + c] =
                                    gb4[((size_t)px * 4 + s) * 3 + c];
                }
                float *rgbq = nullptr;
                if (synthesizer_texture_forward(impl_->syn, planes, 1, pH, pW,
                                                pts.data(), m * 4, &rgbq) &&
                    rgbq) {
                    std::vector<float> avg((size_t)m * 3);
                    for (int i = 0; i < m; ++i)
                        for (int c = 0; c < 3; ++c)
                            avg[(size_t)i * 3 + c] =
                                    0.25f * (rgbq[((size_t)i * 4 + 0) * 3 + c] +
                                             rgbq[((size_t)i * 4 + 1) * 3 + c] +
                                             rgbq[((size_t)i * 4 + 2) * 3 + c] +
                                             rgbq[((size_t)i * 4 + 3) * 3 + c]);
                    std::free(rgbq);
                    for (int i = 0; i < m; ++i) {
                        int px = covered[off + i];
                        for (int c = 0; c < 3; ++c) {
                            float x = avg[(size_t)i * 3 + c] *
                                              (1.0f + 2.0f * 0.001f) -
                                      0.001f;
                            x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
                            tex[3 * (size_t)px + c] =
                                    (uint8_t)(x * 255.0f + 0.5f);
                        }
                    }
                }
                if (progress)
                    progress(user, 8 /* TEXTURE */,
                             (int)std::min(covered.size(), off + (size_t)m),
                             (int)covered.size());
            }

            // Dilate to fill the silhouette rim (cv2.dilate 3x3 mirror).
            {
                std::vector<uint8_t> dil(tex.size(), 0);
                for (int py = 0; py < tr; ++py)
                    for (int px = 0; px < tr; ++px)
                        for (int c = 0; c < 3; ++c) {
                            uint8_t mx = 0;
                            for (int dy = -1; dy <= 1; ++dy)
                                for (int dx = -1; dx <= 1; ++dx) {
                                    int nx = px + dx, ny = py + dy;
                                    if (nx < 0 || ny < 0 || nx >= tr ||
                                        ny >= tr)
                                        continue;
                                    mx = std::max(
                                            mx, tex[3 * ((size_t)ny * tr + nx) +
                                                    c]);
                                }
                            dil[3 * ((size_t)py * tr + px) + c] = mx;
                        }
                for (int px = 0; px < tr * tr; ++px)
                    if (!mask[px])
                        for (int c = 0; c < 3; ++c)
                            tex[3 * (size_t)px + c] = dil[3 * (size_t)px + c];
            }

            // PNG flips vertically (row 0 = top = v=1) so OBJ vt (v=0 bottom)
            // maps correctly; RGBA atlas uses the same top-first convention.
            out->tex_w = out->tex_h = tr;
            out->tex_rgba.resize((size_t)tr * tr * 4);
            for (int y = 0; y < tr; ++y)
                for (int x = 0; x < tr; ++x) {
                    const size_t s = (size_t)(tr - 1 - y) * tr + x;
                    out->tex_rgba[((size_t)y * tr + x) * 4 + 0] =
                            tex[s * 3 + 0];
                    out->tex_rgba[((size_t)y * tr + x) * 4 + 1] =
                            tex[s * 3 + 1];
                    out->tex_rgba[((size_t)y * tr + x) * 4 + 2] =
                            tex[s * 3 + 2];
                    out->tex_rgba[((size_t)y * tr + x) * 4 + 3] = 255;
                }
            out->uvs = std::move(suv);  // per-corner UVs of the split mesh
        }
    }

    // Vertex colors: sample net_rgb at the final vertex positions (mirrors
    // upstream get_texture_prediction; gray fallback keeps a valid result).
    {
        float *rgbv = nullptr;
        if (synthesizer_texture_forward(impl_->syn, planes, 1, pH, pW,
                                        out->verts.data(), (int)nv, &rgbv) &&
            rgbv) {
            for (size_t i = 0; i < nv * 3; ++i) {
                float c = rgbv[i] * (1.0f + 2.0f * 0.001f) - 0.001f;
                out->vcolors[i] = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
            }
            std::free(rgbv);
        }
    }
    planes_owner.reset();
    t_post += now_ms() - t_tex0;

    if (timings) {
        timings->abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
        timings->valid_fields = AICORE_TIMING_PREPROCESS |
                                AICORE_TIMING_INFERENCE |
                                AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
        timings->preprocess_ms = t_pre;
        timings->inference_ms = t_inf;
        timings->postprocess_ms = t_post;
        timings->serialization_ms = 0;
        timings->e2e_ms = now_ms() - t_e2e;
    }
    return true;
}

// ---------------------------------------------------------------------------
// GLB export (minimal glTF 2.0 binary: POSITION, indices, TEXCOORD_0, one
// embedded RGBA texture).
// ---------------------------------------------------------------------------
namespace {

void append_u32(std::vector<uint8_t> &b, uint32_t v) {
    b.push_back((uint8_t)(v & 0xff));
    b.push_back((uint8_t)((v >> 8) & 0xff));
    b.push_back((uint8_t)((v >> 16) & 0xff));
    b.push_back((uint8_t)((v >> 24) & 0xff));
}
void append_f32v(std::vector<uint8_t> &b, const float *v, size_t n) {
    const auto *p = reinterpret_cast<const uint8_t *>(v);
    b.insert(b.end(), p, p + n * sizeof(float));
}
void append_u32v(std::vector<uint8_t> &b, const int32_t *v, size_t n) {
    const auto *p = reinterpret_cast<const uint8_t *>(v);
    b.insert(b.end(), p, p + n * sizeof(int32_t));
}

}  // namespace

uint8_t *bake_glb_buffer(const float *verts,
                         int n_verts,
                         const int *tris,
                         int n_tris,
                         const float *uvs,
                         const uint8_t *tex_rgba,
                         int tex_w,
                         int tex_h,
                         int *out_len,
                         std::string *error) {
    if (out_len) *out_len = 0;
    if (!verts || n_verts <= 0 || !tris || n_tris <= 0 ||
        n_verts > std::numeric_limits<int>::max() / 12 ||
        n_tris > std::numeric_limits<int>::max() / 96) {
        set_error(error, "instantmesh: invalid mesh for GLB");
        return nullptr;
    }
    const size_t nc = static_cast<size_t>(n_tris) * 3;
    for (size_t i = 0; i < nc; ++i) {
        if (tris[i] < 0 || tris[i] >= n_verts) {
            set_error(error, "instantmesh: invalid triangle index for GLB");
            return nullptr;
        }
    }
    for (size_t i = 0; i < static_cast<size_t>(n_verts) * 3; ++i) {
        if (!std::isfinite(verts[i])) {
            set_error(error, "instantmesh: non-finite position for GLB");
            return nullptr;
        }
    }
    const bool has_uv = tex_rgba && uvs;
    if ((tex_rgba || uvs || tex_w || tex_h) &&
        (!has_uv || tex_w <= 0 || tex_h <= 0 || tex_w > 4096 || tex_h > 4096)) {
        set_error(error, "instantmesh: invalid texture for GLB");
        return nullptr;
    }
    std::vector<float> positions, texcoords;
    std::vector<int32_t> indices;
    if (has_uv) {
        // glTF attributes share vertex indices; split corners at the export
        // boundary to preserve UV seams.
        positions.resize(nc * 3);
        texcoords.resize(nc * 2);
        indices.resize(nc);
        for (size_t i = 0; i < nc; ++i) {
            if (!std::isfinite(uvs[i * 2]) || !std::isfinite(uvs[i * 2 + 1])) {
                set_error(error, "instantmesh: non-finite UV for GLB");
                return nullptr;
            }
            std::memcpy(positions.data() + i * 3,
                        verts + static_cast<size_t>(tris[i]) * 3,
                        3 * sizeof(float));
            indices[i] = static_cast<int32_t>(i);
            texcoords[i * 2] = uvs[i * 2];
            // glTF (0,0) is the image's top-left; typed UV v=0 is the bottom.
            texcoords[i * 2 + 1] = 1.0f - uvs[i * 2 + 1];
        }
        verts = positions.data();
        tris = indices.data();
        n_verts = static_cast<int>(nc);
    }

    std::vector<uint8_t> bin;
    // 0: positions (float3)
    const size_t pos_off = bin.size();
    append_f32v(bin, verts, (size_t)n_verts * 3);
    // 1: indices (u32)
    const size_t idx_off = bin.size();
    append_u32v(bin, tris, (size_t)n_tris * 3);
    // 2: texcoords (float2) — required when textured
    size_t uv_off = 0;
    if (has_uv) {
        uv_off = bin.size();
        append_f32v(bin, texcoords.data(), nc * 2);
    }
    // 3: texture PNG
    size_t png_off = 0;
    size_t png_len = 0;
    if (has_uv) {
        // PNG retains top-first RGBA, including alpha; coordinates are already
        // converted at the UV boundary.
        std::vector<uint8_t> png;
        const aicore_image_view texture{tex_rgba, tex_w, tex_h,
                                        size_t(tex_w) * 4, AICORE_IMAGE_RGBA8};
        if (!aicore::img::encode_png(texture, png)) {
            set_error(error, "instantmesh: texture PNG encode failed");
            return nullptr;
        }
        png_off = bin.size();
        bin.insert(bin.end(), png.begin(), png.end());
        png_len = png.size();
    }
    // 4-byte alignment padding to satisfy the GLB chunk alignment.
    while (bin.size() % 4) bin.push_back(0);

    auto jstr = [](const char *s) { return std::string("\"") + s + "\""; };
    std::string json =
            "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,"
            "\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],";
    json += "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0";
    if (has_uv) json += ",\"TEXCOORD_0\":2";
    json += "},\"indices\":1,\"material\":0}]}],";
    json += "\"materials\":[{\"pbrMetallicRoughness\":{";
    if (has_uv) json += "\"baseColorTexture\":{\"index\":0},";
    json += "\"metallicFactor\":0.0,\"roughnessFactor\":1.0}}],";
    if (has_uv) {
        json += "\"textures\":[{\"sampler\":0,\"source\":0}],"
                "\"samplers\":[{\"wrapS\":33071,\"wrapT\":33071}],"
                "\"images\":[{\"bufferView\":3,\"mimeType\":\"image/png\"}],";
    }
    json += "\"buffers\":[{\"byteLength\":" + std::to_string(bin.size()) +
            "}],";
    json += "\"bufferViews\":["
            "{\"buffer\":0,\"byteOffset\":" +
            std::to_string(pos_off) +
            ",\"byteLength\":" + std::to_string((size_t)n_verts * 3 * 4) +
            ",\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":" +
            std::to_string(idx_off) +
            ",\"byteLength\":" + std::to_string((size_t)n_tris * 3 * 4) +
            ",\"target\":34963}";
    if (has_uv) {
        json += ",{\"buffer\":0,\"byteOffset\":" + std::to_string(uv_off) +
                ",\"byteLength\":" +
                std::to_string((size_t)n_tris * 3 * 2 * 4) +
                ",\"target\":34962}";
        json += ",{\"buffer\":0,\"byteOffset\":" + std::to_string(png_off) +
                ",\"byteLength\":" + std::to_string(png_len) + "}";
    }
    json += "],\"accessors\":["
            "{\"bufferView\":0,\"componentType\":5126,\"count\":" +
            std::to_string(n_verts) + ",\"type\":\"VEC3\",\"min\":[0,0,0]," +
            "\"max\":[0,0,0]},"
            "{\"bufferView\":1,\"componentType\":5125,\"count\":" +
            std::to_string(n_tris * 3) + ",\"type\":\"SCALAR\"}";
    if (has_uv) {
        json += ",{\"bufferView\":2,\"componentType\":5126,\"count\":" +
                std::to_string(n_tris * 3) + ",\"type\":\"VEC2\"}";
    }
    json += "]}";

    // Fill min/max on positions.
    {
        float mn[3] = {verts[0], verts[1], verts[2]};
        float mx[3] = {verts[0], verts[1], verts[2]};
        for (int i = 1; i < n_verts; ++i)
            for (int c = 0; c < 3; ++c) {
                float v = verts[(size_t)i * 3 + c];
                mn[c] = std::min(mn[c], v);
                mx[c] = std::max(mx[c], v);
            }
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]",
                      mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
        const std::string want = "\"min\":[0,0,0],\"max\":[0,0,0]";
        auto at = json.find(want);
        if (at != std::string::npos) json.replace(at, want.size(), buf);
    }

    // GLB container.
    std::vector<uint8_t> glb;
    const std::string pad = has_uv ? "" : "";
    (void)pad;
    while (json.size() % 4) json += ' ';
    const size_t total_size = 28 + json.size() + bin.size();
    if (total_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        set_error(error, "instantmesh: GLB exceeds ABI length limit");
        return nullptr;
    }
    const uint32_t total = static_cast<uint32_t>(total_size);
    append_u32(glb, 0x46546C67);  // magic 'glTF'
    append_u32(glb, 2);
    append_u32(glb, total);
    append_u32(glb, (uint32_t)json.size());
    append_u32(glb, 0x4E4F534A);  // 'JSON'
    glb.insert(glb.end(), json.begin(), json.end());
    append_u32(glb, (uint32_t)bin.size());
    append_u32(glb, 0x004E4942);  // 'BIN'
    glb.insert(glb.end(), bin.begin(), bin.end());

    uint8_t *out = (uint8_t *)malloc(glb.size());
    if (!out) {
        set_error(error, "instantmesh: GLB alloc failed");
        return nullptr;
    }
    std::memcpy(out, glb.data(), glb.size());
    if (out_len) *out_len = (int)glb.size();
    return out;
}

}  // namespace instantmesh
