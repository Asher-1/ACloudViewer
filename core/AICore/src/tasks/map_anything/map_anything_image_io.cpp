// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Official MapAnything preprocessing over borrowed views. Zero third-party
// image dependencies: the resize is the in-tree separable Catmull-Rom
// resampler below (same Keys a=-0.5 kernel family as the upstream stb
// smoke path; the bit-exact parity anchor stays the --bin f32 feed).

#include "map_anything_image_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "aicore/map_anything_capi.h"
#include "common/capi_utils.hpp"
#include "map_anything_common.hpp"

namespace map_anything {

namespace {

// Python round(): round half to even (the official loader relies on
// builtins.round / np.round for every target-shape decision)
int round_half_even(double v) {
    const double f = std::floor(v);
    const double diff = v - f;
    if (diff > 0.5) return (int)f + 1;
    if (diff < 0.5) return (int)f;
    return ((int)f % 2 == 0) ? (int)f : (int)f + 1;
}

struct ImageBuf {
    int w = 0, h = 0;
    std::vector<unsigned char> px;  // RGB uint8, row-major
};

// official _crop_to_supported_aspect_ratio: center-crop extreme aspect
// ratios (height/width) into [0.5, 2.0]
void crop_to_supported_aspect(ImageBuf& im) {
    const double aspect = (double)im.h / (double)std::max(im.w, 1);
    if (aspect < 0.5) {
        const int crop_w =
                std::min(im.w, std::max(1, round_half_even(im.h / 0.5)));
        const int left = std::max((im.w - crop_w) / 2, 0);
        ImageBuf out;
        out.w = crop_w;
        out.h = im.h;
        out.px.resize((size_t)crop_w * im.h * 3);
        for (int y = 0; y < im.h; y++) {
            memcpy(&out.px[(size_t)y * crop_w * 3],
                   &im.px[((size_t)y * im.w + left) * 3], (size_t)crop_w * 3);
        }
        im = std::move(out);
    } else if (aspect > 2.0) {
        const int crop_h =
                std::min(im.h, std::max(1, round_half_even(im.w * 2.0)));
        const int top = std::max((im.h - crop_h) / 2, 0);
        ImageBuf out;
        out.w = im.w;
        out.h = crop_h;
        out.px.resize((size_t)im.w * crop_h * 3);
        memcpy(out.px.data(), &im.px[(size_t)top * im.w * 3],
               (size_t)im.w * crop_h * 3);
        im = std::move(out);
    }
}

// Catmull-Rom == Keys cubic with a=-0.5 (the PIL "BICUBIC" kernel family;
// B-spline form B=0, C=0.5).
inline float catmullrom_kernel(float x) {
    x = std::fabs(x);
    if (x < 1.0f) return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    if (x < 2.0f) return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    return 0.0f;
}

// One 1-D resample of a float row/column set with half-pixel-centered
// sample mapping, weight normalization and EDGE_CLAMP (the stbir default
// contract). filter_scale widens the support when downscaling so every
// input pixel contributes with total weight 1.
//
// Complexity: O(out_n * taps) per pass with taps <= 4 * filter_scale + 1;
// the two separable passes replace stbir's fused ring-buffer traversal at
// one extra (width*height*3) float staging buffer.
void resample_1d(const float* src,
                 int src_n,
                 int channels,
                 int src_stride,  // floats between consecutive lines
                 int dst_n,
                 int dst_stride,  // floats between consecutive lines
                 int line_count,
                 float* dst) {
    const bool downscaling = dst_n < src_n;
    const float scale = (float)dst_n / (float)src_n;
    const float filter_scale = downscaling ? 1.0f / scale : 1.0f;
    const float support = 2.0f * filter_scale;

    std::vector<float> weights;
    std::vector<int> indices;
    weights.reserve((size_t)(2 * support + 2));
    indices.reserve((size_t)(2 * support + 2));

    for (int o = 0; o < dst_n; ++o) {
        const float center = ((float)o + 0.5f) / scale - 0.5f;
        const int first = (int)std::ceil(center - support);
        const int last = (int)std::floor(center + support);

        weights.clear();
        indices.clear();
        float total = 0.0f;
        for (int i = first; i <= last; ++i) {
            const float w = catmullrom_kernel((i - center) / filter_scale);
            if (w == 0.0f) continue;
            const int clamped = std::min(src_n - 1, std::max(0, i));
            weights.push_back(w);
            indices.push_back(clamped);
            total += w;
        }
        if (total == 0.0f || indices.empty()) {
            // Degenerate support (never reached with the Catmull-Rom
            // kernel, but keeps the pass total-weight safe).
            const int clamped =
                    std::min(src_n - 1, std::max(0, (int)std::lround(center)));
            for (int l = 0; l < line_count; ++l) {
                for (int c = 0; c < channels; ++c) {
                    dst[(size_t)l * dst_stride + (size_t)o * channels + c] =
                            src[(size_t)l * src_stride +
                                (size_t)clamped * channels + c];
                }
            }
            continue;
        }
        const float inv_total = 1.0f / total;
        for (int l = 0; l < line_count; ++l) {
            float* dst_line =
                    dst + (size_t)l * dst_stride + (size_t)o * channels;
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (size_t k = 0; k < indices.size(); ++k) {
                const float w = weights[k] * inv_total;
                const float* src_px = src + (size_t)l * src_stride +
                                      (size_t)indices[k] * channels;
                for (int c = 0; c < channels; ++c) acc[c] += w * src_px[c];
            }
            for (int c = 0; c < channels; ++c) dst_line[c] = acc[c];
        }
    }
}

inline unsigned char to_uint8(float v) {
    const long rounded = std::lroundf(v);
    return (unsigned char)std::min(255L, std::max(0L, rounded));
}

// stbir_resize replacement: uint8 RGB -> uint8 RGB, separable
// Catmull-Rom, EDGE_CLAMP. Horizontal pass stages through a float buffer,
// the vertical pass writes the quantized output.
void resize_rgb_catmullrom(const unsigned char* src,
                           int sw,
                           int sh,
                           unsigned char* dst,
                           int dw,
                           int dh) {
    const int channels = 3;
    // Horizontal: (sh, sw, 3) -> (sh, dw, 3) floats.
    std::vector<float> tmp((size_t)sh * dw * channels);
    std::vector<float> src_row;
    src_row.reserve((size_t)sw * channels);
    for (int y = 0; y < sh; ++y) {
        // resample_1d walks "lines" of samples; build one row view by
        // transposing in-place would cost extra bandwidth — instead feed
        // the single-line variant: source line = row y, stride = sw*3.
        src_row.assign((size_t)sw * channels, 0.0f);
        for (int x = 0; x < sw; ++x) {
            for (int c = 0; c < channels; ++c) {
                src_row[(size_t)x * channels + c] =
                        src[((size_t)y * sw + x) * channels + c];
            }
        }
        resample_1d(src_row.data(), sw, channels, sw * channels, dw,
                    dw * channels, 1, tmp.data() + (size_t)y * dw * channels);
    }
    // Vertical: (sh, dw, 3) -> (dh, dw, 3) floats, quantized to uint8.
    // The 1-D pass works line-by-line over columns: transpose the tmp
    // buffer column-major into (dw, sh, 3) so a line == one column.
    std::vector<float> cols((size_t)dw * sh * channels);
    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < dw; ++x) {
            for (int c = 0; c < channels; ++c) {
                cols[((size_t)x * sh + y) * channels + c] =
                        tmp[((size_t)y * dw + x) * channels + c];
            }
        }
    }
    // Output staging must hold dh rows (a tall target resamples dw->dh
    // with dh > dw; sizing it by dw overflows the heap for portrait
    // inputs).
    std::vector<float> out_col((size_t)dh * channels);
    for (int x = 0; x < dw; ++x) {
        resample_1d(cols.data() + (size_t)x * sh * channels, sh, channels,
                    channels, dh, channels, 1, out_col.data());
        for (int y = 0; y < dh; ++y) {
            for (int c = 0; c < channels; ++c) {
                dst[((size_t)y * dw + x) * channels + c] =
                        to_uint8(out_col[(size_t)y * channels + c]);
            }
        }
    }
}

}  // namespace

// Official load_and_preprocess_images(mode="balanced"|"max_size") core:
// input buffers are already RGB8 row-major.
static bool preprocess_official(std::vector<ImageBuf>& ims,
                                int image_resolution,
                                int patch_size,
                                const std::string& mode,
                                std::vector<float>& out,
                                int& height,
                                int& width,
                                std::vector<std::pair<int, int>>* valid_sizes) {
    if (image_resolution <= 0 || image_resolution % patch_size != 0) {
        log_error(
                "image_resolution (%d) must be positive and divisible by "
                "patch_size (%d)",
                image_resolution, patch_size);
        return false;
    }
    if (mode != "balanced" && mode != "max_size" && mode != "square") {
        log_error("unknown resize mode: %s (balanced|max_size|square)",
                  mode.c_str());
        return false;
    }
    out.clear();
    height = width = 0;
    if (valid_sizes != nullptr) valid_sizes->clear();

    const int token_number =
            (image_resolution / patch_size) * (image_resolution / patch_size);

    struct Plane {
        int w = 0, h = 0;
        std::vector<float> px;  // (h, w, c) rows in [0,1]
    };
    std::vector<Plane> planes;

    for (ImageBuf& im : ims) {
        // load_and_preprocess_images_square pads any aspect to the square
        // and never crops; the aspect crop belongs to the other loaders.
        if (mode != "square") crop_to_supported_aspect(im);

        const double ar = (double)im.h / (double)std::max(im.w, 1);
        // official _balanced_target_shape: w_patches = round(sqrt(T/ar)),
        // then h_patches = round(T / w_patches_FLOAT) — the float w (NOT the
        // rounded one) feeds the h division, i.e. h = round(sqrt(T*ar)).
        // Using the rounded w here shifts h by one patch for some aspect
        // ratios (e.g. ar=1.24: official 576 vs 560) and desyncs the two
        // runtimes' preprocessing for mixed-size image sets.
        const double w_patches_f = std::sqrt((double)token_number / ar);
        int target_w = 0, target_h = 0;
        if (mode == "balanced") {
            const int w_patches = std::max(1, round_half_even(w_patches_f));
            const int h_patches = std::max(
                    1, round_half_even((double)token_number / w_patches_f));
            target_w = w_patches * patch_size;
            target_h = h_patches * patch_size;
        } else {
            // official _max_size_target_shape + _round_to_patch_multiple:
            // longest side = image_resolution, other side rounded to a
            // patch multiple (max(patch, round(v/patch)*patch))
            auto round_to_patch = [&](double v) {
                return std::max(patch_size,
                                round_half_even(v / patch_size) * patch_size);
            };
            if (ar >= 1.0) {
                target_h = image_resolution;
                target_w = round_to_patch(image_resolution / ar);
            } else {
                target_w = image_resolution;
                target_h = round_to_patch(image_resolution * ar);
            }
        }

        if (mode == "square") {
            // Official VGGT load_and_preprocess_images_square: paste the
            // source centered on a BLACK square canvas (Image.new("RGB",
            // (max_dim, max_dim), (0, 0, 0))), then resize the whole square
            // to (R, R). The valid content rect keeps the original aspect
            // centered in the square. (demo_colmap loads at 1024 and
            // bilinearly re-samples to 518; this resizes once, straight to
            // R.)
            const int side = std::max(im.w, im.h);
            std::vector<unsigned char> sq((size_t)side * side * 3, 0);
            const int ox = (side - im.w) / 2, oy = (side - im.h) / 2;
            for (int y = 0; y < im.h; ++y) {
                std::memcpy(sq.data() + ((size_t)(oy + y) * side + ox) * 3,
                            im.px.data() + (size_t)y * im.w * 3,
                            (size_t)im.w * 3);
            }
            target_w = image_resolution;
            target_h = image_resolution;
            std::vector<unsigned char> out8((size_t)target_w * target_h * 3);
            resize_rgb_catmullrom(sq.data(), side, side, out8.data(), target_w,
                                  target_h);
            Plane sqp;
            sqp.w = target_w;
            sqp.h = target_h;
            sqp.px.resize((size_t)target_w * target_h * 3);
            for (size_t i = 0; i < sqp.px.size(); i++)
                sqp.px[i] = out8[i] / 255.0f;
            planes.push_back(std::move(sqp));
            if (valid_sizes != nullptr) {
                const double ar = (double)im.h / (double)std::max(im.w, 1);
                int cw = image_resolution, ch = image_resolution;
                if (ar >= 1.0) {
                    ch = image_resolution;
                    cw = std::max(1, (int)std::lround(image_resolution / ar));
                } else {
                    cw = image_resolution;
                    ch = std::max(1, (int)std::lround(image_resolution * ar));
                }
                valid_sizes->emplace_back(cw, ch);
            }
            continue;
        }
        Plane pl;
        pl.w = target_w;
        pl.h = target_h;
        // The resampler writes quantized uint8 (the stb contract this path
        // keeps); stage through a byte buffer — writing floats straight in
        // would reinterpret every 4 bytes as one float.
        std::vector<unsigned char> out8((size_t)target_w * target_h * 3);
        resize_rgb_catmullrom(im.px.data(), im.w, im.h, out8.data(), target_w,
                              target_h);
        pl.px.resize((size_t)target_w * target_h * 3);
        for (size_t i = 0; i < pl.px.size(); i++) pl.px[i] = out8[i] / 255.0f;
        planes.push_back(std::move(pl));
    }
    // square already recorded each view's letterboxed content rect; the
    // plane itself is the full (R, R) canvas.
    if (valid_sizes != nullptr && mode != "square") {
        valid_sizes->clear();
        valid_sizes->reserve(planes.size());
        for (const auto& pl : planes) valid_sizes->emplace_back(pl.w, pl.h);
    }

    // official _pad_images_to_common_size: white padding (value 1.0),
    // centered (pad_top = (maxh-h)//2 etc.). Output layout must match the
    // graph input ggml ne{W,H,3,S} == (S,3,H,W) C-order: channel before
    // height/width per frame (the previous (h,w,c) layout produced finite
    // but WRONG pixels — the model never saw the real image).
    int max_h = 0, max_w = 0;
    for (const auto& pl : planes) {
        max_h = std::max(max_h, pl.h);
        max_w = std::max(max_w, pl.w);
    }
    for (const auto& pl : planes) {
        const int pad_top = (max_h - pl.h) / 2;
        const int pad_bottom = (max_h - pl.h) - pad_top;
        const int pad_left = (max_w - pl.w) / 2;
        const int pad_right = (max_w - pl.w) - pad_left;
        const auto push_white = [&](int pixels) {
            for (int i = 0; i < pixels; i++) out.push_back(1.0f);
        };
        for (int c = 0; c < 3; c++) {
            push_white(pad_top * max_w);
            for (int y = 0; y < pl.h; y++) {
                push_white(pad_left);
                const float* row = &pl.px[(size_t)y * pl.w * 3];
                for (int x = 0; x < pl.w; x++) {
                    out.push_back(row[(size_t)x * 3 + c]);
                }
                push_white(pad_right);
            }
            push_white(pad_bottom * max_w);
        }
    }
    height = max_h;
    width = max_w;
    // (h, w, c) per image matches (H, W, C) = ggml {W, H, 3, S} rows.
    return true;
}

// Convert one borrowed view (RGB8/RGBA8/GRAY8/BGR8/BGRA8, arbitrary stride)
// into a packed RGB8 ImageBuf through the shared AICore view converter (the
// one owner of format/stride semantics; it rejects a short stride).
static bool view_to_rgb(const aicore_image_view& v, ImageBuf& im) {
    if (!aicore::capi::image_view_to_packed_rgb(v, im.px)) {
        log_error("invalid image view (format %d, %dx%d, stride %zu)",
                  (int)v.format, v.width, v.height, v.row_stride_bytes);
        return false;
    }
    im.w = v.width;
    im.h = v.height;
    return true;
}

bool load_images_official_views(const aicore_image_view* views,
                                int n,
                                int image_resolution,
                                int patch_size,
                                std::vector<float>& out,
                                int& height,
                                int& width,
                                const std::string& mode,
                                std::vector<std::pair<int, int>>* valid_sizes) {
    if (!views || n <= 0) {
        log_error("no input image views");
        return false;
    }
    std::vector<ImageBuf> ims;
    ims.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        ImageBuf im;
        if (!view_to_rgb(views[i], im)) return false;
        ims.push_back(std::move(im));
    }
    return preprocess_official(ims, image_resolution, patch_size, mode, out,
                               height, width, valid_sizes);
}

}  // namespace map_anything
