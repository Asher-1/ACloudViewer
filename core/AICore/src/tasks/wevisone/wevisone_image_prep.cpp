// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/wevisone/wevisone_image_prep.h"

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <thread>

#if defined(ACV_WEV_HAVE_LIBJPEG)
#include <jerror.h>
#include <jpeglib.h>
#endif

#include "common/image_utils.hpp"

namespace wwv {

// The released processor_config.json normalizes with 0.5/0.5 per channel
// (qwen3vl-era config; NOT the Qwen2-VL CLIP stats — the round-10 mask
// investigation proved a CLIP-stats preprocessing diverges the manual ViT
// features while the mtmd path, which reads the GGUF metadata written from
// this same config, stays correct).
const float kQwenImageMean[3] = {0.5f, 0.5f, 0.5f};
const float kQwenImageStd[3] = {0.5f, 0.5f, 0.5f};

namespace {

// Cubic convolution kernel with a = -0.5 (PIL / torch bicubic weight).
float cubic_weight(float t) {
    const float at = std::fabs(t);
    const float t2 = at * at;
    const float t3 = t2 * at;
    if (at <= 1.0f) {
        return 1.5f * t3 - 2.5f * t2 + 1.0f;
    }
    if (at < 2.0f) {
        return -0.5f * t3 + 2.5f * t2 - 4.0f * at + 2.0f;
    }
    return 0.0f;
}

// torch upsample_bicubic2d(antialias=True) tap layout for one output
// coordinate: center = (dst + 0.5) * scale - 0.5 with scale = in / out;
// the kernel support doubles when downscaling (ks = max(1, scale)).
struct TapList {
    int begin = 0;
    std::vector<float> weights;  // weight per input index begin + i
};

TapList bicubic_taps(int32_t in_size, int32_t out_size, int32_t dst) {
    TapList taps;
    const float scale =
            static_cast<float>(in_size) / static_cast<float>(out_size);
    const float center = (static_cast<float>(dst) + 0.5f) * scale - 0.5f;
    const float ks = std::max(1.0f, scale);
    const float inv_ks = 1.0f / ks;
    const float support = 2.0f * ks;
    const int lo = static_cast<int>(std::floor(center - support + 0.5f));
    const int hi = static_cast<int>(std::ceil(center + support - 0.5f));
    taps.begin = std::max(0, lo);
    const int last = std::min(in_size - 1, hi);
    std::vector<float> raw;
    raw.reserve(static_cast<size_t>(last - taps.begin + 1));
    for (int i = taps.begin; i <= last; ++i) {
        raw.push_back(cubic_weight((static_cast<float>(i) - center) * inv_ks));
    }
    // torch normalizes the tap weights (window sum == 1).
    float sum = 0.0f;
    for (const float w : raw) sum += w;
    taps.weights.reserve(raw.size());
    for (const float w : raw) taps.weights.push_back(w / sum);
    return taps;
}

// One separable pass over interleaved RGB rows: "rows" in the source have
// `in_size` samples (3 channels each) and produce `out_size` samples.
void bicubic_pass(const std::vector<float>& in,
                  int32_t n_rows,
                  int32_t in_size,
                  int32_t out_size,
                  std::vector<float>* out) {
    out->assign(static_cast<size_t>(n_rows) * out_size * 3, 0.0f);
    // The tap weights depend on the destination column only — precompute
    // them once, then run the row loops on worker threads. Rows are
    // independent and the per-element FMA order is untouched, so the
    // -ffp-contract=off / torch-exact contract below still holds bit-exact.
    std::vector<TapList> taps;
    taps.reserve(static_cast<size_t>(out_size));
    for (int32_t d = 0; d < out_size; ++d) {
        taps.push_back(bicubic_taps(in_size, out_size, d));
    }
    auto row_range = [&](int32_t r0, int32_t r1) {
        for (int32_t r = r0; r < r1; ++r) {
            for (int32_t d = 0; d < out_size; ++d) {
                const TapList& tap = taps[static_cast<size_t>(d)];
                float* dst = out->data() +
                             (static_cast<size_t>(r) * out_size + d) * 3;
                for (size_t k = 0; k < tap.weights.size(); ++k) {
                    const float* src =
                            in.data() +
                            (static_cast<size_t>(r) * in_size +
                             static_cast<size_t>(tap.begin +
                                                 static_cast<int>(k))) *
                                    3;
                    const float w = tap.weights[k];
                    // The single FMA torch issues per tap (the accum).
                    for (int c = 0; c < 3; ++c) {
                        dst[c] = std::fma(w, src[c], dst[c]);
                    }
                }
            }
        }
    };
    const unsigned hw = std::thread::hardware_concurrency();
    const int n_threads =
            static_cast<int>(std::min<unsigned>(std::max(1u, hw), 16u));
    if (n_threads <= 1 || n_rows < 16) {
        row_range(0, n_rows);
        return;
    }
    std::vector<std::thread> workers;
    const int32_t chunk = (n_rows + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        const int32_t r0 = t * chunk;
        if (r0 >= n_rows) break;
        workers.emplace_back(row_range, r0, std::min(n_rows, r0 + chunk));
    }
    for (auto& w : workers) w.join();
}

// Interleaved RGB8 -> CHW f32 in [0, 1] resized to an arbitrary target.
std::vector<float> bicubic_rect_chw(const RgbImage& src,
                                    int32_t out_w,
                                    int32_t out_h) {
    std::vector<float> interleaved(static_cast<size_t>(src.height) * src.width *
                                   3);
    for (size_t i = 0; i < interleaved.size(); ++i) {
        interleaved[i] = static_cast<float>(src.pixels[i]);
    }
    // Horizontal: H x W -> H x out_w (rows = image rows).
    std::vector<float> horiz;
    bicubic_pass(interleaved, src.height, src.width, out_w, &horiz);
    // Vertical: transpose to W x H so rows become columns, then pass again.
    std::vector<float> vert_in(static_cast<size_t>(out_w) * src.height * 3);
    for (int32_t y = 0; y < src.height; ++y) {
        for (int32_t x = 0; x < out_w; ++x) {
            std::memcpy(vert_in.data() +
                                (static_cast<size_t>(x) * src.height + y) * 3,
                        horiz.data() + (static_cast<size_t>(y) * out_w + x) * 3,
                        3 * sizeof(float));
        }
    }
    std::vector<float> vert;
    bicubic_pass(vert_in, out_w, src.height, out_h, &vert);
    // vert: (x: out_w, y: out_h, c) interleaved -> CHW planes.
    std::vector<float> out(static_cast<size_t>(out_w) * out_h * 3);
    const size_t plane = static_cast<size_t>(out_w) * out_h;
    for (int32_t y = 0; y < out_h; ++y) {
        for (int32_t x = 0; x < out_w; ++x) {
            const float* p =
                    vert.data() + (static_cast<size_t>(x) * out_h + y) * 3;
            const size_t i = static_cast<size_t>(y) * out_w + x;
            out[i] = p[0] / 255.0f;
            out[plane + i] = p[1] / 255.0f;
            out[2 * plane + i] = p[2] / 255.0f;
        }
    }
    return out;
}

#if defined(ACV_WEV_HAVE_LIBJPEG)
// Minimal libjpeg error shim: jump back to the decode call site on error.
struct JpegErrorMgr {
    jpeg_error_mgr pub;
    jmp_buf jump;
};

void jpeg_error_exit(j_common_ptr cinfo) {
    auto* err = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
    longjmp(err->jump, 1);
}
#endif

bool load_path_jpeg(const std::string& path, RgbImage* out, std::string* err) {
#if defined(ACV_WEV_HAVE_LIBJPEG)
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        if (err != nullptr) *err = "cannot open " + path;
        return false;
    }
    jpeg_decompress_struct cinfo{};
    JpegErrorMgr jerr{};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    if (setjmp(jerr.jump) != 0) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(f);
        if (err != nullptr) *err = "libjpeg decode failed: " + path;
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(f);
        if (err != nullptr) *err = "bad jpeg header: " + path;
        return false;
    }
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    out->width = static_cast<int32_t>(cinfo.output_width);
    out->height = static_cast<int32_t>(cinfo.output_height);
    out->pixels.resize(static_cast<size_t>(out->width) * out->height * 3);
    std::vector<JSAMPLE> row(static_cast<size_t>(cinfo.output_width) * 3);
    for (int y = 0; y < cinfo.output_height; ++y) {
        JSAMPROW rows[1] = {row.data()};
        jpeg_read_scanlines(&cinfo, rows, 1);
        std::memcpy(
                out->pixels.data() + static_cast<size_t>(y) * out->width * 3,
                row.data(), static_cast<size_t>(out->width) * 3);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(f);
    return true;
#else
    (void)path;
    (void)out;
    (void)err;
    return false;
#endif
}

}  // namespace

bool load_path_rgb(const std::string& path, RgbImage* out, std::string* err) {
    // libjpeg first (byte-identical to PIL ISLOW + fancy upsampling); the
    // shared AICore Qt decoder covers PNG and exotic JPEGs (no stb).
    if (load_path_jpeg(path, out, err)) return true;
    aicore::img::Image img;
    if (!aicore::img::load_image_rgb(path, img)) {
        if (err != nullptr) *err = "cannot decode " + path;
        return false;
    }
    out->width = img.width;
    out->height = img.height;
    out->pixels = std::move(img.rgb);
    return true;
}

bool view_to_rgb(const aicore_image_view& img,
                 RgbImage* out,
                 std::string* err) {
    if (img.data == nullptr || img.width <= 0 || img.height <= 0) {
        if (err != nullptr) *err = "invalid image view";
        return false;
    }
    const int32_t w = img.width;
    const int32_t h = img.height;
    size_t min_stride = 0;
    switch (img.format) {
        case AICORE_IMAGE_RGB8:
        case AICORE_IMAGE_BGR8:
            min_stride = static_cast<size_t>(w) * 3;
            break;
        case AICORE_IMAGE_RGBA8:
        case AICORE_IMAGE_BGRA8:
            min_stride = static_cast<size_t>(w) * 4;
            break;
        case AICORE_IMAGE_GRAY8:
            min_stride = static_cast<size_t>(w);
            break;
        default:
            if (err != nullptr) *err = "unsupported image format";
            return false;
    }
    if (img.row_stride_bytes < min_stride) {
        if (err != nullptr) *err = "row_stride_bytes too small";
        return false;
    }
    out->width = w;
    out->height = h;
    out->pixels.assign(static_cast<size_t>(w) * h * 3, 0);
    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src =
                img.data + static_cast<size_t>(y) * img.row_stride_bytes;
        uint8_t* dst = out->pixels.data() + static_cast<size_t>(y) * w * 3;
        for (int32_t x = 0; x < w; ++x) {
            switch (img.format) {
                case AICORE_IMAGE_RGB8:
                    dst[x * 3 + 0] = src[x * 3 + 0];
                    dst[x * 3 + 1] = src[x * 3 + 1];
                    dst[x * 3 + 2] = src[x * 3 + 2];
                    break;
                case AICORE_IMAGE_RGBA8:
                    dst[x * 3 + 0] = src[x * 4 + 0];
                    dst[x * 3 + 1] = src[x * 4 + 1];
                    dst[x * 3 + 2] = src[x * 4 + 2];
                    break;
                case AICORE_IMAGE_BGRA8:
                    dst[x * 3 + 0] = src[x * 4 + 2];
                    dst[x * 3 + 1] = src[x * 4 + 1];
                    dst[x * 3 + 2] = src[x * 4 + 0];
                    break;
                case AICORE_IMAGE_BGR8:
                    dst[x * 3 + 0] = src[x * 3 + 2];
                    dst[x * 3 + 1] = src[x * 3 + 1];
                    dst[x * 3 + 2] = src[x * 3 + 0];
                    break;
                case AICORE_IMAGE_GRAY8:
                    dst[x * 3 + 0] = src[x];
                    dst[x * 3 + 1] = src[x];
                    dst[x * 3 + 2] = src[x];
                    break;
            }
        }
    }
    return true;
}

std::vector<float> bicubic_square_chw(const RgbImage& src, int32_t size) {
    return bicubic_rect_chw(src, size, size);
}

void smart_resize_dims(int32_t width,
                       int32_t height,
                       int32_t factor,
                       int64_t min_pixels,
                       int64_t max_pixels,
                       int32_t* out_width,
                       int32_t* out_height) {
    // qwen_vl_utils.vision_process.smart_resize (released checkpoint math):
    // round to the nearest factor multiple, clamp the pixel count, re-round.
    auto round_by = [&](int64_t v) {
        return std::max<int64_t>(
                factor,
                static_cast<int64_t>(std::round(static_cast<double>(v) /
                                                static_cast<double>(factor))) *
                        factor);
    };
    int64_t w = round_by(width);
    int64_t h = round_by(height);
    if (w * h > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(width) * height /
                                      static_cast<double>(max_pixels));
        w = round_by(static_cast<int64_t>(
                std::floor(static_cast<double>(width) / beta)));
        h = round_by(static_cast<int64_t>(
                std::floor(static_cast<double>(height) / beta)));
    } else if (w * h < min_pixels) {
        const double beta = std::sqrt(
                static_cast<double>(min_pixels) /
                static_cast<double>(static_cast<int64_t>(width) * height));
        w = round_by(static_cast<int64_t>(
                std::ceil(static_cast<double>(width) * beta)));
        h = round_by(static_cast<int64_t>(
                std::ceil(static_cast<double>(height) * beta)));
    }
    *out_width = static_cast<int32_t>(w);
    *out_height = static_cast<int32_t>(h);
}

VitInput smart_resize_normalized(const RgbImage& src,
                                 int32_t patch_size,
                                 int32_t merge_size,
                                 int64_t min_pixels,
                                 int64_t max_pixels) {
    const int32_t factor = patch_size * merge_size;
    VitInput out;
    smart_resize_dims(src.width, src.height, factor, min_pixels, max_pixels,
                      &out.width, &out.height);
    std::vector<float> chw = bicubic_rect_chw(src, out.width, out.height);
    // Normalize per channel with the released Qwen2-VL statistics.
    const size_t plane = static_cast<size_t>(out.width) * out.height;
    for (int c = 0; c < 3; ++c) {
        for (size_t i = 0; i < plane; ++i) {
            chw[c * plane + i] =
                    (chw[c * plane + i] - kQwenImageMean[c]) / kQwenImageStd[c];
        }
    }
    out.pixels = std::move(chw);
    return out;
}

}  // namespace wwv
