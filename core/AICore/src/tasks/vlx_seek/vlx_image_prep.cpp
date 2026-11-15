// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "tasks/vlx_seek/vlx_image_prep.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "FileSystem.h"

#if defined(ACV_VLX_HAVE_LIBJPEG)
#include <jpeglib.h>

#include <csetjmp>
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_NO_SIMD
#include "tasks/vlx_seek/third_party/stb/stb_image.h"
#include "tasks/vlx_seek/third_party/stb/stb_image_write.h"

namespace vlx {

// --------------------------------------------------------------------------
// decode
// --------------------------------------------------------------------------
#if defined(ACV_VLX_HAVE_LIBJPEG)
namespace {
struct JpegErr {
    jpeg_error_mgr pub;
    std::jmp_buf jb;
    char msg[JMSG_LENGTH_MAX];
};
void jpeg_err_exit(j_common_ptr c) {
    JpegErr* e = reinterpret_cast<JpegErr*>(c->err);
    (*c->err->format_message)(c, e->msg);
    std::longjmp(e->jb, 1);
}

// libjpeg with PIL's decoder settings: library defaults (JDCT_ISLOW, fancy
// upsampling on), out_color_space RGB for YCbCr/RGB, gray decoded as L and
// replicated (PIL's convert("RGB") of an "L" image copies the channel).
bool decode_jpeg(const std::vector<uint8_t>& buf,
                 ImageU8* out,
                 std::string* err) {
    jpeg_decompress_struct cinfo;
    JpegErr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_err_exit;
    std::vector<uint8_t> row;
    if (setjmp(jerr.jb)) {
        if (err) *err = std::string("libjpeg: ") + jerr.msg;
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, buf.data(), static_cast<unsigned long>(buf.size()));
    jpeg_read_header(&cinfo, TRUE);
    const bool gray = cinfo.jpeg_color_space == JCS_GRAYSCALE;
    if (cinfo.jpeg_color_space == JCS_CMYK ||
        cinfo.jpeg_color_space == JCS_YCCK) {
        if (err) *err = "CMYK/YCCK JPEG is not supported";
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    cinfo.out_color_space = gray ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_start_decompress(&cinfo);
    const int W = int(cinfo.output_width), H = int(cinfo.output_height);
    const int nc = int(cinfo.output_components);
    out->width = W;
    out->height = H;
    out->rgb.resize(size_t(W) * H * 3);
    row.resize(size_t(W) * nc);
    while (cinfo.output_scanline < cinfo.output_height) {
        const int y = int(cinfo.output_scanline);
        JSAMPROW rp = row.data();
        jpeg_read_scanlines(&cinfo, &rp, 1);
        uint8_t* dst = out->rgb.data() + size_t(y) * W * 3;
        if (nc == 3) {
            std::memcpy(dst, row.data(), size_t(W) * 3);
        } else {
            for (int x = 0; x < W; ++x)
                dst[3 * x] = dst[3 * x + 1] = dst[3 * x + 2] = row[x];
        }
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;
}
}  // namespace
#endif

bool decode_image(const std::string& path, ImageU8* out, std::string* err) {
    std::vector<uint8_t> buf;
    {
        std::FILE* f =
                cloudViewer::utility::filesystem::FOpen(path.c_str(), "rb");
        if (!f) {
            if (err) *err = "cannot open " + path;
            return false;
        }
        size_t size = 0;
        if (!cloudViewer::utility::filesystem::FSize(f, size)) {
            std::fclose(f);
            if (err) *err = "cannot determine size of " + path;
            return false;
        }
        buf.resize(size);
        const size_t got = std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        if (got != buf.size()) {
            if (err) *err = "short read " + path;
            return false;
        }
    }
#if defined(ACV_VLX_HAVE_LIBJPEG)
    if (buf.size() > 2 && buf[0] == 0xFF && buf[1] == 0xD8)
        return decode_jpeg(buf, out, err);
#endif
    int w = 0, h = 0, ch = 0;
    uint8_t* px =
            stbi_load_from_memory(buf.data(), int(buf.size()), &w, &h, &ch, 3);
    if (!px) {
        if (err) *err = std::string("stb_image: ") + stbi_failure_reason();
        return false;
    }
    out->width = w;
    out->height = h;
    out->rgb.assign(px, px + size_t(w) * h * 3);
    stbi_image_free(px);
    return true;
}

// --------------------------------------------------------------------------
// separable AA resample, 8bpc fixed point
// --------------------------------------------------------------------------
namespace {

double filter_bilinear(double x) {
    x = std::fabs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}
double filter_bicubic(double x) {  // a = -0.5 (PIL and torch AA both)
    const double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

struct Coeffs {
    int ksize = 0;               // max taps per output
    std::vector<int> xmin, xsz;  // per output
    std::vector<int32_t> k;      // out * ksize quantized taps
    int prec = 22;
};

// PIL precompute_coeffs + normalize_coeffs_8bpc, or torch's
// _compute_index_ranges_int16_weights (same double weights; int16 with the
// largest precision whose max tap fits, capped at 22 bits).
Coeffs make_coeffs(int in, int out, ResampleFilter f, ResampleImpl impl) {
    const double support0 = f == ResampleFilter::Bilinear ? 1.0 : 2.0;
    const double scale = double(in) / out;
    const double fs = std::max(scale, 1.0);
    const double support = support0 * fs;
    const double ss = 1.0 / fs;
    Coeffs c;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.xmin.resize(out);
    c.xsz.resize(out);
    std::vector<double> w(size_t(out) * c.ksize, 0.0);
    double wmax = 0.0;
    for (int xx = 0; xx < out; ++xx) {
        const double center = (xx + 0.5) * scale;
        int xmin = int(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = int(center + support + 0.5);
        if (xmax > in) xmax = in;
        xmax -= xmin;
        double* k = &w[size_t(xx) * c.ksize];
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double t = (x + xmin - center + 0.5) * ss;
            const double v = f == ResampleFilter::Bilinear ? filter_bilinear(t)
                                                           : filter_bicubic(t);
            k[x] = v;
            ww += v;
        }
        for (int x = 0; x < xmax; ++x) {
            if (ww != 0.0) k[x] /= ww;
            wmax = std::max(wmax, k[x]);
        }
        c.xmin[xx] = xmin;
        c.xsz[xx] = xmax;
    }
    if (impl == ResampleImpl::kTorch) {
        int p = 0;
        for (; p < 22; ++p)
            if (int(0.5 + wmax * double(1 << (p + 1))) >= (1 << 15)) break;
        c.prec = p;
    } else {
        c.prec = 22;
    }
    c.k.resize(w.size());
    const double m = double(1 << c.prec);
    for (size_t i = 0; i < w.size(); ++i)
        c.k[i] = w[i] < 0 ? int32_t(-0.5 + w[i] * m) : int32_t(0.5 + w[i] * m);
    return c;
}

inline uint8_t clip8(int64_t v, int prec) {
    v >>= prec;
    return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v);
}

ImageU8 pass_h(const ImageU8& s, int ow, const Coeffs& c) {
    ImageU8 d;
    d.width = ow;
    d.height = s.height;
    d.rgb.resize(size_t(ow) * s.height * 3);
    const int64_t init = int64_t(1) << (c.prec - 1);
    for (int y = 0; y < s.height; ++y) {
        const uint8_t* sr = s.rgb.data() + size_t(y) * s.width * 3;
        uint8_t* dr = d.rgb.data() + size_t(y) * ow * 3;
        for (int x = 0; x < ow; ++x) {
            const int32_t* k = &c.k[size_t(x) * c.ksize];
            const int x0 = c.xmin[x], n = c.xsz[x];
            int64_t a0 = init, a1 = init, a2 = init;
            for (int j = 0; j < n; ++j) {
                const uint8_t* p = sr + size_t(x0 + j) * 3;
                a0 += int64_t(p[0]) * k[j];
                a1 += int64_t(p[1]) * k[j];
                a2 += int64_t(p[2]) * k[j];
            }
            dr[3 * x] = clip8(a0, c.prec);
            dr[3 * x + 1] = clip8(a1, c.prec);
            dr[3 * x + 2] = clip8(a2, c.prec);
        }
    }
    return d;
}

ImageU8 pass_v(const ImageU8& s, int oh, const Coeffs& c) {
    ImageU8 d;
    d.width = s.width;
    d.height = oh;
    d.rgb.resize(size_t(s.width) * oh * 3);
    const int64_t init = int64_t(1) << (c.prec - 1);
    const size_t row = size_t(s.width) * 3;
    std::vector<int64_t> acc(row);
    for (int y = 0; y < oh; ++y) {
        const int32_t* k = &c.k[size_t(y) * c.ksize];
        std::fill(acc.begin(), acc.end(), init);
        for (int j = 0; j < c.xsz[y]; ++j) {
            const uint8_t* sr = s.rgb.data() + size_t(c.xmin[y] + j) * row;
            const int64_t kj = k[j];
            for (size_t i = 0; i < row; ++i) acc[i] += int64_t(sr[i]) * kj;
        }
        uint8_t* dr = d.rgb.data() + size_t(y) * row;
        for (size_t i = 0; i < row; ++i) dr[i] = clip8(acc[i], c.prec);
    }
    return d;
}

}  // namespace

ImageU8 resize_u8(const ImageU8& src,
                  int out_w,
                  int out_h,
                  ResampleFilter f,
                  ResampleImpl impl) {
    ImageU8 cur = src;
    if (out_w != src.width)
        cur = pass_h(cur, out_w, make_coeffs(src.width, out_w, f, impl));
    if (out_h != src.height)
        cur = pass_v(cur, out_h, make_coeffs(src.height, out_h, f, impl));
    return cur;
}

// --------------------------------------------------------------------------
// torch float AA bilinear (aux tower _resize_to_supported)
// --------------------------------------------------------------------------
namespace {
struct CoeffsF {
    int ksize = 0;
    std::vector<int> xmin, xsz;
    std::vector<float> w;
};
// _compute_indices_min_size_weights_aa<float>: scalar_t = float, but the
// source mixes in double literals, and C++ promotion makes those steps double
// arithmetic rounded back to float on assignment. Mirrored term by term: an
// all-float version moves weights by an ulp and flips bf16 rounding ties.
CoeffsF make_coeffs_f(int in, int out) {
    const float scale = float(in) / float(out);
    const float support = scale >= 1.0f ? float(1.0 * double(scale))
                                        : 1.0f;  // interp_size*0.5
    const float invscale = scale >= 1.0f ? float(1.0 / double(scale)) : 1.0f;
    CoeffsF c;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.xmin.resize(out);
    c.xsz.resize(out);
    c.w.assign(size_t(out) * c.ksize, 0.f);
    for (int i = 0; i < out; ++i) {
        const float center = float(double(scale) * (double(i) + 0.5));
        const int64_t xmin =
                std::max<int64_t>(int64_t(double(center - support) + 0.5), 0);
        const int64_t xsize =
                std::min<int64_t>(int64_t(double(center + support) + 0.5), in) -
                xmin;
        float* w = &c.w[size_t(i) * c.ksize];
        float total = 0.f;
        for (int64_t j = 0; j < xsize; ++j) {
            // (j + xmin - center) is int64 - float = float; + 0.5 and *
            // invscale in double
            const float t =
                    std::fabs(float((double(float(j + xmin) - center) + 0.5) *
                                    double(invscale)));
            const float v = t < 1.f ? float(1.0 - double(t)) : 0.f;
            w[j] = v;
            total += v;
        }
        if (total != 0.f)
            for (int64_t j = 0; j < xsize; ++j) w[j] /= total;
        c.xmin[i] = int(xmin);
        c.xsz[i] = int(xsize);
    }
    return c;
}
}  // namespace

std::vector<float> resize_f32_bilinear_aa(
        const std::vector<float>& chw, int C, int H, int W, int oh, int ow) {
    // Accumulation = interpolate_aa_single_dim: out = t0*w0, then out += tj*wj.
    // torch's native/cpu kernels are built with -mavx2 -mfma and GCC contracts
    // that `+=` into an FMA; plain mul+add differs in ~12% of outputs by one
    // f32 ulp, which flips bf16 ties downstream. std::fma reproduces it
    // bit-exactly.
    std::vector<float> cur = chw;
    int cw = W;
    if (ow != W) {
        const CoeffsF c = make_coeffs_f(W, ow);
        std::vector<float> d(size_t(C) * H * ow);
        for (int ch = 0; ch < C; ++ch)
            for (int y = 0; y < H; ++y) {
                const float* s = &cur[(size_t(ch) * H + y) * W];
                float* o = &d[(size_t(ch) * H + y) * ow];
                for (int x = 0; x < ow; ++x) {
                    const float* w = &c.w[size_t(x) * c.ksize];
                    float t = s[c.xmin[x]] * w[0];
                    for (int j = 1; j < c.xsz[x]; ++j)
                        t = std::fma(s[c.xmin[x] + j], w[j], t);
                    o[x] = t;
                }
            }
        cur.swap(d);
        cw = ow;
    }
    if (oh != H) {
        const CoeffsF c = make_coeffs_f(H, oh);
        std::vector<float> d(size_t(C) * oh * cw);
        for (int ch = 0; ch < C; ++ch)
            for (int y = 0; y < oh; ++y) {
                const float* w = &c.w[size_t(y) * c.ksize];
                float* o = &d[(size_t(ch) * oh + y) * cw];
                for (int x = 0; x < cw; ++x) {
                    const float* s =
                            &cur[(size_t(ch) * H + c.xmin[y]) * cw + x];
                    float t = s[0] * w[0];
                    for (int j = 1; j < c.xsz[y]; ++j)
                        t = std::fma(s[size_t(j) * cw], w[j], t);
                    o[x] = t;
                }
            }
        cur.swap(d);
    }
    return cur;
}

float bf16_round(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) {  // inf / nan: keep (quiet nan)
        if (u & 0x007fffffu) u |= 0x00400000u;
        u &= 0xffff0000u;
    } else {
        u += 0x7fffu + ((u >> 16) & 1u);
        u &= 0xffff0000u;
    }
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

Letterbox letterbox(const ImageU8& img, int new_size) {
    Letterbox lb;
    const double r = std::min(double(new_size) / img.width,
                              double(new_size) / img.height);
    // python round() = half-to-even = nearbyint under the default rounding mode
    const int nw = int(std::nearbyint(img.width * r));
    const int nh = int(std::nearbyint(img.height * r));
    const ImageU8 rs = resize_u8(img, nw, nh, ResampleFilter::Bilinear,
                                 ResampleImpl::kPil);
    const int dw = new_size - nw, dh = new_size - nh;
    const int left = dw / 2, top = dh / 2;
    lb.canvas.width = lb.canvas.height = new_size;
    lb.canvas.rgb.assign(size_t(new_size) * new_size * 3, 114);
    for (int y = 0; y < nh; ++y)
        std::memcpy(&lb.canvas.rgb[(size_t(y + top) * new_size + left) * 3],
                    &rs.rgb[size_t(y) * nw * 3], size_t(nw) * 3);
    lb.ratio = r;
    lb.off_x = float(dw / 2.0);
    lb.off_y = float(dh / 2.0);
    return lb;
}

void smart_resize(int height,
                  int width,
                  int factor,
                  double min_pixels,
                  double max_pixels,
                  int* h_bar,
                  int* w_bar) {
    int hb = int(std::nearbyint(double(height) / factor)) * factor;
    int wb = int(std::nearbyint(double(width) / factor)) * factor;
    if (double(hb) * wb > max_pixels) {
        const double beta = std::sqrt(double(height) * width / max_pixels);
        hb = std::max(factor, int(std::floor(height / beta / factor)) * factor);
        wb = std::max(factor, int(std::floor(width / beta / factor)) * factor);
    } else if (double(hb) * wb < min_pixels) {
        const double beta = std::sqrt(min_pixels / (double(height) * width));
        hb = int(std::ceil(height * beta / factor)) * factor;
        wb = int(std::ceil(width * beta / factor)) * factor;
    }
    *h_bar = hb;
    *w_bar = wb;
}

}  // namespace vlx
