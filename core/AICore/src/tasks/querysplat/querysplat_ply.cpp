// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "querysplat_ply.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace qs {

namespace {

constexpr float kShDcScale = 0.28209479177387814f;

// gs.py save_ply runs torch.log / torch.logit on CUDA, i.e. libdevice
// __nv_logf (an fmaf polynomial, deterministic on every GPU), which differs
// from glibc logf by 1 ulp on ~3% of inputs. Same fma sequence on the host
// -> bit-identical opacity / scale columns (verified on 4M inputs).
float cuda_logf(float a) {
    auto f2i = [](float f) {
        int32_t i;
        std::memcpy(&i, &f, 4);
        return i;
    };
    auto i2f = [](int32_t i) {
        float f;
        std::memcpy(&f, &i, 4);
        return f;
    };
    float i = 0.0f;
    if (a < 0x1p-126f) {
        a *= 0x1p23f;
        i = -23.0f;
    }
    const int32_t e = (int32_t)(((uint32_t)f2i(a) - 0x3f2aaaabu) & 0xff800000u);
    const float m = i2f(f2i(a) - e);
    i = std::fma((float)e, 0x1p-23f, i);
    const float f = m - 1.0f;
    float r = std::fma(-0x1.0aa04ep-3f, f, 0x1.2073ecp-3f);
    r = std::fma(r, f, -0x1.f19b98p-4f);
    r = std::fma(r, f, 0x1.1e52aap-3f);
    r = std::fma(r, f, -0x1.55b172p-3f);
    r = std::fma(r, f, 0x1.99da16p-3f);
    r = std::fma(r, f, -0x1.fffe44p-3f);
    r = std::fma(r, f, 0x1.5554f0p-2f);
    r = std::fma(r, f, -0.5f);
    r = r * f;
    r = std::fma(r, f, f);
    r = std::fma(i, 0x1.62e430p-1f, r);
    if ((uint32_t)f2i(a) >= 0x7f800000u) r = std::fma(a, INFINITY, INFINITY);
    if (a == 0.0f) r = -INFINITY;
    return r;
}

// torch logit (no eps): log(x / (1 - x)), IEEE f32 division.
inline float logit(float x) { return cuda_logf(x / (1.0f - x)); }

}  // namespace

bool save_gaussian_ply(const GaussianOutput& g,
                       const std::string& path,
                       float opacity_threshold) {
    const int color_dim = g.color_dim;
    const int sh_degree = color_dim == 3 ? 0 : 1;  // released configs are 0/1
    const int stride = 11 + color_dim;
    const int n_rest = sh_degree > 0 ? color_dim - 3 : 0;
    const int props = 3 + 3 + n_rest + 1 + 3 + 4;

    std::vector<int> keep;
    keep.reserve((size_t)g.N);
    for (int64_t i = 0; i < g.N; i++) {
        if (g.opacity[(size_t)i] >= opacity_threshold) keep.push_back((int)i);
    }

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    // std::string header: the property list length varies with the SH degree
    // (9 f_rest props at degree 1) and a fixed stack buffer once overflowed.
    // plyfile writes no comment lines.
    std::string header =
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex " +
            std::to_string(keep.size()) +
            "\n"
            "property float x\nproperty float y\nproperty float z\n";
    for (int i = 0; i < 3; i++)
        header += "property float f_dc_" + std::to_string(i) + "\n";
    for (int i = 0; i < n_rest; i++)
        header += "property float f_rest_" + std::to_string(i) + "\n";
    header += "property float opacity\n";
    for (int i = 0; i < 3; i++)
        header += "property float scale_" + std::to_string(i) + "\n";
    for (int i = 0; i < 4; i++)
        header += "property float rot_" + std::to_string(i) + "\n";
    header += "end_header\n";
    fwrite(header.data(), 1, header.size(), fp);

    std::vector<float> row(props);
    // gs.py runs on CUDA, where dividing by a CPU scalar multiplies by its
    // reciprocal, computed in f64 and then rounded to f32
    const float inv_c0 = (float)(1.0 / 0.28209479177387814);
    for (int idx : keep) {
        const float* src = &g.data[(size_t)idx * stride];
        const float* pos = src + 0;
        const float op = src[3];
        const float* scl = src + 4;
        const float* rot = src + 7;
        const float* col = src + 11;
        float* o = row.data();
        for (int i = 0; i < 3; i++) *o++ = pos[i];
        for (int i = 0; i < 3; i++) {
            *o++ = sh_degree == 0 ? (col[i] - 0.5f) * inv_c0 : col[i];
        }
        // sh (N, K, 3) -> transpose(1, 2) -> flatten: channel-major,
        // f_rest[c * (K-1) + (k-1)] = sh[k][c]
        const int k_rest = n_rest / 3;
        for (int c = 0; c < 3; c++)
            for (int k = 0; k < k_rest; k++) *o++ = col[3 + k * 3 + c];
        const float oc = std::min(std::max(op, 1e-8f), 1.0f - 1e-8f);
        *o++ = logit(oc);
        for (int i = 0; i < 3; i++) *o++ = cuda_logf(scl[i] + 1e-8f);
        for (int i = 0; i < 4; i++) *o++ = rot[i];
        fwrite(row.data(), sizeof(float), props, fp);
    }
    fclose(fp);
    return true;
}

bool save_gaussians_bin(const GaussianOutput& g, const std::string& path) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    fwrite(g.data.data(), sizeof(float), g.data.size(), fp);
    fclose(fp);
    return true;
}

// Official save_colored_pointcloud parity: vertex-only colored point cloud
// (no GS attributes), opacity-filtered, RGB decoded via color_to_rgb.
bool save_colored_pointcloud_ply(const GaussianOutput& g,
                                 const std::string& path,
                                 float threshold) {
    const int stride = 11 + g.color_dim;
    const float C0 = 0.28209479177387814f;
    std::vector<std::array<float, 3>> pts;
    std::vector<std::array<unsigned char, 3>> cols;
    for (int64_t i = 0; i < g.N; i++) {
        const float* src = &g.data[(size_t)i * stride];
        if (src[3] < threshold) continue;
        float rgb[3];
        if (g.color_dim <= 3) {
            rgb[0] = src[11];
            rgb[1] = src[12];
            rgb[2] = src[13];
        } else {
            for (int c = 0; c < 3; c++)
                rgb[c] = std::clamp(src[11 + c] * C0 + 0.5f, 0.0f, 1.0f);
        }
        pts.push_back({src[0], src[1], src[2]});
        // official: (rgb.clip(0,1) * 255).astype(np.uint8) — truncation
        auto u8 = [](float v) {
            return (unsigned char)(std::clamp(v, 0.0f, 1.0f) * 255.0f);
        };
        cols.push_back({u8(rgb[0]), u8(rgb[1]), u8(rgb[2])});
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    std::string header =
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex " +
            std::to_string(pts.size()) +
            "\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n";
    fwrite(header.data(), 1, header.size(), f);
    for (size_t i = 0; i < pts.size(); i++) {
        fwrite(pts[i].data(), 4, 3, f);
        fwrite(cols[i].data(), 1, 3, f);
    }
    fclose(f);
    return true;
}

}  // namespace qs
