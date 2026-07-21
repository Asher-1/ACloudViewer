// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// PLY export implementation for SIBR Gaussian viewer compatibility.
// Converts free-splatter's activated Gaussians to 3DGS PLY format.
#include "ply_export.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace aicore {
namespace gaussian {

namespace {

static inline float sane_f(float v, float fallback = 0.0f) {
    return std::isfinite(v) ? v : fallback;
}

static const float C0 = 0.28209479177387814f;

static inline float logit(float p) {
    p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
    return std::log(p / (1.0f - p));
}

struct SibrGaussianVertex {
    float x, y, z;
    float nx, ny, nz;
    float dc[3];
    float rest[45];
    float opacity;
    float scale[3];
    float rot[4];
};

static bool collect_sibr_gaussians(const float* gaussians,
                                   int32_t n_views,
                                   int32_t height,
                                   int32_t width,
                                   int32_t gaussian_channels,
                                   int32_t sh_degree,
                                   float opacity_threshold,
                                   std::vector<SibrGaussianVertex>& valid,
                                   std::string& err) {
    if (!gaussians || n_views < 1 || height < 1 || width < 1 ||
        gaussian_channels < 16) {
        err = "invalid gaussians buffer";
        return false;
    }

    const int64_t P = (int64_t)n_views * height * width;
    const int gc = gaussian_channels;
    const int source_rest_count = (sh_degree + 1) * (sh_degree + 1) - 1;

    valid.clear();
    valid.reserve(static_cast<size_t>(P / 4));

    for (int64_t i = 0; i < P; i++) {
        const float* g = gaussians + i * gc;
        const float op = g[15];
        if (op <= opacity_threshold) continue;

        SibrGaussianVertex v{};
        v.x = sane_f(g[0]);
        v.y = -sane_f(g[1]);
        v.z = -sane_f(g[2]);
        v.dc[0] = sane_f(g[3]);
        v.dc[1] = sane_f(g[4]);
        v.dc[2] = sane_f(g[5]);
        std::fill(v.rest, v.rest + 45, 0.0f);
        for (int c = 0; c < 3 && c < source_rest_count; c++) {
            for (int j = 0; j < std::min(source_rest_count, 15); j++) {
                int src_idx = 6 + c * source_rest_count + j;
                if (src_idx < gc) {
                    v.rest[c * 15 + j] = sane_f(g[src_idx]);
                }
            }
        }
        v.opacity = logit(sane_f(op, 0.5f));
        v.scale[0] = std::log(std::max(sane_f(g[16], 1e-4f), 1e-8f));
        v.scale[1] = std::log(std::max(sane_f(g[17], 1e-4f), 1e-8f));
        v.scale[2] = std::log(std::max(sane_f(g[18], 1e-4f), 1e-8f));
        v.rot[0] = sane_f(g[19], 1.0f);
        v.rot[1] = sane_f(g[20]);
        v.rot[2] = sane_f(g[21]);
        v.rot[3] = sane_f(g[22]);
        valid.push_back(v);
    }

    if (valid.empty()) {
        err = "no valid gaussians after pruning (try lowering opacity "
              "threshold)";
        return false;
    }
    return true;
}

static void write_ply_header(std::ostream& out, size_t count) {
    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "element vertex " << count << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property float nx\n";
    out << "property float ny\n";
    out << "property float nz\n";
    for (int c = 0; c < 3; c++) out << "property float f_dc_" << c << "\n";
    for (int i = 0; i < 45; i++) out << "property float f_rest_" << i << "\n";
    out << "property float opacity\n";
    out << "property float scale_0\n";
    out << "property float scale_1\n";
    out << "property float scale_2\n";
    out << "property float rot_0\n";
    out << "property float rot_1\n";
    out << "property float rot_2\n";
    out << "property float rot_3\n";
    out << "end_header\n";
}

static void write_ply_vertices(std::ostream& out,
                               const std::vector<SibrGaussianVertex>& valid) {
    for (const auto& v : valid) {
        out.write(reinterpret_cast<const char*>(&v.x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&v.y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&v.z), sizeof(float));
        out.write(reinterpret_cast<const char*>(&v.nx), sizeof(float));
        out.write(reinterpret_cast<const char*>(&v.ny), sizeof(float));
        out.write(reinterpret_cast<const char*>(&v.nz), sizeof(float));
        out.write(reinterpret_cast<const char*>(v.dc), 3 * sizeof(float));
        out.write(reinterpret_cast<const char*>(v.rest), 45 * sizeof(float));
        out.write(reinterpret_cast<const char*>(&v.opacity), sizeof(float));
        out.write(reinterpret_cast<const char*>(v.scale), 3 * sizeof(float));
        out.write(reinterpret_cast<const char*>(v.rot), 4 * sizeof(float));
    }
}

}  // namespace

bool export_ply_sibr(const float* gaussians,
                     int32_t n_views,
                     int32_t height,
                     int32_t width,
                     int32_t gaussian_channels,
                     int32_t sh_degree,
                     float opacity_threshold,
                     const std::string& out_path,
                     std::string& err) {
    std::vector<SibrGaussianVertex> valid;
    if (!collect_sibr_gaussians(gaussians, n_views, height, width,
                                gaussian_channels, sh_degree, opacity_threshold,
                                valid, err)) {
        return false;
    }

    std::ofstream f(out_path, std::ios::binary);
    if (!f) {
        err = "cannot write to: " + out_path;
        return false;
    }
    write_ply_header(f, valid.size());
    write_ply_vertices(f, valid);
    return true;
}

bool export_ply_sibr_to_buffer(const float* gaussians,
                               int32_t n_views,
                               int32_t height,
                               int32_t width,
                               int32_t gaussian_channels,
                               int32_t sh_degree,
                               float opacity_threshold,
                               std::vector<uint8_t>& out,
                               std::string& err) {
    std::vector<SibrGaussianVertex> valid;
    if (!collect_sibr_gaussians(gaussians, n_views, height, width,
                                gaussian_channels, sh_degree, opacity_threshold,
                                valid, err)) {
        return false;
    }

    std::ostringstream oss(std::ios::binary);
    write_ply_header(oss, valid.size());
    write_ply_vertices(oss, valid);
    const std::string bytes = oss.str();
    out.assign(bytes.begin(), bytes.end());
    return true;
}

}  // namespace gaussian
}  // namespace aicore
