// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "glb_export.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "reconstruct.hpp"

namespace aicore {
namespace depth {

namespace {

// Apply a row-major 4x4 (double) to a 3D point, returning the [:3] of the
// homogeneous result (last row of A is assumed [0,0,0,1] in our usage, but the
// w-divide is omitted to match trimesh.transform_points for affine matrices).
inline void apply44(const double A[16],
                    double x,
                    double y,
                    double z,
                    double& ox,
                    double& oy,
                    double& oz) {
    ox = A[0] * x + A[1] * y + A[2] * z + A[3];
    oy = A[4] * x + A[5] * y + A[6] * z + A[7];
    oz = A[8] * x + A[9] * y + A[10] * z + A[11];
}

// HSV->RGB matching the reference _hsv_to_rgb (h,s,v in [0,1]).
void hsv_to_rgb(double h, double s, double v, double& r, double& g, double& b) {
    int i = (int)(h * 6.0);
    double f = h * 6.0 - i;
    double p = v * (1.0 - s);
    double q = v * (1.0 - f * s);
    double t = v * (1.0 - (1.0 - f) * s);
    i = ((i % 6) + 6) % 6;
    switch (i) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        default:
            r = v;
            g = p;
            b = q;
            break;
    }
}

// _index_color_rgb(i,n): HSV at hue (i+0.5)/max(n,1), s=0.85, v=0.95 -> u8 rgb.
std::array<uint8_t, 3> index_color_rgb(int i, int n) {
    double h = (i + 0.5) / (double)std::max(n, 1);
    double r, g, b;
    hsv_to_rgb(h, 0.85, 0.95, r, g, b);
    std::array<uint8_t, 3> out;
    out[0] = (uint8_t)(int)(r * 255.0);
    out[1] = (uint8_t)(int)(g * 255.0);
    out[2] = (uint8_t)(int)(b * 255.0);
    return out;
}

// Port of _camera_frustum_lines: returns 8 segments (16 world-frame points,
// pairs of [a,b]) for one camera, BEFORE the alignment transform A.
void camera_frustum_lines(const std::array<float, 9>& K,
                          const std::array<float, 16>& ext,
                          int W,
                          int H,
                          double scale,
                          std::vector<double>& out /*16*3*/) {
    out.clear();
    std::array<float, 9> Kinv;
    std::array<float, 16> c2w;
    if (!inv3(K, Kinv)) return;
    if (!inv4(ext, c2w)) return;

    double ki[9];
    for (int k = 0; k < 9; ++k) ki[k] = (double)Kinv[k];
    double cw[16];
    for (int k = 0; k < 16; ++k) cw[k] = (double)c2w[k];

    // camera center in world: c2w @ [0,0,0,1]
    double Cw[3] = {cw[3], cw[7], cw[11]};

    double corners[4][3] = {
            {0.0, 0.0, 1.0},
            {(double)(W - 1), 0.0, 1.0},
            {(double)(W - 1), (double)(H - 1), 1.0},
            {0.0, (double)(H - 1), 1.0},
    };
    double plane_w[4][3];
    for (int c = 0; c < 4; ++c) {
        double rx = ki[0] * corners[c][0] + ki[1] * corners[c][1] +
                    ki[2] * corners[c][2];
        double ry = ki[3] * corners[c][0] + ki[4] * corners[c][1] +
                    ki[5] * corners[c][2];
        double rz = ki[6] * corners[c][0] + ki[7] * corners[c][1] +
                    ki[8] * corners[c][2];
        double z = (rz == 0.0) ? 1.0 : rz;
        double px = (rx / z) * scale, py = (ry / z) * scale,
               pz = (rz / z) * scale;
        // to world: c2w @ [px,py,pz,1]
        plane_w[c][0] = cw[0] * px + cw[1] * py + cw[2] * pz + cw[3];
        plane_w[c][1] = cw[4] * px + cw[5] * py + cw[6] * pz + cw[7];
        plane_w[c][2] = cw[8] * px + cw[9] * py + cw[10] * pz + cw[11];
    }

    auto push = [&](const double* a, const double* b) {
        out.push_back(a[0]);
        out.push_back(a[1]);
        out.push_back(a[2]);
        out.push_back(b[0]);
        out.push_back(b[1]);
        out.push_back(b[2]);
    };
    // center to corners
    for (int k = 0; k < 4; ++k) push(Cw, plane_w[k]);
    // rectangle edges: 0-1,1-2,2-3,3-0
    int order[5] = {0, 1, 2, 3, 0};
    for (int e = 0; e < 4; ++e) push(plane_w[order[e]], plane_w[order[e + 1]]);
}

// ---- little-endian appenders (host assumed little-endian; x86/arm64) -------
void put_f32(std::vector<uint8_t>& b, float f) {
    uint8_t tmp[4];
    std::memcpy(tmp, &f, 4);
    b.insert(b.end(), tmp, tmp + 4);
}
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back((uint8_t)(v & 0xFF));
    b.push_back((uint8_t)((v >> 8) & 0xFF));
    b.push_back((uint8_t)((v >> 16) & 0xFF));
    b.push_back((uint8_t)((v >> 24) & 0xFF));
}

std::string fmt_f(float v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", (double)v);
    return std::string(buf);
}

}  // namespace

bool write_glb(const std::string& path,
               const std::vector<float>& depth,
               const std::vector<float>& conf,
               const std::vector<std::array<float, 9>>& K,
               const std::vector<std::array<float, 16>>& ext,
               const std::vector<const uint8_t*>& images_u8,
               int H,
               int W,
               int N,
               const GlbOptions& opt) {
    // 1) GLB adaptive confidence threshold over ALL conf values (no sky mask).
    float conf_thr;
    if (conf.empty()) {
        conf_thr = opt.conf_thresh;
    } else {
        double lower = percentile_linear(conf, opt.conf_thresh_percentile);
        double upper = percentile_linear(conf, opt.ensure_thresh_percentile);
        double thr = std::min(std::max((double)opt.conf_thresh, lower), upper);
        conf_thr = (float)thr;
    }

    // 2) Back-project to world points + colors.
    WorldPoints wp =
            back_project(depth, conf, K, ext, images_u8, H, W, N, conf_thr);
    const size_t npts = wp.xyz.size() / 3;

    // 3) glTF alignment transform A (in double).
    //    w2c0 = ext[0]; M = diag(1,-1,-1,1); A_no_center = M @ w2c0.
    double A_no_center[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    if (N > 0) {
        const auto& w = ext[0];
        for (int c = 0; c < 4; ++c) {
            A_no_center[0 * 4 + c] = (double)w[0 * 4 + c];   // row0
            A_no_center[1 * 4 + c] = -(double)w[1 * 4 + c];  // row1 flipped
            A_no_center[2 * 4 + c] = -(double)w[2 * 4 + c];  // row2 flipped
            A_no_center[3 * 4 + c] = (double)w[3 * 4 + c];   // row3
        }
    }
    // pts_tmp = A_no_center @ [X;1]; center = per-axis median.
    double center[3] = {0.0, 0.0, 0.0};
    std::vector<float> tmpx, tmpy, tmpz;
    tmpx.reserve(npts);
    tmpy.reserve(npts);
    tmpz.reserve(npts);
    for (size_t p = 0; p < npts; ++p) {
        double ox, oy, oz;
        apply44(A_no_center, wp.xyz[3 * p], wp.xyz[3 * p + 1],
                wp.xyz[3 * p + 2], ox, oy, oz);
        // store as float to mirror numpy float32 point cloud for the median.
        tmpx.push_back((float)ox);
        tmpy.push_back((float)oy);
        tmpz.push_back((float)oz);
    }
    if (npts > 0) {
        center[0] = percentile_linear(tmpx, 50.0);
        center[1] = percentile_linear(tmpy, 50.0);
        center[2] = percentile_linear(tmpz, 50.0);
    }
    // A = T(-center) @ A_no_center: since last row of A_no_center is [0,0,0,1],
    // this only subtracts `center` from the translation column of rows 0..2.
    double A[16];
    std::memcpy(A, A_no_center, sizeof(A));
    A[3] -= center[0];
    A[7] -= center[1];
    A[11] -= center[2];

    // Final aligned points = A @ [X;1] = pts_tmp - center.
    std::vector<float> pos;
    pos.reserve(npts * 3);
    for (size_t p = 0; p < npts; ++p) {
        pos.push_back((float)((double)tmpx[p] - center[0]));
        pos.push_back((float)((double)tmpy[p] - center[1]));
        pos.push_back((float)((double)tmpz[p] - center[2]));
    }
    // (num_max_points downsample intentionally disabled for determinism.)

    // 4) Optional camera frustums (LINES), aligned by A.
    std::vector<float> line_pos;    // 3 per vertex
    std::vector<uint8_t> line_col;  // 4 per vertex (rgba)
    if (opt.show_cameras && N > 0) {
        // scene scale: p5/p95 per-axis diagonal of the ALIGNED points.
        double scene_scale = 1.0;
        if (npts >= 2) {
            std::vector<float> ax(npts), ay(npts), az(npts);
            for (size_t p = 0; p < npts; ++p) {
                ax[p] = pos[3 * p];
                ay[p] = pos[3 * p + 1];
                az[p] = pos[3 * p + 2];
            }
            double lo[3] = {percentile_linear(ax, 5.0),
                            percentile_linear(ay, 5.0),
                            percentile_linear(az, 5.0)};
            double hi[3] = {percentile_linear(ax, 95.0),
                            percentile_linear(ay, 95.0),
                            percentile_linear(az, 95.0)};
            double dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
            double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (std::isfinite(diag) && diag > 0.0) scene_scale = diag;
        }
        double scale = scene_scale * (double)opt.camera_size;

        std::vector<double> segs;
        for (int i = 0; i < N; ++i) {
            camera_frustum_lines(K[i], ext[i], W, H, scale, segs);
            std::array<uint8_t, 3> col = index_color_rgb(i, N);
            for (size_t k = 0; k < segs.size() / 3; ++k) {
                double ox, oy, oz;
                apply44(A, segs[3 * k], segs[3 * k + 1], segs[3 * k + 2], ox,
                        oy, oz);
                line_pos.push_back((float)ox);
                line_pos.push_back((float)oy);
                line_pos.push_back((float)oz);
                line_col.push_back(col[0]);
                line_col.push_back(col[1]);
                line_col.push_back(col[2]);
                line_col.push_back(255);
            }
        }
    }
    const size_t nlines = line_pos.size() / 3;
    const bool have_lines = nlines > 0;
    const bool have_points = npts > 0;

    // 5) Build BIN buffer with 4-byte-aligned bufferViews. Sections (points,
    // lines) are emitted only when non-empty, and bufferView/accessor indices
    // are assigned dynamically — a zero-length bufferView/accessor/buffer is
    // invalid per glTF 2.0, so an empty point cloud must omit them entirely.
    std::vector<uint8_t> bin;
    auto align4 = [&]() {
        while (bin.size() % 4 != 0) bin.push_back(0);
    };

    struct BV {
        uint32_t off, len;
    };
    std::vector<BV> bvs;
    int bv_ppos = -1, bv_pcol = -1, bv_lpos = -1, bv_lcol = -1;

    if (have_points) {
        uint32_t off = (uint32_t)bin.size();
        for (size_t i = 0; i < pos.size(); ++i) put_f32(bin, pos[i]);
        bvs.push_back({off, (uint32_t)bin.size() - off});
        bv_ppos = (int)bvs.size() - 1;
        align4();
        off = (uint32_t)bin.size();
        for (size_t p = 0; p < npts; ++p) {
            bin.push_back(wp.rgb[3 * p]);
            bin.push_back(wp.rgb[3 * p + 1]);
            bin.push_back(wp.rgb[3 * p + 2]);
            bin.push_back(255);
        }
        bvs.push_back({off, (uint32_t)bin.size() - off});
        bv_pcol = (int)bvs.size() - 1;
        align4();
    }
    if (have_lines) {
        uint32_t off = (uint32_t)bin.size();
        for (size_t i = 0; i < line_pos.size(); ++i) put_f32(bin, line_pos[i]);
        bvs.push_back({off, (uint32_t)bin.size() - off});
        bv_lpos = (int)bvs.size() - 1;
        align4();
        off = (uint32_t)bin.size();
        bin.insert(bin.end(), line_col.begin(), line_col.end());
        bvs.push_back({off, (uint32_t)bin.size() - off});
        bv_lcol = (int)bvs.size() - 1;
        align4();
    }

    // POSITION min/max for accessors.
    float pmin[3] = {0, 0, 0}, pmax[3] = {0, 0, 0};
    if (have_points) {
        for (int c = 0; c < 3; ++c) {
            pmin[c] = pos[c];
            pmax[c] = pos[c];
        }
        for (size_t p = 0; p < npts; ++p)
            for (int c = 0; c < 3; ++c) {
                pmin[c] = std::min(pmin[c], pos[3 * p + c]);
                pmax[c] = std::max(pmax[c], pos[3 * p + c]);
            }
    }
    float lmin[3] = {0, 0, 0}, lmax[3] = {0, 0, 0};
    if (have_lines) {
        for (int c = 0; c < 3; ++c) {
            lmin[c] = line_pos[c];
            lmax[c] = line_pos[c];
        }
        for (size_t p = 0; p < nlines; ++p)
            for (int c = 0; c < 3; ++c) {
                lmin[c] = std::min(lmin[c], line_pos[3 * p + c]);
                lmax[c] = std::max(lmax[c], line_pos[3 * p + c]);
            }
    }

    // Build accessors + mesh primitives with dynamic indices.
    std::ostringstream acc, prim;
    int nacc = 0, nprim = 0;
    if (have_points) {
        int a_pos = nacc++, a_col = nacc++;
        acc << "{\"bufferView\":" << bv_ppos
            << ",\"componentType\":5126,\"count\":" << npts
            << ",\"type\":\"VEC3\",\"min\":[" << fmt_f(pmin[0]) << ","
            << fmt_f(pmin[1]) << "," << fmt_f(pmin[2]) << "],\"max\":["
            << fmt_f(pmax[0]) << "," << fmt_f(pmax[1]) << "," << fmt_f(pmax[2])
            << "]}";
        acc << ",{\"bufferView\":" << bv_pcol
            << ",\"componentType\":5121,\"normalized\":true,\"count\":" << npts
            << ",\"type\":\"VEC4\"}";
        prim << "{\"attributes\":{\"POSITION\":" << a_pos
             << ",\"COLOR_0\":" << a_col << "},\"mode\":0}";
        ++nprim;
    }
    if (have_lines) {
        int a_pos = nacc++, a_col = nacc++;
        if (nacc > 2) acc << ",";
        acc << "{\"bufferView\":" << bv_lpos
            << ",\"componentType\":5126,\"count\":" << nlines
            << ",\"type\":\"VEC3\",\"min\":[" << fmt_f(lmin[0]) << ","
            << fmt_f(lmin[1]) << "," << fmt_f(lmin[2]) << "],\"max\":["
            << fmt_f(lmax[0]) << "," << fmt_f(lmax[1]) << "," << fmt_f(lmax[2])
            << "]}";
        acc << ",{\"bufferView\":" << bv_lcol
            << ",\"componentType\":5121,\"normalized\":true,\"count\":"
            << nlines << ",\"type\":\"VEC4\"}";
        if (nprim) prim << ",";
        prim << "{\"attributes\":{\"POSITION\":" << a_pos
             << ",\"COLOR_0\":" << a_col << "},\"mode\":1}";
        ++nprim;
    }

    // 6) Build JSON.
    std::ostringstream js;
    js << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"depth-anything."
          "cpp\"}";
    if (nprim > 0) {
        js << ",\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":"
              "0}],";
        js << "\"bufferViews\":[";
        for (size_t i = 0; i < bvs.size(); ++i) {
            if (i) js << ",";
            js << "{\"buffer\":0,\"byteOffset\":" << bvs[i].off
               << ",\"byteLength\":" << bvs[i].len << ",\"target\":34962}";
        }
        js << "],\"accessors\":[" << acc.str() << "],";
        js << "\"meshes\":[{\"primitives\":[" << prim.str() << "]}],";
        js << "\"buffers\":[{\"byteLength\":" << bin.size() << "}]}";
    } else {
        // Nothing to draw (no valid points, no cameras): emit a minimal valid
        // glTF — an empty scene, no buffers/bufferViews/accessors.
        js << ",\"scene\":0,\"scenes\":[{\"nodes\":[]}]}";
    }

    std::string json = js.str();
    // pad JSON to 4 bytes with spaces (0x20)
    while (json.size() % 4 != 0) json.push_back(' ');
    // pad BIN to 4 bytes with zeros
    while (bin.size() % 4 != 0) bin.push_back(0);

    uint32_t json_len = (uint32_t)json.size();
    uint32_t bin_len = (uint32_t)bin.size();
    const bool emit_bin = bin_len > 0;
    uint32_t total = 12 + 8 + json_len + (emit_bin ? 8 + bin_len : 0);

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> hdr;
    put_u32(hdr, 0x46546C67);  // "glTF"
    put_u32(hdr, 2);           // version
    put_u32(hdr, total);
    put_u32(hdr, json_len);
    put_u32(hdr, 0x4E4F534A);  // "JSON"
    f.write((const char*)hdr.data(), hdr.size());
    f.write(json.data(), json.size());
    if (emit_bin) {
        std::vector<uint8_t> bhdr;
        put_u32(bhdr, bin_len);
        put_u32(bhdr, 0x004E4942);  // "BIN\0"
        f.write((const char*)bhdr.data(), bhdr.size());
        f.write((const char*)bin.data(), bin.size());
    }
    return (bool)f;
}

}  // namespace depth
}  // namespace aicore
