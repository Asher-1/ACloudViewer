// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "colmap_export.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "reconstruct.hpp"

namespace aicore {
namespace depth {

namespace {

// ---------------------------------------------------------------------------
// Little-endian POD writers. The host is assumed little-endian (every platform
// we target is); we serialize byte-by-byte from the native value, which is
// correct on LE and keeps the code dependency-free.
// ---------------------------------------------------------------------------
template <class T>
void put(std::ostream& o, T v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

// Per-frame derived camera/pose data.
struct FrameData {
    int orig_w, orig_h;
    double fx, fy, cx, cy;      // rescaled PINHOLE params
    std::array<float, 4> qvec;  // qw,qx,qy,qz
    double tx, ty, tz;          // tvec
};

}  // namespace

bool write_colmap(const std::string& dir,
                  const std::vector<float>& depth,
                  const std::vector<float>& conf,
                  const std::vector<std::array<float, 9>>& K,
                  const std::vector<std::array<float, 16>>& ext,
                  const std::vector<const uint8_t*>& images_u8,
                  const std::vector<std::string>& image_names,
                  const std::vector<std::pair<int, int>>& orig_wh,
                  int H,
                  int W,
                  int N,
                  bool binary) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return false;

    // 1. Confidence threshold (numpy percentile, linear interp) over all conf.
    const float conf_thr =
            conf.empty() ? 0.0f : (float)percentile_linear(conf, 40.0);

    // 2. Back-project -> world points, colors, and per-point (frame,u,v).
    WorldPoints wp =
            back_project(depth, conf, K, ext, images_u8, H, W, N, conf_thr);
    const size_t num_points = wp.frame.size();

    // 3. Per-frame derived data (rescaled intrinsics + pose).
    std::vector<FrameData> fd((size_t)N);
    for (int i = 0; i < N; ++i) {
        const int ow = orig_wh[(size_t)i].first;
        const int oh = orig_wh[(size_t)i].second;
        const double sw = (double)ow / (double)W;
        const double sh = (double)oh / (double)H;
        const auto& k = K[(size_t)i];
        FrameData& f = fd[(size_t)i];
        f.orig_w = ow;
        f.orig_h = oh;
        f.fx = (double)k[0] * sw;  // fx, row 0
        f.cx = (double)k[2] * sw;  // cx, row 0
        f.fy = (double)k[4] * sh;  // fy, row 1
        f.cy = (double)k[5] * sh;  // cy, row 1
        std::array<float, 9> R = {
                ext[(size_t)i][0], ext[(size_t)i][1], ext[(size_t)i][2],
                ext[(size_t)i][4], ext[(size_t)i][5], ext[(size_t)i][6],
                ext[(size_t)i][8], ext[(size_t)i][9], ext[(size_t)i][10]};
        f.qvec = rotmat2qvec(R);
        f.tx = ext[(size_t)i][3];
        f.ty = ext[(size_t)i][7];
        f.tz = ext[(size_t)i][11];
    }

    // 4. Per-image point2D lists + per-point track (each point observed once).
    //    Back-projection orders points frame-outer, so each frame's points are
    //    contiguous; point2D_idx is the running index within the image.
    //    point3D id = global index + 1.
    struct Pt2D {
        double x, y;
        int64_t point3D_id;
    };
    std::vector<std::vector<Pt2D>> img_pts((size_t)N);
    std::vector<int> track_image_id(num_points);  // observing image id
    std::vector<int> track_pt2d_idx(num_points);  // running idx within image
    for (size_t p = 0; p < num_points; ++p) {
        const int fr = wp.frame[p];
        const FrameData& f = fd[(size_t)fr];
        const double sw = (double)f.orig_w / (double)W;
        const double sh = (double)f.orig_h / (double)H;
        // Reference scales an int32 array in place -> truncation toward zero.
        Pt2D pt;
        pt.x = (double)(long long)((double)wp.u[p] * sw);
        pt.y = (double)(long long)((double)wp.v[p] * sh);
        pt.point3D_id = (int64_t)(p + 1);
        const int idx = (int)img_pts[(size_t)fr].size();
        img_pts[(size_t)fr].push_back(pt);
        track_image_id[p] = fr + 1;
        track_pt2d_idx[p] = idx;
    }

    if (binary) {
        // -- cameras.bin --
        {
            std::ofstream o(dir + "/cameras.bin", std::ios::binary);
            if (!o) return false;
            put<uint64_t>(o, (uint64_t)N);
            for (int i = 0; i < N; ++i) {
                const FrameData& f = fd[(size_t)i];
                put<int32_t>(o, (int32_t)(i + 1));  // camera id
                put<int32_t>(o, (int32_t)1);        // model_id = PINHOLE
                put<uint64_t>(o, (uint64_t)f.orig_w);
                put<uint64_t>(o, (uint64_t)f.orig_h);
                put<double>(o, f.fx);
                put<double>(o, f.fy);
                put<double>(o, f.cx);
                put<double>(o, f.cy);
            }
        }
        // -- images.bin --
        {
            std::ofstream o(dir + "/images.bin", std::ios::binary);
            if (!o) return false;
            put<uint64_t>(o, (uint64_t)N);
            for (int i = 0; i < N; ++i) {
                const FrameData& f = fd[(size_t)i];
                put<int32_t>(o, (int32_t)(i + 1));  // image id
                put<double>(o, (double)f.qvec[0]);
                put<double>(o, (double)f.qvec[1]);
                put<double>(o, (double)f.qvec[2]);
                put<double>(o, (double)f.qvec[3]);
                put<double>(o, f.tx);
                put<double>(o, f.ty);
                put<double>(o, f.tz);
                put<int32_t>(o, (int32_t)(i + 1));  // camera id
                const std::string& nm = image_names[(size_t)i];
                o.write(nm.data(), (std::streamsize)nm.size());
                put<char>(o, '\0');
                const auto& pts = img_pts[(size_t)i];
                put<uint64_t>(o, (uint64_t)pts.size());
                for (const Pt2D& p : pts) {
                    put<double>(o, p.x);
                    put<double>(o, p.y);
                    put<int64_t>(o, p.point3D_id);
                }
            }
        }
        // -- points3D.bin --
        {
            std::ofstream o(dir + "/points3D.bin", std::ios::binary);
            if (!o) return false;
            put<uint64_t>(o, (uint64_t)num_points);
            for (size_t p = 0; p < num_points; ++p) {
                put<uint64_t>(o, (uint64_t)(p + 1));
                put<double>(o, (double)wp.xyz[3 * p + 0]);
                put<double>(o, (double)wp.xyz[3 * p + 1]);
                put<double>(o, (double)wp.xyz[3 * p + 2]);
                put<uint8_t>(o, wp.rgb[3 * p + 0]);
                put<uint8_t>(o, wp.rgb[3 * p + 1]);
                put<uint8_t>(o, wp.rgb[3 * p + 2]);
                put<double>(o, 0.0);            // error
                put<uint64_t>(o, (uint64_t)1);  // track length (single obs)
                put<int32_t>(o, (int32_t)track_image_id[p]);
                put<int32_t>(o, (int32_t)track_pt2d_idx[p]);
            }
        }
        return true;
    }

    // ----------------------- text format -----------------------
    // -- cameras.txt --
    {
        std::ofstream o(dir + "/cameras.txt");
        if (!o) return false;
        o << "# Camera list with one line of data per camera:\n"
          << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
          << "# Number of cameras: " << N << "\n";
        char buf[256];
        for (int i = 0; i < N; ++i) {
            const FrameData& f = fd[(size_t)i];
            std::snprintf(buf, sizeof(buf),
                          "%d PINHOLE %d %d %.17g %.17g %.17g %.17g", i + 1,
                          f.orig_w, f.orig_h, f.fx, f.fy, f.cx, f.cy);
            o << buf << "\n";
        }
    }
    // -- images.txt --
    {
        std::ofstream o(dir + "/images.txt");
        if (!o) return false;
        double mean_obs = 0.0;
        if (N > 0) {
            size_t total = 0;
            for (int i = 0; i < N; ++i) total += img_pts[(size_t)i].size();
            mean_obs = (double)total / (double)N;
        }
        o << "# Image list with two lines of data per image:\n"
          << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
          << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
          << "# Number of images: " << N
          << ", mean observations per image: " << mean_obs << "\n";
        char buf[256];
        for (int i = 0; i < N; ++i) {
            const FrameData& f = fd[(size_t)i];
            std::snprintf(buf, sizeof(buf),
                          "%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %s",
                          i + 1, (double)f.qvec[0], (double)f.qvec[1],
                          (double)f.qvec[2], (double)f.qvec[3], f.tx, f.ty,
                          f.tz, i + 1, image_names[(size_t)i].c_str());
            o << buf << "\n";
            const auto& pts = img_pts[(size_t)i];
            for (size_t j = 0; j < pts.size(); ++j) {
                if (j) o << " ";
                std::snprintf(buf, sizeof(buf), "%.17g %.17g %lld", pts[j].x,
                              pts[j].y, (long long)pts[j].point3D_id);
                o << buf;
            }
            o << "\n";
        }
    }
    // -- points3D.txt --
    {
        std::ofstream o(dir + "/points3D.txt");
        if (!o) return false;
        const double mean_track =
                num_points ? 1.0 : 0.0;  // single obs per point
        o << "# 3D point list with one line of data per point:\n"
          << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, "
             "POINT2D_IDX)\n"
          << "# Number of points: " << num_points
          << ", mean track length: " << mean_track << "\n";
        char buf[256];
        for (size_t p = 0; p < num_points; ++p) {
            std::snprintf(buf, sizeof(buf),
                          "%llu %.17g %.17g %.17g %d %d %d %.17g %d %d",
                          (unsigned long long)(p + 1),
                          (double)wp.xyz[3 * p + 0], (double)wp.xyz[3 * p + 1],
                          (double)wp.xyz[3 * p + 2], (int)wp.rgb[3 * p + 0],
                          (int)wp.rgb[3 * p + 1], (int)wp.rgb[3 * p + 2], 0.0,
                          track_image_id[p], track_pt2d_idx[p]);
            o << buf << "\n";
        }
    }
    return true;
}

}  // namespace depth
}  // namespace aicore
