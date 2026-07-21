// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>
#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>

#include "BevRemapCuda.cuh"

namespace mcalib {
namespace bev_cuda {

namespace {

constexpr int kBlock = 16;

__global__ void remap_kernel(int src_w,
                             int src_h,
                             int dst_w,
                             int dst_h,
                             int channels,
                             const unsigned char* src,
                             const float* mapx,
                             const float* mapy,
                             unsigned char* dst) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int dst_idx = y * dst_w + x;
    const float xx = mapx[dst_idx];
    const float yy = mapy[dst_idx];
    const int X = static_cast<int>(floorf(xx));
    const int Y = static_cast<int>(floorf(yy));
    const float xfrac = xx - static_cast<float>(X);
    const float yfrac = yy - static_cast<float>(Y);

    if (X < 0 || Y < 0 || X >= src_w - 1 || Y >= src_h - 1) return;

    unsigned char* p_dst = &dst[dst_idx * channels];
    for (int c = 0; c < channels; ++c) {
        const unsigned char p00 = src[(Y * src_w + X) * channels + c];
        const unsigned char p01 = src[(Y * src_w + X + 1) * channels + c];
        const unsigned char p10 = src[((Y + 1) * src_w + X) * channels + c];
        const unsigned char p11 = src[((Y + 1) * src_w + X + 1) * channels + c];
        const float v = (p00 * (1.f - xfrac) + p01 * xfrac) * (1.f - yfrac) +
                        (p10 * (1.f - xfrac) + p11 * xfrac) * yfrac;
        p_dst[c] = static_cast<unsigned char>(v + 0.5f);
    }
}

__global__ void alpha_fusion_kernel(int w,
                                    int h,
                                    const unsigned char* src,
                                    const float* weight,
                                    float* dst_acc,
                                    unsigned char* dst_out) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int idx = y * w + x;
    const float wgt = weight[idx];
    const int pix = idx * 3;
    const unsigned char b = src[pix + 0];
    const unsigned char g = src[pix + 1];
    const unsigned char r = src[pix + 2];

    float fb = dst_acc[pix + 0] + wgt * static_cast<float>(b);
    float fg = dst_acc[pix + 1] + wgt * static_cast<float>(g);
    float fr = dst_acc[pix + 2] + wgt * static_cast<float>(r);
    dst_acc[pix + 0] = fb;
    dst_acc[pix + 1] = fg;
    dst_acc[pix + 2] = fr;

    dst_out[pix + 0] = static_cast<unsigned char>(fb + 0.5f);
    dst_out[pix + 1] = static_cast<unsigned char>(fg + 0.5f);
    dst_out[pix + 2] = static_cast<unsigned char>(fr + 0.5f);
}

int divUp(int total, int block) { return (total + block - 1) / block; }

bool checkCuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return true;
    CVLog::Warning("[BevCuda] %s failed: %s", what, cudaGetErrorString(err));
    return false;
}

struct DeviceBuffers {
    unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    float* mapx = nullptr;
    float* mapy = nullptr;
    size_t src_bytes = 0;
    size_t dst_bytes = 0;
    size_t map_bytes = 0;
};

DeviceBuffers g_bufs;
cudaStream_t g_stream = nullptr;
bool g_checked = false;
bool g_available = false;

struct FusionState {
    cv::Size size;
    std::vector<std::string> camera_order;
    std::map<std::string, float*> weight_dev;
    unsigned char* dst_acc = nullptr;
    unsigned char* dst_out = nullptr;
    size_t acc_bytes = 0;
    size_t out_bytes = 0;
    bool ready = false;
};

FusionState g_fusion;

void releaseFusionBuffers() {
    for (auto& [name, ptr] : g_fusion.weight_dev) {
        if (ptr) cudaFree(ptr);
    }
    g_fusion.weight_dev.clear();
    if (g_fusion.dst_acc) cudaFree(g_fusion.dst_acc);
    if (g_fusion.dst_out) cudaFree(g_fusion.dst_out);
    g_fusion.dst_acc = nullptr;
    g_fusion.dst_out = nullptr;
    g_fusion.acc_bytes = 0;
    g_fusion.out_bytes = 0;
    g_fusion.ready = false;
}

void releaseBuffers() {
    if (g_bufs.src) cudaFree(g_bufs.src);
    if (g_bufs.dst) cudaFree(g_bufs.dst);
    if (g_bufs.mapx) cudaFree(g_bufs.mapx);
    if (g_bufs.mapy) cudaFree(g_bufs.mapy);
    g_bufs = {};
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
}

bool ensureCapacity(const cv::Mat& src,
                    const cv::Mat& dst,
                    const cv::Mat& mapx,
                    const cv::Mat& mapy) {
    const size_t src_bytes =
            static_cast<size_t>(src.cols * src.rows * src.channels());
    const size_t dst_bytes =
            static_cast<size_t>(dst.cols * dst.rows * dst.channels());
    const size_t map_bytes = static_cast<size_t>(mapx.total() * sizeof(float));

    if (src_bytes <= g_bufs.src_bytes && dst_bytes <= g_bufs.dst_bytes &&
        map_bytes <= g_bufs.map_bytes) {
        return true;
    }

    releaseBuffers();
    if (!checkCuda(cudaMalloc(&g_bufs.src, src_bytes), "cudaMalloc src") ||
        !checkCuda(cudaMalloc(&g_bufs.dst, dst_bytes), "cudaMalloc dst") ||
        !checkCuda(cudaMalloc(&g_bufs.mapx, map_bytes), "cudaMalloc mapx") ||
        !checkCuda(cudaMalloc(&g_bufs.mapy, map_bytes), "cudaMalloc mapy") ||
        !checkCuda(cudaStreamCreate(&g_stream), "cudaStreamCreate")) {
        releaseBuffers();
        return false;
    }

    g_bufs.src_bytes = src_bytes;
    g_bufs.dst_bytes = dst_bytes;
    g_bufs.map_bytes = map_bytes;
    return true;
}

}  // namespace

bool isAvailable() {
    if (!g_checked) {
        int count = 0;
        g_available = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
        g_checked = true;
        if (g_available) {
            CVLog::Print(
                    "[BevCuda] CUDA remap backend available (%d device(s))",
                    count);
        }
    }
    return g_available;
}

bool remap(const cv::Mat& src,
           cv::Mat& dst,
           const cv::Mat& mapx,
           const cv::Mat& mapy) {
    if (!isAvailable() || src.empty() || mapx.empty() || mapy.empty()) {
        return false;
    }

    if (dst.empty() || dst.size() != mapx.size() || dst.type() != src.type()) {
        dst.create(mapx.size(), src.type());
    }

    cv::Mat src_c = src.isContinuous() ? src : src.clone();
    cv::Mat mapx_c = mapx.isContinuous() ? mapx : mapx.clone();
    cv::Mat mapy_c = mapy.isContinuous() ? mapy : mapy.clone();

    if (!ensureCapacity(src_c, dst, mapx_c, mapy_c)) return false;

    const int channels = src_c.channels();
    const size_t src_bytes =
            static_cast<size_t>(src_c.cols * src_c.rows * channels);
    const size_t dst_bytes =
            static_cast<size_t>(dst.cols * dst.rows * channels);
    const size_t map_bytes =
            static_cast<size_t>(mapx_c.total() * sizeof(float));

    if (!checkCuda(cudaMemcpyAsync(g_bufs.src, src_c.data, src_bytes,
                                   cudaMemcpyHostToDevice, g_stream),
                   "H2D src") ||
        !checkCuda(cudaMemcpyAsync(g_bufs.mapx, mapx_c.ptr<float>(), map_bytes,
                                   cudaMemcpyHostToDevice, g_stream),
                   "H2D mapx") ||
        !checkCuda(cudaMemcpyAsync(g_bufs.mapy, mapy_c.ptr<float>(), map_bytes,
                                   cudaMemcpyHostToDevice, g_stream),
                   "H2D mapy") ||
        !checkCuda(cudaMemsetAsync(g_bufs.dst, 0, dst_bytes, g_stream),
                   "memset dst")) {
        return false;
    }

    const dim3 block(kBlock, kBlock);
    const dim3 grid(divUp(dst.cols, kBlock), divUp(dst.rows, kBlock));
    remap_kernel<<<grid, block, 0, g_stream>>>(
            src_c.cols, src_c.rows, dst.cols, dst.rows, channels, g_bufs.src,
            g_bufs.mapx, g_bufs.mapy, g_bufs.dst);

    if (!checkCuda(cudaGetLastError(), "remap_kernel launch") ||
        !checkCuda(cudaMemcpyAsync(dst.data, g_bufs.dst, dst_bytes,
                                   cudaMemcpyDeviceToHost, g_stream),
                   "D2H dst") ||
        !checkCuda(cudaStreamSynchronize(g_stream), "sync")) {
        return false;
    }
    return true;
}

bool initAlphaFusion(cv::Size bev_size,
                     const std::vector<std::string>& camera_order,
                     const std::map<std::string, cv::Mat>& weights) {
    if (!isAvailable()) return false;

    releaseFusionBuffers();
    g_fusion.size = bev_size;
    g_fusion.camera_order = camera_order;

    const size_t map_bytes = static_cast<size_t>(
            bev_size.width * bev_size.height * sizeof(float));
    g_fusion.acc_bytes = static_cast<size_t>(bev_size.width * bev_size.height *
                                             3 * sizeof(float));
    g_fusion.out_bytes =
            static_cast<size_t>(bev_size.width * bev_size.height * 3);

    if (!g_stream && !checkCuda(cudaStreamCreate(&g_stream), "fusion stream")) {
        return false;
    }

    for (const auto& name : camera_order) {
        auto it = weights.find(name);
        if (it == weights.end() || it->second.empty()) return false;

        cv::Mat w = it->second.isContinuous() ? it->second : it->second.clone();
        float* dev = nullptr;
        if (!checkCuda(cudaMalloc(&dev, map_bytes), "cudaMalloc weight") ||
            !checkCuda(cudaMemcpyAsync(dev, w.ptr<float>(), map_bytes,
                                       cudaMemcpyHostToDevice, g_stream),
                       "H2D weight")) {
            if (dev) cudaFree(dev);
            releaseFusionBuffers();
            return false;
        }
        g_fusion.weight_dev[name] = dev;
    }

    if (!checkCuda(cudaMalloc(&g_fusion.dst_acc, g_fusion.acc_bytes),
                   "cudaMalloc dst_acc") ||
        !checkCuda(cudaMalloc(&g_fusion.dst_out, g_fusion.out_bytes),
                   "cudaMalloc dst_out") ||
        !checkCuda(cudaStreamSynchronize(g_stream), "fusion init sync")) {
        releaseFusionBuffers();
        return false;
    }

    g_fusion.ready = true;
    return true;
}

void updateAlphaFusionWeights(const std::map<std::string, cv::Mat>& weights) {
    if (!g_fusion.ready || !g_stream) return;

    const size_t map_bytes = static_cast<size_t>(
            g_fusion.size.width * g_fusion.size.height * sizeof(float));

    for (const auto& name : g_fusion.camera_order) {
        auto dev_it = g_fusion.weight_dev.find(name);
        auto w_it = weights.find(name);
        if (dev_it == g_fusion.weight_dev.end() || w_it == weights.end())
            continue;

        cv::Mat w = w_it->second.isContinuous() ? w_it->second
                                                : w_it->second.clone();
        checkCuda(cudaMemcpyAsync(dev_it->second, w.ptr<float>(), map_bytes,
                                  cudaMemcpyHostToDevice, g_stream),
                  "H2D weight update");
    }
    checkCuda(cudaStreamSynchronize(g_stream), "fusion weight sync");
}

void releaseAlphaFusion() { releaseFusionBuffers(); }

bool alphaFusion(const std::vector<std::string>& camera_order,
                 const std::map<std::string, cv::Mat>& warped_images,
                 cv::Mat& output) {
    if (!g_fusion.ready || !g_stream) return false;

    if (output.empty() || output.size() != g_fusion.size ||
        output.type() != CV_8UC3) {
        output.create(g_fusion.size, CV_8UC3);
    }

    if (!checkCuda(cudaMemsetAsync(g_fusion.dst_acc, 0, g_fusion.acc_bytes,
                                   g_stream),
                   "memset acc") ||
        !checkCuda(cudaMemsetAsync(g_fusion.dst_out, 0, g_fusion.out_bytes,
                                   g_stream),
                   "memset out")) {
        return false;
    }

    const int w = g_fusion.size.width;
    const int h = g_fusion.size.height;
    const dim3 block(kBlock, kBlock);
    const dim3 grid(divUp(w, kBlock), divUp(h, kBlock));

    for (const auto& name : camera_order) {
        auto img_it = warped_images.find(name);
        auto w_it = g_fusion.weight_dev.find(name);
        if (img_it == warped_images.end() || img_it->second.empty() ||
            w_it == g_fusion.weight_dev.end()) {
            continue;
        }

        cv::Mat img = img_it->second.isContinuous() ? img_it->second
                                                    : img_it->second.clone();
        if (img.size() != g_fusion.size || img.type() != CV_8UC3) continue;

        unsigned char* dev_src = nullptr;
        const size_t img_bytes =
                static_cast<size_t>(img.total() * img.elemSize());
        if (!checkCuda(cudaMalloc(&dev_src, img_bytes), "cudaMalloc src") ||
            !checkCuda(cudaMemcpyAsync(dev_src, img.data, img_bytes,
                                       cudaMemcpyHostToDevice, g_stream),
                       "H2D src")) {
            if (dev_src) cudaFree(dev_src);
            return false;
        }

        alpha_fusion_kernel<<<grid, block, 0, g_stream>>>(
                w, h, dev_src, w_it->second,
                reinterpret_cast<float*>(g_fusion.dst_acc), g_fusion.dst_out);

        cudaFree(dev_src);

        if (!checkCuda(cudaGetLastError(), "alpha_fusion_kernel")) {
            return false;
        }
    }

    if (!checkCuda(cudaMemcpyAsync(output.data, g_fusion.dst_out,
                                   g_fusion.out_bytes, cudaMemcpyDeviceToHost,
                                   g_stream),
                   "D2H fusion") ||
        !checkCuda(cudaStreamSynchronize(g_stream), "fusion sync")) {
        return false;
    }
    return true;
}

__global__ void lidar_project_kernel(int n,
                                     const float* pts,
                                     const float* rot,
                                     const float* trans,
                                     float fx,
                                     float fy,
                                     float cx,
                                     float cy,
                                     float* uv,
                                     float* depth,
                                     unsigned char* valid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float x = pts[i * 3 + 0];
    const float y = pts[i * 3 + 1];
    const float z = pts[i * 3 + 2];

    const float pcx = rot[0] * x + rot[1] * y + rot[2] * z + trans[0];
    const float pcy = rot[3] * x + rot[4] * y + rot[5] * z + trans[1];
    const float pcz = rot[6] * x + rot[7] * y + rot[8] * z + trans[2];
    if (pcz <= 0.f) {
        valid[i] = 0;
        return;
    }

    valid[i] = 1;
    uv[i * 2 + 0] = fx * pcx / pcz + cx;
    uv[i * 2 + 1] = fy * pcy / pcz + cy;
    depth[i] = pcz;
}

__device__ float kb_radius(float theta, const float* kb) {
    const float t2 = theta * theta;
    const float t3 = t2 * theta;
    const float t5 = t2 * t3;
    const float t7 = t2 * t5;
    const float t9 = t2 * t7;
    return theta + kb[0] * t3 + kb[1] * t5 + kb[2] * t7 + kb[3] * t9;
}

__global__ void lidar_project_kb_kernel(int n,
                                        const float* pts,
                                        const float* rot,
                                        const float* trans,
                                        float fx,
                                        float fy,
                                        float cx,
                                        float cy,
                                        const float* kb,
                                        float* uv,
                                        float* depth,
                                        unsigned char* valid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float x = pts[i * 3 + 0];
    const float y = pts[i * 3 + 1];
    const float z = pts[i * 3 + 2];

    const float pcx = rot[0] * x + rot[1] * y + rot[2] * z + trans[0];
    const float pcy = rot[3] * x + rot[4] * y + rot[5] * z + trans[1];
    const float pcz = rot[6] * x + rot[7] * y + rot[8] * z + trans[2];
    const float len = sqrtf(pcx * pcx + pcy * pcy + pcz * pcz);
    if (len < 1e-12f || pcz <= 0.f) {
        valid[i] = 0;
        return;
    }

    const float theta = acosf(fminf(fmaxf(pcz / len, -1.f), 1.f));
    const float phi = atan2f(pcy, pcx);
    const float r = kb_radius(theta, kb);
    valid[i] = 1;
    uv[i * 2 + 0] = fx * r * cosf(phi) + cx;
    uv[i * 2 + 1] = fy * r * sinf(phi) + cy;
    depth[i] = pcz;
}

bool projectPoints(const float* points_xyz,
                   int num_points,
                   const float rotation[9],
                   const float translation[3],
                   float fx,
                   float fy,
                   float cx,
                   float cy,
                   std::vector<cv::Point2f>& image_points,
                   std::vector<float>& depths) {
    image_points.clear();
    depths.clear();
    if (!points_xyz || num_points <= 0 || !isAvailable()) return false;

    float* dev_pts = nullptr;
    float* dev_rot = nullptr;
    float* dev_trans = nullptr;
    float* dev_uv = nullptr;
    float* dev_depth = nullptr;
    unsigned char* dev_valid = nullptr;

    const size_t pts_bytes =
            static_cast<size_t>(num_points) * 3 * sizeof(float);
    const size_t uv_bytes = static_cast<size_t>(num_points) * 2 * sizeof(float);
    const size_t depth_bytes = static_cast<size_t>(num_points) * sizeof(float);
    const size_t valid_bytes =
            static_cast<size_t>(num_points) * sizeof(unsigned char);

    if (!checkCuda(cudaMalloc(&dev_pts, pts_bytes), "malloc pts") ||
        !checkCuda(cudaMalloc(&dev_rot, 9 * sizeof(float)), "malloc rot") ||
        !checkCuda(cudaMalloc(&dev_trans, 3 * sizeof(float)), "malloc trans") ||
        !checkCuda(cudaMalloc(&dev_uv, uv_bytes), "malloc uv") ||
        !checkCuda(cudaMalloc(&dev_depth, depth_bytes), "malloc depth") ||
        !checkCuda(cudaMalloc(&dev_valid, valid_bytes), "malloc valid")) {
        if (dev_pts) cudaFree(dev_pts);
        if (dev_rot) cudaFree(dev_rot);
        if (dev_trans) cudaFree(dev_trans);
        if (dev_uv) cudaFree(dev_uv);
        if (dev_depth) cudaFree(dev_depth);
        if (dev_valid) cudaFree(dev_valid);
        return false;
    }

    checkCuda(
            cudaMemcpy(dev_pts, points_xyz, pts_bytes, cudaMemcpyHostToDevice),
            "H2D pts");
    checkCuda(cudaMemcpy(dev_rot, rotation, 9 * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D rot");
    checkCuda(cudaMemcpy(dev_trans, translation, 3 * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D trans");

    const int block = 256;
    const int grid = divUp(num_points, block);
    lidar_project_kernel<<<grid, block>>>(num_points, dev_pts, dev_rot,
                                          dev_trans, fx, fy, cx, cy, dev_uv,
                                          dev_depth, dev_valid);

    std::vector<float> host_uv(num_points * 2);
    std::vector<float> host_depth(num_points);
    std::vector<unsigned char> host_valid(num_points);

    bool ok = checkCuda(cudaGetLastError(), "lidar_project_kernel") &&
              checkCuda(cudaMemcpy(host_uv.data(), dev_uv, uv_bytes,
                                   cudaMemcpyDeviceToHost),
                        "D2H uv") &&
              checkCuda(cudaMemcpy(host_depth.data(), dev_depth, depth_bytes,
                                   cudaMemcpyDeviceToHost),
                        "D2H depth") &&
              checkCuda(cudaMemcpy(host_valid.data(), dev_valid, valid_bytes,
                                   cudaMemcpyDeviceToHost),
                        "D2H valid");

    cudaFree(dev_pts);
    cudaFree(dev_rot);
    cudaFree(dev_trans);
    cudaFree(dev_uv);
    cudaFree(dev_depth);
    cudaFree(dev_valid);

    if (!ok) return false;

    image_points.reserve(num_points);
    depths.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
        if (!host_valid[i]) continue;
        image_points.emplace_back(host_uv[i * 2], host_uv[i * 2 + 1]);
        depths.push_back(host_depth[i]);
    }
    return !image_points.empty();
}

bool projectPointsKb(const float* points_xyz,
                     int num_points,
                     const float rotation[9],
                     const float translation[3],
                     float fx,
                     float fy,
                     float cx,
                     float cy,
                     const float kb[4],
                     std::vector<cv::Point2f>& image_points,
                     std::vector<float>& depths) {
    image_points.clear();
    depths.clear();
    if (!points_xyz || num_points <= 0 || !kb || !isAvailable()) return false;

    float* dev_pts = nullptr;
    float* dev_rot = nullptr;
    float* dev_trans = nullptr;
    float* dev_kb = nullptr;
    float* dev_uv = nullptr;
    float* dev_depth = nullptr;
    unsigned char* dev_valid = nullptr;

    const size_t pts_bytes =
            static_cast<size_t>(num_points) * 3 * sizeof(float);
    const size_t uv_bytes = static_cast<size_t>(num_points) * 2 * sizeof(float);
    const size_t depth_bytes = static_cast<size_t>(num_points) * sizeof(float);
    const size_t valid_bytes =
            static_cast<size_t>(num_points) * sizeof(unsigned char);

    if (!checkCuda(cudaMalloc(&dev_pts, pts_bytes), "malloc pts") ||
        !checkCuda(cudaMalloc(&dev_rot, 9 * sizeof(float)), "malloc rot") ||
        !checkCuda(cudaMalloc(&dev_trans, 3 * sizeof(float)), "malloc trans") ||
        !checkCuda(cudaMalloc(&dev_kb, 4 * sizeof(float)), "malloc kb") ||
        !checkCuda(cudaMalloc(&dev_uv, uv_bytes), "malloc uv") ||
        !checkCuda(cudaMalloc(&dev_depth, depth_bytes), "malloc depth") ||
        !checkCuda(cudaMalloc(&dev_valid, valid_bytes), "malloc valid")) {
        if (dev_pts) cudaFree(dev_pts);
        if (dev_rot) cudaFree(dev_rot);
        if (dev_trans) cudaFree(dev_trans);
        if (dev_kb) cudaFree(dev_kb);
        if (dev_uv) cudaFree(dev_uv);
        if (dev_depth) cudaFree(dev_depth);
        if (dev_valid) cudaFree(dev_valid);
        return false;
    }

    checkCuda(
            cudaMemcpy(dev_pts, points_xyz, pts_bytes, cudaMemcpyHostToDevice),
            "H2D pts");
    checkCuda(cudaMemcpy(dev_rot, rotation, 9 * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D rot");
    checkCuda(cudaMemcpy(dev_trans, translation, 3 * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D trans");
    checkCuda(cudaMemcpy(dev_kb, kb, 4 * sizeof(float), cudaMemcpyHostToDevice),
              "H2D kb");

    const int block = 256;
    const int grid = divUp(num_points, block);
    lidar_project_kb_kernel<<<grid, block>>>(num_points, dev_pts, dev_rot,
                                             dev_trans, fx, fy, cx, cy, dev_kb,
                                             dev_uv, dev_depth, dev_valid);

    std::vector<float> host_uv(num_points * 2);
    std::vector<float> host_depth(num_points);
    std::vector<unsigned char> host_valid(num_points);

    bool ok = checkCuda(cudaGetLastError(), "lidar_project_kb_kernel") &&
              checkCuda(cudaMemcpy(host_uv.data(), dev_uv, uv_bytes,
                                   cudaMemcpyDeviceToHost),
                        "D2H uv") &&
              checkCuda(cudaMemcpy(host_depth.data(), dev_depth, depth_bytes,
                                   cudaMemcpyDeviceToHost),
                        "D2H depth") &&
              checkCuda(cudaMemcpy(host_valid.data(), dev_valid, valid_bytes,
                                   cudaMemcpyDeviceToHost),
                        "D2H valid");

    cudaFree(dev_pts);
    cudaFree(dev_rot);
    cudaFree(dev_trans);
    cudaFree(dev_kb);
    cudaFree(dev_uv);
    cudaFree(dev_depth);
    cudaFree(dev_valid);

    if (!ok) return false;

    image_points.reserve(num_points);
    depths.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
        if (!host_valid[i]) continue;
        image_points.emplace_back(host_uv[i * 2], host_uv[i * 2 + 1]);
        depths.push_back(host_depth[i]);
    }
    return !image_points.empty();
}

}  // namespace bev_cuda
}  // namespace mcalib
