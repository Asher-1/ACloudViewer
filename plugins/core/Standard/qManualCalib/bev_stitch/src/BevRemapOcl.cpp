// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BevRemapOcl.h"

#include <CL/cl.h>
#include <CVLog.h>

#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace mcalib {
namespace bev_ocl {

namespace {

constexpr int kBlock = 16;

const char* kRemapKernel = R"(
__kernel void remap_kernel(__global const uchar* src,
                          __global const float* mapx,
                          __global const float* mapy,
                          __global uchar* dst,
                          int src_width, int src_height,
                          int dst_width, int dst_height,
                          int channels) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= dst_width || y >= dst_height) return;

  const int dst_idx = y * dst_width + x;
  const float xx = mapx[dst_idx];
  const float yy = mapy[dst_idx];
  const int X = (int)floor(xx);
  const int Y = (int)floor(yy);
  const float xfrac = xx - X;
  const float yfrac = yy - Y;

  if (X >= 0 && X < src_width - 1 && Y >= 0 && Y < src_height - 1) {
    const int dst_pixel_idx = (y * dst_width + x) * channels;
    for (int c = 0; c < channels; ++c) {
      const int i00 = (Y * src_width + X) * channels + c;
      const int i01 = (Y * src_width + (X + 1)) * channels + c;
      const int i10 = ((Y + 1) * src_width + X) * channels + c;
      const int i11 = ((Y + 1) * src_width + (X + 1)) * channels + c;
      const float p00 = src[i00];
      const float p01 = src[i01];
      const float p10 = src[i10];
      const float p11 = src[i11];
      const float result = (p00 * (1.f - xfrac) + p01 * xfrac) * (1.f - yfrac) +
                           (p10 * (1.f - xfrac) + p11 * xfrac) * yfrac;
      dst[dst_pixel_idx + c] = convert_uchar_sat(result + 0.5f);
    }
  }
}
)";

const char* kAlphaFusionKernel = R"(
__kernel void alpha_fusion_kernel(int width, int height, int num_cameras, int channels,
                                  __global const uchar* src1, __global const float* weight1,
                                  __global const uchar* src2, __global const float* weight2,
                                  __global const uchar* src3, __global const float* weight3,
                                  __global const uchar* src4, __global const float* weight4,
                                  __global const uchar* src5, __global const float* weight5,
                                  __global const uchar* src6, __global const float* weight6,
                                  __global uchar* dst_uchar) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= width || y >= height) return;

  const int pixel_idx = y * width + x;
  for (int c = 0; c < channels; ++c) {
    const int idx = pixel_idx * channels + c;
    float result = 0.0f;
    if (num_cameras >= 1) result += weight1[pixel_idx] * src1[idx];
    if (num_cameras >= 2) result += weight2[pixel_idx] * src2[idx];
    if (num_cameras >= 3) result += weight3[pixel_idx] * src3[idx];
    if (num_cameras >= 4) result += weight4[pixel_idx] * src4[idx];
    if (num_cameras >= 5) result += weight5[pixel_idx] * src5[idx];
    if (num_cameras >= 6) result += weight6[pixel_idx] * src6[idx];
    dst_uchar[idx] = convert_uchar_sat(result + 0.5f);
  }
}
)";

const char* kProjectKernel = R"(
__kernel void lidar_project_kernel(__global const float* pts,
                                 __global const float* rot,
                                 __global const float* trans,
                                 int num_points,
                                 float fx, float fy, float cx, float cy,
                                 __global float* uv,
                                 __global float* depth,
                                 __global uchar* valid) {
  const int i = get_global_id(0);
  if (i >= num_points) return;
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

float kb_radius(float theta, float k1, float k2, float k3, float k4) {
  const float t2 = theta * theta;
  const float t3 = t2 * theta;
  const float t5 = t2 * t3;
  const float t7 = t2 * t5;
  const float t9 = t2 * t7;
  return theta + k1 * t3 + k2 * t5 + k3 * t7 + k4 * t9;
}

__kernel void lidar_project_kb_kernel(__global const float* pts,
                                   __global const float* rot,
                                   __global const float* trans,
                                   int num_points,
                                   float fx, float fy, float cx, float cy,
                                   float k1, float k2, float k3, float k4,
                                   __global float* uv,
                                   __global float* depth,
                                   __global uchar* valid) {
  const int i = get_global_id(0);
  if (i >= num_points) return;
  const float x = pts[i * 3 + 0];
  const float y = pts[i * 3 + 1];
  const float z = pts[i * 3 + 2];
  const float pcx = rot[0] * x + rot[1] * y + rot[2] * z + trans[0];
  const float pcy = rot[3] * x + rot[4] * y + rot[5] * z + trans[1];
  const float pcz = rot[6] * x + rot[7] * y + rot[8] * z + trans[2];
  const float len = sqrt(pcx * pcx + pcy * pcy + pcz * pcz);
  if (len < 1e-12f || pcz <= 0.f) {
    valid[i] = 0;
    return;
  }
  const float theta = acos(fmin(fmax(pcz / len, -1.f), 1.f));
  const float phi = atan2(pcy, pcx);
  const float r = kb_radius(theta, k1, k2, k3, k4);
  valid[i] = 1;
  uv[i * 2 + 0] = fx * r * cos(phi) + cx;
  uv[i * 2 + 1] = fy * r * sin(phi) + cy;
  depth[i] = pcz;
}
)";

struct OclState {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_program fusion_program = nullptr;
    cl_kernel fusion_kernel = nullptr;
    cl_program project_program = nullptr;
    cl_kernel project_kernel = nullptr;
    cl_kernel project_kb_kernel = nullptr;
    bool initialized = false;
    bool available = false;
};

struct FusionState {
    cv::Size size;
    std::vector<std::string> camera_order;
    std::map<std::string, cl_mem> weight_bufs;
    bool ready = false;
};

std::mutex g_mutex;
OclState g_state;
FusionState g_fusion;

cl_command_queue createCommandQueue(cl_context context,
                                    cl_device_id device,
                                    cl_int* err_out) {
    cl_int err = CL_SUCCESS;
#if defined(CL_VERSION_2_0)
    const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue2 =
            clCreateCommandQueueWithProperties(context, device, props, &err);
    if (queue2) {
        if (err_out) *err_out = err;
        return queue2;
    }
#endif
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err_out) *err_out = err;
    return queue;
}

bool probeOpenCL() {
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS ||
        num_platforms == 0) {
        return false;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    if (clGetPlatformIDs(num_platforms, platforms.data(), nullptr) !=
        CL_SUCCESS) {
        return false;
    }
    cl_device_id device = nullptr;
    if (clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, nullptr) ==
        CL_SUCCESS) {
        return true;
    }
    return clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &device,
                          nullptr) == CL_SUCCESS;
}

bool initOnce() {
    if (g_state.initialized) return g_state.available;
    g_state.initialized = true;

    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS ||
        num_platforms == 0) {
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    if (clGetPlatformIDs(num_platforms, platforms.data(), nullptr) !=
        CL_SUCCESS) {
        return false;
    }

    cl_platform_id platform = platforms[0];
    cl_device_id device = nullptr;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) !=
        CL_SUCCESS) {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr) !=
            CL_SUCCESS) {
            return false;
        }
    }

    cl_int err = CL_SUCCESS;
    g_state.context =
            clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !g_state.context) return false;

    g_state.queue = createCommandQueue(g_state.context, device, &err);
    if (err != CL_SUCCESS || !g_state.queue) return false;

    g_state.program = clCreateProgramWithSource(g_state.context, 1,
                                                &kRemapKernel, nullptr, &err);
    if (err != CL_SUCCESS || !g_state.program) return false;

    err = clBuildProgram(g_state.program, 1, &device, nullptr, nullptr,
                         nullptr);
    if (err != CL_SUCCESS) {
        char log[4096] = {};
        clGetProgramBuildInfo(g_state.program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, nullptr);
        CVLog::Warning("[BevOCL] remap build failed: %s", log);
        return false;
    }

    g_state.kernel = clCreateKernel(g_state.program, "remap_kernel", &err);
    if (err != CL_SUCCESS || !g_state.kernel) return false;

    g_state.fusion_program = clCreateProgramWithSource(
            g_state.context, 1, &kAlphaFusionKernel, nullptr, &err);
    if (err != CL_SUCCESS || !g_state.fusion_program) return false;

    err = clBuildProgram(g_state.fusion_program, 1, &device, nullptr, nullptr,
                         nullptr);
    if (err != CL_SUCCESS) {
        char log[4096] = {};
        clGetProgramBuildInfo(g_state.fusion_program, device,
                              CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        CVLog::Warning("[BevOCL] fusion build failed: %s", log);
        return false;
    }

    g_state.fusion_kernel =
            clCreateKernel(g_state.fusion_program, "alpha_fusion_kernel", &err);
    if (err != CL_SUCCESS || !g_state.fusion_kernel) return false;

    g_state.project_program = clCreateProgramWithSource(
            g_state.context, 1, &kProjectKernel, nullptr, &err);
    if (err != CL_SUCCESS || !g_state.project_program) return false;

    err = clBuildProgram(g_state.project_program, 1, &device, nullptr, nullptr,
                         nullptr);
    if (err != CL_SUCCESS) {
        char log[4096] = {};
        clGetProgramBuildInfo(g_state.project_program, device,
                              CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        CVLog::Warning("[BevOCL] project build failed: %s", log);
        return false;
    }

    g_state.project_kernel = clCreateKernel(g_state.project_program,
                                            "lidar_project_kernel", &err);
    if (err != CL_SUCCESS || !g_state.project_kernel) return false;

    g_state.project_kb_kernel = clCreateKernel(g_state.project_program,
                                               "lidar_project_kb_kernel", &err);
    if (err != CL_SUCCESS || !g_state.project_kb_kernel) return false;

    g_state.available = true;
    CVLog::Print("[BevOCL] OpenCL remap backend initialized");
    return true;
}

}  // namespace

bool probePlatform() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return probeOpenCL();
}

bool isAvailable() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return initOnce();
}

bool remap(const cv::Mat& src,
           cv::Mat& dst,
           const cv::Mat& mapx,
           const cv::Mat& mapy) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!initOnce() || src.empty() || mapx.empty() || mapy.empty()) {
        return false;
    }

    if (dst.empty() || dst.size() != mapx.size() || dst.type() != src.type()) {
        dst.create(mapx.size(), src.type());
    }

    cv::Mat src_c = src.isContinuous() ? src : src.clone();
    cv::Mat mapx_c = mapx.isContinuous() ? mapx : mapx.clone();
    cv::Mat mapy_c = mapy.isContinuous() ? mapy : mapy.clone();

    const int channels = src_c.channels();
    const size_t src_bytes =
            static_cast<size_t>(src_c.cols * src_c.rows * channels);
    const size_t dst_bytes =
            static_cast<size_t>(dst.cols * dst.rows * channels);
    const size_t map_bytes =
            static_cast<size_t>(mapx_c.total() * sizeof(float));

    cl_int err = CL_SUCCESS;
    cl_mem buf_src = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                    src_bytes, nullptr, &err);
    cl_mem buf_dst = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY,
                                    dst_bytes, nullptr, &err);
    cl_mem buf_mapx = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                     map_bytes, nullptr, &err);
    cl_mem buf_mapy = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                     map_bytes, nullptr, &err);
    if (!buf_src || !buf_dst || !buf_mapx || !buf_mapy) {
        if (buf_src) clReleaseMemObject(buf_src);
        if (buf_dst) clReleaseMemObject(buf_dst);
        if (buf_mapx) clReleaseMemObject(buf_mapx);
        if (buf_mapy) clReleaseMemObject(buf_mapy);
        return false;
    }

    bool ok = true;
    ok &= clEnqueueWriteBuffer(g_state.queue, buf_src, CL_TRUE, 0, src_bytes,
                               src_c.data, 0, nullptr, nullptr) == CL_SUCCESS;
    ok &= clEnqueueWriteBuffer(g_state.queue, buf_mapx, CL_TRUE, 0, map_bytes,
                               mapx_c.ptr<float>(), 0, nullptr,
                               nullptr) == CL_SUCCESS;
    ok &= clEnqueueWriteBuffer(g_state.queue, buf_mapy, CL_TRUE, 0, map_bytes,
                               mapy_c.ptr<float>(), 0, nullptr,
                               nullptr) == CL_SUCCESS;
    ok &= clEnqueueFillBuffer(g_state.queue, buf_dst, "\0", 1, 0, dst_bytes, 0,
                              nullptr, nullptr) == CL_SUCCESS;

    const int src_w = src_c.cols;
    const int src_h = src_c.rows;
    const int dst_w = dst.cols;
    const int dst_h = dst.rows;

    ok &= clSetKernelArg(g_state.kernel, 0, sizeof(cl_mem), &buf_src) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.kernel, 1, sizeof(cl_mem), &buf_mapx) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.kernel, 2, sizeof(cl_mem), &buf_mapy) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.kernel, 3, sizeof(cl_mem), &buf_dst) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.kernel, 4, sizeof(int), &src_w) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.kernel, 5, sizeof(int), &src_h) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.kernel, 6, sizeof(int), &dst_w) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.kernel, 7, sizeof(int), &dst_h) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.kernel, 8, sizeof(int), &channels) ==
          CL_SUCCESS;

    const size_t global[2] = {static_cast<size_t>(dst_w),
                              static_cast<size_t>(dst_h)};
    const size_t local[2] = {static_cast<size_t>(kBlock),
                             static_cast<size_t>(kBlock)};
    ok &= clEnqueueNDRangeKernel(g_state.queue, g_state.kernel, 2, nullptr,
                                 global, local, 0, nullptr,
                                 nullptr) == CL_SUCCESS;
    ok &= clEnqueueReadBuffer(g_state.queue, buf_dst, CL_TRUE, 0, dst_bytes,
                              dst.data, 0, nullptr, nullptr) == CL_SUCCESS;

    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_dst);
    clReleaseMemObject(buf_mapx);
    clReleaseMemObject(buf_mapy);
    return ok;
}

void releaseFusionBuffers() {
    for (auto& [name, buf] : g_fusion.weight_bufs) {
        if (buf) clReleaseMemObject(buf);
    }
    g_fusion.weight_bufs.clear();
    g_fusion.camera_order.clear();
    g_fusion.ready = false;
}

bool initAlphaFusion(cv::Size bev_size,
                     const std::vector<std::string>& camera_order,
                     const std::map<std::string, cv::Mat>& weights) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!initOnce()) return false;

    releaseFusionBuffers();
    g_fusion.size = bev_size;
    g_fusion.camera_order = camera_order;

    const size_t map_bytes = static_cast<size_t>(
            bev_size.width * bev_size.height * sizeof(float));

    for (const auto& name : camera_order) {
        auto it = weights.find(name);
        if (it == weights.end() || it->second.empty()) return false;

        cv::Mat w = it->second.isContinuous() ? it->second : it->second.clone();
        cl_int err = CL_SUCCESS;
        cl_mem buf = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                    map_bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf) return false;

        if (clEnqueueWriteBuffer(g_state.queue, buf, CL_TRUE, 0, map_bytes,
                                 w.ptr<float>(), 0, nullptr,
                                 nullptr) != CL_SUCCESS) {
            clReleaseMemObject(buf);
            releaseFusionBuffers();
            return false;
        }
        g_fusion.weight_bufs[name] = buf;
    }

    g_fusion.ready = true;
    return true;
}

void updateAlphaFusionWeights(const std::map<std::string, cv::Mat>& weights) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_fusion.ready) return;

    const size_t map_bytes = static_cast<size_t>(
            g_fusion.size.width * g_fusion.size.height * sizeof(float));

    for (const auto& name : g_fusion.camera_order) {
        auto buf_it = g_fusion.weight_bufs.find(name);
        auto w_it = weights.find(name);
        if (buf_it == g_fusion.weight_bufs.end() || w_it == weights.end())
            continue;

        cv::Mat w = w_it->second.isContinuous() ? w_it->second
                                                : w_it->second.clone();
        clEnqueueWriteBuffer(g_state.queue, buf_it->second, CL_TRUE, 0,
                             map_bytes, w.ptr<float>(), 0, nullptr, nullptr);
    }
}

void releaseAlphaFusion() {
    std::lock_guard<std::mutex> lock(g_mutex);
    releaseFusionBuffers();
}

bool alphaFusion(const std::vector<std::string>& camera_order,
                 const std::map<std::string, cv::Mat>& warped_images,
                 cv::Mat& output) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!initOnce() || !g_fusion.ready || camera_order.empty()) return false;

    if (output.empty() || output.size() != g_fusion.size ||
        output.type() != CV_8UC3) {
        output.create(g_fusion.size, CV_8UC3);
    }

    const int width = g_fusion.size.width;
    const int height = g_fusion.size.height;
    const int channels = 3;
    const int num_cameras = static_cast<int>(camera_order.size());
    const size_t img_bytes = static_cast<size_t>(width * height * channels);

    std::vector<cl_mem> src_bufs(6, nullptr);
    std::vector<cl_mem> weight_bufs(6, nullptr);
    std::vector<cv::Mat> src_mats(6);

    cl_int err = CL_SUCCESS;
    cl_mem zero_weight_buf = clCreateBuffer(
            g_state.context, CL_MEM_READ_ONLY,
            static_cast<size_t>(width * height) * sizeof(float), nullptr, &err);
    cl_mem zero_src_buf = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                         img_bytes, nullptr, &err);
    if (!zero_weight_buf || !zero_src_buf) {
        if (zero_weight_buf) clReleaseMemObject(zero_weight_buf);
        if (zero_src_buf) clReleaseMemObject(zero_src_buf);
        return false;
    }
    const std::vector<float> zero_weight(static_cast<size_t>(width * height),
                                         0.f);
    const std::vector<unsigned char> zero_src(img_bytes, 0);
    bool ok = clEnqueueWriteBuffer(g_state.queue, zero_weight_buf, CL_TRUE, 0,
                                   zero_weight.size() * sizeof(float),
                                   zero_weight.data(), 0, nullptr,
                                   nullptr) == CL_SUCCESS;
    ok &= clEnqueueWriteBuffer(g_state.queue, zero_src_buf, CL_TRUE, 0,
                               img_bytes, zero_src.data(), 0, nullptr,
                               nullptr) == CL_SUCCESS;
    if (!ok) {
        clReleaseMemObject(zero_weight_buf);
        clReleaseMemObject(zero_src_buf);
        return false;
    }

    for (int i = 0; i < 6; ++i) {
        if (i < num_cameras) {
            const auto& name = camera_order[static_cast<size_t>(i)];
            auto img_it = warped_images.find(name);
            auto w_it = g_fusion.weight_bufs.find(name);
            if (img_it == warped_images.end() || img_it->second.empty() ||
                w_it == g_fusion.weight_bufs.end()) {
                clReleaseMemObject(zero_weight_buf);
                clReleaseMemObject(zero_src_buf);
                return false;
            }

            src_mats[static_cast<size_t>(i)] = img_it->second.isContinuous()
                                                       ? img_it->second
                                                       : img_it->second.clone();
            if (src_mats[static_cast<size_t>(i)].size() != g_fusion.size ||
                src_mats[static_cast<size_t>(i)].type() != CV_8UC3) {
                clReleaseMemObject(zero_weight_buf);
                clReleaseMemObject(zero_src_buf);
                return false;
            }

            src_bufs[static_cast<size_t>(i)] =
                    clCreateBuffer(g_state.context, CL_MEM_READ_ONLY, img_bytes,
                                   nullptr, &err);
            if (err != CL_SUCCESS || !src_bufs[static_cast<size_t>(i)]) {
                clReleaseMemObject(zero_weight_buf);
                clReleaseMemObject(zero_src_buf);
                return false;
            }
            weight_bufs[static_cast<size_t>(i)] = w_it->second;
        } else {
            src_bufs[static_cast<size_t>(i)] = zero_src_buf;
            weight_bufs[static_cast<size_t>(i)] = zero_weight_buf;
        }
    }

    err = CL_SUCCESS;
    cl_mem dst_buf = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY,
                                    img_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !dst_buf) {
        for (int i = 0; i < num_cameras && i < 6; ++i) {
            if (src_bufs[static_cast<size_t>(i)] &&
                src_bufs[static_cast<size_t>(i)] != zero_src_buf) {
                clReleaseMemObject(src_bufs[static_cast<size_t>(i)]);
            }
        }
        clReleaseMemObject(zero_weight_buf);
        clReleaseMemObject(zero_src_buf);
        return false;
    }

    for (int i = 0; i < num_cameras && i < 6; ++i) {
        ok &= clEnqueueWriteBuffer(
                      g_state.queue, src_bufs[static_cast<size_t>(i)], CL_TRUE,
                      0, img_bytes, src_mats[static_cast<size_t>(i)].data, 0,
                      nullptr, nullptr) == CL_SUCCESS;
    }

    int arg = 0;
    ok &= clSetKernelArg(g_state.fusion_kernel, arg++, sizeof(int), &width) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.fusion_kernel, arg++, sizeof(int), &height) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.fusion_kernel, arg++, sizeof(int),
                         &num_cameras) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.fusion_kernel, arg++, sizeof(int),
                         &channels) == CL_SUCCESS;

    for (int i = 0; i < 6; ++i) {
        cl_mem src_arg = src_bufs[static_cast<size_t>(i)];
        cl_mem w_arg = weight_bufs[static_cast<size_t>(i)];
        ok &= clSetKernelArg(g_state.fusion_kernel, arg++, sizeof(cl_mem),
                             &src_arg) == CL_SUCCESS;
        ok &= clSetKernelArg(g_state.fusion_kernel, arg++, sizeof(cl_mem),
                             &w_arg) == CL_SUCCESS;
    }
    ok &= clSetKernelArg(g_state.fusion_kernel, arg++, sizeof(cl_mem),
                         &dst_buf) == CL_SUCCESS;

    const size_t global[2] = {static_cast<size_t>(width),
                              static_cast<size_t>(height)};
    const size_t local[2] = {static_cast<size_t>(kBlock),
                             static_cast<size_t>(kBlock)};
    ok &= clEnqueueNDRangeKernel(g_state.queue, g_state.fusion_kernel, 2,
                                 nullptr, global, local, 0, nullptr,
                                 nullptr) == CL_SUCCESS;
    ok &= clEnqueueReadBuffer(g_state.queue, dst_buf, CL_TRUE, 0, img_bytes,
                              output.data, 0, nullptr, nullptr) == CL_SUCCESS;

    for (auto buf : src_bufs) {
        if (buf && buf != zero_src_buf) clReleaseMemObject(buf);
    }
    clReleaseMemObject(zero_weight_buf);
    clReleaseMemObject(zero_src_buf);
    clReleaseMemObject(dst_buf);
    return ok;
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
    std::lock_guard<std::mutex> lock(g_mutex);
    image_points.clear();
    depths.clear();
    if (!points_xyz || num_points <= 0 || !initOnce()) return false;

    const size_t pts_bytes =
            static_cast<size_t>(num_points) * 3 * sizeof(float);
    const size_t uv_bytes = static_cast<size_t>(num_points) * 2 * sizeof(float);
    const size_t depth_bytes = static_cast<size_t>(num_points) * sizeof(float);
    const size_t valid_bytes =
            static_cast<size_t>(num_points) * sizeof(unsigned char);

    cl_int err = CL_SUCCESS;
    cl_mem buf_pts = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                    pts_bytes, nullptr, &err);
    cl_mem buf_rot = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                    9 * sizeof(float), nullptr, &err);
    cl_mem buf_trans = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                      3 * sizeof(float), nullptr, &err);
    cl_mem buf_uv = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY, uv_bytes,
                                   nullptr, &err);
    cl_mem buf_depth = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY,
                                      depth_bytes, nullptr, &err);
    cl_mem buf_valid = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY,
                                      valid_bytes, nullptr, &err);
    if (!buf_pts || !buf_rot || !buf_trans || !buf_uv || !buf_depth ||
        !buf_valid) {
        if (buf_pts) clReleaseMemObject(buf_pts);
        if (buf_rot) clReleaseMemObject(buf_rot);
        if (buf_trans) clReleaseMemObject(buf_trans);
        if (buf_uv) clReleaseMemObject(buf_uv);
        if (buf_depth) clReleaseMemObject(buf_depth);
        if (buf_valid) clReleaseMemObject(buf_valid);
        return false;
    }

    bool ok =
            clEnqueueWriteBuffer(g_state.queue, buf_pts, CL_TRUE, 0, pts_bytes,
                                 points_xyz, 0, nullptr, nullptr) == CL_SUCCESS;
    ok &= clEnqueueWriteBuffer(g_state.queue, buf_rot, CL_TRUE, 0,
                               9 * sizeof(float), rotation, 0, nullptr,
                               nullptr) == CL_SUCCESS;
    ok &= clEnqueueWriteBuffer(g_state.queue, buf_trans, CL_TRUE, 0,
                               3 * sizeof(float), translation, 0, nullptr,
                               nullptr) == CL_SUCCESS;

    int arg = 0;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(cl_mem),
                         &buf_pts) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(cl_mem),
                         &buf_rot) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(cl_mem),
                         &buf_trans) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(int),
                         &num_points) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(float), &fx) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(float), &fy) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(float), &cx) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(float), &cy) ==
          CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(cl_mem),
                         &buf_uv) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(cl_mem),
                         &buf_depth) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kernel, arg++, sizeof(cl_mem),
                         &buf_valid) == CL_SUCCESS;

    const size_t global = static_cast<size_t>(num_points);
    ok &= clEnqueueNDRangeKernel(g_state.queue, g_state.project_kernel, 1,
                                 nullptr, &global, nullptr, 0, nullptr,
                                 nullptr) == CL_SUCCESS;

    std::vector<float> host_uv(num_points * 2);
    std::vector<float> host_depth(num_points);
    std::vector<unsigned char> host_valid(num_points);
    ok &= clEnqueueReadBuffer(g_state.queue, buf_uv, CL_TRUE, 0, uv_bytes,
                              host_uv.data(), 0, nullptr,
                              nullptr) == CL_SUCCESS;
    ok &= clEnqueueReadBuffer(g_state.queue, buf_depth, CL_TRUE, 0, depth_bytes,
                              host_depth.data(), 0, nullptr,
                              nullptr) == CL_SUCCESS;
    ok &= clEnqueueReadBuffer(g_state.queue, buf_valid, CL_TRUE, 0, valid_bytes,
                              host_valid.data(), 0, nullptr,
                              nullptr) == CL_SUCCESS;

    clReleaseMemObject(buf_pts);
    clReleaseMemObject(buf_rot);
    clReleaseMemObject(buf_trans);
    clReleaseMemObject(buf_uv);
    clReleaseMemObject(buf_depth);
    clReleaseMemObject(buf_valid);

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
    std::lock_guard<std::mutex> lock(g_mutex);
    image_points.clear();
    depths.clear();
    if (!points_xyz || num_points <= 0 || !kb || !initOnce()) return false;

    const size_t pts_bytes =
            static_cast<size_t>(num_points) * 3 * sizeof(float);
    const size_t uv_bytes = static_cast<size_t>(num_points) * 2 * sizeof(float);
    const size_t depth_bytes = static_cast<size_t>(num_points) * sizeof(float);
    const size_t valid_bytes =
            static_cast<size_t>(num_points) * sizeof(unsigned char);

    cl_int err = CL_SUCCESS;
    cl_mem buf_pts = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                    pts_bytes, nullptr, &err);
    cl_mem buf_rot = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                    9 * sizeof(float), nullptr, &err);
    cl_mem buf_trans = clCreateBuffer(g_state.context, CL_MEM_READ_ONLY,
                                      3 * sizeof(float), nullptr, &err);
    cl_mem buf_uv = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY, uv_bytes,
                                   nullptr, &err);
    cl_mem buf_depth = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY,
                                      depth_bytes, nullptr, &err);
    cl_mem buf_valid = clCreateBuffer(g_state.context, CL_MEM_WRITE_ONLY,
                                      valid_bytes, nullptr, &err);
    if (!buf_pts || !buf_rot || !buf_trans || !buf_uv || !buf_depth ||
        !buf_valid) {
        if (buf_pts) clReleaseMemObject(buf_pts);
        if (buf_rot) clReleaseMemObject(buf_rot);
        if (buf_trans) clReleaseMemObject(buf_trans);
        if (buf_uv) clReleaseMemObject(buf_uv);
        if (buf_depth) clReleaseMemObject(buf_depth);
        if (buf_valid) clReleaseMemObject(buf_valid);
        return false;
    }

    bool ok =
            clEnqueueWriteBuffer(g_state.queue, buf_pts, CL_TRUE, 0, pts_bytes,
                                 points_xyz, 0, nullptr, nullptr) == CL_SUCCESS;
    ok &= clEnqueueWriteBuffer(g_state.queue, buf_rot, CL_TRUE, 0,
                               9 * sizeof(float), rotation, 0, nullptr,
                               nullptr) == CL_SUCCESS;
    ok &= clEnqueueWriteBuffer(g_state.queue, buf_trans, CL_TRUE, 0,
                               3 * sizeof(float), translation, 0, nullptr,
                               nullptr) == CL_SUCCESS;

    const float k1 = kb[0];
    const float k2 = kb[1];
    const float k3 = kb[2];
    const float k4 = kb[3];
    int arg = 0;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(cl_mem),
                         &buf_pts) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(cl_mem),
                         &buf_rot) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(cl_mem),
                         &buf_trans) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(int),
                         &num_points) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(float),
                         &fx) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(float),
                         &fy) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(float),
                         &cx) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(float),
                         &cy) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(float),
                         &k1) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(float),
                         &k2) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(float),
                         &k3) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(float),
                         &k4) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(cl_mem),
                         &buf_uv) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(cl_mem),
                         &buf_depth) == CL_SUCCESS;
    ok &= clSetKernelArg(g_state.project_kb_kernel, arg++, sizeof(cl_mem),
                         &buf_valid) == CL_SUCCESS;

    const size_t global = static_cast<size_t>(num_points);
    ok &= clEnqueueNDRangeKernel(g_state.queue, g_state.project_kb_kernel, 1,
                                 nullptr, &global, nullptr, 0, nullptr,
                                 nullptr) == CL_SUCCESS;

    std::vector<float> host_uv(num_points * 2);
    std::vector<float> host_depth(num_points);
    std::vector<unsigned char> host_valid(num_points);
    ok &= clEnqueueReadBuffer(g_state.queue, buf_uv, CL_TRUE, 0, uv_bytes,
                              host_uv.data(), 0, nullptr,
                              nullptr) == CL_SUCCESS;
    ok &= clEnqueueReadBuffer(g_state.queue, buf_depth, CL_TRUE, 0, depth_bytes,
                              host_depth.data(), 0, nullptr,
                              nullptr) == CL_SUCCESS;
    ok &= clEnqueueReadBuffer(g_state.queue, buf_valid, CL_TRUE, 0, valid_bytes,
                              host_valid.data(), 0, nullptr,
                              nullptr) == CL_SUCCESS;

    clReleaseMemObject(buf_pts);
    clReleaseMemObject(buf_rot);
    clReleaseMemObject(buf_trans);
    clReleaseMemObject(buf_uv);
    clReleaseMemObject(buf_depth);
    clReleaseMemObject(buf_valid);

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

}  // namespace bev_ocl
}  // namespace mcalib
