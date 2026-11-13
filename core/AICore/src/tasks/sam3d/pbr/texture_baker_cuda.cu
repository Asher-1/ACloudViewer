// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <csrc/common/interpolate.h>
#include <csrc/common/rasterize.h>
#include <csrc/common/texture.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <CudaRaster.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "texture_baker.hpp"
#include "texture_inpaint.hpp"

// The symbols below are the standalone CUDA entry points compiled from the
// pinned nvdiffrast core in CMake. They are the same kernels called through
// nvdiffrast's PyTorch binding, but no Python/Torch objects cross this
// boundary.
void RasterizeCudaFwdShaderKernel(RasterizeCudaFwdShaderParams p);
void InterpolateFwdKernelDa(InterpolateKernelParams p);
void MipBuildKernel1(TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinear1(TextureKernelParams p);
void TextureGradKernelLinearMipmapLinear(TextureKernelParams p);
void MipGradKernel1(TextureKernelParams p);

namespace sam3d {
namespace {

constexpr float kAdamBeta1 = 0.5f;
constexpr float kAdamBeta2 = 0.9f;
constexpr float kAdamEpsilon = 1.0e-8f;
constexpr float kPi = 3.14159265358979323846f;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) : count_(count) {
        if (count_ != 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count_ * sizeof(T)),
                       "cudaMalloc native texture bake buffer");
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    ~DeviceArray() { cudaFree(data_); }

    T* data() const { return data_; }
    size_t size() const { return count_; }

    void upload(const T* source, size_t count, const char* operation) {
        if (count != count_)
            throw std::runtime_error(std::string(operation) + " size mismatch");
        if (count != 0) {
            check_cuda(cudaMemcpy(data_, source, count * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       operation);
        }
    }

    void download(T* destination, size_t count, const char* operation) const {
        if (count != count_)
            throw std::runtime_error(std::string(operation) + " size mismatch");
        if (count != 0) {
            check_cuda(cudaMemcpy(destination, data_, count * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       operation);
        }
    }

    void zero(const char* operation) {
        if (count_ != 0)
            check_cuda(cudaMemset(data_, 0, count_ * sizeof(T)), operation);
    }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

class CublasHandle {
public:
    CublasHandle() {
        check_cublas(cublasCreate(&handle_), "cublasCreate texture bake");
        check_cublas(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH),
                     "cublasSetMathMode texture bake");
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    ~CublasHandle() {
        if (handle_ != nullptr) cublasDestroy(handle_);
    }

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

dim3 block_8x8() { return dim3(8, 8, 1); }

dim3 grid_8x8(int width, int height, int depth = 1) {
    return dim3(static_cast<unsigned>((width + 7) / 8),
                static_cast<unsigned>((height + 7) / 8),
                static_cast<unsigned>(depth));
}

__global__ void l1_gradient_masked(const float* rendered,
                                   const float* observation,
                                   float* gradient,
                                   int pixel_count,
                                   int valid_pixels) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const float* target = observation + static_cast<size_t>(pixel) * 3;
    float* out = gradient + static_cast<size_t>(pixel) * 3;
    const bool valid =
            target[0] != 0.0f || target[1] != 0.0f || target[2] != 0.0f;
    if (!valid) {
        out[0] = 0.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
        return;
    }
    const float scale = 1.0f / static_cast<float>(valid_pixels * 3);
    const float* source = rendered + static_cast<size_t>(pixel) * 3;
    for (int channel = 0; channel < 3; ++channel) {
        const float delta = source[channel] - target[channel];
        out[channel] = delta > 0.0f ? scale : (delta < 0.0f ? -scale : 0.0f);
    }
}

__global__ void add_total_variation_gradient(const float* texture,
                                             float* gradient,
                                             int width,
                                             int height,
                                             float weight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int pixel = x + width * y;
    for (int channel = 0; channel < 3; ++channel) {
        const int index = pixel * 3 + channel;
        if (y + 1 < height) {
            const float delta = texture[index] - texture[index + width * 3];
            const float value =
                    delta > 0.0f
                            ? weight / static_cast<float>((height - 1) * width *
                                                          3)
                            : (delta < 0.0f ? -weight / static_cast<float>(
                                                                (height - 1) *
                                                                width * 3)
                                            : 0.0f);
            atomicAdd(gradient + index, value);
            atomicAdd(gradient + index + width * 3, -value);
        }
        if (x + 1 < width) {
            const float delta = texture[index] - texture[index + 3];
            const float value =
                    delta > 0.0f
                            ? weight / static_cast<float>(height * (width - 1) *
                                                          3)
                            : (delta < 0.0f ? -weight / static_cast<float>(
                                                                height *
                                                                (width - 1) * 3)
                                            : 0.0f);
            atomicAdd(gradient + index, value);
            atomicAdd(gradient + index + 3, -value);
        }
    }
}

__global__ void adam_step(float* texture,
                          const float* gradient,
                          float* first_moment,
                          float* second_moment,
                          size_t count,
                          float learning_rate,
                          float bias_correction1,
                          float bias_correction2_sqrt) {
    const size_t index =
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float grad = gradient[index];
    const float moment1 = first_moment[index] =
            kAdamBeta1 * first_moment[index] + (1.0f - kAdamBeta1) * grad;
    const float moment2 = second_moment[index] =
            kAdamBeta2 * second_moment[index] +
            (1.0f - kAdamBeta2) * grad * grad;
    const float step_size = learning_rate / bias_correction1;
    const float denominator =
            sqrtf(moment2) / bias_correction2_sqrt + kAdamEpsilon;
    texture[index] -= step_size * moment1 / denominator;
}

__global__ void coverage_from_raster(const float* raster,
                                     uint8_t* coverage,
                                     int pixel_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;
    coverage[index] = raster[index * 4 + 3] != 0.0f ? 1 : 0;
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

bool valid_mesh_for_bake(const NativeMesh& mesh, std::string& error) {
    if (mesh.positions.empty() || mesh.positions.size() % 3 != 0 ||
        mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        error = "optimized texture bake requires a non-empty indexed triangle "
                "mesh";
        return false;
    }
    const size_t vertices = mesh.positions.size() / 3;
    if (mesh.texcoords.size() != vertices * 2) {
        error = "optimized texture bake requires one xatlas UV pair per vertex";
        return false;
    }
    for (uint32_t index : mesh.indices) {
        if (index >= vertices ||
            index > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            error = "optimized texture bake received an out-of-range triangle "
                    "index";
            return false;
        }
    }
    return true;
}

std::array<float, 16> to_column_major(const std::array<float, 16>& row_major) {
    std::array<float, 16> column_major{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            column_major[static_cast<size_t>(column) * 4 + row] =
                    row_major[static_cast<size_t>(row) * 4 + column];
        }
    }
    return column_major;
}

void make_view_projection(const GaussianCamera& camera,
                          const TextureBakeConfig& config,
                          std::array<float, 16>& view,
                          std::array<float, 16>& projection) {
    // Exact utils3d.torch.extrinsics_to_view / intrinsics_to_perspective
    // convention.
    view = {camera.extrinsics[0],
            camera.extrinsics[1],
            camera.extrinsics[2],
            camera.extrinsics[3],
            -camera.extrinsics[4],
            -camera.extrinsics[5],
            -camera.extrinsics[6],
            -camera.extrinsics[7],
            -camera.extrinsics[8],
            -camera.extrinsics[9],
            -camera.extrinsics[10],
            -camera.extrinsics[11],
            0.0f,
            0.0f,
            0.0f,
            1.0f};
    const float fx = camera.intrinsics[0];
    const float fy = camera.intrinsics[4];
    const float cx = camera.intrinsics[2];
    const float cy = camera.intrinsics[5];
    const float near_plane = config.near_plane;
    const float far_plane = config.far_plane;
    projection = {2.0f * fx,
                  0.0f,
                  1.0f - 2.0f * cx,
                  0.0f,
                  0.0f,
                  2.0f * fy,
                  2.0f * cy - 1.0f,
                  0.0f,
                  0.0f,
                  0.0f,
                  (near_plane + far_plane) / (near_plane - far_plane),
                  2.0f * near_plane * far_plane / (near_plane - far_plane),
                  0.0f,
                  0.0f,
                  -1.0f,
                  0.0f};
}

class NumpyRandomState {
public:
    explicit NumpyRandomState(uint32_t seed) {
        state_[0] = seed;
        for (uint32_t index = 1; index < state_.size(); ++index) {
            const uint32_t previous = state_[index - 1];
            state_[index] = 1812433253u * (previous ^ (previous >> 30)) + index;
        }
    }

    int randint(int high) {
        uint32_t mask = static_cast<uint32_t>(high - 1);
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        uint32_t value = 0;
        do {
            value = next_u32() & mask;
        } while (value >= static_cast<uint32_t>(high));
        return static_cast<int>(value);
    }

private:
    uint32_t next_u32() {
        if (index_ >= state_.size()) twist();
        uint32_t value = state_[index_++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680u;
        value ^= (value << 15) & 0xefc60000u;
        return value ^ (value >> 18);
    }

    void twist() {
        for (uint32_t index = 0; index < state_.size(); ++index) {
            const uint32_t bits =
                    (state_[index] & 0x80000000u) |
                    (state_[(index + 1) % state_.size()] & 0x7fffffffu);
            state_[index] = state_[(index + 397) % state_.size()] ^
                            (bits >> 1) ^ ((bits & 1u) == 0 ? 0u : 0x9908b0dfu);
        }
        index_ = 0;
    }

    std::array<uint32_t, 624> state_{};
    uint32_t index_ = 624;
};

void launch(void* function,
            dim3 grid,
            dim3 block,
            void** arguments,
            size_t shared_bytes,
            const char* operation) {
    check_cuda(cudaLaunchKernel(function, grid, block, arguments, shared_bytes,
                                nullptr),
               operation);
}

void rasterize_shader(CR::CudaRaster& rasterizer,
                      const float4* positions,
                      const int* indices,
                      int vertices,
                      int triangles,
                      int width,
                      int height,
                      float* raster,
                      float* raster_derivatives) {
    RasterizeCudaFwdShaderParams parameters{};
    parameters.pos = reinterpret_cast<const float*>(positions);
    parameters.tri = indices;
    parameters.in_idx = static_cast<const int*>(rasterizer.getColorBuffer());
    parameters.out = raster;
    parameters.out_db = raster_derivatives;
    parameters.numTriangles = triangles;
    parameters.numVertices = vertices;
    parameters.width_in = (width + 7) & ~7;
    parameters.height_in = (height + 7) & ~7;
    parameters.width_out = width;
    parameters.height_out = height;
    parameters.depth = 1;
    parameters.instance_mode = 0;
    parameters.xs = 2.0f / static_cast<float>(width);
    parameters.xo = 1.0f / static_cast<float>(width) - 1.0f;
    parameters.ys = 2.0f / static_cast<float>(height);
    parameters.yo = 1.0f / static_cast<float>(height) - 1.0f;
    void* arguments[] = {&parameters};
    launch(reinterpret_cast<void*>(RasterizeCudaFwdShaderKernel),
           grid_8x8(width, height), block_8x8(), arguments, 0,
           "nvdiffrast RasterizeCudaFwdShaderKernel");
}

void interpolate_uv(const int* indices,
                    const float* texcoords,
                    int vertices,
                    int triangles,
                    int width,
                    int height,
                    const float* raster,
                    const float* raster_derivatives,
                    float* uv,
                    float* uv_derivatives) {
    InterpolateKernelParams parameters{};
    parameters.tri = indices;
    parameters.attr = texcoords;
    parameters.rast = raster;
    parameters.rastDB = raster_derivatives;
    parameters.out = uv;
    parameters.outDA = uv_derivatives;
    parameters.numTriangles = triangles;
    parameters.numVertices = vertices;
    parameters.numAttr = 2;
    parameters.numDiffAttr = 2;
    parameters.width = width;
    parameters.height = height;
    parameters.depth = 1;
    parameters.attrBC = 0;
    parameters.instance_mode = 0;
    parameters.diff_attrs_all = 1;
    void* arguments[] = {&parameters};
    launch(reinterpret_cast<void*>(InterpolateFwdKernelDa),
           grid_8x8(width, height), block_8x8(), arguments, 0,
           "nvdiffrast InterpolateFwdKernelDa");
}

void configure_texture_params(TextureKernelParams& parameters,
                              float* texture,
                              float* mip,
                              float* texture_gradient,
                              float* mip_gradient,
                              int texture_size,
                              int image_width,
                              int image_height,
                              const float* uv,
                              const float* uv_derivatives,
                              float* rendered,
                              float* rendered_gradient,
                              float* uv_gradient,
                              float* uv_derivative_gradient) {
    parameters = {};
    parameters.tex[0] = texture;
    parameters.gradTex[0] = texture_gradient;
    parameters.uv = uv;
    parameters.uvDA = uv_derivatives;
    parameters.out = rendered;
    parameters.dy = rendered_gradient;
    parameters.gradUV = uv_gradient;
    parameters.gradUVDA = uv_derivative_gradient;
    parameters.enableMip = 1;
    parameters.filterMode = TEX_MODE_LINEAR_MIPMAP_LINEAR;
    parameters.boundaryMode = TEX_BOUNDARY_MODE_WRAP;
    parameters.mipLevelLimit = -1;
    parameters.channels = 3;
    parameters.imgWidth = image_width;
    parameters.imgHeight = image_height;
    parameters.texWidth = texture_size;
    parameters.texHeight = texture_size;
    parameters.texDepth = 1;
    parameters.n = 1;
    int offsets[TEX_MAX_MIP_LEVEL]{};
    (void)calculateMipInfo(parameters, offsets);
    for (int level = 1; level <= parameters.mipLevelMax; ++level) {
        parameters.tex[level] = mip + offsets[level];
        parameters.gradTex[level] = mip_gradient + offsets[level];
    }
}

void build_mipmaps(TextureKernelParams parameters) {
    void* arguments[] = {&parameters};
    for (int level = 1; level <= parameters.mipLevelMax; ++level) {
        parameters.mipLevelOut = level;
        const int level_width = std::max(parameters.texWidth >> level, 1);
        const int level_height = std::max(parameters.texHeight >> level, 1);
        launch(reinterpret_cast<void*>(MipBuildKernel1),
               grid_8x8(level_width, level_height), block_8x8(), arguments, 0,
               "nvdiffrast MipBuildKernel1");
    }
}

void texture_gradient(TextureKernelParams parameters) {
    void* arguments[] = {&parameters};
    launch(reinterpret_cast<void*>(TextureGradKernelLinearMipmapLinear),
           grid_8x8(parameters.imgWidth, parameters.imgHeight), block_8x8(),
           arguments, 0, "nvdiffrast TextureGradKernelLinearMipmapLinear");
    const size_t shared_bytes =
            static_cast<size_t>(8 * 8 * parameters.channels) * sizeof(float);
    launch(reinterpret_cast<void*>(MipGradKernel1),
           grid_8x8(parameters.texWidth, parameters.texHeight), block_8x8(),
           arguments, shared_bytes, "nvdiffrast MipGradKernel1");
}

}  // namespace

float texture_bake_learning_rate_for_update(const TextureBakeConfig& config,
                                            int completed_updates) {
    if (completed_updates <= 0 || config.steps <= 0) {
        return config.learning_rate;
    }

    // This is intentionally evaluated in double precision, as NumPy computes
    // the official scheduler scalar in float64 before PyTorch consumes it.
    const int schedule_step = std::min(completed_updates - 1, config.steps - 1);
    const double progress = static_cast<double>(schedule_step) /
                            static_cast<double>(config.steps);
    const double rate =
            static_cast<double>(config.eta_min) +
            (static_cast<double>(config.learning_rate) -
             static_cast<double>(config.eta_min)) *
                    0.5 * (1.0 + std::cos(static_cast<double>(kPi) * progress));
    return static_cast<float>(rate);
}

bool rasterize_texture_bake_view_official_cuda(const NativeMesh& mesh,
                                               const GaussianCamera& camera,
                                               const TextureBakeConfig& config,
                                               int width,
                                               int height,
                                               TextureBakeRaster& result,
                                               std::string& error) {
    if (!valid_mesh_for_bake(mesh, error)) return false;
    if (width <= 0 || height <= 0 || !(config.near_plane > 0.0f) ||
        !(config.far_plane > config.near_plane)) {
        error = "invalid texture-bake raster dimensions or clip range";
        return false;
    }
    const size_t vertices = mesh.positions.size() / 3;
    const size_t triangles = mesh.indices.size() / 3;
    const size_t pixels = static_cast<size_t>(width) * height;
    try {
        std::vector<int> host_indices(mesh.indices.begin(), mesh.indices.end());
        std::vector<float4> host_positions(vertices);
        for (size_t index = 0; index < vertices; ++index) {
            host_positions[index] = make_float4(
                    mesh.positions[index * 3], mesh.positions[index * 3 + 1],
                    mesh.positions[index * 3 + 2], 1.0f);
        }
        DeviceArray<int> device_indices(host_indices.size());
        DeviceArray<float> device_texcoords(mesh.texcoords.size());
        DeviceArray<float4> device_positions(vertices);
        DeviceArray<float4> device_clip_positions(vertices);
        DeviceArray<float> device_projection(16);
        DeviceArray<float> device_view(16);
        DeviceArray<float> device_mvp(16);
        DeviceArray<float> device_raster(pixels * 4);
        DeviceArray<float> device_raster_derivatives(pixels * 4);
        DeviceArray<float> device_uv(pixels * 2);
        DeviceArray<float> device_uv_derivatives(pixels * 4);
        DeviceArray<uint8_t> device_coverage(pixels);
        device_indices.upload(host_indices.data(), host_indices.size(),
                              "cudaMemcpy texture raster indices");
        device_texcoords.upload(mesh.texcoords.data(), mesh.texcoords.size(),
                                "cudaMemcpy texture raster UVs");
        device_positions.upload(host_positions.data(), host_positions.size(),
                                "cudaMemcpy texture raster positions");
        std::array<float, 16> view{};
        std::array<float, 16> projection{};
        make_view_projection(camera, config, view, projection);
        const std::array<float, 16> view_column_major = to_column_major(view);
        const std::array<float, 16> projection_column_major =
                to_column_major(projection);
        device_view.upload(view_column_major.data(), view_column_major.size(),
                           "cudaMemcpy texture raster view");
        device_projection.upload(projection_column_major.data(),
                                 projection_column_major.size(),
                                 "cudaMemcpy texture raster projection");
        CublasHandle cublas;
        constexpr float one = 1.0f;
        constexpr float zero = 0.0f;
        check_cublas(
                cublasSgemm(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4,
                            &one, device_projection.data(), 4,
                            device_view.data(), 4, &zero, device_mvp.data(), 4),
                "cublasSgemm texture raster projection-view");
        check_cublas(
                cublasSgemm(
                        cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, 4,
                        static_cast<int>(vertices), 4, &one, device_mvp.data(),
                        4,
                        reinterpret_cast<const float*>(device_positions.data()),
                        4, &zero,
                        reinterpret_cast<float*>(device_clip_positions.data()),
                        4),
                "cublasSgemm texture raster clip positions");
        CR::CudaRaster rasterizer;
        rasterizer.setBufferSize(width, height, 1);
        rasterizer.setViewport(width, height, 0, 0);
        rasterizer.setRenderModeFlags(0);
        rasterizer.setIndexBuffer(device_indices.data(),
                                  static_cast<int>(triangles));
        rasterizer.setVertexBuffer(device_clip_positions.data(),
                                   static_cast<int>(vertices));
        rasterizer.deferredClear(0);
        if (!rasterizer.drawTriangles(nullptr, false, nullptr)) {
            error = "nvdiffrast CUDA rasterizer exceeded its internal "
                    "subtriangle capacity";
            return false;
        }
        rasterize_shader(
                rasterizer, device_clip_positions.data(), device_indices.data(),
                static_cast<int>(vertices), static_cast<int>(triangles), width,
                height, device_raster.data(), device_raster_derivatives.data());
        interpolate_uv(device_indices.data(), device_texcoords.data(),
                       static_cast<int>(vertices), static_cast<int>(triangles),
                       width, height, device_raster.data(),
                       device_raster_derivatives.data(), device_uv.data(),
                       device_uv_derivatives.data());
        const int threads = 256;
        const int blocks = static_cast<int>((pixels + threads - 1) / threads);
        coverage_from_raster<<<blocks, threads>>>(device_raster.data(),
                                                  device_coverage.data(),
                                                  static_cast<int>(pixels));
        check_cuda(cudaGetLastError(), "native texture raster coverage");
        check_cuda(cudaDeviceSynchronize(),
                   "native texture raster synchronization");
        result.width = width;
        result.height = height;
        result.uv.resize(pixels * 2);
        result.uv_derivatives.resize(pixels * 4);
        result.coverage.resize(pixels);
        device_uv.download(result.uv.data(), result.uv.size(),
                           "cudaMemcpy texture raster UV");
        device_uv_derivatives.download(
                result.uv_derivatives.data(), result.uv_derivatives.size(),
                "cudaMemcpy texture raster UV derivatives");
        device_coverage.download(result.coverage.data(), result.coverage.size(),
                                 "cudaMemcpy texture raster coverage");
        // CudaRaster coverage is bottom-origin, whereas the official utility's
        // exported mask is top-origin. UV and UV derivatives are already in the
        // latter convention after interpolation; keep the diagnostic mask in
        // that same public coordinate system.
        for (int y = 0; y < height / 2; ++y) {
            const size_t top = static_cast<size_t>(y) * width;
            const size_t bottom = static_cast<size_t>(height - 1 - y) * width;
            for (int x = 0; x < width; ++x) {
                std::swap(result.coverage[top + x],
                          result.coverage[bottom + x]);
            }
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool bake_texture_official_cuda(const NativeMesh& mesh,
                                const std::vector<RgbaImage>& observations,
                                const std::vector<GaussianCamera>& cameras,
                                const TextureBakeConfig& config,
                                RgbaImage& texture,
                                std::vector<uint8_t>* hole_mask,
                                std::vector<float>* optimized_texture,
                                std::string& error) {
    if (!valid_mesh_for_bake(mesh, error)) return false;
    if (observations.empty() || observations.size() != cameras.size() ||
        config.steps <= 0 || !is_power_of_two(config.texture_size) ||
        config.texture_size > (1 << TEX_MAX_MIP_LEVEL) ||
        !(config.learning_rate > 0.0f) || !(config.eta_min >= 0.0f) ||
        !(config.tv_weight >= 0.0f) || !(config.near_plane > 0.0f) ||
        !(config.far_plane > config.near_plane)) {
        error = "invalid optimized texture-bake configuration or "
                "observation/camera list";
        return false;
    }
    const int image_width = observations.front().width;
    const int image_height = observations.front().height;
    if (image_width <= 0 || image_height <= 0) {
        error = "optimized texture bake requires non-empty observations";
        return false;
    }
    const size_t image_pixels = static_cast<size_t>(image_width) * image_height;
    const size_t texture_pixels =
            static_cast<size_t>(config.texture_size) * config.texture_size;
    for (const RgbaImage& observation : observations) {
        if (observation.width != image_width ||
            observation.height != image_height ||
            observation.rgba.size() != image_pixels * 4) {
            error = "optimized texture bake observations must have a common "
                    "RGBA size";
            return false;
        }
    }
    const size_t vertices = mesh.positions.size() / 3;
    const size_t triangles = mesh.indices.size() / 3;
    if (vertices > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        triangles > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        image_pixels > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        texture_pixels > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "optimized texture bake input exceeds CUDA kernel index limits";
        return false;
    }
    try {
        std::vector<int> host_indices(mesh.indices.begin(), mesh.indices.end());
        std::vector<float4> host_positions(vertices);
        std::vector<float4> host_uv_positions(vertices);
        for (size_t index = 0; index < vertices; ++index) {
            host_positions[index] = make_float4(
                    mesh.positions[index * 3], mesh.positions[index * 3 + 1],
                    mesh.positions[index * 3 + 2], 1.0f);
            host_uv_positions[index] = make_float4(
                    2.0f * mesh.texcoords[index * 2] - 1.0f,
                    2.0f * mesh.texcoords[index * 2 + 1] - 1.0f, 0.0f, 1.0f);
        }

        std::vector<float> host_observations(observations.size() *
                                             image_pixels * 3);
        std::vector<int> valid_pixels(observations.size(), 0);
        for (size_t view = 0; view < observations.size(); ++view) {
            const RgbaImage& image = observations[view];
            float* destination =
                    host_observations.data() + view * image_pixels * 3;
            for (int y = 0; y < image_height; ++y) {
                const int source_y = image_height - 1 - y;
                for (int x = 0; x < image_width; ++x) {
                    const uint8_t* source =
                            image.rgba.data() +
                            (static_cast<size_t>(source_y) * image_width + x) *
                                    4;
                    float* pixel =
                            destination +
                            (static_cast<size_t>(y) * image_width + x) * 3;
                    pixel[0] = static_cast<float>(source[0]) / 255.0f;
                    pixel[1] = static_cast<float>(source[1]) / 255.0f;
                    pixel[2] = static_cast<float>(source[2]) / 255.0f;
                    valid_pixels[view] +=
                            source[0] != 0 || source[1] != 0 || source[2] != 0;
                }
            }
            if (valid_pixels[view] == 0) {
                error = "optimized texture bake received an all-black Gaussian "
                        "observation";
                return false;
            }
        }

        const size_t max_raster_pixels = std::max(image_pixels, texture_pixels);
        DeviceArray<int> device_indices(host_indices.size());
        DeviceArray<float> device_texcoords(mesh.texcoords.size());
        DeviceArray<float4> device_homogeneous_positions(vertices);
        DeviceArray<float4> device_clip_positions(vertices);
        DeviceArray<float4> device_uv_clip_positions(vertices);
        DeviceArray<float> device_projection(16);
        DeviceArray<float> device_view(16);
        DeviceArray<float> device_mvp(16);
        DeviceArray<float> device_raster(max_raster_pixels * 4);
        DeviceArray<float> device_raster_derivatives(max_raster_pixels * 4);
        DeviceArray<float> device_uv_maps(observations.size() * image_pixels *
                                          2);
        DeviceArray<float> device_uv_derivatives(observations.size() *
                                                 image_pixels * 4);
        DeviceArray<float> device_observations(host_observations.size());
        DeviceArray<float> device_rendered(image_pixels * 3);
        DeviceArray<float> device_rendered_gradient(image_pixels * 3);
        DeviceArray<float> device_uv_gradient(image_pixels * 2);
        DeviceArray<float> device_uv_derivative_gradient(image_pixels * 4);
        DeviceArray<uint8_t> device_coverage(texture_pixels);
        DeviceArray<float> device_texture(texture_pixels * 3);
        DeviceArray<float> device_texture_gradient(texture_pixels * 3);
        DeviceArray<float> device_first_moment(texture_pixels * 3);
        DeviceArray<float> device_second_moment(texture_pixels * 3);

        TextureKernelParams mip_info{};
        mip_info.texWidth = config.texture_size;
        mip_info.texHeight = config.texture_size;
        mip_info.texDepth = 1;
        mip_info.channels = 3;
        mip_info.boundaryMode = TEX_BOUNDARY_MODE_WRAP;
        mip_info.mipLevelLimit = -1;
        int mip_offsets[TEX_MAX_MIP_LEVEL]{};
        const int mip_float_count = calculateMipInfo(mip_info, mip_offsets);
        DeviceArray<float> device_mips(static_cast<size_t>(mip_float_count));
        DeviceArray<float> device_mip_gradients(
                static_cast<size_t>(mip_float_count));

        device_indices.upload(host_indices.data(), host_indices.size(),
                              "cudaMemcpy texture bake indices");
        device_texcoords.upload(mesh.texcoords.data(), mesh.texcoords.size(),
                                "cudaMemcpy texture bake UVs");
        device_homogeneous_positions.upload(
                host_positions.data(), host_positions.size(),
                "cudaMemcpy texture bake positions");
        device_uv_clip_positions.upload(
                host_uv_positions.data(), host_uv_positions.size(),
                "cudaMemcpy texture bake UV clip positions");
        device_observations.upload(host_observations.data(),
                                   host_observations.size(),
                                   "cudaMemcpy texture bake observations");
        device_texture.zero("cudaMemset texture bake texture");
        device_first_moment.zero("cudaMemset texture bake first moment");
        device_second_moment.zero("cudaMemset texture bake second moment");

        CublasHandle cublas;
        CR::CudaRaster rasterizer;
        rasterizer.setBufferSize(image_width, image_height, 1);
        rasterizer.setViewport(image_width, image_height, 0, 0);
        rasterizer.setRenderModeFlags(0);
        rasterizer.setIndexBuffer(device_indices.data(),
                                  static_cast<int>(triangles));

        for (size_t view_index = 0; view_index < cameras.size(); ++view_index) {
            std::array<float, 16> view{};
            std::array<float, 16> projection{};
            make_view_projection(cameras[view_index], config, view, projection);
            const std::array<float, 16> view_column_major =
                    to_column_major(view);
            const std::array<float, 16> projection_column_major =
                    to_column_major(projection);
            device_view.upload(view_column_major.data(),
                               view_column_major.size(),
                               "cudaMemcpy texture bake view");
            device_projection.upload(projection_column_major.data(),
                                     projection_column_major.size(),
                                     "cudaMemcpy texture bake projection");
            constexpr float one = 1.0f;
            constexpr float zero = 0.0f;
            check_cublas(cublasSgemm(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, 4,
                                     4, 4, &one, device_projection.data(), 4,
                                     device_view.data(), 4, &zero,
                                     device_mvp.data(), 4),
                         "cublasSgemm texture bake projection-view");
            check_cublas(
                    cublasSgemm(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, 4,
                                static_cast<int>(vertices), 4, &one,
                                device_mvp.data(), 4,
                                reinterpret_cast<const float*>(
                                        device_homogeneous_positions.data()),
                                4, &zero,
                                reinterpret_cast<float*>(
                                        device_clip_positions.data()),
                                4),
                    "cublasSgemm texture bake clip positions");
            rasterizer.setVertexBuffer(device_clip_positions.data(),
                                       static_cast<int>(vertices));
            rasterizer.deferredClear(0);
            if (!rasterizer.drawTriangles(nullptr, false, nullptr)) {
                error = "nvdiffrast CUDA rasterizer exceeded its internal "
                        "subtriangle capacity";
                return false;
            }
            rasterize_shader(rasterizer, device_clip_positions.data(),
                             device_indices.data(), static_cast<int>(vertices),
                             static_cast<int>(triangles), image_width,
                             image_height, device_raster.data(),
                             device_raster_derivatives.data());
            interpolate_uv(
                    device_indices.data(), device_texcoords.data(),
                    static_cast<int>(vertices), static_cast<int>(triangles),
                    image_width, image_height, device_raster.data(),
                    device_raster_derivatives.data(),
                    device_uv_maps.data() + view_index * image_pixels * 2,
                    device_uv_derivatives.data() +
                            view_index * image_pixels * 4);
        }

        NumpyRandomState random(config.random_seed);
        const size_t texture_values = texture_pixels * 3;
        const int adam_threads = 256;
        const int adam_blocks = static_cast<int>(
                (texture_values + adam_threads - 1) / adam_threads);
        const int image_threads = 256;
        const int image_blocks = static_cast<int>(
                (image_pixels + image_threads - 1) / image_threads);
        float learning_rate = texture_bake_learning_rate_for_update(config, 0);
        for (int step = 0; step < config.steps; ++step) {
            const size_t view_index = static_cast<size_t>(
                    random.randint(static_cast<int>(observations.size())));
            TextureKernelParams parameters{};
            configure_texture_params(
                    parameters, device_texture.data(), device_mips.data(),
                    device_texture_gradient.data(), device_mip_gradients.data(),
                    config.texture_size, image_width, image_height,
                    device_uv_maps.data() + view_index * image_pixels * 2,
                    device_uv_derivatives.data() +
                            view_index * image_pixels * 4,
                    device_rendered.data(), device_rendered_gradient.data(),
                    device_uv_gradient.data(),
                    device_uv_derivative_gradient.data());
            device_texture_gradient.zero("cudaMemset texture bake gradient");
            device_mip_gradients.zero("cudaMemset texture bake mip gradient");
            build_mipmaps(parameters);
            void* forward_arguments[] = {&parameters};
            launch(reinterpret_cast<void*>(TextureFwdKernelLinearMipmapLinear1),
                   grid_8x8(image_width, image_height), block_8x8(),
                   forward_arguments, 0,
                   "nvdiffrast TextureFwdKernelLinearMipmapLinear1");
            l1_gradient_masked<<<image_blocks, image_threads>>>(
                    device_rendered.data(),
                    device_observations.data() + view_index * image_pixels * 3,
                    device_rendered_gradient.data(),
                    static_cast<int>(image_pixels), valid_pixels[view_index]);
            check_cuda(cudaGetLastError(), "native texture bake L1 gradient");
            texture_gradient(parameters);
            add_total_variation_gradient<<<grid_8x8(config.texture_size,
                                                    config.texture_size),
                                           block_8x8()>>>(
                    device_texture.data(), device_texture_gradient.data(),
                    config.texture_size, config.texture_size, config.tv_weight);
            check_cuda(cudaGetLastError(), "native texture bake TV gradient");
            const float step_number = static_cast<float>(step + 1);
            const float bias_correction1 = 1.0f - powf(kAdamBeta1, step_number);
            const float bias_correction2_sqrt =
                    sqrtf(1.0f - powf(kAdamBeta2, step_number));
            adam_step<<<adam_blocks, adam_threads>>>(
                    device_texture.data(), device_texture_gradient.data(),
                    device_first_moment.data(), device_second_moment.data(),
                    texture_values, learning_rate, bias_correction1,
                    bias_correction2_sqrt);
            check_cuda(cudaGetLastError(), "native texture bake Adam update");
            learning_rate =
                    texture_bake_learning_rate_for_update(config, step + 1);
        }
        check_cuda(cudaDeviceSynchronize(),
                   "native texture bake optimizer synchronization");

        rasterizer.setBufferSize(config.texture_size, config.texture_size, 1);
        rasterizer.setViewport(config.texture_size, config.texture_size, 0, 0);
        rasterizer.setVertexBuffer(device_uv_clip_positions.data(),
                                   static_cast<int>(vertices));
        rasterizer.deferredClear(0);
        if (!rasterizer.drawTriangles(nullptr, false, nullptr)) {
            error = "nvdiffrast CUDA UV rasterizer exceeded its internal "
                    "subtriangle capacity";
            return false;
        }
        rasterize_shader(rasterizer, device_uv_clip_positions.data(),
                         device_indices.data(), static_cast<int>(vertices),
                         static_cast<int>(triangles), config.texture_size,
                         config.texture_size, device_raster.data(),
                         device_raster_derivatives.data());
        const int coverage_blocks = static_cast<int>(
                (texture_pixels + image_threads - 1) / image_threads);
        coverage_from_raster<<<coverage_blocks, image_threads>>>(
                device_raster.data(), device_coverage.data(),
                static_cast<int>(texture_pixels));
        check_cuda(cudaGetLastError(), "native texture bake atlas coverage");
        check_cuda(cudaDeviceSynchronize(),
                   "native texture bake atlas synchronization");

        std::vector<float> host_texture(texture_values);
        std::vector<uint8_t> coverage(texture_pixels);
        device_texture.download(host_texture.data(), host_texture.size(),
                                "cudaMemcpy texture bake texture");
        device_coverage.download(coverage.data(), coverage.size(),
                                 "cudaMemcpy texture bake coverage");
        RgbaImage output;
        output.width = config.texture_size;
        output.height = config.texture_size;
        output.rgba.resize(texture_pixels * 4);
        std::vector<uint8_t> holes(texture_pixels);
        if (optimized_texture != nullptr)
            optimized_texture->resize(texture_values);
        for (int y = 0; y < config.texture_size; ++y) {
            const int source_y = config.texture_size - 1 - y;
            for (int x = 0; x < config.texture_size; ++x) {
                const size_t output_pixel =
                        static_cast<size_t>(y) * config.texture_size + x;
                const size_t source_pixel =
                        static_cast<size_t>(source_y) * config.texture_size + x;
                for (int channel = 0; channel < 3; ++channel) {
                    if (optimized_texture != nullptr) {
                        (*optimized_texture)[output_pixel * 3 + channel] =
                                host_texture[source_pixel * 3 + channel];
                    }
                    const float value = std::max(
                            0.0f,
                            std::min(255.0f,
                                     host_texture[source_pixel * 3 + channel] *
                                             255.0f));
                    output.rgba[output_pixel * 4 + channel] =
                            static_cast<uint8_t>(value);
                }
                output.rgba[output_pixel * 4 + 3] = 255;
                // The final atlas is vertically flipped to match the official
                // PyTorch image convention. Apply the same flip to coverage
                // before passing holes to OpenCV Telea.
                holes[output_pixel] = coverage[source_pixel] == 0 ? 1 : 0;
            }
        }
        if (config.apply_telea &&
            !inpaint_texture_telea(output, holes, 3.0f, error))
            return false;
        texture = std::move(output);
        if (hole_mask != nullptr) *hole_mask = std::move(holes);
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace sam3d
