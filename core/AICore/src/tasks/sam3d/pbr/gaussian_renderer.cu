// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "gaussian_renderer.hpp"
#include "rasterizer.h"

namespace sam3d {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(data_); }

    char* reserve(size_t bytes) {
        if (bytes > bytes_) {
            check_cuda(cudaFree(data_), "cudaFree renderer scratch buffer");
            data_ = nullptr;
            bytes_ = 0;
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes),
                       "cudaMalloc renderer scratch buffer");
            bytes_ = bytes;
        }
        return data_;
    }

private:
    char* data_ = nullptr;
    size_t bytes_ = 0;
};

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) : count_(count) {
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count_ * sizeof(T)),
                       "cudaMalloc renderer input");
        }
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { cudaFree(data_); }

    void upload(const T* source, size_t count) {
        if (count != count_)
            throw std::runtime_error("renderer upload size mismatch");
        if (count > 0)
            check_cuda(cudaMemcpy(data_, source, count * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy renderer input");
    }
    void download(T* destination, size_t count) const {
        if (count != count_)
            throw std::runtime_error("renderer download size mismatch");
        if (count > 0)
            check_cuda(cudaMemcpy(destination, data_, count * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy renderer output");
    }
    T* data() const { return data_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

std::array<float, 16> transpose_4x4(const std::array<float, 16>& matrix) {
    std::array<float, 16> result{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            result[static_cast<size_t>(column) * 4 + row] =
                    matrix[static_cast<size_t>(row) * 4 + column];
        }
    }
    return result;
}

std::array<float, 16> multiply_4x4(const std::array<float, 16>& left,
                                   const std::array<float, 16>& right) {
    std::array<float, 16> result{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            float value = 0.0f;
            for (int inner = 0; inner < 4; ++inner) {
                value += left[static_cast<size_t>(row) * 4 + inner] *
                         right[static_cast<size_t>(inner) * 4 + column];
            }
            result[static_cast<size_t>(row) * 4 + column] = value;
        }
    }
    return result;
}

std::array<float, 16> intrinsics_to_official_projection(
        const GaussianCamera& camera, const GaussianRenderConfig& config) {
    const float fx = camera.intrinsics[0];
    const float fy = camera.intrinsics[4];
    const float cx = camera.intrinsics[2];
    const float cy = camera.intrinsics[5];
    std::array<float, 16> projection{};
    projection[0] = 2.0f * fx;
    projection[5] = 2.0f * fy;
    projection[2] = 2.0f * cx - 1.0f;
    projection[6] = -2.0f * cy + 1.0f;
    projection[10] = config.far_plane / (config.far_plane - config.near_plane);
    projection[11] = config.near_plane * config.far_plane /
                     (config.near_plane - config.far_plane);
    projection[14] = 1.0f;
    return projection;
}

std::array<float, 3> camera_center(const GaussianCamera& camera) {
    // For rigid OpenCV extrinsics, camera position is -R^T t.
    std::array<float, 3> center{};
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] = -(camera.extrinsics[axis] * camera.extrinsics[3] +
                         camera.extrinsics[4 + axis] * camera.extrinsics[7] +
                         camera.extrinsics[8 + axis] * camera.extrinsics[11]);
    }
    return center;
}

uint8_t float_to_u8(float value) {
    value = std::max(0.0f, std::min(1.0f, value));
    return static_cast<uint8_t>(value * 255.0f);
}

}  // namespace

bool render_gaussian_views_cuda(const GaussianSplatSet& splats,
                                const std::vector<GaussianCamera>& cameras,
                                const GaussianRenderConfig& config,
                                std::vector<RgbaImage>& images,
                                std::string& error) {
    if (!splats.valid()) {
        error = "Gaussian renderer received inconsistent splat arrays";
        return false;
    }
    if (cameras.empty() || config.width <= 0 || config.height <= 0 ||
        !(config.kernel_size > 0.0f)) {
        error = "Gaussian renderer received an invalid camera or render "
                "configuration";
        return false;
    }
    try {
        const size_t count = splats.size();
        const size_t pixels = static_cast<size_t>(config.width) * config.height;
        DeviceArray<float> positions(count * 3);
        DeviceArray<float> sh0(count * 3);
        DeviceArray<float> opacities(count);
        DeviceArray<float> scales(count * 3);
        DeviceArray<float> rotations(count * 4);
        DeviceArray<float> color(pixels * 3);
        DeviceArray<float> background(3);
        DeviceArray<float> subpixel_offsets(pixels * 2);
        DeviceArray<float> viewmatrix(16);
        DeviceArray<float> projmatrix(16);
        DeviceArray<float> campos(3);
        DeviceArray<int> radii(2 * count);  // gsplat parity: per-axis radii
        positions.upload(splats.positions.data(), splats.positions.size());
        sh0.upload(splats.sh0.data(), splats.sh0.size());
        opacities.upload(splats.opacities.data(), splats.opacities.size());
        scales.upload(splats.scales.data(), splats.scales.size());
        rotations.upload(splats.rotations.data(), splats.rotations.size());
        const float black[3] = {0.0f, 0.0f, 0.0f};
        background.upload(black, 3);
        std::vector<float> zero_offsets(pixels * 2, 0.0f);
        subpixel_offsets.upload(zero_offsets.data(), zero_offsets.size());

        DeviceBuffer geometry;
        DeviceBuffer binning;
        DeviceBuffer image;
        std::vector<float> rendered(pixels * 3);
        images.clear();
        images.reserve(cameras.size());
        const float tan_fovx = 0.5f / cameras[0].intrinsics[0];
        const float tan_fovy = 0.5f / cameras[0].intrinsics[4];
        for (const GaussianCamera& camera : cameras) {
            const auto projection =
                    intrinsics_to_official_projection(camera, config);
            const auto full_projection =
                    multiply_4x4(projection, camera.extrinsics);
            const auto view_transposed = transpose_4x4(camera.extrinsics);
            const auto projection_transposed = transpose_4x4(full_projection);
            const auto center = camera_center(camera);
            viewmatrix.upload(view_transposed.data(), view_transposed.size());
            projmatrix.upload(projection_transposed.data(),
                              projection_transposed.size());
            campos.upload(center.data(), center.size());

            CudaRasterizer::Rasterizer::forward(
                    [&geometry](size_t bytes) {
                        return geometry.reserve(bytes);
                    },
                    [&binning](size_t bytes) { return binning.reserve(bytes); },
                    [&image](size_t bytes) { return image.reserve(bytes); },
                    static_cast<int>(count), 0, 1, background.data(),
                    config.width, config.height, positions.data(), sh0.data(),
                    nullptr, opacities.data(), scales.data(), 1.0f,
                    rotations.data(), nullptr, viewmatrix.data(),
                    projmatrix.data(), campos.data(), tan_fovx, tan_fovy,
                    config.kernel_size, subpixel_offsets.data(), false,
                    color.data(), radii.data(), false);
            check_cuda(cudaGetLastError(), "Gaussian rasterizer forward");
            check_cuda(cudaDeviceSynchronize(),
                       "Gaussian rasterizer synchronization");
            color.download(rendered.data(), rendered.size());

            RgbaImage output;
            output.width = config.width;
            output.height = config.height;
            output.rgba.resize(pixels * 4);
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                output.rgba[pixel * 4 + 0] = float_to_u8(rendered[pixel]);
                output.rgba[pixel * 4 + 1] =
                        float_to_u8(rendered[pixels + pixel]);
                output.rgba[pixel * 4 + 2] =
                        float_to_u8(rendered[pixels * 2 + pixel]);
                output.rgba[pixel * 4 + 3] = 255;
            }
            images.push_back(std::move(output));
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    return true;
}

}  // namespace sam3d
