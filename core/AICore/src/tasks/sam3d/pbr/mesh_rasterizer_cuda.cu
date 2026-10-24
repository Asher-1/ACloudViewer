// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <CudaRaster.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh_rasterizer_cuda.hpp"
#include "mesh_visibility_camera_reference.hpp"

namespace sam3d {
namespace {

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

class CublasHandle {
public:
    CublasHandle() {
        check_cublas(cublasCreate(&handle_),
                     "cublasCreate native visibility rasterizer");
        check_cublas(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH),
                     "cublasSetMathMode native visibility rasterizer");
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

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) : count_(count) {
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count_ * sizeof(T)),
                       "cudaMalloc native nvdiffrast buffer");
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    ~DeviceArray() { cudaFree(data_); }

    void upload(const T* source, size_t count, const char* label) {
        if (count != count_)
            throw std::runtime_error(std::string(label) +
                                     " upload size mismatch");
        if (count > 0) {
            check_cuda(cudaMemcpy(data_, source, count * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       label);
        }
    }

    void download(T* destination, size_t count, const char* label) const {
        if (count != count_)
            throw std::runtime_error(std::string(label) +
                                     " download size mismatch");
        if (count > 0) {
            check_cuda(cudaMemcpy(destination, data_, count * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       label);
        }
    }

    T* data() const { return data_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

__global__ void mark_visible_faces(const uint32_t* color,
                                   int raster_width,
                                   int width,
                                   int height,
                                   int face_count,
                                   uint32_t* visible_faces) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = width * height;
    if (pixel >= pixel_count) return;
    const int x = pixel % width;
    const int y = pixel / width;
    const uint32_t encoded_face = color[x + y * raster_width];
    if (encoded_face == 0) return;
    const uint32_t face = encoded_face - 1;
    if (face >= static_cast<uint32_t>(face_count)) return;
    atomicOr(visible_faces + face / 32, 1u << (face % 32));
}

__global__ void build_hammersley_view_projection(
        const float* yaws,
        const double* pitches,
        int view_index,
        float* view_column_major,
        float* projection_column_major) {
    const float yaw = yaws[view_index];
    const double pitch = pitches[view_index];
    // This is the exact operation ordering in utils3d.torch.view_look_at().
    // In particular, y is crossed before either x or z is normalized.
    // sphere_hammersley_sequence returns a Python F32 yaw and a NumPy F64
    // pitch. _fill_holes() preserves that mixed tensor dtype, then its
    // torch.tensor(...).cuda().float() call converts the assembled vector.
    const float yaw_sin = sinf(yaw);
    const float yaw_cos = cosf(yaw);
    const float eye_x =
            static_cast<float>(static_cast<double>(yaw_sin) * cos(pitch)) *
            2.0f;
    const float eye_y =
            static_cast<float>(static_cast<double>(yaw_cos) * cos(pitch)) *
            2.0f;
    const float eye_z = static_cast<float>(sin(pitch)) * 2.0f;
    float x_x = -eye_y;
    float x_y = eye_x;
    const float x_z = 0.0f;
    float y_x = -eye_z * x_y;
    float y_y = eye_z * x_x;
    float y_z = eye_x * x_y - eye_y * x_x;
    float z_x = eye_x;
    float z_y = eye_y;
    float z_z = eye_z;
    const float x_norm = sqrtf(x_x * x_x + x_y * x_y + x_z * x_z);
    const float y_norm = sqrtf(y_x * y_x + y_y * y_y + y_z * y_z);
    const float z_norm = sqrtf(z_x * z_x + z_y * z_y + z_z * z_z);
    x_x /= x_norm;
    x_y /= x_norm;
    y_x /= y_norm;
    y_y /= y_norm;
    y_z /= y_norm;
    z_x /= z_norm;
    z_y /= z_norm;
    z_z /= z_norm;
    const float t_x = -(x_x * eye_x + x_y * eye_y + x_z * eye_z);
    const float t_y = -(y_x * eye_x + y_y * eye_y + y_z * eye_z);
    const float t_z = -(z_x * eye_x + z_y * eye_y + z_z * eye_z);
    const float view_values[16] = {x_x, y_x, z_x, 0.0f, x_y, y_y, z_y, 0.0f,
                                   x_z, y_z, z_z, 0.0f, t_x, t_y, t_z, 1.0f};
    for (int element = 0; element < 16; ++element)
        view_column_major[element] = view_values[element];

    // perspective_from_fov_xy(fov, fov, 1, 3). The scalar tensor is F32 in
    // the official path, and the separate tan calls preserve its aspect step.
    constexpr float pi = 3.14159265358979323846f;
    const float fov = 40.0f * (pi / 180.0f);
    const float tan_x = tanf(fov * 0.5f);
    const float tan_y = tanf(fov * 0.5f);
    const float aspect = tan_x / tan_y;
    const float projection_values[16] = {1.0f / (tan_y * aspect),
                                         0.0f,
                                         0.0f,
                                         0.0f,
                                         0.0f,
                                         1.0f / tan_y,
                                         0.0f,
                                         0.0f,
                                         0.0f,
                                         0.0f,
                                         -2.0f,
                                         -1.0f,
                                         0.0f,
                                         0.0f,
                                         -3.0f,
                                         0.0f};
    for (int element = 0; element < 16; ++element) {
        projection_column_major[element] = projection_values[element];
    }
}

std::array<float, 16> projection_from_intrinsics(
        const MeshRasterCamera& camera, const MeshRasterConfig& config) {
    const float fx = camera.intrinsics[0];
    const float fy = camera.intrinsics[4];
    const float cx = camera.intrinsics[2];
    const float cy = camera.intrinsics[5];
    const float near_plane = config.near_plane;
    const float far_plane = config.far_plane;
    return {2.0f * fx,
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

void camera_view_projection(const MeshRasterCamera& camera,
                            const MeshRasterConfig& config,
                            std::array<float, 16>& view,
                            std::array<float, 16>& projection) {
    if (camera.has_view_projection) {
        view = camera.view;
        projection = camera.projection;
        return;
    }

    // This is utils3d.torch.extrinsics_to_view followed by
    // utils3d.torch.intrinsics_to_perspective. The GEMMs below deliberately run
    // on CUDA so the clip-space arithmetic follows the official tensor path.
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
    projection = projection_from_intrinsics(camera, config);
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

void make_hammersley_angles(int view_count,
                            std::vector<float>& yaws,
                            std::vector<double>& pitches) {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    yaws.resize(static_cast<size_t>(view_count));
    pitches.resize(static_cast<size_t>(view_count));
    for (int index = 0; index < view_count; ++index) {
        double inverse = 0.5;
        double radical_inverse = 0.0;
        for (int bits = index; bits > 0; bits >>= 1, inverse *= 0.5) {
            radical_inverse += static_cast<double>(bits & 1) * inverse;
        }
        const double u =
                static_cast<double>(index) / static_cast<double>(view_count);
        yaws[static_cast<size_t>(index)] =
                static_cast<float>(radical_inverse * 2.0 * kPi);
        pitches[static_cast<size_t>(index)] =
                std::acos(1.0 - 2.0 * u) - kPi * 0.5;
    }
}

bool valid_mesh(const NativeMesh& mesh, std::string& error) {
    if (mesh.positions.empty() || mesh.positions.size() % 3 != 0 ||
        mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        error = "CUDA visibility rasterizer requires non-empty triangle "
                "positions and indices";
        return false;
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    for (uint32_t index : mesh.indices) {
        if (index >= vertex_count ||
            index > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            error = "CUDA visibility rasterizer received an out-of-range index";
            return false;
        }
    }
    return true;
}

}  // namespace

static bool mesh_visibility_frequency_nvdiffrast_cuda_impl(
        const NativeMesh& mesh,
        const std::vector<MeshRasterCamera>* cameras,
        int hammersley_view_count,
        const MeshRasterConfig& config,
        std::vector<float>& frequency,
        std::string& error) {
    const bool generated_hammersley = cameras == nullptr;
    const int view_count = generated_hammersley
                                   ? hammersley_view_count
                                   : static_cast<int>(cameras->size());
    const bool canonical_hammersley =
            generated_hammersley &&
            view_count == detail::kOfficialVisibilityViewCount;
    if (!valid_mesh(mesh, error) || view_count <= 0 || config.width <= 0 ||
        config.height <= 0 || !(config.near_plane > 0.0f) ||
        !(config.far_plane > config.near_plane)) {
        if (error.empty())
            error = "invalid CUDA visibility rasterizer configuration";
        return false;
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    const size_t face_count = mesh.indices.size() / 3;
    if (vertex_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        face_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "CUDA visibility rasterizer input exceeds nvdiffrast index "
                "limits";
        return false;
    }
    try {
        const int raster_width = (config.width + 7) & ~7;
        const size_t bit_word_count = (face_count + 31) / 32;
        DeviceArray<float4> device_positions(vertex_count);
        DeviceArray<float4> device_homogeneous_positions(vertex_count);
        DeviceArray<int> device_indices(mesh.indices.size());
        DeviceArray<uint32_t> device_visible(bit_word_count);
        DeviceArray<float> device_projection(16);
        DeviceArray<float> device_view(16);
        DeviceArray<float> device_mvp(16);
        DeviceArray<float> device_yaws(generated_hammersley &&
                                                       !canonical_hammersley
                                               ? static_cast<size_t>(view_count)
                                               : 0);
        DeviceArray<double> device_pitches(
                generated_hammersley && !canonical_hammersley
                        ? static_cast<size_t>(view_count)
                        : 0);
        DeviceArray<float> device_reference_views(
                canonical_hammersley
                        ? detail::kOfficialVisibilityViewsColumnMajor.size()
                        : 0);
        std::vector<int> indices(mesh.indices.begin(), mesh.indices.end());
        device_indices.upload(indices.data(), indices.size(),
                              "cudaMemcpy native nvdiffrast indices");
        std::vector<float4> homogeneous_positions(vertex_count);
        for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
            const float* position = mesh.positions.data() + vertex * 3;
            homogeneous_positions[vertex] =
                    make_float4(position[0], position[1], position[2], 1.0f);
        }
        device_homogeneous_positions.upload(
                homogeneous_positions.data(), homogeneous_positions.size(),
                "cudaMemcpy native homogeneous positions");
        if (canonical_hammersley) {
            device_reference_views.upload(
                    detail::kOfficialVisibilityViewsColumnMajor.data(),
                    detail::kOfficialVisibilityViewsColumnMajor.size(),
                    "cudaMemcpy official visibility camera table");
            device_projection.upload(
                    detail::kOfficialVisibilityProjectionColumnMajor.data(),
                    detail::kOfficialVisibilityProjectionColumnMajor.size(),
                    "cudaMemcpy official visibility projection");
        } else if (generated_hammersley) {
            std::vector<float> yaws;
            std::vector<double> pitches;
            make_hammersley_angles(view_count, yaws, pitches);
            device_yaws.upload(yaws.data(), yaws.size(),
                               "cudaMemcpy native Hammersley yaws");
            device_pitches.upload(pitches.data(), pitches.size(),
                                  "cudaMemcpy native Hammersley pitches");
        }
        CublasHandle cublas;

        CR::CudaRaster rasterizer;
        rasterizer.setBufferSize(config.width, config.height, 1);
        rasterizer.setViewport(config.width, config.height, 0, 0);
        rasterizer.setRenderModeFlags(0);
        rasterizer.setIndexBuffer(device_indices.data(),
                                  static_cast<int>(face_count));

        std::vector<uint32_t> visible_words(bit_word_count);
        std::vector<uint32_t> hits(face_count, 0);
        const int threads = 256;
        const int blocks =
                (config.width * config.height + threads - 1) / threads;
        for (int view_index = 0; view_index < view_count; ++view_index) {
            const float* view_matrix = device_view.data();
            if (canonical_hammersley) {
                view_matrix = device_reference_views.data() +
                              static_cast<size_t>(view_index) * 16;
            } else if (generated_hammersley) {
                build_hammersley_view_projection<<<1, 1>>>(
                        device_yaws.data(), device_pitches.data(), view_index,
                        device_view.data(), device_projection.data());
                check_cuda(cudaGetLastError(),
                           "native Hammersley camera kernel");
            } else {
                std::array<float, 16> view{};
                std::array<float, 16> projection{};
                camera_view_projection(
                        (*cameras)[static_cast<size_t>(view_index)], config,
                        view, projection);
                const std::array<float, 16> projection_column_major =
                        to_column_major(projection);
                const std::array<float, 16> view_column_major =
                        to_column_major(view);
                device_projection.upload(projection_column_major.data(),
                                         projection_column_major.size(),
                                         "cudaMemcpy native projection matrix");
                device_view.upload(view_column_major.data(),
                                   view_column_major.size(),
                                   "cudaMemcpy native view matrix");
            }
            constexpr float one = 1.0f;
            constexpr float zero = 0.0f;
            check_cublas(
                    cublasSgemm(cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4,
                                &one, device_projection.data(), 4, view_matrix,
                                4, &zero, device_mvp.data(), 4),
                    "cublasSgemm native projection-view");
            check_cublas(
                    cublasSgemm(
                            cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, 4,
                            static_cast<int>(vertex_count), 4, &one,
                            device_mvp.data(), 4,
                            reinterpret_cast<const float*>(
                                    device_homogeneous_positions.data()),
                            4, &zero,
                            reinterpret_cast<float*>(device_positions.data()),
                            4),
                    "cublasSgemm native clip positions");
            rasterizer.setVertexBuffer(device_positions.data(),
                                       static_cast<int>(vertex_count));
            rasterizer.deferredClear(0);
            if (!rasterizer.drawTriangles(nullptr, false, 0)) {
                error = "nvdiffrast CUDA rasterizer exceeded its internal "
                        "subtriangle capacity";
                return false;
            }
            check_cuda(cudaMemset(device_visible.data(), 0,
                                  bit_word_count * sizeof(uint32_t)),
                       "cudaMemset native visibility bits");
            mark_visible_faces<<<blocks, threads>>>(
                    static_cast<const uint32_t*>(rasterizer.getColorBuffer()),
                    raster_width, config.width, config.height,
                    static_cast<int>(face_count), device_visible.data());
            check_cuda(cudaGetLastError(),
                       "native visibility face-mark kernel");
            device_visible.download(visible_words.data(), visible_words.size(),
                                    "cudaMemcpy native visibility bits");
            for (size_t face = 0; face < face_count; ++face) {
                hits[face] += (visible_words[face / 32] >> (face % 32)) & 1u;
            }
        }
        frequency.resize(face_count);
        const float inverse_view_count = 1.0f / static_cast<float>(view_count);
        for (size_t face = 0; face < face_count; ++face) {
            frequency[face] =
                    static_cast<float>(hits[face]) * inverse_view_count;
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool mesh_visibility_frequency_nvdiffrast_cuda(
        const NativeMesh& mesh,
        const std::vector<MeshRasterCamera>& cameras,
        const MeshRasterConfig& config,
        std::vector<float>& frequency,
        std::string& error) {
    return mesh_visibility_frequency_nvdiffrast_cuda_impl(
            mesh, &cameras, 0, config, frequency, error);
}

bool mesh_visibility_hammersley_frequency_nvdiffrast_cuda(
        const NativeMesh& mesh,
        int view_count,
        const MeshRasterConfig& config,
        std::vector<float>& frequency,
        std::string& error) {
    return mesh_visibility_frequency_nvdiffrast_cuda_impl(
            mesh, nullptr, view_count, config, frequency, error);
}

bool mesh_hammersley_camera_matrices_nvdiffrast_cuda(
        int view_count,
        std::vector<float>& views,
        std::vector<float>& projections,
        std::string& error) {
    if (view_count <= 0) {
        error = "Hammersley camera matrix dump requires a positive view count";
        return false;
    }
    try {
        if (view_count == detail::kOfficialVisibilityViewCount) {
            views.resize(detail::kOfficialVisibilityViewsColumnMajor.size());
            projections.resize(views.size());
            for (int index = 0; index < view_count; ++index) {
                for (int row = 0; row < 4; ++row) {
                    for (int column = 0; column < 4; ++column) {
                        const size_t row_major =
                                static_cast<size_t>(index) * 16 + row * 4 +
                                column;
                        const size_t column_major =
                                static_cast<size_t>(index) * 16 + column * 4 +
                                row;
                        views[row_major] =
                                detail::kOfficialVisibilityViewsColumnMajor
                                        [column_major];
                        projections[row_major] =
                                detail::kOfficialVisibilityProjectionColumnMajor
                                        [static_cast<size_t>(column) * 4 + row];
                    }
                }
            }
            return true;
        }
        std::vector<float> yaws;
        std::vector<double> pitches;
        make_hammersley_angles(view_count, yaws, pitches);
        const size_t matrix_values = static_cast<size_t>(view_count) * 16;
        DeviceArray<float> device_yaws(yaws.size());
        DeviceArray<double> device_pitches(pitches.size());
        DeviceArray<float> device_views(matrix_values);
        DeviceArray<float> device_projections(matrix_values);
        device_yaws.upload(yaws.data(), yaws.size(),
                           "cudaMemcpy native Hammersley yaws");
        device_pitches.upload(pitches.data(), pitches.size(),
                              "cudaMemcpy native Hammersley pitches");
        for (int index = 0; index < view_count; ++index) {
            build_hammersley_view_projection<<<1, 1>>>(
                    device_yaws.data(), device_pitches.data(), index,
                    device_views.data() + index * 16,
                    device_projections.data() + index * 16);
        }
        check_cuda(cudaGetLastError(),
                   "native Hammersley camera matrix kernel");
        std::vector<float> views_column_major(matrix_values);
        std::vector<float> projections_column_major(matrix_values);
        device_views.download(views_column_major.data(),
                              views_column_major.size(),
                              "cudaMemcpy native Hammersley views");
        device_projections.download(projections_column_major.data(),
                                    projections_column_major.size(),
                                    "cudaMemcpy native Hammersley projections");
        views.resize(matrix_values);
        projections.resize(matrix_values);
        for (int index = 0; index < view_count; ++index) {
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    const size_t row_major =
                            static_cast<size_t>(index) * 16 + row * 4 + column;
                    const size_t column_major =
                            static_cast<size_t>(index) * 16 + column * 4 + row;
                    views[row_major] = views_column_major[column_major];
                    projections[row_major] =
                            projections_column_major[column_major];
                }
            }
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace sam3d
