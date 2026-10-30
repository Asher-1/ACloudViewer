#include "height_compression_cuda.h"

#include <cuda_runtime.h>

namespace openpcdet::ext {

namespace {

struct coordinate_buffer {
    int32_t * pointer = nullptr;
    size_t capacity = 0;

    ~coordinate_buffer() {
        if (pointer) cudaFree(pointer);
    }

    bool reserve(size_t count) {
        if (count <= capacity) return true;
        if (pointer) cudaFree(pointer);
        pointer = nullptr;
        capacity = 0;
        if (cudaMalloc(&pointer, count * sizeof(int32_t)) != cudaSuccess) return false;
        capacity = count;
        return true;
    }
};

struct spatial_buffer {
    float * pointer = nullptr;
    size_t capacity = 0;

    ~spatial_buffer() {
        if (pointer) cudaFree(pointer);
    }

    bool reserve(size_t count) {
        if (count <= capacity) return true;
        if (pointer) cudaFree(pointer);
        pointer = nullptr;
        capacity = 0;
        if (cudaMalloc(&pointer, count * sizeof(float)) != cudaSuccess) return false;
        capacity = count;
        return true;
    }
};

cudaStream_t height_compression_stream() {
    static thread_local cudaStream_t stream = [] {
        cudaStream_t value = nullptr;
        cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking);
        return value;
    }();
    return stream;
}

template <bool output_whc>
__global__ void scatter_height_kernel(const float * __restrict__ sparse,
                                      const int32_t * __restrict__ coordinates,
                                      float * __restrict__ dense,
                                      int active_count,
                                      int sparse_channels,
                                      int depth,
                                      int height,
                                      int width) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t count = static_cast<int64_t>(active_count) * sparse_channels;
    if (index >= count) return;

    // HWC favors channel-contiguous reads/writes. WHC favors row-contiguous
    // output writes; sparse coordinates are sorted, so neighboring rows usually
    // land next to each other in a channel plane.
    const int row = output_whc
        ? static_cast<int>(index % active_count)
        : static_cast<int>(index / sparse_channels);
    const int channel = output_whc
        ? static_cast<int>(index / active_count)
        : static_cast<int>(index % sparse_channels);
    const int32_t * coordinate = coordinates + static_cast<int64_t>(row) * 4;
    const int z = coordinate[1];
    const int y = coordinate[2];
    const int x = coordinate[3];
    if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) return;

    const int dense_channels = sparse_channels * depth;
    const int output_channel = channel * depth + z;
    if constexpr (output_whc) {
        const int64_t output = static_cast<int64_t>(x) + static_cast<int64_t>(width) *
            (y + static_cast<int64_t>(height) * output_channel);
        dense[output] = sparse[static_cast<int64_t>(row) * sparse_channels + channel];
    } else {
        const int64_t output =
            (static_cast<int64_t>(y) * width + x) * dense_channels + output_channel;
        dense[output] = sparse[index];
    }
}

} // namespace

bool gpu_height_compress(const float * sparse_features_device,
                         const int32_t * coordinates_host,
                         int32_t active_count,
                         int32_t sparse_channels,
                         int32_t depth,
                         int32_t height,
                         int32_t width,
                         float * spatial_device,
                         std::string & error,
                         bool coordinates_are_device,
                         bool output_whc) {
    if (!sparse_features_device || !coordinates_host || !spatial_device ||
        active_count <= 0 || sparse_channels <= 0 || depth <= 0 ||
        height <= 0 || width <= 0) {
        error = "invalid CUDA HeightCompression input";
        return false;
    }

    static thread_local coordinate_buffer device_coordinates;
    static thread_local spatial_buffer staging_spatial;
    const size_t coordinate_count = static_cast<size_t>(active_count) * 4;
    const size_t dense_elements = static_cast<size_t>(height) * width *
                                  sparse_channels * depth;
    if (!coordinates_are_device && !device_coordinates.reserve(coordinate_count)) {
        error = "CUDA HeightCompression staging allocation failed";
        return false;
    }

    const uintptr_t sparse_begin = reinterpret_cast<uintptr_t>(sparse_features_device);
    const uintptr_t sparse_end = sparse_begin +
        static_cast<size_t>(active_count) * sparse_channels * sizeof(float);
    const uintptr_t dense_begin = reinterpret_cast<uintptr_t>(spatial_device);
    const uintptr_t dense_end = dense_begin + dense_elements * sizeof(float);
    const bool overlaps = sparse_begin < dense_end && dense_begin < sparse_end;
    if (overlaps && !staging_spatial.reserve(dense_elements)) {
        error = "CUDA HeightCompression overlap staging allocation failed";
        return false;
    }
    float * dense_target = overlaps ? staging_spatial.pointer : spatial_device;

    const cudaStream_t stream = height_compression_stream();
    const int32_t * coordinates_device = coordinates_host;
    cudaError_t status = cudaSuccess;
    if (!coordinates_are_device) {
        status = cudaMemcpyAsync(device_coordinates.pointer, coordinates_host,
                                 coordinate_count * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream);
        coordinates_device = device_coordinates.pointer;
    }
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(dense_target, 0,
                                 dense_elements * sizeof(float), stream);
    }
    if (status != cudaSuccess) {
        error = std::string("CUDA HeightCompression setup failed: ") +
                cudaGetErrorString(status);
        return false;
    }

    constexpr int threads = 256;
    const int64_t elements = static_cast<int64_t>(active_count) * sparse_channels;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    if (output_whc) {
        scatter_height_kernel<true><<<blocks, threads, 0, stream>>>(
            sparse_features_device, coordinates_device, dense_target,
            active_count, sparse_channels, depth, height, width);
    } else {
        scatter_height_kernel<false><<<blocks, threads, 0, stream>>>(
            sparse_features_device, coordinates_device, dense_target,
            active_count, sparse_channels, depth, height, width);
    }
    status = cudaGetLastError();
    if (status == cudaSuccess && overlaps) {
        status = cudaMemcpyAsync(spatial_device, dense_target,
                                 dense_elements * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream);
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        error = std::string("CUDA HeightCompression failed: ") +
                cudaGetErrorString(status);
        return false;
    }
    return true;
}

} // namespace openpcdet::ext
