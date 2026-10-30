#include "sparse_rulebook_cuda.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace openpcdet::ext {

namespace {

constexpr int k_threads = 256;
constexpr uint64_t k_invalid_key = std::numeric_limits<uint64_t>::max();

template <typename T>
struct device_buffer {
    T * pointer = nullptr;
    size_t capacity = 0;

    ~device_buffer() {
        if (pointer) cudaFree(pointer);
    }

    bool reserve(size_t count) {
        if (count <= capacity) return true;
        if (pointer) cudaFree(pointer);
        pointer = nullptr;
        capacity = 0;
        if (count == 0) return true;
        if (cudaMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)) != cudaSuccess) {
            return false;
        }
        capacity = count;
        return true;
    }
};

struct rulebook_workspace {
    device_buffer<int32_t> coordinates;
    device_buffer<uint64_t> keys_a;
    device_buffer<uint64_t> keys_b;
    device_buffer<uint64_t> hash_keys;
    device_buffer<int32_t> hash_values;
    device_buffer<int32_t> dense_indices;
    device_buffer<int32_t> counts;
    device_buffer<int32_t> offsets;
    device_buffer<int32_t> compact_indices;
    device_buffer<int32_t> scatter_rows;
    device_buffer<int32_t> selected_count;
    device_buffer<int32_t> error_flag;
    device_buffer<uint8_t> cub_temp;
};

struct resident_rulebook {
    device_buffer<int32_t> coordinates;
    device_buffer<int32_t> dense_indices;
    device_buffer<int32_t> compact_offsets;
    device_buffer<int32_t> compact_indices;
    device_buffer<int32_t> scatter_rows;
};

struct conv_geometry {
    int32_t in_shape[3];
    int32_t out_shape[3];
    int32_t kernel[3];
    int32_t stride[3];
    int32_t padding[3];
    int32_t dilation[3];
    int32_t kernel_volume;
};

cudaStream_t rulebook_stream() {
    static thread_local cudaStream_t stream = [] {
        cudaStream_t value = nullptr;
        cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking);
        return value;
    }();
    return stream;
}

__device__ __forceinline__ uint64_t coordinate_key(
        int32_t batch, int32_t z, int32_t y, int32_t x,
        int32_t shape_z, int32_t shape_y, int32_t shape_x) {
    return ((static_cast<uint64_t>(batch) * shape_z + z) * shape_y + y) *
               shape_x + x + 1;
}

__device__ __forceinline__ void decode_key(
        uint64_t key, int32_t shape_z, int32_t shape_y, int32_t shape_x,
        int32_t & batch, int32_t & z, int32_t & y, int32_t & x) {
    uint64_t value = key - 1;
    x = static_cast<int32_t>(value % shape_x);
    value /= shape_x;
    y = static_cast<int32_t>(value % shape_y);
    value /= shape_y;
    z = static_cast<int32_t>(value % shape_z);
    batch = static_cast<int32_t>(value / shape_z);
}

__global__ void encode_input_keys_kernel(const int32_t * coordinates,
                                         uint64_t * keys,
                                         int32_t active_count,
                                         conv_geometry geometry) {
    const int32_t row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= active_count) return;
    const int32_t * coordinate = coordinates + static_cast<int64_t>(row) * 4;
    keys[row] = coordinate_key(coordinate[0], coordinate[1], coordinate[2], coordinate[3],
                               geometry.in_shape[0], geometry.in_shape[1],
                               geometry.in_shape[2]);
}

__global__ void decode_output_coordinates_kernel(const uint64_t * keys,
                                                 int32_t * coordinates,
                                                 int32_t output_count,
                                                 conv_geometry geometry) {
    const int32_t row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= output_count) return;
    int32_t batch, z, y, x;
    decode_key(keys[row], geometry.out_shape[0], geometry.out_shape[1],
               geometry.out_shape[2], batch, z, y, x);
    int32_t * coordinate = coordinates + static_cast<int64_t>(row) * 4;
    coordinate[0] = batch;
    coordinate[1] = z;
    coordinate[2] = y;
    coordinate[3] = x;
}

__global__ void build_input_hash_kernel(const int32_t * coordinates,
                                        uint64_t * hash_keys,
                                        int32_t * hash_values,
                                        int32_t hash_mask,
                                        int32_t active_count,
                                        conv_geometry geometry,
                                        int32_t * error_flag) {
    const int32_t row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= active_count) return;
    const int32_t * coordinate = coordinates + static_cast<int64_t>(row) * 4;
    const uint64_t key = coordinate_key(
        coordinate[0], coordinate[1], coordinate[2], coordinate[3],
        geometry.in_shape[0], geometry.in_shape[1], geometry.in_shape[2]);
    int32_t slot = static_cast<int32_t>(key & static_cast<uint64_t>(hash_mask));
    for (int32_t probe = 0; probe <= hash_mask; ++probe) {
        const auto previous = atomicCAS(
            reinterpret_cast<unsigned long long *>(hash_keys + slot), 0ULL,
            static_cast<unsigned long long>(key));
        if (previous == 0ULL) {
            hash_values[slot] = row;
            return;
        }
        if (previous == key) {
            atomicExch(error_flag, 1);
            return;
        }
        slot = (slot + 1) & hash_mask;
    }
    atomicExch(error_flag, 2);
}

__global__ void generate_output_candidates_kernel(const int32_t * coordinates,
                                                  uint64_t * candidates,
                                                  int32_t active_count,
                                                  conv_geometry geometry) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t candidate_count =
        static_cast<int64_t>(active_count) * geometry.kernel_volume;
    if (index >= candidate_count) return;

    const int32_t row = static_cast<int32_t>(index / geometry.kernel_volume);
    int32_t kernel_index = static_cast<int32_t>(index % geometry.kernel_volume);
    const int32_t kx = kernel_index % geometry.kernel[2];
    kernel_index /= geometry.kernel[2];
    const int32_t ky = kernel_index % geometry.kernel[1];
    const int32_t kz = kernel_index / geometry.kernel[1];
    const int32_t kernel_coord[3] = {kz, ky, kx};
    const int32_t * coordinate = coordinates + static_cast<int64_t>(row) * 4;
    const int32_t input_coord[3] = {coordinate[1], coordinate[2], coordinate[3]};
    int32_t output_coord[3] = {};
    for (int axis = 0; axis < 3; ++axis) {
        const int32_t numerator = input_coord[axis] + geometry.padding[axis] -
                                  kernel_coord[axis] * geometry.dilation[axis];
        if (numerator % geometry.stride[axis] != 0) {
            candidates[index] = k_invalid_key;
            return;
        }
        output_coord[axis] = numerator / geometry.stride[axis];
        if (output_coord[axis] < 0 || output_coord[axis] >= geometry.out_shape[axis]) {
            candidates[index] = k_invalid_key;
            return;
        }
    }
    candidates[index] = coordinate_key(
        coordinate[0], output_coord[0], output_coord[1], output_coord[2],
        geometry.out_shape[0], geometry.out_shape[1], geometry.out_shape[2]);
}

__device__ __forceinline__ int32_t hash_find(const uint64_t * hash_keys,
                                             const int32_t * hash_values,
                                             int32_t hash_mask,
                                             uint64_t key) {
    int32_t slot = static_cast<int32_t>(key & static_cast<uint64_t>(hash_mask));
    for (int32_t probe = 0; probe <= hash_mask; ++probe) {
        const uint64_t found = hash_keys[slot];
        if (found == 0) return -1;
        if (found == key) return hash_values[slot];
        slot = (slot + 1) & hash_mask;
    }
    return -1;
}

__global__ void fill_dense_rulebook_kernel(const uint64_t * output_keys,
                                           const uint64_t * hash_keys,
                                           const int32_t * hash_values,
                                           int32_t hash_mask,
                                           int32_t * dense_indices,
                                           int32_t output_count,
                                           conv_geometry geometry) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t pair_count =
        static_cast<int64_t>(output_count) * geometry.kernel_volume;
    if (index >= pair_count) return;

    const int32_t output_row = static_cast<int32_t>(index / geometry.kernel_volume);
    int32_t kernel_index = static_cast<int32_t>(index % geometry.kernel_volume);
    const int32_t kx = kernel_index % geometry.kernel[2];
    kernel_index /= geometry.kernel[2];
    const int32_t ky = kernel_index % geometry.kernel[1];
    const int32_t kz = kernel_index / geometry.kernel[1];
    const int32_t kernel_coord[3] = {kz, ky, kx};

    int32_t batch, out_z, out_y, out_x;
    decode_key(output_keys[output_row], geometry.out_shape[0], geometry.out_shape[1],
               geometry.out_shape[2], batch, out_z, out_y, out_x);
    const int32_t output_coord[3] = {out_z, out_y, out_x};
    int32_t input_coord[3] = {};
    for (int axis = 0; axis < 3; ++axis) {
        input_coord[axis] = output_coord[axis] * geometry.stride[axis] -
                            geometry.padding[axis] +
                            kernel_coord[axis] * geometry.dilation[axis];
        if (input_coord[axis] < 0 || input_coord[axis] >= geometry.in_shape[axis]) {
            dense_indices[index] = -1;
            return;
        }
    }
    const uint64_t key = coordinate_key(
        batch, input_coord[0], input_coord[1], input_coord[2],
        geometry.in_shape[0], geometry.in_shape[1], geometry.in_shape[2]);
    dense_indices[index] = hash_find(hash_keys, hash_values, hash_mask, key);
}

__global__ void count_valid_per_kernel(const int32_t * dense_indices,
                                       int32_t * counts,
                                       int64_t pair_count,
                                       int32_t kernel_volume) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < pair_count && dense_indices[index] >= 0) {
        atomicAdd(counts + index % kernel_volume, 1);
    }
}

__global__ void stable_compact_rulebook_kernel(const int32_t * dense_indices,
                                               const int32_t * offsets,
                                               int32_t * compact_indices,
                                               int32_t * scatter_rows,
                                               int32_t output_count,
                                               int32_t kernel_volume) {
    const int32_t kernel_index = static_cast<int32_t>(blockIdx.x);
    if (kernel_index >= kernel_volume) return;
    using block_scan = cub::BlockScan<int32_t, k_threads>;
    __shared__ typename block_scan::TempStorage scan_storage;
    __shared__ int32_t tile_base;
    if (threadIdx.x == 0) tile_base = offsets[kernel_index];
    __syncthreads();

    for (int32_t tile = 0; tile < output_count; tile += k_threads) {
        const int32_t output_row = tile + static_cast<int32_t>(threadIdx.x);
        const int64_t pair = static_cast<int64_t>(output_row) * kernel_volume +
                             kernel_index;
        const int32_t input_row = output_row < output_count ? dense_indices[pair] : -1;
        const int32_t valid = input_row >= 0 ? 1 : 0;
        int32_t rank = 0;
        int32_t tile_count = 0;
        block_scan(scan_storage).ExclusiveSum(valid, rank, tile_count);
        const int32_t base = tile_base;
        if (valid) {
            const int32_t compact_row = base + rank;
            compact_indices[compact_row] = input_row;
            scatter_rows[pair] = compact_row;
        }
        __syncthreads();
        if (threadIdx.x == 0) tile_base += tile_count;
        __syncthreads();
    }
}

__global__ void pack_capacity_rulebook_kernel(
        const int32_t * exact_compact,
        const int32_t * exact_scatter,
        const int32_t * exact_offsets,
        const int32_t * capacity_offsets,
        int32_t * capacity_compact,
        int32_t * capacity_scatter,
        int64_t exact_pair_count,
        int32_t kernel_volume) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= exact_pair_count) return;
    const int32_t exact_column = exact_scatter[pair];
    if (exact_column < 0) return;
    const int32_t kernel = static_cast<int32_t>(pair % kernel_volume);
    const int32_t capacity_column = capacity_offsets[kernel] +
                                    exact_column - exact_offsets[kernel];
    capacity_compact[capacity_column] = exact_compact[exact_column];
    capacity_scatter[pair] = capacity_column;
}

bool cuda_ok(cudaError_t status, const char * action, std::string & error) {
    if (status == cudaSuccess) return true;
    error = std::string(action) + ": " + cudaGetErrorString(status);
    return false;
}

bool derive_geometry(const sparse_tensor & input,
                     const sparse_conv_params & params,
                     conv_geometry & geometry,
                     std::string & error,
                     bool coordinates_are_device) {
    if (input.ndim != 3 || input.spatial_shape.size() != 3 || input.channels <= 0 ||
        (!coordinates_are_device && input.coordinates.size() % 4 != 0) ||
        params.kernel_size.size() != 3 ||
        params.stride.size() != 3 || params.padding.size() != 3 ||
        params.dilation.size() != 3 || params.transposed) {
        error = "CUDA rulebook supports non-transposed 3-D sparse convolution";
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (input.spatial_shape[axis] <= 0 || params.kernel_size[axis] <= 0 ||
            params.stride[axis] <= 0 || params.padding[axis] < 0 ||
            params.dilation[axis] <= 0) {
            error = "invalid CUDA sparse convolution geometry";
            return false;
        }
        geometry.in_shape[axis] = input.spatial_shape[axis];
        geometry.kernel[axis] = params.kernel_size[axis];
        geometry.stride[axis] = params.stride[axis];
        geometry.padding[axis] = params.padding[axis];
        geometry.dilation[axis] = params.dilation[axis];
        geometry.out_shape[axis] = params.submanifold
            ? geometry.in_shape[axis]
            : (geometry.in_shape[axis] + 2 * geometry.padding[axis] -
               geometry.dilation[axis] * (geometry.kernel[axis] - 1) - 1) /
                  geometry.stride[axis] + 1;
        if (geometry.out_shape[axis] <= 0) {
            error = "CUDA sparse convolution output shape is empty";
            return false;
        }
    }
    if (params.submanifold &&
        (geometry.stride[0] != 1 || geometry.stride[1] != 1 ||
         geometry.stride[2] != 1)) {
        error = "CUDA SubM rulebook requires stride one";
        return false;
    }
    const int64_t kernel_volume = static_cast<int64_t>(geometry.kernel[0]) *
                                  geometry.kernel[1] * geometry.kernel[2];
    if (kernel_volume <= 0 || kernel_volume > std::numeric_limits<int32_t>::max()) {
        error = "CUDA sparse kernel volume is out of range";
        return false;
    }
    geometry.kernel_volume = static_cast<int32_t>(kernel_volume);
    return true;
}

} // namespace

bool build_sparse_indice_map_cuda(const sparse_tensor & input,
                                  const sparse_conv_params & params,
                                  sparse_indice_map & map,
                                  std::string & error,
                                  const int32_t * input_coordinates_device,
                                  int32_t input_active_count,
                                  bool retain_host_arrays) {
    conv_geometry geometry{};
    const bool device_input = input_coordinates_device != nullptr;
    if (!derive_geometry(input, params, geometry, error, device_input)) return false;
    const int64_t active_count_64 = device_input ? input_active_count : input.active_count();
    if (active_count_64 <= 0 || active_count_64 > std::numeric_limits<int32_t>::max()) {
        error = "CUDA rulebook input active count is out of range";
        return false;
    }
    const int32_t active_count = static_cast<int32_t>(active_count_64);
    if (!device_input) {
        for (int32_t row = 0; row < active_count; ++row) {
            const int32_t * coordinate = input.coordinates.data() + static_cast<int64_t>(row) * 4;
            if (coordinate[0] < 0 || coordinate[1] < 0 ||
                coordinate[1] >= geometry.in_shape[0] || coordinate[2] < 0 ||
                coordinate[2] >= geometry.in_shape[1] || coordinate[3] < 0 ||
                coordinate[3] >= geometry.in_shape[2]) {
                error = "sparse coordinate outside spatial shape";
                return false;
            }
        }
    }

    const int64_t candidate_count_64 =
        static_cast<int64_t>(active_count) * geometry.kernel_volume;
    if (candidate_count_64 > std::numeric_limits<int32_t>::max()) {
        error = "CUDA rulebook candidate count is out of range";
        return false;
    }
    const int32_t candidate_count = static_cast<int32_t>(candidate_count_64);
    if (active_count > std::numeric_limits<int32_t>::max() / 4) {
        error = "CUDA rulebook hash capacity is out of range";
        return false;
    }
    int32_t hash_capacity = 8;
    const int64_t target_hash_capacity = static_cast<int64_t>(active_count) * 2;
    while (hash_capacity < target_hash_capacity) {
        if (hash_capacity > std::numeric_limits<int32_t>::max() / 2) {
            error = "CUDA rulebook hash capacity is out of range";
            return false;
        }
        hash_capacity <<= 1;
    }

    static thread_local rulebook_workspace workspace;
    const size_t coordinate_count = static_cast<size_t>(active_count) * 4;
    if (!workspace.coordinates.reserve(coordinate_count) ||
        !workspace.keys_a.reserve(std::max(active_count, candidate_count)) ||
        !workspace.keys_b.reserve(candidate_count) ||
        !workspace.hash_keys.reserve(hash_capacity) ||
        !workspace.hash_values.reserve(hash_capacity) ||
        !workspace.counts.reserve(geometry.kernel_volume) ||
        !workspace.offsets.reserve(static_cast<size_t>(geometry.kernel_volume) + 1) ||
        !workspace.compact_indices.reserve(candidate_count) ||
        !workspace.scatter_rows.reserve(candidate_count) ||
        !workspace.selected_count.reserve(1) || !workspace.error_flag.reserve(1)) {
        error = "CUDA rulebook workspace allocation failed";
        return false;
    }

    const cudaStream_t stream = rulebook_stream();
    const int32_t * coordinates_device = input_coordinates_device;
    cudaError_t status = cudaSuccess;
    if (!device_input) {
        status = cudaMemcpyAsync(workspace.coordinates.pointer, input.coordinates.data(),
                                 coordinate_count * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream);
        coordinates_device = workspace.coordinates.pointer;
    }
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(workspace.hash_keys.pointer, 0,
                                 hash_capacity * sizeof(uint64_t), stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(workspace.error_flag.pointer, 0, sizeof(int32_t), stream);
    }
    if (!cuda_ok(status, "CUDA rulebook input setup failed", error)) return false;

    const int input_blocks = (active_count + k_threads - 1) / k_threads;
    build_input_hash_kernel<<<input_blocks, k_threads, 0, stream>>>(
        coordinates_device, workspace.hash_keys.pointer,
        workspace.hash_values.pointer, hash_capacity - 1, active_count,
        geometry, workspace.error_flag.pointer);

    int32_t output_count = active_count;
    std::vector<uint64_t> output_keys;
    if (params.submanifold) {
        encode_input_keys_kernel<<<input_blocks, k_threads, 0, stream>>>(
            coordinates_device, workspace.keys_a.pointer,
            active_count, geometry);
    } else {
        const int candidate_blocks = (candidate_count + k_threads - 1) / k_threads;
        generate_output_candidates_kernel<<<candidate_blocks, k_threads, 0, stream>>>(
            coordinates_device, workspace.keys_a.pointer,
            active_count, geometry);
        size_t sort_bytes = 0;
        status = cub::DeviceRadixSort::SortKeys(
            nullptr, sort_bytes, workspace.keys_a.pointer, workspace.keys_b.pointer,
            candidate_count, 0, 64, stream);
        if (!cuda_ok(status, "CUDA rulebook sort query failed", error) ||
            !workspace.cub_temp.reserve(sort_bytes)) {
            if (error.empty()) error = "CUDA rulebook sort workspace allocation failed";
            return false;
        }
        status = cub::DeviceRadixSort::SortKeys(
            workspace.cub_temp.pointer, sort_bytes, workspace.keys_a.pointer,
            workspace.keys_b.pointer, candidate_count, 0, 64, stream);
        size_t unique_bytes = 0;
        if (status == cudaSuccess) {
            status = cub::DeviceSelect::Unique(
                nullptr, unique_bytes, workspace.keys_b.pointer,
                workspace.keys_a.pointer, workspace.selected_count.pointer,
                candidate_count, stream);
        }
        if (!cuda_ok(status, "CUDA rulebook unique query failed", error) ||
            !workspace.cub_temp.reserve(std::max(sort_bytes, unique_bytes))) {
            if (error.empty()) error = "CUDA rulebook unique workspace allocation failed";
            return false;
        }
        status = cub::DeviceSelect::Unique(
            workspace.cub_temp.pointer, unique_bytes, workspace.keys_b.pointer,
            workspace.keys_a.pointer, workspace.selected_count.pointer,
            candidate_count, stream);
        if (status == cudaSuccess) {
            status = cudaMemcpyAsync(&output_count, workspace.selected_count.pointer,
                                     sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        }
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
        if (!cuda_ok(status, "CUDA rulebook sort/unique failed", error)) return false;
        uint64_t last_key = 0;
        if (output_count > 0) {
            status = cudaMemcpyAsync(&last_key, workspace.keys_a.pointer + output_count - 1,
                                     sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        }
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
        if (!cuda_ok(status, "CUDA rulebook output-count check failed", error)) return false;
        if (last_key == k_invalid_key) --output_count;
        if (output_count <= 0) {
            error = "CUDA sparse convolution produced no output coordinates";
            return false;
        }
        if (retain_host_arrays) {
            output_keys.resize(output_count);
            status = cudaMemcpyAsync(output_keys.data(), workspace.keys_a.pointer,
                                     output_keys.size() * sizeof(uint64_t),
                                     cudaMemcpyDeviceToHost, stream);
            if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
            if (!cuda_ok(status, "CUDA rulebook output-key download failed", error)) return false;
        }
    }

    const int64_t pair_count = static_cast<int64_t>(output_count) * geometry.kernel_volume;
    if (pair_count > std::numeric_limits<int32_t>::max()) {
        error = "CUDA rulebook pair count is out of range";
        return false;
    }
    if (!workspace.dense_indices.reserve(static_cast<size_t>(pair_count)) ||
        !workspace.compact_indices.reserve(static_cast<size_t>(pair_count)) ||
        !workspace.scatter_rows.reserve(static_cast<size_t>(pair_count))) {
        error = "CUDA rulebook pair workspace allocation failed";
        return false;
    }
    const int pair_blocks = static_cast<int>((pair_count + k_threads - 1) / k_threads);
    fill_dense_rulebook_kernel<<<pair_blocks, k_threads, 0, stream>>>(
        workspace.keys_a.pointer, workspace.hash_keys.pointer,
        workspace.hash_values.pointer, hash_capacity - 1,
        workspace.dense_indices.pointer, output_count, geometry);
    status = cudaMemsetAsync(workspace.counts.pointer, 0,
                             geometry.kernel_volume * sizeof(int32_t), stream);
    if (status == cudaSuccess) {
        count_valid_per_kernel<<<pair_blocks, k_threads, 0, stream>>>(
            workspace.dense_indices.pointer, workspace.counts.pointer,
            pair_count, geometry.kernel_volume);
    }

    std::vector<int32_t> counts(geometry.kernel_volume);
    int32_t device_error = 0;
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(counts.data(), workspace.counts.pointer,
                                 counts.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(&device_error, workspace.error_flag.pointer,
                                 sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (!cuda_ok(status, "CUDA rulebook neighbor resolution failed", error)) return false;
    if (device_error != 0) {
        error = device_error == 1 ? "duplicate sparse coordinate" :
                                    "CUDA rulebook hash table overflow";
        return false;
    }

    std::vector<int32_t> offsets(static_cast<size_t>(geometry.kernel_volume) + 1, 0);
    for (int32_t kernel = 0; kernel < geometry.kernel_volume; ++kernel) {
        offsets[kernel + 1] = offsets[kernel] + counts[kernel];
    }
    const int32_t compact_count = offsets.back();
    status = cudaMemcpyAsync(workspace.offsets.pointer, offsets.data(),
                             offsets.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(workspace.scatter_rows.pointer, 0xFF,
                                 pair_count * sizeof(int32_t), stream);
    }
    if (status == cudaSuccess) {
        stable_compact_rulebook_kernel<<<geometry.kernel_volume, k_threads, 0, stream>>>(
            workspace.dense_indices.pointer, workspace.offsets.pointer,
            workspace.compact_indices.pointer, workspace.scatter_rows.pointer,
            output_count, geometry.kernel_volume);
        status = cudaGetLastError();
    }

    auto resident = std::make_shared<resident_rulebook>();
    const size_t output_coordinate_count = static_cast<size_t>(output_count) * 4;
    if (!resident->coordinates.reserve(output_coordinate_count) ||
        !resident->dense_indices.reserve(static_cast<size_t>(pair_count)) ||
        !resident->compact_offsets.reserve(offsets.size()) ||
        !resident->compact_indices.reserve(static_cast<size_t>(compact_count)) ||
        !resident->scatter_rows.reserve(static_cast<size_t>(pair_count))) {
        error = "CUDA resident rulebook allocation failed";
        return false;
    }
    if (params.submanifold) {
        status = cudaMemcpyAsync(resident->coordinates.pointer, coordinates_device,
                                 output_coordinate_count * sizeof(int32_t),
                                 cudaMemcpyDeviceToDevice, stream);
    } else {
        const int output_blocks = (output_count + k_threads - 1) / k_threads;
        decode_output_coordinates_kernel<<<output_blocks, k_threads, 0, stream>>>(
            workspace.keys_a.pointer, resident->coordinates.pointer, output_count, geometry);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(resident->dense_indices.pointer,
                                 workspace.dense_indices.pointer,
                                 pair_count * sizeof(int32_t),
                                 cudaMemcpyDeviceToDevice, stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(resident->compact_offsets.pointer,
                                 workspace.offsets.pointer,
                                 offsets.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToDevice, stream);
    }
    if (status == cudaSuccess && compact_count > 0) {
        status = cudaMemcpyAsync(resident->compact_indices.pointer,
                                 workspace.compact_indices.pointer,
                                 static_cast<size_t>(compact_count) * sizeof(int32_t),
                                 cudaMemcpyDeviceToDevice, stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(resident->scatter_rows.pointer,
                                 workspace.scatter_rows.pointer,
                                 pair_count * sizeof(int32_t),
                                 cudaMemcpyDeviceToDevice, stream);
    }
    if (!cuda_ok(status, "CUDA resident rulebook copy failed", error)) return false;

    map = {};
    map.ndim = 3;
    if (retain_host_arrays) {
        map.input_spatial_shape = input.spatial_shape;
        map.input_coordinates = input.coordinates;
    }
    map.kernel_volume = geometry.kernel_volume;
    map.spatial_shape = {geometry.out_shape[0], geometry.out_shape[1],
                         geometry.out_shape[2]};
    if (retain_host_arrays && params.submanifold) {
        map.coordinates = input.coordinates;
    } else if (retain_host_arrays) {
        map.coordinates.reserve(static_cast<size_t>(output_count) * 4);
        for (uint64_t key : output_keys) {
            uint64_t value = key - 1;
            const int32_t x = static_cast<int32_t>(value % geometry.out_shape[2]);
            value /= geometry.out_shape[2];
            const int32_t y = static_cast<int32_t>(value % geometry.out_shape[1]);
            value /= geometry.out_shape[1];
            const int32_t z = static_cast<int32_t>(value % geometry.out_shape[0]);
            const int32_t batch = static_cast<int32_t>(value / geometry.out_shape[0]);
            map.coordinates.insert(map.coordinates.end(), {batch, z, y, x});
        }
    }
    map.compact_offsets = std::move(offsets);
    map.device_owner = resident;
    map.device_coordinates = resident->coordinates.pointer;
    map.device_indices = resident->dense_indices.pointer;
    map.device_compact_offsets = resident->compact_offsets.pointer;
    map.device_compact_indices = resident->compact_indices.pointer;
    map.device_scatter_rows = resident->scatter_rows.pointer;
    map.device_output_active_count = output_count;
    if (retain_host_arrays) {
        map.indices.resize(static_cast<size_t>(pair_count));
        map.compact_indices.resize(compact_count);
        map.scatter_rows.resize(static_cast<size_t>(pair_count));
        status = cudaMemcpyAsync(map.indices.data(), workspace.dense_indices.pointer,
                                 map.indices.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream);
        if (status == cudaSuccess && compact_count > 0) {
            status = cudaMemcpyAsync(map.compact_indices.data(),
                                     workspace.compact_indices.pointer,
                                     map.compact_indices.size() * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, stream);
        }
        if (status == cudaSuccess) {
            status = cudaMemcpyAsync(map.scatter_rows.data(), workspace.scatter_rows.pointer,
                                     map.scatter_rows.size() * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, stream);
        }
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    return cuda_ok(status, "CUDA rulebook download failed", error);
}

bool upload_sparse_indice_map_cuda(const sparse_indice_map & exact,
                                   const sparse_indice_map & capacity,
                                   int32_t * compact_device,
                                   int32_t * scatter_device,
                                   int32_t * dense_device,
                                   std::string & error) {
    if (!exact.has_device_rulebook() || !compact_device || !scatter_device ||
        exact.kernel_volume <= 0 || capacity.kernel_volume != exact.kernel_volume ||
        capacity.compact_offsets.size() != exact.compact_offsets.size() ||
        capacity.output_active_count() < exact.output_active_count()) {
        char diag[512];
        std::snprintf(diag, sizeof(diag),
                      "invalid CUDA sparse rulebook upload "
                      "(has_device=%d compact=%d scatter=%d exact_kv=%d cap_kv=%d "
                      "exact_off=%zu cap_off=%zu exact_active=%lld cap_active=%lld)",
                      exact.has_device_rulebook() ? 1 : 0, compact_device ? 1 : 0,
                      scatter_device ? 1 : 0, exact.kernel_volume, capacity.kernel_volume,
                      exact.compact_offsets.size(), capacity.compact_offsets.size(),
                      (long long) exact.output_active_count(),
                      (long long) capacity.output_active_count());
        error = diag;
        return false;
    }
    for (int32_t k = 0; k < exact.kernel_volume; ++k) {
        const int32_t exact_count = exact.compact_offsets[k + 1] - exact.compact_offsets[k];
        const int32_t capacity_count = capacity.compact_offsets[k + 1] -
                                       capacity.compact_offsets[k];
        if (capacity_count < exact_count) {
            error = "CUDA sparse graph capacity is smaller than the current rulebook";
            return false;
        }
    }

    static thread_local device_buffer<int32_t> capacity_offsets_device;
    if (!capacity_offsets_device.reserve(capacity.compact_offsets.size())) {
        error = "CUDA sparse capacity-offset allocation failed";
        return false;
    }
    const cudaStream_t stream = rulebook_stream();
    const int64_t capacity_pairs = capacity.output_active_count() * capacity.kernel_volume;
    const int64_t exact_pairs = exact.output_active_count() * exact.kernel_volume;
    cudaError_t status = cudaMemcpyAsync(
        capacity_offsets_device.pointer, capacity.compact_offsets.data(),
        capacity.compact_offsets.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(
            scatter_device, 0xFF, static_cast<size_t>(capacity_pairs) * sizeof(int32_t), stream);
    }
    if (status == cudaSuccess && capacity.compact_offsets.back() > 0) {
        status = cudaMemsetAsync(compact_device, 0,
            static_cast<size_t>(capacity.compact_offsets.back()) * sizeof(int32_t), stream);
    }
    if (status == cudaSuccess && dense_device) {
        status = cudaMemsetAsync(
            dense_device, 0xFF, static_cast<size_t>(capacity_pairs) * sizeof(int32_t), stream);
        if (status == cudaSuccess) {
            status = cudaMemcpyAsync(dense_device, exact.device_indices,
                                     static_cast<size_t>(exact_pairs) * sizeof(int32_t),
                                     cudaMemcpyDeviceToDevice, stream);
        }
    }
    if (status == cudaSuccess) {
        const int blocks = static_cast<int>((exact_pairs + k_threads - 1) / k_threads);
        pack_capacity_rulebook_kernel<<<blocks, k_threads, 0, stream>>>(
            exact.device_compact_indices, exact.device_scatter_rows,
            exact.device_compact_offsets, capacity_offsets_device.pointer,
            compact_device, scatter_device, exact_pairs, exact.kernel_volume);
        status = cudaGetLastError();
    }
    return cuda_ok(status, "CUDA sparse capacity upload failed", error);
}

bool synchronize_sparse_rulebook_cuda(std::string & error) {
    return cuda_ok(cudaStreamSynchronize(rulebook_stream()),
                   "CUDA sparse rulebook synchronization failed", error);
}

// AICore addition for the stock-op sparse graph (see the header comment): the
// upstream compact lowering consumes raw -1 entries through the fused gather,
// while the stock get_rows graph needs them remapped to the appended zero row.
// Padding beyond exact_pairs is written to zero_row as well: freshly allocated
// capacity tensors would otherwise keep uninitialized indices that gather
// out-of-bounds rows.
__global__ void remap_dense_indices_kernel(const int32_t * raw_indices,
                                           int32_t * dense_indices,
                                           int64_t exact_pair_count,
                                           int64_t capacity_pair_count,
                                           int32_t zero_row) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= capacity_pair_count) return;
    dense_indices[index] = index < exact_pair_count
        ? (raw_indices[index] < 0 ? zero_row : raw_indices[index])
        : zero_row;
}

bool remap_dense_indices_cuda(const sparse_indice_map & exact, int32_t zero_row,
                              int32_t * dense_device, int64_t capacity_pair_count,
                              std::string & error) {
    if (!exact.has_device_rulebook() || !dense_device ||
        exact.device_indices == nullptr || exact.kernel_volume <= 0 ||
        exact.device_output_active_count <= 0 ||
        capacity_pair_count < exact.device_output_active_count * exact.kernel_volume) {
        error = "invalid CUDA dense rulebook remap request";
        return false;
    }
    const int64_t exact_pairs =
        exact.device_output_active_count * exact.kernel_volume;
    const cudaStream_t stream = rulebook_stream();
    const int blocks = static_cast<int>((capacity_pair_count + k_threads - 1) / k_threads);
    remap_dense_indices_kernel<<<blocks, k_threads, 0, stream>>>(
        exact.device_indices, dense_device, exact_pairs, capacity_pair_count, zero_row);
    return cuda_ok(cudaGetLastError(), "CUDA dense rulebook remap failed", error);
}

} // namespace openpcdet::ext
