// AICore OpenPCDet CUDA extensions — single translation unit.
//
// The three upstream OpenPCDet-GGML extension sources (GPU sparse rulebook
// generation, PillarVFE scatter, HeightCompression) are compiled together on
// purpose. In the monolithic libAICore link, splitting them into separate
// .cu units left the HeightCompression/VFE kernel launches failing with
// cudaErrorInvalidDeviceFunction while the rulebook unit's kernels ran
// (registration data and fatbin contents verified identical, and the same
// objects linked standalone worked). A single whole-program unit sidesteps
// that multi-fatbin registration hazard entirely, matches how the engines'
// consumers link (one extension module per task), and changes no semantics:
// every helper below lives in its own anonymous namespace with distinct
// names (device_buffer/device_buf/coordinate_buffer/spatial_buffer, one
// private stream per subsystem).
#include "sparse_rulebook_cuda.cu"
#include "vfe_scatter_cuda.cu"
#include "height_compression_cuda.cu"

// ---------------------------------------------------------------------------
// Deterministic focal sparse conv (方案 C, see ../DETERMINISTIC_FOCAL_DESIGN.md
// and deterministic_sparse_conv.h). Appended here so the kernel keeps the
// single-translation-unit contract of this extension module. The reduction
// order (k outer / c inner per output cell) and rounding (fmaf single
// rounding) mirror det_sparse_conv_cpu exactly; there is no shared/warp
// reduction and no atomics, so CPU and GPU outputs are bit-identical.
// ---------------------------------------------------------------------------
#include "deterministic_sparse_conv.h"

namespace openpcdet::det {

namespace {

constexpr int k_det_block_x = 128;
constexpr int k_det_block_y = 8;

__global__ void det_sparse_conv_kernel(
        const float * __restrict__ feat, int C_in,
        const int32_t * __restrict__ idx,  // [n_out, K], -1 = empty
        const float * __restrict__ W,      // [C_out, K*C_in] o-major (ggml native)
        const float * __restrict__ bias,   // [C_out]
        float * __restrict__ out,          // [n_out, C_out]
        int n_out, int C_out, int K) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (v >= n_out || o >= C_out) return;
    float acc = bias[o];
    for (int k = 0; k < K; ++k) {
        const int r = idx[static_cast<size_t>(v) * K + k];
        if (r < 0) continue;
        const float * row = feat + static_cast<size_t>(r) * C_in;
        // W is the ggml-native [C_out, K*C_in] o-major layout.
        const float * w = W + static_cast<size_t>(o) * (K * C_in) +
                          static_cast<size_t>(k) * C_in;
        for (int c = 0; c < C_in; ++c)
            acc = fmaf(row[c], w[c], acc);  // single rounding, as CPU
    }
    out[static_cast<size_t>(v) * C_out + o] = acc;
}

template <typename T>
struct det_device_buffer {
    T * pointer = nullptr;
    size_t capacity = 0;

    ~det_device_buffer() {
        if (pointer) cudaFree(pointer);
    }

    bool reserve(size_t count) {
        if (count <= capacity) return true;
        if (pointer) cudaFree(pointer);
        pointer = nullptr;
        capacity = 0;
        if (count == 0) return true;
        if (cudaMalloc(reinterpret_cast<void **>(&pointer),
                       count * sizeof(T)) != cudaSuccess) {
            return false;
        }
        capacity = count;
        return true;
    }
};

cudaStream_t det_conv_stream() {
    static thread_local cudaStream_t stream = [] {
        cudaStream_t value = nullptr;
        cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking);
        return value;
    }();
    return stream;
}

bool det_cuda_ok(cudaError_t status, const char * action, std::string & error) {
    if (status == cudaSuccess) return true;
    error = std::string(action) + ": " + cudaGetErrorString(status);
    return false;
}

}  // namespace

bool det_sparse_conv_cuda(const float * features_host, int in_rows,
                          int in_channels, const int32_t * neighbor_rows_host,
                          const float * weights_host, const float * bias_host,
                          float * out_host, int n_out, int out_channels,
                          int kernel_volume, std::string & error) {
    if (!features_host || !neighbor_rows_host || !weights_host || !out_host ||
        in_rows <= 0 || in_channels <= 0 || n_out <= 0 || out_channels <= 0 ||
        kernel_volume <= 0) {
        error = "invalid deterministic sparse conv CUDA input";
        return false;
    }
    static thread_local det_device_buffer<float> features_device;
    static thread_local det_device_buffer<int32_t> neighbors_device;
    static thread_local det_device_buffer<float> weights_device;
    static thread_local det_device_buffer<float> bias_device;
    static thread_local det_device_buffer<float> output_device;
    const size_t feature_count = static_cast<size_t>(in_rows) * in_channels;
    const size_t neighbor_count = static_cast<size_t>(n_out) * kernel_volume;
    const size_t weight_count =
        static_cast<size_t>(kernel_volume) * in_channels * out_channels;
    if (!features_device.reserve(feature_count) ||
        !neighbors_device.reserve(neighbor_count) ||
        !weights_device.reserve(weight_count) ||
        !output_device.reserve(static_cast<size_t>(n_out) * out_channels) ||
        (bias_host && !bias_device.reserve(static_cast<size_t>(out_channels)))) {
        error = "deterministic sparse conv CUDA staging allocation failed";
        return false;
    }
    const cudaStream_t stream = det_conv_stream();
    cudaError_t status = cudaMemcpyAsync(
        features_device.pointer, features_host, feature_count * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(neighbors_device.pointer, neighbor_rows_host,
                                 neighbor_count * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(weights_device.pointer, weights_host,
                                 weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice, stream);
    }
    if (status == cudaSuccess && bias_host) {
        status = cudaMemcpyAsync(bias_device.pointer, bias_host,
                                 static_cast<size_t>(out_channels) * sizeof(float),
                                 cudaMemcpyHostToDevice, stream);
    }
    if (!det_cuda_ok(status, "deterministic sparse conv CUDA upload failed", error)) {
        return false;
    }
    const float * bias_device_ptr = bias_host ? bias_device.pointer : nullptr;
    if (bias_device_ptr == nullptr) {
        // conv_imp is bias-free: launch with a device zero so the kernel keeps
        // one code path (acc starts at +0.0f, the exact identity element).
        static thread_local det_device_buffer<float> zero_bias;
        if (!zero_bias.reserve(static_cast<size_t>(out_channels))) {
            error = "deterministic sparse conv CUDA zero-bias allocation failed";
            return false;
        }
        status = cudaMemsetAsync(zero_bias.pointer, 0,
                                 static_cast<size_t>(out_channels) * sizeof(float),
                                 stream);
        if (!det_cuda_ok(status, "deterministic sparse conv CUDA zero-bias failed", error)) {
            return false;
        }
        bias_device_ptr = zero_bias.pointer;
    }
    const dim3 block(k_det_block_x, k_det_block_y);
    const dim3 grid(static_cast<unsigned>((n_out + k_det_block_x - 1) / k_det_block_x),
                    static_cast<unsigned>((out_channels + k_det_block_y - 1) / k_det_block_y));
    det_sparse_conv_kernel<<<grid, block, 0, stream>>>(
        features_device.pointer, in_channels, neighbors_device.pointer,
        weights_device.pointer, bias_device_ptr, output_device.pointer,
        n_out, out_channels, kernel_volume);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(out_host, output_device.pointer,
                                 static_cast<size_t>(n_out) * out_channels *
                                     sizeof(float),
                                 cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    return det_cuda_ok(status, "deterministic sparse conv CUDA launch failed", error);
}

}  // namespace openpcdet::det

