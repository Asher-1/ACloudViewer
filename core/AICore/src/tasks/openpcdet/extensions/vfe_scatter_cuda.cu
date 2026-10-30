// GPU-accelerated dynamic PillarVFE + PointPillarScatter.
//
// Mirrors run_dynamic_pillar_vfe_cpu (pillar_vfe.cpp) + the scatter step in
// pointpillar.cpp exactly, but moves the expensive per-point MLP, the
// per-pillar maxpool and the BEV scatter onto the GPU. The voxelization /
// grouping / feature-construction stay on the host (they are cheap); the
// heavy matmul and the strided scatter are the GPU part.

#include "vfe_scatter_cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>

#include <cuda_runtime.h>

namespace openpcdet::ext {

namespace {

struct device_buf {
    void * ptr = nullptr;
    size_t bytes = 0;
    ~device_buf() { if (ptr) cudaFree(ptr); }
    bool alloc(size_t n, size_t esz) {
        if (ptr && n * esz <= bytes) return true;
        if (ptr) cudaFree(ptr), ptr = nullptr;
        bytes = n * esz;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) { ptr = nullptr; bytes = 0; return false; }
        return true;
    }
    template <typename T> T * as() { return static_cast<T *>(ptr); }
    // Upload-once for constant host sources (model weights): keyed by the host
    // pointer + length, which are stable for the lifetime of the model. The
    // first call runs on the graph-capture frame, so the device pointer this
    // fixes is also the pointer replayed by CUDA graphs.
    bool uploaded = false;
    const void * src_key = nullptr;
    size_t src_bytes = 0;
};


// Dedicated non-default stream so we don't hit the legacy default-stream global
// sync that would serialize against ggml's CUDA-graph stream.
static cudaStream_t g_stream() {
    static cudaStream_t s = [] { cudaStream_t r; cudaStreamCreateWithFlags(&r, cudaStreamNonBlocking); return r; }();
    return s;
}
static const cudaStream_t & S = g_stream();

// HtoD that skips when this exact host buffer was already uploaded to d.
static void upload_const(device_buf & d, const void * host, size_t bytes) {
    if (d.uploaded && d.src_key == host && d.src_bytes == bytes) return;
    cudaMemcpyAsync(d.ptr, host, bytes, cudaMemcpyHostToDevice, S);
    d.uploaded = true;
    d.src_key = host;
    d.src_bytes = bytes;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ static void atomic_max_f(float * addr, float value) {
    int * a = (int *) addr;
    int old = *a, assumed;
    do {
        assumed = old;
        const float cur = __int_as_float(assumed);
        const float nv = fmaxf(cur, value);
        old = atomicCAS(a, assumed, __float_as_int(nv));
    } while (assumed != old);
}

// Per-point linear + BN + ReLU.  d_in: [N2][cur_in], W: [out][cur_in].
__global__ static void k_linear_relu(const float * __restrict__ d_in,
                                     const float * __restrict__ W,
                                     const float * __restrict__ b,
                                     const float * __restrict__ scale,
                                     const float * __restrict__ bbias,
                                     float * __restrict__ d_out,
                                     int N2, int cur_in, int out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. N2*out-1
    const int total = N2 * out;
    if (idx >= total) return;
    const int row = idx / out;
    const int oc  = idx % out;
    const float * src = d_in + (size_t) row * cur_in;
    const float * wrow = W + (size_t) oc * cur_in;
    float value = b ? b[oc] : 0.0f;
    for (int ic = 0; ic < cur_in; ic++) value += src[ic] * wrow[ic];
    if (scale) value = value * scale[oc] + (bbias ? bbias[oc] : 0.0f);
    d_out[idx] = fmaxf(value, 0.0f);
}

// Max-pool over each pillar into d_agg (initialised to -1e30).
__global__ static void k_maxpool(const float * __restrict__ d_local,
                                 const int32_t * __restrict__ d_group,
                                 float * __restrict__ d_agg,
                                 int N2, int out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. N2*out-1
    const int total = N2 * out;
    if (idx >= total) return;
    const int row = idx / out;
    const int oc  = idx % out;
    const int g = d_group[row];
    atomic_max_f(&d_agg[(size_t) g * out + oc], d_local[idx]);
}

__global__ static void k_fill(float * __restrict__ d, float v, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] = v;
}

// Build next layer input = concat(local[point], agg[group]) -> d_in [N2][2*out].
__global__ static void k_concat(const float * __restrict__ d_local,
                                const float * __restrict__ d_agg,
                                const int32_t * __restrict__ d_group,
                                float * __restrict__ d_in,
                                int N2, int out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N2 * out;
    if (idx >= total) return;
    const int row = idx / out;
    const int oc  = idx % out;
    const int g = d_group[row];
    float * dst = d_in + (size_t) row * (2 * out);
    dst[oc]           = d_local[idx];
    dst[out + oc]     = d_agg[(size_t) g * out + oc];
}

// Scatter per-pillar features into the BEV [W,H,C] layout (W fastest, C slowest).
__global__ static void k_scatter(const float * __restrict__ d_pf,
                                 const int32_t * __restrict__ d_gx,
                                 const int32_t * __restrict__ d_gy,
                                 float * __restrict__ d_spatial,
                                 int V, int C_bev, int nx, int ny) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. V*C_bev-1
    const int total = V * C_bev;
    if (idx >= total) return;
    const int v = idx / C_bev;
    const int c = idx % C_bev;
    const int gx = d_gx[v];
    const int gy = d_gy[v];
    d_spatial[(size_t) gx + (size_t) nx * gy + (size_t) nx * ny * c] = d_pf[idx];
}

// Per-voxel mean of the first 3 coords + pillar grid coords.
__global__ static void k_vox_mean(const float * __restrict__ voxels,
                                  const int32_t * __restrict__ num_points,
                                  const int32_t * __restrict__ coords,
                                  float * __restrict__ d_mean,
                                  int32_t * __restrict__ d_gx, int32_t * __restrict__ d_gy,
                                  int V, int P, int point_dim) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    const int np = num_points[v];
    const float * vf = voxels + (size_t) v * P * point_dim;
    float mx = 0, my = 0, mz = 0;
    for (int p = 0; p < np; p++) {
        mx += vf[p * point_dim + 0]; my += vf[p * point_dim + 1]; mz += vf[p * point_dim + 2];
    }
    const float inv = np > 0 ? 1.0f / np : 0.0f;
    d_mean[v * 3 + 0] = mx * inv; d_mean[v * 3 + 1] = my * inv; d_mean[v * 3 + 2] = mz * inv;
    d_gx[v] = coords[v * 4 + 3]; d_gy[v] = coords[v * 4 + 2];
}

// Feature expansion per slot (parallelizes the host build loop).
__global__ static void k_expand_vox(const float * __restrict__ voxels,
                                    const int32_t * __restrict__ num_points,
                                    const int32_t * __restrict__ coords,
                                    const float * __restrict__ d_mean,
                                    float * __restrict__ d_features,
                                    int32_t * __restrict__ d_group,
                                    int V, int P, int point_dim, bool use_abs, bool with_distance,
                                    float vx, float vy, float vz, float x0, float y0, float z0,
                                    int in_c) {
    const long idx = (long) blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long) V * P;
    if (idx >= total) return;
    const int v = (int) (idx / P);
    const int p = (int) (idx % P);
    const int np = num_points[v];
    const float * vf = voxels + (size_t) v * P * point_dim;
    const float * src = vf + (size_t) p * point_dim;
    const bool valid = p < np;
    float * dst = d_features + idx * in_c;
    d_group[idx] = v;
    const float mx = d_mean[v * 3 + 0], my = d_mean[v * 3 + 1], mz = d_mean[v * 3 + 2];
    const int32_t gx = coords[v * 4 + 3], gy = coords[v * 4 + 2], gz = coords[v * 4 + 1];
    const float cx = gx * vx + x0 + vx * 0.5f, cy = gy * vy + y0 + vy * 0.5f, cz = gz * vz + z0 + vz * 0.5f;
    int oc = 0;
    if (use_abs) { for (int c = 0; c < point_dim; c++) dst[oc++] = valid ? src[c] : 0.0f; }
    else         { for (int c = 3; c < point_dim; c++) dst[oc++] = valid ? src[c] : 0.0f; }
    dst[oc++] = valid ? (src[0] - mx) : 0.0f;
    dst[oc++] = valid ? (src[1] - my) : 0.0f;
    dst[oc++] = valid ? (src[2] - mz) : 0.0f;
    dst[oc++] = valid ? (src[0] - cx) : 0.0f;
    dst[oc++] = valid ? (src[1] - cy) : 0.0f;
    dst[oc++] = valid ? (src[2] - cz) : 0.0f;
    if (with_distance)
        dst[oc++] = valid ? sqrtf(src[0]*src[0] + src[1]*src[1] + src[2]*src[2]) : 0.0f;
}

} // namespace

bool gpu_pillar_vfe_scatter(const float * points, int32_t n_points, int32_t point_dim,
                            int nx, int ny, int nz,
                            float vx, float vy, float vz,
                            float x0, float y0, float z0,
                            bool use_abs, bool with_distance,
                            const std::vector<gpu_pfn_layer> & layers, int C_bev,
                            std::vector<int32_t> & pillar_coords,
                            std::vector<float> & pillar_features,
                            std::vector<float> & spatial_whc,
                            std::string & error,
                            float * spatial_dst_dev,
                            bool need_pillar_features) {
    (void) nz;
    bool vfe_dbg = getenv("OPCPD_VFEDBG") != nullptr;
    auto vfe_t0 = std::chrono::high_resolution_clock::now();
    // With D2D the BEV is written straight into the ggml device tensor, so the
    // (potentially ~tens of MB) host zero-fill is unnecessary.
    if (!spatial_dst_dev) spatial_whc.assign((size_t) nx * ny * C_bev, 0.0f);
    pillar_coords.clear();
    pillar_features.clear();
    if (!points || n_points <= 0 || point_dim < 3 || layers.empty() || C_bev <= 0) {
        error = "bad gpu vfe input";
        return true; // empty output, no failure
    }

    // ---- host voxelization + grouping (cheap) --------------------------------
    struct entry { int32_t source, gx, gy, group; };
    std::vector<entry> entries;
    std::vector<int64_t> keys;
    entries.reserve(n_points);
    keys.reserve(n_points);
    for (int32_t p = 0; p < n_points; p++) {
        const float * v = points + (int64_t) p * point_dim;
        const int gx = (int) std::floor((v[0] - x0) / vx);
        const int gy = (int) std::floor((v[1] - y0) / vy);
        if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) continue;
        entries.push_back({p, gx, gy, -1});
        keys.push_back((int64_t) gx * ny + gy);
    }
    const int32_t N2 = (int32_t) entries.size();
    if (N2 == 0) return true;
    std::vector<int64_t> unique_keys = keys;
    std::sort(unique_keys.begin(), unique_keys.end());
    unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());
    const int32_t V = (int32_t) unique_keys.size();
    // Assign group ids in O(N2 log N2) by sorting point indices by key and
    // walking in order (instead of O(N2 log V) per-point lower_bound, which
    // had poor cache locality).
    {
        std::vector<int32_t> order((size_t) N2);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return keys[(size_t) a] < keys[(size_t) b]; });
        entries[order[0]].group = 0;
        for (int i = 1; i < N2; i++) {
            const int prev = order[i - 1], cur = order[i];
            entries[cur].group = entries[prev].group + (keys[(size_t) cur] != keys[(size_t) prev]);
        }
    }
    pillar_coords.reserve((size_t) V * 4);
    std::vector<int32_t> pillar_gx(V), pillar_gy(V);
    for (int32_t i = 0; i < V; i++) {
        const int64_t k = unique_keys[i];
        const int32_t gx = (int32_t)(k / ny);
        const int32_t gy = (int32_t)(k % ny);
        pillar_coords.insert(pillar_coords.end(), {0, 0, gy, gx});
        pillar_gx[i] = gx; pillar_gy[i] = gy;
    }

    // means + counts
    std::vector<float> means((size_t) V * 3, 0.0f);
    std::vector<int32_t> counts(V, 0);
    for (const entry & e : entries) {
        const float * pt = points + (int64_t) e.source * point_dim;
        float * m = means.data() + (size_t) e.group * 3;
        m[0] += pt[0]; m[1] += pt[1]; m[2] += pt[2];
        counts[e.group]++;
    }
    for (int32_t g = 0; g < V; g++) {
        const float inv = 1.0f / counts[g];
        means[g * 3 + 0] *= inv; means[g * 3 + 1] *= inv; means[g * 3 + 2] *= inv;
    }

    // per-point input features + group id
    const int input_channels = (use_abs ? point_dim : point_dim - 3) + 6 + (with_distance ? 1 : 0);
    const float xo = vx * 0.5f + x0, yo = vy * 0.5f + y0, zo = vz * 0.5f + z0;
    std::vector<float> features((size_t) N2 * input_channels, 0.0f);
    std::vector<int32_t> group_id(N2);
    for (int64_t r = 0; r < N2; r++) {
        const entry & e = entries[r];
        const float * pt = points + (int64_t) e.source * point_dim;
        const float * m = means.data() + (size_t) e.group * 3;
        float * dst = features.data() + r * input_channels;
        int ch = 0;
        if (use_abs) { for (int s = 0; s < point_dim; s++) dst[ch++] = pt[s]; }
        else         { for (int s = 3; s < point_dim; s++) dst[ch++] = pt[s]; }
        dst[ch++] = pt[0] - m[0];
        dst[ch++] = pt[1] - m[1];
        dst[ch++] = pt[2] - m[2];
        dst[ch++] = pt[0] - (e.gx * vx + xo);
        dst[ch++] = pt[1] - (e.gy * vy + yo);
        dst[ch++] = pt[2] - zo;
        if (with_distance) dst[ch++] = std::sqrt(pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2]);
        group_id[r] = e.group;
    }

    // ---- upload to GPU and run MLP + maxpool + scatter ------------------------
    auto vfe_t_host = std::chrono::high_resolution_clock::now();
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { error = "cudaGetDevice failed"; return false; }
    const int threads = 256;

    // Persistent buffers: reused across forward calls to avoid per-call
    // cudaMalloc. thread_local so concurrent model contexts never share the
    // scratch (AICore can host several openpcdet sessions).
    static thread_local device_buf d_features, d_group, d_local, d_agg, d_pf, d_gx, d_gy, d_spatial;
    if (!d_features.alloc(N2, input_channels * sizeof(float)) ||
        !d_group.alloc(N2, sizeof(int32_t)) ||
        !d_agg.alloc(V, C_bev * sizeof(float)) ||
        !d_pf.alloc(V, C_bev * sizeof(float)) ||
        !d_gx.alloc(V, sizeof(int32_t)) || !d_gy.alloc(V, sizeof(int32_t)) ||
        !d_spatial.alloc((size_t) nx * ny * C_bev, sizeof(float))) {
        error = "cudaMalloc failed"; return false;
    }
    // AICore deviation from upstream (fix): the canvas is only written at the
    // current frame's pillar cells, so a frame sequence whose pillar layout
    // changes would otherwise scatter over stale BEV data retained in the
    // persistent buffer. Zeroing once per frame keeps the D2D copy exact.
    cudaMemsetAsync(d_spatial.ptr, 0, (size_t) nx * ny * C_bev * sizeof(float), S);
    cudaMemcpyAsync(d_features.as<float>(), features.data(), (size_t) N2 * input_channels * sizeof(float), cudaMemcpyHostToDevice, S);
    cudaMemcpyAsync(d_group.as<int32_t>(), group_id.data(), (size_t) N2 * sizeof(int32_t), cudaMemcpyHostToDevice, S);
    cudaMemcpyAsync(d_gx.as<int32_t>(), pillar_gx.data(), (size_t) V * sizeof(int32_t), cudaMemcpyHostToDevice, S);
    cudaMemcpyAsync(d_gy.as<int32_t>(), pillar_gy.data(), (size_t) V * sizeof(int32_t), cudaMemcpyHostToDevice, S);

    // Layer 0 input is the constructed features.
    static device_buf d_in;      // concat buffer used from layer 1 onward
    const float * cur_ptr = d_features.as<float>();
    int cur_in = input_channels;

    for (size_t li = 0; li < layers.size(); li++) {
        const gpu_pfn_layer & L = layers[li];
        const int out = L.out_channels;
        if (L.linear_w.size() != (size_t) out * cur_in) { error = "vfe weight dim mismatch"; return false; }

        if (!d_local.alloc(N2, out * sizeof(float))) { error = "cudaMalloc failed"; return false; }
        // linear_w is [out][cur_in] row-major; upload per layer.
        static device_buf dW, dB, dS, dBB;
        if (!dW.alloc(out, cur_in * sizeof(float))) { error = "cudaMalloc failed"; return false; }
        upload_const(dW, L.linear_w.data(), (size_t) out * cur_in * sizeof(float));
        const float * hB = L.linear_b.empty() ? nullptr : L.linear_b.data();
        const float * hS = L.bn_scale.empty() ? nullptr : L.bn_scale.data();
        const float * hBB = L.bn_bias.empty() ? nullptr : L.bn_bias.data();
        if (hB && !dB.alloc(out, sizeof(float))) { error = "cudaMalloc failed"; return false; }
        if (hS && !dS.alloc(out, sizeof(float))) { error = "cudaMalloc failed"; return false; }
        if (hBB && !dBB.alloc(out, sizeof(float))) { error = "cudaMalloc failed"; return false; }
        if (hB)  upload_const(dB, hB, (size_t) out * sizeof(float));
        if (hS)  upload_const(dS, hS, (size_t) out * sizeof(float));
        if (hBB) upload_const(dBB, hBB, (size_t) out * sizeof(float));

        const int nlin = N2 * out;
        k_linear_relu<<<(nlin + threads - 1) / threads, threads, 0, S>>>(
            cur_ptr, dW.as<float>(), hB ? dB.as<float>() : nullptr,
            hS ? dS.as<float>() : nullptr, hBB ? dBB.as<float>() : nullptr,
            d_local.as<float>(), N2, cur_in, out);

        // maxpool over pillars (init -1e30)
        k_fill<<<(V * out + threads - 1) / threads, threads, 0, S>>>(d_agg.as<float>(), -1e30f, V * out);
        k_maxpool<<<(nlin + threads - 1) / threads, threads, 0, S>>>(
            d_local.as<float>(), d_group.as<int32_t>(), d_agg.as<float>(), N2, out);

        if (L.last_layer) {
            cudaMemcpyAsync(d_pf.as<float>(), d_agg.as<float>(), (size_t) V * out * sizeof(float), cudaMemcpyDeviceToDevice, S);
            break;
        }
        // next input = concat(local[point], agg[group]) -> N2 * (2*out)
        if (!d_in.alloc(N2, 2 * out * sizeof(float))) { error = "cudaMalloc failed"; return false; }
        k_concat<<<(nlin + threads - 1) / threads, threads, 0, S>>>(
            d_local.as<float>(), d_agg.as<float>(), d_group.as<int32_t>(),
            d_in.as<float>(), N2, out);
        cur_ptr = d_in.as<float>();
        cur_in = 2 * out;
    }

    // scatter -> BEV [W,H,C]
    const int ns = V * C_bev;
    k_scatter<<<(ns + threads - 1) / threads, threads, 0, S>>>(
        d_pf.as<float>(), d_gx.as<int32_t>(), d_gy.as<int32_t>(),
        d_spatial.as<float>(), V, C_bev, nx, ny);

    // Copy back only the host buffers the caller will actually read. The
    // steady CUDA path writes BEV directly into the ggml input tensor, so the
    // per-pillar features are only needed for validation taps.
    pillar_features.resize((size_t) V * C_bev);
    if (need_pillar_features) {
        cudaMemcpyAsync(pillar_features.data(), d_pf.as<float>(),
                        (size_t) V * C_bev * sizeof(float), cudaMemcpyDeviceToHost, S);
    }
    if (spatial_dst_dev) {
        cudaMemcpyAsync(spatial_dst_dev, d_spatial.as<float>(),
                        (size_t) nx * ny * C_bev * sizeof(float), cudaMemcpyDeviceToDevice, S);
    } else {
        cudaMemcpyAsync(spatial_whc.data(), d_spatial.as<float>(), (size_t) nx * ny * C_bev * sizeof(float), cudaMemcpyDeviceToHost, S);
    }
    if (cudaStreamSynchronize(S) != cudaSuccess) { error = "cuda sync failed"; return false; }
    if (vfe_dbg) {
        auto vfe_t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(vfe_t1 - vfe_t0).count();
        double host = std::chrono::duration<double, std::milli>(vfe_t_host - vfe_t0).count();
        fprintf(stderr, "[vfedbg] dyn total=%.2fms host=%.2fms gpu=%.2fms N2=%d V=%d grid=%dx%dx%d Cbev=%d D2D=%d\n",
                ms, host, ms - host, N2, V, nx, ny, nz, C_bev, spatial_dst_dev != nullptr);
    }
    return true;
}

bool gpu_pillar_vfe_scatter_vox(const float * voxels, const int32_t * num_points,
                                const int32_t * coords, int32_t V, int32_t P, int32_t point_dim,
                                int nx, int ny,
                                float vx, float vy, float vz,
                                float x0, float y0, float z0,
                                bool use_abs, bool with_distance,
                                const std::vector<gpu_pfn_layer> & layers, int C_bev,
                                std::vector<float> & pillar_features,
                                std::vector<float> & spatial_whc,
                                std::string & error,
                                float * spatial_dst_dev,
                                bool need_pillar_features) {
    // With D2D the BEV is written straight into the ggml device tensor, so the
    // (potentially ~tens of MB) host zero-fill is unnecessary.
    if (!spatial_dst_dev) spatial_whc.assign((size_t) nx * ny * C_bev, 0.0f);
    pillar_features.clear();
    if (V <= 0 || P <= 0 || layers.empty() || C_bev <= 0) {
        error = "bad gpu vox vfe input";
        return true;
    }

    const int in_c = (use_abs ? point_dim : point_dim - 3) + 6 + (with_distance ? 1 : 0);
    const int64_t slots = (int64_t) V * P;
    const int32_t N2 = (int32_t) slots;

    // ---- GPU MLP + maxpool + scatter ------------------------------------------
    const int threads = 256;
    // thread_local: concurrent model contexts never share the scratch.
    static thread_local device_buf d_voxels, d_npoints, d_coords, d_mean,
                      d_features, d_group, d_local, d_agg, d_pf, d_gx, d_gy, d_spatial;
    if (!d_voxels.alloc(slots, point_dim * sizeof(float)) ||
        !d_npoints.alloc(V, sizeof(int32_t)) ||
        !d_coords.alloc(V, 4 * sizeof(int32_t)) ||
        !d_mean.alloc(V, 3 * sizeof(float)) ||
        !d_features.alloc(N2, in_c * sizeof(float)) ||
        !d_group.alloc(N2, sizeof(int32_t)) ||
        !d_agg.alloc(V, C_bev * sizeof(float)) ||
        !d_pf.alloc(V, C_bev * sizeof(float)) ||
        !d_gx.alloc(V, sizeof(int32_t)) || !d_gy.alloc(V, sizeof(int32_t)) ||
        !d_spatial.alloc((size_t) nx * ny * C_bev, sizeof(float))) {
        error = "cudaMalloc failed"; return false;
    }
    // Same staleness fix as the dynamic variant above.
    cudaMemsetAsync(d_spatial.ptr, 0, (size_t) nx * ny * C_bev * sizeof(float), S);
    cudaMemcpyAsync(d_voxels.as<float>(), voxels, (size_t) slots * point_dim * sizeof(float), cudaMemcpyHostToDevice, S);
    cudaMemcpyAsync(d_npoints.as<int32_t>(), num_points, (size_t) V * sizeof(int32_t), cudaMemcpyHostToDevice, S);
    cudaMemcpyAsync(d_coords.as<int32_t>(), coords, (size_t) V * 4 * sizeof(int32_t), cudaMemcpyHostToDevice, S);
    const int nb = (V + threads - 1) / threads;
    k_vox_mean<<<nb, threads, 0, S>>>(d_voxels.as<float>(), d_npoints.as<int32_t>(), d_coords.as<int32_t>(),
                                      d_mean.as<float>(), d_gx.as<int32_t>(), d_gy.as<int32_t>(),
                                      V, P, point_dim);
    const int ne = (N2 + threads - 1) / threads;
    k_expand_vox<<<ne, threads, 0, S>>>(d_voxels.as<float>(), d_npoints.as<int32_t>(), d_coords.as<int32_t>(),
                                        d_mean.as<float>(), d_features.as<float>(), d_group.as<int32_t>(),
                                        V, P, point_dim, use_abs, with_distance, vx, vy, vz, x0, y0, z0, in_c);

    static device_buf d_in_buf;
    const float * cur_ptr = d_features.as<float>();
    int cur_in = in_c;
    for (size_t li = 0; li < layers.size(); li++) {
        const gpu_pfn_layer & L = layers[li];
        const int out = L.out_channels;
        if (L.linear_w.size() != (size_t) out * cur_in) { error = "vfe weight dim mismatch"; return false; }
        if (!d_local.alloc(N2, out * sizeof(float))) { error = "cudaMalloc failed"; return false; }
        static device_buf dW, dB, dS, dBB;
        if (!dW.alloc(out, cur_in * sizeof(float))) { error = "cudaMalloc failed"; return false; }
        upload_const(dW, L.linear_w.data(), (size_t) out * cur_in * sizeof(float));
        const float * hB = L.linear_b.empty() ? nullptr : L.linear_b.data();
        const float * hS = L.bn_scale.empty() ? nullptr : L.bn_scale.data();
        const float * hBB = L.bn_bias.empty() ? nullptr : L.bn_bias.data();
        if (hB && !dB.alloc(out, sizeof(float))) { error = "cudaMalloc failed"; return false; }
        if (hS && !dS.alloc(out, sizeof(float))) { error = "cudaMalloc failed"; return false; }
        if (hBB && !dBB.alloc(out, sizeof(float))) { error = "cudaMalloc failed"; return false; }
        if (hB)  upload_const(dB, hB, (size_t) out * sizeof(float));
        if (hS)  upload_const(dS, hS, (size_t) out * sizeof(float));
        if (hBB) upload_const(dBB, hBB, (size_t) out * sizeof(float));

        const int nlin = N2 * out;
        k_linear_relu<<<(nlin + threads - 1) / threads, threads, 0, S>>>(
            cur_ptr, dW.as<float>(), hB ? dB.as<float>() : nullptr,
            hS ? dS.as<float>() : nullptr, hBB ? dBB.as<float>() : nullptr,
            d_local.as<float>(), N2, cur_in, out);
        k_fill<<<(V * out + threads - 1) / threads, threads, 0, S>>>(d_agg.as<float>(), -1e30f, V * out);
        k_maxpool<<<(nlin + threads - 1) / threads, threads, 0, S>>>(
            d_local.as<float>(), d_group.as<int32_t>(), d_agg.as<float>(), N2, out);
        if (L.last_layer) {
            cudaMemcpyAsync(d_pf.as<float>(), d_agg.as<float>(), (size_t) V * out * sizeof(float), cudaMemcpyDeviceToDevice, S);
            break;
        }
        if (!d_in_buf.alloc(N2, 2 * out * sizeof(float))) { error = "cudaMalloc failed"; return false; }
        k_concat<<<(nlin + threads - 1) / threads, threads, 0, S>>>(
            d_local.as<float>(), d_agg.as<float>(), d_group.as<int32_t>(),
            d_in_buf.as<float>(), N2, out);
        cur_ptr = d_in_buf.as<float>();
        cur_in = 2 * out;
    }

    const int ns = V * C_bev;
    k_scatter<<<(ns + threads - 1) / threads, threads, 0, S>>>(
        d_pf.as<float>(), d_gx.as<int32_t>(), d_gy.as<int32_t>(),
        d_spatial.as<float>(), V, C_bev, nx, ny);

    pillar_features.resize((size_t) V * C_bev);
    if (need_pillar_features) {
        cudaMemcpyAsync(pillar_features.data(), d_pf.as<float>(),
                        (size_t) V * C_bev * sizeof(float), cudaMemcpyDeviceToHost, S);
    }
    if (spatial_dst_dev) {
        cudaMemcpyAsync(spatial_dst_dev, d_spatial.as<float>(),
                        (size_t) nx * ny * C_bev * sizeof(float), cudaMemcpyDeviceToDevice, S);
    } else {
        cudaMemcpyAsync(spatial_whc.data(), d_spatial.as<float>(), (size_t) nx * ny * C_bev * sizeof(float), cudaMemcpyDeviceToHost, S);
    }
    if (cudaStreamSynchronize(S) != cudaSuccess) { error = "cuda sync failed"; return false; }
    return true;
}

} // namespace openpcdet::ext
