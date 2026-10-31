// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Implementation of the geometry-prediction (OSGDecoder) as a ggml graph.
//
// Two parts:
//  1. Host triplane sampling: the fixed bilinear grid_sample that maps grid
//     vertices onto the 3 planes (no weights, not a ggml graph). This is the
//     memory-heavy step (sampled [3C, M] where M = (grid_res+1)^3).
//  2. The OSGDecoder MLPs (sdf / deformation / weight) as a single ggml graph
//     scheduled on the selected backend, so those weights run on CPU/CUDA/
//     Vulkan identically.
#include "synthesizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "aicore/runtime_capi.h"
#include "numerics.hpp"
#ifdef IM_USE_OPENMP
#include <omp.h>
#endif

#include "common/ggml_backend_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

namespace instantmesh {

namespace {

void check_canceled(ggml_backend_t backend) {
    if (!aicore_cancel_requested()) return;
    ggml_backend_synchronize(backend);
    throw std::runtime_error("instantmesh: synthesizer canceled");
}

ggml_tensor* get_t(const SynthesizerModel& m, const char* name) {
    std::string key = std::string("decoder.") + name;
    auto it = m.gguf.tensors.find(key);
    if (it == m.gguf.tensors.end()) {
        std::fprintf(stderr, "synthesizer: missing tensor '%s'\n", key.c_str());
        std::abort();
    }
    return it->second;
}

ggml_tensor* linear(ggml_context* ctx,
                    ggml_tensor* x,
                    ggml_tensor* W,
                    ggml_tensor* b,
                    const char* node_name) {
    ggml_tensor* y = mul_mat_exact(ctx, W, x);
    if (node_name) ggml_set_name(y, node_name);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// One OSGDecoder MLP: Linear(3C->64) ReLU (64->64) ReLU ... ReLU
// Linear(64->out). Weight/bias indexed 2*i (net_sdf.0/.2/.4/.6). Input x is
// [fin, n], output [fout, n].
ggml_tensor* mlp(ggml_context* ctx,
                 const SynthesizerModel& m,
                 ggml_tensor* x,
                 const char* tag) {
    const SynthesizerHparams& hp = m.hp;
    char name[128];
    for (int i = 0; i < hp.num_layers; ++i) {
        std::snprintf(name, sizeof(name), "%s.%d.weight", tag, 2 * i);
        ggml_tensor* W = get_t(m, name);
        // The corrected weight-branch reference retains F32 input; quantized
        // GEMM also quantizes activations, which the strict hint alone cannot
        // prevent. Dequantize published weights only in this small MLP graph;
        // storage precision and other branches/tasks remain unchanged.
        if (std::strcmp(tag, "net_weight") == 0 && W->type != GGML_TYPE_F32)
            W = ggml_cast(ctx, W, GGML_TYPE_F32);
        std::snprintf(name, sizeof(name), "%s.%d.bias", tag, 2 * i);
        ggml_tensor* b = get_t(m, name);
        // Single-row F16 SDF retains frozen GEMM dispatch; swapped-operand
        // matvec changes reduction order and exceeds the original compatibility
        // gate.
        const char* node_name = i == hp.num_layers - 1 &&
                                                W->type == GGML_TYPE_F16 &&
                                                std::strcmp(tag, "net_sdf") == 0
                                        ? "instantmesh_sdf_matmul_"
                                        : nullptr;
        x = linear(ctx, x, W, b, node_name);
        if (i < hp.num_layers - 1) x = ggml_relu(ctx, x);
    }
    return x;
}

// Bilinear grid sample (align_corners=False, padding zeros) of one plane.
// plane: [C, H, W] contiguous; nx,ny normalized in [-1,1]. Writes out[c].
// Maps nx -> px = (nx+1)/2*W - 0.5, then 4-neighbour weighted sum.
static void sample_plane(const float* plane,
                         int C,
                         int H,
                         int W,
                         float nx,
                         float ny,
                         float* out) {
    const float px = (nx + 1.0f) * 0.5f * W - 0.5f;
    const float py = (ny + 1.0f) * 0.5f * H - 0.5f;
    const int x0 = (int)std::floor(px);
    const int y0 = (int)std::floor(py);
    const float fx = px - (float)x0;
    const float fy = py - (float)y0;
    // weights of the 4 neighbours (x0/x0+1, y0/y0+1).
    const float w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy,
                        fx * fy};
    const int xs[4] = {x0, x0 + 1, x0, x0 + 1};
    const int ys[4] = {y0, y0, y0 + 1, y0 + 1};
    for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        for (int k = 0; k < 4; ++k) {
            if (xs[k] >= 0 && xs[k] < W && ys[k] >= 0 && ys[k] < H) {
                acc += w[k] * plane[c * H * W + ys[k] * W + xs[k]];
            }
        }
        out[c] = acc;
    }
}

// Host triplane sampling. planes [N,3,C,H,W]; verts [M,3]; writes sampled
// [N, 3C, M] (row = plane*C + channel). For N>1 the same verts are used for
// every batch element.
void sample_triplanes(const SynthesizerModel& m,
                      const float* planes,
                      int N,
                      int H,
                      int W,
                      const float* verts,
                      int M,
                      float* sampled) {
    const int C = m.hp.plane_dim;
    const int plane_stride = C * H * W;
    const int total = N * M;
#ifdef IM_USE_OPENMP
#pragma omp parallel for schedule(static) \
        num_threads(std::max(1, omp_get_max_threads()))
#endif
    for (int it = 0; it < total; ++it) {
        const int n = it / M;
        const int v = it % M;
        float tmp[3 * 128];  // plane_dim <= 128
        const float* pb = planes + (size_t)n * 3 * plane_stride;
        const float* p = verts + 3 * v;
        const float x = p[0], y = p[1], z = p[2];
        // plane 0 : (x,y) ; plane 1 : (x,z) ; plane 2 : (z,y)
        sample_plane(pb + 0 * plane_stride, C, H, W, x, y, &tmp[0 * C]);
        sample_plane(pb + 1 * plane_stride, C, H, W, x, z, &tmp[1 * C]);
        sample_plane(pb + 2 * plane_stride, C, H, W, z, y, &tmp[2 * C]);
        std::memcpy(sampled + (size_t)it * (3 * C), tmp,
                    (size_t)3 * C * sizeof(float));
    }
}

}  // namespace

bool synthesizer_load(const std::string& path,
                      ggml_backend_t backend,
                      SynthesizerModel& out,
                      std::string* error) {
    out.gguf.unload();
    out.backend = backend;
    const auto fail = [&](const std::string& message) {
        if (error) *error = message;
        out.gguf.unload();
        out.backend = nullptr;
        return false;
    };
    if (!load_gguf(path, backend, out.gguf, error)) {
        out.gguf.unload();
        out.backend = nullptr;
        return false;
    }
    const gguf_context* g = out.gguf.gguf;
    // Missing metadata retains published model defaults; reject wrong types or
    // truncated integers when present.
    const auto read_dimension = [&](const char* key, int fallback, int& value) {
        const int64_t id = gguf_find_key(g, key);
        if (id < 0) {
            value = fallback;
            return true;
        }
        int64_t raw = 0;
        switch (gguf_get_kv_type(g, id)) {
            case GGUF_TYPE_INT32:
                raw = gguf_get_val_i32(g, id);
                break;
            case GGUF_TYPE_UINT32:
                raw = gguf_get_val_u32(g, id);
                break;
            case GGUF_TYPE_INT64:
                raw = gguf_get_val_i64(g, id);
                break;
            case GGUF_TYPE_UINT64: {
                const uint64_t unsigned_raw = gguf_get_val_u64(g, id);
                if (unsigned_raw > uint64_t(std::numeric_limits<int>::max()))
                    return false;
                raw = int64_t(unsigned_raw);
                break;
            }
            default:
                return false;
        }
        if (raw <= 0 || raw > std::numeric_limits<int>::max()) return false;
        value = int(raw);
        return true;
    };
    if (!read_dimension("synthesizer.plane_dim", 80, out.hp.plane_dim) ||
        !read_dimension("synthesizer.hidden", 64, out.hp.hidden) ||
        !read_dimension("synthesizer.num_layers", 4, out.hp.num_layers) ||
        out.hp.plane_dim > 128 || out.hp.num_layers < 2 ||
        size_t(out.hp.num_layers) > out.gguf.tensors.size() / 8) {
        return fail("synthesizer: invalid hparams");
    }
    // Validate all four branches (including RGB); reject missing tensors and
    // mismatched MLP dimensions before graph construction.
    for (const char* branch :
         {"net_sdf", "net_deformation", "net_weight", "net_rgb"}) {
        const bool weight_branch = std::strcmp(branch, "net_weight") == 0;
        const int output = weight_branch                         ? 21
                           : std::strcmp(branch, "net_sdf") == 0 ? 1
                                                                 : 3;
        for (int layer = 0; layer < out.hp.num_layers; ++layer) {
            const int64_t input =
                    layer == 0 ? (weight_branch ? 24 : 3) * out.hp.plane_dim
                               : out.hp.hidden;
            const int64_t rows =
                    layer == out.hp.num_layers - 1 ? output : out.hp.hidden;
            const std::string prefix = std::string("decoder.") + branch + "." +
                                       std::to_string(2 * layer);
            for (bool bias : {false, true}) {
                const std::string name = prefix + (bias ? ".bias" : ".weight");
                const auto found = out.gguf.tensors.find(name);
                if (found == out.gguf.tensors.end())
                    return fail("synthesizer: missing tensor '" + name + "'");
                const ggml_tensor* tensor = found->second;
                if (tensor->ne[0] != (bias ? rows : input) ||
                    tensor->ne[1] != (bias ? 1 : rows) || tensor->ne[2] != 1 ||
                    tensor->ne[3] != 1) {
                    return fail("synthesizer: invalid tensor shape '" + name +
                                "'");
                }
            }
        }
    }
    return true;
}

void synthesizer_forward(const SynthesizerModel& model,
                         const float* planes,
                         int N,
                         int H,
                         int W,
                         const float* verts,
                         int M,
                         const int32_t* cubes,
                         int n_cubes,
                         float** outsdf,
                         float** outdeform,
                         float** outweight,
                         const std::function<void(int, int)>& progress) {
    *outsdf = *outdeform = *outweight = nullptr;
    check_canceled(model.backend);
    const SynthesizerHparams& hp = model.hp;
    const int F = 3 * hp.plane_dim;  // per-vertex feature width
    const int GF = 8 * F;            // per-cube feature width

    // ---- host geometry (fixed ops, no weights) ---------------------------
    std::vector<float> sampled((size_t)N * M * F);
    sample_triplanes(model, planes, N, H, W, verts, M, sampled.data());

    size_t ns = (size_t)N * M;
    size_t nd = (size_t)N * M * 3;
    size_t nw = (size_t)N * n_cubes * 21;
    std::unique_ptr<float, decltype(&std::free)> S(
            static_cast<float*>(malloc(ns * sizeof(float))), std::free);
    std::unique_ptr<float, decltype(&std::free)> D(
            static_cast<float*>(malloc(nd * sizeof(float))), std::free);
    std::unique_ptr<float, decltype(&std::free)> Wt(
            static_cast<float*>(malloc(nw * sizeof(float))), std::free);
    if (!S || !D || !Wt) throw std::bad_alloc();
    check_canceled(model.backend);

    // ---- ggml graph for the OSGDecoder MLPs ------------------------------
    // net_sdf / net_deformation run once on the sampled triplane features
    // [F, M*N]. F32 activations: the FlexiCubes SDF/deformation/weight outputs
    // are too precision-sensitive for F16 (quantizing them to F16 corrupts the
    // extracted mesh — it balloons from ~39k to ~1.8M faces), so geometry MLPs
    // stay F32.
    ggml_init_params iparams = {/*mem_size=*/16u << 20, nullptr,
                                /*no_alloc=*/true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_owner(
            ggml_init(iparams), ggml_free);
    if (!ctx_owner) throw std::bad_alloc();
    ggml_context* ctx = ctx_owner.get();

    ggml_tensor* sampled_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, M * N);
    ggml_set_input(sampled_t);

    ggml_tensor* sdf = mlp(ctx, model, sampled_t, "net_sdf");  // [1, M*N]
    ggml_tensor* deformation =
            mlp(ctx, model, sampled_t, "net_deformation");  // [3, M*N]

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc_owner(
            ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(model.backend)),
            ggml_gallocr_free);
    if (!alloc_owner) throw std::bad_alloc();
    ggml_gallocr_t alloc = alloc_owner.get();
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sdf);
    ggml_build_forward_expand(gf, deformation);
    preserve_cpu_matmul_dispatch(gf, model.backend);
    if (!ggml_gallocr_alloc_graph(alloc, gf)) throw std::bad_alloc();

    ggml_backend_tensor_set(sampled_t, sampled.data(), 0,
                            sampled.size() * sizeof(float));
    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error(
                "instantmesh: synthesizer graph compute failed");
    check_canceled(model.backend);
    ggml_backend_tensor_get(sdf, S.get(), 0, ns * sizeof(float));
    ggml_backend_tensor_get(deformation, D.get(), 0, nd * sizeof(float));

    alloc_owner.reset();
    ctx_owner.reset();

    // ---- net_weight (FlexiCubes): grid [8F, n_cubes] built on the GPU with
    // ggml_get_rows(sampled, cubes) -> [F, 8*nc] then a corner-major
    // reshape/cont into [8F, n_cubes]. This keeps the multi-GB
    // intermediate on the GPU (avoids a ~3s CPU gather + multi-GB host->device
    // copy). Because that intermediate is large, the cubes are processed in
    // chunks to bound GPU memory. ----
    // Chunk size trades graph-setup overhead (favours large chunks on GPU)
    // against peak intermediate memory (get_rows + cont buffers). Larger chunks
    // cut the per-chunk ggml context/gallocr/graph overhead ~2x (1<<16 ->
    // 1<<18), but on the CPU backend big contiguous gathers are slower, so keep
    // a small chunk there.
    const int chunk =
            ggml_common::is_cpu_backend(model.backend) ? (1 << 16) : (1 << 18);
    for (int n = 0; n < N; ++n) {
        const int32_t off = n * M;
        for (int base = 0; base < n_cubes; base += chunk) {
            check_canceled(model.backend);
            const int cn = std::min(chunk, n_cubes - base);
            std::vector<int32_t> idx((size_t)8 * cn);
            for (int c = 0; c < cn; ++c)
                for (int v = 0; v < 8; ++v)
                    idx[c * 8 + v] = cubes[(base + c) * 8 + v] + off;

            ggml_init_params ip2 = {/*mem_size=*/16u << 20, nullptr,
                                    /*no_alloc=*/true};
            std::unique_ptr<ggml_context, decltype(&ggml_free)> c2_owner(
                    ggml_init(ip2), ggml_free);
            if (!c2_owner) throw std::bad_alloc();
            ggml_context* c2 = c2_owner.get();
            ggml_tensor* st = ggml_new_tensor_2d(c2, GGML_TYPE_F32, F, M * N);
            ggml_set_input(st);
            ggml_tensor* b =
                    ggml_new_tensor_1d(c2, GGML_TYPE_I32, (int64_t)8 * cn);
            ggml_set_input(b);
            ggml_tensor* g = ggml_get_rows(c2, st, b);  // [F, 8*cn]
            g = ggml_reshape_3d(c2, g, F, 8, cn);       // [F,8,cn]
            // get_rows is already corner-major with F varying fastest; do not
            // swap feature/corner axes.
            g = ggml_cont_2d(c2, g, GF, cn);                   // [8F,cn]
            ggml_tensor* w = mlp(c2, model, g, "net_weight");  // [21, cn]
            w = ggml_scale(c2, w, 0.1f);

            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>
                    a2_owner(ggml_gallocr_new(
                                     ggml_backend_get_default_buffer_type(
                                             model.backend)),
                             ggml_gallocr_free);
            if (!a2_owner) throw std::bad_alloc();
            ggml_gallocr_t a2 = a2_owner.get();
            ggml_cgraph* g2 = ggml_new_graph(c2);
            ggml_build_forward_expand(g2, w);
            if (!ggml_gallocr_alloc_graph(a2, g2)) throw std::bad_alloc();

            ggml_backend_tensor_set(st, sampled.data(), 0,
                                    sampled.size() * sizeof(float));
            ggml_backend_tensor_set(b, idx.data(), 0,
                                    idx.size() * sizeof(int32_t));
            if (ggml_backend_graph_compute(model.backend, g2) !=
                GGML_STATUS_SUCCESS)
                throw std::runtime_error(
                        "instantmesh: weight graph compute failed");
            check_canceled(model.backend);

            float* wcol = Wt.get() + ((size_t)n * n_cubes + base) * 21;
            ggml_backend_tensor_get(w, wcol, 0,
                                    (size_t)21 * cn * sizeof(float));

            a2_owner.reset();
            c2_owner.reset();
            if (progress) progress(n * n_cubes + base + cn, N * n_cubes);
            check_canceled(model.backend);
        }
    }

    *outsdf = S.release();
    *outdeform = D.release();
    *outweight = Wt.release();
}

bool synthesizer_texture_forward(const SynthesizerModel& model,
                                 const float* planes,
                                 int N,
                                 int H,
                                 int W,
                                 const float* points,
                                 int M,
                                 float** rgb) {
    *rgb = nullptr;
    check_canceled(model.backend);
    const SynthesizerHparams& hp = model.hp;
    const int F = 3 * hp.plane_dim;

    // Sample the triplane at the query points, then run net_rgb.
    std::vector<float> sampled((size_t)N * M * F);
    sample_triplanes(model, planes, N, H, W, points, M, sampled.data());

    ggml_init_params iparams = {/*mem_size=*/8u << 20, nullptr,
                                /*no_alloc=*/true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_owner(
            ggml_init(iparams), ggml_free);
    if (!ctx_owner) throw std::bad_alloc();
    ggml_context* ctx = ctx_owner.get();

    ggml_tensor* sampled_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, M * N);
    ggml_set_input(sampled_t);

    ggml_tensor* rgb_t = mlp(ctx, model, sampled_t, "net_rgb");  // [3, M*N]
    rgb_t = ggml_sigmoid(ctx, rgb_t);  // affine clamp applied on host

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc_owner(
            ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(model.backend)),
            ggml_gallocr_free);
    if (!alloc_owner) throw std::bad_alloc();
    ggml_gallocr_t alloc = alloc_owner.get();
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rgb_t);
    preserve_cpu_matmul_dispatch(gf, model.backend);
    if (!ggml_gallocr_alloc_graph(alloc, gf)) throw std::bad_alloc();

    ggml_backend_tensor_set(sampled_t, sampled.data(), 0,
                            sampled.size() * sizeof(float));
    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error(
                "instantmesh: synthesizer graph compute failed");
    check_canceled(model.backend);

    size_t n = (size_t)N * M * 3;
    std::unique_ptr<float, decltype(&std::free)> R(
            static_cast<float*>(malloc(n * sizeof(float))), std::free);
    if (!R) throw std::bad_alloc();
    ggml_backend_tensor_get(rgb_t, R.get(), 0, n * sizeof(float));

    *rgb = R.release();
    return true;
}

}  // namespace instantmesh