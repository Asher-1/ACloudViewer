// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// End-to-end pipeline session: condition embedders -> SS flow sampling ->
// occupancy -> coords -> SLat flow sampling -> Gaussian decode -> PLY.
//
// All stage graphs are the parity-verified ones (dino_graph / pointpatch_graph
// / ss_flow_graph / ss_decoder_graph / slat_flow_graph / gs_decoder_graph).
// The sampling loops live host-side (Euler integration + CFG blending); the
// graphs are built once per stage and re-run with fresh inputs.
//
// Inputs (first milestone): the preprocessed condition tensors dumped by
// scripts/dump_e2e_stages.py (ss_input_*.samt). Sampling uses the
// PyTorch-compatible Philox stream; CPU/Vulkan receive the recorded CUDA
// distribution-block contract. --noise-dir replays the torch reference noise.
#include "asset_io.hpp"
#include "dino_graph.hpp"
#include "dtype_contract.hpp"
#include "e2e_options.hpp"
#include "flexicubes.hpp"
#include "gs_decoder_graph.hpp"
#include "image_preprocess.hpp"
#include "mesh_decoder_runner.hpp"
#include "moge_inference.hpp"
#include "pointpatch_graph.hpp"
#include "pose_decoder.hpp"
#include "pytorch_philox_rng.hpp"
#include "slat_flow_graph.hpp"
#include "sparse_ops.hpp"
#include "ss_decoder_graph.hpp"
#include "ss_flow_graph.hpp"
#if defined(SAM3D_USE_CUDA)
#include "pytorch_cuda_rng.hpp"
#endif
#include <fstream>

#include "backend.hpp"
#include "common.hpp"
#include "common/ggml_backend_registry.hpp"
#include "gguf_loader.hpp"
#include "graph_builder.hpp"
#include "sam3dggml.h"
#if defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
#include "gaussian_renderer.hpp"
#include "native_pbr_pipeline.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
/* MSVC provides no strncasecmp (POSIX); _strnicmp is the equivalent
 * case-insensitive n-char compare. Defined after all includes so the
 * replacement never rewrites declarations inside vendored headers. */
#define strncasecmp _strnicmp
#endif

namespace sam3d {

namespace {

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

// CUDA-family match for the per-stage specializations (F16 autocast, the
// CUDA spconv bridge, shared-backend eligibility). Keys off the *family*,
// not the raw request string: "cuda[:N]" and the resolved registry names
// ("CUDA0") are CUDA, and on the SAM3D_USE_CUDA build "auto" resolves to
// the registered CUDA device exactly like the PytorchCudaNormalRng branch
// below. A "cuda" request that falls back to CPU keeps the generic path
// only when the runtime resolved a non-CUDA backend.
static bool be_is_cuda(const char* be) {
    if (be == nullptr) return false;
    if (strncasecmp(be, "cuda", 4) == 0) return true;
#if defined(SAM3D_USE_CUDA)
    return std::strcmp(be, "auto") == 0;
#else
    return false;
#endif
}

struct RawImg {  // HWC f32, ne = [C, W, H] (dino graph convention)
    std::vector<float> data;
    int64_t w = 0, h = 0, c = 0;
};

// torch (1, C, H, W) SAMT dump (ne = [W, H, C], CHW memory) -> HWC buffer
RawImg chw_to_hwc(const RawTensor& t) {
    const int64_t C = t.ne[2], H = t.ne[1], W = t.ne[0];
    GGML_ASSERT(static_cast<size_t>(C * H * W) ==
                t.data.size() / sizeof(float));
    const float* src = (const float*)t.data.data();
    RawImg out;
    out.c = C;
    out.h = H;
    out.w = W;
    out.data.resize((size_t)C * H * W);
    for (int64_t y = 0; y < H; y++)
        for (int64_t x = 0; x < W; x++)
            for (int64_t c = 0; c < C; c++)
                out.data[(y * W + x) * C + c] = src[c * H * W + y * W + x];
    return out;
}

// torch (1, C, H, W) SAMT dump kept in CHW order (pointpatch graph convention)
std::vector<float> as_chw(const RawTensor& t) {
    return std::vector<float>((const float*)t.data.data(),
                              (const float*)t.data.data() + t.data.size() / 4);
}

struct Stage {
    std::unique_ptr<Backend> owned_backend;
    // Points at owned_backend, or at a caller-owned shared backend when the
    // session reuses one backend across requests. Never null while open.
    Backend* backend = nullptr;
    std::unique_ptr<GGUFModel> model;

    bool open(const std::string& gguf,
              const char* be_name,
              int nthreads,
              const char* profile_label = nullptr,
              Backend* shared_backend = nullptr) {
        if (shared_backend) {
            backend = shared_backend;
        } else {
            owned_backend = Backend::create(be_name, nthreads);
            backend = owned_backend.get();
        }
        if (!backend) return false;
        backend->set_profile_label(profile_label ? profile_label : "unlabeled");
        model = std::make_unique<GGUFModel>();
        return model->load(gguf, backend->weights_buffer_type());
    }
    void close() {
        // Frees the GPU weights regardless of ownership; the shared backend
        // itself stays alive for the next stage/request.
        model.reset();
        owned_backend.reset();
        backend = nullptr;
    }
};

struct EulerStep {
    float t = 0.0f;   // model time in [0, 1]
    float dt = 0.0f;  // integration interval to the next model time
};

// Mirrors FlowMatching._prepare_t and ODESolver.solve_iter in the shipped
// Python pipeline. The model receives t * 1000, while Euler integrates over
// the rescaled [0, 1] interval.
std::vector<EulerStep> make_euler_schedule(int steps, float rescale_t) {
    GGML_ASSERT(steps > 0);
    std::vector<float> times(static_cast<size_t>(steps) + 1);
    for (int i = 0; i <= steps; ++i) {
        const double u = static_cast<double>(i) / steps;
        const double t = rescale_t != 0.0f
                                 ? u / (1.0 + (static_cast<double>(rescale_t) -
                                               1.0) * (1.0 - u))
                                 : u;
        times[static_cast<size_t>(i)] = static_cast<float>(t);
    }
    std::vector<EulerStep> schedule;
    schedule.reserve(static_cast<size_t>(steps));
    for (int i = 0; i < steps; ++i) {
        schedule.push_back({times[static_cast<size_t>(i)],
                            times[static_cast<size_t>(i + 1)] -
                                    times[static_cast<size_t>(i)]});
    }
    return schedule;
}

using SparseCoordinate = std::array<int32_t, 4>;

// Coordinate identity must retain all four signed 32-bit components. A packed
// uint64_t key aliases values outside an int16 domain, including ordinary
// negative sparse-grid coordinates. The hash only selects buckets; array
// equality remains the uniqueness contract.
struct SparseCoordinateHash {
    size_t operator()(const SparseCoordinate& coordinate) const noexcept {
        uint64_t hash = 1469598103934665603ULL;
        for (int32_t value : coordinate) {
            hash ^= static_cast<uint32_t>(value);
            hash *= 1099511628211ULL;
        }
        return static_cast<size_t>(hash);
    }
};

SparseCoordinate sparse_coordinate(int32_t b, int32_t x, int32_t y, int32_t z) {
    return {b, x, y, z};
}

uint64_t sparse_coordinate_seed(const SparseCoordinate& coordinate) {
    return static_cast<uint64_t>(SparseCoordinateHash{}(coordinate));
}

// Mirrors prune_sparse_structure(..., max_neighbor_axes_dist=1). The source
// occupancy is binary and coordinates are unique, so testing the 3^3
// neighbourhood directly is identical to the Python conv3d count while
// preserving torch.argwhere's input order.
std::vector<int32_t> prune_surface_coords(const std::vector<int32_t>& coords,
                                          int max_neighbor_axes_dist = 1) {
    GGML_ASSERT(coords.size() % 4 == 0);
    std::unordered_set<SparseCoordinate, SparseCoordinateHash> occupied;
    occupied.reserve(coords.size() / 2);
    for (size_t i = 0; i < coords.size(); i += 4) {
        occupied.insert(sparse_coordinate(coords[i], coords[i + 1],
                                          coords[i + 2], coords[i + 3]));
    }

    std::vector<int32_t> surface;
    surface.reserve(coords.size());
    for (size_t i = 0; i < coords.size(); i += 4) {
        const int32_t b = coords[i];
        bool is_surface = false;
        for (int dx = -max_neighbor_axes_dist;
             dx <= max_neighbor_axes_dist && !is_surface; ++dx) {
            for (int dy = -max_neighbor_axes_dist;
                 dy <= max_neighbor_axes_dist && !is_surface; ++dy) {
                for (int dz = -max_neighbor_axes_dist;
                     dz <= max_neighbor_axes_dist; ++dz) {
                    if (occupied.find(sparse_coordinate(
                                b, coords[i + 1] + dx, coords[i + 2] + dy,
                                coords[i + 3] + dz)) == occupied.end()) {
                        is_surface = true;
                        break;
                    }
                }
            }
        }
        if (is_surface)
            surface.insert(surface.end(),
                           coords.begin() + static_cast<ptrdiff_t>(i),
                           coords.begin() + static_cast<ptrdiff_t>(i + 4));
    }
    return surface;
}

}  // namespace

// Match sam3d_objects.pipeline.inference_utils.downsample_sparse_structure.
// The sparse DiT uses int32 coordinate tables, so keeping this policy in the
// C++ runtime is required for large (dense) occupancy fields.
std::vector<int32_t> downsample_sparse_coords_pytorch(
        const std::vector<int32_t>& coords,
        PytorchPhiloxNormalRng& rng,
        std::string& error,
        bool& randomly_subsampled,
        int64_t max_coords,
        int downsample_factor) {
    randomly_subsampled = false;
    const int64_t n = static_cast<int64_t>(coords.size() / 4);
    if (n <= max_coords) return coords;
    if (n == 0) return {};

    std::array<int32_t, 3> lo{coords[1], coords[2], coords[3]};
    std::array<int32_t, 3> hi = lo;
    for (int64_t i = 1; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            const int32_t v = coords[static_cast<size_t>(i) * 4 + c + 1];
            lo[c] = std::min(lo[c], v);
            hi[c] = std::max(hi[c], v);
        }
    }
    // The Python pipeline casts to float before each arithmetic operation.
    // Preserve F32 rounding, including torch.round's tie-to-even behavior.
    std::array<float, 3> original{}, target{}, target_min{}, target_max{};
    std::array<int32_t, 3> target_min_i{}, target_max_i{};
    for (int c = 0; c < 3; ++c) {
        original[c] = static_cast<float>(hi[c] - lo[c] + 1);
        target[c] = original[c] / static_cast<float>(downsample_factor);
        const float offset = (original[c] - target[c]) * 0.5f;
        target_min[c] = static_cast<float>(lo[c]) + offset;
        target_max[c] = target_min[c] + target[c] - 1.0f;
        // torch.Tensor.int() truncates toward zero (rather than floor/ceil).
        target_min_i[c] = static_cast<int32_t>(target_min[c]);
        target_max_i[c] = static_cast<int32_t>(target_max[c]);
    }

    std::vector<int32_t> unique;
    unique.reserve(static_cast<size_t>(n));
    std::unordered_set<SparseCoordinate, SparseCoordinateHash> seen;
    seen.reserve(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t* p = &coords[static_cast<size_t>(i) * 4];
        int32_t out[4] = {p[0], 0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            const float denom = static_cast<float>(hi[c] - lo[c]);
            const float normalized = denom > 0.0f
                                             ? (static_cast<float>(p[c + 1]) -
                                                static_cast<float>(lo[c])) /
                                                       denom
                                             : 0.0f;
            const float value = normalized * (target[c] - 1.0f) + target_min[c];
            // torch.round uses round-to-nearest with ties to even, unlike
            // std::round and floor(x + 0.5) for negative/halfway values.
            const float floor_value = std::floor(value);
            const float fractional = value - floor_value;
            int64_t rounded = static_cast<int64_t>(floor_value);
            if (fractional > 0.5f ||
                (fractional == 0.5f && (rounded & 1LL) != 0)) {
                ++rounded;
            }
            rounded = std::max<int64_t>(rounded, target_min_i[c]);
            rounded = std::min<int64_t>(rounded, target_max_i[c]);
            out[c + 1] = static_cast<int32_t>(rounded);
        }
        const SparseCoordinate k =
                sparse_coordinate(out[0], out[1], out[2], out[3]);
        if (seen.insert(k).second) unique.insert(unique.end(), out, out + 4);
    }
    const size_t unique_n = unique.size() / 4;
    // torch.unique(..., dim=0) returns rows in lexicographic order.  Insertion
    // order is observably different when randperm selects a subset.
    std::vector<size_t> lexicographic_order(unique_n);
    std::iota(lexicographic_order.begin(), lexicographic_order.end(), 0U);
    std::sort(lexicographic_order.begin(), lexicographic_order.end(),
              [&unique](size_t left, size_t right) {
                  for (size_t component = 0; component < 4; ++component) {
                      const int32_t a = unique[left * 4 + component];
                      const int32_t b = unique[right * 4 + component];
                      if (a != b) return a < b;
                  }
                  return false;
              });
    std::vector<int32_t> sorted_unique;
    sorted_unique.reserve(unique.size());
    for (size_t index : lexicographic_order) {
        const int32_t* point = unique.data() + index * 4;
        sorted_unique.insert(sorted_unique.end(), point, point + 4);
    }
    unique.swap(sorted_unique);
    if (unique_n > static_cast<size_t>(max_coords)) {
        std::vector<uint32_t> order;
        if (!rng.randperm(unique_n, order, error)) return {};
        std::vector<int32_t> sampled;
        sampled.reserve(static_cast<size_t>(max_coords) * 4);
        for (size_t j = 0; j < static_cast<size_t>(max_coords); ++j) {
            const int32_t* p = &unique[order[j] * 4];
            sampled.insert(sampled.end(), p, p + 4);
        }
        randomly_subsampled = true;
        return sampled;
    }
    return unique;
}

namespace {

// batched DINO forwards: ONE graph per call, several (prefix, image) pairs
// sharing the same weights model - avoids the multi-alloc CUDA graph-capture
// trouble of running many small graphs on one backend.
struct DinoDbg {
    bool enabled = false;
    std::string debug_stage;
    std::string out_path;
};

std::vector<std::vector<float>> run_dino_batch(
        Stage& st,
        const std::vector<std::pair<std::string, const RawImg*>>& jobs,
        bool prenorm,
        int64_t& out_n,
        const DinoDbg* dbg = nullptr,
        bool manual_attention = false) {
    GraphContext gctx;
    struct Job {
        DinoGraph dg;
        ggml_tensor* in;
        ggml_tensor* out;
    };
    // dbg: build/run only the first DINO job and write its debug_stage
    // boundary tensor to out_path for parity bisects (mixed-size debug
    // graphs cannot share one batched allocation).
    const bool dino_dbg = dbg && dbg->enabled;
    const char* dino_dbg_out = dbg ? dbg->out_path.c_str() : nullptr;
    std::vector<Job> js;
    for (auto& [prefix, img] : jobs) {
        if (dino_dbg && !js.empty()) break;
        js.emplace_back();
        js.back().dg.g = &gctx;
        js.back().dg.m = st.model.get();
        js.back().dg.prefix = prefix;
        js.back().dg.prenorm = prenorm;
        // The condition DINO contract is the explicit F32 attention form; the
        // flash path (F16 K/V) measurably perturbs the outlier channels and
        // collapses the SS occupancy decision downstream.
        js.back().dg.manual_attention = manual_attention;
        // optional per-boundary parity dumps, same contract as the other
        // condition-chain graphs
        if (dbg && !dbg->debug_stage.empty())
            js.back().dg.debug_stage = dbg->debug_stage;
        js.back().in = gctx.input_f32("image", {img->c, img->w, img->h});
        js.back().dg.inputs.push_back(js.back().in);
        js.back().dg.table_data.push_back(nullptr);
        js.back().out = js.back().dg.build(js.back().in);
    }
    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 32768, false);
    for (auto& j : js) ggml_build_forward_expand(graph, j.out);
    record_dtype_contract_graph(
            prenorm ? "slat_condition_dino" : "ss_condition_dino", graph);
    if (!st.backend->alloc(graph)) {
        out_n = -1;
        return {};
    }
    // upload everything (images + build-internal tables)
    for (size_t k = 0; k < js.size(); k++) {
        const RawImg* img = jobs[k].second;
        if (!st.backend->set_input_f32(js[k].in, img->data.data(),
                                       (size_t)img->c * img->w * img->h)) {
            out_n = -1;
            return {};
        }
    }
    for (auto& j : js) {
        for (size_t ti = 0; ti < j.dg.inputs.size(); ti++) {
            ggml_tensor* t = j.dg.inputs[ti];
            if (!t->buffer || !j.dg.table_data[ti]) continue;
            const auto& host = j.dg.table_data[ti];
            bool ok = t->type == GGML_TYPE_F32
                              ? st.backend->set_input_f32(
                                        t, (const float*)host->data(),
                                        host->size())
                              : st.backend->set_input_i32(t, host->data(),
                                                          host->size());
            if (!ok) {
                out_n = -1;
                return {};
            }
        }
    }
    if (!st.backend->run(graph)) {
        out_n = -1;
        return {};
    }
    std::vector<std::vector<float>> outs(js.size());
    for (size_t k = 0; k < js.size(); k++)
        st.backend->get_tensor_f32(js[k].out, outs[k]);
    out_n = js[0].out->ne[1];
    if (dino_dbg && dino_dbg_out) {
        const auto& t = js[0].out;
        std::ofstream df(dino_dbg_out, std::ios::binary);
        df.write(reinterpret_cast<const char*>(outs[0].data()),
                 std::streamsize(outs[0].size() * sizeof(float)));
        df.close();
        LOGI("dino dbg: wrote %s raw f32 (ne %lldx%lld, %zu floats)",
             dino_dbg_out, (long long)t->ne[0], (long long)t->ne[1],
             outs[0].size());
    }
    return outs;
}

// batched PointPatch forwards: one graph, one run per input pair
std::vector<std::vector<float>> run_pointpatch_batch(
        Stage& st,
        const std::string& prefix,
        const std::vector<std::pair<const std::vector<float>*,
                                    std::pair<int64_t, int64_t>>>& jobs) {
    GraphContext gctx;
    struct Job {
        PointPatchGraph pp;
        ggml_tensor* in;
        ggml_tensor* out;
    };
    std::vector<Job> js;
    for (auto& [pm, wh] : jobs) {
        js.emplace_back();
        js.back().pp.g = &gctx;
        js.back().pp.m = st.model.get();
        js.back().pp.prefix = prefix;
        js.back().in = gctx.input_f32("pointmap", {wh.first, wh.second, 3});
        js.back().pp.inputs.push_back(js.back().in);
        js.back().pp.table_data.push_back(nullptr);
        js.back().out = js.back().pp.build(js.back().in);
    }
    ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 32768, false);
    for (auto& j : js) ggml_build_forward_expand(graph, j.out);
    record_dtype_contract_graph("ss_condition_pointpatch", graph);
    if (!st.backend->alloc(graph)) return {};
    const int64_t n_px = jobs[0].second.first * jobs[0].second.second;
    std::vector<float> ones(n_px, 1.0f), zeros(n_px, 0.0f);
    for (size_t k = 0; k < js.size(); k++) {
        if (!st.backend->set_input_f32(js[k].in, jobs[k].first->data(),
                                       jobs[k].first->size()))
            return {};
        for (ggml_tensor* mk : js[k].pp.mask_inputs) {
            if (!mk->buffer) continue;
            const bool is_valid = !strcmp(mk->name, "pp_valid");
            if (!st.backend->set_input_f32(
                        mk, is_valid ? ones.data() : zeros.data(),
                        (size_t)ggml_nelements(mk)))
                return {};
        }
    }
    // upload the build-internal gather tables (resize indices, cls indices)
    for (auto& j : js) {
        for (size_t ti = 0; ti < j.pp.inputs.size(); ti++) {
            ggml_tensor* t = j.pp.inputs[ti];
            if (!t->buffer || !j.pp.table_data[ti]) continue;
            const auto& host = j.pp.table_data[ti];
            bool ok = t->type == GGML_TYPE_F32
                              ? st.backend->set_input_f32(
                                        t, (const float*)host->data(),
                                        host->size())
                              : st.backend->set_input_i32(t, host->data(),
                                                          host->size());
            if (!ok) return {};
        }
    }
    if (!st.backend->run(graph)) return {};
    std::vector<std::vector<float>> outs(js.size());
    for (size_t k = 0; k < js.size(); k++)
        st.backend->get_tensor_f32(js[k].out, outs[k]);
    return outs;
}

// EmbedderFuser in-graph: per segment LayerNorm -> SwiGLU -> + idx_emb, concat.
// Segments may share a projection net (same embedder, different inputs).
struct FuserSeg {
    int64_t in_ch;  // token channel count (1024 dino / 512 pointpatch)
    int64_t n;
    int embedder;  // proj prefix index (cemb.emb{E})
    int pos_idx;   // idx_emb column
    std::vector<float> host;
    ggml_tensor* tokens = nullptr;  // created inside build_fuser
};

// builds the fuser subgraph; returns the concat output and fills upload list
ggml_tensor* build_fuser(GraphContext& gctx,
                         GGUFModel& m,
                         std::vector<FuserSeg>& segs,
                         const std::string& cemb_prefix) {
    ggml_context* ctx = gctx.ctx();
    auto as_f32 = [&](ggml_tensor* t) {
        return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
    };
    auto matmul_weight = [&](ggml_tensor* t) {
        return ggml_is_quantized(t->type) ? t : as_f32(t);
    };
    ggml_tensor* idx_emb =
            as_f32(m.get(cemb_prefix + ".idx_emb"));  // ne = [C, n_groups]
    const int64_t C = idx_emb->ne[0];
    ggml_tensor* cat = nullptr;
    for (auto& s : segs) {
        const std::string p = cemb_prefix + ".emb" + std::to_string(s.embedder);
        ggml_tensor* x = gctx.input_f32("fuse_seg", {s.in_ch, s.n});
        s.tokens = x;
        ggml_tensor* h =
                gb_layer_norm(ctx, x, as_f32(m.get(p + ".proj_ln.weight")),
                              as_f32(m.get(p + ".proj_ln.bias")), 1e-5f);
        ggml_tensor* g1 = gb_linear(
                ctx, matmul_weight(m.get(p + ".proj.w1.weight")), nullptr, h);
        ggml_tensor* g3 = gb_linear(
                ctx, matmul_weight(m.get(p + ".proj.w3.weight")), nullptr, h);
        ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g1), g3);
        ggml_tensor* o = gb_linear(
                ctx, matmul_weight(m.get(p + ".proj.w2.weight")), nullptr, act);
        ggml_tensor* pe =
                ggml_view_1d(ctx, idx_emb, C, s.pos_idx * C * sizeof(float));
        o = ggml_add(ctx, o, pe);  // (C,1) broadcasts over (C,n)
        cat = cat ? ggml_concat(ctx, cat, o, 1) : o;
    }
    return cat;
}

// load condition inputs dumped by dump_e2e_stages.py
struct CondInputs {
    RawImg image, rgb_image, mask3, rgb_image_mask3;  // HWC (dino)
    std::vector<float> pointmap, rgb_pointmap;        // CHW (pointpatch)
    int64_t W = 518, H = 518;
};

// 1-channel masks: the Dino wrapper repeats them to 3 channels
void repeat_mask_channels_to_3(RawImg& m) {
    if (m.c != 1) return;
    std::vector<float> m3(m.data.size() * 3);
    for (size_t i = 0; i < m.data.size(); i++)
        for (int c = 0; c < 3; c++) m3[i * 3 + c] = m.data[i];
    m.c = 3;
    m.data = std::move(m3);
}

bool load_cond_inputs(const std::string& dir, CondInputs& ci) {
    auto load = [&](const char* name, RawTensor& t) {
        if (!load_raw_tensor(dir + "/" + name, t)) {
            LOGE("e2e: missing %s/%s", dir.c_str(), name);
            return false;
        }
        return true;
    };
    RawTensor image, rgb_image, mask, rgb_image_mask, pointmap, rgb_pointmap;
    if (!load("ss_input_image.samt", image)) return false;
    if (!load("ss_input_rgb_image.samt", rgb_image)) return false;
    if (!load("ss_input_mask.samt", mask)) return false;
    if (!load("ss_input_rgb_image_mask.samt", rgb_image_mask)) return false;
    if (!load("ss_input_pointmap.samt", pointmap)) return false;
    if (!load("ss_input_rgb_pointmap.samt", rgb_pointmap)) return false;

    ci.image = chw_to_hwc(image);
    ci.rgb_image = chw_to_hwc(rgb_image);
    ci.W = ci.image.w;
    ci.H = ci.image.h;
    ci.mask3 = chw_to_hwc(mask);
    ci.rgb_image_mask3 = chw_to_hwc(rgb_image_mask);
    repeat_mask_channels_to_3(ci.mask3);
    repeat_mask_channels_to_3(ci.rgb_image_mask3);
    ci.pointmap = as_chw(pointmap);
    ci.rgb_pointmap = as_chw(rgb_pointmap);
    return true;
}

// In-memory equivalent of load_cond_inputs(): the file path stores the CHW
// f32 payloads verbatim (ne = [W, H, C, 1]) and reads them back through
// chw_to_hwc/as_chw; this performs the identical index transforms on the
// native condition struct, so the full-pipeline numerics are bit-exact
// without the disk round-trip.
bool cond_inputs_from_native(const NativeConditionInputs& in, CondInputs& ci) {
    const int64_t W = in.width;
    const int64_t H = in.height;
    if (W <= 0 || H <= 0) {
        LOGE("e2e: invalid in-memory condition dimensions %lldx%lld",
             (long long)W, (long long)H);
        return false;
    }
    const size_t image_elems = static_cast<size_t>(W) * H * 3;
    const size_t mask_elems = static_cast<size_t>(W) * H;
    if (in.image.size() != image_elems || in.rgb_image.size() != image_elems ||
        in.pointmap.size() != image_elems ||
        in.rgb_pointmap.size() != image_elems || in.mask.size() != mask_elems ||
        in.rgb_image_mask.size() != mask_elems) {
        LOGE("e2e: in-memory condition payload sizes mismatch %lldx%lld",
             (long long)W, (long long)H);
        return false;
    }
    auto to_hwc = [&](const std::vector<float>& chw, int64_t C) {
        RawImg out;
        out.c = C;
        out.h = H;
        out.w = W;
        out.data.resize(static_cast<size_t>(C) * H * W);
        const float* src = chw.data();
        for (int64_t y = 0; y < H; y++)
            for (int64_t x = 0; x < W; x++)
                for (int64_t c = 0; c < C; c++)
                    out.data[(y * W + x) * C + c] = src[c * H * W + y * W + x];
        return out;
    };
    ci.image = to_hwc(in.image, 3);
    ci.rgb_image = to_hwc(in.rgb_image, 3);
    ci.W = W;
    ci.H = H;
    ci.mask3 = to_hwc(in.mask, 1);
    ci.rgb_image_mask3 = to_hwc(in.rgb_image_mask, 1);
    repeat_mask_channels_to_3(ci.mask3);
    repeat_mask_channels_to_3(ci.rgb_image_mask3);
    ci.pointmap = in.pointmap;  // CHW passthrough (bit-exact)
    ci.rgb_pointmap = in.rgb_pointmap;
    return true;
}

// Cross-backend parity sampling requires the reference PyTorch CUDA
// distribution-block count. The source runtime read it from an environment
// variable; AICore tasks must not use environment control, so the caller
// passes it through the explicit philox_blocks option instead.
bool pytorch_philox_distribution_blocks_from_environment(uint32_t& blocks,
                                                         std::string& error) {
    (void)blocks;
    error = "philox_blocks option required: CPU/Vulkan native-seed sampling "
            "needs the reference PyTorch CUDA distribution-block count; set "
            "aicore_sam3d_options_set_philox_blocks() explicitly";
    return false;
}

bool load_noise_opt(const std::string& dir,
                    const char* name,
                    std::vector<float>& v,
                    int64_t expect_elems) {
    if (dir.empty()) return false;
    RawTensor t;
    if (!load_raw_tensor(dir + "/" + name, t)) return false;
    const size_t n = t.data.size() / 4;
    if ((int64_t)n != expect_elems) {
        LOGE("noise %s: expected %lld elems, got %zu", name,
             (long long)expect_elems, n);
        return false;
    }
    v.assign((const float*)t.data.data(), (const float*)t.data.data() + n);
    return true;
}

bool load_f32_vector(const std::string& path,
                     size_t expected_elements,
                     std::array<float, 3>& output) {
    RawTensor tensor;
    if (!load_raw_tensor(path, tensor) || tensor.type != GGML_TYPE_F32 ||
        tensor.data.size() != expected_elements * sizeof(float)) {
        LOGE("e2e: expected F32[%zu] tensor %s", expected_elements,
             path.c_str());
        return false;
    }
    const float* values = reinterpret_cast<const float*>(tensor.data.data());
    for (size_t index = 0; index < expected_elements; ++index)
        output[index] = values[index];
    return true;
}

// Replaying an official SLat noise tensor by position, rather than by its
// compact token index, makes an SS-boundary change reproducible. Most target
// cells retain the exact official initial state; cells introduced by the
// backend's occupancy decision receive a stable coordinate-derived state.
bool load_slat_noise_for_coords(const std::string& dir,
                                const std::vector<int32_t>& target_coords,
                                unsigned seed,
                                std::vector<float>& target_noise) {
    if (dir.empty() || target_coords.size() % 4 != 0) return false;
    RawTensor ref_noise;
    RawTensor ref_coords;
    if (!load_raw_tensor(dir + "/slat_x0.samt", ref_noise) ||
        !load_raw_tensor(dir + "/slat_coords.samt", ref_coords) ||
        ref_noise.type != GGML_TYPE_F32 || ref_coords.type != GGML_TYPE_I32) {
        return false;
    }
    const size_t n_ref = ref_noise.data.size() / (8 * sizeof(float));
    if (n_ref == 0 || ref_noise.data.size() != n_ref * 8 * sizeof(float) ||
        ref_coords.data.size() != n_ref * 4 * sizeof(int32_t)) {
        LOGE("slat noise replay: malformed reference support");
        return false;
    }

    const int32_t* ref_coord_data =
            reinterpret_cast<const int32_t*>(ref_coords.data.data());
    const float* ref_noise_data =
            reinterpret_cast<const float*>(ref_noise.data.data());
    std::unordered_map<SparseCoordinate, size_t, SparseCoordinateHash>
            reference_index;
    reference_index.reserve(n_ref * 2);
    for (size_t i = 0; i < n_ref; ++i) {
        const int32_t* coord = ref_coord_data + 4 * i;
        reference_index.emplace(
                sparse_coordinate(coord[0], coord[1], coord[2], coord[3]), i);
    }

    const size_t n_target = target_coords.size() / 4;
    target_noise.resize(n_target * 8);
    size_t replayed = 0;
    for (size_t i = 0; i < n_target; ++i) {
        const int32_t* coord = target_coords.data() + 4 * i;
        const SparseCoordinate key =
                sparse_coordinate(coord[0], coord[1], coord[2], coord[3]);
        const auto it = reference_index.find(key);
        float* output = target_noise.data() + 8 * i;
        if (it != reference_index.end()) {
            std::memcpy(output, ref_noise_data + 8 * it->second,
                        8 * sizeof(float));
            ++replayed;
            continue;
        }
        // This branch is diagnostic only: it supplies a stable state for a
        // coordinate absent from the official replay support. It is never a
        // claim that the added coordinate exists in PyTorch. Philox removes
        // the former host-library-dependent random distribution.
        const uint64_t key_seed = sparse_coordinate_seed(key);
        const uint64_t local_seed =
                key_seed ^ (key_seed >> 32U) ^ static_cast<uint64_t>(seed);
        PytorchPhiloxNormalRng local_rng(local_seed, 1);
        std::vector<float> local_noise(8);
        std::string local_error;
        if (!local_rng.fill(local_noise, local_error)) {
            LOGE("slat noise replay: coordinate Philox fallback failed: %s",
                 local_error.c_str());
            return false;
        }
        std::memcpy(output, local_noise.data(), 8 * sizeof(float));
    }
    LOGI("slat noise replay: %zu/%zu cells matched official support", replayed,
         n_target);
    return true;
}

// --- SLat denorm constants (checkpoints/hf/pipeline.yaml) -----------------
// The inference_utils defaults describe a different checkpoint family. The
// shipped pipeline overrides both vectors, and these are therefore part of
// the model contract rather than generic runtime defaults.
constexpr float SLAT_STD[8] = {2.37326008f, 2.13174402f, 2.24139530f,
                               2.30589401f, 2.11918940f, 1.89695110f,
                               2.41684989f, 2.08374642f};
constexpr float SLAT_MEAN[8] = {0.12211431f,  0.37204156f,  -1.26521907f,
                                -2.05276058f, -3.10432536f, -0.11294304f,
                                -0.85146744f, 0.45506954f};

// Gaussian decoder representation constants (slat_decoder_gs.yaml)
constexpr int GS_NGAUSS = 32;
constexpr float GS_VOXEL_SIZE = 1.5f;
constexpr float GS_SCALING_BIAS = 0.004f;
constexpr float GS_OPACITY_BIAS = 0.1f;
constexpr float GS_MIN_KERNEL = 0.0009f;

inline float softplus(float x) { return x > 20.f ? x : logf(1.f + expf(x)); }
inline float softplus_inv(float y) {
    return y + logf(1.f - expf(-y));
}  // y > 0
inline float inv_sigmoid(float p) { return logf(p / (1.f - p)); }

// write the official 3DGS PLY (matches gaussian_model.save_ply: sh_degree 0)
bool write_gaussian_ply(const std::string& path,
                        int64_t n_gs,
                        const std::vector<float>& xyz,
                        const std::vector<float>& fdc,
                        const std::vector<float>& opacity,
                        const std::vector<float>& scaling,
                        const std::vector<float>& rotation) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    char head[1024];
    snprintf(head, sizeof(head),
             "ply\nformat binary_little_endian 1.0\ncomment NeoGaussian\n"
             "element vertex %lld\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property float nx\nproperty float ny\nproperty float nz\n"
             "property float f_dc_0\nproperty float f_dc_1\nproperty float "
             "f_dc_2\n"
             "property float opacity\n"
             "property float scale_0\nproperty float scale_1\nproperty float "
             "scale_2\n"
             "property float rot_0\nproperty float rot_1\nproperty float "
             "rot_2\nproperty float rot_3\n"
             "end_header\n",
             (long long)n_gs);
    fwrite(head, 1, strlen(head), f);
    std::vector<float> row(3 + 3 + 3 + 1 + 3 + 4);
    const float scale_bias = softplus_inv(GS_SCALING_BIAS);
    const float op_bias = inv_sigmoid(GS_OPACITY_BIAS);
    for (int64_t i = 0; i < n_gs; i++) {
        // PLY xyz = _xyz * aabb[3:] + aabb[:3] with aabb = [-.5,-.5,-.5,1,1,1]
        row[0] = xyz[i * 3 + 0] - 0.5f;
        row[1] = xyz[i * 3 + 1] - 0.5f;
        row[2] = xyz[i * 3 + 2] - 0.5f;
        row[3] = row[4] = row[5] = 0.f;
        for (int k = 0; k < 3; k++) row[6 + k] = fdc[i * 3 + k];
        // scale = log(sqrt(softplus(s + softplus_inv(bias))^2 + min_kernel^2))
        for (int k = 0; k < 3; k++) {
            float sp = softplus(scaling[i * 3 + k] + scale_bias);
            row[10 + k] = logf(sqrtf(sp * sp + GS_MIN_KERNEL * GS_MIN_KERNEL));
        }
        // opacity = inverse_sigmoid(sigmoid(op + inverse_sigmoid(bias))) = op +
        // bias
        row[9] = opacity[i] + op_bias;
        // rotation = raw * lr + rots_bias [1,0,0,0] (unnormalized in the PLY)
        row[13] = rotation[i * 4 + 0] + 1.f;
        row[14] = rotation[i * 4 + 1];
        row[15] = rotation[i * 4 + 2];
        row[16] = rotation[i * 4 + 3];
        fwrite(row.data(), sizeof(float), row.size(), f);
    }
    fclose(f);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// e2e command
// ---------------------------------------------------------------------------

// Every e2e knob is an explicit field: the CLI fills this struct from argv
// flags, and the pipeline reads no process environment at all. This keeps
// the stage contracts discoverable, unit-testable and free of the leaked-
// environment failure mode.

int cmd_e2e(const E2eOptions& opt) {
    const std::string& models_dir = opt.models_dir;
    const std::string& cond_dir = opt.cond_dir;
    const std::string& noise_dir = opt.noise_dir;
    const std::string& dbg_dir = opt.dbg_dir;
    const std::string& out_ply = opt.out_ply;
    const std::string& out_pbr = opt.out_pbr;
    const std::string& out_mesh_vertices = opt.out_mesh_vertices;
    const std::string& out_mesh_faces = opt.out_mesh_faces;
    const std::string& out_pose = opt.out_pose;
    const std::string& out_dtype_contract = opt.out_dtype_contract;
    const char* be_name = opt.backend.c_str();
    const std::string dt = opt.dtype;
    const std::string ss_dt = opt.ss_dtype.empty() ? dt : opt.ss_dtype;
    // The occupancy decoder determines the sparse support consumed by every
    // later stage. Keep it independently selectable so Q4 flow experiments
    // can retain this small, threshold-sensitive model without promoting the
    // multi-gigabyte SS transformer.
    const std::string ss_decoder_dt =
            opt.ss_decoder_dtype.empty() ? ss_dt : opt.ss_decoder_dtype;
    // Keep the structured-latent generator independently selectable for
    // quantization localization. The default preserves the homogeneous E2E
    // model family selected by `dtype`.
    const std::string slat_dt = opt.slat_dtype.empty() ? dt : opt.slat_dtype;
    const std::string gs_dt = opt.gs_dtype.empty() ? dt : opt.gs_dtype;
    const std::string mesh_dt = opt.mesh_dtype.empty() ? dt : opt.mesh_dtype;
    const unsigned seed = opt.seed;
    const int nthreads = opt.threads;
    const auto valid_dtype = [](const std::string& dtype) {
        return dtype == "f32" || dtype == "f16" || dtype == "q4_0" ||
               dtype == "q4_1" || dtype == "q4_k" || dtype == "q8_0";
    };
    if (!valid_dtype(dt) || !valid_dtype(ss_dt) ||
        !valid_dtype(ss_decoder_dt) || !valid_dtype(slat_dt) ||
        !valid_dtype(mesh_dt)) {
        LOGE("e2e: unsupported model dtype: base=%s ss=%s ss_decoder=%s "
             "slat=%s mesh=%s",
             dt.c_str(), ss_dt.c_str(), ss_decoder_dt.c_str(), slat_dt.c_str(),
             mesh_dt.c_str());
        return 1;
    }
    // The decoder graph accepts every model dtype accepted above. CUDA and
    // Vulkan both have native Q4_1 matrix and copy support; rejecting the
    // exported Q4_1 decoder here made the E2E path unreachable.
    if (!valid_dtype(gs_dt)) {
        LOGE("e2e: unsupported Gaussian decoder dtype: %s", gs_dt.c_str());
        return 1;
    }
    DtypeContractMetadata dtype_metadata;
    dtype_metadata.backend = be_name;
    dtype_metadata.base_dtype = dt;
    dtype_metadata.ss_dtype = ss_dt;
    dtype_metadata.ss_decoder_dtype = ss_decoder_dt;
    dtype_metadata.slat_dtype = slat_dt;
    dtype_metadata.gs_dtype = gs_dt;
    dtype_metadata.mesh_dtype = mesh_dt;
    std::string dtype_contract_error;
    ScopedDtypeContractRecording dtype_contract(
            out_dtype_contract, dtype_metadata, dtype_contract_error);
    if (!out_dtype_contract.empty() && !dtype_contract.enabled()) {
        LOGE("e2e: cannot start dtype contract recording: %s",
             dtype_contract_error.c_str());
        return 1;
    }
    if (!out_pbr.empty()) {
#if !defined(SAM3D_USE_NVDIFFRAST_CUDARASTER)
        LOGE("e2e: --pbr-out requires CUDA native PBR with "
             "SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON");
        return 1;
#elif !defined(SAM3D_USE_MESHFIX_GPL)
        LOGE("e2e: --pbr-out requires SAM3D_GGML_MESHFIX_GPL=ON for the "
             "official "
             "pymeshfix TMesh cleanup; no approximate repair is substituted");
        return 1;
#endif
    }
    if (out_mesh_vertices.empty() != out_mesh_faces.empty()) {
        LOGE("e2e: --mesh-vertices-out and --mesh-faces-out must be provided "
             "together");
        return 1;
    }
    const auto t_start = std::chrono::steady_clock::now();
    const auto elapsed_seconds = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             t_start)
                .count();
    };
    // Split the pipeline into independent processes when GPU memory is tight:
    //   cond: embedder/fuser -> fused condition tokens
    //   ss:   SS flow/decoder -> sparse coordinates
    //   slat: SLat flow/Gaussian decoder -> PLY
    // This prevents a prior model's CUDA allocator pool from consuming the
    // memory required by the next model's graph. "flow" remains compatible.
    const bool cond_stage = opt.stage == "cond";
    const bool ss_stage = opt.stage == "ss";
    const bool slat_stage = opt.stage == "slat";
    const bool flow_stage_env = opt.stage == "flow";

    // ---------------- stage A: sparse structure --------------------------
    // The SLat process receives coordinates and does not need SS weights.
    Stage ss;
    if (!flow_stage_env && !slat_stage) {
        if (!ss.open(models_dir + "/ss_generator-" + ss_dt + ".gguf", be_name,
                     nthreads, "condition_ss_generator",
                     static_cast<Backend*>(opt.shared_backend))) {
            LOGE("e2e: failed to load ss_generator");
            return 1;
        }
        LOGI("e2e: ss_generator loaded (%s)", ss_dt.c_str());
    }

    // SKIP_COND=1: no embedder graphs at all (cond from SAMT)
    // SKIP_COND=2: only the fuser graph (dino/pp tokens from the dbg dumps)
    // Isolated flow stages consume tokens written by the cond process.
    const int skip_cond = (flow_stage_env || ss_stage || slat_stage)
                                  ? 1
                                  : (opt.fuser_only ? 2 : 0);
    CondInputs ci;
    std::vector<float> ss_cond;
    std::vector<float> slat_cond;
    if (skip_cond >= 1) {
        if (!slat_stage) {
            const std::string ss_cond_path =
                    opt.ss_cond_path.empty() ? cond_dir + "/ss_cond_tokens.samt"
                                             : opt.ss_cond_path;
            RawTensor ct;
            if (!load_raw_tensor(ss_cond_path, ct) || ct.type != GGML_TYPE_F32)
                return 1;
            ss_cond.assign(reinterpret_cast<const float*>(ct.data.data()),
                           reinterpret_cast<const float*>(ct.data.data()) +
                                   ct.data.size() / 4);
        }
        if (flow_stage_env || slat_stage) {
            const std::string slat_cond_path =
                    opt.slat_cond_path.empty()
                            ? cond_dir + "/slat_cond_tokens.samt"
                            : opt.slat_cond_path;
            RawTensor st3;
            if (!load_raw_tensor(slat_cond_path, st3) ||
                st3.type != GGML_TYPE_F32)
                return 1;
            slat_cond.assign(reinterpret_cast<const float*>(st3.data.data()),
                             reinterpret_cast<const float*>(st3.data.data()) +
                                     st3.data.size() / 4);
        }
        LOGI("e2e: SKIP_COND - fused cond tokens loaded");
    }
    if (!skip_cond) {
        // In-memory condition handoff (C API path) bypasses cond_dir; the
        // directory input stays the CLI/stage-replay contract.
        if (opt.conditions ? !cond_inputs_from_native(*opt.conditions, ci)
                           : !load_cond_inputs(cond_dir, ci)) {
            return 1;
        }
    }
    if (opt.progress) opt.progress(kStageCondition, 0, 0);
    LOGI("e2e: condition inputs %lldx%lld", (long long)ci.W, (long long)ci.H);

    if (cond_stage) {
        int64_t nd = 0;
        DinoDbg dino_dbg_cfg;
        dino_dbg_cfg.enabled = opt.dino_dbg;
        dino_dbg_cfg.debug_stage = opt.debug_stage;
        dino_dbg_cfg.out_path = opt.dino_dbg_out;
        auto d4 = run_dino_batch(ss,
                                 {{"cemb.emb0", &ci.image},
                                  {"cemb.emb0", &ci.rgb_image},
                                  {"cemb.emb1", &ci.mask3},
                                  {"cemb.emb1", &ci.rgb_image_mask3}},
                                 false, nd, &dino_dbg_cfg,
                                 opt.cond_manual_attention);
        if (d4.empty() || d4[0].empty()) {
            LOGE("e2e-cond: dino failed");
            return 1;
        }
        if (dino_dbg_cfg.enabled) {
            // The debug graph returns a single boundary tensor, but the fuser
            // below consumes all four DINO outputs: stop once the dump is on
            // disk instead of indexing out-of-bounds vectors.
            return 0;
        }
        auto p2 = run_pointpatch_batch(ss, "cemb.emb2",
                                       {{&ci.pointmap, {ci.W, ci.H}},
                                        {&ci.rgb_pointmap, {ci.W, ci.H}}});
        if (p2.empty() || p2[0].empty()) {
            LOGE("e2e-cond: pointpatch failed");
            return 1;
        }
        const int64_t n_pp2 = (int64_t)p2[0].size() / 512;
        std::vector<float> cond_out;
        {
            GraphContext gctx;
            std::vector<FuserSeg> segs = {
                    {1024, nd, 0, 0, std::move(d4[0])},
                    {1024, nd, 0, 1, std::move(d4[1])},
                    {1024, nd, 1, 0, std::move(d4[2])},
                    {1024, nd, 1, 1, std::move(d4[3])},
                    {512, n_pp2, 2, 0, std::move(p2[0])},
                    {512, n_pp2, 2, 1, std::move(p2[1])},
            };
            ggml_tensor* out = build_fuser(gctx, *ss.model, segs, "cemb");
            ggml_cgraph* graph =
                    ggml_new_graph_custom(gctx.ctx(), 16384, false);
            ggml_build_forward_expand(graph, out);
            record_dtype_contract_graph("ss_condition_fuser", graph);
            if (!ss.backend->alloc(graph)) return 1;
            for (auto& sg2 : segs)
                if (!ss.backend->set_input_f32(sg2.tokens, sg2.host.data(),
                                               sg2.host.size()))
                    return 1;
            if (!ss.backend->run(graph)) return 1;
            ss.backend->get_tensor_f32(out, cond_out);
        }
        // The side files are PLY-path artifacts (stage probes read them);
        // with an empty out_ply (in-memory artifacts path) there is no file
        // family to extend.
        if (!out_ply.empty()) {
            save_raw_tensor_f32(out_ply + ".ss_cond.samt", {1024, 7528},
                                cond_out.data());
        }
        ss.close();  // free the SS weights before the slat embedder stage
        Stage slat_emb;
        if (!slat_emb.open(models_dir + "/slat_generator-" + slat_dt + ".gguf",
                           be_name, nthreads, "condition_slat_generator",
                           static_cast<Backend*>(opt.shared_backend)))
            return 1;
        auto sd = run_dino_batch(slat_emb,
                                 {{"cemb.emb0", &ci.image},
                                  {"cemb.emb0", &ci.rgb_image},
                                  {"cemb.emb1", &ci.mask3},
                                  {"cemb.emb1", &ci.rgb_image_mask3}},
                                 true, nd, nullptr, opt.cond_manual_attention);
        if (sd.empty() || sd[0].empty()) {
            LOGE("e2e-cond: slat dino failed");
            return 1;
        }
        std::vector<float> slat_out;
        {
            GraphContext gctx;
            std::vector<FuserSeg> segs = {
                    {1024, nd, 0, 0, std::move(sd[0])},
                    {1024, nd, 0, 1, std::move(sd[1])},
                    {1024, nd, 1, 0, std::move(sd[2])},
                    {1024, nd, 1, 1, std::move(sd[3])},
            };
            ggml_tensor* out = build_fuser(gctx, *slat_emb.model, segs, "cemb");
            ggml_cgraph* graph =
                    ggml_new_graph_custom(gctx.ctx(), 16384, false);
            ggml_build_forward_expand(graph, out);
            record_dtype_contract_graph("slat_condition_fuser", graph);
            if (!slat_emb.backend->alloc(graph)) return 1;
            for (auto& sg2 : segs)
                if (!slat_emb.backend->set_input_f32(
                            sg2.tokens, sg2.host.data(), sg2.host.size()))
                    return 1;
            if (!slat_emb.backend->run(graph)) return 1;
            slat_emb.backend->get_tensor_f32(out, slat_out);
        }
        // The side files are PLY-path artifacts (stage probes read them).
        if (!out_ply.empty()) {
            save_raw_tensor_f32(out_ply + ".slat_cond.samt", {1024, 5496},
                                slat_out.data());
        }
        LOGI("e2e-cond: done in %.1fs", elapsed_seconds());
        if (!dtype_contract.complete(dtype_contract_error)) {
            LOGE("e2e-cond: failed to write dtype contract: %s",
                 dtype_contract_error.c_str());
            return 1;
        }
        return 0;
    }

    if (!skip_cond) {
        // 4 DINO forwards: emb0(image), emb0(rgb_image), emb1(mask),
        // emb1(rgb_mask)
        int64_t n_dino = 0;
        DinoDbg dino_dbg_cfg;
        dino_dbg_cfg.enabled = opt.dino_dbg;
        dino_dbg_cfg.debug_stage = opt.debug_stage;
        dino_dbg_cfg.out_path = opt.dino_dbg_out;
        auto d = run_dino_batch(ss,
                                {{"cemb.emb0", &ci.image},
                                 {"cemb.emb0", &ci.rgb_image},
                                 {"cemb.emb1", &ci.mask3},
                                 {"cemb.emb1", &ci.rgb_image_mask3}},
                                false, n_dino, &dino_dbg_cfg,
                                opt.cond_manual_attention);
        if (d.empty() || d[0].empty()) {
            LOGE("e2e: dino forward failed");
            return 1;
        }
        if (dino_dbg_cfg.enabled) {
            // Same as the cond stage: the debug dump is a single boundary
            // tensor; stop before the fuser that needs all four outputs.
            return 0;
        }
        std::vector<float> d_img = std::move(d[0]), d_rgb = std::move(d[1]);
        std::vector<float> d_mask = std::move(d[2]), d_rgbm = std::move(d[3]);
        LOGI("e2e: dino tokens done (n=%lld each)", (long long)n_dino);

        // 2 PointPatch forwards (one graph)
        auto p = run_pointpatch_batch(ss, "cemb.emb2",
                                      {{&ci.pointmap, {ci.W, ci.H}},
                                       {&ci.rgb_pointmap, {ci.W, ci.H}}});
        if (p.empty() || p[0].empty()) {
            LOGE("e2e: pointpatch failed");
            return 1;
        }
        std::vector<float> p_pm = std::move(p[0]), p_rgbpm = std::move(p[1]);
        const int64_t n_pp = (int64_t)p_pm.size() / 512;
        if (!dbg_dir.empty()) {
            save_raw_tensor_f32(dbg_dir + "/e2e_pp_tokens.samt", {512, n_pp},
                                p_pm.data());
            save_raw_tensor_f32(dbg_dir + "/e2e_dino_tokens.samt",
                                {1024, n_dino}, d_img.data());
        }
        LOGI("e2e: pointpatch tokens done (n=%lld each)", (long long)n_pp);

        // fuser -> ss condition tokens (1024, 7528)
        // ss_cond filled by the fuser below
        {
            GraphContext gctx;
            std::vector<FuserSeg> segs = {
                    {1024, n_dino, 0, 0, std::move(d_img)},
                    {1024, n_dino, 0, 1, std::move(d_rgb)},
                    {1024, n_dino, 1, 0, std::move(d_mask)},
                    {1024, n_dino, 1, 1, std::move(d_rgbm)},
                    {512, n_pp, 2, 0, std::move(p_pm)},
                    {512, n_pp, 2, 1, std::move(p_rgbpm)},
            };
            ggml_tensor* out = build_fuser(gctx, *ss.model, segs, "cemb");
            ggml_cgraph* graph =
                    ggml_new_graph_custom(gctx.ctx(), 16384, false);
            ggml_build_forward_expand(graph, out);
            record_dtype_contract_graph("ss_condition_fuser", graph);
            if (!ss.backend->alloc(graph)) {
                LOGE("e2e: fuser alloc failed");
                return 1;
            }
            for (auto& s : segs) {
                if (!ss.backend->set_input_f32(s.tokens, s.host.data(),
                                               s.host.size()))
                    return 1;
            }
            if (!ss.backend->run(graph)) {
                LOGE("e2e: fuser run failed");
                return 1;
            }
            ss.backend->get_tensor_f32(out, ss_cond);
        }
    }
    LOGI("e2e: ss condition tokens %lld", (long long)ss_cond.size() / 1024);
    if (!dbg_dir.empty() && !ss_cond.empty())
        save_raw_tensor_f32(dbg_dir + "/e2e_ss_cond_tokens.samt", {1024, 7528},
                            ss_cond.data());

    // fresh backend for the flow graph: a shared gallocr across the embedder
    // graphs and the flow graph changed its allocation layout and produced
    // NaNs on the 4th flow run (CUDA only); a clean gallocr per stage is
    // also the CLI-verified configuration.
    ss.close();

    // A strict replay consumes immutable official support. The dedicated SLat
    // stage likewise starts from coordinates produced by the dedicated SS
    // process, so neither path allocates SS model state in this process.
    const bool strict_reference_coords = opt.reference_coords;
    std::vector<int32_t> coords;  // (N,4) [b, x, y, z], z fastest
    uint32_t distribution_blocks = 0;
#if defined(SAM3D_USE_CUDA)
    std::unique_ptr<PytorchCudaNormalRng> pytorch_cuda_rng;
    // A CUDA-only build resolves `auto` to its registered CUDA device. The
    // verified PyTorch CUDA kernel remains the source implementation there.
    if (std::strcmp(be_name, "cuda") == 0 ||
        std::strcmp(be_name, "auto") == 0) {
        std::string rng_contract_error;
        if (!PytorchCudaNormalRng::distribution_blocks(distribution_blocks,
                                                       rng_contract_error)) {
            LOGE("e2e: %s", rng_contract_error.c_str());
            return 1;
        }
        pytorch_cuda_rng = std::make_unique<PytorchCudaNormalRng>(
                seed, distribution_blocks);
    }
#endif
    std::unique_ptr<PytorchPhiloxNormalRng> pytorch_rng_contract;
#if !defined(SAM3D_USE_CUDA)
    constexpr bool uses_cuda_rng = false;
#else
    const bool uses_cuda_rng = pytorch_cuda_rng != nullptr;
#endif
    if (opt.philox_blocks > 0) {
        distribution_blocks = opt.philox_blocks;
    } else if (!uses_cuda_rng) {
        std::string rng_contract_error;
        if (!pytorch_philox_distribution_blocks_from_environment(
                    distribution_blocks, rng_contract_error)) {
            LOGE("e2e: %s", rng_contract_error.c_str());
            return 1;
        }
    }
    // CUDA materializes normal samples with the verified device kernel, while
    // this parallel state tracks the same counter for the later randperm.
    // CPU/Vulkan also materialize from this state directly.
    pytorch_rng_contract =
            std::make_unique<PytorchPhiloxNormalRng>(seed, distribution_blocks);
    auto fill_sampling_noise = [&](std::vector<float>& values) -> bool {
#if defined(SAM3D_USE_CUDA)
        if (pytorch_cuda_rng) {
            std::string rng_error;
            if (!pytorch_cuda_rng->fill(values, rng_error)) {
                LOGE("e2e: PyTorch-compatible CUDA RNG failed: %s",
                     rng_error.c_str());
                return false;
            }
            if (!pytorch_rng_contract->advance_normal(values.size(),
                                                      rng_error)) {
                LOGE("e2e: cannot advance Philox randperm state: %s",
                     rng_error.c_str());
                return false;
            }
            return true;
        }
#endif
        std::string rng_error;
        if (!pytorch_rng_contract->fill(values, rng_error)) {
            LOGE("e2e: PyTorch-compatible portable Philox RNG failed: %s",
                 rng_error.c_str());
            return false;
        }
        return true;
    };
    if (!slat_stage && (!flow_stage_env || !strict_reference_coords)) {
        // ---- SS sampling: 25 rescaled Euler steps; CFG covers every modality
        // ---- Values match pipeline.yaml plus InferencePipeline's SS defaults.
        // The Python wrapper enables no_shortcut, so d remains zero for every
        // step.
        constexpr int kSsSteps = 25;
        constexpr float kSsRescaleT = 3.0f;
        constexpr float kSsCfgStrength = 7.0f;
        constexpr float kSsCfgStart = 0.0f;
        constexpr float kSsCfgEnd = 500.0f;
        const int64_t n_shape = 4096;
        std::vector<float> x_shape(8 * n_shape), x_6d(6), x_sc(3), x_tr(3),
                x_ts(1);
        const bool replayed_ss_noise =
                load_noise_opt(noise_dir, "ss_x0_shape.samt", x_shape,
                               8 * n_shape) &&
                load_noise_opt(noise_dir, "ss_x0_6drotation_normalized.samt",
                               x_6d, 6) &&
                load_noise_opt(noise_dir, "ss_x0_scale.samt", x_sc, 3) &&
                load_noise_opt(noise_dir, "ss_x0_translation.samt", x_tr, 3) &&
                load_noise_opt(noise_dir, "ss_x0_translation_scale.samt", x_ts,
                               1);
        if (!replayed_ss_noise) {
            if (!noise_dir.empty()) {
                LOGE("e2e: --noise-dir must contain valid F32 SS initial-noise "
                     "tensors");
                return 1;
            }
            LOGI("e2e: reference noise unavailable, using seed %u", seed);
            // `latent_mapping` is an insertion-ordered Python dict.  The
            // official generator draws in this order from one CUDA Philox
            // state.
            if (!fill_sampling_noise(x_6d) || !fill_sampling_noise(x_sc) ||
                !fill_sampling_noise(x_shape) || !fill_sampling_noise(x_tr) ||
                !fill_sampling_noise(x_ts))
                return 1;
        }

        Stage ssf;
        if (!ssf.open(models_dir + "/ss_generator-" + ss_dt + ".gguf", be_name,
                      nthreads, "ss_flow",
                      static_cast<Backend*>(opt.shared_backend))) {
            LOGE("e2e: failed to reload ss_generator for the flow stage");
            return 1;
        }
        GraphContext fg;
        SsFlowGraph sfg;
        sfg.g = &fg;
        sfg.m = ssf.model.get();
        sfg.n_cond_tokens = 7528;
        // The official SS backbone is an F32 model (use_fp16=false): its
        // attention contract is F32 SDPA, so K/V stay in F32 by default.
        sfg.strict_attention = opt.ss_strict_attention;
        // Exact raw-string match like the upstream reference implementation:
        // only an explicit "cuda" request enables the F16 autocast chain, so
        // the default "auto" keeps the SS flow in F32 and bit-exact with the
        // upstream production path (888224 gaussian count parity).
        sfg.f16_autocast = std::strcmp(be_name, "cuda") == 0;
        // SS debug probe: ss_flow_graph's build() short-circuits and returns
        // the debug-stage tensor as the only output when debug_stage is set.
        if (!opt.debug_stage.empty()) sfg.debug_stage = opt.debug_stage;
        auto outs = sfg.build();  // dict order: 6drot, scale, shape,
                                  // translation, ts
        ggml_cgraph* fgraph = ggml_new_graph_custom(fg.ctx(), 32768, false);
        for (auto* o : outs) {
            ggml_set_output(o);
            ggml_build_forward_expand(fgraph, o);
        }
        record_dtype_contract_graph("ss_flow", fgraph);
        if (!ssf.backend->alloc(fgraph)) {
            LOGE("e2e: ss flow alloc failed");
            return 1;
        }
        if (opt.progress) opt.progress(kStageSsFlow, 0, 0);
        LOGI("e2e: ss flow graph built (%d nodes)", ggml_graph_n_nodes(fgraph));

        std::vector<float> zeros_cond(ss_cond.size(), 0.0f);
        // Table values are invariant for all Euler/CFG replays. A Vulkan A/B
        // experiment produced different sparse coordinates when caching this
        // input, so the path remains opt-in until buffer lifetime is proven.
        const bool cache_vulkan_ss_tables =
                strstr(ssf.backend->backend_name(), "Vulkan") != nullptr &&
                opt.vulkan_ss_table_cache && !opt.vulkan_table_cache;
        auto up_scalar = [&](ggml_tensor* t, float v) {
            return !t->buffer || ssf.backend->set_input_f32(t, &v, 1);
        };
        auto upload_ss_tables = [&]() -> bool {
            for (size_t ti = 0; ti < sfg.inputs.size(); ++ti) {
                ggml_tensor* input = sfg.inputs[ti];
                if (!input->buffer || !sfg.table_data[ti]) continue;
                const auto& host = sfg.table_data[ti];
                const bool ok =
                        input->type == GGML_TYPE_F32
                                ? ssf.backend->set_input_f32(
                                          input,
                                          reinterpret_cast<const float*>(
                                                  host->data()),
                                          host->size())
                                : ssf.backend->set_input_i32(
                                          input, host->data(), host->size());
                if (!ok) {
                    LOGE("e2e: failed to upload SS table %s", input->name);
                    return false;
                }
            }
            return true;
        };
        if (cache_vulkan_ss_tables && !upload_ss_tables()) return 1;
        auto upload_and_run = [&](float t_v, bool zero_cond) -> bool {
            auto up = [&](ggml_tensor* t, const std::vector<float>& v) {
                if (!t->buffer) return true;
                return ssf.backend->set_input_f32(t, v.data(), v.size());
            };
            if (!up(sfg.x_shape, x_shape)) return false;
            if (!up(sfg.x_6drot, x_6d)) return false;
            if (!up(sfg.x_scale, x_sc)) return false;
            if (!up(sfg.x_trans, x_tr)) return false;
            if (!up(sfg.x_ts, x_ts)) return false;
            if (!up_scalar(sfg.t, t_v)) return false;
            if (!up_scalar(sfg.d, 0.0f)) return false;
            if (!up(sfg.cond, zero_cond ? zeros_cond : ss_cond)) return false;
            if (!cache_vulkan_ss_tables && !upload_ss_tables()) return false;
            if (opt.verify) {
                float rb_t = -9;
                std::vector<float> rb_t1(1), rb_c(4), rb_x(4);
                ssf.backend->get_tensor_f32(sfg.t, rb_t1);
                rb_t = rb_t1[0];
                ssf.backend->get_tensor_f32(sfg.cond, rb_c);
                ssf.backend->get_tensor_f32(sfg.x_shape, rb_x);
                LOGI("verify: t=%g c0=%g x0=%g (expect t=%g c0=%g x0=%g)", rb_t,
                     rb_c[0], rb_x[0], t_v, (zero_cond ? 0.0f : ss_cond[0]),
                     x_shape[0]);
            }
            return ssf.backend->run(fgraph);
        };
        auto get_outs = [&](std::vector<std::vector<float>>& v) {
            v.resize(outs.size());
            for (size_t i = 0; i < outs.size(); i++)
                ssf.backend->get_tensor_f32(outs[i], v[i]);
        };

        int ss_steps = opt.ss_steps;
        if (ss_steps < 1 || ss_steps > kSsSteps) {
            LOGE("e2e: --ss-steps must be in [1, %d], got %d", kSsSteps,
                 ss_steps);
            return 1;
        }
        const auto ss_schedule = make_euler_schedule(kSsSteps, kSsRescaleT);
        if (!opt.debug_stage.empty() &&
            strcmp(opt.debug_stage.c_str(), "im2col") != 0 &&
            strcmp(opt.debug_stage.c_str(), "gemm") != 0 &&
            strcmp(opt.debug_stage.c_str(), "convout") != 0) {
            // SS debug probe: replay only the first Euler step (the same t the
            // main loop would use) and dump the debug-stage tensor. Used by
            // the sam3d parity ladder to bisect SS DiT internals block by
            // block (b0_adaln -> b0_qkv -> b0_attn_out -> b0_res ...).
            // The decoder boundary names (im2col/gemm/convout) are handled by
            // the SS decoder probe further down.
            if (outs.size() != 1) {
                LOGE("e2e: ss debug stage '%s' did not short-circuit build",
                     opt.debug_stage.c_str());
                return 1;
            }
            const float t_v = ss_schedule[0].t * 1000.0f;
            if (!upload_and_run(t_v, false)) {
                LOGE("e2e: ss debug probe run failed");
                return 1;
            }
            std::vector<float> vals;
            if (!ssf.backend->get_tensor_f32(outs[0], vals)) {
                LOGE("e2e: ss debug probe readback failed");
                return 1;
            }
            const std::string dbg_out =
                    dbg_dir.empty()
                            ? "ss_dbg_" + opt.debug_stage + ".samt"
                            : dbg_dir + "/ss_dbg_" + opt.debug_stage + ".samt";
            save_raw_tensor_f32(
                    dbg_out,
                    std::vector<int64_t>{outs[0]->ne[0], outs[0]->ne[1]},
                    vals.data());
            LOGI("e2e: ss debug stage '%s' dumped (%zu elems) -> %s",
                 opt.debug_stage.c_str(), vals.size(), dbg_out.c_str());
            return 0;
        }
        for (int step = 0; step < ss_steps; ++step) {
            const float t_v = ss_schedule[step].t * 1000.0f;
            if (!upload_and_run(t_v, false)) {
                LOGE("e2e: ss cond run failed");
                return 1;
            }
            std::vector<std::vector<float>> vc;
            get_outs(vc);
            std::vector<std::vector<float>> vu;
            const bool cfg_active = t_v >= kSsCfgStart && t_v <= kSsCfgEnd;
            if (cfg_active) {
                if (!upload_and_run(t_v, true)) {
                    LOGE("e2e: ss uncond run failed");
                    return 1;
                }
                get_outs(vu);
            }
            if (!dbg_dir.empty()) {
                for (size_t oi = 0; oi < outs.size(); oi++) {
                    save_raw_tensor_f32(dbg_dir + "/e2e_ss_vc" +
                                                std::to_string(step) + "_" +
                                                std::to_string(oi) + ".samt",
                                        std::vector<int64_t>{outs[oi]->ne[0],
                                                             outs[oi]->ne[1]},
                                        vc[oi].data());
                    if (cfg_active) {
                        save_raw_tensor_f32(
                                dbg_dir + "/e2e_ss_vu" + std::to_string(step) +
                                        "_" + std::to_string(oi) + ".samt",
                                std::vector<int64_t>{outs[oi]->ne[0],
                                                     outs[oi]->ne[1]},
                                vu[oi].data());
                    }
                }
                // dump the step-1 inputs for offline CLI repro
                if (step == 1) {
                    save_raw_tensor_f32(dbg_dir + "/e2e_ss_x1_shape.samt",
                                        {8, n_shape}, x_shape.data());
                    save_raw_tensor_f32(dbg_dir + "/e2e_ss_x1_6d.samt", {6},
                                        x_6d.data());
                }
            }
            for (size_t i = 0; i < outs.size(); i++) {
                std::vector<float>& x = i == 2   ? x_shape
                                        : i == 0 ? x_6d
                                        : i == 1 ? x_sc
                                        : i == 3 ? x_tr
                                                 : x_ts;
                for (size_t k = 0; k < x.size(); k++) {
                    // ``cfg_modalities`` belongs to the Shortcut training
                    // target; the inference wrapper's scalar
                    // ClassifierFreeGuidance strength is applied to every
                    // output modality.
                    const float velocity =
                            cfg_active ? vc[i][k] + kSsCfgStrength * (vc[i][k] -
                                                                      vu[i][k])
                                       : vc[i][k];
                    x[k] += ss_schedule[step].dt * velocity;
                }
            }
            if (!dbg_dir.empty()) {
                const char* x_names[] = {"6drotation_normalized", "scale",
                                         "shape", "translation",
                                         "translation_scale"};
                const std::vector<float>* states[] = {&x_6d, &x_sc, &x_shape,
                                                      &x_tr, &x_ts};
                // 2-D records match the reference SAMT files (the pose states
                // are single-column tensors, and a 1-D record broke shape
                // equality).
                const std::vector<int64_t> x_ne[] = {
                        {6, 1}, {3, 1}, {8, n_shape}, {3, 1}, {1, 1}};
                for (size_t i = 0; i < 5; i++) {
                    save_raw_tensor_f32(dbg_dir + "/e2e_ss_state" +
                                                std::to_string(step + 1) + "_" +
                                                x_names[i] + ".samt",
                                        x_ne[i], states[i]->data());
                }
            }
            LOGI("e2e: ss step %d/%d done", step + 1, ss_steps);
        }
        LOGI("e2e: ss flow done in %.1fs", elapsed_seconds());
        ssf.close();
        if (!dbg_dir.empty())
            save_raw_tensor_f32(dbg_dir + "/e2e_ss_shape_latent.samt",
                                {8, n_shape}, x_shape.data());
        if (ss_stage && opt.ss_flow_only) {
            LOGI("e2e: SS flow-only diagnostic done in %.1fs",
                 elapsed_seconds());
            return 0;
        }

        // ---- ss_decoder -> occupancy -> coords ------------------------------
        {
            // the SS decoder lives in its own GGUF (ss_decoder-*.gguf)
            Stage dec;
            if (opt.progress) opt.progress(kStageSsDecode, 0, 0);
            if (!dec.open(models_dir + "/ss_decoder-" + ss_decoder_dt + ".gguf",
                          be_name, nthreads, "ss_decoder",
                          static_cast<Backend*>(opt.shared_backend))) {
                LOGE("e2e: failed to load ss_decoder");
                return 1;
            }
            GraphContext gctx;
            SsDecoderGraph sdg;
            sdg.ctx = gctx.ctx();
            sdg.m = dec.model.get();
            // SS decoder debug probe: short-circuit build() at the requested
            // boundary (im2col/gemm/convout) for parity bisects.
            if (!opt.debug_stage.empty())
                sdg.debug_stage = opt.debug_stage.c_str();
            // torch lat_in =
            // shape_latent(1,4096,8).permute(0,2,1).view(1,8,16,16,16): memory
            // becomes channel-major c*4096 + (x*16+y)*16+z -> transpose here
            std::vector<float> lat_cm(x_shape.size());
            for (int64_t n = 0; n < n_shape; n++)
                for (int c = 0; c < 8; c++)
                    lat_cm[c * n_shape + n] = x_shape[n * 8 + c];
            ggml_tensor* lat = gctx.input_f32("latent", {16, 16, 16, 8});
            ggml_tensor* occ = sdg.build(lat);
            ggml_cgraph* graph = ggml_new_graph_custom(gctx.ctx(), 8192, false);
            ggml_build_forward_expand(graph, lat);
            ggml_build_forward_expand(graph, occ);
            record_dtype_contract_graph("ss_decoder", graph);
            if (!dec.backend->alloc(graph)) {
                LOGE("e2e: ss decoder alloc failed");
                return 1;
            }
            if (!dec.backend->set_input_f32(lat, lat_cm.data(), lat_cm.size()))
                return 1;
            if (!dec.backend->run(graph)) {
                LOGE("e2e: ss decoder run failed");
                return 1;
            }
            std::vector<float> occ_f;
            dec.backend->get_tensor_f32(occ, occ_f);
            dec.close();
            if (!opt.debug_stage.empty()) {
                // Debug build returned the boundary tensor; dump it and stop
                // before the occupancy expansion that expects (64,64,64).
                const std::string dbg_out2 =
                        dbg_dir.empty()
                                ? "ss_dbg_dec_" + opt.debug_stage + ".samt"
                                : dbg_dir + "/ss_dbg_dec_" + opt.debug_stage +
                                          ".samt";
                save_raw_tensor_f32(
                        dbg_out2,
                        {occ->ne[0], occ->ne[1], occ->ne[2], occ->ne[3]},
                        occ_f.data());
                LOGI("e2e: ss decoder debug stage '%s' dumped (%zu elems) -> "
                     "%s",
                     opt.debug_stage.c_str(), occ_f.size(), dbg_out2.c_str());
                return 0;
            }
            LOGI("e2e: ss decoder done in %.1fs", elapsed_seconds());
            // torch (1,1,64,64,64) C-order: flat = (x*64 + y)*64 + z
            for (int x = 0; x < 64; x++)
                for (int y = 0; y < 64; y++)
                    for (int z = 0; z < 64; z++)
                        if (occ_f[((x * 64) + y) * 64 + z] > 0.0f)
                            coords.insert(coords.end(), {0, x, y, z});
        }
        // Pre-compute the effective downsample factor before the pose decode:
        // the official pipeline rescales the decoded instance scale by
        // downsample_factor whenever the pruned sparse support exceeds
        // max_coords (inference_pipeline.py "Rescaling scale by <factor> after
        // downsampling"). prune_surface_coords is deterministic, so re-running
        // it here on the same pre-prune coords yields the same decision the
        // SLat-stage branch below will make; the factor must be known here
        // because the pose decode runs before that branch in this function.
        int effective_pose_downsample_factor = 1;
        if (!(slat_stage && !strict_reference_coords)) {
            const std::vector<int32_t> pruned = prune_surface_coords(coords);
            if (static_cast<int64_t>(pruned.size() / 4) > 42000) {
                effective_pose_downsample_factor = 2;
            }
        }
        const bool want_pose_json = !out_pose.empty();
        const bool want_pose_artifact =
                opt.artifacts != nullptr && opt.scene_attributes;
        if (want_pose_json || want_pose_artifact) {
            std::array<float, 3> pointmap_scale{};
            std::array<float, 3> pointmap_shift{};
            if (opt.conditions) {
                pointmap_scale = opt.conditions->pointmap_scale;
                pointmap_shift = opt.conditions->pointmap_shift;
            } else if (!load_f32_vector(
                               cond_dir + "/ss_input_pointmap_scale.samt", 3,
                               pointmap_scale) ||
                       !load_f32_vector(
                               cond_dir + "/ss_input_pointmap_shift.samt", 3,
                               pointmap_shift)) {
                return 1;
            }
            NativeInstancePose pose;
            std::string pose_error;
            if (!decode_scale_shift_invariant_pose(
                        x_6d.data(), x_sc.data(), x_tr.data(), x_ts[0],
                        pointmap_scale.data(), pointmap_shift.data(), pose,
                        pose_error, effective_pose_downsample_factor)) {
                LOGE("e2e: native pose decode/export failed: %s",
                     pose_error.c_str());
                return 1;
            }
            if (want_pose_json &&
                !write_native_pose_json(out_pose, pose, pose_error)) {
                LOGE("e2e: native pose decode/export failed: %s",
                     pose_error.c_str());
                return 1;
            }
            if (want_pose_artifact) {
                opt.artifacts->pose = pose;
                opt.artifacts->has_pose = true;
            }
            LOGI("e2e: decoded official ScaleShiftInvariant pose");
        }
        LOGI("e2e: occupancy coords %lld (of 262144)",
             (long long)coords.size() / 4);
        ss.close();  // free the SS weights before the SLat stage
    }

    int64_t n_coord = 0;
    if (slat_stage && !strict_reference_coords) {
        const char* coords_path_env = opt.coords_path.c_str();
        if (!coords_path_env) {
            LOGE("e2e: SAM3D_E2E_COORDS_PATH is required for the slat stage");
            return 1;
        }
        RawTensor saved_coords;
        if (!load_raw_tensor(coords_path_env, saved_coords) ||
            saved_coords.type != GGML_TYPE_I32 || saved_coords.ne.size() != 2 ||
            saved_coords.ne[0] != 4 ||
            saved_coords.data.size() % (4 * sizeof(int32_t)) != 0) {
            LOGE("e2e: invalid sparse coordinate SAMT: %s", coords_path_env);
            return 1;
        }
        const int32_t* values =
                reinterpret_cast<const int32_t*>(saved_coords.data.data());
        coords.assign(values,
                      values + saved_coords.data.size() / sizeof(int32_t));
        n_coord = static_cast<int64_t>(coords.size() / 4);
        LOGI("e2e: loaded %lld sparse coords from the SS stage",
             static_cast<long long>(n_coord));
    } else {
        const size_t before_prune = coords.size() / 4;
        coords = prune_surface_coords(coords);
        LOGI("e2e: surface-pruned sparse coords %zu -> %zu", before_prune,
             coords.size() / 4);
        n_coord = static_cast<int64_t>(coords.size() / 4);
        if (n_coord > 42000) {
            const size_t before = coords.size() / 4;
            bool randomly_subsampled = false;
            std::string downsample_error;
            coords = downsample_sparse_coords_pytorch(
                    coords, *pytorch_rng_contract, downsample_error,
                    randomly_subsampled, 42000, 2);
            if (coords.empty()) {
                LOGE("e2e: PyTorch-compatible coordinate randperm failed: %s",
                     downsample_error.c_str());
                return 1;
            }
            n_coord = static_cast<int64_t>(coords.size() / 4);
            LOGI("e2e: downsampled sparse coords %zu -> %zu with PyTorch "
                 "Philox randperm%s",
                 before, coords.size() / 4,
                 randomly_subsampled ? "" : " (dedup only)");
        }
    }
    // A strict replay holds the sparse support fixed to the official output.
    // It isolates SLat and Gaussian-decoder quantization from a boundary flip
    // in SS occupancy; normal image-to-3D inference never sets this switch.
    if (strict_reference_coords) {
        RawTensor reference_coords;
        if (!load_raw_tensor(cond_dir + "/coords.samt", reference_coords) ||
            reference_coords.type != GGML_TYPE_I32 ||
            reference_coords.data.size() % (4 * sizeof(int32_t)) != 0) {
            LOGE("e2e: invalid reference coords.samt");
            return 1;
        }
        const int32_t* values =
                reinterpret_cast<const int32_t*>(reference_coords.data.data());
        coords.assign(values,
                      values + reference_coords.data.size() / sizeof(int32_t));
        n_coord = static_cast<int64_t>(coords.size() / 4);
        LOGI("e2e: strict replay replaces sparse support with %lld official "
             "coords",
             static_cast<long long>(n_coord));
    }
    if (ss_stage) {
        RawTensor saved_coords;
        saved_coords.ne = {4, n_coord};
        saved_coords.type = GGML_TYPE_I32;
        saved_coords.data.resize(coords.size() * sizeof(int32_t));
        std::memcpy(saved_coords.data.data(), coords.data(),
                    saved_coords.data.size());
        const std::string coords_path = out_ply + ".coords.samt";
        if (!save_raw_tensor(coords_path, saved_coords)) {
            LOGE("e2e: failed to save SS coordinates: %s", coords_path.c_str());
            return 1;
        }
        LOGI("e2e-ss: wrote %lld sparse coords to %s",
             static_cast<long long>(n_coord), coords_path.c_str());
        return 0;
    }

    // ---------------- stage B: structured latent -------------------------
    if (!skip_cond) {
        Stage slat_emb;
        if (!slat_emb.open(models_dir + "/slat_generator-" + slat_dt + ".gguf",
                           be_name, nthreads, "slat_condition_generator",
                           static_cast<Backend*>(opt.shared_backend))) {
            LOGE("e2e: failed to load slat_generator");
            return 1;
        }
        LOGI("e2e: slat_generator loaded (%s)", slat_dt.c_str());

        // slat DINOs (prenorm mode: 1374 tokens each, registers kept) run on a
        // dedicated stage (fresh gallocr); the flow stage is opened after.
        int64_t n_sdino = 0;
        auto sd = run_dino_batch(slat_emb,
                                 {{"cemb.emb0", &ci.image},
                                  {"cemb.emb0", &ci.rgb_image},
                                  {"cemb.emb1", &ci.mask3},
                                  {"cemb.emb1", &ci.rgb_image_mask3}},
                                 true, n_sdino, nullptr,
                                 opt.cond_manual_attention);
        if (sd.empty() || sd[0].empty()) {
            LOGE("e2e: slat dino failed");
            return 1;
        }
        std::vector<float> s_dimg = std::move(sd[0]), s_drgb = std::move(sd[1]);
        std::vector<float> s_dmask = std::move(sd[2]),
                           s_drgbm = std::move(sd[3]);
        LOGI("e2e: slat dino tokens done (n=%lld each)", (long long)n_sdino);

        // slat_cond loaded (embedder or flow dump)
        {
            GraphContext gctx;
            std::vector<FuserSeg> segs = {
                    {1024, n_sdino, 0, 0, std::move(s_dimg)},
                    {1024, n_sdino, 0, 1, std::move(s_drgb)},
                    {1024, n_sdino, 1, 0, std::move(s_dmask)},
                    {1024, n_sdino, 1, 1, std::move(s_drgbm)},
            };
            ggml_tensor* out = build_fuser(gctx, *slat_emb.model, segs, "cemb");
            ggml_cgraph* graph =
                    ggml_new_graph_custom(gctx.ctx(), 16384, false);
            ggml_build_forward_expand(graph, out);
            record_dtype_contract_graph("slat_condition_fuser", graph);
            if (!slat_emb.backend->alloc(graph)) {
                LOGE("e2e: slat fuser alloc failed");
                return 1;
            }
            for (auto& s : segs) {
                if (!slat_emb.backend->set_input_f32(s.tokens, s.host.data(),
                                                     s.host.size()))
                    return 1;
            }
            if (!slat_emb.backend->run(graph)) {
                LOGE("e2e: slat fuser run failed");
                return 1;
            }
            slat_emb.backend->get_tensor_f32(out, slat_cond);
        }
        slat_emb.close();
    }

    // fresh stage for the slat flow loop (CLI-verified single-graph gallocr)
    Stage slat;
    if (!slat.open(models_dir + "/slat_generator-" + slat_dt + ".gguf", be_name,
                   nthreads, "slat_flow",
                   static_cast<Backend*>(opt.shared_backend))) {
        LOGE("e2e: failed to reload slat_generator");
        return 1;
    }
    const int64_t n_slat_cond = (int64_t)slat_cond.size() / 1024;
    LOGI("e2e: slat condition tokens %lld (%.1fs)", (long long)n_slat_cond,
         elapsed_seconds());
    if (!dbg_dir.empty())
        save_raw_tensor_f32(dbg_dir + "/e2e_slat_cond_tokens.samt",
                            {1024, n_slat_cond}, slat_cond.data());

    // sparse tables from the (fixed) coords
    SlatTables tb;
    if (!tb.build(coords.data(), n_coord)) {
        LOGE("e2e: slat tables failed");
        return 1;
    }
    LOGI("e2e: sparse tables nf=%lld nc=%lld", (long long)tb.nf,
         (long long)tb.nc);

    // ---- SLat sampling: Euler steps with the pipeline CFG schedule ----
    // pipeline.yaml overrides the SLat configuration to strength=1 and
    // rescale_t=1; override_slat_generator_cfg_config supplies [0, 500].
    // The deployment pipeline distills the trajectory to 12 steps while the
    // repository trajectory gate fixes 25; the count is opt-driven.
    constexpr int kSlatGateSteps = 25;
    if (opt.slat_steps < 1 || opt.slat_steps > kSlatGateSteps) {
        LOGE("e2e: slat steps must be in [1, %d], got %d", kSlatGateSteps,
             opt.slat_steps);
        return 1;
    }
    const int kSlatSteps = opt.slat_steps;
    constexpr float kSlatRescaleT = 1.0f;
    constexpr float kSlatCfgStrength = 1.0f;
    constexpr float kSlatCfgStart = 0.0f;
    constexpr float kSlatCfgEnd = 500.0f;
    std::vector<float> x_slat(8 * tb.nf);
    const bool replayed_slat_noise =
            load_slat_noise_for_coords(noise_dir, coords, seed, x_slat);
    if (!replayed_slat_noise) {
        if (!noise_dir.empty()) {
            LOGE("e2e: --noise-dir must contain valid SLat initial noise and "
                 "coordinates");
            return 1;
        }
        if (!fill_sampling_noise(x_slat)) return 1;
    }

    GraphContext sg;
    SlatFlowGraph slg;
    slg.g = &sg;
    slg.m = slat.model.get();
    slg.tb = &tb;
    slg.n_cond_tokens = n_slat_cond;
    slg.use_cuda_spconv = be_is_cuda(be_name);
    const bool split_slat_final =
            ggml_is_quantized(slat.model->get("dit.out_layer.weight")->type) &&
            slat.backend->has_gpu() && opt.debug_stage.empty();
    // Quantized final projections need a split graph to avoid changing the
    // trunk's allocation layout.  Keep an independent allocator for that
    // small final graph so both graphs stay allocated across all Euler/CFG
    // evaluations instead of reallocating them 100 times per sample.
    std::unique_ptr<Backend> slat_final_backend;
    if (split_slat_final) {
        slat_final_backend = Backend::create(be_name, nthreads);
        if (!slat_final_backend) {
            LOGE("e2e: failed to create SLat final-projection backend");
            return 1;
        }
        slat_final_backend->set_profile_label("slat_final_projection");
    }
    if (split_slat_final) {
        slg.debug_stage = "pre_final";
    } else if (const char* debug_stage = opt.debug_stage.c_str()) {
        slg.debug_stage = debug_stage;
    }
    auto s_outs = slg.build();
    // Matches the independently validated slat-step graph. The graph has
    // about 2.3k nodes, so 32k leaves ample headroom without changing the
    // context allocation layout between CLI and end-to-end execution.
    ggml_cgraph* s_graph = ggml_new_graph_custom(sg.ctx(), 32768, false);
    for (auto* o : s_outs) {
        ggml_set_output(o);
        ggml_build_forward_expand(s_graph, o);
    }
    record_dtype_contract_graph("slat_flow", s_graph);
    if (!slat.backend->alloc(s_graph)) {
        LOGE("e2e: slat flow alloc failed");
        return 1;
    }
    if (opt.progress) opt.progress(kStageSlatFlow, 0, 0);
    LOGI("e2e: slat flow graph built (%d nodes)", ggml_graph_n_nodes(s_graph));

    // A full Q8 SLat graph changes the gallocr liveness layout enough for the
    // CUDA get_rows kernel to read an invalid source row. Keep the large sparse
    // trunk and the final norm/projection in independent graph allocations.
    GraphContext final_ctx;
    ggml_tensor* final_in = nullptr;
    ggml_tensor* final_out = nullptr;
    ggml_cgraph* final_graph = nullptr;
    if (split_slat_final) {
        final_in = final_ctx.input_f32("slat_pre_final", {128, tb.nf});
        ggml_tensor* final_weight = slat.model->get("dit.out_layer.weight");
        if (!ggml_is_quantized(final_weight->type) &&
            final_weight->type != GGML_TYPE_F32)
            final_weight =
                    ggml_cast(final_ctx.ctx(), final_weight, GGML_TYPE_F32);
        ggml_tensor* final_bias = slat.model->get("dit.out_layer.bias");
        if (final_bias->type != GGML_TYPE_F32)
            final_bias = ggml_cast(final_ctx.ctx(), final_bias, GGML_TYPE_F32);
        final_out = gb_linear(final_ctx.ctx(), final_weight, final_bias,
                              gb_layer_norm(final_ctx.ctx(), final_in, nullptr,
                                            nullptr, 1e-6f));
        final_graph = ggml_new_graph_custom(final_ctx.ctx(), 64, false);
        ggml_set_output(final_out);
        ggml_build_forward_expand(final_graph, final_out);
        record_dtype_contract_graph("slat_final_projection", final_graph);
        if (!slat_final_backend->alloc(final_graph)) {
            LOGE("e2e: SLat final projection alloc failed");
            return 1;
        }
    }

    {
        const auto slat_schedule =
                make_euler_schedule(kSlatSteps, kSlatRescaleT);
        const std::vector<float> zeros_slat_cond(slat_cond.size(), 0.0f);
        const bool dump_slat_steps = opt.dump_slat_steps;
        // Reusing these input buffers across SLat graphs is not yet safe on
        // Vulkan: the cached path changes the final rendered entity. Keep the
        // verified per-forward upload as the default until ownership across
        // graph submissions is made explicit and regression-tested.
        const bool cache_vulkan_tables =
                strstr(slat.backend->backend_name(), "Vulkan") != nullptr &&
                opt.vulkan_table_cache;
        auto upload_slat_tables = [&]() {
            for (size_t ti = 0; ti < slg.inputs.size(); ti++) {
                ggml_tensor* input = slg.inputs[ti];
                if (!input->buffer || !slg.table_data[ti]) continue;
                const auto& host = slg.table_data[ti];
                const bool ok =
                        input->type == GGML_TYPE_F32
                                ? slat.backend->set_input_f32(
                                          input,
                                          reinterpret_cast<const float*>(
                                                  host->data()),
                                          host->size())
                                : slat.backend->set_input_i32(
                                          input, host->data(), host->size());
                if (!ok) {
                    LOGE("e2e: failed to upload SLat table %s", input->name);
                    return false;
                }
            }
            return true;
        };
        if (cache_vulkan_tables && !upload_slat_tables()) return 1;
        if (dump_slat_steps && !dbg_dir.empty()) {
            save_raw_tensor_f32(dbg_dir + "/slat_ggml_x000.samt", {8, tb.nf},
                                x_slat.data());
        }
        size_t debug_slat_forwards = 0;
        if (opt.debug_slat_forwards < 0) {
            LOGE("e2e: --debug-slat-forwards must be a positive integer");
            return 1;
        }
        if (opt.debug_slat_forwards > 0) {
            debug_slat_forwards = static_cast<size_t>(opt.debug_slat_forwards);
        } else if (opt.debug_once) {
            debug_slat_forwards = 1;
        }
        const char* debug_slat_output = opt.debug_slat_output.c_str();
        size_t completed_debug_slat_forwards = 0;

        // Loop-timing diagnostic was an environment opt-in in the source
        // runtime; AICore forbids task-level environment control, so the
        // diagnostic split is disabled and stage timings flow through the
        // common aicore_pipeline_timings contract instead.
        const bool loop_timing = false;
        double t_upload = 0, t_run = 0, t_read = 0;
        auto now_ms = [] {
            return std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now().time_since_epoch())
                    .count();
        };
        for (size_t s = 0; s < slat_schedule.size(); ++s) {
            const float t_v = slat_schedule[s].t * 1000.0f;
            const double tu0 = loop_timing ? now_ms() : 0;
            if (slg.x->buffer && !slat.backend->set_input_f32(
                                         slg.x, x_slat.data(), x_slat.size()))
                return 1;
            if (slg.cond->buffer &&
                !slat.backend->set_input_f32(slg.cond, slat_cond.data(),
                                             slat_cond.size()))
                return 1;
            if (slg.t->buffer && !slat.backend->set_input_f32(slg.t, &t_v, 1))
                return 1;
            if (!cache_vulkan_tables) {
                if (!upload_slat_tables()) return 1;
                if (opt.verify) {
                    for (ggml_tensor* t : slg.inputs) {
                        if (!t->buffer || t->type != GGML_TYPE_I32) continue;
                        std::vector<int32_t> device;
                        slat.backend->get_tensor_i32(t, device);
                        const auto [lo, hi] = std::minmax_element(
                                device.begin(), device.end());
                        LOGI("verify: %s rows [%d, %d] (%zu values)", t->name,
                             *lo, *hi, device.size());
                    }
                }
            }
            const double tu1 = loop_timing ? now_ms() : 0;
            if (!slat.backend->run(s_graph)) {
                LOGE("e2e: slat step %zu failed", s);
                return 1;
            }
            const double tr0 = loop_timing ? now_ms() : 0;
            std::vector<float> vc;
            slat.backend->get_tensor_f32(s_outs[0], vc);
            const double tr1 = loop_timing ? now_ms() : 0;
            if (loop_timing) {
                t_upload += tu1 - tu0;
                t_run += tr0 - tu1;
                t_read += tr1 - tr0;
            }
            if (split_slat_final) {
                if (!slat_final_backend->set_input_f32(final_in, vc.data(),
                                                       vc.size()) ||
                    !slat_final_backend->run(final_graph) ||
                    !slat_final_backend->get_tensor_f32(final_out, vc)) {
                    LOGE("e2e: SLat final projection failed");
                    return 1;
                }
            }
            if (debug_slat_forwards != 0 &&
                ++completed_debug_slat_forwards >= debug_slat_forwards) {
                if (debug_slat_output && *debug_slat_output) {
                    if (!save_raw_tensor_f32(debug_slat_output, {8, tb.nf},
                                             vc.data())) {
                        LOGE("e2e: failed to write SLat debug output %s",
                             debug_slat_output);
                        return 1;
                    }
                }
                LOGI("e2e: completed %zu SLat debug forward(s)",
                     completed_debug_slat_forwards);
                return 0;
            }
            const bool cfg_active = t_v >= kSlatCfgStart && t_v <= kSlatCfgEnd;
            std::vector<float> vu;
            if (cfg_active) {
                if (!cache_vulkan_tables && !upload_slat_tables()) return 1;
                if (slg.x->buffer &&
                    !slat.backend->set_input_f32(slg.x, x_slat.data(),
                                                 x_slat.size()))
                    return 1;
                if (slg.t->buffer &&
                    !slat.backend->set_input_f32(slg.t, &t_v, 1))
                    return 1;
                if (!slat.backend->set_input_f32(slg.cond,
                                                 zeros_slat_cond.data(),
                                                 zeros_slat_cond.size()))
                    return 1;
                if (!slat.backend->run(s_graph)) {
                    LOGE("e2e: slat uncond step %zu failed", s);
                    return 1;
                }
                slat.backend->get_tensor_f32(s_outs[0], vu);
                if (split_slat_final) {
                    if (!slat_final_backend->set_input_f32(final_in, vu.data(),
                                                           vu.size()) ||
                        !slat_final_backend->run(final_graph) ||
                        !slat_final_backend->get_tensor_f32(final_out, vu)) {
                        LOGE("e2e: SLat final uncond projection failed");
                        return 1;
                    }
                }
            }
            for (size_t k = 0; k < x_slat.size(); k++) {
                const float velocity =
                        cfg_active ? vc[k] + kSlatCfgStrength * (vc[k] - vu[k])
                                   : vc[k];
                x_slat[k] += slat_schedule[s].dt * velocity;
            }
            if (dump_slat_steps && !dbg_dir.empty()) {
                char name[64];
                snprintf(name, sizeof(name), "/slat_ggml_x%03zu.samt", s + 1);
                save_raw_tensor_f32(dbg_dir + name, {8, tb.nf}, x_slat.data());
            }
            if (s % 5 == 4)
                LOGI("e2e: slat step %zu/%zu", s + 1, slat_schedule.size());
        }
        if (loop_timing) {
            LOGI("e2e: slat loop timing: upload=%.2fs graph=%.2fs "
                 "readback=%.2fs over %zu runs",
                 t_upload / 1000.0, t_run / 1000.0, t_read / 1000.0,
                 slat_schedule.size());
        }
    }
    // The deployment trajectory gate consumes the normalized latent written
    // above.  Stop before decoder work so a SLat numerical diagnosis neither
    // changes the observed tensor nor spends time producing a throwaway PLY.
    if (opt.slat_flow_only) {
        LOGI("e2e: SLat flow-only replay complete");
        return 0;
    }
    LOGI("e2e: slat flow done in %.1fs", elapsed_seconds());
    // denormalize: feats * STD + MEAN (per channel)
    for (int64_t n = 0; n < tb.nf; n++)
        for (int c = 0; c < 8; c++)
            x_slat[n * 8 + c] = x_slat[n * 8 + c] * SLAT_STD[c] + SLAT_MEAN[c];
    if (!dbg_dir.empty())
        save_raw_tensor_f32(dbg_dir + "/e2e_slat_feats.samt", {8, tb.nf},
                            x_slat.data());
    slat.close();

    // ---------------- stage C: Gaussian decode + PLY ---------------------
    Stage gs;
    if (opt.progress) opt.progress(kStageGsDecode, 0, 0);
    if (!gs.open(models_dir + "/slat_decoder_gs-" + gs_dt + ".gguf", be_name,
                 nthreads, "gaussian_decoder",
                 static_cast<Backend*>(opt.shared_backend))) {
        LOGE("e2e: failed to load slat_decoder_gs-%s", gs_dt.c_str());
        return 1;
    }
    GsTables gtb;
    if (!gtb.build(coords.data(), n_coord)) {
        LOGE("e2e: gs tables failed");
        return 1;
    }
    // offset perturbation (hammersley) from the GGUF: ggml ne = [3, 32].
    // The converter writes torch's [3, 32] C-order buffer, which is already
    // flattened as [gaussian * 3 + channel] in ggml memory.
    ggml_tensor* pert = gs.model->get("gsdec.offset_perturbation");
    GGML_ASSERT(pert && pert->ne[0] == 3 && pert->ne[1] == GS_NGAUSS);
    std::vector<float> perturb;
    gs.backend->get_tensor_f32(pert, perturb);
    GGML_ASSERT(perturb.size() == static_cast<size_t>(GS_NGAUSS * 3));

    GraphContext gg;
    GsDecoderGraph gdg;
    gdg.g = &gg;
    gdg.m = gs.model.get();
    gdg.tb = &gtb;
#ifdef SAM3D_USE_CUDA
    gdg.use_pytorch_cuda_attention =
            std::strstr(gs.backend->backend_name(), "CUDA") != nullptr &&
            !opt.gs_portable_attention;
#endif
    ggml_tensor* g_x = gg.input_f32("x", {8, tb.nf});
    gdg.x = g_x;
    gdg.inputs.push_back(g_x);
    gdg.table_data.push_back(nullptr);
    auto g_outs = gdg.build();  // [0] = raw feats (448, N)
    ggml_cgraph* g_graph = ggml_new_graph_custom(gg.ctx(), 65536, false);
    for (auto* o : g_outs) {
        ggml_set_output(o);
        ggml_build_forward_expand(g_graph, o);
    }
    record_dtype_contract_graph("gaussian_decoder", g_graph);
    if (!gs.backend->alloc(g_graph)) {
        LOGE("e2e: gs alloc failed");
        return 1;
    }
    if (!gs.backend->set_input_f32(g_x, x_slat.data(), x_slat.size())) return 1;
    for (size_t ti = 0; ti < gdg.inputs.size(); ti++) {
        ggml_tensor* t = gdg.inputs[ti];
        if (!t->buffer || !gdg.table_data[ti]) continue;
        const auto& host = gdg.table_data[ti];
        const bool ok =
                t->type == GGML_TYPE_F32
                        ? gs.backend->set_input_f32(
                                  t,
                                  reinterpret_cast<const float*>(host->data()),
                                  host->size())
                        : gs.backend->set_input_i32(t, host->data(),
                                                    host->size());
        if (!ok) return 1;
    }
    if (!gs.backend->run(g_graph)) {
        LOGE("e2e: gs run failed");
        return 1;
    }
    std::vector<float> raw;
    gs.backend->get_tensor_f32(g_outs[0], raw);
    gs.close();
    LOGI("e2e: gaussian decode done in %.1fs", elapsed_seconds());

    // host to_representation (decoder_gs.py) + PLY
    const int64_t n_tok = (int64_t)raw.size() / 448;
    const int64_t n_gs_total = n_tok * GS_NGAUSS;
    std::vector<float> xyz(n_gs_total * 3), fdc(n_gs_total * 3), op(n_gs_total),
            scl(n_gs_total * 3), rot(n_gs_total * 4);
    for (int64_t n = 0; n < n_tok; n++) {
        const float* r = &raw[n * 448];
        const float cb = (float)coords[n * 4 + 1];
        const float cby = (float)coords[n * 4 + 2];
        const float cbz = (float)coords[n * 4 + 3];
        for (int g = 0; g < GS_NGAUSS; g++) {
            float* X = &xyz[(n * GS_NGAUSS + g) * 3];
            X[0] = (cb + 0.5f) / 64.f;
            X[1] = (cby + 0.5f) / 64.f;
            X[2] = (cbz + 0.5f) / 64.f;
            // offset: raw*lr(1.0) + perturbation; tanh / 64 * 0.5 * voxel_size
            for (int c = 0; c < 3; c++) {
                float o = r[g * 3 + c] + perturb[g * 3 + c];
                X[c] += tanhf(o) / 64.f * 0.5f * GS_VOXEL_SIZE;
            }
            float* F = &fdc[(n * GS_NGAUSS + g) * 3];
            for (int c = 0; c < 3; c++) F[c] = r[96 + g * 3 + c];
            float* S = &scl[(n * GS_NGAUSS + g) * 3];
            for (int c = 0; c < 3; c++) S[c] = r[192 + g * 3 + c];
            float* R = &rot[(n * GS_NGAUSS + g) * 4];
            for (int c = 0; c < 4; c++) R[c] = r[288 + g * 4 + c] * 0.1f;
            op[n * GS_NGAUSS + g] = r[416 + g];
        }
    }
    if (opt.artifacts) {
        // Display-ready in-memory artifacts for the C API path: the splat
        // centers use the PLY/world domain (xyz - 0.5, matching the mesh
        // vertices) and the colors are 0.5 + SH_C0 * f_dc clamped to 0..1.
        opt.artifacts->gaussian_count = n_gs_total;
        opt.artifacts->splat_centers.resize(static_cast<size_t>(n_gs_total) *
                                            3);
        opt.artifacts->splat_rgb.resize(static_cast<size_t>(n_gs_total) * 3);
        for (int64_t i = 0; i < n_gs_total; ++i) {
            for (int c = 0; c < 3; ++c) {
                opt.artifacts->splat_centers[i * 3 + c] = xyz[i * 3 + c] - 0.5f;
                float rgb = 0.5f + 0.28209479177387814f * fdc[i * 3 + c];
                opt.artifacts->splat_rgb[i * 3 + c] =
                        rgb < 0.f ? 0.f : (rgb > 1.f ? 1.f : rgb);
            }
        }
        if (opt.scene_attributes) {
            // Composer-facing PLY-semantic interchange rows. Every expression
            // mirrors write_gaussian_ply in the same order so the in-memory
            // path and the PLY export stay bit-identical inputs for the
            // scene composer and the upstream scene-assemble flow.
            const float scale_bias = softplus_inv(GS_SCALING_BIAS);
            const float op_bias = inv_sigmoid(GS_OPACITY_BIAS);
            opt.artifacts->splat_sh0.resize(static_cast<size_t>(n_gs_total) *
                                            3);
            opt.artifacts->splat_log_scale.resize(
                    static_cast<size_t>(n_gs_total) * 3);
            opt.artifacts->splat_opacity_logit.resize(
                    static_cast<size_t>(n_gs_total));
            opt.artifacts->splat_rot_ply.resize(
                    static_cast<size_t>(n_gs_total) * 4);
            for (int64_t i = 0; i < n_gs_total; ++i) {
                for (int c = 0; c < 3; ++c) {
                    opt.artifacts->splat_sh0[i * 3 + c] = fdc[i * 3 + c];
                    const float sp = softplus(scl[i * 3 + c] + scale_bias);
                    opt.artifacts->splat_log_scale[i * 3 + c] = logf(
                            sqrtf(sp * sp + GS_MIN_KERNEL * GS_MIN_KERNEL));
                }
                opt.artifacts->splat_opacity_logit[i] = op[i] + op_bias;
                opt.artifacts->splat_rot_ply[i * 4 + 0] = rot[i * 4 + 0] + 1.f;
                opt.artifacts->splat_rot_ply[i * 4 + 1] = rot[i * 4 + 1];
                opt.artifacts->splat_rot_ply[i * 4 + 2] = rot[i * 4 + 2];
                opt.artifacts->splat_rot_ply[i * 4 + 3] = rot[i * 4 + 3];
            }
        }
    }
    const bool export_ply = !out_ply.empty();
    if (export_ply &&
        !write_gaussian_ply(out_ply, n_gs_total, xyz, fdc, op, scl, rot)) {
        LOGE("e2e: failed to write %s", out_ply.c_str());
        return 1;
    }
    // The PLY is the durable Gaussian interchange artifact. Release the
    // decoder-only host buffers before the optional mesh/PBR branch so the
    // native 100-view bake has the same memory envelope as pbr-assemble.
    raw.clear();
    raw.shrink_to_fit();
    xyz.clear();
    xyz.shrink_to_fit();
    fdc.clear();
    fdc.shrink_to_fit();
    op.clear();
    op.shrink_to_fit();
    scl.clear();
    scl.shrink_to_fit();
    rot.clear();
    rot.shrink_to_fit();

    const bool export_raw_mesh = !out_mesh_vertices.empty();
    // The mesh stage runs for file exports (CLI/PBR interchange), for the
    // in-memory artifact sink (opt.decode_mesh), or for the PBR branch.
    const bool want_mesh = export_raw_mesh || !out_pbr.empty() ||
                           (opt.artifacts && opt.decode_mesh);
    if (want_mesh) {
        MeshDecoderRunOptions mesh_options;
        mesh_options.model_path =
                models_dir + "/slat_decoder_mesh-" + mesh_dt + ".gguf";
        mesh_options.backend = be_name;
        mesh_options.threads = nthreads;
        MeshDecoderRunResult mesh_decode;
        std::string mesh_error;
        if (opt.progress) opt.progress(kStageMeshDecode, 0, 0);
        if (!run_mesh_decoder(x_slat.data(), coords.data(), tb.nf, mesh_options,
                              mesh_decode, mesh_error)) {
            LOGE("e2e: mesh decoder failed: %s", mesh_error.c_str());
            return 1;
        }
        if (mesh_decode.channels != 101 || mesh_decode.resolution != 256) {
            LOGE("e2e: mesh decoder output is [%lld, %lld] at resolution %d, "
                 "expected [101, N] at 256",
                 static_cast<long long>(mesh_decode.channels),
                 static_cast<long long>(mesh_decode.token_count),
                 mesh_decode.resolution);
            return 1;
        }
        FlexiCubesResult flexicubes;
        if (!decode_flexicubes(mesh_decode.features.data(),
                               mesh_decode.coordinates.data(),
                               mesh_decode.token_count, mesh_decode.resolution,
                               flexicubes, mesh_error)) {
            LOGE("e2e: FlexiCubes extraction failed: %s", mesh_error.c_str());
            return 1;
        }
        if (opt.artifacts) {
            // In-memory mesh for the C API path: same world domain as the
            // splat centers; the SAMT file exports stay a CLI-only boundary.
            opt.artifacts->mesh_vertices = flexicubes.positions;
            opt.artifacts->mesh_triangles = flexicubes.indices;
        }
        if (export_raw_mesh) {
            std::vector<int32_t> mesh_faces;
            mesh_faces.reserve(flexicubes.indices.size());
            for (uint32_t index : flexicubes.indices) {
                if (index > static_cast<uint32_t>(
                                    std::numeric_limits<int32_t>::max())) {
                    LOGE("e2e: FlexiCubes index exceeds SAMT I32 range");
                    return 1;
                }
                mesh_faces.push_back(static_cast<int32_t>(index));
            }
            const int64_t vertex_count =
                    static_cast<int64_t>(flexicubes.positions.size() / 3);
            const int64_t face_count =
                    static_cast<int64_t>(mesh_faces.size() / 3);
            if (!save_raw_tensor_f32(out_mesh_vertices, {3, vertex_count},
                                     flexicubes.positions.data()) ||
                !save_raw_tensor_i32(out_mesh_faces, {3, face_count},
                                     mesh_faces.data())) {
                LOGE("e2e: failed to write raw FlexiCubes mesh interchange "
                     "artifacts");
                return 1;
            }
            LOGI("e2e: wrote raw FlexiCubes mesh (%lld vertices, %lld faces)",
                 static_cast<long long>(vertex_count),
                 static_cast<long long>(face_count));
        }
        if (!out_pbr.empty()) {
#if !defined(SAM3D_USE_NVDIFFRAST_CUDARASTER) || !defined(SAM3D_USE_MESHFIX_GPL)
            LOGE("e2e: --pbr-out requires a CUDA native-PBR build with "
                 "SAM3D_GGML_NVDIFFRAST_NONCOMMERCIAL=ON and "
                 "SAM3D_GGML_MESHFIX_GPL=ON");
            return 1;
#else
            if (!export_ply) {
                LOGE("e2e: --pbr-out reads back the Gaussian PLY; request an "
                     "out_ply export together with the PBR assembly");
                return 1;
            }
            NativeMesh decoded_mesh;
            decoded_mesh.positions = std::move(flexicubes.positions);
            decoded_mesh.indices = std::move(flexicubes.indices);
            decoded_mesh.vertex_attributes =
                    std::move(flexicubes.vertex_attributes);
            GaussianSplatSet splats;
            if (!load_gaussian_splat_ply(out_ply, splats, mesh_error)) {
                LOGE("e2e: failed to reopen native Gaussian PLY for PBR: %s",
                     mesh_error.c_str());
                return 1;
            }
            NativePbrPipelineConfig pbr_config;
            pbr_config.random_seed = seed;
            NativePbrPipelineResult pbr_result;
            if (!assemble_official_pbr_cuda(decoded_mesh, splats, pbr_config,
                                            pbr_result, mesh_error) ||
                !write_pbr_glb(out_pbr, pbr_result.mesh, mesh_error)) {
                LOGE("e2e: native PBR assembly failed: %s", mesh_error.c_str());
                return 1;
            }
            auto texture_path = std::filesystem::path(out_pbr);
            texture_path.replace_extension(".base_color.png");
            if (!write_rgba_png(texture_path.string(),
                                pbr_result.mesh.material.base_color_texture,
                                mesh_error)) {
                LOGE("e2e: failed to write base-color texture: %s",
                     mesh_error.c_str());
                return 1;
            }
            LOGI("e2e: wrote %s (%zu rendered views, %zu final triangles)",
                 out_pbr.c_str(), pbr_result.rendered_views,
                 pbr_result.final_faces);
#endif
        }
    }
    LOGI("e2e: %s (%lld gaussians) in %.1fs for native generation stages",
         export_ply ? out_ply.c_str() : "in-memory artifacts requested",
         (long long)n_gs_total, elapsed_seconds());
    if (!dtype_contract.complete(dtype_contract_error)) {
        LOGE("e2e: failed to write dtype contract: %s",
             dtype_contract_error.c_str());
        return 1;
    }
    return 0;
}

// Legacy entry point kept for the remaining CLI commands.
RunResult run_pipeline(const CliOptions& opts) {
    RunResult r;
    if (opts.condition_dir.empty()) {
        r.error =
                "condition_dir is required; provide a dump_e2e_stages.py "
                "output directory";
        return r;
    }
    if (opts.out_ply.empty()) {
        r.error = "out_ply is required";
        return r;
    }
    if (opts.models_dir.empty()) {
        r.error = "models_dir is required";
        return r;
    }
    // Keep the C API and CLI on one implementation so parity fixes cannot
    // drift between entry points. All knobs travel through E2eOptions.
    E2eOptions e2e;
    e2e.models_dir = opts.models_dir;
    e2e.cond_dir = opts.condition_dir;
    // image-to-3d generates noise internally (empty noise_dir = fresh noise)
    e2e.out_ply = opts.out_ply;
    e2e.dbg_dir = opts.dump_dir;
    e2e.backend = opts.backend.empty() ? "auto" : opts.backend;
    e2e.threads = opts.n_threads;
    e2e.seed = static_cast<unsigned>(opts.seed);
    e2e.dtype =
            "f32";  // historical image-to-3d default (SAM3D_E2E_DTYPE unset)
    const int rc = cmd_e2e(e2e);
    r.ok = rc == 0;
    if (!r.ok) r.error = "end-to-end graph session failed; see log output";
    return r;
}

namespace {

class ScopedConditionDirectory {
public:
    ~ScopedConditionDirectory() {
        if (!remove_on_exit_) return;
        std::error_code error;
        std::filesystem::remove_all(path_, error);
        if (error) {
            LOGW("image-to-3d: unable to remove temporary conditions %s: %s",
                 path_.c_str(), error.message().c_str());
        }
    }

    bool create(const std::string& requested_path, std::string& error) {
        if (!requested_path.empty()) {
            path_ = requested_path;
            std::error_code filesystem_error;
            std::filesystem::create_directories(path_, filesystem_error);
            if (filesystem_error) {
                error = "cannot create condition directory '" + path_ +
                        "': " + filesystem_error.message();
                return false;
            }
            return true;
        }

        std::error_code filesystem_error;
        const std::filesystem::path root =
                std::filesystem::temp_directory_path(filesystem_error);
        if (filesystem_error) {
            error = "cannot determine the system temporary directory: " +
                    filesystem_error.message();
            return false;
        }
        const auto nonce = static_cast<unsigned long long>(
                std::chrono::steady_clock::now().time_since_epoch().count());
        for (unsigned attempt = 0; attempt < 128; ++attempt) {
            const std::filesystem::path candidate =
                    root / ("sam3d-native-conditions-" + std::to_string(nonce) +
                            "-" + std::to_string(attempt));
            filesystem_error.clear();
            if (std::filesystem::create_directory(candidate,
                                                  filesystem_error)) {
                path_ = candidate.string();
                remove_on_exit_ = true;
                return true;
            }
            if (filesystem_error &&
                filesystem_error != std::errc::file_exists) {
                error = "cannot create temporary condition directory: " +
                        filesystem_error.message();
                return false;
            }
        }
        error = "cannot reserve a unique temporary condition directory";
        return false;
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
    bool remove_on_exit_ = false;
};

bool merge_binary_mask_into_alpha(RgbaImage& image,
                                  const std::string& mask_path,
                                  std::string& error) {
    if (mask_path.empty()) return true;
    RgbaImage mask;
    if (!load_rgba_image(mask_path, mask, error)) return false;
    if (mask.width != image.width || mask.height != image.height) {
        error = "mask dimensions must match the input image";
        return false;
    }
    for (size_t pixel = 0;
         pixel < static_cast<size_t>(image.width) * image.height; ++pixel) {
        // notebook/inference.py::load_mask applies `mask > 0` and selects the
        // last decoded channel. Preserve that binary alpha contract exactly.
        image.rgba[pixel * 4 + 3] = mask.rgba[pixel * 4 + 3] == 0 ? 0 : 255;
    }
    return true;
}

}  // namespace

// Shared validation used by both the session init and the one-shot wrapper.
namespace {

bool validate_session_config(const ImageTo3DOptions& opts, std::string& error) {
    if (opts.models_dir.empty() || opts.moge_model.empty()) {
        error = "models_dir and moge_model are required";
        return false;
    }
    if (opts.dtype.empty()) {
        error = "dtype is required";
        return false;
    }
    const auto valid_dtype = [](const std::string& dtype) {
        return dtype == "f32" || dtype == "f16" || dtype == "q4_0" ||
               dtype == "q4_1" || dtype == "q4_k" || dtype == "q8_0";
    };
    if (!valid_dtype(opts.dtype)) {
        error = "unsupported dtype '" + opts.dtype + "'";
        return false;
    }
    if (opts.n_threads <= 0) {
        error = "n_threads must be positive";
        return false;
    }
    return true;
}

}  // namespace

struct ImageTo3DSession::Impl {
    ImageTo3DOptions config;
    std::unique_ptr<Backend> backend;
    // The MoGe weights are loaded lazily per request and released right after
    // the forward: keeping 600 MB resident next to the generative stages
    // overflows a 12 GiB card, and mmap makes the reload nearly free while a
    // cache hit skips it entirely.
    std::unique_ptr<GGUFModel> moge_model;

    // Process-shared physical backend handles (runtime registry leases).
    std::vector<aicore::runtime::BackendLease> leases() const {
        return backend ? backend->leases()
                       : std::vector<aicore::runtime::BackendLease>();
    }

    // RGB-only MoGe reuse: the point map depends on the RGB channels and the
    // inference options only, never on the mask alpha, so a request whose RGB
    // bytes match the previous request skips the MoGe forward entirely. The
    // mask merge and condition preprocessing still run every request.
    std::vector<uint8_t> cached_rgb;
    int cached_width = 0;
    int cached_height = 0;
    bool cached_valid = false;
    MogeInferenceResult cached_moge;

    bool init(const ImageTo3DOptions& config_in, std::string& error) {
        if (!validate_session_config(config_in, error)) return false;
        config = config_in;
        backend = Backend::create(config.backend.c_str(), config.n_threads);
        if (!backend) {
            error = "failed to create the requested MoGe backend";
            return false;
        }
        return true;
    }

    RunResult run(const ImageTo3DOptions& opts) {
        RunResult result;
        if (!opts.image_override && opts.image_path.empty()) {
            result.error = "image_path is required";
            return result;
        }
        if (opts.out_ply.empty() && opts.artifacts == nullptr) {
            // The pipeline needs at least one output: the file export or the
            // in-memory artifact sink (the C API path).
            result.error = "out_ply or artifacts is required";
            return result;
        }
        if (opts.progress) opts.progress(0 /*load*/, 0, 0);
        if (!validate_session_config(opts, result.error)) return result;
        // The session fixes models/backend/dtype/threads; a mismatching
        // request would silently change numerics or reuse foreign weights.
        if (opts.models_dir != config.models_dir ||
            opts.moge_model != config.moge_model ||
            opts.backend != config.backend || opts.dtype != config.dtype ||
            opts.n_threads != config.n_threads) {
            result.error =
                    "request settings must match the session configuration "
                    "(models_dir, moge_model, backend, dtype, n_threads)";
            return result;
        }
        if (opts.out_mesh_vertices.empty() != opts.out_mesh_faces.empty()) {
            result.error =
                    "out_mesh_vertices and out_mesh_faces must be provided "
                    "together";
            return result;
        }
        if (!opts.noise_dir.empty()) {
            std::error_code filesystem_error;
            if (!std::filesystem::is_directory(opts.noise_dir,
                                               filesystem_error) ||
                filesystem_error) {
                result.error =
                        "noise_dir must be a readable official stage directory";
                return result;
            }
        }

        RgbaImage image;
        std::string error;
        if (opts.image_override) {
            image.width = opts.image_override->width;
            image.height = opts.image_override->height;
            image.rgba = opts.image_override->rgba;
            if (image.width <= 0 || image.height <= 0 ||
                image.rgba.size() !=
                        static_cast<size_t>(image.width) * image.height * 4) {
                result.error = "invalid in-memory RGBA input";
                return result;
            }
        } else if (!load_rgba_image(opts.image_path, image, error) ||
                   !merge_binary_mask_into_alpha(image, opts.mask_path,
                                                 error)) {
            result.error = error;
            return result;
        }

        // RGB fingerprint of the mask-merged image; only the first three
        // channels participate because MoGe never sees alpha.
        const size_t pixels = static_cast<size_t>(image.width) * image.height;
        std::vector<uint8_t> rgb(pixels * 3);
        for (size_t p = 0; p < pixels; ++p) {
            rgb[p * 3 + 0] = image.rgba[p * 4 + 0];
            rgb[p * 3 + 1] = image.rgba[p * 4 + 1];
            rgb[p * 3 + 2] = image.rgba[p * 4 + 2];
        }
        const bool cache_hit = !opts.disable_moge_pointmap_cache &&
                               cached_valid && cached_width == image.width &&
                               cached_height == image.height &&
                               cached_rgb == rgb;

        MogeInferenceResult moge_result;
        if (cache_hit) {
            // The cached result is immutable and the condition preprocessing
            // below only reads it, so sharing across requests is safe.
            moge_result = cached_moge;
            result.output.stats.moge_pointmap_reused = true;
            LOGI("image-to-3d: reusing the cached MoGe point map for identical "
                 "RGB");
        } else {
            if (opts.progress) opts.progress(1 /*pointmap*/, 0, 0);
            const auto moge_started = std::chrono::steady_clock::now();
            // A request-scoped MoGe backend keeps the CUDA pool from
            // accumulating cross-request fragmentation: per-object token
            // counts vary, the pool never returns chunks, and late objects in
            // a batch would OOM on the pointpatch reserve. The mmap-backed
            // weight reload is cheap and the pool dies with the backend.
            auto moge_backend =
                    Backend::create(config.backend.c_str(), config.n_threads);
            if (!moge_backend) {
                result.error = "failed to create the MoGe backend";
                return result;
            }
            // The request-scoped MoGe backend owns its own registry leases
            // (invisible to the caller-held session lock); hold them for the
            // whole forward+teardown so shared GPU handles stay serialized.
            aicore::runtime::BackendLeaseLock moge_lease_lock =
                    aicore::runtime::lock_backend_leases(
                            moge_backend->leases());
            moge_model = std::make_unique<GGUFModel>();
            if (!moge_model->load(config.moge_model,
                                  moge_backend->weights_buffer_type())) {
                result.error = "failed to load MoGe GGUF: " + config.moge_model;
                return result;
            }
            MogeInferenceOptions moge_options;
            // The official pipeline feeds the condition chain the FULL finite
            // point map (its raw receipt has zero NaN/Inf; the predicted-mask
            // inf fill is never seen downstream). Keeping apply_mask on would
            // inject Inf that the bilinear upsample turns into NaN and that
            // resize_pointmap_to_image then re-spreads, which is what diverged
            // the seven small-object SS trajectories.
            moge_options.apply_mask = false;
            if (!run_moge_inference(image, *moge_model, *moge_backend,
                                    moge_options, moge_result, error)) {
                result.error = "native MoGe inference failed: " + error;
                return result;
            }
            // Release the MoGe weights AND its backend before the generative
            // stages allocate; the mmap-backed reload on the next cache miss
            // is cheap and the freed pool removes the fragmentation source.
            moge_model.reset();
            moge_backend.reset();
            LOGI("image-to-3d: moge done in %.1fs",
                 std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - moge_started)
                         .count());
            cached_rgb = std::move(rgb);
            cached_width = image.width;
            cached_height = image.height;
            cached_moge = moge_result;
            cached_valid = true;
        }

        NativeConditionInputs condition_inputs;
        if (!preprocess_ss_conditions(image, moge_result.pointmap_pytorch3d,
                                      moge_result.width, moge_result.height, {},
                                      condition_inputs, error)) {
            result.error = "native condition preprocessing failed: " + error;
            return result;
        }
        // The conditions_out path is a caller-requested condition dump (a
        // replay/inspection artifact): create the directory and write the
        // SAMT files as before. The default C API path hands the condition
        // tensors to cmd_e2e in memory — no scratch directory and no
        // condition file IO in the end-to-end request.
        bool cond_memory_path = false;
        std::string cond_dir_for_e2e;
        if (opts.conditions_out.empty()) {
            cond_memory_path = true;
        } else {
            ScopedConditionDirectory conditions;
            if (!conditions.create(opts.conditions_out, error)) {
                result.error = error;
                return result;
            }
            if (!write_ss_conditions(conditions.path(), condition_inputs,
                                     error)) {
                result.error =
                        "native condition preprocessing failed: " + error;
                return result;
            }
            cond_dir_for_e2e = conditions.path();
        }

        E2eOptions e2e;
        e2e.models_dir = opts.models_dir;
        e2e.cond_dir = cond_dir_for_e2e;
        // image-to-3d generates noise internally (empty noise_dir = fresh
        // noise)
        e2e.out_ply = opts.out_ply;
        e2e.out_pbr = opts.out_pbr;
        e2e.out_mesh_vertices = opts.out_mesh_vertices;
        e2e.out_mesh_faces = opts.out_mesh_faces;
        e2e.out_pose = opts.out_pose;
        e2e.out_dtype_contract = opts.out_dtype_contract;
        e2e.backend = opts.backend.empty() ? "auto" : opts.backend;
        e2e.threads = opts.n_threads;
        e2e.seed = static_cast<unsigned>(opts.seed);
        e2e.dtype = opts.dtype;
        // The image-to-3D production path uses strict F32 condition attention
        // (the flash K/V path measurably perturbs the DINO outlier channels)
        // and the official F32 SS attention contract.
        e2e.cond_manual_attention = true;
        e2e.ss_strict_attention = opts.strict_ss_attention;
        e2e.gs_portable_attention = opts.gs_portable_attention;
        e2e.philox_blocks = opts.philox_blocks;
        e2e.ss_steps = opts.ss_steps;
        e2e.slat_steps = opts.slat_steps;
        // Accuracy-diagnostic fixed noise (upstream official stage dir);
        // empty = generate from seed as the production path does.
        e2e.noise_dir = opts.noise_dir;
        // Share the backend across stages only where its allocator survives a
        // stream of differently-shaped graphs. Vulkan's sub-allocator reuses
        // freed ranges; CUDA's pool never returns chunks to the driver, so a
        // batch whose per-object token counts vary fragments the pool and the
        // late objects OOM. On CUDA each stage keeps its own backend (the
        // per-object layout already verified 27/27) while the session still
        // removes the process launch and the repeated MoGe load. The CUDA
        // family is decided from the RESOLVED backend name so an "auto"
        // request that lands on CUDA takes the same per-stage layout as an
        // explicit "cuda" request.
        e2e.shared_backend = (backend && strncasecmp(backend->backend_name(),
                                                     "cuda", 4) == 0)
                                     ? nullptr
                                     : backend.get();
        e2e.artifacts = opts.artifacts;
        e2e.decode_mesh = opts.decode_mesh;
        e2e.scene_attributes = opts.scene_attributes;
        e2e.conditions = cond_memory_path ? &condition_inputs : nullptr;
        e2e.progress = opts.progress;
        const int rc = cmd_e2e(e2e);

        result.ok = rc == 0;
        if (!result.ok) {
            result.error = "native image-to-3D session failed; see log output";
        }
        return result;
    }
};

ImageTo3DSession::ImageTo3DSession() : impl_(std::make_unique<Impl>()) {}
ImageTo3DSession::~ImageTo3DSession() = default;

std::vector<aicore::runtime::BackendLease> ImageTo3DSession::leases() const {
    return impl_->leases();
}

std::string ImageTo3DSession::backend_name() const {
    return impl_->backend ? impl_->backend->backend_name()
                          : std::string("none");
}

bool ImageTo3DSession::init(const ImageTo3DOptions& config,
                            std::string& error) {
    return impl_->init(config, error);
}

RunResult ImageTo3DSession::run(const ImageTo3DOptions& opts) {
    return impl_->run(opts);
}

RunResult run_image_to_3d(const ImageTo3DOptions& opts) {
    // One-shot wrapper: identical numerics to the session path because it is
    // the session path with a throwaway session object.
    ImageTo3DSession session;
    std::string error;
    if (!session.init(opts, error)) {
        RunResult result;
        result.error = error;
        return result;
    }
    return session.run(opts);
}

}  // namespace sam3d
