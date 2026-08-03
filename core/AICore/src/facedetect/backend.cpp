// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "backend.hpp"

#include "aicore/runtime_capi.h"
#include "common.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml_backend_utils.hpp"
#include "model_loader.hpp"
#if !defined(AICORE_BACKEND_DL)
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fd {

namespace {
// Number of graph nodes the metadata context must hold. Generous upper bound
// for the SCRFD conv heads / ArcFace / genderage / MiniFASNet graphs.
constexpr size_t kGraphSize = 16384;

struct PendingInput {
    ggml_tensor* tensor;
    const void* host;
    size_t nbytes;
};
struct PendingCapture {
    ggml_tensor* tensor;
    std::vector<float>* dst;
};

// Structural fingerprint of a freshly-built graph. Two graphs that hash equal
// compute the identical function (same node ops, types, shapes, input/capture
// counts, output shape) and so are interchangeable for compute - this is the
// per-shape cache key. Built from the graph metadata only (cheap), it both keys
// the cache and guards against two DIFFERENT call sites (e.g. the SCRFD
// detector vs the ArcFace recognizer at the same input size) colliding.
uint64_t fingerprint_graph(ggml_cgraph* gf,
                           ggml_tensor* output,
                           size_t n_inputs,
                           size_t n_caps) {
    uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t)ggml_graph_n_nodes(gf));
    mix((uint64_t)n_inputs);
    mix((uint64_t)n_caps);
    if (output) {
        mix((uint64_t)output->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) mix((uint64_t)output->ne[d]);
    }
    const int nn = ggml_graph_n_nodes(gf);
    for (int i = 0; i < nn; ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        mix((uint64_t)t->op);
        mix((uint64_t)t->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) mix((uint64_t)t->ne[d]);
        // Fold in the op params. For GGML_OP_CUSTOM nodes these hold the
        // compute callback pointer + userdata (the per-op state carrying
        // kind/pad/stride), which op/type/ne alone do NOT distinguish: two
        // blocked convs (e.g. 3x3 and 1x1 strided) can build structurally
        // identical graphs (same node count, shapes, input/capture counts,
        // output shape) while running completely different kernels. Without
        // this the per-shape cache could false-hit and replay the wrong custom
        // op, producing wrong output and out-of-bounds writes. op_params is
        // zero-initialised at tensor creation, so the unused tail is
        // deterministic across builds.
        for (int p = 0; p < (int)(GGML_MAX_OP_PARAMS / sizeof(int32_t)); ++p)
            mix((uint64_t)(uint32_t)t->op_params[p]);
        // Hash each source's data pointer. In the no_alloc build context only
        // the realized WEIGHT leaves carry a non-null ->data (their addresses
        // are owned by the ModelLoader and stable per model);
        // inputs/intermediates are null. Folding these in binds the fingerprint
        // to THIS model's weight buffers, so two distinct models that share an
        // architecture + input size cannot false-hit the same cached graph.
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor* src = t->src[s];
            mix(src ? (uint64_t)(uintptr_t)src->data : 0ull);
        }
    }
    return h;
}

// Collect the distinct backend buffers backing this graph's WEIGHT leaves. Must
// be called on the freshly-built graph BEFORE gallocr allocation: at that point
// only realized weight leaves carry a non-null ->buffer (it points at the
// owning ModelLoader's weights buffer), while inputs/intermediates are still
// unallocated (null buffer). Recording these per cache entry is what lets
// invalidate_weights_buffer() purge an entry the moment its model is freed.
std::vector<ggml_backend_buffer_t> collect_weight_buffers(ggml_cgraph* gf) {
    std::vector<ggml_backend_buffer_t> bufs;
    const int nn = ggml_graph_n_nodes(gf);
    for (int i = 0; i < nn; ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor* src = t->src[s];
            if (!src || !src->buffer) continue;
            if (std::find(bufs.begin(), bufs.end(), src->buffer) == bufs.end())
                bufs.push_back(src->buffer);
        }
    }
    return bufs;
}
}  // namespace

struct Backend::Impl {
    std::vector<ggml_backend_t> gpu_backends;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool use_sched = false;

    // Persistent per-shape graph cache (parakeet.cpp pattern). Each entry owns
    // its ggml_context (graph node STRUCTURE + metadata), the built
    // ggml_cgraph, and its OWN gallocr (a dedicated device buffer, so entries
    // never alias each other's tensor storage). The graph object AND its tensor
    // data pointers stay STABLE across calls, so on CUDA the captured CUDA
    // graph replays through cudaGraphExecUpdate instead of churning destroy +
    // reinstantiate every call.
    struct GraphEntry {
        uint64_t fp = 0;
        ggml_context* ctx = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_tensor* output = nullptr;
        std::vector<ggml_tensor*> inputs;  // stable input tensors (build order)
        std::vector<ggml_tensor*> caps;  // stable capture tensors (build order)
        // Distinct backend buffers backing this graph's WEIGHT leaves (each is
        // a ModelLoader's weights buffer). invalidate_weights_buffer() drops
        // the entry the instant any of these is freed, so a freed-then-reused
        // address can never alias into a stale replay (multi-model hosting
        // safety).
        std::vector<ggml_backend_buffer_t> weight_bufs;
        uint64_t lru = 0;
    };
    std::vector<GraphEntry> cache;
    uint64_t lru_clock = 0;
    uint64_t hits = 0;  // cache hit / miss counters (test introspection)
    uint64_t misses = 0;
    size_t cache_cap = 8;  // 0 == caching disabled (per-call)

    // Scratch context reused for the per-call graph build. The build lambda
    // runs exactly ONCE per compute() (the conv/bn helpers populate
    // caller-owned scratch vectors mid-build, so a second build pass would
    // double them) - we always build here to harvest the fresh input host
    // pointers and the structural fingerprint. On a cache MISS this scratch is
    // PROMOTED to a cache entry and a fresh scratch is allocated next call; on
    // a HIT it is ggml_reset and reused.
    ggml_context* scratch = nullptr;

    std::vector<PendingInput> pending;
    std::vector<PendingCapture> captures;
};

// Thread-local pointer to the Backend whose build lambda is currently running,
// so add_graph_input()/capture_graph_output() can route registrations. compute
// is not re-entrant on a single thread, so a single pointer suffices.
static thread_local Backend* t_active = nullptr;

Backend::Backend(int n_threads) : impl_(new Impl()) {
    ggml_common::load_backends_once();

    // Persistent-graph cache capacity (number of distinct input shapes / call
    // sites kept hot). 0 disables caching (one-shot per-call build, the legacy
    // behaviour) as an A/B kill-switch.
    if (const char* env = std::getenv("FACEDETECT_GRAPH_CACHE")) {
        const long v = std::atol(env);
        impl_->cache_cap = v >= 0 ? (size_t)v : impl_->cache_cap;
    }

    // Optional override via FACEDETECT_DEVICE: "cpu" forces CPU; a device name
    // selects that registry device (case-insensitive); unset auto-picks the
    // first GPU/IGPU device.
    const char* force = std::getenv("FACEDETECT_DEVICE");
    const std::string want = force ? force : "";
    const bool force_cpu = want == "cpu" || want == "CPU";

    auto iequals = [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::tolower((unsigned char)a[i]) !=
                std::tolower((unsigned char)b[i]))
                return false;
        return true;
    };

    if (!force_cpu) {
        if (want.empty()) {
            ggml_common::GpuBackendGroup group =
                    ggml_common::resolve_gpu_group("auto");
            if (group.primary()) {
                impl_->gpu_backends = std::move(group.gpus);
                impl_->backend = impl_->gpu_backends.front();
                device_name_ = group.primary_name();
                if (impl_->gpu_backends.size() > 1) {
                    device_name_ += " (x" +
                                    std::to_string(impl_->gpu_backends.size()) +
                                    " GPUs)";
                }
                impl_->use_sched = true;
                FD_LOG("fd::Backend using device: %s", device_name_.c_str());
            }
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                const auto type = ggml_backend_dev_type(dev);
                const char* name = ggml_backend_dev_name(dev);
                if (!name || !iequals(want, name)) continue;
                if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
                    type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    continue;
                }
                if (ggml_backend_t be = ggml_backend_dev_init(dev, nullptr)) {
                    impl_->gpu_backends = {be};
                    impl_->backend = be;
                    device_name_ = name;
                    impl_->use_sched = true;
                    FD_LOG("fd::Backend using device: %s",
                           device_name_.c_str());
                    break;
                }
            }
            if (!impl_->backend)
                FD_LOG("fd::Backend: FACEDETECT_DEVICE=%s not found; falling "
                       "back to CPU",
                       want.c_str());
        }
    }
    if (!impl_->backend) {
        impl_->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,
                                                   nullptr);
        device_name_ = "cpu";
    }
    if (!impl_->backend) {
        FD_LOG("backend init returned null");
        return;
    }
    if (impl_->use_sched) {
        impl_->cpu_backend = ggml_backend_init_by_type(
                GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!impl_->cpu_backend) {
            FD_LOG("fd::Backend: CPU fallback init failed; disabling sched");
            impl_->use_sched = false;
        }
    }
    set_n_threads(n_threads);
}

Backend::~Backend() {
    if (impl_) {
        for (Impl::GraphEntry& e : impl_->cache) {
            if (e.galloc) ggml_gallocr_free(e.galloc);
            if (e.ctx) ggml_free(e.ctx);
        }
        impl_->cache.clear();
        if (impl_->scratch) ggml_free(impl_->scratch);
        if (impl_->sched) ggml_backend_sched_free(impl_->sched);
        if (impl_->cpu_backend) ggml_backend_free(impl_->cpu_backend);
        for (ggml_backend_t gpu : impl_->gpu_backends) {
            if (gpu) ggml_backend_free(gpu);
        }
        impl_->gpu_backends.clear();
        impl_->backend = nullptr;
        delete impl_;
        impl_ = nullptr;
    }
}

void Backend::set_n_threads(int n_threads) {
    n_threads_ = n_threads > 0 ? n_threads : 1;
    if (impl_ && impl_->backend &&
        ggml_common::is_cpu_backend(impl_->backend)) {
        ggml_common::set_cpu_threads(impl_->backend, n_threads_);
    }
    if (impl_ && impl_->cpu_backend) {
        ggml_common::set_cpu_threads(impl_->cpu_backend, n_threads_);
    }
}

ggml_backend_t Backend::handle() const {
    return impl_ ? impl_->backend : nullptr;
}

void Backend::invalidate_weights_buffer(ggml_backend_buffer_t buf) {
    if (!impl_ || !buf) return;
    for (size_t i = 0; i < impl_->cache.size();) {
        Impl::GraphEntry& e = impl_->cache[i];
        const bool refs = std::find(e.weight_bufs.begin(), e.weight_bufs.end(),
                                    buf) != e.weight_bufs.end();
        if (refs) {
            if (e.galloc) ggml_gallocr_free(e.galloc);
            if (e.ctx) ggml_free(e.ctx);
            impl_->cache.erase(impl_->cache.begin() + (long)i);
        } else {
            ++i;
        }
    }
}

size_t Backend::cache_size() const { return impl_ ? impl_->cache.size() : 0; }
uint64_t Backend::cache_hits() const { return impl_ ? impl_->hits : 0; }
uint64_t Backend::cache_misses() const { return impl_ ? impl_->misses : 0; }

void Backend::register_input(ggml_tensor* t, const void* host, size_t nbytes) {
    impl_->pending.push_back({t, host, nbytes});
}
void Backend::register_capture(ggml_tensor* t, std::vector<float>* dst) {
    impl_->captures.push_back({t, dst});
}

bool Backend::compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                      std::vector<float>& out) {
    if (!impl_ || !impl_->backend) {
        FD_LOG("Backend::compute called on an uninitialised backend");
        return false;
    }
    const struct ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * kGraphSize +
                    ggml_graph_overhead_custom(kGraphSize, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
    };

    // --- Build the graph ONCE per call into the scratch context.
    // ---------------- This harvests the fresh input host pointers (the image
    // differs every call) and the structural fingerprint. ggml_reset reuses the
    // scratch buffer on a hit (no malloc churn); a fresh ctx is allocated only
    // after a miss promoted the previous scratch into a cache entry.
    if (!impl_->scratch) {
        impl_->scratch = ggml_init(params);
        if (!impl_->scratch) {
            FD_LOG("Backend::compute: ggml_init failed");
            return false;
        }
    } else {
        ggml_reset(impl_->scratch);
    }
    ggml_context* ctx = impl_->scratch;

    impl_->pending.clear();
    impl_->captures.clear();
    Backend* prev_active = t_active;
    t_active = this;
    struct ggml_tensor* output = build(ctx);
    t_active = prev_active;

    if (!output) {
        FD_LOG("Backend::compute: build() returned null output tensor");
        impl_->pending.clear();
        impl_->captures.clear();
        return false;
    }
    ggml_set_output(output);
    for (const PendingCapture& pc : impl_->captures) ggml_set_output(pc.tensor);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    for (const PendingCapture& pc : impl_->captures)
        ggml_build_forward_expand(gf, pc.tensor);
    ggml_build_forward_expand(gf, output);

    bool need_sched = false;
    if (impl_->use_sched) {
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            if (!ggml_backend_supports_op(impl_->backend,
                                          ggml_graph_node(gf, i))) {
                need_sched = true;
                break;
            }
        }
    }

    // === Sched path: a node is unsupported on the primary device, so the graph
    // is scheduled across {primary, cpu}. Kept per-call (no persistent caching)
    // exactly as before; the scratch context is reused (reset) next call.
    // =========
    if (need_sched) {
        if (!impl_->sched) {
            impl_->sched = ggml_common::new_gpu_sched(
                    impl_->gpu_backends, impl_->cpu_backend, kGraphSize);
            if (!impl_->sched) {
                FD_LOG("Backend::compute: ggml_backend_sched_new failed");
                impl_->pending.clear();
                impl_->captures.clear();
                return false;
            }
        }
        ggml_backend_sched_reset(impl_->sched);
        if (!ggml_backend_sched_alloc_graph(impl_->sched, gf)) {
            FD_LOG("Backend::compute: ggml_backend_sched_alloc_graph failed");
            impl_->pending.clear();
            impl_->captures.clear();
            return false;
        }
        for (const PendingInput& pi : impl_->pending)
            ggml_backend_tensor_set(pi.tensor, pi.host, 0, pi.nbytes);
        impl_->pending.clear();

        if (aicore_cancel_requested()) {
            FD_LOG("Backend::compute: cancelled before sched compute");
            impl_->captures.clear();
            return false;
        }

        enum ggml_status status =
                ggml_backend_sched_graph_compute(impl_->sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            FD_LOG("Backend::compute: sched compute failed (status=%d)",
                   (int)status);
            impl_->captures.clear();
            return false;
        }
        for (const PendingCapture& pc : impl_->captures) {
            size_t cn = (size_t)ggml_nelements(pc.tensor);
            pc.dst->resize(cn);
            ggml_backend_tensor_get(pc.tensor, pc.dst->data(), 0,
                                    cn * sizeof(float));
        }
        impl_->captures.clear();
        size_t n = (size_t)ggml_nelements(output);
        out.resize(n);
        ggml_backend_tensor_get(output, out.data(), 0, n * sizeof(float));
        return true;
    }

    // === gallocr path: persistent per-shape cached graph.
    // =======================
    const uint64_t fp = fingerprint_graph(gf, output, impl_->pending.size(),
                                          impl_->captures.size());

    Impl::GraphEntry* hit = nullptr;
    for (Impl::GraphEntry& e : impl_->cache) {
        if (e.fp == fp) {
            hit = &e;
            break;
        }
    }

    if (hit) {
        ++impl_->hits;
        // Re-set ONLY the input tensor DATA into the cached (stable-address)
        // input tensors, then replay the cached graph. The cgraph object and
        // every node data pointer are unchanged, so the CUDA graph updates in
        // place (cudaGraphExecUpdate) and replays - no destroy + reinstantiate.
        GGML_ASSERT(hit->inputs.size() == impl_->pending.size() &&
                    "graph cache hit: input count mismatch");
        for (size_t i = 0; i < impl_->pending.size(); ++i) {
            ggml_backend_tensor_set(hit->inputs[i], impl_->pending[i].host, 0,
                                    impl_->pending[i].nbytes);
        }
        impl_->pending.clear();

        enum ggml_status status =
                ggml_backend_graph_compute(impl_->backend, hit->gf);
        if (status != GGML_STATUS_SUCCESS) {
            FD_LOG("Backend::compute: cached graph_compute failed (status=%d)",
                   (int)status);
            impl_->captures.clear();
            return false;
        }

        GGML_ASSERT(hit->caps.size() == impl_->captures.size() &&
                    "graph cache hit: capture count mismatch");
        for (size_t i = 0; i < impl_->captures.size(); ++i) {
            ggml_tensor* src = hit->caps[i];
            size_t cn = (size_t)ggml_nelements(src);
            impl_->captures[i].dst->resize(cn);
            ggml_backend_tensor_get(src, impl_->captures[i].dst->data(), 0,
                                    cn * sizeof(float));
        }
        impl_->captures.clear();

        size_t n = (size_t)ggml_nelements(hit->output);
        out.resize(n);
        ggml_backend_tensor_get(hit->output, out.data(), 0, n * sizeof(float));
        hit->lru = ++impl_->lru_clock;
        return true;
    }

    // --- MISS: allocate a dedicated gallocr, run, and (unless caching is
    // disabled) promote this scratch graph into a new cache entry.
    // ------------------------
    ++impl_->misses;
    // Snapshot the weight buffers NOW (pre-alloc): only weight leaves are
    // buffered at this point, so this captures exactly the ModelLoader weights
    // this graph depends on (inputs/intermediates get their buffer from gallocr
    // below).
    std::vector<ggml_backend_buffer_t> weight_bufs = collect_weight_buffers(gf);
    ggml_gallocr_t galloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(impl_->backend));
    if (!galloc) {
        FD_LOG("Backend::compute: ggml_gallocr_new failed");
        impl_->pending.clear();
        impl_->captures.clear();
        return false;
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        FD_LOG("Backend::compute: ggml_gallocr_alloc_graph failed");
        ggml_gallocr_free(galloc);
        impl_->pending.clear();
        impl_->captures.clear();
        return false;
    }

    for (const PendingInput& pi : impl_->pending) {
        ggml_backend_tensor_set(pi.tensor, pi.host, 0, pi.nbytes);
    }

    enum ggml_status status = ggml_backend_graph_compute(impl_->backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        FD_LOG("Backend::compute: ggml_backend_graph_compute failed "
               "(status=%d)",
               (int)status);
        ggml_gallocr_free(galloc);
        impl_->pending.clear();
        impl_->captures.clear();
        return false;
    }

    for (const PendingCapture& pc : impl_->captures) {
        size_t cn = (size_t)ggml_nelements(pc.tensor);
        pc.dst->resize(cn);
        ggml_backend_tensor_get(pc.tensor, pc.dst->data(), 0,
                                cn * sizeof(float));
    }

    size_t n = (size_t)ggml_nelements(output);
    out.resize(n);
    ggml_backend_tensor_get(output, out.data(), 0, n * sizeof(float));

    if (impl_->cache_cap == 0) {
        // Caching disabled: free this graph's resources, mirroring the legacy
        // per-call build/free. The scratch ctx is reused (reset) next call.
        ggml_gallocr_free(galloc);
        impl_->pending.clear();
        impl_->captures.clear();
        return true;
    }

    Impl::GraphEntry e;
    e.fp = fp;
    e.ctx = impl_->scratch;  // the scratch ctx now OWNS this graph
    e.gf = gf;
    e.galloc = galloc;
    e.output = output;
    e.inputs.reserve(impl_->pending.size());
    for (const PendingInput& pi : impl_->pending) e.inputs.push_back(pi.tensor);
    e.caps.reserve(impl_->captures.size());
    for (const PendingCapture& pc : impl_->captures)
        e.caps.push_back(pc.tensor);
    e.weight_bufs = std::move(weight_bufs);
    e.lru = ++impl_->lru_clock;
    impl_->pending.clear();
    impl_->captures.clear();

    // The scratch ctx is now owned by the cache entry; force a fresh one next
    // call.
    impl_->scratch = nullptr;
    impl_->cache.push_back(std::move(e));

    // Evict the least-recently-used entries beyond the capacity cap. The entry
    // we just appended carries the highest lru, so it is never the victim.
    while (impl_->cache.size() > impl_->cache_cap) {
        size_t victim = 0;
        for (size_t i = 1; i < impl_->cache.size(); ++i)
            if (impl_->cache[i].lru < impl_->cache[victim].lru) victim = i;
        Impl::GraphEntry& v = impl_->cache[victim];
        if (v.galloc) ggml_gallocr_free(v.galloc);
        if (v.ctx) ggml_free(v.ctx);
        impl_->cache.erase(impl_->cache.begin() + (long)victim);
    }
    return true;
}

void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes) {
    GGML_ASSERT(
            t_active != nullptr &&
            "add_graph_input called outside a Backend::compute build lambda");
    ggml_set_input(t);
    t_active->register_input(t, host, nbytes);
}

ggml_tensor* graph_input_tensor(ggml_context* ctx,
                                int type,
                                int n_dims,
                                const int64_t* ne,
                                const void* host,
                                size_t nbytes) {
    ggml_tensor* t = ggml_new_tensor(ctx, (ggml_type)type, n_dims, ne);
    add_graph_input(t, host, nbytes);
    return t;
}

void capture_graph_output(ggml_tensor* t, std::vector<float>* dst) {
    GGML_ASSERT(t_active != nullptr &&
                "capture_graph_output called outside a Backend::compute build "
                "lambda");
    t_active->register_capture(t, dst);
}

// --- process-global backend -------------------------------------------------

namespace {
std::unique_ptr<Backend> g_backend;
std::once_flag g_shutdown_hook_once;

void register_process_shutdown_hook() {
    std::call_once(g_shutdown_hook_once, []() {
        // Release ggml CUDA/Vulkan buffers while the GPU runtime is still
        // alive. Without this, libAICore unload can run ~Backend after CUDA
        // teardown and abort.
        std::atexit([]() { shutdown_backend(); });
    });
}

// Default CPU thread count for the process-global backend. The previous default
// of 1 left a ~4x latency win unclaimed (it is below even ggml's own default of
// 4): these conv/matmul forwards are compute-bound and scale near-linearly to a
// handful of threads. Profiling on a 16-core/2-CCD host shows the 640x640 SCRFD
// detector (the dominant pipeline cost, im2col-heavy and bandwidth-bound)
// plateaus at ~8 threads, beyond which there is no gain and a cross-CCD/SMT
// regression risk. Cap accordingly; FACEDETECT_THREADS overrides for callers
// who know their box, and fd::set_num_threads()/CLI --threads still win at
// runtime.
int default_n_threads() {
    if (const char* env = std::getenv("FACEDETECT_THREADS")) {
        const int v = std::atoi(env);
        if (v > 0) return v;
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    return (int)std::min(hw, 8u);
}
}  // namespace

Backend& global_backend() {
    register_process_shutdown_hook();
    if (!g_backend) g_backend = std::make_unique<Backend>(default_n_threads());
    return *g_backend;
}

void set_num_threads(int n_threads) {
    global_backend().set_n_threads(n_threads);
}

bool backend_is_cpu() {
    ggml_backend_t h = global_backend().handle();
    return h && ggml_common::is_cpu_backend(h);
}

void shutdown_backend() { g_backend.reset(); }

void invalidate_graph_cache_for_weights(ggml_backend_buffer_t buf) {
    // Do NOT lazily create the global backend here: a ModelLoader destroyed
    // after shutdown_backend() (or before any compute) has nothing to
    // invalidate.
    if (g_backend) g_backend->invalidate_weights_buffer(buf);
}

void ensure_weights_realized(const ModelLoader& ml) {
    if (ml.weights_realized()) return;
    ModelLoader& mut = const_cast<ModelLoader&>(ml);
    mut.realize_weights(global_backend().handle());
}

ggml_tensor* clone_weight(ggml_context* /*ctx*/,
                          const ModelLoader& ml,
                          const char* name) {
    ensure_weights_realized(ml);
    ggml_tensor* src = ml.tensor(name);
    assert(src && "missing tensor");
    return src;
}

ggml_tensor* clone_weight_opt(ggml_context* ctx,
                              const ModelLoader& ml,
                              const char* name) {
    // null name == "no such weight" (e.g. a bias-less conv whose following BN
    // is an explicit node, not folded - the genderage head).
    if (!name || !ml.tensor(name)) return nullptr;
    return clone_weight(ctx, ml, name);
}

}  // namespace fd
