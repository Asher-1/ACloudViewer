#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

namespace fd {

class ModelLoader;

// Persistent compute backend + reusable graph allocator.
//
// face-detect.cpp's pipeline runs several small ggml graphs per image (SCRFD
// detection heads, the ArcFace embed, optional genderage / MiniFASNet). Mirrors
// parakeet.cpp's Backend: a persistent `ggml_backend_t` (CPU by default, or a
// GPU device selected from the registry) plus a persistent `ggml_gallocr_t`
// reused across every graph computation, so the steady-state loop sees no
// allocator churn. GPU devices route through ggml_backend_sched only when the
// graph contains an op the device lacks a kernel for (cheap per-graph scan),
// otherwise they stay on the fast gallocr path.
//
// CORRECTNESS-CRITICAL ordering: with `no_alloc=true`, a tensor's `->data` is
// NULL until `ggml_gallocr_alloc_graph`. So input tensor data MUST be written
// AFTER alloc (via `ggml_backend_tensor_set`), never inline in the build
// lambda. The build lambda registers each host-backed input by calling
// `fd::add_graph_input(...)`; Backend defers the copy until after alloc.
class Backend {
public:
    // Construct a backend with `n_threads` CPU worker threads (<=0 -> 1). The
    // device is auto-picked (first GPU/IGPU, else CPU) unless FACEDETECT_DEVICE
    // overrides it ("cpu", or a registry device name like "CUDA0"). The gallocr
    // is created lazily on the first compute and reused afterwards.
    explicit Backend(int n_threads);
    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Update the CPU worker-thread count (cheap). <=0 is clamped to 1.
    void set_n_threads(int n_threads);
    int  n_threads() const { return n_threads_; }

    // Name of the selected compute device ("cpu" or a registry device name).
    const char* device_name() const { return device_name_.c_str(); }

    // The underlying ggml backend. Exposed so the loader can give its weight
    // tensors a backend buffer over the SAME backend graphs run on.
    ggml_backend_t handle() const;

    // --- Multi-model graph-cache safety (see backend.cpp). ----------------------
    // Drop every persistent graph-cache entry that references `buf` (a freed
    // ModelLoader's weights buffer). Called from ~ModelLoader BEFORE the buffer is
    // released, so a later model reallocating the same address cannot false-hit a
    // stale cached graph built over the now-freed weights. Entries belonging to
    // other (still-live) models are left untouched, so a concurrently-hosted model
    // keeps its hot cache.
    void invalidate_weights_buffer(ggml_backend_buffer_t buf);

    // Cache introspection (exercised by the cache-safety smoke test).
    size_t   cache_size()   const;
    uint64_t cache_hits()   const;
    uint64_t cache_misses() const;

    // Build + run a single graph on the persistent backend/gallocr and copy the
    // output tensor's f32 contents into `out`. See the class comment for the
    // build-lambda contract (register inputs via add_graph_input; reference
    // loader weights directly). Returns true on success.
    bool compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                 std::vector<float>& out);

    // Backend-internal hooks used by add_graph_input() / capture_graph_output().
    void register_input(ggml_tensor* t, const void* host, size_t nbytes);
    void register_capture(ggml_tensor* t, std::vector<float>* dst);

private:
    struct Impl;
    Impl* impl_;
    int   n_threads_ = 1;
    std::string device_name_ = "cpu";
};

// Register a host-backed graph input for the active Backend::compute build
// phase. Marks `t` as a ggml graph input and records that `nbytes` from `host`
// must be copied into `t` AFTER the graph is allocated. Must be called from
// inside a build lambda passed to Backend::compute.
void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes);

// Create a graph input tensor of the given type and ne[] inside `ctx`, mark it
// as a graph input, and register a host->device copy of `nbytes` from `host` to
// be performed after the gallocr allocates it. Returns the new tensor.
ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host, size_t nbytes);

// Capture an intermediate graph tensor for readback after Backend::compute.
// `*dst` is resized to the tensor's element count and filled with its f32
// contents once the graph has run. Must be called from inside a build lambda.
void capture_graph_output(ggml_tensor* t, std::vector<float>* dst);

// Reference a weight tensor from the loader DIRECTLY as a graph leaf (zero
// per-call copy); see parakeet.cpp's clone_weight. clone_weight_opt returns
// nullptr when absent; clone_weight asserts present.
ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml, const char* name);
ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml, const char* name);

// Ensure the loader's weights have a backend buffer (zero-copy) on the
// process-global Backend. Idempotent; called automatically by clone_weight.
void ensure_weights_realized(const ModelLoader& ml);

// Process-global Backend, lazily created on first use. The model graphs + the
// loader's weight realization all run on this one backend (so weight buffers are
// compatible with the graphs that reference them).
Backend& global_backend();

// Set the CPU worker-thread count on the process-global backend.
void set_num_threads(int n_threads);

// True when the process-global backend computes on the CPU (vs a GPU/IGPU
// device). Used to route convs: ggml_conv_2d_direct wins on CPU for K>1 kernels,
// while on CUDA the im2col + mul_mat (cuBLAS-class) path is faster, so the conv
// helper keeps im2col on GPU. Mirrors depth-anything.cpp's da::gpu_mode().
bool backend_is_cpu();

// Free the process-global backend (call before process exit / static teardown
// to avoid GPU-driver shutdown-ordering aborts). Idempotent.
void shutdown_backend();

// Invalidate any persistent graph-cache entries on the process-global Backend
// that reference `buf`. Safe during teardown: it does NOT create the global
// Backend if none exists yet, so a ModelLoader destroyed after shutdown_backend()
// (or before any compute ran) is a harmless no-op. Called by ~ModelLoader.
void invalidate_graph_cache_for_weights(ggml_backend_buffer_t buf);

} // namespace fd
