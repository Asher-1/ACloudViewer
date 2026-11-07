// Backend handling: one CPU backend plus at most one GPU backend, wired
// through a ggml gallocr for direct graph execution.
//
// ACloudViewer adaptation: the physical ggml backend handles are shared with
// the whole process through aicore::runtime::BackendLease (trellis2 pattern).
// Sessions keep allocators, weights, graphs, and result buffers private; the
// lease handles are never freed here. The public C ABI caller holds
// lock_backend_leases(leases()) around every graph-consuming call.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/ggml_backend_registry.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

namespace sam3d {

class Backend {
  public:
    // backend_name: "auto" | "cpu" | "cuda" | "vulkan"
    static std::unique_ptr<Backend> create(const std::string& backend_name, int n_threads);

    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Highest-priority (GPU if present) buffer type for weight preloading.
    ggml_backend_buffer_type_t weights_buffer_type() const;

    bool has_gpu() const { return gpu_backend_ != nullptr; }
    const char* device_name() const;
    const char* backend_name() const { return backend_name_.c_str(); }
    int n_threads() const { return n_threads_; }

    // Process-shared physical backend handles owned by the runtime registry.
    const std::vector<aicore::runtime::BackendLease>& leases() const { return leases_; }

    // No-op compatibility hook: the source repo wired per-file JSONL tracing
    // through environment variables, which AICore forbids. Stage timings are
    // reported through aicore_pipeline_timings instead.
    void set_profile_label(std::string label);

    // Build helpers -----------------------------------------------------------
    // alloc(): reset + allocate the graph on the gallocr (inputs can then
    // be filled with set_input_*). run(): compute + synchronize.
    bool alloc(struct ggml_cgraph* graph, std::vector<struct ggml_tensor*> inputs = {});
    bool run(ggml_cgraph* graph);
    bool compute(struct ggml_cgraph* graph, std::vector<struct ggml_tensor*> inputs = {});

    // Copy host data into an input tensor (must be called after alloc()).
    bool set_input_f32(struct ggml_tensor* t, const float* data, size_t n_floats);
    bool set_input_i32(struct ggml_tensor* t, const int32_t* data, size_t n_ints);

    // Read a tensor back to host (handles CPU fallback transparently).
    bool get_tensor_f32(struct ggml_tensor* t, std::vector<float>& out);
    bool get_tensor_i32(struct ggml_tensor* t, std::vector<int32_t>& out);

  private:
    Backend() = default;
    bool init(const std::string& backend_name, int n_threads);

    ggml_backend_t gpu_backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_t compute_backend_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    std::vector<aicore::runtime::BackendLease> leases_;
    std::string backend_name_ = "cpu";
    int n_threads_ = 8;
};

}  // namespace sam3d
