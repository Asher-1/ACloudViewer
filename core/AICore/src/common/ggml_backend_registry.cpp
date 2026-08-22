// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/ggml_backend_registry.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "common/ggml_backend_utils.hpp"

namespace aicore {
namespace runtime {
namespace {

struct Candidate {
    ggml_backend_t handle = nullptr;
    std::string device;
};

std::mutex g_registry_mutex;
std::unordered_map<std::string, std::weak_ptr<BackendLease::State>> g_registry;

std::string lease_key(const std::string& device, int n_threads, bool is_cpu) {
    return ggml_common::to_lower(device) +
           (is_cpu ? ":threads=" + std::to_string(n_threads) : "");
}

Candidate create_candidate(const std::string& request,
                           int n_threads,
                           std::string* error) {
    ggml_common::load_backends_once();

    std::string name;
    int device_index = 0;
    ggml_common::parse_device(request, name, device_index);
    Candidate candidate;
    if (name.empty() || name == "auto") {
        candidate.handle = ggml_common::find_auto_backend(candidate.device);
        if (candidate.handle == nullptr) {
            name = "cpu";
        }
    }
    if (candidate.handle == nullptr && name == "cpu") {
        candidate.handle = ggml_backend_init_by_type(
                GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        candidate.device = "cpu";
        if (candidate.handle != nullptr) {
            ggml_common::set_cpu_threads(candidate.handle, n_threads);
        }
    } else if (candidate.handle == nullptr &&
               (name == "gpu" || name == "cuda" || name == "opencl" ||
                name == "metal" || name == "sycl" || name == "vulkan")) {
        candidate.handle = ggml_common::find_gpu_backend(name, device_index,
                                                         candidate.device);
    } else if (candidate.handle == nullptr && name != "cpu") {
        if (error) {
            *error = "unknown device '" + request +
                     "' (want auto|cpu|gpu|sycl|vulkan|cuda|metal)";
        }
        return candidate;
    }

    if (candidate.handle == nullptr && error && error->empty()) {
        *error = "no usable '" + (name.empty() ? request : name) +
                 "' device (backend built and runtime driver present?)";
    }
    if (candidate.handle != nullptr) {
        // Match public UI/task IDs (cuda, cuda:1, vulkan, metal) instead of
        // backend-specific display names such as CUDA0.
        candidate.device = ggml_common::resolve_device_request(request);
    }
    return candidate;
}

}  // namespace

struct BackendLease::State {
    State(ggml_backend_t backend, std::string resolved_device)
        : handle(backend), device(std::move(resolved_device)) {}
    ~State() {
        if (handle != nullptr) ggml_backend_free(handle);
    }

    ggml_backend_t handle = nullptr;
    std::string device;
    // A public extraction may call helpers that also need the physical device
    // lock (scheduler, teardown). Recursive ownership keeps that layering safe
    // while still serializing distinct sessions.
    std::recursive_mutex execution_mutex;
};

ggml_backend_t BackendLease::handle() const {
    return state_ ? state_->handle : nullptr;
}

const std::string& BackendLease::device() const {
    static const std::string kEmpty;
    return state_ ? state_->device : kEmpty;
}

bool BackendLease::is_cpu() const {
    return state_ != nullptr && ggml_common::is_cpu_backend(state_->handle);
}

std::unique_lock<std::recursive_mutex> BackendLease::lock() const {
    return state_ ? std::unique_lock<std::recursive_mutex>(
                            state_->execution_mutex)
                  : std::unique_lock<std::recursive_mutex>();
}

void BackendLease::reset() { state_.reset(); }

BackendLease adopt_backend_lease(ggml_backend_t backend,
                                 const std::string& resolved_device,
                                 int n_threads) {
    if (backend == nullptr) return BackendLease();
    if (n_threads <= 0)
        n_threads = static_cast<int>(ggml_common::default_cpu_threads());

    const bool cpu = ggml_common::is_cpu_backend(backend);
    const std::string key = lease_key(resolved_device, n_threads, cpu);
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_registry.find(key);
    if (it != g_registry.end()) {
        if (std::shared_ptr<BackendLease::State> existing = it->second.lock()) {
            ggml_backend_free(backend);
            return BackendLease(std::move(existing));
        }
        g_registry.erase(it);
    }

    std::shared_ptr<BackendLease::State> state =
            std::make_shared<BackendLease::State>(backend, resolved_device);
    g_registry.emplace(key, state);
    return BackendLease(std::move(state));
}

BackendLease acquire_backend_lease(const std::string& device_request,
                                   int n_threads,
                                   std::string* error) {
    if (error) error->clear();
    if (n_threads <= 0) {
        n_threads = static_cast<int>(ggml_common::default_cpu_threads());
    }

    Candidate candidate = create_candidate(device_request, n_threads, error);
    if (candidate.handle == nullptr) return BackendLease();
    return adopt_backend_lease(candidate.handle, candidate.device, n_threads);
}

BackendLeaseLock lock_backend_leases(const std::vector<BackendLease>& leases) {
    std::vector<std::shared_ptr<BackendLease::State>> states;
    states.reserve(leases.size());
    for (const BackendLease& lease : leases) {
        if (lease.state_) states.push_back(lease.state_);
    }
    std::sort(states.begin(), states.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.get() < rhs.get();
              });
    states.erase(std::unique(states.begin(), states.end(),
                             [](const auto& lhs, const auto& rhs) {
                                 return lhs.get() == rhs.get();
                             }),
                 states.end());

    BackendLeaseLock result;
    result.locks_.reserve(states.size());
    for (const auto& state : states) {
        result.locks_.emplace_back(state->execution_mutex);
    }
    return result;
}

void purge_inactive_backend_leases() {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    for (auto it = g_registry.begin(); it != g_registry.end();) {
        if (it->second.expired()) {
            it = g_registry.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace runtime
}  // namespace aicore
