// ----------------------------------------------------------------------------
// Process-wide ggml backend leases. Sessions keep allocators, weights, graphs,
// and result buffers private; only the physical backend handle is shared.
// ----------------------------------------------------------------------------

#pragma once

#include <ggml-backend.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace aicore {
namespace runtime {

class BackendLease;
class BackendLeaseLock;
BackendLeaseLock lock_backend_leases(const std::vector<BackendLease>& leases);

class BackendLease {
public:
    struct State;

    BackendLease() = default;

    explicit operator bool() const { return static_cast<bool>(state_); }
    ggml_backend_t handle() const;
    const std::string& device() const;
    bool is_cpu() const;
    std::unique_lock<std::recursive_mutex> lock() const;
    void reset();

private:
    std::shared_ptr<State> state_;

    explicit BackendLease(std::shared_ptr<State> state)
        : state_(std::move(state)) {}
    friend BackendLease acquire_backend_lease(const std::string&, int,
                                              std::string*);
    friend BackendLease adopt_backend_lease(ggml_backend_t,
                                            const std::string&, int);
    friend BackendLeaseLock lock_backend_leases(
            const std::vector<BackendLease>&);
};

class BackendLeaseLock {
public:
    BackendLeaseLock() = default;
    bool owns_lock() const { return !locks_.empty(); }

private:
    std::vector<std::unique_lock<std::recursive_mutex>> locks_;

    friend BackendLeaseLock lock_backend_leases(
            const std::vector<BackendLease>&);
};

// Acquires a physical ggml backend for a resolved device. CPU thread count is
// part of the lease key because it configures the backend instance. The caller
// must keep gallocr/scheduler/weights session-local and hold lock() around all
// graph work using handle().
BackendLease acquire_backend_lease(const std::string& device_request,
                                   int n_threads,
                                   std::string* error);

// Takes ownership of an already-created backend handle. Used by multi-GPU
// schedulers after resolving a GPU group; a compatible existing lease wins and
// the candidate handle is immediately released.
BackendLease adopt_backend_lease(ggml_backend_t backend,
                                 const std::string& resolved_device,
                                 int n_threads);

// Locks a GPU lease group and CPU fallback in a stable order. Every lease must
// remain alive while the returned lock object is in scope.
BackendLeaseLock lock_backend_leases(const std::vector<BackendLease>& leases);

// Drops registry entries whose owners are gone (no live context holds the
// backend). Live sessions are never touched; this only reclaims the key table
// memory of expired leases. Used by the per-task *shutdown entry points.
void purge_inactive_backend_leases();

}  // namespace runtime
}  // namespace aicore
