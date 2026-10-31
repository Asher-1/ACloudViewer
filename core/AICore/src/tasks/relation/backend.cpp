// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "relateanything/backend.hpp"

#include <stdexcept>
namespace relateanything {
Backend::Backend(const ModelOptions& options) : name(options.backend) {
    options.validate_model();
    binding = options.backend_binding;
    if (binding.struct_size != sizeof(binding) || binding.abi_version != 1 ||
        !binding.cpu_handle || !binding.accelerator_handle ||
        !binding.retain_lease || !binding.release_lease)
        throw std::invalid_argument("AICore backend lease binding is required");
    cpu = static_cast<ggml_backend_t>(binding.cpu_handle);
    accelerator = static_cast<ggml_backend_t>(binding.accelerator_handle);
    ggml_backend_t handles[2] = {accelerator, cpu};
    scheduler = ggml_backend_sched_new(
            handles, nullptr, accelerator == cpu ? 1 : 2, 8192, false, true);
    if (!scheduler)
        throw std::runtime_error("cannot create relation scheduler");
    binding.retain_lease(binding.lease_user);
    lease_retained = true;
}
Backend::~Backend() {
    if (scheduler) ggml_backend_sched_free(scheduler);
    if (lease_retained) binding.release_lease(binding.lease_user);
}
}  // namespace relateanything
