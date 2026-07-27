// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ggml-backend.h>

#include <string>

namespace lightglue::internal {

struct Backend {
    ggml_backend_t handle = nullptr;
    ggml_gallocr_t allocator = nullptr;
    std::string device;
    std::string error;

    bool Init(const std::string &request, int num_threads);
    bool IsCpu() const;
    bool IsGpu() const;
    bool IsCuda() const;
    bool IsVulkan() const;
    void Release();
    ~Backend() { Release(); }
};

}  // namespace lightglue::internal
