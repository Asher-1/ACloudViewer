#pragma once

#include "relateanything/options.hpp"

#include "ggml-backend.h"

#include <string>

namespace relateanything {

struct Backend {
    ggml_backend_t accelerator = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    std::string name;
    ra_backend_binding binding{};
    bool lease_retained = false;

    explicit Backend(const ModelOptions & options);
    Backend(const Backend &) = delete;
    Backend & operator=(const Backend &) = delete;
    ~Backend();
};

} // namespace relateanything
