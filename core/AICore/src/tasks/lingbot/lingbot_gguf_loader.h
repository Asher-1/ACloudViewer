// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <unordered_map>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace lingbot {

class gguf_loader {
public:
    bool open(const std::string& path);
    void close();
    ggml_tensor* require(const std::string& name) const;
    int32_t i32(const char* key, int32_t fallback) const;
    std::string str(const char* key, const std::string& fallback) const;
    ggml_context* context() const { return ctx_; }
    const std::string& error() const { return error_; }

private:
    gguf_context* gctx_ = nullptr;
    ggml_context* ctx_ = nullptr;
    std::string error_;
};

}  // namespace lingbot
