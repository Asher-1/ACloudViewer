// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "lingbot_gguf_loader.h"

#include <ggml.h>
#include <gguf.h>

#include "common/gguf_file_io.hpp"

namespace lingbot {

bool gguf_loader::open(const std::string& path) {
    close();
    // Load tensor bytes into one contiguous GGML context. The model owns the
    // context for its lifetime and binds it to the selected backend buffer.
    std::string open_error;
    gctx_ = ggml_common::open_gguf_file(path, /*no_alloc=*/false, &ctx_,
                                        "lingbot", &open_error);
    if (!gctx_) {
        error_ = open_error;
        return false;
    }
    return true;
}

void gguf_loader::close() {
    if (gctx_) gguf_free(gctx_);
    if (ctx_) ggml_free(ctx_);
    gctx_ = nullptr;
    ctx_ = nullptr;
    error_.clear();
}

ggml_tensor* gguf_loader::require(const std::string& name) const {
    return ctx_ ? ggml_get_tensor(ctx_, name.c_str()) : nullptr;
}

int32_t gguf_loader::i32(const char* key, int32_t fallback) const {
    const int64_t id = gctx_ ? gguf_find_key(gctx_, key) : -1;
    return id >= 0 ? static_cast<int32_t>(gguf_get_val_u32(gctx_, id))
                   : fallback;
}

std::string gguf_loader::str(const char* key,
                             const std::string& fallback) const {
    const int64_t id = gctx_ ? gguf_find_key(gctx_, key) : -1;
    return id >= 0 ? gguf_get_val_str(gctx_, id) : fallback;
}

}  // namespace lingbot
