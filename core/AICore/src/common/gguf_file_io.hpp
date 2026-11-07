// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared GGUF file-open helper (F-06): one open + failure-reporting shape
// for every task loader, without constraining how each task reads metadata
// keys or maps tensor payloads afterwards. Header-only on purpose — tasks
// are compiled into libAICore, and the sam3 whitebox test archive must be
// able to consume call sites without a new TU in its manual closure.

#pragma once

#include <gguf.h>

#include <string>

namespace ggml_common {

/**
 * Standard AICore GGUF open. Wraps gguf_init_from_file with the caller's
 * no_alloc flag and metadata-context out-parameter.
 *
 * On failure returns nullptr and, when `error` is non-null, sets the shared
 * loader message "<context>: failed to open GGUF '<path>'" so every task
 * reports the same failure shape (grep-addressable via
 * "failed to open GGUF").
 *
 * Ownership: the caller owns the returned gguf_context and, when non-null,
 * *meta_ctx, and must release them with gguf_free()/ggml_free() (or the
 * task's existing teardown that already does so).
 *
 * @param path      GGUF file path.
 * @param no_alloc  passed to gguf_init_params.no_alloc. true = metadata and
 *                  tensor descriptors only (weights are streamed into
 *                  backend buffers by the caller); false = tensor data is
 *                  read into the metadata context as well.
 * @param meta_ctx  out ggml_context holding the tensor descriptors (same
 *                  lifetime rules as gguf_init_params.ctx).
 * @param context   short owner tag for the error message, e.g. "gkd",
 *                  "sam3", "depth".
 * @param error     optional failure message receiver.
 * @return the gguf_context, or nullptr on failure.
 */
inline gguf_context* open_gguf_file(const std::string& path, bool no_alloc,
                                    struct ggml_context** meta_ctx,
                                    const char* context,
                                    std::string* error = nullptr) {
    gguf_init_params params{};
    params.no_alloc = no_alloc;
    params.ctx = meta_ctx;
    gguf_context* gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf && error) {
        *error = std::string(context ? context : "gguf") +
                 ": failed to open GGUF '" + path + "'";
    }
    return gguf;
}

}  // namespace ggml_common
