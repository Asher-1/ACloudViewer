// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// ReID C API contract test (no model assets): ABI version, options
// defaults/setters, NULL safety, the ready-or-queryable-error context
// contract, argument rejection, and shutdown idempotency.

#include <cstring>

#include "aicore/reid_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_reid_abi_version() == 1);

    /* Options: create, NULL-safe setters, free. */
    aicore_reid_options* opts = aicore_reid_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_reid_options_set_device(opts, "cpu");
    aicore_reid_options_set_threads(opts, 2);
    aicore_reid_options_set_device(nullptr, "cpu");  // NULL-safe no-op
    aicore_reid_options_set_threads(nullptr, 4);     // NULL-safe no-op

    /* NULL/empty path -> a context that is not ready and carries the
     * queryable error (the "ready or error" contract). */
    aicore_reid_ctx* empty = aicore_reid_load_opts(nullptr, opts);
    AICORE_CHECK(empty != nullptr);
    AICORE_CHECK(aicore_reid_is_ready(empty) == 0);
    AICORE_CHECK(aicore_reid_last_error(empty) != nullptr);
    aicore_reid_ctx* blank = aicore_reid_load_opts("", opts);
    AICORE_CHECK(blank != nullptr);
    AICORE_CHECK(aicore_reid_is_ready(blank) == 0);
    AICORE_CHECK(aicore_reid_last_error(blank) != nullptr);
    /* Nonexistent path -> not ready with an error, no crash. */
    aicore_reid_ctx* missing =
            aicore_reid_load_opts("/nonexistent/reid-encoder.gguf", opts);
    AICORE_CHECK(missing != nullptr);
    AICORE_CHECK(aicore_reid_is_ready(missing) == 0);
    AICORE_CHECK(aicore_reid_last_error(missing) != nullptr);
    aicore_reid_free(empty);
    aicore_reid_free(blank);
    aicore_reid_free(missing);
    aicore_reid_free(nullptr);  // NULL-safe

    /* NULL-safe queries. */
    AICORE_CHECK(aicore_reid_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_reid_last_error(nullptr) == nullptr);
    AICORE_CHECK(aicore_reid_embed_dim(nullptr) == -1);
    aicore_pipeline_timings timings{};
    AICORE_CHECK(aicore_reid_last_pipeline_timings(nullptr, &timings) != 0);
    aicore_reid_free_buffer(nullptr);  // NULL-safe

    /* embed_image argument contract: out pointers are zeroed before any
     * rejection, and every NULL-required argument returns -1. */
    float* out = nullptr;
    int32_t n = -1;
    int32_t dim = -1;
    AICORE_CHECK(aicore_reid_embed_image(nullptr, nullptr, nullptr, 0, &out, &n,
                                         &dim) == -1);
    AICORE_CHECK(out == nullptr && n == 0 && dim == 0);
    AICORE_CHECK(aicore_reid_embed_image(nullptr, nullptr, nullptr, 0, nullptr,
                                         nullptr, nullptr) == -1);

    aicore_reid_options_free(opts);
    aicore_reid_options_free(nullptr);  // NULL-safe

    /* Process-wide shutdown delegates to the shared runtime shutdown and is
     * idempotent; with no live context it must be safe to call twice. */
    aicore_reid_shutdown();
    aicore_reid_shutdown();

    if (failures == 0) {
        std::printf("[reid] contract test passed\n");
    }
    return failures;
}
