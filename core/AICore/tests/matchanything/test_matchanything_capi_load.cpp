// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aicore/matchanything_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <matchanything.gguf>\n", argv[0]);
        return 77;
    }

    aicore_matchanything_options* opts = aicore_matchanything_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_matchanything_options_set_device(opts, "cpu");
    aicore_matchanything_options_set_threads(opts, 1);
    aicore_matchanything_options_set_variant(opts, AICORE_MATCHANYTHING_ELOFTR);

    aicore_matchanything_ctx* ctx =
            aicore_matchanything_load_opts(argv[1], opts);
    aicore_matchanything_options_free(opts);

    if (ctx == nullptr) {
        std::fprintf(stderr, "SKIP: could not load GGUF (missing asset?)\n");
        return 77;
    }

    char* info = aicore_matchanything_info_json(ctx);
    AICORE_CHECK(info != nullptr &&
                 std::strstr(info, "matchanything") != nullptr);
    aicore_matchanything_free_string(info);

    const int32_t w = 256;
    const int32_t h = 256;
    std::vector<uint8_t> img0(static_cast<size_t>(w) * h, 128);
    std::vector<uint8_t> img1(static_cast<size_t>(w) * h, 128);

    aicore_matchanything_match* matches = nullptr;
    int32_t count = 0;
    const int rc = aicore_matchanything_match_gray(
            ctx, img0.data(), img1.data(), w, h, w, &matches, &count);
    if (rc != 0) {
        const char* err = aicore_matchanything_last_error(ctx);
        std::fprintf(stderr, "match failed: %s\n", err ? err : "unknown");
        aicore_matchanything_free(ctx);
        return 1;
    }
    aicore_matchanything_free_matches(matches);
    aicore_matchanything_free(ctx);
    return failures;
}
