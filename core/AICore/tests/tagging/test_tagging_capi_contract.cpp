// SPDX-License-Identifier: MIT
#include <cstring>
#include <set>
#include <string>

#include "aicore/runtime_capi.h"
#include "aicore/tagging_capi.h"
#include "tests/common/test_macros.hpp"
static int failures = 0;
int main() {
    aicore_tagging_free(nullptr);
    aicore_tagging_result_free(nullptr);
    aicore_tagging_free_buffer(nullptr);
    AICORE_CHECK(!aicore_tagging_ready(nullptr));
    AICORE_CHECK(aicore_tagging_model_count() == 9);
    AICORE_CHECK(aicore_tagging_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_tagging_model_at(9) == nullptr);
    std::set<std::string> names;
    int captions = 0;
    for (int i = 0; i < 9; ++i) {
        const auto* e = aicore_tagging_model_at(i);
        AICORE_CHECK(e && e->size_bytes > 0);
        if (!e) continue;
        AICORE_CHECK(std::strlen(e->sha256) == 64);
        AICORE_CHECK(
                std::strstr(e->download_url,
                            "huggingface.co/Asher-1/RAM_GGUF/resolve/main/"));
        names.insert(e->filename);
        captions += e->supports_caption;
    }
    AICORE_CHECK(names.size() == 9 && captions == 3);
    auto opts = aicore_tagging_default_options();
    auto request = aicore_tagging_default_request();
    AICORE_CHECK(opts.struct_size == sizeof(opts) && request.threshold == -1);
    AICORE_CHECK(opts.cpu_precision == AICORE_TAGGING_CPU_AUTO);
    for (int mode = 0; mode < 5; ++mode) {
        auto invalid = opts;
        if (mode == 1) invalid.struct_size = 0;
        if (mode == 2) invalid.abi_version = 999;
        if (mode == 3) invalid.max_length = 0;
        if (mode == 4) invalid.cpu_precision = 99;
        auto* ctx = aicore_tagging_create(mode == 0 ? nullptr : &invalid);
        AICORE_CHECK(ctx && !aicore_tagging_ready(ctx));
        AICORE_CHECK(std::strlen(aicore_tagging_last_error(ctx)) > 0);
        auto* result = reinterpret_cast<aicore_tagging_result*>(1);
        AICORE_CHECK(aicore_tagging_infer_image(ctx, nullptr, &request,
                                                &result) == -1 &&
                     result == nullptr);
        aicore_pipeline_timings timings{};
        AICORE_CHECK(aicore_tagging_last_pipeline_timings(ctx, &timings) == -1);
        aicore_tagging_free(ctx);
    }
    char* dir = aicore_tagging_model_cache_dir();
    AICORE_CHECK(dir && std::strstr(dir, "ram_models"));
    aicore_tagging_free_buffer(dir);
    aicore_tagging_shutdown();
    aicore_tagging_shutdown();
    return failures;
}
