// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Asset-free C ABI contract for the local LLM / VLM task: catalog shape and
// pairing, digest pinning, option/request validation, error contexts and
// the null-safety of every release entry point.

#include <cstring>
#include <set>
#include <string>

#include "aicore/asset_digests.h"
#include "aicore/llm_capi.h"
#include "aicore/runtime_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    aicore_llm_free(nullptr);
    aicore_llm_result_free(nullptr);
    aicore_llm_free_buffer(nullptr);
    aicore_llm_cancel(nullptr);
    AICORE_CHECK(!aicore_llm_ready(nullptr));
    AICORE_CHECK(!aicore_llm_supports_vision(nullptr));
    AICORE_CHECK(std::strcmp(aicore_llm_context_device(nullptr), "") == 0);
    AICORE_CHECK(std::strlen(aicore_llm_last_error(nullptr)) > 0);
    AICORE_CHECK(std::strlen(aicore_llm_media_marker()) > 0);

    // ---- catalog: every row pinned, every vision model paired in-family
    const int count = aicore_llm_model_count();
    AICORE_CHECK(count == 10);
    AICORE_CHECK(aicore_llm_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_llm_model_at(count) == nullptr);
    AICORE_CHECK(aicore_llm_model_by_filename(nullptr) == nullptr);
    AICORE_CHECK(aicore_llm_model_by_filename("missing.gguf") == nullptr);
    std::set<std::string> names;
    int models = 0, projectors = 0, vision = 0;
    for (int i = 0; i < count; ++i) {
        const auto* e = aicore_llm_model_at(i);
        AICORE_CHECK(e && e->size_bytes > 0);
        if (!e) continue;
        names.insert(e->filename);
        AICORE_CHECK(std::strlen(e->sha256) == 64);
        const char* pinned = aicore::AssetDigestForFile(e->filename);
        AICORE_CHECK(pinned && std::strcmp(pinned, e->sha256) == 0);
        // Revision-pinned resolver URL ending in the cached file name.
        const std::string url = e->download_url;
        AICORE_CHECK(url.rfind("https://huggingface.co/", 0) == 0);
        AICORE_CHECK(url.find("/resolve/main/") == std::string::npos);
        AICORE_CHECK(url.size() > std::strlen(e->filename) &&
                     url.compare(url.size() - std::strlen(e->filename),
                                 std::string::npos, e->filename) == 0);
        AICORE_CHECK(aicore_llm_model_by_filename(e->filename) == e);
        AICORE_CHECK(e->quantization && *e->quantization);
        if (e->role == AICORE_LLM_ROLE_MODEL) {
            ++models;
            if (e->mmproj_filename) {
                ++vision;
                const auto* proj =
                        aicore_llm_model_by_filename(e->mmproj_filename);
                AICORE_CHECK(proj && proj->role == AICORE_LLM_ROLE_MMPROJ);
                AICORE_CHECK(proj && std::strcmp(proj->family, e->family) == 0);
            }
        } else {
            AICORE_CHECK(e->role == AICORE_LLM_ROLE_MMPROJ);
            AICORE_CHECK(e->mmproj_filename == nullptr);
            ++projectors;
        }
    }
    AICORE_CHECK(int(names.size()) == count);
    AICORE_CHECK(models == 6 && projectors == 4 && vision == 4);
    const auto* def = aicore_llm_model_at(aicore_llm_model_default_index());
    AICORE_CHECK(def && def->role == AICORE_LLM_ROLE_MODEL &&
                 def->mmproj_filename);
    char* cache = aicore_llm_model_cache_dir();
    AICORE_CHECK(cache && std::strstr(cache, "llm_models"));
    aicore_llm_free_buffer(cache);

    // ---- defaults
    const auto opts = aicore_llm_default_options();
    const auto request = aicore_llm_default_request();
    AICORE_CHECK(opts.struct_size == sizeof(opts) &&
                 opts.abi_version == AICORE_LLM_ABI_VERSION);
    AICORE_CHECK(opts.flash_attn == -1 &&
                 std::strcmp(opts.device, "auto") == 0);
    AICORE_CHECK(request.struct_size == sizeof(request) &&
                 request.max_tokens == 512 && request.temperature == 0.f);

    // ---- invalid creates yield error contexts that refuse every call
    for (int mode = 0; mode < 7; ++mode) {
        auto invalid = opts;
        invalid.model_path = "/nonexistent/aicore-llm-model.gguf";
        if (mode == 1) invalid.struct_size = 0;
        if (mode == 2) invalid.abi_version = 999;
        if (mode == 3) invalid.model_path = nullptr;
        if (mode == 4) invalid.flash_attn = 7;
        if (mode == 5) invalid.image_max_tokens = -1;
        if (mode == 6) invalid.device = "no-such-backend:3";
        auto* ctx = aicore_llm_create(mode == 0 ? nullptr : &invalid);
        AICORE_CHECK(ctx && !aicore_llm_ready(ctx));
        AICORE_CHECK(std::strlen(aicore_llm_last_error(ctx)) > 0);
        AICORE_CHECK(!aicore_llm_supports_vision(ctx));
        auto* result = reinterpret_cast<aicore_llm_result*>(1);
        AICORE_CHECK(aicore_llm_generate(ctx, &request, &result) == -1 &&
                     result == nullptr);
        aicore_pipeline_timings timings{};
        AICORE_CHECK(aicore_llm_last_pipeline_timings(ctx, &timings) == -1);
        aicore_llm_cancel(ctx);
        aicore_llm_free(ctx);
    }
    // A missing model file on CPU is a typed load error, not a crash.
    auto cpu = opts;
    cpu.device = "cpu";
    cpu.model_path = "/nonexistent/aicore-llm-model.gguf";
    auto* ctx = aicore_llm_create(&cpu);
    AICORE_CHECK(ctx && !aicore_llm_ready(ctx) &&
                 std::strstr(aicore_llm_last_error(ctx), "failed to load"));
    aicore_llm_free(ctx);
    AICORE_CHECK(aicore_llm_generate(nullptr, &request, nullptr) == -1);

    aicore_llm_shutdown();
    aicore_llm_shutdown();  // idempotent
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
