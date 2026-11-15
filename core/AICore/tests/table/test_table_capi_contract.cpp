// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Model-free contract of the public `table` C ABI: catalog + digest closure,
// NULL safety of every accessor, option validation (struct_size /
// abi_version / model path) and the shutdown idempotence.

#include <cstdio>
#include <cstring>
#include <string>

#include "aicore/asset_digests.h"
#include "aicore/table_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

namespace {

void testCatalog() {
    AICORE_CHECK(aicore_table_model_count() == 3);
    const std::string base = aicore_table_model_download_base();
    AICORE_CHECK(!base.empty() && base.back() == '/');
    const char* quants[] = {"f32", "f16", "q8_0"};
    for (int i = 0; i < aicore_table_model_count(); ++i) {
        const aicore_table_model_entry* e = aicore_table_model_at(i);
        AICORE_CHECK(e != nullptr);
        if (!e) continue;
        const std::string name = e->filename;
        AICORE_CHECK(name ==
                     std::string("slanet-plus-table-") + quants[i] + ".gguf");
        AICORE_CHECK(std::string(e->download_url) == base + name);
        AICORE_CHECK(std::strcmp(e->role, "table") == 0);
        AICORE_CHECK(std::strcmp(e->arch, "slanet_plus") == 0);
        AICORE_CHECK(e->display_name && e->display_name[0]);
        // Every published variant is anchored by a content digest.
        const char* sha = aicore::AssetDigestForFile(e->filename);
        AICORE_CHECK(sha != nullptr && std::strlen(sha) == 64);
        AICORE_CHECK(aicore_table_model_by_filename(e->filename) != nullptr);
    }
    AICORE_CHECK(aicore_table_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_table_model_at(3) == nullptr);
    AICORE_CHECK(aicore_table_model_by_filename("missing.gguf") == nullptr);
    AICORE_CHECK(aicore_table_model_by_filename(nullptr) == nullptr);
    char* cache = aicore_table_model_cache_dir();
    AICORE_CHECK(cache != nullptr &&
                 std::strstr(cache, "ocr_models") != nullptr);
    aicore_table_free_buffer(cache);
    aicore_table_free_buffer(nullptr);
}

void testNullSafety() {
    AICORE_CHECK(aicore_table_context_create(nullptr) == nullptr);
    AICORE_CHECK(aicore_table_last_error()[0] != '\0');
    aicore_table_context_destroy(nullptr);
    AICORE_CHECK(aicore_table_context_ready(nullptr) == 0);
    AICORE_CHECK(aicore_table_context_device(nullptr) != nullptr);
    AICORE_CHECK(aicore_table_run(nullptr, nullptr, nullptr, nullptr,
                                  nullptr) == nullptr);
    AICORE_CHECK(aicore_table_count(nullptr) == 0);
    AICORE_CHECK(aicore_table_at(nullptr, 0) == nullptr);
    AICORE_CHECK(aicore_table_cell_count(nullptr) == 0);
    AICORE_CHECK(aicore_table_cell_at(nullptr, 0) == nullptr);
    AICORE_CHECK(aicore_table_token_at(nullptr, 0, 0) == nullptr);
    aicore_table_result_release(nullptr);
    const aicore_pipeline_timings t = aicore_table_timings(nullptr);
    AICORE_CHECK(t.valid_fields == 0);
}

void testOptionValidation() {
    aicore_table_options o{};
    o.struct_size = sizeof(o);
    o.abi_version = AICORE_TABLE_ABI_VERSION;
    o.device = "cpu";
    // Missing model path.
    AICORE_CHECK(aicore_table_context_create(&o) == nullptr);
    o.model_path = "/nonexistent/slanet-plus-table-f32.gguf";
    AICORE_CHECK(aicore_table_context_create(&o) == nullptr);
    AICORE_CHECK(std::strlen(aicore_table_last_error()) > 0);
    aicore_table_options small = o;
    small.struct_size = sizeof(o) - 1;
    AICORE_CHECK(aicore_table_context_create(&small) == nullptr);
    aicore_table_options abi = o;
    abi.abi_version = AICORE_TABLE_ABI_VERSION + 1;
    AICORE_CHECK(aicore_table_context_create(&abi) == nullptr);
}

}  // namespace

int main() {
    testCatalog();
    testNullSafety();
    testOptionValidation();
    aicore_table_runtime_shutdown();
    aicore_table_runtime_shutdown();
    if (failures) {
        std::fprintf(stderr, "test_table_capi_contract: %d failure(s)\n",
                     failures);
        return 1;
    }
    std::printf("test_table_capi_contract: OK\n");
    return 0;
}
