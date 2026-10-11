// SPDX-License-Identifier: MIT
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/relation_capi.h"
#include "aicore/backend_capi.h"
#define REQUIRE(x)                                                      \
    do {                                                                \
        if (!(x)) {                                                     \
            std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); \
            return 1;                                                   \
        }                                                               \
    } while (0)
int main() {
    auto o = aicore_relation_default_options();
    auto r = aicore_relation_default_request();
    REQUIRE(o.struct_size == sizeof(o) && r.struct_size == sizeof(r));
    REQUIRE(o.abi_version == 1 && r.abi_version == 1);
    REQUIRE(o.strict_f32 == 0 && o.flash_attention == 1 &&
            o.compact_pairs == 1 && o.fused_vocabulary == 1);
    aicore_model_device_info info{};
    info.struct_size = sizeof(info);
    REQUIRE(aicore_model_device_info_query(AICORE_MODEL_RELATION, "cpu",
                                          &info) == 0);
    REQUIRE(info.capabilities & AICORE_MODEL_CAP_FULL_GRAPH);
    REQUIRE(info.capabilities & AICORE_MODEL_CAP_TASK_CANCEL);
    auto* ctx = aicore_relation_create(nullptr);
    REQUIRE(ctx && !aicore_relation_ready(ctx));
    REQUIRE(std::strlen(aicore_relation_last_error(ctx)));
    aicore_relation_free(ctx);
    o.struct_size = 1;
    ctx = aicore_relation_create(&o);
    REQUIRE(!aicore_relation_ready(ctx));
    aicore_relation_free(ctx);
    o = aicore_relation_default_options();
    o.abi_version = 99;
    ctx = aicore_relation_create(&o);
    REQUIRE(!aicore_relation_ready(ctx));
    aicore_relation_free(ctx);
    aicore_relation_result* out = reinterpret_cast<aicore_relation_result*>(1);
    REQUIRE(aicore_relation_infer_image(nullptr, nullptr, &r, &out) < 0 &&
            !out);
    aicore_relation_embeddings* text =
            reinterpret_cast<aicore_relation_embeddings*>(1);
    REQUIRE(aicore_relation_encode_text(nullptr, nullptr, 0, &text) < 0 &&
            !text);
    aicore_pipeline_timings timings{};
    REQUIRE(aicore_relation_last_pipeline_timings(nullptr, &timings) < 0);
    REQUIRE(!aicore_relation_ready(nullptr) &&
            aicore_relation_last_error(nullptr));
    REQUIRE(aicore_relation_context_device(nullptr));
    REQUIRE(aicore_relation_model_count() == 13);
    REQUIRE(!aicore_relation_model_at(-1) && !aicore_relation_model_at(13));
    int relations = 0, students = 0;
    for (int i = 0; i < 13; ++i) {
        const auto* e = aicore_relation_model_at(i);
        REQUIRE(e && e->size_bytes > 0 && std::strlen(e->sha256) == 64);
        if (e->role == AICORE_RELATION_MODEL_RELATIONS) ++relations;
        if (e->role == AICORE_RELATION_MODEL_TEXT) ++students;
    }
    REQUIRE(relations == 9 && students == 3);
    REQUIRE(aicore_relation_model_at(aicore_relation_model_default_index())
                    ->role == AICORE_RELATION_MODEL_RELATIONS);
    char* dir = aicore_relation_model_cache_dir();
    REQUIRE(dir && std::strstr(dir, "relation_models"));
    aicore_relation_free_buffer(dir);
    aicore_relation_free(nullptr);
    aicore_relation_result_free(nullptr);
    aicore_relation_embeddings_free(nullptr);
    aicore_relation_shutdown();
    aicore_relation_shutdown();
    return 0;
}
