// AnyCalib 公共 ABI 的无模型契约：目录查找、分配释放与错误输出所有权。
#include "aicore/anycalib_capi.h"
#include "tests/common/test_macros.hpp"

#include <cstring>
#include <filesystem>
#include <string>

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_anycalib_abi_version() == 1);
    AICORE_CHECK(aicore_anycalib_model_by_filename(nullptr) == nullptr);
    AICORE_CHECK(aicore_anycalib_model_by_filename("not-a-model.gguf") == nullptr);
    AICORE_CHECK(aicore_anycalib_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_anycalib_model_at(aicore_anycalib_model_count()) == nullptr);
    AICORE_CHECK(aicore_anycalib_model_count() > 0);
    for (int i = 0; i < aicore_anycalib_model_count(); ++i) {
        const auto* entry = aicore_anycalib_model_at(i);
        AICORE_CHECK(entry != nullptr && entry->filename != nullptr);
        if (entry && entry->filename)
            AICORE_CHECK(aicore_anycalib_model_by_filename(entry->filename) == entry);
    }
    char* cache = aicore_anycalib_model_cache_dir();
    AICORE_CHECK(cache != nullptr);
    if (cache) {
        AICORE_CHECK(std::filesystem::path(cache).filename() == "anycalib_models");
        aicore_anycalib_free_buffer(cache);
    }
    aicore_anycalib_options_set_threads(nullptr, 2);
    auto* options = aicore_anycalib_options_new();
    AICORE_CHECK(options != nullptr);
    aicore_anycalib_options_set_threads(options, 2);
    aicore_anycalib_options_set_device(options, "cpu");
    auto* ctx = aicore_anycalib_load_opts(nullptr, options);
    aicore_anycalib_options_free(options);
    AICORE_CHECK(ctx != nullptr && aicore_anycalib_is_ready(ctx) == 0);
    const char* error = aicore_anycalib_last_error(ctx);
    AICORE_CHECK(error && std::strstr(error, "model path"));
    float* rays = reinterpret_cast<float*>(1);
    float* tangent = reinterpret_cast<float*>(1);
    int32_t count = 99;
    AICORE_CHECK(aicore_anycalib_run(ctx, nullptr, &rays, &tangent, &count) < 0);
    AICORE_CHECK(rays == nullptr && tangent == nullptr && count == 0);
    aicore_anycalib_free(ctx);
    aicore_anycalib_shutdown();
    aicore_anycalib_shutdown();
    return failures ? 1 : 0;
}
