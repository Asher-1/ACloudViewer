// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "aicore/attribute_capi.h"
#include <cstdio>
#include <cstring>
#include <type_traits>

static int errors = 0;
static void check(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); ++errors; }
}

int main() {
    static_assert(std::is_standard_layout<aicore_attribute_options>::value, "C ABI");
    static_assert(std::is_standard_layout<aicore_attribute_result>::value, "C ABI");
    static_assert(std::is_standard_layout<aicore_attribute_preset_entry>::value, "C ABI");
    check(aicore_attribute_preset_count() == 2, "缺少 person/vehicle 本地 recipe");
    for (int i = 0; i < aicore_attribute_preset_count(); ++i) {
        const auto* preset = aicore_attribute_preset_at(i);
        check(preset && preset->struct_size == sizeof(*preset) && preset->abi_version == 1,
              "本地 recipe ABI 无效");
        check(preset && std::strstr(preset->filename, "-f32.gguf") &&
              std::strlen(preset->source_sha256) == 64, "recipe 不能冒充未验量化下载资产");
    }
    check(!aicore_attribute_preset_at(-1) && !aicore_attribute_preset_at(2), "recipe 索引越界");
    char* directory = aicore_attribute_model_cache_dir();
    check(directory && std::strstr(directory, "attribute_models"), "缺少 owning 模型缓存目录");
    aicore_attribute_free_buffer(directory);
    aicore_attribute_free_buffer(nullptr);
    const auto defaults = aicore_attribute_default_options();
    check(defaults.struct_size == sizeof(defaults) &&
          defaults.abi_version == AICORE_ATTRIBUTE_ABI_VERSION, "默认 options ABI 无效");
    check(defaults.model_path == nullptr, "默认值不能绑定开发机模型");
    for (int kind = 0; kind < 4; ++kind) {
        auto options = defaults;
        options.device = "cpu";
        if (kind == 0) options.struct_size = 0;
        if (kind == 1) options.abi_version = 0;
        if (kind == 2) options.abi_version = AICORE_ATTRIBUTE_ABI_VERSION + 1;
        if (kind == 3) options.model_path = "attribute-contract-missing.gguf";
        auto* context = aicore_attribute_create(&options);
        check(aicore_attribute_ready(context) == 0, "无效参数或缺失模型不能 ready");
        check(aicore_attribute_last_error(context) &&
              std::strlen(aicore_attribute_last_error(context)) > 0, "失败必须暴露错误");
        check(std::strlen(aicore_attribute_context_device(context)) == 0,
              "加载失败不能虚报设备");
        aicore_attribute_result* output = nullptr;
        check(aicore_attribute_run(context, nullptr, &output) == -1 && !output,
              "无效 run 必须清空输出");
        aicore_pipeline_timings timing{};
        timing.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
        check(aicore_attribute_last_pipeline_timings(context, &timing) == -1,
              "未推理不能虚报 timing");
        aicore_attribute_free(context);
    }
    aicore_attribute_result* output = nullptr;
    check(aicore_attribute_run(nullptr, nullptr, &output) == -1 && !output,
          "NULL run 没有失败");
    check(aicore_attribute_ready(nullptr) == 0, "NULL ready");
    check(aicore_attribute_last_error(nullptr) != nullptr, "NULL last_error");
    check(aicore_attribute_last_pipeline_timings(nullptr, nullptr) == -1, "NULL timing");
    aicore_attribute_result_free(nullptr);
    aicore_attribute_free(nullptr);
    aicore_attribute_shutdown();
    std::printf("attribute C ABI: %d failures\n", errors);
    return errors ? 1 : 0;
}
