// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AICORE_ATTRIBUTE_ABI_VERSION 1u

typedef struct aicore_attribute_ctx aicore_attribute_ctx;

/** 字符串仅在 create 期间借用。当前接受已核对 source 身份的本地 PULC GGUF；
 *  本地文件支持不代表下载 catalog、量化精度或发布资格。 */
typedef struct aicore_attribute_options {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* model_path;
    const char* device;
    int32_t threads;
} aicore_attribute_options;

/** 严格阈值或组内 argmax 选择的单个属性。
 *  score 为所选分支概率；二值否定分支使用 1-p。
 *  probability_index 指向原始概率，不对输出做 softmax 或重新校准。 */
typedef struct aicore_attribute_item {
    const char* name;
    const char* value;
    float score;
    int32_t probability_index;
} aicore_attribute_item;

/** 独立持有的结果，释放 context 后仍有效。其几何为原图的全图 rectangle，
 *  左上角 (0,0)，右下角 (width,height)。原始概率保留完整 source 输出次序。
 *  调用方不得修改成员或 private_owner；只可用 result_free 释放一次。 */
typedef struct aicore_attribute_result {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* label;
    int32_t width;
    int32_t height;
    const aicore_attribute_item* attributes;
    size_t attribute_count;
    const float* probabilities;
    size_t probability_count;
    void* private_owner;
} aicore_attribute_result;

/** 本地转换 recipe，不是已发布的下载 catalog。仅列出已验证的 F32 路径；
 *  source_sha256 标识原 ONNX，不是 GGUF 的下载摘要。成员及字符串进程期借用。 */
typedef struct aicore_attribute_preset_entry {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* filename;
    const char* display_name;
    const char* label;
    const char* source_sha256;
    size_t probability_count;
    size_t attribute_count;
} aicore_attribute_preset_entry;

AICORE_CAPI int aicore_attribute_preset_count(void);
/** 越界索引返回 NULL；不执行模型加载或推理。 */
AICORE_CAPI const aicore_attribute_preset_entry* aicore_attribute_preset_at(int index);
/** 返回共享数据根目录下的 extract/attribute_models；失败返回 NULL。
 *  返回的独立缓冲区必须使用 free_buffer 释放；不创建目录或下载文件。 */
AICORE_CAPI char* aicore_attribute_model_cache_dir(void);
AICORE_CAPI void aicore_attribute_free_buffer(char* buffer);

AICORE_CAPI aicore_attribute_options aicore_attribute_default_options(void);
/** 失败返回 error context；除分配失败外均可查询 ready/last_error。 */
AICORE_CAPI aicore_attribute_ctx* aicore_attribute_create(
        const aicore_attribute_options* options);
AICORE_CAPI void aicore_attribute_free(aicore_attribute_ctx* ctx);
AICORE_CAPI int aicore_attribute_ready(const aicore_attribute_ctx* ctx);
AICORE_CAPI const char* aicore_attribute_last_error(const aicore_attribute_ctx* ctx);
AICORE_CAPI const char* aicore_attribute_context_device(const aicore_attribute_ctx* ctx);
/** 输入只读借用并保留实际 stride/format；同一 context 的调用须串行。
 *  0 成功，-1 错误；任何错误均将 *output 清空。 */
AICORE_CAPI int aicore_attribute_run(aicore_attribute_ctx* ctx,
                                     const aicore_image_view* image,
                                     aicore_attribute_result** output);
AICORE_CAPI void aicore_attribute_result_free(aicore_attribute_result* result);
/** 尚无成功推理、NULL 或 timing ABI 错误返回 -1；否则填充 timing，返回 0。 */
AICORE_CAPI int aicore_attribute_last_pipeline_timings(
        const aicore_attribute_ctx* ctx, aicore_pipeline_timings* output);
/** 委托公共 runtime shutdown，不销毁仍存活的 context。 */
AICORE_CAPI void aicore_attribute_shutdown(void);

#ifdef __cplusplus
}
#endif
