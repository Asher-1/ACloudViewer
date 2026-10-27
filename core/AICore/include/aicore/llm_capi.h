// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** Local chat LLM / VLM generation backed by the pinned llama.cpp (libllama
 *  + libmtmd) that shares AICore's embedded ggml.
 *
 *  One context = one text model plus an optional multimodal projector
 *  (mmproj). Without a projector the context is a text-only LLM; with one it
 *  also accepts images. A VLM is therefore not a second task: it is the same
 *  session with one more owned asset.
 *
 *  Each generate call is stateless: the request carries the full message
 *  history and the context clears its KV memory before prefill, so results
 *  never depend on a previous request. Calls on one context must be
 *  serialized; aicore_llm_cancel may be called from any thread. */

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AICORE_LLM_ABI_VERSION 1u
typedef struct aicore_llm_ctx aicore_llm_ctx;

typedef enum aicore_llm_finish_reason {
    AICORE_LLM_FINISH_STOP = 1,      /**< end-of-generation token or stop
                                        sequence */
    AICORE_LLM_FINISH_LENGTH = 2,    /**< max_tokens or context exhausted */
    AICORE_LLM_FINISH_CANCELLED = 3, /**< aicore_llm_cancel / runtime cancel */
} aicore_llm_finish_reason;

/** Strings are borrowed for the duration of create only. */
typedef struct aicore_llm_options {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* model_path;  /**< text model GGUF (required) */
    const char* mmproj_path; /**< projector GGUF; NULL/"" = text-only */
    const char* device; /**< auto, cpu, cuda, cuda:1, vulkan, vulkan:1, metal */
    int32_t threads;    /**< <=0 selects the runtime CPU default */
    uint32_t context_tokens; /**< KV size; 0 = min(model train ctx, 4096) */
    uint32_t batch_tokens;   /**< logical prefill batch; 0 = 512 */
    int32_t flash_attn;      /**< -1 auto (llama default), 0 off, 1 on */
    int32_t image_max_tokens; /**< 0 = projector metadata default */
} aicore_llm_options;

typedef struct aicore_llm_message {
    const char* role;    /**< "system", "user" or "assistant" */
    const char* content; /**< UTF-8; may contain aicore_llm_media_marker() */
} aicore_llm_message;

/** Called with each decoded UTF-8 fragment (complete code points only). The
 *  fragment is borrowed for the duration of the callback. Return non-zero to
 *  stop generation (reported as AICORE_LLM_FINISH_CANCELLED). */
typedef int (*aicore_llm_stream_cb)(const char* chunk_utf8,
                                    size_t chunk_bytes,
                                    void* user_data);

/** Images are borrowed for the duration of generate. When the messages
 *  contain no media marker, one marker per image is prepended to the last
 *  user message; otherwise the marker count must equal image_count. */
typedef struct aicore_llm_request {
    uint32_t struct_size;
    uint32_t abi_version;
    const aicore_llm_message* messages;
    size_t message_count;
    const aicore_image_view* images;
    size_t image_count;
    uint32_t max_tokens; /**< 1..context; default 512 */
    float temperature;   /**< 0 = greedy (deterministic) */
    float top_p;         /**< (0,1]; used only when temperature > 0 */
    uint32_t seed;       /**< sampling seed when temperature > 0 */
    const char* const* stop_sequences;
    size_t stop_count;
    aicore_llm_stream_cb stream_cb;
    void* stream_user_data;
} aicore_llm_request;

/** Independently owned result; valid after the context is freed. Release
 *  exactly once with aicore_llm_result_free.
 *
 *  Timing stages: preprocess = chat template + tokenization + image
 *  preprocessing; inference = prefill (text + image encode) + decode loop;
 *  postprocess = stop-sequence trim and result construction; e2e = API entry
 *  until the result is ready. prefill_ms/decode_ms split inference. */
typedef struct aicore_llm_result {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* text; /**< UTF-8, NUL-terminated */
    size_t text_bytes;
    uint32_t finish_reason; /**< aicore_llm_finish_reason */
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    const int32_t* token_ids; /**< generated token ids (generated_tokens) */
    double prefill_ms;
    double decode_ms;
    aicore_pipeline_timings timings;
    void* private_owner; /**< opaque; callers must not inspect or modify */
} aicore_llm_result;

typedef enum aicore_llm_model_role {
    AICORE_LLM_ROLE_MODEL = 0,
    AICORE_LLM_ROLE_MMPROJ = 1,
} aicore_llm_model_role;

typedef struct aicore_llm_model_entry {
    const char* filename;
    const char* download_url; /**< revision-pinned resolver URL */
    const char* display_name;
    const char* family;       /**< pairs a model with its projector */
    int32_t role;             /**< aicore_llm_model_role */
    const char* quantization; /**< q8_0, f16, ... */
    const char* mmproj_filename; /**< recommended projector for vision
                                      models; NULL for text-only/projectors */
    const char* sha256;
    int64_t size_bytes;
} aicore_llm_model_entry;

AICORE_CAPI aicore_llm_options aicore_llm_default_options(void);
AICORE_CAPI aicore_llm_request aicore_llm_default_request(void);
/** Returns an error context on load failure; inspect ready/last_error. A
 *  projector that does not match the text model fails the load. */
AICORE_CAPI aicore_llm_ctx* aicore_llm_create(const aicore_llm_options* options);
AICORE_CAPI void aicore_llm_free(aicore_llm_ctx* ctx);
AICORE_CAPI int aicore_llm_ready(const aicore_llm_ctx* ctx);
AICORE_CAPI const char* aicore_llm_last_error(const aicore_llm_ctx* ctx);
AICORE_CAPI const char* aicore_llm_context_device(const aicore_llm_ctx* ctx);
/** 1 when a projector is loaded and supports image input. */
AICORE_CAPI int aicore_llm_supports_vision(const aicore_llm_ctx* ctx);
/** Media marker understood inside message content (static string). */
AICORE_CAPI const char* aicore_llm_media_marker(void);
/** Returns 0 on success (including stop/length), 1 when cancelled (the
 *  partial result is still returned), -1 on error (output set to NULL). */
AICORE_CAPI int aicore_llm_generate(aicore_llm_ctx* ctx,
                                    const aicore_llm_request* request,
                                    aicore_llm_result** output);
/** Thread-safe request to stop the in-flight generate of this context. */
AICORE_CAPI void aicore_llm_cancel(aicore_llm_ctx* ctx);
AICORE_CAPI void aicore_llm_result_free(aicore_llm_result* result);
AICORE_CAPI int aicore_llm_last_pipeline_timings(
        const aicore_llm_ctx* ctx, aicore_pipeline_timings* output);
AICORE_CAPI int aicore_llm_model_count(void);
AICORE_CAPI const aicore_llm_model_entry* aicore_llm_model_at(int index);
AICORE_CAPI const aicore_llm_model_entry* aicore_llm_model_by_filename(
        const char* filename);
AICORE_CAPI int aicore_llm_model_default_index(void);
AICORE_CAPI char* aicore_llm_model_cache_dir(void);
AICORE_CAPI void aicore_llm_free_buffer(void* buffer);
/** Delegates to aicore_runtime_shutdown(); never destroys live contexts. */
AICORE_CAPI void aicore_llm_shutdown(void);

#ifdef __cplusplus
}
#endif
