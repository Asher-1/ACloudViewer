// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/rfdetr_capi.h"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "tasks/rfdetr/backend.hpp"
#include "tasks/rfdetr/common.hpp"
#include "tasks/rfdetr/image_io.hpp"
#include "tasks/rfdetr/rfdetr.h"
#include "tasks/rfdetr/rfdetr_model.hpp"

namespace {}  // namespace

using aicore::capi::dup_cstr;
using aicore::capi::json_escape;

struct aicore_rfdetr_options {
    aicore::capi::CommonOptions common;
    /* Class allowlist (whitelist) copied from the caller; empty = detect
     * every class the model knows. Applied at post-processing time (see
     * run_detect). */
    std::vector<uint32_t> class_filter;
};

struct aicore_rfdetr_ctx {
    rfdetr_context* engine = nullptr;
    std::string model_path;
    std::string device;
    int32_t threads = 0;
    std::string last_error;

    /* Class allowlist copied from the options at load time; empty = detect
     * all classes. Lives here (not in the engine) so the C API owns the
     * memory and can hand it to rfdetr_detect_params per call. */
    std::vector<uint32_t> class_filter;

    // Per-detection store from the most recent detect call (masks are kept
    // raw at model resolution so the caller-owned rfdetr_detection array can
    // be freed immediately; the PNG variant is encoded on demand for
    // metadata/export callers only — the live video path reads raw bytes).
    struct DetectionStore {
        uint32_t class_id = 0;
        std::string class_name;
        float score = 0.0f;
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        std::vector<uint8_t> mask_raw;  // empty for detection-only models
        int mask_width = 0;
        int mask_height = 0;
    };
    std::vector<DetectionStore> detections;
};

AICORE_CAPI int aicore_rfdetr_abi_version(void) { return 1; }

AICORE_CAPI aicore_rfdetr_options* aicore_rfdetr_options_new(void) {
    return new (std::nothrow) aicore_rfdetr_options();
}

AICORE_CAPI void aicore_rfdetr_options_free(aicore_rfdetr_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_rfdetr_options_set_device(aicore_rfdetr_options* opts,
                                                  const char* device) {
    if (opts != nullptr) aicore::capi::set_device(opts->common, device);
}

AICORE_CAPI void aicore_rfdetr_options_set_threads(aicore_rfdetr_options* opts,
                                                   int n_threads) {
    if (opts != nullptr) aicore::capi::set_threads(opts->common, n_threads);
}

AICORE_CAPI void aicore_rfdetr_options_set_class_filter(
        aicore_rfdetr_options* opts,
        const uint32_t* class_ids,
        size_t n) {
    if (opts == nullptr) return;
    opts->class_filter.clear();
    if (class_ids == nullptr || n == 0) return;
    opts->class_filter.assign(class_ids, class_ids + n);
}

AICORE_CAPI aicore_rfdetr_ctx* aicore_rfdetr_load_opts(
        const char* gguf_path, const aicore_rfdetr_options* opts) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_rfdetr_ctx();
    if (ctx == nullptr) return nullptr;

    ctx->model_path = gguf_path;
    ctx->device = opts != nullptr ? opts->common.device : "auto";
    ctx->threads = opts != nullptr ? opts->common.threads : 0;
    if (opts != nullptr) ctx->class_filter = opts->class_filter;

    try {
        rfdetr_params p{};
        p.model_path = gguf_path;
        p.n_threads = ctx->threads;
        p.device = ctx->device.c_str();
        rfdetr_status st = RFDETR_OK;
        ctx->engine = rfdetr_init(&p, &st);
        if (ctx->engine == nullptr) {
            ctx->last_error = std::string("failed to load RF-DETR GGUF: ") +
                              rfdetr_status_str(st);
        }
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    }
    return ctx;
}

AICORE_CAPI void aicore_rfdetr_free(aicore_rfdetr_ctx* ctx) {
    if (ctx == nullptr) return;
    if (ctx->engine != nullptr) rfdetr_free(ctx->engine);
    delete ctx;
}

AICORE_CAPI int aicore_rfdetr_is_ready(const aicore_rfdetr_ctx* ctx) {
    return ctx != nullptr && ctx->engine != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_rfdetr_last_error(const aicore_rfdetr_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI void aicore_rfdetr_free_buffer(void* p) { std::free(p); }

AICORE_CAPI int aicore_rfdetr_load_path_rgb(const char* image_path,
                                            uint8_t** out_rgb,
                                            int32_t* out_width,
                                            int32_t* out_height) {
    if (image_path == nullptr || out_rgb == nullptr || out_width == nullptr ||
        out_height == nullptr) {
        return -1;
    }
    *out_rgb = nullptr;
    *out_width = 0;
    *out_height = 0;
    rfdetr_status st = RFDETR_OK;
    rfdetr_image* img = rfdetr_image_load_file(image_path, &st);
    if (img == nullptr) return -1;
    const size_t nbytes = static_cast<size_t>(img->width) *
                          static_cast<size_t>(img->height) * 3;
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(nbytes));
    if (buf != nullptr) {
        std::memcpy(buf, rfdetr_image_rgb_data(img), nbytes);
        *out_rgb = buf;
        *out_width = img->width;
        *out_height = img->height;
    }
    rfdetr_image_free(img);
    return buf != nullptr ? 0 : -1;
}

namespace {

// Shared detect core: runs inference, persists the per-detection store and
// serializes the JSON envelope. The store survives until the next detect call
// so segmentation masks can be fetched lazily by the plugin.
char* run_detect(aicore_rfdetr_ctx* ctx,
                 const rfdetr_image* img,
                 float threshold,
                 uint32_t top_k,
                 int* out_rc) {
    *out_rc = -1;
    if (ctx == nullptr || ctx->engine == nullptr || img == nullptr) {
        return nullptr;
    }
    ctx->detections.clear();

    /* dets lives across the try so an unwinding post-processing step can
     * still release it. Everything below may allocate (mask PNG encode,
     * detections store, JSON envelope); an uncaught bad_alloc would cross
     * the extern "C" boundary and the Qt event loop (queued worker slot)
     * and terminate the process with SIGABRT. */
    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    try {
        rfdetr_detect_params dp{};
        dp.threshold = threshold;
        dp.top_k = top_k;
        /* Class allowlist: empty vector means "all classes" (NULL/0 to the
         * engine — the post-process fast path skips filtering). */
        dp.class_filter =
                ctx->class_filter.empty() ? nullptr : ctx->class_filter.data();
        dp.class_filter_len = ctx->class_filter.size();

        const rfdetr_status st =
                rfdetr_detect(ctx->engine, img, &dp, &dets, &n);
        if (st != RFDETR_OK) {
            ctx->last_error = std::string("RF-DETR inference failed: ") +
                              rfdetr_status_str(st);
            return nullptr;
        }

        // Persist detections (class names are borrowed from the model config —
        // copy them; masks are PNG-encoded so the raw arrays can be freed).
        if (dets != nullptr && n > 0) {
            ctx->detections.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const rfdetr_detection& d = dets[i];
                aicore_rfdetr_ctx::DetectionStore s;
                s.class_id = d.class_id;
                s.class_name = d.class_name != nullptr ? d.class_name : "";
                s.score = d.score;
                s.x1 = d.x1;
                s.y1 = d.y1;
                s.x2 = d.x2;
                s.y2 = d.y2;
                if (d.mask != nullptr && d.mask_width > 0 &&
                    d.mask_height > 0) {
                    /* Keep the thresholded mask raw (0/255, model resolution):
                     * PNG-encoding it here cost 1-3 ms per detection per frame
                     * in the video path, and the plugin decodes it right back.
                     * Raw bytes are memcpy'd once (~0.1 ms) and the PNG form
                     * is encoded lazily for callers that really need it. */
                    const size_t mask_bytes =
                            static_cast<size_t>(d.mask_width) * d.mask_height;
                    s.mask_raw.assign(d.mask, d.mask + mask_bytes);
                    s.mask_width = d.mask_width;
                    s.mask_height = d.mask_height;
                }
                ctx->detections.push_back(std::move(s));
            }
        }
        rfdetr_detections_free(dets, n);
        dets = nullptr;
        n = 0;

        const uint32_t image_size = rfdetr_context_image_size(ctx->engine);
        const uint32_t num_classes = rfdetr_context_num_classes(ctx->engine);
        const uint32_t num_queries = rfdetr_context_num_queries(ctx->engine);
        const int has_seg = rfdetr_context_has_segmentation(ctx->engine);
        std::ostringstream o;
        o << "{\"model\":\"" << json_escape(rfdetr_context_variant(ctx->engine))
          << "\","
          << "\"segmentation\":" << has_seg << ","
          << "\"image_size\":" << image_size << ","
          << "\"num_classes\":" << num_classes << ","
          << "\"num_queries\":" << num_queries << ","
          << "\"image\":{\"width\":" << img->width
          << ",\"height\":" << img->height << "},\"detections\":[";
        for (size_t i = 0; i < ctx->detections.size(); ++i) {
            const auto& d = ctx->detections[i];
            if (i) o << ',';
            o << "{\"class_id\":" << d.class_id << ",\"class_name\":\""
              << json_escape(d.class_name) << "\",\"score\":" << d.score
              << ",\"box\":[" << d.x1 << ',' << d.y1 << ',' << d.x2 << ','
              << d.y2 << "]}";
        }
        o << "]}";
        *out_rc = 0;
        return dup_cstr(o.str());
    } catch (const std::bad_alloc&) {
        ctx->detections.clear();
        rfdetr_detections_free(dets, n);
        ctx->last_error = "RF-DETR out of memory in post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->detections.clear();
        rfdetr_detections_free(dets, n);
        ctx->last_error =
                std::string("RF-DETR post-processing error: ") + e.what();
        return nullptr;
    }
}

}  // namespace

AICORE_CAPI char* aicore_rfdetr_detect_path_json(aicore_rfdetr_ctx* ctx,
                                                 const char* image_path,
                                                 float threshold,
                                                 uint32_t top_k) {
    if (ctx == nullptr || ctx->engine == nullptr || image_path == nullptr) {
        return nullptr;
    }
    rfdetr_status st = RFDETR_OK;
    rfdetr_image* img = rfdetr_image_load_file(image_path, &st);
    if (img == nullptr) {
        ctx->last_error = std::string("failed to load image: ") + image_path;
        return nullptr;
    }
    int rc = -1;
    char* json = run_detect(ctx, img, threshold, top_k, &rc);
    rfdetr_image_free(img);
    return json;
}

AICORE_CAPI char* aicore_rfdetr_detect_rgb_json(aicore_rfdetr_ctx* ctx,
                                                const uint8_t* rgb,
                                                int32_t width,
                                                int32_t height,
                                                float threshold,
                                                uint32_t top_k) {
    if (ctx == nullptr || ctx->engine == nullptr || rgb == nullptr ||
        width <= 0 || height <= 0) {
        return nullptr;
    }
    rfdetr_status st = RFDETR_OK;
    /* Borrow (no copy): the caller's buffer must stay alive for the whole
     * call, which the synchronous C API contract guarantees (preprocess
     * only reads from it). Saves a full-frame copy per detection call. */
    rfdetr_image* img = rfdetr_image_borrow_rgb(rgb, width, height, &st);
    if (img == nullptr) {
        ctx->last_error = "failed to wrap rgb buffer";
        return nullptr;
    }
    int rc = -1;
    char* json = run_detect(ctx, img, threshold, top_k, &rc);
    rfdetr_image_free(img);
    return json;
}

AICORE_CAPI int aicore_rfdetr_detection_count(const aicore_rfdetr_ctx* ctx) {
    if (ctx == nullptr) return -1;
    return static_cast<int>(ctx->detections.size());
}

AICORE_CAPI int aicore_rfdetr_detection_mask(aicore_rfdetr_ctx* ctx,
                                             int index,
                                             unsigned char* buf,
                                             int buf_size,
                                             int32_t* out_width,
                                             int32_t* out_height) {
    if (ctx == nullptr || index < 0 ||
        static_cast<size_t>(index) >= ctx->detections.size()) {
        return -1;
    }
    if (out_width != nullptr) *out_width = 0;
    if (out_height != nullptr) *out_height = 0;
    const auto& d = ctx->detections[static_cast<size_t>(index)];
    if (d.mask_raw.empty()) return 0;  // detection-only model / no mask
    if (out_width != nullptr) *out_width = d.mask_width;
    if (out_height != nullptr) *out_height = d.mask_height;
    const int needed = static_cast<int>(d.mask_raw.size());
    if (buf == nullptr || buf_size < needed) return needed;
    std::memcpy(buf, d.mask_raw.data(), d.mask_raw.size());
    return needed;
}

AICORE_CAPI int aicore_rfdetr_detection_mask_png(aicore_rfdetr_ctx* ctx,
                                                 int index,
                                                 unsigned char* buf,
                                                 int buf_size) {
    if (ctx == nullptr || index < 0 ||
        static_cast<size_t>(index) >= ctx->detections.size()) {
        return -1;
    }
    const auto& d = ctx->detections[static_cast<size_t>(index)];
    if (d.mask_raw.empty()) return 0;  // detection-only model / no mask
    try {
        /* Encoded on demand: the hot video path fetches raw masks through
         * aicore_rfdetr_detection_mask and never pays for PNG; this stays
         * for metadata/export callers. */
        std::vector<uint8_t> png;
        rfdetr_encode_gray_png(d.mask_raw.data(), d.mask_width, d.mask_height,
                               png);
        const int needed = static_cast<int>(png.size());
        if (buf == nullptr || buf_size < needed) return needed;
        std::memcpy(buf, png.data(), png.size());
        return needed;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "RF-DETR out of memory encoding mask PNG";
        return -1;
    } catch (const std::exception& e) {
        ctx->last_error =
                std::string("RF-DETR mask PNG encode error: ") + e.what();
        return -1;
    }
}

AICORE_CAPI const char* aicore_rfdetr_context_variant(
        const aicore_rfdetr_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? rfdetr_context_variant(ctx->engine)
                   : "";
}

AICORE_CAPI uint32_t
aicore_rfdetr_context_image_size(const aicore_rfdetr_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? rfdetr_context_image_size(ctx->engine)
                   : 0;
}

AICORE_CAPI uint32_t
aicore_rfdetr_context_num_classes(const aicore_rfdetr_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? rfdetr_context_num_classes(ctx->engine)
                   : 0;
}

AICORE_CAPI int aicore_rfdetr_context_has_segmentation(
        const aicore_rfdetr_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return 0;
    return rfdetr_context_has_segmentation(ctx->engine);
}

AICORE_CAPI const char* const* aicore_rfdetr_context_class_names(
        aicore_rfdetr_ctx* ctx, uint32_t* out_count) {
    if (out_count != nullptr) *out_count = 0;
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    return rfdetr_context_class_names(ctx->engine, out_count);
}

AICORE_CAPI char* aicore_rfdetr_info_json(aicore_rfdetr_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    std::ostringstream o;
    o << "{\"model\":\"" << json_escape(rfdetr_context_variant(ctx->engine))
      << "\","
      << "\"image_size\":" << rfdetr_context_image_size(ctx->engine) << ","
      << "\"num_classes\":" << rfdetr_context_num_classes(ctx->engine) << ","
      << "\"num_queries\":" << rfdetr_context_num_queries(ctx->engine) << ","
      << "\"segmentation\":" << rfdetr_context_has_segmentation(ctx->engine)
      << ","
      << "\"tensors\":" << rfdetr_context_n_tensors(ctx->engine)
      << ","
      /* The backend-RESOLVED device ("CUDA0", "cpu", ...), not the request:
       * makes a silent CPU fallback visible to callers comparing against
       * GPU latency expectations. */
      << "\"device\":\"" << json_escape(aicore_rfdetr_context_device(ctx))
      << "\","
      << "\"threads\":" << rfdetr_context_n_threads(ctx->engine) << "}";
    return dup_cstr(o.str());
}

AICORE_CAPI const char* aicore_rfdetr_context_device(aicore_rfdetr_ctx* ctx) {
    /* Pointer owned by ctx (stable until the next load); callers must copy
     * it before the context is freed. */
    if (ctx == nullptr || ctx->engine == nullptr) return "";
    const char* resolved = rfdetr_context_device_name(ctx->engine);
    return resolved != nullptr ? resolved : "";
}

AICORE_CAPI int aicore_rfdetr_context_threads(aicore_rfdetr_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return 0;
    return rfdetr_context_n_threads(ctx->engine);
}

AICORE_CAPI int aicore_rfdetr_warmup_backend(const char* device) {
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI void aicore_rfdetr_shutdown(void) {}

AICORE_CAPI char* aicore_rfdetr_model_cache_dir(void) {
    return dup_cstr(aicore::rfdetr_model_cache_dir());
}
