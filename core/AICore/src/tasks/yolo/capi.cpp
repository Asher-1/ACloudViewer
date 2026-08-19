// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QImage>
#include <QString>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/yolo_capi.h"
#include "backend.hpp"
#include "ggml_backend_utils.hpp"
#include "model_cache.hpp"
#include "yolo_graph.hpp"
#include "yolo_image.hpp"
#include "yolo_postprocess.hpp"

namespace {

char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

// JSON-escape a class name (COCO labels are plain ASCII, but models converted
// from other taxonomies may carry quotes / backslashes).
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
                break;
        }
    }
    return out;
}

}  // namespace

struct aicore_yolo_options {
    std::string device = "auto";
    int32_t threads = 0;
};

struct aicore_yolo_ctx {
    yolo::Session* engine = nullptr;
    std::string model_path;
    std::string device;
    int32_t threads = 0;
    std::string last_error;

    // Statistics of the most recent depth call (surfaced through
    // aicore_yolo_last_depth_json; the float map itself is handed to the
    // caller and not kept here).
    struct DepthStats {
        bool valid = false;
        int width = 0, height = 0;
        int image_w = 0, image_h = 0;
        float min_d = 0, max_d = 0, mean_d = 0, p95_d = 0;
        size_t valid_pixels = 0;
    };
    DepthStats depth;
};

AICORE_CAPI int aicore_yolo_abi_version(void) { return 1; }

AICORE_CAPI aicore_yolo_options* aicore_yolo_options_new(void) {
    return new (std::nothrow) aicore_yolo_options();
}

AICORE_CAPI void aicore_yolo_options_free(aicore_yolo_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_yolo_options_set_device(aicore_yolo_options* opts,
                                                const char* device) {
    if (opts != nullptr && device != nullptr) opts->device = device;
}

AICORE_CAPI void aicore_yolo_options_set_threads(aicore_yolo_options* opts,
                                                 int n_threads) {
    if (opts != nullptr) opts->threads = n_threads;
}

AICORE_CAPI aicore_yolo_ctx* aicore_yolo_load_opts(
        const char* gguf_path, const aicore_yolo_options* opts) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_yolo_ctx();
    if (ctx == nullptr) return nullptr;

    ctx->model_path = gguf_path;
    ctx->device = opts != nullptr ? opts->device : "auto";
    ctx->threads = opts != nullptr ? opts->threads : 0;

    try {
        ctx->engine =
                yolo::create_session(gguf_path, ctx->threads, ctx->device);
        if (ctx->engine == nullptr) {
            ctx->last_error = "failed to load YOLO GGUF";
        }
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    }
    return ctx;
}

AICORE_CAPI void aicore_yolo_free(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr) return;
    if (ctx->engine != nullptr) yolo::free_session(ctx->engine);
    delete ctx;
}

AICORE_CAPI int aicore_yolo_is_ready(const aicore_yolo_ctx* ctx) {
    return ctx != nullptr && ctx->engine != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_yolo_last_error(const aicore_yolo_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI void aicore_yolo_free_string(char* s) { std::free(s); }

AICORE_CAPI void aicore_yolo_free_vec(float* v) { std::free(v); }

AICORE_CAPI int aicore_yolo_load_path_rgb(const char* image_path,
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
    QImage img(QString::fromUtf8(image_path));
    if (img.isNull()) return -1;
    QImage rgb = img.convertToFormat(QImage::Format_RGB888);
    const size_t nbytes = static_cast<size_t>(rgb.width()) *
                          static_cast<size_t>(rgb.height()) * 3;
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(nbytes));
    if (buf == nullptr) return -1;
    const int stride = rgb.bytesPerLine();
    if (stride == rgb.width() * 3) {
        std::memcpy(buf, rgb.constBits(), nbytes);
    } else {
        for (int y = 0; y < rgb.height(); ++y) {
            std::memcpy(buf + static_cast<size_t>(y) * rgb.width() * 3,
                        rgb.constScanLine(y),
                        static_cast<size_t>(rgb.width()) * 3);
        }
    }
    *out_rgb = buf;
    *out_width = rgb.width();
    *out_height = rgb.height();
    return 0;
}

namespace {

// Shared detect core: letterbox, inference, postprocess, JSON envelope.
// Everything may allocate; an uncaught bad_alloc would cross the extern "C"
// boundary and the Qt event loop (queued worker slot) and terminate the
// process with SIGABRT — hence the catch fencing.
char* run_detect(aicore_yolo_ctx* ctx,
                 const uint8_t* rgb,
                 int32_t width,
                 int32_t height,
                 float conf_thres,
                 float iou_thres,
                 uint32_t top_k,
                 int* out_rc) {
    *out_rc = -1;
    yolo::Session* s = ctx->engine;
    if (s == nullptr || rgb == nullptr || width <= 0 || height <= 0) {
        return nullptr;
    }
    if (s->model.meta.task != "detect") {
        ctx->last_error =
                "model is not a detect model (task=" + s->model.meta.task + ")";
        return nullptr;
    }
    ctx->depth.valid = false;

    try {
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(yolo::Image{width, height, rgb},
                              s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        if (!yolo::session_run(s, canvas.data())) {
            ctx->last_error = "YOLO inference failed";
            return nullptr;
        }
        std::vector<float> raw;
        int no = 0, na = 0;
        if (!yolo::session_read_output(s, raw, no, na)) {
            ctx->last_error = "output readback failed";
            return nullptr;
        }
        yolo::PostprocConfig cfg;
        cfg.conf_thres =
                conf_thres > 0.0f && conf_thres < 1.0f ? conf_thres : 0.25f;
        cfg.iou_thres = iou_thres > 0.0f && iou_thres < 1.0f ? iou_thres : 0.7f;
        cfg.max_det = top_k > 0 ? (int)top_k : s->model.meta.max_det;
        std::vector<yolo::Detection> dets =
                yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(),
                                  s->anchor_strides.data(), cfg);
        yolo::unscale_boxes(dets, info);

        const auto& names = s->model.meta.class_names;
        std::ostringstream o;
        o << "{\"model\":\"" << json_escape(s->model.meta.name) << "\","
          << "\"task\":\"detect\","
          << "\"image_size\":" << s->model.meta.imgsz << ","
          << "\"num_classes\":" << s->model.meta.nc << ","
          << "\"end2end\":" << (s->model.meta.end2end ? 1 : 0) << ","
          << "\"image\":{\"width\":" << width << ",\"height\":" << height
          << "},\"detections\":[";
        for (size_t i = 0; i < dets.size(); ++i) {
            const auto& d = dets[i];
            std::string cname =
                    d.class_id >= 0 && d.class_id < (int)names.size()
                            ? names[d.class_id]
                            : ("class " + std::to_string(d.class_id));
            if (i) o << ',';
            o << "{\"class_id\":" << d.class_id << ",\"class_name\":\""
              << json_escape(cname) << "\",\"score\":" << d.score
              << ",\"box\":[" << d.x1 << ',' << d.y1 << ',' << d.x2 << ','
              << d.y2 << "]}";
        }
        o << "]}";
        *out_rc = 0;
        return dup_cstr(o.str());
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error =
                std::string("YOLO post-processing error: ") + e.what();
        return nullptr;
    }
}

// Shared depth core: letterbox, inference, model-resolution readback, restore
// to the original image size, summary statistics, malloc'd float array.
float* run_depth(aicore_yolo_ctx* ctx,
                 const uint8_t* rgb,
                 int32_t width,
                 int32_t height,
                 int32_t* out_width,
                 int32_t* out_height) {
    yolo::Session* s = ctx->engine;
    if (s == nullptr || rgb == nullptr || width <= 0 || height <= 0) {
        return nullptr;
    }
    if (s->model.meta.task != "depth") {
        ctx->last_error =
                "model is not a depth model (task=" + s->model.meta.task + ")";
        return nullptr;
    }

    try {
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(yolo::Image{width, height, rgb},
                              s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        if (!yolo::session_run(s, canvas.data())) {
            ctx->last_error = "YOLO inference failed";
            return nullptr;
        }
        std::vector<float> model_depth;
        int dw = 0, dh = 0;
        if (!yolo::session_read_depth(s, model_depth, dw, dh)) {
            ctx->last_error = "depth readback failed";
            return nullptr;
        }
        std::vector<float> restored =
                yolo::restore_depth(model_depth, dw, dh, info, width, height);
        if (restored.empty()) {
            ctx->last_error = "depth restoration failed";
            return nullptr;
        }

        // Summary statistics over valid pixels (finite && > 0), mirroring the
        // upstream write_depth_png normalization inputs.
        std::vector<float> valid;
        valid.reserve(restored.size());
        double sum = 0.0;
        for (float v : restored) {
            if (std::isfinite(v) && v > 0.0f) {
                valid.push_back(v);
                sum += v;
            }
        }
        ctx->depth.valid = true;
        ctx->depth.width = width;
        ctx->depth.height = height;
        ctx->depth.image_w = width;
        ctx->depth.image_h = height;
        ctx->depth.valid_pixels = valid.size();
        if (!valid.empty()) {
            const float mn = *std::min_element(valid.begin(), valid.end());
            const float mx = *std::max_element(valid.begin(), valid.end());
            const size_t p95 =
                    std::min(valid.size() - 1, valid.size() * 95 / 100);
            std::nth_element(valid.begin(), valid.begin() + p95, valid.end());
            ctx->depth.min_d = mn;
            ctx->depth.max_d = mx;
            ctx->depth.mean_d = (float)(sum / (double)valid.size());
            ctx->depth.p95_d = valid[p95];
        } else {
            ctx->depth.min_d = ctx->depth.max_d = ctx->depth.mean_d =
                    ctx->depth.p95_d = 0.0f;
        }

        float* out = static_cast<float*>(
                std::malloc(restored.size() * sizeof(float)));
        if (out == nullptr) {
            ctx->depth.valid = false;
            ctx->last_error = "YOLO out of memory returning depth map";
            return nullptr;
        }
        std::memcpy(out, restored.data(), restored.size() * sizeof(float));
        if (out_width != nullptr) *out_width = width;
        if (out_height != nullptr) *out_height = height;
        return out;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in depth post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO depth error: ") + e.what();
        return nullptr;
    }
}

}  // namespace

AICORE_CAPI char* aicore_yolo_detect_path_json(aicore_yolo_ctx* ctx,
                                               const char* image_path,
                                               float conf_thres,
                                               float iou_thres,
                                               uint32_t top_k) {
    if (ctx == nullptr || ctx->engine == nullptr || image_path == nullptr) {
        return nullptr;
    }
    QImage img(QString::fromUtf8(image_path));
    if (img.isNull()) {
        ctx->last_error = std::string("failed to load image: ") + image_path;
        return nullptr;
    }
    QImage rgb = img.convertToFormat(QImage::Format_RGB888);
    std::vector<uint8_t> packed;
    const int w = rgb.width(), h = rgb.height();
    packed.resize((size_t)w * h * 3);
    if (rgb.bytesPerLine() == w * 3) {
        std::memcpy(packed.data(), rgb.constBits(), packed.size());
    } else {
        for (int y = 0; y < h; ++y) {
            std::memcpy(packed.data() + (size_t)y * w * 3, rgb.constScanLine(y),
                        (size_t)w * 3);
        }
    }
    int rc = -1;
    return run_detect(ctx, packed.data(), w, h, conf_thres, iou_thres, top_k,
                      &rc);
}

AICORE_CAPI char* aicore_yolo_detect_rgb_json(aicore_yolo_ctx* ctx,
                                              const uint8_t* rgb,
                                              int32_t width,
                                              int32_t height,
                                              float conf_thres,
                                              float iou_thres,
                                              uint32_t top_k) {
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    int rc = -1;
    /* Borrow (no copy): the caller's buffer must stay alive for the whole
     * call, which the synchronous C API contract guarantees (preprocess
     * only reads from it). Saves a full-frame copy per detection call. */
    return run_detect(ctx, rgb, width, height, conf_thres, iou_thres, top_k,
                      &rc);
}

AICORE_CAPI float* aicore_yolo_depth_path(aicore_yolo_ctx* ctx,
                                          const char* image_path,
                                          int32_t* out_width,
                                          int32_t* out_height) {
    if (ctx == nullptr || ctx->engine == nullptr || image_path == nullptr) {
        return nullptr;
    }
    QImage img(QString::fromUtf8(image_path));
    if (img.isNull()) {
        ctx->last_error = std::string("failed to load image: ") + image_path;
        return nullptr;
    }
    QImage rgb = img.convertToFormat(QImage::Format_RGB888);
    std::vector<uint8_t> packed;
    const int w = rgb.width(), h = rgb.height();
    packed.resize((size_t)w * h * 3);
    if (rgb.bytesPerLine() == w * 3) {
        std::memcpy(packed.data(), rgb.constBits(), packed.size());
    } else {
        for (int y = 0; y < h; ++y) {
            std::memcpy(packed.data() + (size_t)y * w * 3, rgb.constScanLine(y),
                        (size_t)w * 3);
        }
    }
    return run_depth(ctx, packed.data(), w, h, out_width, out_height);
}

AICORE_CAPI float* aicore_yolo_depth_rgb(aicore_yolo_ctx* ctx,
                                         const uint8_t* rgb,
                                         int32_t width,
                                         int32_t height,
                                         int32_t* out_width,
                                         int32_t* out_height) {
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    if (out_width != nullptr) *out_width = 0;
    if (out_height != nullptr) *out_height = 0;
    return run_depth(ctx, rgb, width, height, out_width, out_height);
}

AICORE_CAPI char* aicore_yolo_last_depth_json(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr || !ctx->depth.valid) {
        return nullptr;
    }
    try {
        const auto& st = ctx->depth;
        std::ostringstream o;
        o << "{\"model\":\"" << json_escape(ctx->engine->model.meta.name)
          << "\","
          << "\"task\":\"depth\","
          << "\"image_size\":" << ctx->engine->model.meta.imgsz << ","
          << "\"image\":{\"width\":" << st.image_w
          << ",\"height\":" << st.image_h << "},"
          << "\"depth_width\":" << st.width << ",\"depth_height\":" << st.height
          << ","
          << "\"min_depth\":" << st.min_d << ",\"max_depth\":" << st.max_d
          << ","
          << "\"mean_depth\":" << st.mean_d << ",\"p95_depth\":" << st.p95_d
          << ","
          << "\"valid_pixels\":" << st.valid_pixels << "}";
        return dup_cstr(o.str());
    } catch (const std::exception& e) {
        ctx->last_error =
                std::string("depth stats serialization failed: ") + e.what();
        return nullptr;
    }
}

AICORE_CAPI const char* aicore_yolo_context_task(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? ctx->engine->model.meta.task.c_str()
                   : "";
}

AICORE_CAPI const char* aicore_yolo_context_model_name(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? ctx->engine->model.meta.name.c_str()
                   : "";
}

AICORE_CAPI uint32_t aicore_yolo_context_image_size(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? (uint32_t)ctx->engine->model.meta.imgsz
                   : 0;
}

AICORE_CAPI uint32_t aicore_yolo_context_num_classes(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? (uint32_t)ctx->engine->model.meta.nc
                   : 0;
}

AICORE_CAPI int aicore_yolo_context_end2end(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr &&
            ctx->engine->model.meta.end2end)
                   ? 1
                   : 0;
}

AICORE_CAPI const char* aicore_yolo_context_device(aicore_yolo_ctx* ctx) {
    /* Pointer owned by ctx (stable until the next load); callers must copy
     * it before the context is freed. */
    if (ctx == nullptr || ctx->engine == nullptr) return "";
    const std::string& resolved = ctx->engine->backend.device_name;
    return resolved.empty() ? "" : resolved.c_str();
}

AICORE_CAPI int aicore_yolo_context_threads(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return 0;
    return ctx->engine->backend.n_threads;
}

AICORE_CAPI char* aicore_yolo_info_json(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    const yolo::ModelMeta& meta = ctx->engine->model.meta;
    std::ostringstream o;
    o << "{\"model\":\"" << json_escape(meta.name) << "\","
      << "\"task\":\"" << json_escape(meta.task) << "\","
      << "\"image_size\":" << meta.imgsz << ","
      << "\"num_classes\":" << meta.nc << ","
      << "\"end2end\":" << (meta.end2end ? 1 : 0) << ","
      << "\"reg_max\":" << meta.reg_max << ","
      << "\"dtype\":\"" << json_escape(meta.dtype)
      << "\","
      /* The backend-RESOLVED device, not the request: makes a silent CPU
       * fallback visible to callers comparing against GPU latency
       * expectations. */
      << "\"device\":\"" << json_escape(aicore_yolo_context_device(ctx))
      << "\","
      << "\"threads\":" << ctx->engine->backend.n_threads << "}";
    return dup_cstr(o.str());
}

AICORE_CAPI int aicore_yolo_warmup_backend(const char* device) {
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI void aicore_yolo_shutdown(void) {}

AICORE_CAPI char* aicore_yolo_model_cache_dir(void) {
    return dup_cstr(aicore::yolo_model_cache_dir());
}
