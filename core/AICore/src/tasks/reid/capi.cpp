// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// ReID C API implementation: a classify-task YOLO session wrapped with the
// official appearance-embedding extraction semantics (save_one_box crops,
// square stretch to the model input, pooled pre-linear feature readback).

#include <QImage>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/reid_capi.h"
#include "aicore/yolo_capi.h"
#include "common/capi_utils.hpp"
#include "tasks/yolo/yolo_gguf_loader.hpp"
#include "tasks/yolo/yolo_graph.hpp"

constexpr int kReidAbiVersion = 1;

// The opaque context behind the aicore_reid_ctx typedef in the public
// header (global scope; the internal state stays private to this TU).
struct aicore_reid_ctx {
    yolo::Session* engine = nullptr;
    std::string last_error;
    int embed_dim = 0;
    bool has_embed = false;
    aicore_pipeline_timings timings{};
};

struct aicore_reid_options {
    aicore::capi::CommonOptions common;
};

namespace {

// Official save_one_box default semantics (utils/ops.py): wh * gain + pad
// per side, center kept, truncating integer cast, boundary clip. The crop
// is NOT squared — the caller stretches it to the model input afterwards.
QImage crop_box_save_one_box(const QImage& img, const float* box) {
    constexpr float kGain = 1.02f;
    constexpr int kPad = 10;
    const float w = box[2] - box[0], h = box[3] - box[1];
    const float cx = (box[0] + box[2]) / 2.0f;
    const float cy = (box[1] + box[3]) / 2.0f;
    const float nw = w * kGain + 2 * kPad;
    const float nh = h * kGain + 2 * kPad;
    // numpy .long() truncates toward zero.
    int x1 = (int)(cx - nw / 2), y1 = (int)(cy - nh / 2);
    int x2 = (int)(cx + nw / 2), y2 = (int)(cy + nh / 2);
    x1 = std::clamp(x1, 0, img.width());
    y1 = std::clamp(y1, 0, img.height());
    x2 = std::clamp(x2, 0, img.width());
    y2 = std::clamp(y2, 0, img.height());
    if (x2 - x1 < 1 || y2 - y1 < 1) return QImage();
    return img.copy(x1, y1, x2 - x1, y2 - y1);
}

// One crop -> [3, S, S] float tensor in [0, 1] (RGB, /255), stretched to
// the model's square input with bilinear interpolation, mirroring the
// upstream _crops_to_tensor geometry.
bool crop_to_chw(const QImage& crop, int imgsz, std::vector<float>& chw) {
    if (crop.isNull()) return false;
    QImage rgb = crop;
    if (rgb.format() != QImage::Format_RGB888) {
        rgb = rgb.convertToFormat(QImage::Format_RGB888);
    }
    if (rgb.width() != imgsz || rgb.height() != imgsz) {
        rgb = rgb.scaled(imgsz, imgsz, Qt::IgnoreAspectRatio,
                         Qt::SmoothTransformation);
        // QImage::scaled(SmoothTransformation) routes through
        // ARGB32_Premultiplied and returns an RGB32-family image even when
        // the source is RGB888. Convert back or the scanline walk below
        // reads 4-byte pixels with a 3-byte stride (channel-shifted input
        // with alpha spikes — measured cos 0.36 vs the torch baseline).
        if (rgb.format() != QImage::Format_RGB888)
            rgb = rgb.convertToFormat(QImage::Format_RGB888);
    }
    if (rgb.isNull() || rgb.width() != imgsz || rgb.height() != imgsz) {
        return false;
    }
    chw.assign((size_t)3 * imgsz * imgsz, 0.0f);
    const int plane = imgsz * imgsz;
    for (int y = 0; y < imgsz; ++y) {
        const uchar* row = rgb.constScanLine(y);
        for (int x = 0; x < imgsz; ++x) {
            const size_t p = (size_t)y * imgsz + x;
            chw[p] = row[x * 3 + 0] / 255.0f;
            chw[plane + p] = row[x * 3 + 1] / 255.0f;
            chw[2 * (size_t)plane + p] = row[x * 3 + 2] / 255.0f;
        }
    }
    return true;
}

QImage view_to_rgb(const aicore_image_view* view) {
    if (view == nullptr || view->data == nullptr || view->width <= 0 ||
        view->height <= 0) {
        return QImage();
    }
    QImage img;
    switch (view->format) {
        case AICORE_IMAGE_RGB8:
            img = QImage(view->data, view->width, view->height,
                         (int)view->row_stride_bytes, QImage::Format_RGB888);
            break;
        case AICORE_IMAGE_BGR8:
            img = QImage(view->data, view->width, view->height,
                         (int)view->row_stride_bytes, QImage::Format_BGR888);
            break;
        case AICORE_IMAGE_RGBA8:
            img = QImage(view->data, view->width, view->height,
                         (int)view->row_stride_bytes, QImage::Format_RGBA8888);
            break;
        case AICORE_IMAGE_BGRA8:
            img = QImage(view->data, view->width, view->height,
                         (int)view->row_stride_bytes, QImage::Format_ARGB32);
            break;
        case AICORE_IMAGE_GRAY8:
            img = QImage(view->data, view->width, view->height,
                         (int)view->row_stride_bytes,
                         QImage::Format_Grayscale8);
            break;
        default:
            return QImage();
    }
    if (img.isNull()) return QImage();
    // Deep-copy once so the crop/scale pipeline never reads caller storage
    // after the call returns, normalizing to RGB888 on the way.
    return img.convertToFormat(QImage::Format_RGB888);
}

}  // namespace

AICORE_CAPI int aicore_reid_abi_version(void) { return kReidAbiVersion; }

AICORE_CAPI aicore_reid_options* aicore_reid_options_new(void) {
    return new (std::nothrow) aicore_reid_options();
}

AICORE_CAPI void aicore_reid_options_free(aicore_reid_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_reid_options_set_device(aicore_reid_options* opts,
                                                const char* device) {
    if (opts != nullptr) aicore::capi::set_device(opts->common, device);
}

AICORE_CAPI void aicore_reid_options_set_threads(aicore_reid_options* opts,
                                                 int n_threads) {
    if (opts != nullptr) aicore::capi::set_threads(opts->common, n_threads);
}

AICORE_CAPI aicore_reid_ctx* aicore_reid_load_opts(
        const char* gguf_path, const aicore_reid_options* opts) {
    auto* ctx = new (std::nothrow) aicore_reid_ctx();
    if (ctx == nullptr) return nullptr;
    try {
        if (gguf_path == nullptr || gguf_path[0] == '\0') {
            ctx->last_error = "model path is empty";
            return ctx;
        }
        yolo::SessionOptions sopts;
        sopts.threads = opts != nullptr ? opts->common.threads : 0;
        // Two encoder flavors exist: native reid-task GGUFs (converted from
        // the official yolo26{n,s,m,l,x}-reid.onnx assets; the graph output is
        // the embedding) and legacy classify-task GGUFs (the embedding is the
        // pooled feature tapped before the final linear). The graph flavor is
        // decided from the file's task kv before building.
        const bool native_reid = yolo::read_gguf_meta(gguf_path).task == "reid";
        sopts.export_embed = !native_reid;
        ctx->engine = yolo::create_session(
                gguf_path,
                opts != nullptr ? opts->common.device : std::string("auto"),
                sopts);
        if (ctx->engine == nullptr) {
            ctx->last_error = "failed to load the ReID encoder GGUF";
            return ctx;
        }
        const std::string& task = ctx->engine->model.meta.task;
        if (task != "reid" && task != "classify") {
            ctx->last_error =
                    "ReID encoders must be reid-task or classify-task GGUFs "
                    "(task=" +
                    task + ")";
            yolo::free_session(ctx->engine);
            ctx->engine = nullptr;
            return ctx;
        }
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    }
    return ctx;
}

AICORE_CAPI void aicore_reid_free(aicore_reid_ctx* ctx) {
    if (ctx == nullptr) return;
    if (ctx->engine != nullptr) yolo::free_session(ctx->engine);
    delete ctx;
}

AICORE_CAPI int aicore_reid_is_ready(const aicore_reid_ctx* ctx) {
    return ctx != nullptr && ctx->engine != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_reid_last_error(const aicore_reid_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI void aicore_reid_free_buffer(void* p) { std::free(p); }

AICORE_CAPI int aicore_reid_embed_dim(const aicore_reid_ctx* ctx) {
    if (ctx == nullptr) return -1;
    return ctx->has_embed ? ctx->embed_dim : 0;
}

AICORE_CAPI int aicore_reid_embed_image(aicore_reid_ctx* ctx,
                                        const aicore_image_view* image,
                                        const float* boxes_xyxy,
                                        int32_t count,
                                        float** out_embed,
                                        int32_t* out_count,
                                        int32_t* out_dim) {
    if (out_embed != nullptr) *out_embed = nullptr;
    if (out_count != nullptr) *out_count = 0;
    if (out_dim != nullptr) *out_dim = 0;
    if (ctx == nullptr || out_embed == nullptr || out_count == nullptr ||
        out_dim == nullptr) {
        return -1;
    }
    yolo::Session* s = ctx->engine;
    if (s == nullptr) {
        ctx->last_error = "ReID context is not ready";
        return -1;
    }
    if (image == nullptr && count > 0) {
        ctx->last_error = "image view is required for a non-empty batch";
        return -1;
    }
    if (count < 0 || (count > 0 && boxes_xyxy == nullptr)) {
        ctx->last_error = "invalid box batch";
        return -1;
    }

    try {
        const auto t_e2e = std::chrono::steady_clock::now();
        // Phase timers accumulate the per-box work; stage boundaries follow
        // aicore/pipeline_timing.h: preprocess = view validation +
        // save_one_box crop + stretch/normalize (CHW staging); inference =
        // upload + graph execution + embedding readback; postprocess =
        // embedding aggregation + typed-result materialization.
        double preprocess_ms = 0.0;
        double inference_ms = 0.0;
        double postprocess_ms = 0.0;
        auto t_phase = t_e2e;
        QImage img;
        int imgsz = s->model.meta.imgsz;
        if (count > 0) {
            img = view_to_rgb(image);
            if (img.isNull()) {
                ctx->last_error = "unsupported image view format or dimensions";
                return -1;
            }
            if (!yolo::session_ensure_canvas(s, imgsz, imgsz)) {
                ctx->last_error = "graph rebuild for the encoder input failed";
                return -1;
            }
            preprocess_ms += yolo::ms_since(t_phase);
        }

        std::vector<float> rows;
        int dim = 0;
        int produced = 0;
        std::vector<float> chw;  // reused across boxes (assign keeps capacity)
        for (int32_t i = 0; i < count; ++i) {
            t_phase = std::chrono::steady_clock::now();
            const QImage crop =
                    crop_box_save_one_box(img, boxes_xyxy + 4 * (size_t)i);
            if (!crop_to_chw(crop, imgsz, chw)) {
                ctx->last_error = "crop preprocessing failed";
                return -1;
            }
            preprocess_ms += yolo::ms_since(t_phase);
            t_phase = std::chrono::steady_clock::now();
            if (!yolo::session_run(s, chw.data())) {
                ctx->last_error = "ReID inference failed";
                return -1;
            }
            std::vector<float> vec;
            int d = 0;
            if (!yolo::session_read_embed(s, vec, d) || d <= 0) {
                ctx->last_error = "embedding readback failed";
                return -1;
            }
            // Readback belongs to inference per the timing contract.
            inference_ms += yolo::ms_since(t_phase);
            t_phase = std::chrono::steady_clock::now();
            if (dim == 0) {
                dim = d;
                rows.reserve((size_t)count * (size_t)dim);
            } else if (d != dim) {
                ctx->last_error = "inconsistent embedding dimension";
                return -1;
            }
            rows.insert(rows.end(), vec.begin(), vec.end());
            ++produced;
            postprocess_ms += yolo::ms_since(t_phase);
        }
        ctx->embed_dim = dim;
        ctx->has_embed = dim > 0;

        t_phase = std::chrono::steady_clock::now();
        auto* out = (float*)std::malloc(
                rows.empty() ? 1 : rows.size() * sizeof(float));
        if (out == nullptr) {
            ctx->last_error = "ReID out of memory for embeddings";
            return -1;
        }
        if (!rows.empty()) {
            std::memcpy(out, rows.data(), rows.size() * sizeof(float));
        }
        *out_embed = out;
        if (out_count != nullptr) *out_count = produced;
        if (out_dim != nullptr) *out_dim = dim;
        postprocess_ms += yolo::ms_since(t_phase);

        ctx->timings = {};
        ctx->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
        // Honest valid_fields: preprocess/inference only ran (and were
        // measured) for a non-empty batch; the aggregation/copy postprocess
        // stage always executes; e2e always covers the whole call.
        ctx->timings.valid_fields =
                AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
        if (count > 0) {
            ctx->timings.valid_fields |=
                    AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE;
        }
        ctx->timings.preprocess_ms = preprocess_ms;
        ctx->timings.inference_ms = inference_ms;
        ctx->timings.postprocess_ms = postprocess_ms;
        ctx->timings.e2e_ms = yolo::ms_since(t_e2e);
        return 0;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "ReID out of memory";
        return -1;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("ReID error: ") + e.what();
        return -1;
    }
}

AICORE_CAPI int aicore_reid_last_pipeline_timings(
        const aicore_reid_ctx* ctx, aicore_pipeline_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr ||
        ctx->timings.valid_fields == 0) {
        return -1;
    }
    *out_timings = ctx->timings;
    return 0;
}

AICORE_CAPI void aicore_reid_shutdown(void) {
    // ReID shares the process-wide backend registry with the other tasks;
    // the common shutdown is idempotent and never destroys live contexts.
    aicore_yolo_shutdown();
}
