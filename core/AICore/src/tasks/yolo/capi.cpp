// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QImage>
#include <QString>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/yolo_capi.h"
#include "common/capi_utils.hpp"
#include "common/debug_dump.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "common/runtime_cleanup.hpp"
#include "gguf.h"
#include "tasks/yolo/backend.hpp"
#include "tasks/yolo/yolo_clip_text_graph.hpp"
#include "tasks/yolo/yolo_graph.hpp"
#include "tasks/yolo/yolo_image.hpp"
#include "tasks/yolo/yolo_mclip_text_graph.hpp"
#include "tasks/yolo/yolo_mobileclip_graph.hpp"
#include "tasks/yolo/yolo_postprocess.hpp"

struct aicore_yolo_ctx {
    yolo::Session* engine = nullptr;
    std::string model_path;
    std::string device;
    int32_t threads = 0;
    std::string last_error;

    // Detection thresholds: the single configuration point for detect and
    // segment calls (defaults 0.25 / 0.7 / model max_det). Seeded from the
    // options struct at load; adjustable at runtime via
    // aicore_yolo_set_detect_thresholds without rebuilding the context.
    float conf_thres = 0.25f;
    float iou_thres = 0.7f;
    uint32_t top_k = 0;

    // Open-vocabulary state: the user class list (empty = checkpoint
    // vocabulary) that overrides the GGUF class names in every result, so
    // the vocabulary is part of the result (mirrors the upstream CLI
    // semantics where --classes re-derives the category table).
    std::vector<std::string> class_names_override;

    // Per-stage wall-clock timings of the most recent inference call
    // (mirrors the upstream ultralytics-ggml bench fields; surfaced through
    // aicore_yolo_last_timings for 1:1 latency comparisons).
    aicore_yolo_timings timings{};
    bool has_timings = false;

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
    std::vector<yolo::Detection> last_detections;
};

struct aicore_yolo_options {
    aicore::capi::CommonOptions common;
    float conf_thres = 0.25f;
    float iou_thres = 0.7f;
    uint32_t top_k = 0;
    // Mirrors the yolo::SessionOptions debug/tuning fields (complete bridge;
    // defaults equal the SessionOptions defaults).
    int log_level = 1;  // 0=DEBUG,1=INFO,2=WARN,3=ERROR
    int input_w = 0;    // 0: square imgsz from GGUF metadata
    int input_h = 0;
    bool keep_all_ops = false;
    bool profile_ops = false;
    bool profile_gaps = false;
    // Open-vocabulary knobs (YOLO-World / YOLOE).
    std::vector<std::string> classes;
    std::string text_model_path;
    // YOLOE visual prompts (SAVPE): original-image pixel boxes, four floats
    // per box; empty = text/vocabulary path. See
    // aicore_yolo_options_set_visual_prompts.
    std::vector<float> visual_boxes;
};

struct aicore_yolo_segment_result {
    std::vector<yolo::Detection> dets;
    std::vector<yolo::SegMask> masks;
    // Canvas-space mask data (absolute coordinates), also stored per-mask
    int canvas_w = 0;
    int canvas_h = 0;
    // Class-name table copied at result time (open-vocabulary override or
    // the model metadata), so aicore_yolo_seg_det_class_name stays valid for
    // the result's lifetime (the typed API has no ctx handle to query after
    // the call).
    std::vector<std::string> class_names;
};

struct aicore_yolo_pose_result {
    std::vector<yolo::PoseDetection> poses;
    int kpt_count = 0;  // keypoints per detection (17 for the COCO models)
    std::vector<std::string> class_names;
};

struct aicore_yolo_obb_result {
    std::vector<yolo::OBBDetection> boxes;
    std::vector<std::string> class_names;
};

struct aicore_yolo_semantic_result {
    std::vector<uint8_t> class_map;  // full source-image resolution
    int width = 0;
    int height = 0;
    int num_classes = 0;
    std::vector<std::string> class_names;
};

struct aicore_yolo_classify_result {
    std::vector<float> probs;  // softmax over all classes
    std::vector<std::string> class_names;
};

using aicore::capi::dup_cstr;
using aicore::capi::json_escape;

namespace {

// Process-level cache for open-vocabulary text encodings. Text encoding is
// a pure function of (text-model GGUF, class list), and the cached embedding
// also sidesteps an in-process re-encoding defect (the second
// clip/mobileclip session in one process currently produces a different
// embedding; under investigation — see test_yolo_world_optrace). Key:
// text-model path + '\x1f' + joined class names.
struct TextEmbedCacheEntry {
    std::vector<float> embed;
    int nc = 0;
};
std::mutex g_text_embed_cache_mutex;
std::map<std::string, TextEmbedCacheEntry> g_text_embed_cache;
std::once_flag g_text_embed_cleanup_registration;
constexpr size_t kMaxTextEmbedCacheEntries = 32;

void clear_text_embed_cache() {
    std::lock_guard<std::mutex> lock(g_text_embed_cache_mutex);
    g_text_embed_cache.clear();
}

// Probe a text-encoder GGUF for the mclip architecture and its projection
// target space (KVs "mclip.arch" / "mclip.target_space", written by
// core/AICore/src/tasks/yolo/tools/convert_mclip_gguf.py). Header-only read, no
// tensor data mapped. Returns "" when the file is not an mclip bridge;
// otherwise the target space id ("clipb32" = OpenAI CLIP ViT-B/32 for World,
// "mobileclip2b" = MobileCLIP2-B for YOLOE). Older bridge files without the key
// default to "clipb32".
std::string text_gguf_target_space(const std::string& path) {
    gguf_init_params ip{};
    ip.no_alloc = true;
    ip.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) return "";
    const int arch = gguf_find_key(g, "mclip.arch");
    if (arch < 0) {
        gguf_free(g);
        return "";
    }
    std::string space = "clipb32";
    const int kid = gguf_find_key(g, "mclip.target_space");
    if (kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_STRING) {
        const char* v = gguf_get_val_str(g, kid);
        if (v && v[0]) space = v;
    }
    gguf_free(g);
    return space;
}

// Encode the open-vocabulary class list through the matching text tower
// (MobileCLIP for YOLOE — its GGUF declares yolo.text_model — and CLIP for
// YOLO-World) and queue the embedding as the session's text input. The
// text sessions are torn down after encoding: the embedding is the only
// state the detector graph consumes.
bool encode_open_vocab_classes(aicore_yolo_ctx* ctx,
                               const aicore_yolo_options* opts) {
    std::call_once(g_text_embed_cleanup_registration, [] {
        aicore::runtime::register_cleanup(clear_text_embed_cache);
    });
    yolo::Session* s = ctx->engine;
    if (opts->text_model_path.empty()) {
        ctx->last_error =
                "text-conditioned model requires a text encoder GGUF "
                "(aicore_yolo_options_set_text_model)";
        return false;
    }
    const int nc = s->world_nc;
    const size_t dim = 512;  // CLIP / MobileCLIP / M-CLIP embedding dim

    // Cache lookup (pure-function memoization; see the comment above).
    std::string cache_key = opts->text_model_path + '\x1f';
    for (const std::string& c : ctx->class_names_override) {
        cache_key += c;
        cache_key += '\x1f';
    }
    {
        std::lock_guard<std::mutex> lock(g_text_embed_cache_mutex);
        auto it = g_text_embed_cache.find(cache_key);
        if (it != g_text_embed_cache.end() && it->second.nc == nc) {
            if (!yolo::session_set_text(s, it->second.embed.data())) {
                ctx->last_error = "failed to queue the class text embedding";
                return false;
            }
            return true;
        }
    }

    std::vector<float> embed((size_t)nc * dim, 0.0f);
    const bool yoloe = !s->model.meta.text_model.empty();
    const std::string mclip_target =
            text_gguf_target_space(opts->text_model_path);
    const bool mclip = !mclip_target.empty();
    if (mclip) {
        // Space gate: the bridge GGUF declares which text space its
        // projection lands in ("clipb32" for World, "mobileclip2b" for
        // YOLOE) — the detector head consumes that space and nothing else.
        const bool compatible = yoloe ? (mclip_target == "mobileclip2b")
                                      : (mclip_target == "clipb32");
        if (!compatible) {
            ctx->last_error =
                    yoloe ? "YOLOE detectors require a text tower projecting "
                            "into the MobileCLIP2-B space (mclip.target_space="
                            "mobileclip2b)"
                          : "YOLO-World detectors require a text tower "
                            "projecting into the CLIP ViT-B/32 space "
                            "(mclip.target_space=clipb32)";
            return false;
        }
        // Multilingual bridge: DistilBERT tower projected into the detector's
        // text space — the head consumes it unchanged.
        mclip::TextSession* ms =
                mclip::text_create_session(opts->text_model_path, ctx->threads);
        if (ms == nullptr) {
            ctx->last_error = "failed to load M-CLIP text model: " +
                              opts->text_model_path;
            return false;
        }
        for (int i = 0; i < nc; ++i) {
            if (!mclip::text_encode_string(ms,
                                           ctx->class_names_override[i].c_str(),
                                           embed.data() + (size_t)i * dim)) {
                ctx->last_error = "failed to encode class '" +
                                  ctx->class_names_override[i] + "'";
                mclip::text_free_session(ms);
                return false;
            }
        }
        mclip::text_free_session(ms);
    } else if (yoloe) {
        mobileclip::Session* ms =
                mobileclip::create_session(opts->text_model_path, ctx->threads);
        if (ms == nullptr) {
            ctx->last_error = "failed to load MobileCLIP text model: " +
                              opts->text_model_path;
            return false;
        }
        for (int i = 0; i < nc; ++i) {
            if (!mobileclip::encode_string(ms,
                                           ctx->class_names_override[i].c_str(),
                                           embed.data() + (size_t)i * dim)) {
                ctx->last_error = "failed to encode class '" +
                                  ctx->class_names_override[i] + "'";
                mobileclip::free_session(ms);
                return false;
            }
        }
        mobileclip::free_session(ms);
    } else {
        clip::TextSession* cs =
                clip::text_create_session(opts->text_model_path, ctx->threads);
        if (cs == nullptr) {
            ctx->last_error =
                    "failed to load CLIP text model: " + opts->text_model_path;
            return false;
        }
        for (int i = 0; i < nc; ++i) {
            if (!clip::text_encode_string(cs,
                                          ctx->class_names_override[i].c_str(),
                                          embed.data() + (size_t)i * dim)) {
                ctx->last_error = "failed to encode class '" +
                                  ctx->class_names_override[i] + "'";
                clip::text_free_session(cs);
                return false;
            }
        }
        clip::text_free_session(cs);
    }
    if (!yolo::session_set_text(s, embed.data())) {
        ctx->last_error = "failed to queue the class text embedding";
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g_text_embed_cache_mutex);
        TextEmbedCacheEntry entry;
        entry.embed = embed;
        entry.nc = nc;
        if (g_text_embed_cache.size() >= kMaxTextEmbedCacheEntries &&
            g_text_embed_cache.find(cache_key) == g_text_embed_cache.end()) {
            g_text_embed_cache.erase(g_text_embed_cache.begin());
        }
        g_text_embed_cache[cache_key] = std::move(entry);
    }
    return true;
}

// Load an image file and hand the tightly-packed RGB buffer to f(rgb, w, h).
// The buffer is freed before returning; f must not keep a pointer to it.
template <class F>
auto with_path_rgb(const char* image_path,
                   aicore_yolo_ctx* ctx,
                   F&& f) -> decltype(f(nullptr, 0, 0)) {
    QImage img(QString::fromUtf8(image_path));
    if (img.isNull()) {
        ctx->last_error = std::string("failed to load image: ") + image_path;
        return decltype(f(nullptr, 0, 0))();
    }
    aicore::capi::PackedRgb packed = aicore::capi::qimage_to_packed_rgb(img);
    if (packed.data == nullptr) {
        ctx->last_error = "out of memory decoding image";
        return decltype(f(nullptr, 0, 0))();
    }
    auto result = f(packed.data, packed.width, packed.height);
    std::free(packed.data);
    return result;
}

// Effective class-name table: the open-vocabulary override (user class
// list) wins over the GGUF metadata; result structs copy it so accessors
// stay valid for the result lifetime.
const std::vector<std::string>& effective_class_names(aicore_yolo_ctx* ctx) {
    return !ctx->class_names_override.empty()
                   ? ctx->class_names_override
                   : ctx->engine->model.meta.class_names;
}

}  // namespace

AICORE_CAPI int aicore_yolo_abi_version(void) { return 4; }

AICORE_CAPI aicore_yolo_options* aicore_yolo_options_new(void) {
    return new (std::nothrow) aicore_yolo_options();
}

AICORE_CAPI void aicore_yolo_options_free(aicore_yolo_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_yolo_options_set_device(aicore_yolo_options* opts,
                                                const char* device) {
    if (opts != nullptr) aicore::capi::set_device(opts->common, device);
}

AICORE_CAPI void aicore_yolo_options_set_threads(aicore_yolo_options* opts,
                                                 int n_threads) {
    if (opts != nullptr) aicore::capi::set_threads(opts->common, n_threads);
}

AICORE_CAPI void aicore_yolo_options_set_conf_thres(aicore_yolo_options* opts,
                                                    float conf_thres) {
    if (opts != nullptr && conf_thres > 0.0f && conf_thres < 1.0f)
        opts->conf_thres = conf_thres;
}

AICORE_CAPI void aicore_yolo_options_set_iou_thres(aicore_yolo_options* opts,
                                                   float iou_thres) {
    if (opts != nullptr && iou_thres > 0.0f && iou_thres < 1.0f)
        opts->iou_thres = iou_thres;
}

AICORE_CAPI void aicore_yolo_options_set_top_k(aicore_yolo_options* opts,
                                               uint32_t top_k) {
    if (opts != nullptr) opts->top_k = top_k;
}

AICORE_CAPI void aicore_yolo_options_set_log_level(aicore_yolo_options* opts,
                                                   int log_level) {
    if (opts == nullptr) return;
    if (log_level < 0 || log_level > 3) return;  // invalid keeps current
    opts->log_level = log_level;
}

AICORE_CAPI void aicore_yolo_options_set_input_size(aicore_yolo_options* opts,
                                                    int width,
                                                    int height) {
    if (opts == nullptr) return;
    if (width <= 0 || height <= 0) {  // 0/invalid clears to model default
        opts->input_w = 0;
        opts->input_h = 0;
        return;
    }
    opts->input_w = width;
    opts->input_h = height;
}

AICORE_CAPI void aicore_yolo_options_set_keep_all_ops(aicore_yolo_options* opts,
                                                      int enabled) {
    if (opts != nullptr && enabled >= 0) opts->keep_all_ops = enabled != 0;
}

AICORE_CAPI void aicore_yolo_options_set_profile_ops(aicore_yolo_options* opts,
                                                     int enabled) {
    if (opts != nullptr && enabled >= 0) opts->profile_ops = enabled != 0;
}

AICORE_CAPI void aicore_yolo_options_set_profile_gaps(aicore_yolo_options* opts,
                                                      int enabled) {
    if (opts != nullptr && enabled >= 0) opts->profile_gaps = enabled != 0;
}

AICORE_CAPI void aicore_yolo_options_set_classes(aicore_yolo_options* opts,
                                                 const char* const* classes,
                                                 int32_t count) {
    if (opts == nullptr) return;
    opts->classes.clear();
    if (classes == nullptr || count <= 0) return;
    opts->classes.reserve((size_t)count);
    for (int32_t i = 0; i < count; ++i) {
        // An empty string is a real class row (background prompt), so no
        // filtering here — only NULL entries are dropped.
        opts->classes.emplace_back(classes[i] != nullptr ? classes[i] : "");
    }
}

AICORE_CAPI void aicore_yolo_options_set_text_model(
        aicore_yolo_options* opts, const char* text_model_path) {
    if (opts == nullptr) return;
    opts->text_model_path = text_model_path != nullptr ? text_model_path : "";
}

// Sanity cap: every prompt adds a graph branch and a mask plane; 64 boxes
// is far past any interactive labeling session.
constexpr int32_t kMaxVisualPrompts = 64;

AICORE_CAPI void aicore_yolo_options_set_visual_prompts(
        aicore_yolo_options* opts, const float* boxes_xyxy, int32_t count) {
    if (opts == nullptr) return;
    opts->visual_boxes.clear();
    if (boxes_xyxy == nullptr || count <= 0) return;
    const int32_t n = std::min(count, kMaxVisualPrompts);
    for (int32_t i = 0; i < n; ++i) {
        const float x1 = boxes_xyxy[i * 4 + 0];
        const float y1 = boxes_xyxy[i * 4 + 1];
        const float x2 = boxes_xyxy[i * 4 + 2];
        const float y2 = boxes_xyxy[i * 4 + 3];
        // Degenerate boxes carry no usable appearance; drop them instead of
        // handing the encoder an empty mask (which would yield a zero
        // embedding that matches nothing).
        if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) ||
            !std::isfinite(y2) || x2 <= x1 || y2 <= y1) {
            continue;
        }
        opts->visual_boxes.insert(opts->visual_boxes.end(), {x1, y1, x2, y2});
    }
}

AICORE_CAPI int32_t
aicore_yolo_options_get_visual_prompt_count(const aicore_yolo_options* opts) {
    return opts != nullptr ? (int32_t)(opts->visual_boxes.size() / 4) : 0;
}

AICORE_CAPI int aicore_yolo_gguf_has_savpe(const char* gguf_path) {
    if (gguf_path == nullptr) return 0;
    gguf_init_params ip{};  // header only, no tensor mapping
    ip.no_alloc = true;
    ip.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(gguf_path, ip);
    if (!g) return 0;
    const int64_t id = gguf_find_key(g, "yolo.savpe");
    const int has = id >= 0 ? (gguf_get_val_u32(g, id) != 0) : 0;
    gguf_free(g);
    return has;
}

AICORE_CAPI int aicore_yolo_context_has_visual_prompts(
        const aicore_yolo_ctx* ctx) {
    return ctx != nullptr && ctx->engine != nullptr &&
                           ctx->engine->visual_mode()
                   ? 1
                   : 0;
}

AICORE_CAPI float aicore_yolo_options_get_conf_thres(
        const aicore_yolo_options* opts) {
    return opts != nullptr ? opts->conf_thres : 0.25f;
}

AICORE_CAPI float aicore_yolo_options_get_iou_thres(
        const aicore_yolo_options* opts) {
    return opts != nullptr ? opts->iou_thres : 0.7f;
}

AICORE_CAPI aicore_yolo_ctx* aicore_yolo_load_opts(
        const char* gguf_path, const aicore_yolo_options* opts) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_yolo_ctx();
    if (ctx == nullptr) return nullptr;

    ctx->model_path = gguf_path;
    ctx->device = opts != nullptr ? opts->common.device : "auto";
    ctx->threads = opts != nullptr ? opts->common.threads : 0;
    if (opts != nullptr) {
        ctx->conf_thres = opts->conf_thres;
        ctx->iou_thres = opts->iou_thres;
        ctx->top_k = opts->top_k;
    }

    try {
        yolo::SessionOptions sopts;
        if (opts != nullptr) {
            sopts.threads = opts->common.threads;
            sopts.input_w = opts->input_w;
            sopts.input_h = opts->input_h;
            sopts.log_level = opts->log_level;
            sopts.keep_all_ops = opts->keep_all_ops;
            sopts.profile_ops = opts->profile_ops;
            sopts.profile_gaps = opts->profile_gaps;
            // The class count fixes the graph text-input shape, so it must
            // ride into the session options at creation.
            if (!opts->classes.empty())
                sopts.world_nc = (int)opts->classes.size();
            // Visual prompts take precedence over a class list: the head's
            // cls_pe comes from the savpe encoder, nc = prompt count.
            if (!opts->visual_boxes.empty()) {
                // Pre-flight the SAVPE KV only when the file actually opens:
                // a missing/unresolvable path used to fail the header probe
                // and masquerade as "ships none" (observed from the GUI,
                // which passes a not-yet-downloaded cache-dir path here).
                // Let the loader report the real open error instead.
                std::error_code probe_ec;
                const bool path_exists =
                        std::filesystem::exists(gguf_path, probe_ec);
                if (path_exists && !aicore_yolo_gguf_has_savpe(gguf_path)) {
                    ctx->last_error =
                            "visual prompts need a YOLOE GGUF converted with "
                            "savpe weights (yolo.savpe=1); this checkpoint "
                            "ships none";
                    return ctx;
                }
                sopts.visual_count = (int)(opts->visual_boxes.size() / 4);
                sopts.visual_boxes = opts->visual_boxes;
            }
        } else {
            sopts.threads = ctx->threads;
        }
        // Text-conditioned families run on every backend: the CUDA open-
        // vocabulary detection-count divergence was root-caused to the F32
        // TF32 MMF GEMM path in ggml-cuda (upstream ultralytics-ggml routes
        // F32 through cuBLAS — integrated via
        // patches/upstream_accuracy/0001-world-f32-gemm-tf32-route-...),
        // verified by test_yolo_capi_parity on real World/YOLOE GGUFs.
        ctx->engine = yolo::create_session(gguf_path, ctx->device, sopts);
        if (ctx->engine == nullptr) {
            ctx->last_error = "failed to load YOLO GGUF";
        }
        // Open-vocabulary setup: user classes override the checkpoint
        // vocabulary in every result, then get encoded through the matching
        // text tower. The override is the text-encoding contract, so it only
        // applies to text-conditioned models: a prompt-free YOLOE checkpoint
        // rejects set_classes outright (upstream AssertionError) and matches
        // its built-in LRPC vocabulary. Storing the caller's list for such a
        // model used to REPLACE the GGUF's stored class-name table, so every
        // detection cid left the tiny override and the plugin label degraded
        // to "class <id>" (the class-name accessor returns nullptr there).
        if (ctx->engine != nullptr && opts != nullptr &&
            !opts->classes.empty() && ctx->engine->model.has_text_input &&
            sopts.visual_count == 0) {
            ctx->class_names_override = opts->classes;
            if (!encode_open_vocab_classes(ctx, opts)) {
                yolo::free_session(ctx->engine);
                ctx->engine = nullptr;
            }
        }
        // Visual-prompt sessions label detections object0..object{Q-1}
        // (official semantics: visual prompts group examples, they do not
        // carry names), so the typed result accessors resolve every cid.
        if (ctx->engine != nullptr && sopts.visual_count > 0) {
            ctx->class_names_override.reserve((size_t)sopts.visual_count);
            for (int i = 0; i < sopts.visual_count; ++i) {
                ctx->class_names_override.push_back("object" +
                                                    std::to_string(i));
            }
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

AICORE_CAPI void aicore_yolo_free_buffer(void* p) { std::free(p); }

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
    aicore::capi::PackedRgb packed = aicore::capi::qimage_to_packed_rgb(img);
    if (packed.data == nullptr) return -1;
    *out_rgb = packed.data;
    *out_width = packed.width;
    *out_height = packed.height;
    return 0;
}

namespace {

yolo::Image packed_rgb_image(const uint8_t* rgb,
                             int32_t width,
                             int32_t height) {
    return yolo::Image{width, height, rgb,
                       width > 0 ? static_cast<size_t>(width) * 3 : 0, 3};
}

bool image_from_view(const aicore_image_view* view, yolo::Image* out) {
    if (view == nullptr || out == nullptr || view->data == nullptr ||
        view->width <= 0 || view->height <= 0) {
        return false;
    }
    int channels = 0;
    bool bgr = false;
    switch (view->format) {
        case AICORE_IMAGE_RGB8:
            channels = 3;
            break;
        case AICORE_IMAGE_RGBA8:
            channels = 4;
            break;
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            break;
        case AICORE_IMAGE_BGR8:
            channels = 3;
            bgr = true;
            break;
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            bgr = true;
            break;
        default:
            return false;
    }
    const size_t min_stride = static_cast<size_t>(view->width) * channels;
    if (view->row_stride_bytes < min_stride) return false;
    *out = yolo::Image{view->width, view->height,
                       view->data,  view->row_stride_bytes,
                       channels,    bgr};
    return true;
}

// Shared detect core: letterbox, inference, postprocess, JSON envelope.
// Everything may allocate; an uncaught bad_alloc would cross the extern "C"
// boundary and the Qt event loop (queued worker slot) and terminate the
// process with SIGABRT — hence the catch fencing.
char* run_detect(aicore_yolo_ctx* ctx,
                 const yolo::Image& image,
                 int* out_rc,
                 bool serialize_json = true) {
    *out_rc = -1;
    yolo::Session* s = ctx->engine;
    const int32_t width = image.w;
    const int32_t height = image.h;
    if (s == nullptr || image.rgb == nullptr || width <= 0 || height <= 0 ||
        (image.channels != 1 && image.channels != 3 && image.channels != 4)) {
        return nullptr;
    }
    if (s->model.meta.task != "detect") {
        ctx->last_error =
                "model is not a detect model (task=" + s->model.meta.task + ")";
        return nullptr;
    }
    ctx->depth.valid = false;

    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(image, s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        if (s->visual_mode() && !yolo::session_prepare_visual_masks(s, info)) {
            ctx->last_error = "visual prompt rasterization failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
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
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        yolo::PostprocConfig cfg;
        cfg.conf_thres = ctx->conf_thres;
        cfg.iou_thres = ctx->iou_thres;
        cfg.max_det = ctx->top_k > 0 ? (int)ctx->top_k : s->model.meta.max_det;
        std::vector<yolo::Detection> dets =
                yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(),
                                  s->anchor_strides.data(), cfg);
        yolo::unscale_boxes(dets, info, width, height);
        ctx->last_detections = dets;
        const double postprocess_ms = yolo::ms_since(t0);

        if (!serialize_json) {
            ctx->timings = aicore_yolo_timings{preprocess_ms, inference_ms,
                                               postprocess_ms, 0.0,
                                               yolo::ms_since(t_e2e)};
            ctx->has_timings = true;
            *out_rc = 0;
            return nullptr;
        }
        t0 = yolo::Clock::now();
        const auto& names = effective_class_names(ctx);
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
        char* json = dup_cstr(o.str());
        const double json_ms = yolo::ms_since(t0);
        const double e2e_ms = yolo::ms_since(t_e2e);
        ctx->timings = aicore_yolo_timings{preprocess_ms, inference_ms,
                                           postprocess_ms, json_ms, e2e_ms};
        ctx->has_timings = true;
        *out_rc = 0;
        return json;
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
                 const yolo::Image& image,
                 int32_t* out_width,
                 int32_t* out_height) {
    yolo::Session* s = ctx->engine;
    const int32_t width = image.w;
    const int32_t height = image.h;
    if (s == nullptr || image.rgb == nullptr || width <= 0 || height <= 0 ||
        (image.channels != 1 && image.channels != 3 && image.channels != 4)) {
        return nullptr;
    }
    if (s->model.meta.task != "depth") {
        ctx->last_error =
                "model is not a depth model (task=" + s->model.meta.task + ")";
        return nullptr;
    }

    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(image, s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
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
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
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
        const double postprocess_ms = yolo::ms_since(t0);
        ctx->timings =
                aicore_yolo_timings{preprocess_ms, inference_ms, postprocess_ms,
                                    0.0, yolo::ms_since(t_e2e)};
        ctx->has_timings = true;
        return out;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in depth post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO depth error: ") + e.what();
        return nullptr;
    }
}

// Shared pose core: letterbox, inference, box + keypoint decode.
aicore_yolo_pose_result* run_pose(aicore_yolo_ctx* ctx,
                                  const yolo::Image& image) {
    yolo::Session* s = ctx->engine;
    const int32_t width = image.w;
    const int32_t height = image.h;
    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(image, s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
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
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        yolo::PostprocConfig cfg;
        cfg.conf_thres = ctx->conf_thres;
        cfg.iou_thres = ctx->iou_thres;
        cfg.max_det = ctx->top_k > 0 ? (int)ctx->top_k : s->model.meta.max_det;
        std::vector<yolo::PoseDetection> poses = yolo::postprocess_pose(
                raw, no, na, s->model.meta, s->anchors.data(),
                s->anchor_strides.data(), cfg);
        yolo::unscale_pose(poses, info);
        const double postprocess_ms = yolo::ms_since(t0);

        auto* res = new (std::nothrow) aicore_yolo_pose_result();
        if (res == nullptr) {
            ctx->last_error = "YOLO out of memory for pose result";
            return nullptr;
        }
        res->poses = std::move(poses);
        res->kpt_count = s->model.meta.nk > 0
                                 ? s->model.meta.nk / s->model.meta.kpt_ndim
                                 : 0;
        res->class_names = effective_class_names(ctx);
        ctx->timings =
                aicore_yolo_timings{preprocess_ms, inference_ms, postprocess_ms,
                                    0.0, yolo::ms_since(t_e2e)};
        ctx->has_timings = true;
        return res;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in pose post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO pose error: ") + e.what();
        return nullptr;
    }
}

// Shared OBB core: letterbox, inference, dist2rbox decode + unscale.
aicore_yolo_obb_result* run_obb(aicore_yolo_ctx* ctx,
                                const yolo::Image& image) {
    yolo::Session* s = ctx->engine;
    const int32_t width = image.w;
    const int32_t height = image.h;
    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(image, s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
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
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        yolo::PostprocConfig cfg;
        cfg.conf_thres = ctx->conf_thres;
        cfg.iou_thres = ctx->iou_thres;
        cfg.max_det = ctx->top_k > 0 ? (int)ctx->top_k : s->model.meta.max_det;
        std::vector<yolo::OBBDetection> boxes = yolo::postprocess_obb(
                raw, no, na, s->model.meta, s->anchors.data(),
                s->anchor_strides.data(), cfg);
        yolo::unscale_obb(boxes, info);
        const double postprocess_ms = yolo::ms_since(t0);

        auto* res = new (std::nothrow) aicore_yolo_obb_result();
        if (res == nullptr) {
            ctx->last_error = "YOLO out of memory for OBB result";
            return nullptr;
        }
        res->boxes = std::move(boxes);
        res->class_names = effective_class_names(ctx);
        ctx->timings =
                aicore_yolo_timings{preprocess_ms, inference_ms, postprocess_ms,
                                    0.0, yolo::ms_since(t_e2e)};
        ctx->has_timings = true;
        return res;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in OBB post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO OBB error: ") + e.what();
        return nullptr;
    }
}

// Shared semantic core: letterbox, inference, full-resolution logit restore,
// then argmax (the ordering is part of the Ultralytics semantic contract).
aicore_yolo_semantic_result* run_semantic(aicore_yolo_ctx* ctx,
                                          const yolo::Image& image) {
    yolo::Session* s = ctx->engine;
    const int32_t width = image.w;
    const int32_t height = image.h;
    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(image, s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        if (!yolo::session_run(s, canvas.data())) {
            ctx->last_error = "YOLO inference failed";
            return nullptr;
        }
        std::vector<float> logits;
        int nc = 0, gw = 0, gh = 0;
        if (!yolo::session_read_semantic(s, logits, nc, gw, gh)) {
            ctx->last_error = "semantic readback failed";
            return nullptr;
        }
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        std::vector<uint8_t> class_map = yolo::semantic_restore_logits(
                logits, nc, gw, gh, info.imgsz_w, info.imgsz_h, width, height);
        if ((int)class_map.size() != width * height) {
            ctx->last_error = "semantic logit restoration failed";
            return nullptr;
        }
        auto* res = new (std::nothrow) aicore_yolo_semantic_result();
        if (res == nullptr) {
            ctx->last_error = "YOLO out of memory for semantic result";
            return nullptr;
        }
        res->class_map = std::move(class_map);
        res->width = width;
        res->height = height;
        res->num_classes = nc;
        res->class_names = effective_class_names(ctx);
        const double postprocess_ms = yolo::ms_since(t0);
        ctx->timings =
                aicore_yolo_timings{preprocess_ms, inference_ms, postprocess_ms,
                                    0.0, yolo::ms_since(t_e2e)};
        ctx->has_timings = true;
        return res;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in semantic post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO semantic error: ") + e.what();
        return nullptr;
    }
}

// Shared classify core: checkpoint-baked resize+crop preprocessing,
// inference, softmax.
aicore_yolo_classify_result* run_classify(aicore_yolo_ctx* ctx,
                                          const yolo::Image& image) {
    yolo::Session* s = ctx->engine;
    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        // Classification uses the checkpoint-baked resize + center crop (no
        // letterbox), so the canvas is the fixed square imgsz and the graph
        // never rebuilds.
        std::vector<float> input;
        yolo::classify_preprocess(image, s->model.meta.imgsz, input);
        if (!yolo::session_ensure_canvas(s, s->model.meta.imgsz,
                                         s->model.meta.imgsz)) {
            ctx->last_error = "graph rebuild for the classify canvas failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        if (!yolo::session_run(s, input.data())) {
            ctx->last_error = "YOLO inference failed";
            return nullptr;
        }
        std::vector<float> logits;
        if (!yolo::session_read_logits(s, logits)) {
            ctx->last_error = "logits readback failed";
            return nullptr;
        }
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        auto* res = new (std::nothrow) aicore_yolo_classify_result();
        if (res == nullptr) {
            ctx->last_error = "YOLO out of memory for classify result";
            return nullptr;
        }
        res->probs = yolo::classify_softmax(logits);
        res->class_names = effective_class_names(ctx);
        const double postprocess_ms = yolo::ms_since(t0);
        ctx->timings =
                aicore_yolo_timings{preprocess_ms, inference_ms, postprocess_ms,
                                    0.0, yolo::ms_since(t_e2e)};
        ctx->has_timings = true;
        return res;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in classify post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO classify error: ") + e.what();
        return nullptr;
    }
}

}  // namespace

AICORE_CAPI char* aicore_yolo_detect_path_json(aicore_yolo_ctx* ctx,
                                               const char* image_path) {
    if (ctx == nullptr || ctx->engine == nullptr || image_path == nullptr) {
        return nullptr;
    }
    return with_path_rgb(
            image_path, ctx, [&](const uint8_t* rgb, int w, int h) {
                int rc = -1;
                return run_detect(ctx, packed_rgb_image(rgb, w, h), &rc);
            });
}

AICORE_CAPI char* aicore_yolo_detect_rgb_json(aicore_yolo_ctx* ctx,
                                              const uint8_t* rgb,
                                              int32_t width,
                                              int32_t height) {
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    int rc = -1;
    /* Borrow (no copy): the caller's buffer must stay alive for the whole
     * call, which the synchronous C API contract guarantees (preprocess
     * only reads from it). Saves a full-frame copy per detection call. */
    return run_detect(ctx, packed_rgb_image(rgb, width, height), &rc);
}

AICORE_CAPI int aicore_yolo_detect_rgb(aicore_yolo_ctx* ctx,
                                       const uint8_t* rgb,
                                       int32_t width,
                                       int32_t height) {
    if (ctx == nullptr || ctx->engine == nullptr) return -1;
    int rc = -1;
    (void)run_detect(ctx, packed_rgb_image(rgb, width, height), &rc, false);
    return rc;
}

AICORE_CAPI int aicore_yolo_detect_image(aicore_yolo_ctx* ctx,
                                         const aicore_image_view* image) {
    if (!ctx || !ctx->engine) return -1;
    yolo::Image input;
    if (!image_from_view(image, &input)) return -1;
    int rc = -1;
    (void)run_detect(ctx, input, &rc, false);
    return rc;
}

AICORE_CAPI int aicore_yolo_detection_count(const aicore_yolo_ctx* ctx) {
    return ctx != nullptr ? static_cast<int>(ctx->last_detections.size()) : -1;
}

AICORE_CAPI aicore_yolo_detection
aicore_yolo_detection_at(const aicore_yolo_ctx* ctx, int index) {
    aicore_yolo_detection out = {};
    if (ctx == nullptr || index < 0 ||
        static_cast<size_t>(index) >= ctx->last_detections.size())
        return out;
    const auto& d = ctx->last_detections[static_cast<size_t>(index)];
    out.x1 = d.x1;
    out.y1 = d.y1;
    out.x2 = d.x2;
    out.y2 = d.y2;
    out.score = d.score;
    out.class_id = d.class_id;
    return out;
}

/** Runtime threshold update without rebuilding the context (validated: out
 *  of range values keep the previous value). */
AICORE_CAPI void aicore_yolo_set_detect_thresholds(aicore_yolo_ctx* ctx,
                                                   float conf_thres,
                                                   float iou_thres,
                                                   uint32_t top_k) {
    if (ctx == nullptr) return;
    if (conf_thres > 0.0f && conf_thres < 1.0f) ctx->conf_thres = conf_thres;
    if (iou_thres > 0.0f && iou_thres < 1.0f) ctx->iou_thres = iou_thres;
    ctx->top_k = top_k;
}

/** Drop the host-side copies of the model weights (halves the host memory
 *  footprint; the device weight buffer is untouched, so inference keeps
 *  working). Reload on demand with aicore_yolo_ensure_host_weights. */
AICORE_CAPI int aicore_yolo_release_host_weights(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return -1;
    return yolo::session_release_host_weights(ctx->engine) ? 0 : -1;
}

/** Reload released host weight copies from the GGUF file (no-op when they
 *  are present). Returns 0 on success, -1 when the GGUF cannot be reopened. */
AICORE_CAPI int aicore_yolo_ensure_host_weights(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return -1;
    return yolo::session_ensure_host_weights(ctx->engine) ? 0 : -1;
}

AICORE_CAPI float* aicore_yolo_depth_path(aicore_yolo_ctx* ctx,
                                          const char* image_path,
                                          int32_t* out_width,
                                          int32_t* out_height) {
    if (ctx == nullptr || ctx->engine == nullptr || image_path == nullptr) {
        return nullptr;
    }
    return with_path_rgb(image_path, ctx,
                         [&](const uint8_t* rgb, int w, int h) {
                             return run_depth(ctx, packed_rgb_image(rgb, w, h),
                                              out_width, out_height);
                         });
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
    return run_depth(ctx, packed_rgb_image(rgb, width, height), out_width,
                     out_height);
}

AICORE_CAPI float* aicore_yolo_depth_image(aicore_yolo_ctx* ctx,
                                           const aicore_image_view* image,
                                           int32_t* out_width,
                                           int32_t* out_height) {
    if (out_width != nullptr) *out_width = 0;
    if (out_height != nullptr) *out_height = 0;
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    yolo::Image input;
    if (!image_from_view(image, &input)) return nullptr;
    return run_depth(ctx, input, out_width, out_height);
}

// ---- Segment result (typed API) ----

static aicore_yolo_segment_result* run_segment(aicore_yolo_ctx* ctx,
                                               const yolo::Image& image) {
    const int32_t width = image.w;
    const int32_t height = image.h;
    if (ctx == nullptr || ctx->engine == nullptr || image.rgb == nullptr ||
        width <= 0 || height <= 0 ||
        (image.channels != 1 && image.channels != 3 && image.channels != 4)) {
        return nullptr;
    }
    yolo::Session* s = ctx->engine;
    if (s->model.meta.task != "segment") {
        ctx->last_error =
                "model is not a segment model (task=" + s->model.meta.task +
                ")";
        return nullptr;
    }

    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(image, s->model.meta.imgsz, info, canvas);

        // Canvas resize if needed
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild failed";
            return nullptr;
        }
        if (s->visual_mode() && !yolo::session_prepare_visual_masks(s, info)) {
            ctx->last_error = "visual prompt rasterization failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        if (!yolo::session_run(s, canvas.data())) {
            ctx->last_error = "YOLO segment inference failed";
            return nullptr;
        }
        if (const char* dump = aicore::debug::savpe_dump_path()) {
            if (s->savpe_out != nullptr) {
                std::vector<float> vpe((size_t)ggml_nelements(s->savpe_out));
                ggml_backend_tensor_get(s->savpe_out, vpe.data(), 0,
                                        vpe.size() * sizeof(float));
                FILE* f = std::fopen(dump, "wb");
                if (f != nullptr) {
                    std::fwrite(vpe.data(), sizeof(float), vpe.size(), f);
                    std::fclose(f);
                }
            }
            std::vector<std::pair<ggml_tensor*, std::string>> nodes = {
                    {s->savpe_x, "_x.bin"}, {s->savpe_y, "_y.bin"}};
            for (int l = 0; l < 3; l++) {
                nodes.push_back({s->savpe_fpn_dbg[l],
                                 "_fpn" + std::to_string(l) + ".bin"});
                nodes.push_back({s->savpe_cv2_dbg[l],
                                 "_cv2" + std::to_string(l) + ".bin"});
            }
            for (auto& [node, suffix] : nodes) {
                if (node == nullptr) continue;
                std::vector<float> d((size_t)ggml_nelements(node));
                ggml_backend_tensor_get(node, d.data(), 0,
                                        d.size() * sizeof(float));
                std::string path = std::string(dump) + suffix;
                FILE* f2 = std::fopen(path.c_str(), "wb");
                if (f2 != nullptr) {
                    std::fwrite(d.data(), sizeof(float), d.size(), f2);
                    std::fclose(f2);
                }
            }
        }

        // Debug: dump the level-0 embed op output (savpe/contrastive input)
        // when op tracing is on — pairs with the upstream YOLO_EMB_DUMP.
        if (const char* dump = aicore::debug::savpe_dump_path();
            dump != nullptr && s->opts.keep_all_ops && s->model.has_savpe &&
            s->model.detect_op_index >= 0 &&
            s->model.ops[s->model.detect_op_index].inputs.size() > 1) {
            const int emb_op = s->model.ops[s->model.detect_op_index].inputs[1];
            if (emb_op >= 0 && emb_op < (int)s->op_values.size() &&
                s->op_values[emb_op] != nullptr) {
                std::string ep = std::string(dump) + "_emb0.bin";
                ggml_tensor* t = s->op_values[emb_op];
                if (t->type == GGML_TYPE_F32) {
                    std::vector<float> d((size_t)ggml_nelements(t));
                    ggml_backend_tensor_get(t, d.data(), 0,
                                            d.size() * sizeof(float));
                    FILE* fe2 = std::fopen(ep.c_str(), "wb");
                    if (fe2 != nullptr) {
                        std::fwrite(d.data(), sizeof(float), d.size(), fe2);
                        std::fclose(fe2);
                    }
                }
            }
        }

        // Read detect output
        std::vector<float> raw;
        int no = 0, na = 0;
        if (!yolo::session_read_output(s, raw, no, na)) {
            ctx->last_error = "output readback failed";
            return nullptr;
        }
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        // Read proto output — counted as postprocess, mirroring the upstream
        // bench (proto readback + mask composition belong to post_ms).
        std::vector<float> proto;
        int nm = 0, proto_w = 0, proto_h = 0;
        if (!yolo::session_read_proto(s, proto, nm, proto_w, proto_h)) {
            ctx->last_error = "proto readback failed";
            return nullptr;
        }

        // Postprocess
        yolo::PostprocConfig cfg;
        cfg.conf_thres = ctx->conf_thres;
        cfg.iou_thres = ctx->iou_thres;
        cfg.max_det = ctx->top_k > 0 ? (int)ctx->top_k : s->model.meta.max_det;
        std::vector<yolo::Detection> dets =
                yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(),
                                  s->anchor_strides.data(), cfg);

        // Compose masks (before unscale_boxes — masks are in canvas coords)
        std::vector<yolo::SegMask> masks = yolo::compose_masks(
                dets, raw, na, s->model.meta, proto, proto_w, proto_h,
                info.imgsz_w, info.imgsz_h);

        // Unscale boxes to original image coordinates (clipped to the
        // source image — upstream clip_boxes).
        yolo::unscale_boxes(dets, info, width, height);

        // Masks follow the boxes into the original image space: the canvas
        // windows compose_masks produced would overlay the wrong region on
        // the source image (canvas dims differ from source dims, and the
        // window origins are lost across the C API). Full-size source masks
        // make the typed result directly drawable at 1:1 with the boxes.
        yolo::unscale_masks(masks, info, width, height);

        auto* res = new (std::nothrow) aicore_yolo_segment_result();
        if (!res) {
            ctx->last_error = "YOLO out of memory for segment result";
            return nullptr;
        }
        res->dets = std::move(dets);
        res->masks = std::move(masks);
        res->canvas_w = info.imgsz_w;
        res->canvas_h = info.imgsz_h;
        // Copy the model's class table (open-vocabulary override or model
        // metadata) so seg_det_class_name can serve names for the whole
        // result lifetime (no ctx dependency).
        res->class_names = effective_class_names(ctx);
        const double postprocess_ms = yolo::ms_since(t0);
        ctx->timings =
                aicore_yolo_timings{preprocess_ms, inference_ms, postprocess_ms,
                                    0.0, yolo::ms_since(t_e2e)};
        ctx->has_timings = true;
        return res;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in segment post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO segment error: ") + e.what();
        return nullptr;
    }
}

AICORE_CAPI aicore_yolo_segment_result* aicore_yolo_seg_rgb(
        aicore_yolo_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height) {
    return run_segment(ctx, packed_rgb_image(rgb, width, height));
}

AICORE_CAPI aicore_yolo_segment_result* aicore_yolo_seg_image(
        aicore_yolo_ctx* ctx, const aicore_image_view* image) {
    yolo::Image input;
    return image_from_view(image, &input) ? run_segment(ctx, input) : nullptr;
}

AICORE_CAPI int aicore_yolo_seg_det_count(
        const aicore_yolo_segment_result* res) {
    return res != nullptr ? (int)res->dets.size() : 0;
}

AICORE_CAPI aicore_yolo_detection
aicore_yolo_seg_det_at(const aicore_yolo_segment_result* res, int index) {
    aicore_yolo_detection det = {};
    if (res != nullptr && index >= 0 && index < (int)res->dets.size()) {
        const auto& d = res->dets[index];
        det.x1 = d.x1;
        det.y1 = d.y1;
        det.x2 = d.x2;
        det.y2 = d.y2;
        det.score = d.score;
        det.class_id = d.class_id;
    }
    return det;
}

AICORE_CAPI const char* aicore_yolo_seg_det_class_name(
        const aicore_yolo_segment_result* res, int index) {
    if (res == nullptr || index < 0 || index >= (int)res->dets.size()) {
        return nullptr;
    }
    const int cid = res->dets[index].class_id;
    if (cid < 0 || cid >= (int)res->class_names.size()) {
        return nullptr;
    }
    const std::string& name = res->class_names[cid];
    return name.empty() ? nullptr : name.c_str();
}

AICORE_CAPI aicore_yolo_plane_view
aicore_yolo_seg_mask_at(const aicore_yolo_segment_result* res, int index) {
    aicore_yolo_plane_view view = {};
    if (res != nullptr && index >= 0 && index < (int)res->masks.size()) {
        const auto& m = res->masks[index];
        view.data = m.bits.data();
        view.width = m.w;
        view.height = m.h;
        view.row_stride_bytes = (size_t)m.w;
    }
    return view;
}

AICORE_CAPI void aicore_yolo_seg_result_free(aicore_yolo_segment_result* res) {
    delete res;
}

// ---- Pose result (typed API) ----

AICORE_CAPI aicore_yolo_pose_result* aicore_yolo_pose_rgb(aicore_yolo_ctx* ctx,
                                                          const uint8_t* rgb,
                                                          int32_t width,
                                                          int32_t height) {
    if (ctx == nullptr || ctx->engine == nullptr || rgb == nullptr ||
        width <= 0 || height <= 0) {
        return nullptr;
    }
    yolo::Session* s = ctx->engine;
    if (s->model.meta.task != "pose") {
        ctx->last_error =
                "model is not a pose model (task=" + s->model.meta.task + ")";
        return nullptr;
    }
    return run_pose(ctx, packed_rgb_image(rgb, width, height));
}

AICORE_CAPI aicore_yolo_pose_result* aicore_yolo_pose_image(
        aicore_yolo_ctx* ctx, const aicore_image_view* image) {
    yolo::Image input;
    if (ctx == nullptr || ctx->engine == nullptr ||
        !image_from_view(image, &input)) {
        return nullptr;
    }
    if (ctx->engine->model.meta.task != "pose") {
        ctx->last_error = "model is not a pose model (task=" +
                          ctx->engine->model.meta.task + ")";
        return nullptr;
    }
    return run_pose(ctx, input);
}

AICORE_CAPI int aicore_yolo_pose_det_count(const aicore_yolo_pose_result* res) {
    return res != nullptr ? (int)res->poses.size() : 0;
}

AICORE_CAPI aicore_yolo_detection
aicore_yolo_pose_det_at(const aicore_yolo_pose_result* res, int index) {
    aicore_yolo_detection det = {};
    if (res != nullptr && index >= 0 && index < (int)res->poses.size()) {
        const auto& d = res->poses[index].det;
        det.x1 = d.x1;
        det.y1 = d.y1;
        det.x2 = d.x2;
        det.y2 = d.y2;
        det.score = d.score;
        det.class_id = d.class_id;
    }
    return det;
}

AICORE_CAPI int aicore_yolo_pose_kpt_count(const aicore_yolo_pose_result* res) {
    return res != nullptr ? res->kpt_count : 0;
}

AICORE_CAPI aicore_yolo_keypoint aicore_yolo_pose_kpt_at(
        const aicore_yolo_pose_result* res, int index, int kpt) {
    aicore_yolo_keypoint kp = {};
    if (res == nullptr || index < 0 || index >= (int)res->poses.size()) {
        return kp;
    }
    const int kpt_ndim =
            res->kpt_count > 0
                    ? (int)res->poses[index].kpts.size() / res->kpt_count
                    : 0;
    if (kpt < 0 || kpt >= res->kpt_count || kpt_ndim < 2 || kpt_ndim > 3) {
        return kp;
    }
    const auto& kpts = res->poses[index].kpts;
    kp.x = kpts[(size_t)kpt * kpt_ndim];
    kp.y = kpts[(size_t)kpt * kpt_ndim + 1];
    kp.visibility = kpt_ndim == 3 ? kpts[(size_t)kpt * kpt_ndim + 2] : 1.0f;
    return kp;
}

AICORE_CAPI const char* aicore_yolo_pose_det_class_name(
        const aicore_yolo_pose_result* res, int index) {
    if (res == nullptr || index < 0 || index >= (int)res->poses.size()) {
        return nullptr;
    }
    const int cid = res->poses[index].det.class_id;
    if (cid < 0 || cid >= (int)res->class_names.size()) return nullptr;
    const std::string& name = res->class_names[cid];
    return name.empty() ? nullptr : name.c_str();
}

AICORE_CAPI void aicore_yolo_pose_result_free(aicore_yolo_pose_result* res) {
    delete res;
}

// ---- OBB result (typed API) ----

AICORE_CAPI aicore_yolo_obb_result* aicore_yolo_obb_rgb(aicore_yolo_ctx* ctx,
                                                        const uint8_t* rgb,
                                                        int32_t width,
                                                        int32_t height) {
    if (ctx == nullptr || ctx->engine == nullptr || rgb == nullptr ||
        width <= 0 || height <= 0) {
        return nullptr;
    }
    yolo::Session* s = ctx->engine;
    if (s->model.meta.task != "obb") {
        ctx->last_error =
                "model is not an obb model (task=" + s->model.meta.task + ")";
        return nullptr;
    }
    return run_obb(ctx, packed_rgb_image(rgb, width, height));
}

AICORE_CAPI aicore_yolo_obb_result* aicore_yolo_obb_image(
        aicore_yolo_ctx* ctx, const aicore_image_view* image) {
    yolo::Image input;
    if (ctx == nullptr || ctx->engine == nullptr ||
        !image_from_view(image, &input)) {
        return nullptr;
    }
    if (ctx->engine->model.meta.task != "obb") {
        ctx->last_error = "model is not an obb model (task=" +
                          ctx->engine->model.meta.task + ")";
        return nullptr;
    }
    return run_obb(ctx, input);
}

AICORE_CAPI int aicore_yolo_obb_count(const aicore_yolo_obb_result* res) {
    return res != nullptr ? (int)res->boxes.size() : 0;
}

AICORE_CAPI aicore_yolo_obb_box
aicore_yolo_obb_at(const aicore_yolo_obb_result* res, int index) {
    aicore_yolo_obb_box box = {};
    if (res != nullptr && index >= 0 && index < (int)res->boxes.size()) {
        const auto& b = res->boxes[index];
        box.cx = b.cx;
        box.cy = b.cy;
        box.w = b.w;
        box.h = b.h;
        box.angle = b.angle;
        box.score = b.score;
        box.class_id = b.class_id;
    }
    return box;
}

AICORE_CAPI const char* aicore_yolo_obb_class_name(
        const aicore_yolo_obb_result* res, int index) {
    if (res == nullptr || index < 0 || index >= (int)res->boxes.size()) {
        return nullptr;
    }
    const int cid = res->boxes[index].class_id;
    if (cid < 0 || cid >= (int)res->class_names.size()) return nullptr;
    const std::string& name = res->class_names[cid];
    return name.empty() ? nullptr : name.c_str();
}

AICORE_CAPI void aicore_yolo_obb_result_free(aicore_yolo_obb_result* res) {
    delete res;
}

// ---- Semantic result (typed API) ----

AICORE_CAPI aicore_yolo_semantic_result* aicore_yolo_semantic_rgb(
        aicore_yolo_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height) {
    if (ctx == nullptr || ctx->engine == nullptr || rgb == nullptr ||
        width <= 0 || height <= 0) {
        return nullptr;
    }
    yolo::Session* s = ctx->engine;
    if (s->model.meta.task != "semantic") {
        ctx->last_error =
                "model is not a semantic model (task=" + s->model.meta.task +
                ")";
        return nullptr;
    }
    return run_semantic(ctx, packed_rgb_image(rgb, width, height));
}

AICORE_CAPI aicore_yolo_semantic_result* aicore_yolo_semantic_image(
        aicore_yolo_ctx* ctx, const aicore_image_view* image) {
    yolo::Image input;
    if (ctx == nullptr || ctx->engine == nullptr ||
        !image_from_view(image, &input)) {
        return nullptr;
    }
    if (ctx->engine->model.meta.task != "semantic") {
        ctx->last_error = "model is not a semantic model (task=" +
                          ctx->engine->model.meta.task + ")";
        return nullptr;
    }
    return run_semantic(ctx, input);
}

AICORE_CAPI aicore_yolo_plane_view
aicore_yolo_semantic_class_map(const aicore_yolo_semantic_result* res) {
    aicore_yolo_plane_view view = {};
    if (res != nullptr) {
        view.data = res->class_map.data();
        view.width = res->width;
        view.height = res->height;
        view.row_stride_bytes = (size_t)res->width;
    }
    return view;
}

AICORE_CAPI int aicore_yolo_semantic_num_classes(
        const aicore_yolo_semantic_result* res) {
    return res != nullptr ? res->num_classes : 0;
}

AICORE_CAPI const char* aicore_yolo_semantic_class_name(
        const aicore_yolo_semantic_result* res, int class_id) {
    if (res == nullptr || class_id < 0 ||
        class_id >= (int)res->class_names.size()) {
        return nullptr;
    }
    const std::string& name = res->class_names[class_id];
    return name.empty() ? nullptr : name.c_str();
}

AICORE_CAPI void aicore_yolo_semantic_result_free(
        aicore_yolo_semantic_result* res) {
    delete res;
}

// ---- Classify result (typed API) ----

AICORE_CAPI aicore_yolo_classify_result* aicore_yolo_classify_rgb(
        aicore_yolo_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height) {
    if (ctx == nullptr || ctx->engine == nullptr || rgb == nullptr ||
        width <= 0 || height <= 0) {
        return nullptr;
    }
    yolo::Session* s = ctx->engine;
    if (s->model.meta.task != "classify") {
        ctx->last_error =
                "model is not a classify model (task=" + s->model.meta.task +
                ")";
        return nullptr;
    }
    return run_classify(ctx, packed_rgb_image(rgb, width, height));
}

AICORE_CAPI aicore_yolo_classify_result* aicore_yolo_classify_image(
        aicore_yolo_ctx* ctx, const aicore_image_view* image) {
    yolo::Image input;
    if (ctx == nullptr || ctx->engine == nullptr ||
        !image_from_view(image, &input)) {
        return nullptr;
    }
    if (ctx->engine->model.meta.task != "classify") {
        ctx->last_error = "model is not a classify model (task=" +
                          ctx->engine->model.meta.task + ")";
        return nullptr;
    }
    return run_classify(ctx, input);
}

AICORE_CAPI int aicore_yolo_classify_count(
        const aicore_yolo_classify_result* res) {
    return res != nullptr ? (int)res->probs.size() : 0;
}

AICORE_CAPI float aicore_yolo_classify_prob_at(
        const aicore_yolo_classify_result* res, int index) {
    if (res == nullptr || index < 0 || index >= (int)res->probs.size()) {
        return 0.0f;
    }
    return res->probs[index];
}

AICORE_CAPI const char* aicore_yolo_classify_class_name(
        const aicore_yolo_classify_result* res, int index) {
    if (res == nullptr || index < 0 || index >= (int)res->probs.size()) {
        return nullptr;
    }
    if (index >= (int)res->class_names.size()) return nullptr;
    const std::string& name = res->class_names[index];
    return name.empty() ? nullptr : name.c_str();
}

AICORE_CAPI void aicore_yolo_classify_result_free(
        aicore_yolo_classify_result* res) {
    delete res;
}

AICORE_CAPI int aicore_yolo_last_depth_stats(
        const aicore_yolo_ctx* ctx, aicore_yolo_depth_stats* out_stats) {
    if (ctx == nullptr || out_stats == nullptr || !ctx->depth.valid) return -1;
    const auto& st = ctx->depth;
    *out_stats = aicore_yolo_depth_stats{
            st.image_w, st.image_h, st.width,
            st.height,  st.min_d,   st.max_d,
            st.mean_d,  st.p95_d,   static_cast<uint64_t>(st.valid_pixels)};
    return 0;
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

AICORE_CAPI int aicore_yolo_context_has_text_input(const aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr &&
            ctx->engine->model.has_text_input)
                   ? 1
                   : 0;
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

AICORE_CAPI int aicore_yolo_last_timings(const aicore_yolo_ctx* ctx,
                                         aicore_yolo_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr || !ctx->has_timings) {
        return -1;
    }
    *out_timings = ctx->timings;
    return 0;
}

AICORE_CAPI int aicore_yolo_last_pipeline_timings(
        const aicore_yolo_ctx* ctx, aicore_pipeline_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr || !ctx->has_timings) {
        return -1;
    }
    *out_timings = aicore_pipeline_timings{
            AICORE_PIPELINE_TIMINGS_ABI_VERSION,
            AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
                    AICORE_TIMING_POSTPROCESS | AICORE_TIMING_SERIALIZATION |
                    AICORE_TIMING_E2E,
            ctx->timings.preprocess_ms,
            ctx->timings.inference_ms,
            ctx->timings.postprocess_ms,
            ctx->timings.json_ms,
            ctx->timings.e2e_ms};
    return 0;
}

AICORE_CAPI int aicore_yolo_warmup_backend(const char* device) {
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI void aicore_yolo_shutdown(void) { aicore_runtime_shutdown(); }

AICORE_CAPI char* aicore_yolo_model_cache_dir(void) {
    return dup_cstr(aicore::yolo_model_cache_dir());
}
