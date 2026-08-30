// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// YOLOE visual-prompt (SAVPE) model-tier test — exercises the integrated
// visual-prompt path on a real GGUF converted with savpe weights:
//
//   1. aicore_yolo_gguf_has_savpe probes the GGUF (must be 1),
//   2. the visual-prompt context loads (no class list, no text model — the
//      head's cls_pe comes from the savpe encoder),
//   3. inference returns detections with class ids < Q labeled
//      object0..object{Q-1} (official visual-prompt semantics),
//   4. self-consistency gate: each drawn box's own class is detected inside
//      that box region (a wrong savpe computation would not localize the
//      prompted objects),
//   5. a prompt-free GGUF (no yolo.savpe) rejects visual prompts.
//
// Assets (location-only env vars; unset => skip with 77, same contract as
// test_yolo_capi_parity):
//   AICORE_TEST_YOLO_SAVPE_GGUF     yoloe GGUF with yolo.savpe = 1
//   AICORE_TEST_YOLO_IMAGE          prompt image
//   AICORE_TEST_YOLO_VP_BOXES       comma-separated x1,y1,x2,y2 per box
//   AICORE_TEST_YOLO_SAVPE_PF_GGUF  optional -pf GGUF (expects load fail)
//   AICORE_TEST_YOLO_SAVPE_MINIOU   optional gate (default 0.3)
//   AICORE_TEST_YOLO_SAVPE_STRICT   set to require every box matched

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/yolo_capi.h"

static const char* env_or_null(const char* name) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

static int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

namespace {

std::vector<float> parse_boxes(const std::string& csv) {
    std::vector<float> values;
    std::string item;
    for (const char c : csv) {
        if (c == ',') {
            values.push_back(std::strtof(item.c_str(), nullptr));
            item.clear();
        } else if (c != ' ') {
            item.push_back(c);
        }
    }
    if (!item.empty()) values.push_back(std::strtof(item.c_str(), nullptr));
    return values;
}

// Minimal detection view of the JSON envelope (class_id + score + box).
struct Det {
    int class_id = -1;
    float score = 0.f;
    float box[4] = {0.f, 0.f, 0.f, 0.f};
};

std::vector<Det> parse_detections(const char* json) {
    std::vector<Det> dets;
    const std::string s(json);
    size_t pos = 0;
    while ((pos = s.find("\"class_id\"", pos)) != std::string::npos) {
        Det d;
        size_t v = s.find(':', pos) + 1;
        d.class_id = std::atoi(s.c_str() + v);
        const size_t score_pos = s.find("\"score\"", pos);
        if (score_pos != std::string::npos) {
            v = s.find(':', score_pos) + 1;
            d.score = std::strtof(s.c_str() + v, nullptr);
        }
        const size_t box_pos = s.find("\"box\"", pos);
        if (box_pos != std::string::npos) {
            v = s.find('[', box_pos) + 1;
            for (int k = 0; k < 4; ++k) {
                d.box[k] = std::strtof(s.c_str() + v, nullptr);
                v = s.find(',', v) + 1;
            }
        }
        dets.push_back(d);
        pos += 10;
    }
    return dets;
}

float iou(const float* a, const float* b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const float areaB = (b[2] - b[0]) * (b[3] - b[1]);
    const float uni = areaA + areaB - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// Self-consistency: for every prompt q, the best object-q detection should
// sit inside (or near) its own example box. Returns the matched count.
int matched_boxes(const std::vector<Det>& dets,
                  const std::vector<float>& boxes,
                  int q,
                  float min_iou) {
    int matched = 0;
    for (int c = 0; c < q; ++c) {
        float best = 0.f;
        const Det* best_det = nullptr;
        for (const Det& d : dets) {
            if (d.class_id == c && d.score > best) {
                best = d.score;
                best_det = &d;
            }
        }
        if (best_det != nullptr &&
            iou(best_det->box, &boxes[(size_t)c * 4]) >= min_iou) {
            ++matched;
        }
    }
    return matched;
}

}  // namespace

int main() {
    const char* gguf = env_or_null("AICORE_TEST_YOLO_SAVPE_GGUF");
    const char* image = env_or_null("AICORE_TEST_YOLO_IMAGE");
    const char* boxes_csv = env_or_null("AICORE_TEST_YOLO_VP_BOXES");
    if (gguf == nullptr || image == nullptr || boxes_csv == nullptr) {
        std::printf(
                "[yolo] savpe model test skipped (set AICORE_TEST_YOLO_"
                "SAVPE_GGUF / AICORE_TEST_YOLO_IMAGE / "
                "AICORE_TEST_YOLO_VP_BOXES)\n");
        return 77;
    }

    const std::vector<float> boxes = parse_boxes(boxes_csv);
    if (boxes.size() < 4 || boxes.size() % 4 != 0) {
        std::printf(
                "[yolo] savpe test skipped: bad AICORE_TEST_YOLO_VP_BOXES\n");
        return 77;
    }
    const int q = (int)(boxes.size() / 4);
    float min_iou = 0.3f;
    if (const char* v = env_or_null("AICORE_TEST_YOLO_SAVPE_MINIOU")) {
        min_iou = std::strtof(v, nullptr);
    }
    const bool strict = env_or_null("AICORE_TEST_YOLO_SAVPE_STRICT") != nullptr;

    // 1. The GGUF must declare savpe support.
    CHECK(aicore_yolo_gguf_has_savpe(gguf) == 1);

    // 2. Visual-prompt load: no classes, no text model.
    aicore_yolo_options* opts = aicore_yolo_options_new();
    aicore_yolo_options_set_device(opts, "cpu");
    aicore_yolo_options_set_visual_prompts(opts, boxes.data(), q);
    CHECK(aicore_yolo_options_get_visual_prompt_count(opts) == q);
    aicore_yolo_ctx* ctx = aicore_yolo_load_opts(gguf, opts);
    aicore_yolo_options_free(opts);
    CHECK(ctx != nullptr && aicore_yolo_is_ready(ctx) == 1);
    if (ctx == nullptr || aicore_yolo_is_ready(ctx) != 1) {
        std::printf("[yolo] savpe load failed: %s\n",
                    ctx != nullptr && aicore_yolo_last_error(ctx)
                            ? aicore_yolo_last_error(ctx)
                            : "unknown");
        return failures;
    }
    CHECK(aicore_yolo_context_has_visual_prompts(ctx) == 1);
    CHECK(aicore_yolo_context_num_classes(ctx) == (uint32_t)q);

    // 3. Inference (segment GGUFs via the typed API, detect via JSON).
    uint8_t* rgb = nullptr;
    int32_t w = 0, h = 0;
    CHECK(aicore_yolo_load_path_rgb(image, &rgb, &w, &h) == 0);
    if (rgb != nullptr) {
        std::vector<Det> dets;
        if (aicore_yolo_context_task(ctx) != nullptr &&
            std::strcmp(aicore_yolo_context_task(ctx), "segment") == 0) {
            aicore_yolo_segment_result* seg =
                    aicore_yolo_seg_rgb(ctx, rgb, w, h);
            CHECK(seg != nullptr);
            if (seg != nullptr) {
                const int n = aicore_yolo_seg_det_count(seg);
                for (int i = 0; i < n; ++i) {
                    const aicore_yolo_detection d =
                            aicore_yolo_seg_det_at(seg, i);
                    Det t;
                    t.class_id = d.class_id;
                    t.score = d.score;
                    t.box[0] = d.x1;
                    t.box[1] = d.y1;
                    t.box[2] = d.x2;
                    t.box[3] = d.y2;
                    dets.push_back(t);
                    // 3b. names resolve to objectN for every cid < q.
                    const char* name = aicore_yolo_seg_det_class_name(seg, i);
                    CHECK(name != nullptr &&
                          std::strncmp(name, "object", 6) == 0);
                }
                aicore_yolo_seg_result_free(seg);
            }
        } else {
            char* json = aicore_yolo_detect_rgb_json(ctx, rgb, w, h);
            CHECK(json != nullptr);
            if (json != nullptr) {
                dets = parse_detections(json);
                aicore_yolo_free_buffer(json);
            }
        }
        // 3. class ids stay inside the prompt range.
        for (const Det& d : dets) {
            CHECK(d.class_id >= 0 && d.class_id < q);
        }
        // 4. Self-consistency gate (skipped when nothing crossed conf).
        if (!dets.empty()) {
            const int matched = matched_boxes(dets, boxes, q, min_iou);
            CHECK(matched >= 1);
            if (strict) CHECK(matched == q);
        }
        aicore_yolo_free_buffer(rgb);
    }

    aicore_yolo_free(ctx);

    // 5. A prompt-free GGUF must reject visual prompts at load.
    if (const char* pf = env_or_null("AICORE_TEST_YOLO_SAVPE_PF_GGUF")) {
        aicore_yolo_options* pf_opts = aicore_yolo_options_new();
        aicore_yolo_options_set_device(pf_opts, "cpu");
        aicore_yolo_options_set_visual_prompts(pf_opts, boxes.data(), q);
        aicore_yolo_ctx* pf_ctx = aicore_yolo_load_opts(pf, pf_opts);
        CHECK(pf_ctx == nullptr || aicore_yolo_is_ready(pf_ctx) == 0);
        if (pf_ctx != nullptr) aicore_yolo_free(pf_ctx);
        aicore_yolo_options_free(pf_opts);
    }

    if (failures == 0) std::printf("[yolo] savpe model test passed\n");
    return failures;
}
