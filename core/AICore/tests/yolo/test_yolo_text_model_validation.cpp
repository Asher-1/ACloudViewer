// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// End-to-end validation for standalone YOLO text-tower catalog entries.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/yolo_capi.h"
#include "common/validation_probe.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin)
            .count();
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
            std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

std::string lowercase(const char* value) {
    std::string result = value != nullptr ? value : "";
    std::transform(
            result.begin(), result.end(), result.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

bool finiteDetection(const aicore_yolo_detection& detection) {
    return std::isfinite(detection.x1) && std::isfinite(detection.y1) &&
           std::isfinite(detection.x2) && std::isfinite(detection.y2) &&
           std::isfinite(detection.score);
}

bool inferAndHash(aicore_yolo_ctx* ctx,
                  const aicore_image_view& image,
                  uint64_t* outputHash,
                  double* e2eMs) {
    uint64_t hash = 1469598103934665603ULL;
    const char* task = aicore_yolo_context_task(ctx);
    if (std::strcmp(task, "detect") == 0) {
        if (aicore_yolo_detect_image(ctx, &image) != 0) return false;
        const int count = aicore_yolo_detection_count(ctx);
        if (count <= 0) return false;
        hash = aicore::test::fnv1aAppend(hash, &count, sizeof(count));
        for (int i = 0; i < count; ++i) {
            const aicore_yolo_detection detection =
                    aicore_yolo_detection_at(ctx, i);
            if (!finiteDetection(detection)) return false;
            hash = aicore::test::fnv1aAppend(hash, &detection,
                                             sizeof(detection));
        }
    } else if (std::strcmp(task, "segment") == 0) {
        aicore_yolo_segment_result* result = aicore_yolo_seg_image(ctx, &image);
        if (result == nullptr) return false;
        const int count = aicore_yolo_seg_det_count(result);
        bool valid = count > 0;
        hash = aicore::test::fnv1aAppend(hash, &count, sizeof(count));
        for (int i = 0; valid && i < count; ++i) {
            const aicore_yolo_detection detection =
                    aicore_yolo_seg_det_at(result, i);
            valid = finiteDetection(detection);
            hash = aicore::test::fnv1aAppend(hash, &detection,
                                             sizeof(detection));
            const aicore_yolo_plane_view mask =
                    aicore_yolo_seg_mask_at(result, i);
            valid = valid && mask.data != nullptr &&
                    mask.width == image.width && mask.height == image.height &&
                    mask.row_stride_bytes >= static_cast<size_t>(mask.width);
            const auto* bytes = static_cast<const uint8_t*>(mask.data);
            for (int y = 0; valid && y < mask.height; ++y) {
                hash = aicore::test::fnv1aAppend(
                        hash,
                        bytes + static_cast<size_t>(y) * mask.row_stride_bytes,
                        static_cast<size_t>(mask.width));
            }
        }
        aicore_yolo_seg_result_free(result);
        if (!valid) return false;
    } else {
        return false;
    }

    aicore_pipeline_timings timings{};
    if (aicore_yolo_last_pipeline_timings(ctx, &timings) != 0 ||
        !std::isfinite(timings.e2e_ms) || timings.e2e_ms <= 0.0) {
        return false;
    }
    *outputHash = hash;
    *e2eMs = timings.e2e_ms;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <text.gguf> <detector.gguf> <image> <device> "
                     "<runs> [warmups]\n",
                     argv[0]);
        return 2;
    }
    const int runs = std::max(1, std::atoi(argv[5]));
    const int warmups = argc > 6 ? std::max(0, std::atoi(argv[6])) : 2;

    uint8_t* rgb = nullptr;
    int32_t width = 0;
    int32_t height = 0;
    if (aicore_yolo_load_path_rgb(argv[3], &rgb, &width, &height) != 0 ||
        rgb == nullptr) {
        std::fprintf(stderr, "failed to load image: %s\n", argv[3]);
        return 1;
    }
    const aicore_image_view image{rgb, width, height,
                                  static_cast<size_t>(width) * 3,
                                  AICORE_IMAGE_RGB8};
    const char* classes[] = {"person", "car", "bus", "dog"};
    aicore_yolo_options* options = aicore_yolo_options_new();
    aicore_yolo_options_set_device(options, argv[4]);
    aicore_yolo_options_set_conf_thres(options, 0.05f);
    aicore_yolo_options_set_classes(options, classes, 4);
    aicore_yolo_options_set_text_model(options, argv[1]);
    const auto loadBegin = Clock::now();
    aicore_yolo_ctx* ctx = aicore_yolo_load_opts(argv[2], options);
    const double loadMs = elapsedMs(loadBegin);
    aicore_yolo_options_free(options);
    if (ctx == nullptr || !aicore_yolo_is_ready(ctx) ||
        !aicore_yolo_context_has_text_input(ctx)) {
        std::fprintf(stderr, "failed to load text-conditioned detector: %s\n",
                     ctx != nullptr && aicore_yolo_last_error(ctx) != nullptr
                             ? aicore_yolo_last_error(ctx)
                             : "unknown");
        aicore_yolo_free(ctx);
        aicore_yolo_free_buffer(rgb);
        return 1;
    }
    const std::string requestedDevice = lowercase(argv[4]);
    const std::string resolvedDevice =
            lowercase(aicore_yolo_context_device(ctx));
    if (requestedDevice != "auto" &&
        resolvedDevice.find(requestedDevice) == std::string::npos) {
        std::fprintf(stderr,
                     "backend fallback rejected: requested=%s resolved=%s\n",
                     argv[4], aicore_yolo_context_device(ctx));
        aicore_yolo_free(ctx);
        aicore_yolo_free_buffer(rgb);
        return 1;
    }

    uint64_t referenceHash = 0;
    double ignoredMs = 0.0;
    for (int i = 0; i < warmups; ++i) {
        if (!inferAndHash(ctx, image, &referenceHash, &ignoredMs)) {
            std::fprintf(stderr, "warmup failed: %s\n",
                         aicore_yolo_last_error(ctx));
            aicore_yolo_free(ctx);
            aicore_yolo_free_buffer(rgb);
            return 1;
        }
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        uint64_t hash = 0;
        double e2eMs = 0.0;
        if (!inferAndHash(ctx, image, &hash, &e2eMs) ||
            (i > 0 && hash != referenceHash)) {
            std::fprintf(stderr, "unstable or invalid inference at run %d\n",
                         i);
            aicore_yolo_free(ctx);
            aicore_yolo_free_buffer(rgb);
            return 1;
        }
        referenceHash = hash;
        timings.push_back(e2eMs);
    }

    std::printf(
            "{\"suite\":\"aicore-validation\",\"task\":\"yolo-text\","
            "\"device\":\"%s\","
            "\"load_ms\":%.6f,\"inference_p50_ms\":%.6f,"
            "\"inference_p95_ms\":%.6f,\"output_hash\":\"%016llx\"}\n",
            aicore_yolo_context_device(ctx), loadMs, percentile(timings, 0.50),
            percentile(timings, 0.95),
            static_cast<unsigned long long>(referenceHash));
    aicore_yolo_free(ctx);
    aicore_yolo_free_buffer(rgb);
    return 0;
}
