// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "aicore/facedetect_capi.h"
#include "tests/common/validation_probe.hpp"

namespace {

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(q * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

aicore_facedetect_ctx* load(const char* path, const char* device) {
    aicore_facedetect_options* options = aicore_facedetect_options_new();
    aicore_facedetect_options_set_device(options, device);
    aicore_facedetect_ctx* ctx = aicore_facedetect_load_opts(path, options);
    aicore_facedetect_options_free(options);
    return ctx;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s <detector.gguf> <landmark.gguf> <image> "
                     "<device> [warmups=2] [runs=10]\n",
                     argv[0]);
        return 2;
    }
    const int warmups = argc >= 6 ? std::max(0, std::atoi(argv[5])) : 2;
    const int runs = argc >= 7 ? std::max(1, std::atoi(argv[6])) : 10;
    uint8_t* rgb = nullptr;
    int32_t width = 0;
    int32_t height = 0;
    if (aicore_facedetect_load_path_rgb(argv[3], &rgb, &width, &height) != 0) {
        std::fprintf(stderr, "unable to decode face image\n");
        return 1;
    }
    aicore_facedetect_ctx* detector = load(argv[1], argv[4]);
    aicore_facedetect_ctx* landmark = load(argv[2], argv[4]);
    if (!aicore_facedetect_is_ready(detector) ||
        !aicore_facedetect_is_ready(landmark)) {
        std::fprintf(stderr, "unable to load detector/landmark pair\n");
        aicore_facedetect_free_buffer(rgb);
        aicore_facedetect_free(detector);
        aicore_facedetect_free(landmark);
        return 1;
    }

    std::vector<double> timings;
    uint64_t reference_hash = 0;
    size_t reference_faces = 0;
    const aicore_image_view view{rgb, width, height,
                                 static_cast<size_t>(width) * 3,
                                 AICORE_IMAGE_RGB8};
    for (int run = -warmups; run < runs; ++run) {
        const auto started = std::chrono::steady_clock::now();
        const int rc = aicore_facedetect_dense_landmarks_image(
                detector, landmark, &view, 0.0f);
        const auto stopped = std::chrono::steady_clock::now();
        const size_t faces = aicore_facedetect_dense_face_count(detector);
        if (rc != 0 || faces == 0) {
            std::fprintf(stderr, "dense landmarks failed/non-empty gate: %s\n",
                         aicore_facedetect_last_error(detector));
            aicore_facedetect_free_buffer(rgb);
            aicore_facedetect_free(detector);
            aicore_facedetect_free(landmark);
            return 1;
        }
        uint64_t hash = 1469598103934665603ULL;
        for (size_t face = 0; face < faces; ++face) {
            aicore_facedetect_detection detection{};
            if (aicore_facedetect_dense_detection_at(detector, face,
                                                     &detection) != 0 ||
                !std::isfinite(detection.score)) {
                std::fprintf(stderr, "invalid dense-landmark detection\n");
                return 1;
            }
            hash = aicore::test::fnv1aAppend(hash, &detection,
                                             sizeof(detection));
            const size_t count2d =
                    aicore_facedetect_dense_point_count(detector, face, 0);
            const size_t count3d =
                    aicore_facedetect_dense_point_count(detector, face, 1);
            if (count2d != 106 || count3d != 68) {
                std::fprintf(stderr, "dense-landmark count mismatch: %zu/%zu\n",
                             count2d, count3d);
                return 1;
            }
            for (int three_d = 0; three_d <= 1; ++three_d) {
                const size_t count = three_d ? count3d : count2d;
                for (size_t point = 0; point < count; ++point) {
                    aicore_facedetect_landmark_point value{};
                    if (aicore_facedetect_dense_point_at(
                                detector, face, three_d, point, &value) != 0 ||
                        !std::isfinite(value.x) || !std::isfinite(value.y) ||
                        !std::isfinite(value.z)) {
                        std::fprintf(stderr, "invalid dense-landmark point\n");
                        return 1;
                    }
                    hash = aicore::test::fnv1aAppend(hash, &value,
                                                     sizeof(value));
                }
            }
        }
        if (run >= 0) {
            if (reference_faces != 0 &&
                (reference_faces != faces || reference_hash != hash)) {
                std::fprintf(stderr,
                             "dense-landmark output changed across runs\n");
                return 1;
            }
            reference_faces = faces;
            reference_hash = hash;
            timings.push_back(
                    std::chrono::duration<double, std::milli>(stopped - started)
                            .count());
        }
    }
    std::printf(
            "{\"suite\":\"aicore-validation\",\"task\":\"facedetect\","
            "\"device\":\"%s\",\"faces\":%zu,"
            "\"inference_p50_ms\":%.6f,\"inference_p95_ms\":%.6f,"
            "\"output_hash\":\"%016llx\"}\n",
            argv[4], reference_faces, percentile(timings, 0.5),
            percentile(timings, 0.95),
            static_cast<unsigned long long>(reference_hash));
    aicore_facedetect_free_buffer(rgb);
    aicore_facedetect_free(detector);
    aicore_facedetect_free(landmark);
    return 0;
}
