// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// C API embed consistency: detect landmarks + embed_rgb_landmarks must match
// embed_path on the same libjpeg-loaded buffer (InsightFace recognize path).
// SKIP (77) when AICORE_TEST_FACEDETECT_GGUF or AICORE_TEST_FACEDETECT_IMAGE
// unset.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aicore/facedetect_capi.h"

namespace {

float cosineDistance(const float* a, const float* b, int dim) {
    double dot = 0.0;
    for (int i = 0; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(1.0 - dot);
}

bool embedPath(aicore_facedetect_ctx* ctx,
               const char* path,
               float minDet,
               std::vector<float>* out) {
    float* vec = nullptr;
    int dim = 0;
    const int rc = aicore_facedetect_embed_path(ctx, path, minDet, &vec, &dim);
    if (rc != 0 || !vec || dim <= 0) {
        if (vec) aicore_facedetect_free_vec(vec);
        return false;
    }
    out->assign(vec, vec + dim);
    aicore_facedetect_free_vec(vec);
    return true;
}

}  // namespace

int main() {
    const char* gguf = std::getenv("AICORE_TEST_FACEDETECT_GGUF");
    const char* image = std::getenv("AICORE_TEST_FACEDETECT_IMAGE");
    if (!gguf || !image) return 77;

    aicore_facedetect_ctx* ctx = aicore_facedetect_load_opts(gguf, nullptr);
    if (!ctx) {
        std::fprintf(stderr, "failed to load model %s\n", gguf);
        return 1;
    }

    std::vector<float> embPathA;
    std::vector<float> embPathB;
    if (!embedPath(ctx, image, 0.f, &embPathA) ||
        !embedPath(ctx, image, 0.f, &embPathB)) {
        std::fprintf(stderr, "embed_path failed: %s\n",
                     aicore_facedetect_last_error(ctx) ?: "(null)");
        aicore_facedetect_free(ctx);
        return 1;
    }
    const float selfDist = cosineDistance(embPathA.data(), embPathB.data(),
                                          static_cast<int>(embPathA.size()));
    if (selfDist > 1e-4f) {
        std::fprintf(stderr, "embed_path not reproducible: d=%f\n", selfDist);
        aicore_facedetect_free(ctx);
        return 1;
    }

    uint8_t* rgb = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    if (aicore_facedetect_load_path_rgb(image, &rgb, &w, &h) != 0 || !rgb) {
        std::fprintf(stderr, "load_path_rgb failed\n");
        aicore_facedetect_free(ctx);
        return 1;
    }

    char* detJson = aicore_facedetect_detect_rgb_json(ctx, rgb, w, h);
    if (!detJson) {
        std::fprintf(stderr, "detect_rgb_json failed: %s\n",
                     aicore_facedetect_last_error(ctx) ?: "(null)");
        aicore_facedetect_free_vec(reinterpret_cast<float*>(rgb));
        aicore_facedetect_free(ctx);
        return 1;
    }

    // Parse detect JSON: pick the largest face (same rule as Model::embed).
    float landmarks[10] = {};
    bool haveLmk = false;
    float bestArea = 0.f;
    const std::string json(detJson);
    aicore_facedetect_free_string(detJson);

    size_t searchFrom = 0;
    while (true) {
        const size_t facePos = json.find("\"box\":", searchFrom);
        if (facePos == std::string::npos) break;
        const size_t lmkPos = json.find("\"landmarks\":", facePos);
        if (lmkPos == std::string::npos) break;

        float box[4] = {};
        size_t i = facePos + 6;
        for (int bi = 0; bi < 4; ++bi) {
            while (i < json.size() && (json[i] < '0' || json[i] > '9') &&
                   json[i] != '-' && json[i] != '.') {
                ++i;
            }
            char* end = nullptr;
            box[bi] = std::strtof(json.c_str() + i, &end);
            i = end ? static_cast<size_t>(end - json.c_str()) : i + 1;
        }
        const float area =
                std::max(0.f, box[2] - box[0]) * std::max(0.f, box[3] - box[1]);

        float candidate[10] = {};
        i = lmkPos + 12;
        int idx = 0;
        while (i < json.size() && idx < 10) {
            if ((json[i] >= '0' && json[i] <= '9') || json[i] == '-') {
                char* end = nullptr;
                candidate[idx++] = std::strtof(json.c_str() + i, &end);
                i = end ? static_cast<size_t>(end - json.c_str()) : i + 1;
            } else {
                ++i;
            }
        }
        if (idx >= 10 && area >= bestArea) {
            bestArea = area;
            for (int k = 0; k < 10; ++k) landmarks[k] = candidate[k];
            haveLmk = true;
        }
        searchFrom = lmkPos + 12;
    }

    if (!haveLmk) {
        std::fprintf(stderr, "no landmarks in detect json\n");
        aicore_facedetect_free_vec(reinterpret_cast<float*>(rgb));
        aicore_facedetect_free(ctx);
        return 1;
    }

    float* embLmk = nullptr;
    int dimLmk = 0;
    const int rc = aicore_facedetect_embed_rgb_landmarks(
            ctx, rgb, w, h, landmarks, &embLmk, &dimLmk);
    aicore_facedetect_free_vec(reinterpret_cast<float*>(rgb));
    if (rc != 0 || !embLmk || dimLmk <= 0) {
        std::fprintf(stderr, "embed_rgb_landmarks failed: %s\n",
                     aicore_facedetect_last_error(ctx) ?: "(null)");
        aicore_facedetect_free(ctx);
        return 1;
    }

    const float crossDist =
            cosineDistance(embPathA.data(), embLmk,
                           std::min(static_cast<int>(embPathA.size()), dimLmk));
    aicore_facedetect_free_vec(embLmk);
    aicore_facedetect_free(ctx);

    if (crossDist > 0.05f) {
        std::fprintf(stderr,
                     "embed_path vs embed_rgb_landmarks mismatch: d=%f "
                     "(expected < 0.05)\n",
                     crossDist);
        return 1;
    }

    std::fprintf(stderr, "test_facedetect_embed_capi ok (cross d=%f)\n",
                 crossDist);
    return 0;
}
