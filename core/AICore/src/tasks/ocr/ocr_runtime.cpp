// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ocr_runtime.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>

#if (defined(__GNUC__) || defined(__clang__)) && \
        (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define AICORE_OCR_HAS_AVX2_DISPATCH 1
#else
#define AICORE_OCR_HAS_AVX2_DISPATCH 0
#endif
#include <sstream>
#include <unordered_set>

#include "common/blocked_conv.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/winograd.hpp"

namespace aicore {
namespace ocr {

namespace {

// CTC argmax is O(T*V), with V=6625 for PP-OCR Chinese. Keep the decode
// state machine scalar and only vectorize the independent max reduction.
// The vector path has exactly the scalar `>` semantics: NaNs never replace
// the current winner and equal values retain their earliest class index.
inline int ctcArgmaxScalar(const float* row, int vocab, float* bestProb) {
    int best = 0;
    float bestValue = row[0];
    for (int i = 1; i < vocab; ++i) {
        if (row[i] > bestValue) {
            bestValue = row[i];
            best = i;
        }
    }
    *bestProb = bestValue;
    return best;
}

#if AICORE_OCR_HAS_AVX2_DISPATCH
__attribute__((target("avx2"))) int ctcArgmaxAvx2(const float* row,
                                                  int vocab,
                                                  float* bestProb) {
    // Scalar starts from row[0], therefore a leading NaN is itself the
    // winner (all later `value > NaN` comparisons are false). Preserve that
    // unusual but defined behavior before entering the vector path.
    if (std::isnan(row[0])) return ctcArgmaxScalar(row, vocab, bestProb);

    const __m256 negInf =
            _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    __m256 laneMax = negInf;
    int i = 0;
    for (; i + 8 <= vocab; i += 8) {
        const __m256 values = _mm256_loadu_ps(row + i);
        // Ordered compare is false for NaNs, matching scalar `>`.
        const __m256 replace = _mm256_cmp_ps(values, laneMax, _CMP_GT_OQ);
        laneMax = _mm256_blendv_ps(laneMax, values, replace);
    }
    alignas(32) float lane[8];
    _mm256_store_ps(lane, laneMax);
    float bestValue = lane[0];
    for (int laneIndex = 1; laneIndex < 8; ++laneIndex)
        if (lane[laneIndex] > bestValue) bestValue = lane[laneIndex];
    for (; i < vocab; ++i)
        if (row[i] > bestValue) bestValue = row[i];

    // A second vector pass finds the earliest equal class, preserving the
    // scalar tie rule without carrying an index through the reduction.
    const __m256 target = _mm256_set1_ps(bestValue);
    i = 0;
    for (; i + 8 <= vocab; i += 8) {
        const int mask = _mm256_movemask_ps(
                _mm256_cmp_ps(_mm256_loadu_ps(row + i), target, _CMP_EQ_OQ));
        if (mask != 0) {
            *bestProb = bestValue;
            return i + __builtin_ctz(static_cast<unsigned>(mask));
        }
    }
    for (; i < vocab; ++i) {
        if (row[i] == bestValue) {
            *bestProb = bestValue;
            return i;
        }
    }
    // Only reachable for an all-NaN row, already handled above; retain a
    // scalar fallback rather than manufacture an index.
    return ctcArgmaxScalar(row, vocab, bestProb);
}

bool ctcAvx2Available() {
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
}
#endif

inline int ctcArgmax(const float* row, int vocab, float* bestProb) {
#if AICORE_OCR_HAS_AVX2_DISPATCH
    if (ctcAvx2Available()) return ctcArgmaxAvx2(row, vocab, bestProb);
#endif
    return ctcArgmaxScalar(row, vocab, bestProb);
}

}  // namespace

namespace {

// CTC collapse shared by both decode entry points. `pick(t, &prob)` yields
// the argmax class of timestep t and its probability; everything after that
// (blank, repeat collapse, dictionary layout) is identical whether the
// argmax was reduced on the CPU or inside the graph.
template <typename Pick>
std::string ctcCollapse(int timeSteps,
                        int vocab,
                        const std::vector<std::string>& dictionary,
                        float* meanScore,
                        Pick&& pick) {
    std::string text;
    double scoreSum = 0.0;
    int scoreCount = 0;
    int previous = 0;
    // Class-index layout over the rec head (PP-OCR CTC): index 0 is the
    // blank and the remaining classes follow one of two published
    // container conventions:
    //   vocab == dict.size() + 2 -> 1..N = dictionary, N+1 = " " (the raw
    //          ppocr_keys_v1 vocabulary; use_space_char appends a trailing
    //          space class at export time),
    //   vocab == dict.size() + 1 -> 1..N = dictionary (the stored table
    //          already carries its own trailing " " entry).
    // The trailing space class is real recognized text, not a separator:
    // unconditionally skipping vocab-1 deleted every recognized space (and
    // the last vocabulary character whenever the container carries no
    // space class).
    const int64_t dictSize = static_cast<int64_t>(dictionary.size());
    const bool spaceTail = static_cast<int64_t>(vocab) - dictSize == 2;
    for (int t = 0; t < timeSteps; ++t) {
        float bestProb = 0.0f;
        const int best = pick(t, &bestProb);
        // 0 = CTC blank; collapse repeats, then map the class through the
        // layout above.
        if (best != 0 && best != previous) {
            if (best == vocab - 1 && spaceTail) {
                text += " ";
                scoreSum += bestProb;
                ++scoreCount;
            } else {
                const int dictIndex = best - 1;
                if (dictIndex >= 0 && dictIndex < static_cast<int>(dictSize)) {
                    text += dictionary[dictIndex];
                    scoreSum += bestProb;
                    ++scoreCount;
                }
            }
        }
        previous = best;
    }
    if (meanScore)
        *meanScore =
                scoreCount ? static_cast<float>(scoreSum / scoreCount) : 0.0f;
    return text;
}

}  // namespace

std::string ctcGreedyDecode(const float* probs,
                            int timeSteps,
                            int vocab,
                            const std::vector<std::string>& dictionary,
                            float* meanScore) {
    return ctcCollapse(timeSteps, vocab, dictionary, meanScore,
                       [&](int t, float* bestProb) {
                           return ctcArgmax(
                                   probs + static_cast<size_t>(t) * vocab,
                                   vocab, bestProb);
                       });
}

std::string ctcGreedyDecodeIndexed(const int32_t* bestIndex,
                                   const float* bestProb,
                                   int timeSteps,
                                   int vocab,
                                   const std::vector<std::string>& dictionary,
                                   float* meanScore) {
    return ctcCollapse(timeSteps, vocab, dictionary, meanScore,
                       [&](int t, float* prob) {
                           *prob = bestProb[t];
                           // Clamp defensively: a backend bug can only
                           // degrade to "unknown class", never index out
                           // of the dictionary.
                           const int32_t idx = bestIndex[t];
                           return (idx < 0 || idx >= vocab) ? 0 : idx;
                       });
}

std::vector<float> preprocessRec(const unsigned char* rgb,
                                 int width,
                                 int height,
                                 int targetWidth,
                                 int* outW) {
    // PP-OCRv4 rec: resize to height 48 (aspect preserved), normalize with
    // the PP-OCR ImageNet statistics, NCHW float output.
    constexpr int kHeight = 48;
    int w = std::max(1, static_cast<int>(1LL * width * kHeight / height));
    if (targetWidth > 0) w = std::min(w, targetWidth);
    if (outW) *outW = w;

    std::vector<float> out(3ULL * kHeight * w);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};
    for (int y = 0; y < kHeight; ++y) {
        const int sy = std::min(height - 1, y * height / kHeight);
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(width - 1, x * width / std::max(1, w));
            const unsigned char* pixel =
                    rgb + (static_cast<size_t>(sy) * width + sx) * 3;
            for (int c = 0; c < 3; ++c)
                out[(static_cast<size_t>(c) * kHeight + y) * w + x] =
                        (pixel[c] / 255.0f - mean[c]) / stdv[c];
        }
    }
    return out;
}

}  // namespace ocr
}  // namespace aicore
