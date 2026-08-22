// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Score-map / feature-map checksum diagnostics (LIGHTGLUE_ALIKED_DKD_DEBUG=1).

#include "tasks/aliked/score_debug.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tasks/aliked/gpu_sync.hpp"

namespace lightglue::aliked_internal {
namespace {

DkdDebugCpuRefs g_cpu_refs;
DkdDebugCpuRefs *g_capture_refs = nullptr;

uint64_t Fnv1a64(const float *data, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &data[i], sizeof(bits));
        hash ^= static_cast<uint64_t>(bits);
        hash *= 1099511628211ull;
    }
    return hash;
}

const std::vector<float> *FindScoreRef(int32_t h,
                                       int32_t w,
                                       const char **out_tag) {
    const size_t count = static_cast<size_t>(h) * static_cast<size_t>(w);
    if (g_cpu_refs.score_padded.size() == count && g_cpu_refs.padded_h == h &&
        g_cpu_refs.padded_w == w) {
        if (out_tag) {
            *out_tag = "padded";
        }
        return &g_cpu_refs.score_padded;
    }
    if (g_cpu_refs.score_cropped.size() == count && g_cpu_refs.crop_h == h &&
        g_cpu_refs.crop_w == w) {
        if (out_tag) {
            *out_tag = "cropped";
        }
        return &g_cpu_refs.score_cropped;
    }
    if (out_tag) {
        *out_tag = nullptr;
    }
    return nullptr;
}

ScoreMapStats StatsAgainstRef(const std::vector<float> &nchw,
                              const std::vector<float> *ref) {
    ScoreMapStats stats;
    if (nchw.empty()) {
        return stats;
    }
    stats.min_val = nchw[0];
    stats.max_val = nchw[0];
    for (float v : nchw) {
        stats.sum += static_cast<double>(v);
        stats.min_val = std::min(stats.min_val, v);
        stats.max_val = std::max(stats.max_val, v);
    }
    stats.fnv_hash = Fnv1a64(nchw.data(), nchw.size());
    if (ref != nullptr && ref->size() == nchw.size()) {
        for (size_t i = 0; i < nchw.size(); ++i) {
            const float diff = std::fabs(nchw[i] - (*ref)[i]);
            if (diff > 0.0f) {
                ++stats.mismatch_count;
                stats.max_abs_diff = std::max(stats.max_abs_diff, diff);
            }
        }
    }
    return stats;
}

void PrintStatsLine(const char *kind,
                    const char *stage,
                    int32_t c,
                    int32_t h,
                    int32_t w,
                    const ScoreMapStats &stats) {
    std::fprintf(stderr,
                 "[dkd-debug] kind=%s stage=%s c=%d h=%d w=%d sum=%.6f "
                 "min=%.6f max=%.6f hash=%016llx",
                 kind, stage, c, h, w, stats.sum, stats.min_val, stats.max_val,
                 static_cast<unsigned long long>(stats.fnv_hash));
    if (stats.ref_tag != nullptr) {
        std::fprintf(stderr, " ref=%s mismatch=%zu max_abs_diff=%.6e",
                     stats.ref_tag, stats.mismatch_count, stats.max_abs_diff);
    }
    std::fprintf(stderr, "\n");
}

}  // namespace

bool DkdDebugEnabled() {
    // The historical LIGHTGLUE_ALIKED_DKD_DEBUG env switch was development
    // scaffolding and is removed; the checksum/parity plumbing below stays
    // dormant until an explicit API re-enables it.
    return false;
}

void ClearDkdDebugCpuRefs() { g_cpu_refs = DkdDebugCpuRefs{}; }

void SetDkdDebugCpuRefs(const DkdDebugCpuRefs &refs) { g_cpu_refs = refs; }

void CaptureBackboneStage(DkdDebugCpuRefs *refs,
                          const char *stage,
                          const std::vector<float> &nchw,
                          int32_t c,
                          int32_t h,
                          int32_t w) {
    if (refs == nullptr || stage == nullptr || stage[0] == '\0') {
        return;
    }
    const size_t expected = static_cast<size_t>(c) * static_cast<size_t>(h) *
                            static_cast<size_t>(w);
    if (nchw.size() != expected || expected == 0) {
        return;
    }
    refs->backbone[stage] = BackboneStageRef{nchw, c, h, w};
}

void BeginDkdDebugCapture(DkdDebugCpuRefs *refs) { g_capture_refs = refs; }

void EndDkdDebugCapture() { g_capture_refs = nullptr; }

bool DkdDebugCaptureActive() { return g_capture_refs != nullptr; }

DkdDebugCpuRefs *DkdDebugCaptureTarget() { return g_capture_refs; }

const BackboneStageRef *FindBackboneRef(const char *stage) {
    if (stage == nullptr) {
        return nullptr;
    }
    const auto it = g_cpu_refs.backbone.find(stage);
    if (it == g_cpu_refs.backbone.end()) {
        return nullptr;
    }
    return &it->second;
}

ScoreMapStats ScoreMapStatsFromNchw(const std::vector<float> &nchw,
                                    int32_t h,
                                    int32_t w,
                                    int32_t c) {
    const size_t expected = static_cast<size_t>(c) * static_cast<size_t>(h) *
                            static_cast<size_t>(w);
    if (nchw.size() != expected || expected == 0) {
        return ScoreMapStats{};
    }
    const char *tag = nullptr;
    const std::vector<float> *ref = nullptr;
    if (c == 1) {
        ref = FindScoreRef(h, w, &tag);
    } else if (g_cpu_refs.feature_padded.size() == expected &&
               g_cpu_refs.padded_h == h && g_cpu_refs.padded_w == w &&
               c == 128) {
        ref = &g_cpu_refs.feature_padded;
        tag = "feature_padded";
    }
    ScoreMapStats stats = StatsAgainstRef(nchw, ref);
    stats.ref_tag = tag;
    return stats;
}

bool LogScoreMapStage(internal::Backend *backend,
                      const GpuTensor &score,
                      int32_t h,
                      int32_t w,
                      const char *stage,
                      std::string *error) {
    if (!DkdDebugEnabled() || stage == nullptr) {
        return true;
    }
    std::vector<float> nchw;
    if (!score.DownloadNchw(backend, &nchw, 1, h, w, error)) {
        return false;
    }
    PrintStatsLine("score", stage, 1, h, w, ScoreMapStatsFromNchw(nchw, h, w));
    return true;
}

bool LogFeatureMapStage(internal::Backend *backend,
                        const GpuTensor &feature,
                        int32_t c,
                        int32_t h,
                        int32_t w,
                        const char *stage,
                        std::string *error) {
    if (!DkdDebugEnabled() || stage == nullptr) {
        return true;
    }
    std::vector<float> nchw;
    if (!feature.DownloadNchw(backend, &nchw, c, h, w, error)) {
        return false;
    }
    PrintStatsLine("feature", stage, c, h, w,
                   ScoreMapStatsFromNchw(nchw, h, w, c));
    return true;
}

bool LogBackboneStage(internal::Backend *backend,
                      const GpuTensor &tensor,
                      int32_t c,
                      int32_t h,
                      int32_t w,
                      const char *stage,
                      std::string *error) {
    if (!DkdDebugEnabled() || stage == nullptr) {
        return true;
    }
    std::vector<float> nchw;
    if (!tensor.DownloadNchw(backend, &nchw, c, h, w, error)) {
        return false;
    }
    const size_t expected = static_cast<size_t>(c) * static_cast<size_t>(h) *
                            static_cast<size_t>(w);
    if (nchw.size() != expected) {
        if (error) {
            *error = std::string("backbone debug download size mismatch at ") +
                     stage;
        }
        return false;
    }
    const BackboneStageRef *ref = FindBackboneRef(stage);
    const char *tag = ref != nullptr ? stage : nullptr;
    const std::vector<float> *ref_data = ref != nullptr ? &ref->nchw : nullptr;
    ScoreMapStats stats = StatsAgainstRef(nchw, ref_data);
    stats.ref_tag = tag;
    PrintStatsLine("backbone", stage, c, h, w, stats);
    return true;
}

}  // namespace lightglue::aliked_internal
