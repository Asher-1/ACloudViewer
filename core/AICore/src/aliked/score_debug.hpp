#pragma once

#include "gpu_tensor.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace lightglue::aliked_internal {

struct ScoreMapStats {
    double sum = 0.0;
    float max_val = 0.0f;
    float min_val = 0.0f;
    uint64_t fnv_hash = 0;
    size_t mismatch_count = 0;
    float max_abs_diff = 0.0f;
    const char *ref_tag = nullptr;
};

struct BackboneStageRef {
    std::vector<float> nchw;
    int32_t c = 0;
    int32_t h = 0;
    int32_t w = 0;
};

struct DkdDebugCpuRefs {
    std::vector<float> score_padded;
    std::vector<float> score_cropped;
    std::vector<float> feature_padded;
    int32_t padded_h = 0;
    int32_t padded_w = 0;
    int32_t crop_h = 0;
    int32_t crop_w = 0;
    std::unordered_map<std::string, BackboneStageRef> backbone;
};

bool DkdDebugEnabled();

void ClearDkdDebugCpuRefs();
void SetDkdDebugCpuRefs(const DkdDebugCpuRefs &refs);

void CaptureBackboneStage(DkdDebugCpuRefs *refs, const char *stage,
                          const std::vector<float> &nchw, int32_t c, int32_t h,
                          int32_t w);

void BeginDkdDebugCapture(DkdDebugCpuRefs *refs);
void EndDkdDebugCapture();
bool DkdDebugCaptureActive();
DkdDebugCpuRefs *DkdDebugCaptureTarget();

ScoreMapStats ScoreMapStatsFromNchw(const std::vector<float> &nchw, int32_t h,
                                     int32_t w, int32_t c = 1);

bool LogScoreMapStage(internal::Backend *backend, const GpuTensor &score,
                      int32_t h, int32_t w, const char *stage,
                      std::string *error);

bool LogFeatureMapStage(internal::Backend *backend, const GpuTensor &feature,
                        int32_t c, int32_t h, int32_t w, const char *stage,
                        std::string *error);

bool LogBackboneStage(internal::Backend *backend, const GpuTensor &tensor,
                      int32_t c, int32_t h, int32_t w, const char *stage,
                      std::string *error);

}  // namespace lightglue::aliked_internal
