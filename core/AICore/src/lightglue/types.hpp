// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aicore {
namespace lightglue {

enum class FeatureType {
    kSuperPoint,
    kDisk,
    kAliked,
    kSift,
    kDogHardNet,
};

enum class FeatureMatcherType {
    kAuto,
    kSiftLightGlue,
    kAlikedLightGlue,
};

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float scale = 1.0f;
    float orientation = 0.0f;
};

struct Features {
    std::vector<Keypoint> keypoints;
    std::vector<float> descriptors;
    int32_t descriptor_dim = 0;
    int32_t image_width = 0;
    int32_t image_height = 0;
};

struct Match {
    int32_t point2D_idx1 = -1;
    int32_t point2D_idx2 = -1;
    float score = 0.0f;
};

struct RawResult {
    std::vector<int32_t> matches0;
    std::vector<float> mscores0;
};

struct MatchingOptions {
    std::string model_path;
    FeatureMatcherType type = FeatureMatcherType::kAuto;
    std::string device = "cpu";
    int32_t num_threads = 0;
    double min_score = 0.1;

    bool check(std::string* error = nullptr) const;
};

struct ModelGeometry {
    int32_t input_dim = 0;
    int32_t descriptor_dim = 0;
    int32_t num_heads = 0;
    int32_t num_layers = 0;
    int32_t feature_type = 0;
    int32_t add_scale_orientation = 0;
};

class FeatureMatcher {
public:
    virtual ~FeatureMatcher() = default;

    virtual bool match(const Features& image1, const Features& image2,
                       std::vector<Match>* matches) = 0;
    virtual bool run_raw(const Features& image1, const Features& image2,
                         RawResult* result) = 0;
    virtual const std::string& error() const = 0;
    virtual FeatureType model_feature_type() const = 0;
    virtual const std::string& device() const = 0;
    virtual bool geometry(ModelGeometry* out) const = 0;
};

std::unique_ptr<FeatureMatcher> create_feature_matcher(
        const MatchingOptions& options, std::string* error = nullptr);

bool quantize_model(const std::string& input_gguf,
                    const std::string& output_gguf, const std::string& type,
                    std::string* error = nullptr);

const char* feature_type_name(FeatureType type);
const char* feature_matcher_type_name(FeatureMatcherType type);

}  // namespace lightglue
}  // namespace aicore
