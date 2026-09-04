#pragma once

#include <string>

#include "aicore/loma_capi.h"
#include "base/image_reader.h"
#include "feature/types.h"
#include "util/threading.h"

namespace colmap {

enum class LomaMatcherVariant {
    kB,
    kB128,
    kR,
    kL,
    kG,
};

struct LomaExtractionOptions {
    std::string detector_model_path;
    std::string descriptor_model_path;
    int max_num_features = 2048;
    float min_score = 0.0f;
    std::string device = "auto";
    int num_threads = 1;
    FeatureDescriptorType descriptor_type = FeatureDescriptorType::kLomaG;
    bool Check() const;
};

struct LomaMatchingOptions {
    std::string matcher_model_path;
    float min_score = 0.1f;
    std::string device = "auto";
    LomaMatcherVariant matcher_variant = LomaMatcherVariant::kB;
    bool Check() const;
};

std::string DefaultLomaModelPath(aicore_loma_model_role role);
std::string DefaultLomaModelPath(aicore_loma_model_variant variant);
void ResolveDefaultLomaModelPaths(LomaExtractionOptions* options);
void ResolveDefaultLomaModelPaths(LomaMatchingOptions* options);
// Catalog-default paths are provisioned through the shared AICore cache;
// non-default caller-provided paths are never downloaded or replaced.
bool ProvisionDefaultLomaModels(LomaExtractionOptions* options);
bool ProvisionDefaultLomaModels(LomaMatchingOptions* options);
bool ParseLomaMatcherVariant(const std::string& value,
                             LomaMatcherVariant* variant);

bool ExtractLomaFeatures(const LomaExtractionOptions& options,
                         const Bitmap& bitmap,
                         FeatureKeypoints* keypoints,
                         FeatureDescriptorsFloat* descriptors);

bool MatchLomaFeatures(const LomaMatchingOptions& options,
                       const FeatureKeypoints& keypoints1,
                       const FeatureDescriptorsFloat& descriptors1,
                       const FeatureKeypoints& keypoints2,
                       const FeatureDescriptorsFloat& descriptors2,
                       int image_width1,
                       int image_height1,
                       int image_width2,
                       int image_height2,
                       FeatureMatches* matches);

// Sequential LoMa scheduler for the legacy Reconstruction command path. It
// writes keypoints to the canonical keypoints table and float descriptors to
// the explicit float_descriptors protocol.
class LomaFeatureExtractor : public Thread {
public:
    LomaFeatureExtractor(const ImageReaderOptions& reader_options,
                         const LomaExtractionOptions& loma_options);

    bool Succeeded() const { return succeeded_; }

private:
    void Run();
    ImageReaderOptions reader_options_;
    LomaExtractionOptions loma_options_;
    bool succeeded_ = false;
};

}  // namespace colmap
