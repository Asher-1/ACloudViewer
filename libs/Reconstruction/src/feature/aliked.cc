// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Fork wiring of the upstream ALIKED feature extractor / matcher to the
// AICore ggml task family:
//   - extraction runs through aicore_aliked_* (tasks/aliked, ggml CNN);
//   - LightGlue matching runs through feature/lightglue.h (tasks/lightglue,
//     ALIKED geometry);
//   - brute-force matching is the shared Eigen evaluator in
//     feature/learned_matchers.h with upstream's min_cossim / max_ratio /
//     cross_check semantics;
//   - model defaults resolve through the shared aicore model catalog and
//     download into the AICore shared cache with pinned SHA-256 checks.

#include "feature/aliked.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "aicore/aliked_capi.h"
#include "aicore/model_catalog_capi.h"
#include "feature/model_cache.h"
#include "util/logging.h"

namespace colmap {
namespace {

const aicore_model_entry* FindExtractorModel(const std::string& role_prefix,
                                             const std::string& quant) {
    return FindCatalogModel(AICORE_MODEL_FAMILY_ALIKED,
                            [&](const aicore_model_entry* entry) {
        const std::string filename(entry->filename);
        if (filename.find(role_prefix) != 0) return false;
        if (quant.empty()) return true;
        // The quant tag precedes ".gguf" (e.g. aliked-n16rot-f16.gguf).
        const std::string tag = "-" + quant + ".gguf";
        return filename.size() > tag.size() &&
               filename.compare(filename.size() - tag.size(), tag.size(),
                                tag) == 0;
    });
}

void ConvertFeatures(const aicore_lightglue_features& features,
                     FeatureKeypoints* keypoints,
                     FeatureDescriptorsFloatData* descriptors) {
    const int count = features.n_keypoints;
    keypoints->resize(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        // ALIKED keypoints are pixel coordinates without scale/orientation.
        (*keypoints)[static_cast<size_t>(i)] = FeatureKeypoint(
                features.keypoints[i].x, features.keypoints[i].y);
    }
    if (count > 0 && features.descriptors != nullptr &&
        features.descriptor_dim > 0) {
        Eigen::Map<const FeatureDescriptorsFloatData> mapped(
                features.descriptors, features.n_keypoints,
                features.descriptor_dim);
        *descriptors = mapped;
    } else {
        descriptors->resize(0, 0);
    }
}

void FreeFeatures(aicore_lightglue_features* features) {
    aicore_aliked_free_buffer(features->keypoints);
    aicore_aliked_free_buffer(features->descriptors);
    features->keypoints = nullptr;
    features->descriptors = nullptr;
}

class AlikedFeatureExtractor : public FeatureExtractor {
 public:
  explicit AlikedFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
    THROW_CHECK_NOTNULL(options_.aliked);
    AlikedExtractionOptions resolved = *options_.aliked;
    if (!ProvisionDefaultAlikedModels(&resolved)) {
      throw std::runtime_error("ALIKED extractor model provisioning failed");
    }
    const bool use_n32 = options_.type == FeatureExtractorType::ALIKED_N32;
    const std::string& model_path =
            use_n32 ? resolved.n32_model_path : resolved.n16rot_model_path;
    if (model_path.empty()) {
      throw std::runtime_error(
              std::string(use_n32 ? "ALIKED N32" : "ALIKED N16ROT") +
              " extractor model is not published in the shared catalog");
    }

    aicore_aliked_options* engine_options = aicore_aliked_options_new();
    if (engine_options == nullptr) {
      throw std::runtime_error("ALIKED options allocation failed");
    }
    aicore_aliked_options_set_device(engine_options, resolved.device.c_str());
    aicore_aliked_options_set_max_keypoints(engine_options,
                                            resolved.max_num_features);
    ctx_ = aicore_aliked_load_opts(model_path.c_str(), engine_options);
    aicore_aliked_options_free(engine_options);
    if (ctx_ == nullptr || !aicore_aliked_is_ready(ctx_)) {
      const char* error = ctx_ != nullptr ? aicore_aliked_last_error(ctx_)
                                          : "allocation failed";
      const std::string message =
              std::string("ALIKED extractor initialization failed: ") +
              (error != nullptr ? error : "unknown");
      aicore_aliked_free(ctx_);
      ctx_ = nullptr;
      throw std::runtime_error(message);
    }
  }

  ~AlikedFeatureExtractor() override { aicore_aliked_free(ctx_); }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK_NOTNULL(keypoints);
    THROW_CHECK_NOTNULL(descriptors);
    keypoints->clear();
    descriptors->type = options_.type;
    descriptors->data.resize(0, 0);
    if (bitmap.Width() <= 0 || bitmap.Height() <= 0) {
      return false;
    }
    const Bitmap rgb = bitmap.IsRGB() ? bitmap.Clone() : bitmap.CloneAsRGB();
    aicore_image_view view{};
    view.data = rgb.GetScanline(0);
    view.width = rgb.Width();
    view.height = rgb.Height();
    view.row_stride_bytes = static_cast<size_t>(rgb.ScanWidth());
    view.format = AICORE_IMAGE_RGB8;
    if (view.data == nullptr) {
      return false;
    }
    aicore_lightglue_features features{};
    const bool ok =
            aicore_aliked_extract_image_view(ctx_, &view, &features) == 0;
    if (ok) {
      FeatureDescriptorsFloat descriptors_float(
              options_.type, FeatureDescriptorsFloatData());
      ConvertFeatures(features, keypoints, &descriptors_float.data);
      *descriptors = descriptors_float.ToBytes();
    } else {
      const char* error = aicore_aliked_last_error(ctx_);
      LOG(ERROR) << "ALIKED extraction failed: "
                 << (error != nullptr ? error : "unknown");
    }
    FreeFeatures(&features);
    return ok;
  }

 private:
  const FeatureExtractionOptions options_;
  aicore_aliked_ctx* ctx_ = nullptr;
};

}  // namespace

bool AlikedExtractionOptions::Check() const {
    CHECK_OPTION_GT(max_num_features, 0);
    CHECK_OPTION_GE(min_score, 0);
    CHECK_OPTION_LE(min_score, 1);
    return true;
}

bool AlikedMatchingOptions::Check() const {
    CHECK_OPTION(brute_force.Check());
    CHECK_OPTION(lightglue.Check());
    return true;
}

std::string DefaultAlikedExtractorModelPath(const char* role,
                                            const char* quant) {
    const aicore_model_entry* entry = FindExtractorModel(
            role != nullptr ? role : "", quant != nullptr ? quant : "");
    return CatalogModelPath(entry, &aicore_aliked_model_cache_dir);
}

void ResolveDefaultAlikedModelPaths(AlikedExtractionOptions* options) {
    CHECK_NOTNULL(options);
    if (options->n16rot_model_path.empty()) {
        options->n16rot_model_path = DefaultAlikedExtractorModelPath("n16rot");
    }
    if (options->n32_model_path.empty()) {
        // The N32 extractor model is not published in the shared catalog yet.
        options->n32_model_path = DefaultAlikedExtractorModelPath("n32");
    }
}

bool ProvisionDefaultAlikedModels(AlikedExtractionOptions* options) {
    CHECK_NOTNULL(options);
    const std::string default_n16rot =
            DefaultAlikedExtractorModelPath("n16rot");
    const std::string default_n32 = DefaultAlikedExtractorModelPath("n32");
    ResolveDefaultAlikedModelPaths(options);
    if (!default_n16rot.empty() &&
        options->n16rot_model_path == default_n16rot &&
        !EnsureCachedModel(FindExtractorModel("n16rot", ""),
                           options->n16rot_model_path, "ALIKED")) {
        return false;
    }
    if (!default_n32.empty() && options->n32_model_path == default_n32 &&
        !EnsureCachedModel(FindExtractorModel("n32", ""),
                           options->n32_model_path, "ALIKED")) {
        return false;
    }
    return true;  // Caller-provided paths are never re-provisioned.
}

std::unique_ptr<FeatureExtractor> CreateAlikedFeatureExtractor(
        const FeatureExtractionOptions& options) {
    return std::make_unique<AlikedFeatureExtractor>(options);
}

std::unique_ptr<FeatureMatcher> CreateAlikedFeatureMatcher(
        const FeatureMatchingOptions& options) {
    THROW_CHECK_NOTNULL(options.aliked);
    switch (options.type) {
        case FeatureMatcherType::ALIKED_BRUTEFORCE:
            // ALIKED emits unit-norm descriptors; no re-normalization.
            return CreateBruteForceGGMLFeatureMatcher(
                    options, options.aliked->brute_force,
                    /*normalize_descriptors=*/false);
        case FeatureMatcherType::ALIKED_LIGHTGLUE:
            return CreateLightGlueGGMLFeatureMatcher(options,
                                                     options.aliked->lightglue);
        default:
            throw std::runtime_error("Unknown ALIKED feature matcher type.");
    }
}

}  // namespace colmap
