// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "feature/lightglue.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "aicore/lightglue_capi.h"
#include "feature/model_cache.h"
#include "feature/utils.h"
#include "geometry/pose_prior.h"
#include "util/logging.h"

namespace colmap {
namespace {

const aicore_model_entry* FindLightGlueCatalogModel(
        const LightGlueFeatureKind kind, const std::string& quant) {
  const int matcher_type = static_cast<int>(kind);
  return FindCatalogModel(AICORE_MODEL_FAMILY_LIGHTGLUE,
                          [&](const aicore_model_entry* entry) {
    if (entry->matcher_type != matcher_type) return false;
    if (quant.empty()) return true;
    const std::string filename(entry->filename);
    const std::string tag = "-" + quant + ".gguf";
    return filename.size() > tag.size() &&
           filename.compare(filename.size() - tag.size(), tag.size(), tag) ==
                   0;
  });
}

const char* KindName(const LightGlueFeatureKind kind) {
  return kind == LightGlueFeatureKind::SIFT ? "SIFT LightGlue"
                                            : "ALIKED LightGlue";
}

class LightGlueGGMLFeatureMatcher : public FeatureMatcher {
 public:
  LightGlueGGMLFeatureMatcher(const FeatureMatchingOptions& options,
                              const LightGlueGGMLMatchingOptions& lg_options)
      : options_(options),
        kind_(options.type == FeatureMatcherType::SIFT_LIGHTGLUE
                      ? LightGlueFeatureKind::SIFT
                      : LightGlueFeatureKind::ALIKED) {
    THROW_CHECK(options_.type == FeatureMatcherType::SIFT_LIGHTGLUE ||
                options_.type == FeatureMatcherType::ALIKED_LIGHTGLUE);
    THROW_CHECK(lg_options.Check());

    LightGlueGGMLMatchingOptions resolved = lg_options;
    // use_gpu is the upstream-facing placement switch; the fork's device
    // string only takes effect when the caller allows GPU placement.
    if (!options.use_gpu) resolved.device = "cpu";
    if (!ProvisionDefaultLightGlueModel(kind_, &resolved)) {
      throw std::runtime_error(std::string(KindName(kind_)) +
                               " matcher model provisioning failed");
    }
    if (resolved.model_path.empty()) {
      throw std::runtime_error(std::string(KindName(kind_)) +
                               " matcher model is not published in the "
                               "shared catalog");
    }

    aicore_lightglue_options* engine_options = aicore_lightglue_options_new();
    if (engine_options == nullptr) {
      throw std::runtime_error("LightGlue options allocation failed");
    }
    aicore_lightglue_options_set_device(engine_options,
                                        resolved.device.c_str());
    aicore_lightglue_options_set_min_score(engine_options, resolved.min_score);
    aicore_lightglue_options_set_matcher_type(engine_options,
                                              static_cast<int>(kind_));
    ctx_ = aicore_lightglue_load_opts(resolved.model_path.c_str(),
                                      engine_options);
    aicore_lightglue_options_free(engine_options);
    if (ctx_ == nullptr || !aicore_lightglue_is_ready(ctx_)) {
      const char* error = ctx_ != nullptr ? aicore_lightglue_last_error(ctx_)
                                          : "allocation failed";
      const std::string message =
              std::string(KindName(kind_)) +
              " matcher initialization failed: " +
              (error != nullptr ? error : "unknown");
      aicore_lightglue_free(ctx_);
      ctx_ = nullptr;
      throw std::runtime_error(message);
    }
  }

  ~LightGlueGGMLFeatureMatcher() override { aicore_lightglue_free(ctx_); }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    matches->clear();
    THROW_CHECK_NOTNULL(image1.keypoints);
    THROW_CHECK_NOTNULL(image2.keypoints);
    THROW_CHECK_NOTNULL(image1.descriptors);
    THROW_CHECK_NOTNULL(image2.descriptors);

    const Features f1 = PrepareFeatures(image1);
    const Features f2 = PrepareFeatures(image2);
    if (f1.keypoints.empty() || f2.keypoints.empty()) {
      return;
    }

    aicore_lightglue_match* raw_matches = nullptr;
    int32_t count = 0;
    const int rc = aicore_lightglue_run_match(
            ctx_, &f1.features, &f2.features, &raw_matches, &count);
    if (rc != 0) {
      const char* error = aicore_lightglue_last_error(ctx_);
      aicore_lightglue_free_matches(raw_matches);
      LOG_FATAL_THROW(std::runtime_error)
          << KindName(kind_) << " matching failed: "
                       << (error != nullptr ? error : "unknown");
    }
    matches->reserve(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
      matches->emplace_back(static_cast<point2D_t>(raw_matches[i].idx1),
                            static_cast<point2D_t>(raw_matches[i].idx2));
    }
    aicore_lightglue_free_matches(raw_matches);
    if (options_.max_num_matches > 0 &&
        matches->size() > static_cast<size_t>(options_.max_num_matches)) {
      matches->resize(options_.max_num_matches);
    }
  }

  void MatchGuided(double,
                   const Image&,
                   const Image&,
                   TwoViewGeometry*) override {
    throw std::runtime_error(
            "Guided matching is not supported by the LightGlue matcher");
  }

 private:
  struct Features {
    std::vector<aicore_lightglue_keypoint> keypoints;
    FeatureDescriptorsFloatData descriptors;
    aicore_lightglue_features features{};
  };

  Features PrepareFeatures(const Image& image) const {
    Features out;
    // Upstream parity (a2bd15c2 feature/lightglue.cc FeaturesFromImage): a
    // gravity prior rotates the keypoints upright before matching, and the
    // reported image size follows the rotation.
    const int rot90 =
            (image.pose_prior != nullptr && image.pose_prior->HasGravity())
                    ? ComputeRot90FromGravity(image.pose_prior->gravity)
                    : 0;
    const int image_width =
            image.camera != nullptr
                    ? static_cast<int>(image.camera->Width())
                    : 0;
    const int image_height =
            image.camera != nullptr
                    ? static_cast<int>(image.camera->Height())
                    : 0;
    FeatureKeypoints rotated_keypoints;
    const FeatureKeypoints* keypoints_to_use = image.keypoints.get();
    if (rot90 != 0) {
      rotated_keypoints = *image.keypoints;
      for (auto& kp : rotated_keypoints) {
        kp.Rot90(rot90, image_width, image_height);
      }
      keypoints_to_use = &rotated_keypoints;
    }
    const FeatureKeypoints& keypoints = *keypoints_to_use;
    out.keypoints.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
      const FeatureKeypoint& kp = keypoints[i];
      // COLMAP keypoints put the top-left pixel center at (0.5, 0.5); the
      // AICore LightGlue ABI uses the LightGlue/OpenCV convention (0, 0).
      out.keypoints[i] = {kp.x - 0.5f, kp.y - 0.5f, kp.ComputeScale(),
                          kp.ComputeOrientation()};
    }
    if (kind_ == LightGlueFeatureKind::SIFT) {
      // SIFT descriptors are RootSIFT vectors quantized to uint8 (scaled by
      // 512); L2-normalization recovers the unit-norm vectors the model was
      // trained on. Do not root-normalize again.
      out.descriptors = image.descriptors->ToFloat().data;
      L2NormalizeFeatureDescriptors(&out.descriptors);
    } else {
      out.descriptors = image.descriptors->ToFloat().data;
    }
    THROW_CHECK_EQ(out.descriptors.rows(),
                   static_cast<Eigen::Index>(keypoints.size()));
    out.features.keypoints = out.keypoints.data();
    out.features.n_keypoints = static_cast<int32_t>(out.keypoints.size());
    out.features.descriptors = out.descriptors.data();
    out.features.descriptor_dim = static_cast<int32_t>(out.descriptors.cols());
    const bool swap_dims = rot90 % 2 != 0;
    out.features.image_width =
            static_cast<int32_t>(swap_dims ? image_height : image_width);
    out.features.image_height =
            static_cast<int32_t>(swap_dims ? image_width : image_height);
    return out;
  }

  const FeatureMatchingOptions options_;
  const LightGlueFeatureKind kind_;
  aicore_lightglue_ctx* ctx_ = nullptr;
};

}  // namespace

bool LightGlueGGMLMatchingOptions::Check() const {
  CHECK_OPTION_GE(min_score, 0);
  CHECK_OPTION_LE(min_score, 1);
  return true;
}

std::string DefaultLightGlueModelPath(const LightGlueFeatureKind kind,
                                      const char* quant) {
  const aicore_model_entry* entry =
          FindLightGlueCatalogModel(kind, quant != nullptr ? quant : "");
  return CatalogModelPath(entry, &aicore_lightglue_model_cache_dir);
}

void ResolveDefaultLightGlueModelPath(const LightGlueFeatureKind kind,
                                      LightGlueGGMLMatchingOptions* options) {
  CHECK_NOTNULL(options);
  if (options->model_path.empty()) {
    options->model_path = DefaultLightGlueModelPath(kind);
  }
}

bool ProvisionDefaultLightGlueModel(const LightGlueFeatureKind kind,
                                    LightGlueGGMLMatchingOptions* options) {
  CHECK_NOTNULL(options);
  ResolveDefaultLightGlueModelPath(kind, options);
  if (options->model_path.empty() ||
      options->model_path != DefaultLightGlueModelPath(kind)) {
    return true;  // Caller-provided path or unpublished default.
  }
  const aicore_model_entry* entry = FindLightGlueCatalogModel(kind, "");
  if (entry == nullptr) {
    std::cerr << "ERROR: " << KindName(kind)
              << " matcher model is not published in the shared catalog"
              << std::endl;
    return false;
  }
  return EnsureCachedModel(entry, options->model_path, KindName(kind));
}

std::unique_ptr<FeatureMatcher> CreateLightGlueGGMLFeatureMatcher(
        const FeatureMatchingOptions& options,
        const LightGlueGGMLMatchingOptions& lightglue_options) {
  return std::make_unique<LightGlueGGMLFeatureMatcher>(options,
                                                       lightglue_options);
}

}  // namespace colmap
