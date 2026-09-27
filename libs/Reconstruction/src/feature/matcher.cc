// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "feature/matcher.h"

#include "feature/aliked.h"
#include "feature/extractor.h"
#include "util/logging.h"
#include "util/misc.h"

#include <cstring>

#include <sstream>
#include <stdexcept>

namespace colmap {
namespace {

void ThrowUnknownFeatureMatcherType(FeatureMatcherType type) {
  std::ostringstream error;
  error << "Unknown feature matcher type: " << type;
  throw std::runtime_error(error.str());
}

}  // namespace

FeatureMatchingOptions::FeatureMatchingOptions(FeatureMatcherType type)
    : type(type) {
  // Upstream parity (FeatureMatchingTypeOptions): every option group is
  // allocated regardless of the selected type, so callers may switch `type`
  // freely and mutate any group (see feature/matcher_test.cc).
  sift = std::make_shared<SiftMatchingOptions>();
  aliked = std::make_shared<AlikedMatchingOptions>();
  loma = std::make_shared<LomaMatchingOptions>();
}

FeatureMatchingOptions::FeatureMatchingOptions(
        const FeatureMatchingOptions& other)
    : type(other.type),
      num_threads(other.num_threads),
      use_gpu(other.use_gpu),
      gpu_index(other.gpu_index),
      max_num_matches(other.max_num_matches),
      guided_matching(other.guided_matching),
      skip_geometric_verification(other.skip_geometric_verification),
      rig_verification(other.rig_verification),
      skip_image_pairs_in_same_frame(other.skip_image_pairs_in_same_frame),
      sift(CloneSharedPtr(other.sift)),
      loma(CloneSharedPtr(other.loma)),
      aliked(CloneSharedPtr(other.aliked)) {}

FeatureMatchingOptions& FeatureMatchingOptions::operator=(
        const FeatureMatchingOptions& other) {
  if (this == &other) {
    return *this;
  }
  type = other.type;
  num_threads = other.num_threads;
  use_gpu = other.use_gpu;
  gpu_index = other.gpu_index;
  max_num_matches = other.max_num_matches;
  guided_matching = other.guided_matching;
  skip_geometric_verification = other.skip_geometric_verification;
  rig_verification = other.rig_verification;
  skip_image_pairs_in_same_frame = other.skip_image_pairs_in_same_frame;
  sift = CloneSharedPtr(other.sift);
  loma = CloneSharedPtr(other.loma);
  aliked = CloneSharedPtr(other.aliked);
  return *this;
}

bool FeatureMatchingOptions::RequiresOpenGL() const {
  switch (type) {
    case FeatureMatcherType::SIFT_BRUTEFORCE:
#ifdef CUDA_ENABLED
      // The CUDA SIFT matching path does not need an OpenGL context.
      return false;
#else
      return use_gpu;
#endif
    case FeatureMatcherType::SIFT_LIGHTGLUE:
    case FeatureMatcherType::ALIKED_BRUTEFORCE:
    case FeatureMatcherType::ALIKED_LIGHTGLUE:
    case FeatureMatcherType::LOMA_BRUTEFORCE:
    case FeatureMatcherType::LOMA_B:
    case FeatureMatcherType::LOMA_B128:
    case FeatureMatcherType::LOMA_R:
    case FeatureMatcherType::LOMA_L:
    case FeatureMatcherType::LOMA_G:
      return false;
    default:
      ThrowUnknownFeatureMatcherType(type);
  }
  return false;
}

bool FeatureMatchingOptions::Check() const {
  if (type == FeatureMatcherType::SIFT_BRUTEFORCE ||
      type == FeatureMatcherType::SIFT_LIGHTGLUE) {
    THROW_CHECK_NOTNULL(sift);
    return sift->Check();
  }
  if (type == FeatureMatcherType::ALIKED_BRUTEFORCE ||
      type == FeatureMatcherType::ALIKED_LIGHTGLUE) {
    THROW_CHECK_NOTNULL(aliked);
    return aliked->Check();
  }
  THROW_CHECK_NOTNULL(loma);
  // Default-constructed LoMa options carry an empty matcher model path by
  // design; resolve the catalog-default path (no download) before
  // validating, mirroring the MatchLomaFeatures production flow.
  LomaMatchingOptions resolved = *loma;
  ResolveDefaultLomaModelPaths(&resolved);
  return resolved.Check();
}

namespace {

// Fork feature types are plain Eigen matrices; the packed uint8 <-> float
// conversions mirror the binding-layer wrapper semantics (4 bytes per float,
// row-major memcpy).
FeatureDescriptors ToPackedU8(const FeatureDescriptorsFloat& src) {
    const long rows = src.rows();
    const long cols = src.cols();
    FeatureDescriptors dst = FeatureDescriptors::Zero(rows, cols * 4);
    std::memcpy(dst.data(), src.data(), sizeof(float) * rows * cols);
    return dst;
}

FeatureDescriptorsFloat FromPackedU8(const FeatureDescriptors& src) {
    const long rows = src.rows();
    const long cols = src.cols() / 4;
    FeatureDescriptorsFloat dst = FeatureDescriptorsFloat::Zero(rows, cols);
    std::memcpy(dst.data(), src.data(), sizeof(float) * rows * cols);
    return dst;
}

class SiftFeatureMatcherShell : public FeatureMatcher {
 public:
  explicit SiftFeatureMatcherShell(
          std::shared_ptr<SiftMatchingOptions> options)
      : options_(std::move(options)) {}

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    // The fork's in-memory SIFT matching consumes packed uint8 descriptors.
    const FeatureDescriptors desc1 = ToPackedU8(image1.descriptors);
    const FeatureDescriptors desc2 = ToPackedU8(image2.descriptors);
    MatchSiftFeaturesCPU(*options_, desc1, desc2, matches);
    if (matches->size() > static_cast<size_t>(options_->max_num_matches)) {
      matches->resize(options_->max_num_matches);
    }
  }

  void MatchGuided(double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    auto options = *options_;
    options.max_error = max_error;
    const FeatureDescriptors desc1 = ToPackedU8(image1.descriptors);
    const FeatureDescriptors desc2 = ToPackedU8(image2.descriptors);
    MatchGuidedSiftFeaturesCPU(options, image1.keypoints, image2.keypoints,
                               desc1, desc2, two_view_geometry);
  }

 private:
  std::shared_ptr<SiftMatchingOptions> options_;
};

class LomaFeatureMatcherShell : public FeatureMatcher {
 public:
  explicit LomaFeatureMatcherShell(
          std::shared_ptr<LomaMatchingOptions> options)
      : options_(std::move(options)) {}

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK(image1.camera && image2.camera);
    const Camera& camera1 = *image1.camera;
    const Camera& camera2 = *image2.camera;
    MatchLomaFeatures(*options_, image1.keypoints, image1.descriptors,
                      image2.keypoints, image2.descriptors,
                      static_cast<int>(camera1.Width()),
                      static_cast<int>(camera1.Height()), static_cast<int>(camera2.Width()),
                      static_cast<int>(camera2.Height()), matches);
  }

  void MatchGuided(double, const Image&, const Image&,
                   TwoViewGeometry*) override {
    throw std::runtime_error(
            "Guided matching is not supported by the LoMa matcher");
  }

 private:
  std::shared_ptr<LomaMatchingOptions> options_;
};

}  // namespace

std::unique_ptr<FeatureMatcher> FeatureMatcher::Create(
        const FeatureMatchingOptions& options) {
  THROW_CHECK(options.Check());
  if (options.type == FeatureMatcherType::SIFT_BRUTEFORCE ||
      options.type == FeatureMatcherType::SIFT_LIGHTGLUE) {
    return std::make_unique<SiftFeatureMatcherShell>(options.sift);
  }
  if (options.type == FeatureMatcherType::ALIKED_BRUTEFORCE ||
      options.type == FeatureMatcherType::ALIKED_LIGHTGLUE) {
    // Upstream parity: the factory constructs the engine-backed matcher
    // (upstream: onnxruntime model load; fork: AICore ggml LightGlue task
    // or direct brute-force cosine matching). Provisioning failures throw
    // at construction time.
    return CreateAlikedFeatureMatcher(options);
  }
  return std::make_unique<LomaFeatureMatcherShell>(options.loma);
}

}  // namespace colmap
