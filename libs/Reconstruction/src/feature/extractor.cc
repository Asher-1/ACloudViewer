// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "feature/extractor.h"

#include "feature/aliked.h"
#include "feature/loma.h"
#include "feature/sift.h"
#include "util/logging.h"
#include "util/misc.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace colmap {
namespace {

void ThrowUnknownFeatureExtractorType(FeatureExtractorType type) {
  std::ostringstream error;
  error << "Unknown feature extractor type: " << type;
  throw std::runtime_error(error.str());
}

}  // namespace

FeatureExtractionOptions::FeatureExtractionOptions(FeatureExtractorType type)
    : type(type) {
  // Upstream parity (FeatureExtractionTypeOptions): every option group is
  // allocated regardless of the selected type, so callers may switch `type`
  // freely and mutate any group (see feature/extractor_test.cc).
  sift = std::make_shared<SiftExtractionOptions>();
  aliked = std::make_shared<AlikedExtractionOptions>();
  loma = std::make_shared<LomaExtractionOptions>();
}

FeatureExtractionOptions::FeatureExtractionOptions(
        const FeatureExtractionOptions& other)
    : type(other.type),
      max_image_size(other.max_image_size),
      num_threads(other.num_threads),
      use_gpu(other.use_gpu),
      gpu_index(other.gpu_index),
      sift(CloneSharedPtr(other.sift)),
      loma(CloneSharedPtr(other.loma)),
      aliked(CloneSharedPtr(other.aliked)) {}

FeatureExtractionOptions& FeatureExtractionOptions::operator=(
        const FeatureExtractionOptions& other) {
  if (this == &other) {
    return *this;
  }
  type = other.type;
  max_image_size = other.max_image_size;
  num_threads = other.num_threads;
  use_gpu = other.use_gpu;
  gpu_index = other.gpu_index;
  sift = CloneSharedPtr(other.sift);
  loma = CloneSharedPtr(other.loma);
  aliked = CloneSharedPtr(other.aliked);
  return *this;
}

bool FeatureExtractionOptions::RequiresRGB() const {
  switch (type) {
    case FeatureExtractorType::SIFT:
      return false;
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32:
    case FeatureExtractorType::LOMA_B:
    case FeatureExtractorType::LOMA_B128:
      return true;
    default:
      ThrowUnknownFeatureExtractorType(type);
  }
  return false;
}

bool FeatureExtractionOptions::RequiresOpenGL() const {
  // SiftGPU extraction requires an OpenGL context; the fork's default
  // extraction path is CPU, matching upstream's CPU-first layout.
  return false;
}

int FeatureExtractionOptions::EffMaxImageSize() const {
  if (max_image_size > 0) {
    return max_image_size;
  }
  // Upstream type defaults when max_image_size is non-positive.
  switch (type) {
    case FeatureExtractorType::SIFT:
      return 3200;
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32:
    case FeatureExtractorType::LOMA_B:
    case FeatureExtractorType::LOMA_B128:
      return 1600;
    default:
      ThrowUnknownFeatureExtractorType(type);
  }
  return 1600;
}

bool FeatureExtractionOptions::Check() const {
  if (type == FeatureExtractorType::SIFT) {
    THROW_CHECK_NOTNULL(sift);
    return sift->Check();
  }
  if (type == FeatureExtractorType::ALIKED_N16ROT ||
      type == FeatureExtractorType::ALIKED_N32) {
    THROW_CHECK_NOTNULL(aliked);
    return aliked->Check();
  }
  THROW_CHECK_NOTNULL(loma);
  // Default-constructed LoMa options carry empty model paths by design;
  // resolve the catalog-default paths (no download) before validating,
  // mirroring the ExtractLomaFeatures production flow.
  LomaExtractionOptions resolved = *loma;
  ResolveDefaultLomaModelPaths(&resolved);
  return resolved.Check();
}

namespace {

class SiftFeatureExtractorShell : public FeatureExtractor {
 public:
  explicit SiftFeatureExtractorShell(
          std::shared_ptr<SiftExtractionOptions> options)
      : options_(std::move(options)) {}

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptorsFloat* descriptors) override {
    FeatureDescriptors descriptors_u8;
    if (!ExtractSiftFeaturesCPU(*options_, bitmap, keypoints,
                                &descriptors_u8)) {
      return false;
    }
    // The fork's unified extraction surface reports float descriptors so
    // SIFT and LoMa outputs share one downstream representation (packed
    // uint8 -> float, 4 bytes per float, row-major memcpy).
    const long rows = descriptors_u8.rows();
    const long cols = descriptors_u8.cols() / 4;
    *descriptors = FeatureDescriptorsFloat::Zero(rows, cols);
    std::memcpy(descriptors->data(), descriptors_u8.data(),
                sizeof(float) * rows * cols);
    return true;
  }

 private:
  std::shared_ptr<SiftExtractionOptions> options_;
};

class LomaFeatureExtractorShell : public FeatureExtractor {
 public:
  explicit LomaFeatureExtractorShell(
          std::shared_ptr<LomaExtractionOptions> options)
      : options_(std::move(options)) {}

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptorsFloat* descriptors) override {
    return ExtractLomaFeatures(*options_, bitmap, keypoints, descriptors);
  }

 private:
  std::shared_ptr<LomaExtractionOptions> options_;
};

}  // namespace

std::unique_ptr<FeatureExtractor> FeatureExtractor::Create(
        const FeatureExtractionOptions& options) {
  THROW_CHECK(options.Check());
  if (options.type == FeatureExtractorType::SIFT) {
    return std::make_unique<SiftFeatureExtractorShell>(options.sift);
  }
  if (options.type == FeatureExtractorType::ALIKED_N16ROT ||
      options.type == FeatureExtractorType::ALIKED_N32) {
    // Upstream parity: the factory constructs the engine-backed shell
    // (upstream: onnxruntime model load; fork: AICore ggml ALIKED task).
    // Model provisioning failures throw at construction time.
    return CreateAlikedFeatureExtractor(options);
  }
  return std::make_unique<LomaFeatureExtractorShell>(options.loma);
}

}  // namespace colmap
