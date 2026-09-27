// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Fork wiring of the upstream ALIKED feature extractor / matcher to the
// AICore ggml task family:
//   - extraction runs through aicore_aliked_* (tasks/aliked, ggml CNN);
//   - LightGlue matching runs through aicore_lightglue_* (tasks/lightglue,
//     ALIKED geometry);
//   - brute-force matching is direct Eigen cosine similarity with the same
//     min_cossim / max_ratio / cross_check semantics as upstream's
//     brute-force ONNX network;
//   - model defaults resolve through the shared aicore model catalog and
//     download into the AICore shared cache with pinned SHA-256 checks,
//     mirroring feature/loma.cc.

#include "feature/aliked.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "aicore/aliked_capi.h"
#include "aicore/asset_digests.h"
#include "aicore/model_catalog_capi.h"
#include "feature/extractor.h"
#include "feature/matcher.h"
#include "util/download.h"
#include "util/logging.h"

namespace colmap {
namespace {

using aicore::AssetDigestForFile;

// ---------------------------------------------------------------------------
// Shared catalog default resolution
// ---------------------------------------------------------------------------

const aicore_model_entry* FindCatalogModel(
        const aicore_model_family family,
        const std::function<bool(const aicore_model_entry*)>& matches) {
    const int count = aicore_model_count(family);
    for (int index = 0; index < count; ++index) {
        const aicore_model_entry* entry = aicore_model_at(family, index);
        if (entry != nullptr && entry->filename != nullptr && matches(entry)) {
            return entry;
        }
    }
    return nullptr;
}

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

const aicore_model_entry* FindLightGlueModel(const int matcher_type,
                                             const std::string& quant) {
    return FindCatalogModel(AICORE_MODEL_FAMILY_LIGHTGLUE,
                            [&](const aicore_model_entry* entry) {
        if (entry->matcher_type != matcher_type) return false;
        if (quant.empty()) return true;
        const std::string filename(entry->filename);
        const std::string tag = "-" + quant + ".gguf";
        return filename.size() > tag.size() &&
               filename.compare(filename.size() - tag.size(), tag.size(),
                                tag) == 0;
    });
}

std::string CatalogModelPath(const aicore_model_entry* entry,
                             char* (*cache_dir_fn)(void)) {
    if (entry == nullptr || entry->filename == nullptr) return "";
    char* cache_dir = cache_dir_fn();
    if (cache_dir == nullptr) return "";
    std::string path =
            (std::filesystem::path(cache_dir) / entry->filename).string();
    std::free(cache_dir);
    return path;
}

// ---------------------------------------------------------------------------
// Shared-cache provisioning (mirror of feature/loma.cc)
// ---------------------------------------------------------------------------

std::mutex& ModelCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_set<std::string>& VerifiedModels() {
    static std::unordered_set<std::string> paths;
    return paths;
}

bool VerifyModelFile(const std::filesystem::path& path,
                     const char* expected_digest) {
#ifdef COLMAP_DOWNLOAD_ENABLED
    return expected_digest != nullptr &&
           std::filesystem::is_regular_file(path) &&
           ComputeFileSHA256(path) == expected_digest;
#else
    (void)expected_digest;
    return std::filesystem::is_regular_file(path);
#endif
}

bool EnsureCachedModel(const aicore_model_entry* entry,
                       const std::filesystem::path& target,
                       const std::string& engine_name) {
    if (entry == nullptr || entry->filename == nullptr ||
        entry->download_url == nullptr) {
        return false;
    }
    const char* expected_digest = AssetDigestForFile(entry->filename);
    if (expected_digest == nullptr) {
        std::cerr << "ERROR: " << engine_name
                  << " catalog entry has no SHA-256: " << entry->filename
                  << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(ModelCacheMutex());
    if (VerifiedModels().count(target.string()) != 0) {
        return true;
    }
    if (VerifyModelFile(target, expected_digest)) {
        VerifiedModels().insert(target.string());
        return true;
    }
#ifdef COLMAP_DOWNLOAD_ENABLED
    std::error_code error;
    std::filesystem::remove(target, error);
    const std::filesystem::path temporary =
            target.string() + ".partial." +
            std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count());
    const std::string downloaded =
            DownloadAndCacheFile(entry->download_url, temporary);
    if (downloaded.empty() || !VerifyModelFile(temporary, expected_digest)) {
        std::filesystem::remove(temporary, error);
        std::cerr << "ERROR: " << engine_name
                  << " model download or SHA-256 verification failed: "
                  << entry->filename << std::endl;
        return false;
    }
    std::filesystem::rename(temporary, target, error);
    if (error) {
        // A concurrent process may have published the same verified model.
        if (!VerifyModelFile(target, expected_digest)) {
            std::filesystem::remove(temporary, error);
            std::cerr << "ERROR: cannot publish " << engine_name
                      << " model in shared cache: " << target << " ("
                      << error.message() << ")" << std::endl;
            return false;
        }
        std::filesystem::remove(temporary, error);
    }
    VerifiedModels().insert(target.string());
    return true;
#else
    std::cerr << "ERROR: " << engine_name
              << " model is absent from shared cache and reconstruction was "
                 "built with DOWNLOAD_ENABLED=OFF: "
              << target << std::endl;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

Bitmap MakeRgbBitmap(const Bitmap& source) {
    return source.IsRGB() ? source.Clone() : source.CloneAsRGB();
}

void ConvertLightGlueFeatures(const aicore_lightglue_features& features,
                              FeatureKeypoints* keypoints,
                              FeatureDescriptorsFloat* descriptors) {
    const int count = features.n_keypoints;
    keypoints->resize(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        // ALIKED keypoints are pixel coordinates without scale/orientation.
        (*keypoints)[static_cast<size_t>(i)] = FeatureKeypoint(
                features.keypoints[i].x, features.keypoints[i].y);
    }
    if (count > 0 && features.descriptors != nullptr &&
        features.descriptor_dim > 0) {
        Eigen::Map<const FeatureDescriptorsFloat> mapped(
                features.descriptors, features.n_keypoints,
                features.descriptor_dim);
        *descriptors = mapped;
    } else {
        descriptors->resize(0, 0);
    }
}

void FreeLightGlueFeatures(aicore_lightglue_features* features) {
    if (features == nullptr) return;
    aicore_aliked_free_buffer(features->keypoints);
    aicore_aliked_free_buffer(features->descriptors);
    features->keypoints = nullptr;
    features->descriptors = nullptr;
}

bool LoadAlikedExtractor(const AlikedExtractionOptions& options,
                         const char* model_path,
                         aicore_aliked_ctx** ctx) {
    auto engine_options = aicore_aliked_options_new();
    if (engine_options == nullptr) return false;
    aicore_aliked_options_set_device(engine_options, options.device.c_str());
    aicore_aliked_options_set_max_keypoints(engine_options,
                                            options.max_num_features);
    *ctx = aicore_aliked_load_opts(model_path, engine_options);
    aicore_aliked_options_free(engine_options);
    if (*ctx == nullptr || !aicore_aliked_is_ready(*ctx)) {
        const char* error =
                *ctx != nullptr ? aicore_aliked_last_error(*ctx)
                                : "allocation failed";
        std::cerr << "ERROR: ALIKED extractor initialization failed: "
                  << (error != nullptr ? error : "unknown") << std::endl;
        aicore_aliked_free(*ctx);
        *ctx = nullptr;
        return false;
    }
    return true;
}

bool ExtractWithCtx(aicore_aliked_ctx* ctx,
                    const Bitmap& bitmap,
                    FeatureKeypoints* keypoints,
                    FeatureDescriptorsFloat* descriptors) {
    if (ctx == nullptr || bitmap.Width() <= 0 || bitmap.Height() <= 0) {
        return false;
    }
    const Bitmap rgb = MakeRgbBitmap(bitmap);
    aicore_image_view view{};
    view.data = rgb.GetScanline(0);
    view.width = rgb.Width();
    view.height = rgb.Height();
    view.row_stride_bytes = static_cast<size_t>(rgb.ScanWidth());
    view.format = AICORE_IMAGE_RGB8;
    if (view.data == nullptr) return false;
    aicore_lightglue_features features{};
    const bool ok =
            aicore_aliked_extract_image_view(ctx, &view, &features) == 0;
    if (ok) {
        ConvertLightGlueFeatures(features, keypoints, descriptors);
    } else {
        const char* error = aicore_aliked_last_error(ctx);
        std::cerr << "ERROR: ALIKED extraction failed: "
                  << (error != nullptr ? error : "unknown") << std::endl;
    }
    FreeLightGlueFeatures(&features);
    return ok;
}

// ---------------------------------------------------------------------------
// Brute-force matching (Eigen cosine similarity; upstream brute-force ONNX
// network semantics: mutual NN + min_cossim threshold + second-best/best
// similarity ratio + cross check)
// ---------------------------------------------------------------------------

FeatureDescriptorsFloat RowNormalized(
        const FeatureDescriptorsFloat& descriptors) {
    FeatureDescriptorsFloat normalized = descriptors;
    std::vector<Eigen::Index> keep;
    keep.reserve(normalized.rows());
    for (Eigen::Index i = 0; i < normalized.rows(); ++i) {
        const float norm = normalized.row(i).norm();
        if (norm > 0.0f) {
            normalized.row(i) /= norm;
            keep.push_back(i);
        }
    }
    return normalized;
}

}  // namespace

std::string DefaultAlikedExtractorModelPath(const char* role,
                                            const char* quant) {
    const aicore_model_entry* entry =
            FindExtractorModel(role != nullptr ? role : "", quant != nullptr
                                                               ? quant
                                                               : "");
    return CatalogModelPath(entry, &aicore_aliked_model_cache_dir);
}

std::string DefaultAlikedLightGlueModelPath(const char* quant) {
    const aicore_model_entry* entry = FindLightGlueModel(
            /*matcher_type=*/2, quant != nullptr ? quant : "");
    return CatalogModelPath(entry, &aicore_lightglue_model_cache_dir);
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

void ResolveDefaultAlikedModelPaths(AlikedMatchingOptions* options) {
    CHECK_NOTNULL(options);
    if (options->lightglue_model_path.empty()) {
        options->lightglue_model_path = DefaultAlikedLightGlueModelPath();
    }
}

bool ProvisionDefaultAlikedModels(AlikedExtractionOptions* options) {
    CHECK_NOTNULL(options);
    const std::string default_n16rot =
            DefaultAlikedExtractorModelPath("n16rot");
    const std::string default_n32 = DefaultAlikedExtractorModelPath("n32");
    ResolveDefaultAlikedModelPaths(options);
    if (!default_n16rot.empty() &&
        options->n16rot_model_path == default_n16rot) {
        return EnsureCachedModel(FindExtractorModel("n16rot", ""),
                                 options->n16rot_model_path, "ALIKED");
    }
    if (!default_n32.empty() && options->n32_model_path == default_n32) {
        return EnsureCachedModel(FindExtractorModel("n32", ""),
                                 options->n32_model_path, "ALIKED");
    }
    return true;  // Caller-provided paths are never re-provisioned.
}

bool ProvisionDefaultAlikedModels(AlikedMatchingOptions* options) {
    CHECK_NOTNULL(options);
    ResolveDefaultAlikedModelPaths(options);
    if (options->lightglue_model_path.empty() ||
        options->lightglue_model_path !=
                DefaultAlikedLightGlueModelPath()) {
        return true;  // Caller-provided path or unpublished default.
    }
    const aicore_model_entry* entry = FindLightGlueModel(/*matcher_type=*/2, "");
    if (entry == nullptr) {
        std::cerr << "ERROR: ALIKED LightGlue matcher model is not published "
                     "in the shared catalog"
                  << std::endl;
        return false;
    }
    return EnsureCachedModel(entry, options->lightglue_model_path,
                             "ALIKED LightGlue");
}

bool AlikedExtractionOptions::Check() const {
    CHECK_OPTION_GT(max_num_features, 0);
    CHECK_OPTION_GE(min_score, 0);
    CHECK_OPTION_LE(min_score, 1);
    return true;
}

bool AlikedMatchingOptions::Check() const {
    CHECK_OPTION_IN(min_cossim, -1, 1);
    CHECK_OPTION_IN(max_ratio, 0, 1);
    CHECK_OPTION_GE(min_score, 0);
    CHECK_OPTION_LE(min_score, 1);
    return true;
}

bool ExtractAlikedFeatures(const AlikedExtractionOptions& options,
                           const Bitmap& bitmap,
                           FeatureKeypoints* keypoints,
                           FeatureDescriptorsFloat* descriptors) {
    CHECK_NOTNULL(keypoints);
    CHECK_NOTNULL(descriptors);
    keypoints->clear();
    *descriptors = FeatureDescriptorsFloat();
    if (bitmap.Width() <= 0 || bitmap.Height() <= 0 || !options.Check()) {
        return false;
    }
    AlikedExtractionOptions resolved = options;
    if (!ProvisionDefaultAlikedModels(&resolved)) {
        return false;
    }
    const bool use_n32 = !resolved.n32_model_path.empty();
    const std::string& model_path =
            use_n32 ? resolved.n32_model_path : resolved.n16rot_model_path;
    if (model_path.empty()) {
        std::cerr << "ERROR: ALIKED extractor model is not published in the "
                     "shared catalog"
                  << std::endl;
        return false;
    }
    aicore_aliked_ctx* ctx = nullptr;
    if (!LoadAlikedExtractor(resolved, model_path.c_str(), &ctx)) {
        return false;
    }
    const bool ok =
            ExtractWithCtx(ctx, bitmap, keypoints, descriptors);
    aicore_aliked_free(ctx);
    return ok;
}

bool MatchAlikedFeaturesBruteForce(
        const AlikedMatchingOptions& options,
        const FeatureKeypoints& keypoints1,
        const FeatureDescriptorsFloat& descriptors1,
        const FeatureKeypoints& keypoints2,
        const FeatureDescriptorsFloat& descriptors2,
        FeatureMatches* matches) {
    CHECK_NOTNULL(matches);
    if (!options.Check() ||
        descriptors1.rows() != static_cast<Eigen::Index>(keypoints1.size()) ||
        descriptors2.rows() != static_cast<Eigen::Index>(keypoints2.size()) ||
        descriptors1.cols() != descriptors2.cols()) {
        return false;
    }
    matches->clear();
    // Upstream's brute-force ONNX graph requires at least two descriptors
    // per image for its ratio branch; mirror that floor for parity and
    // return an empty match set instead of an error.
    if (descriptors1.rows() < 2 || descriptors2.rows() < 2) {
        return true;
    }

    // O(K1*K2*D) similarity; O(K1*log K2) with a partial sort would only pay
    // off when K1*K2 dominates, which it already does here.
    const FeatureDescriptorsFloat d1 = RowNormalized(descriptors1);
    const FeatureDescriptorsFloat d2 = RowNormalized(descriptors2);
    const Eigen::MatrixXf similarity = d1 * d2.transpose();

    const Eigen::Index num1 = similarity.rows();
    const Eigen::Index num2 = similarity.cols();
    std::vector<Eigen::Index> best1(static_cast<size_t>(num1), -1);
    std::vector<float> best_score1(static_cast<size_t>(num1), -2.0f);
    std::vector<float> second_score1(static_cast<size_t>(num1), -2.0f);
    for (Eigen::Index i = 0; i < num1; ++i) {
        for (Eigen::Index j = 0; j < num2; ++j) {
            const float value = similarity(i, j);
            if (value > best_score1[static_cast<size_t>(i)]) {
                second_score1[static_cast<size_t>(i)] =
                        best_score1[static_cast<size_t>(i)];
                best_score1[static_cast<size_t>(i)] = value;
                best1[static_cast<size_t>(i)] = j;
            } else if (value > second_score1[static_cast<size_t>(i)]) {
                second_score1[static_cast<size_t>(i)] = value;
            }
        }
    }

    std::vector<Eigen::Index> best2(static_cast<size_t>(num2), -1);
    std::vector<float> best_score2(static_cast<size_t>(num2), -2.0f);
    for (Eigen::Index i = 0; i < num1; ++i) {
        const Eigen::Index j = best1[static_cast<size_t>(i)];
        if (j >= 0 &&
            best_score1[static_cast<size_t>(i)] >
                    best_score2[static_cast<size_t>(j)]) {
            best_score2[static_cast<size_t>(j)] =
                    best_score1[static_cast<size_t>(i)];
            best2[static_cast<size_t>(j)] = i;
        }
    }

    for (Eigen::Index i = 0; i < num1; ++i) {
        const Eigen::Index j = best1[static_cast<size_t>(i)];
        if (j < 0) continue;
        const float score = best_score1[static_cast<size_t>(i)];
        if (score < static_cast<float>(options.min_cossim)) continue;
        // Upstream brute-force ONNX graph semantics: the ratio test is
        // disabled for max_ratio <= 0 or >= 1, otherwise Lowe's distance
        // ratio (1 - sim_second) / (1 - sim_best) must not exceed it.
        if (options.max_ratio > 0.0 && options.max_ratio < 1.0) {
            const float second = second_score1[static_cast<size_t>(i)];
            const float dist_best = 1.0f - score;
            const float dist_second = 1.0f - second;
            if (dist_best > 0.0f &&
                dist_second / dist_best >
                        static_cast<float>(options.max_ratio)) {
                continue;
            }
        }
        if (options.cross_check &&
            best2[static_cast<size_t>(j)] != i) {
            continue;
        }
        matches->emplace_back(static_cast<point2D_t>(i),
                              static_cast<point2D_t>(j));
    }
    return true;
}

bool MatchAlikedFeaturesLightGlue(
        const AlikedMatchingOptions& options,
        const FeatureKeypoints& keypoints1,
        const FeatureDescriptorsFloat& descriptors1,
        int image_width1,
        int image_height1,
        const FeatureKeypoints& keypoints2,
        const FeatureDescriptorsFloat& descriptors2,
        int image_width2,
        int image_height2,
        FeatureMatches* matches) {
    CHECK_NOTNULL(matches);
    matches->clear();
    if (!options.Check() ||
        descriptors1.rows() != static_cast<Eigen::Index>(keypoints1.size()) ||
        descriptors2.rows() != static_cast<Eigen::Index>(keypoints2.size())) {
        return false;
    }
    AlikedMatchingOptions resolved = options;
    if (!ProvisionDefaultAlikedModels(&resolved) ||
        resolved.lightglue_model_path.empty()) {
        return false;
    }
    auto engine_options = aicore_lightglue_options_new();
    if (engine_options == nullptr) return false;
    aicore_lightglue_options_set_device(engine_options,
                                        resolved.device.c_str());
    aicore_lightglue_options_set_min_score(engine_options,
                                           resolved.min_score);
    aicore_lightglue_options_set_matcher_type(
            engine_options, /*matcher_type=*/2);
    aicore_lightglue_ctx* ctx = aicore_lightglue_load_opts(
            resolved.lightglue_model_path.c_str(), engine_options);
    aicore_lightglue_options_free(engine_options);
    if (ctx == nullptr || !aicore_lightglue_is_ready(ctx)) {
        const char* error =
                ctx != nullptr ? aicore_lightglue_last_error(ctx)
                               : "allocation failed";
        std::cerr << "ERROR: ALIKED LightGlue matcher initialization failed: "
                  << (error != nullptr ? error : "unknown") << std::endl;
        aicore_lightglue_free(ctx);
        return false;
    }

    const auto to_keypoints =
            [](const FeatureKeypoints& keypoints) {
        std::vector<aicore_lightglue_keypoint> out(keypoints.size());
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = {keypoints[i].x, keypoints[i].y, 0.0f, 0.0f};
        }
        return out;
    };
    const std::vector<aicore_lightglue_keypoint> kp1 = to_keypoints(keypoints1);
    const std::vector<aicore_lightglue_keypoint> kp2 = to_keypoints(keypoints2);
    aicore_lightglue_features f1{const_cast<aicore_lightglue_keypoint*>(
                                         kp1.data()),
                                 static_cast<int32_t>(kp1.size()),
                                 const_cast<float*>(descriptors1.data()),
                                 static_cast<int32_t>(descriptors1.cols()),
                                 image_width1, image_height1};
    aicore_lightglue_features f2{const_cast<aicore_lightglue_keypoint*>(
                                         kp2.data()),
                                 static_cast<int32_t>(kp2.size()),
                                 const_cast<float*>(descriptors2.data()),
                                 static_cast<int32_t>(descriptors2.cols()),
                                 image_width2, image_height2};
    aicore_lightglue_match* raw_matches = nullptr;
    int32_t count = 0;
    const bool ok = aicore_lightglue_run_match(ctx, &f1, &f2, &raw_matches,
                                               &count) == 0;
    if (ok) {
        matches->reserve(static_cast<size_t>(count));
        for (int32_t i = 0; i < count; ++i) {
            matches->emplace_back(
                    static_cast<point2D_t>(raw_matches[i].idx1),
                    static_cast<point2D_t>(raw_matches[i].idx2));
        }
    } else {
        const char* error = aicore_lightglue_last_error(ctx);
        std::cerr << "ERROR: ALIKED LightGlue matching failed: "
                  << (error != nullptr ? error : "unknown") << std::endl;
    }
    aicore_lightglue_free_matches(raw_matches);
    aicore_lightglue_free(ctx);
    return ok;
}

namespace {

class AlikedFeatureExtractorShell : public FeatureExtractor {
 public:
  explicit AlikedFeatureExtractorShell(
          const FeatureExtractionOptions& options)
      : options_(options) {
    THROW_CHECK_NOTNULL(options_.aliked);
    AlikedExtractionOptions resolved = *options_.aliked;
    if (!ProvisionDefaultAlikedModels(&resolved)) {
      throw std::runtime_error(
              "ALIKED extractor model provisioning failed");
    }
    const bool use_n32 =
            options_.type == FeatureExtractorType::ALIKED_N32;
    const std::string& model_path =
            use_n32 ? resolved.n32_model_path : resolved.n16rot_model_path;
    if (model_path.empty()) {
      throw std::runtime_error(
              "ALIKED N32 extractor model is not published in the shared "
              "catalog");
    }
    if (!LoadAlikedExtractor(resolved, model_path.c_str(), &ctx_)) {
      throw std::runtime_error(
              "ALIKED extractor initialization failed");
    }
  }

  ~AlikedFeatureExtractorShell() override { aicore_aliked_free(ctx_); }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptorsFloat* descriptors) override {
    THROW_CHECK_NOTNULL(keypoints);
    THROW_CHECK_NOTNULL(descriptors);
    return ExtractWithCtx(ctx_, bitmap, keypoints, descriptors);
  }

 private:
  FeatureExtractionOptions options_;
  aicore_aliked_ctx* ctx_ = nullptr;
};

class AlikedFeatureMatcherShell : public FeatureMatcher {
 public:
  AlikedFeatureMatcherShell(FeatureMatcherType type,
                            std::shared_ptr<AlikedMatchingOptions> options)
      : type_(type), options_(std::move(options)) {
    THROW_CHECK_NOTNULL(options_);
    if (type_ == FeatureMatcherType::ALIKED_LIGHTGLUE) {
      AlikedMatchingOptions resolved = *options_;
      if (!ProvisionDefaultAlikedModels(&resolved)) {
        throw std::runtime_error(
                "ALIKED LightGlue matcher model provisioning failed");
      }
      if (resolved.lightglue_model_path.empty()) {
        throw std::runtime_error(
                "ALIKED LightGlue matcher model is not published in the "
                "shared catalog");
      }
      lightglue_model_path_ = resolved.lightglue_model_path;
      device_ = resolved.device;
      min_score_ = resolved.min_score;
    }
  }

  void Match(const FeatureMatcher::Image& image1,
             const FeatureMatcher::Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    if (type_ == FeatureMatcherType::ALIKED_BRUTEFORCE) {
      THROW_CHECK(MatchAlikedFeaturesBruteForce(
              *options_, image1.keypoints, image1.descriptors,
              image2.keypoints, image2.descriptors, matches));
      return;
    }
    int width1 = 0;
    int height1 = 0;
    int width2 = 0;
    int height2 = 0;
    if (image1.camera) {
      width1 = static_cast<int>(image1.camera->Width());
      height1 = static_cast<int>(image1.camera->Height());
    }
    if (image2.camera) {
      width2 = static_cast<int>(image2.camera->Width());
      height2 = static_cast<int>(image2.camera->Height());
    }
    AlikedMatchingOptions options;
    options.min_cossim = options_->min_cossim;
    options.max_ratio = options_->max_ratio;
    options.cross_check = options_->cross_check;
    options.min_score = min_score_;
    options.lightglue_model_path = lightglue_model_path_;
    options.device = device_;
    THROW_CHECK(MatchAlikedFeaturesLightGlue(
            options, image1.keypoints, image1.descriptors, width1, height1,
            image2.keypoints, image2.descriptors, width2, height2, matches));
  }

  void MatchGuided(double, const FeatureMatcher::Image&,
                   const FeatureMatcher::Image&,
                   TwoViewGeometry*) override {
    throw std::runtime_error(
            "Guided matching is not supported by the ALIKED matcher");
  }

 private:
  FeatureMatcherType type_;
  std::shared_ptr<AlikedMatchingOptions> options_;
  std::string lightglue_model_path_;
  std::string device_ = "auto";
  double min_score_ = 0.1;
};

}  // namespace

std::unique_ptr<FeatureExtractor> CreateAlikedFeatureExtractor(
        const FeatureExtractionOptions& options) {
  return std::make_unique<AlikedFeatureExtractorShell>(options);
}

std::unique_ptr<FeatureMatcher> CreateAlikedFeatureMatcher(
        const FeatureMatchingOptions& options) {
  if (options.type != FeatureMatcherType::ALIKED_BRUTEFORCE &&
      options.type != FeatureMatcherType::ALIKED_LIGHTGLUE) {
    throw std::runtime_error("Unknown ALIKED feature matcher type.");
  }
  return std::make_unique<AlikedFeatureMatcherShell>(
          options.type, options.aliked);
}

}  // namespace colmap
