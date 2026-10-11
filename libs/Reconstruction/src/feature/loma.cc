// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "feature/loma.h"

#include <sstream>
#include <stdexcept>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <memory>
#include <unordered_set>
#include <vector>

#include "aicore/asset_digests.h"
#include "aicore/loma_capi.h"
#include "feature/utils.h"
#include "util/logging.h"
#include "util/download.h"
#include "util/misc.h"

namespace colmap {
namespace {

bool MakeRgbInput(const Bitmap& source,
                  Bitmap* rgb,
                  aicore_loma_rgb_image* image) {
    *rgb = source.IsRGB() ? source.Clone() : source.CloneAsRGB();
    image->rgb = rgb->GetScanline(0);
    image->width = rgb->Width();
    image->height = rgb->Height();
    image->row_stride_bytes = static_cast<int32_t>(rgb->ScanWidth());
    return image->rgb != nullptr;
}

std::mutex& LomaModelCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_set<std::string>& VerifiedLomaModels() {
    static std::unordered_set<std::string> paths;
    return paths;
}

bool VerifyLomaModel(const std::filesystem::path& path,
                     const char* expected_digest) {
#ifdef COLMAP_DOWNLOAD_ENABLED
    return expected_digest != nullptr && std::filesystem::is_regular_file(path) &&
           ComputeFileSHA256(path) == expected_digest;
#else
    (void)expected_digest;
    return std::filesystem::is_regular_file(path);
#endif
}

bool EnsureCachedLomaModel(const aicore_loma_model_entry* entry,
                           std::string* model_path) {
    if (entry == nullptr || entry->filename == nullptr ||
        entry->download_url == nullptr || model_path == nullptr) {
        return false;
    }
    const char* expected_digest = aicore::AssetDigestForFile(entry->filename);
    if (expected_digest == nullptr) {
        std::cerr << "ERROR: LoMa catalog entry has no SHA-256: "
                  << entry->filename << std::endl;
        return false;
    }
    char* cache_dir = aicore_loma_model_cache_dir();
    if (cache_dir == nullptr) return false;
    const std::string cache_path =
            (std::filesystem::path(cache_dir) / entry->filename).string();
    std::free(cache_dir);

    std::lock_guard<std::mutex> lock(LomaModelCacheMutex());
    if (VerifiedLomaModels().count(cache_path) != 0) {
        *model_path = cache_path;
        return true;
    }
    const std::filesystem::path target(cache_path);
    if (VerifyLomaModel(target, expected_digest)) {
        VerifiedLomaModels().insert(cache_path);
        *model_path = cache_path;
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
    if (downloaded.empty() || !VerifyLomaModel(temporary, expected_digest)) {
        std::filesystem::remove(temporary, error);
        std::cerr << "ERROR: LoMa model download or SHA-256 verification failed: "
                  << entry->filename << std::endl;
        return false;
    }
    std::filesystem::rename(temporary, target, error);
    if (error) {
        // A concurrent process may have published the same verified model.
        if (!VerifyLomaModel(target, expected_digest)) {
            std::filesystem::remove(temporary, error);
            std::cerr << "ERROR: cannot publish LoMa model in shared cache: "
                      << target << " (" << error.message() << ")" << std::endl;
            return false;
        }
        std::filesystem::remove(temporary, error);
    }
    VerifiedLomaModels().insert(cache_path);
    *model_path = cache_path;
    return true;
#else
    std::cerr << "ERROR: LoMa model is absent from shared cache and reconstruction "
                 "was built with DOWNLOAD_ENABLED=OFF: "
              << target << std::endl;
    return false;
#endif
}

bool MakeDescriptorInput(const Bitmap& source,
                         Bitmap* resized,
                         aicore_loma_rgb_image* image) {
    *resized = source.Clone();
    if (resized->Width() != 784 || resized->Height() != 784) {
        resized->Rescale(784, 784);
    }
    image->rgb = resized->GetScanline(0);
    image->width = resized->Width();
    image->height = resized->Height();
    image->row_stride_bytes = static_cast<int32_t>(resized->ScanWidth());
    return image->rgb != nullptr;
}

void ConvertKeypoints(const aicore_loma_detected_features& detected,
                      FeatureKeypoints* keypoints) {
    keypoints->resize(static_cast<size_t>(detected.count));
    for (int i = 0; i < detected.count; ++i) {
        (*keypoints)[static_cast<size_t>(i)] = FeatureKeypoint(
                detected.keypoints[i].x, detected.keypoints[i].y);
    }
}

bool LoadLomaFeatureModels(const LomaExtractionOptions& options,
                           aicore_loma_detector_ctx** detector,
                           aicore_loma_descriptor_ctx** descriptor) {
    CHECK_NOTNULL(detector);
    CHECK_NOTNULL(descriptor);
    *detector = nullptr;
    *descriptor = nullptr;
    auto detector_options = aicore_loma_detector_options_new();
    auto descriptor_options = aicore_loma_descriptor_options_new();
    if (detector_options == nullptr || descriptor_options == nullptr) {
        aicore_loma_detector_options_free(detector_options);
        aicore_loma_descriptor_options_free(descriptor_options);
        return false;
    }
    aicore_loma_detector_options_set_device(detector_options,
                                            options.device.c_str());
    aicore_loma_detector_options_set_max_keypoints(detector_options,
                                                   options.max_num_features);
    aicore_loma_descriptor_options_set_device(descriptor_options,
                                              options.device.c_str());
    *detector = aicore_loma_detector_load(options.detector_model_path.c_str(),
                                          detector_options);
    *descriptor = aicore_loma_descriptor_load(
            options.descriptor_model_path.c_str(), descriptor_options);
    aicore_loma_detector_options_free(detector_options);
    aicore_loma_descriptor_options_free(descriptor_options);
    if (*detector == nullptr || *descriptor == nullptr ||
        !aicore_loma_detector_is_ready(*detector) ||
        !aicore_loma_descriptor_is_ready(*descriptor)) {
        const char* detector_error = *detector != nullptr
                ? aicore_loma_detector_last_error(*detector) : "allocation failed";
        const char* descriptor_error = *descriptor != nullptr
                ? aicore_loma_descriptor_last_error(*descriptor) : "allocation failed";
        std::cerr << "ERROR: LoMa model initialization failed: detector="
                  << (detector_error != nullptr ? detector_error : "unknown")
                  << ", descriptor="
                  << (descriptor_error != nullptr ? descriptor_error : "unknown")
                  << std::endl;
        aicore_loma_detector_free(*detector);
        aicore_loma_descriptor_free(*descriptor);
        *detector = nullptr;
        *descriptor = nullptr;
        return false;
    }
    return true;
}

bool ExtractLomaFeaturesWithModels(const Bitmap& bitmap,
                                   aicore_loma_detector_ctx* detector,
                                   aicore_loma_descriptor_ctx* descriptor,
                                   FeatureKeypoints* keypoints,
                                   FeatureDescriptorsFloatData* descriptors) {
    CHECK_NOTNULL(keypoints);
    CHECK_NOTNULL(descriptors);
    if (detector == nullptr || descriptor == nullptr || bitmap.Width() <= 0 ||
        bitmap.Height() <= 0) {
        return false;
    }
    Bitmap source_rgb;
    Bitmap descriptor_rgb;
    aicore_loma_rgb_image detector_input;
    aicore_loma_rgb_image descriptor_input;
    if (!MakeRgbInput(bitmap, &source_rgb, &detector_input) ||
        !MakeDescriptorInput(source_rgb, &descriptor_rgb, &descriptor_input)) {
        return false;
    }
    aicore_loma_detected_features detected{};
    const bool detect_ok =
            aicore_loma_detector_run(detector, &detector_input, &detected) == 0;
    if (!detect_ok) {
        aicore_loma_detected_features_free(&detected);
        return false;
    }
    ConvertKeypoints(detected, keypoints);
    aicore_loma_described_features described{};
    const bool describe_ok =
            aicore_loma_descriptor_run(descriptor, &descriptor_input,
                                       detected.keypoints, detected.count,
                                       bitmap.Width(), bitmap.Height(),
                                       &described) == 0;
    if (describe_ok) {
        Eigen::Map<const FeatureDescriptorsFloatData> mapped(
                described.descriptors, described.count,
                described.descriptor_dim);
        *descriptors = mapped;
    } else {
        keypoints->clear();
        descriptors->resize(0, 0);
    }
    aicore_loma_described_features_free(&described);
    aicore_loma_detected_features_free(&detected);
    return describe_ok;
}

}  // namespace

std::string DefaultLomaModelPath(const aicore_loma_model_role role) {
    const aicore_loma_model_entry* entry = aicore_loma_model_by_role(role);
    if (entry == nullptr || entry->filename == nullptr) return "";
    char* cache_dir = aicore_loma_model_cache_dir();
    if (cache_dir == nullptr) return "";
    const std::filesystem::path model_path =
            std::filesystem::path(cache_dir) / entry->filename;
    std::free(cache_dir);
    return model_path.string();
}

std::string DefaultLomaModelPath(const aicore_loma_model_variant variant) {
    const aicore_loma_model_entry* entry =
            aicore_loma_model_by_variant(variant);
    char* cache_dir = aicore_loma_model_cache_dir();
    if (entry == nullptr || entry->filename == nullptr ||
        cache_dir == nullptr) {
        std::free(cache_dir);
        return "";
    }
    const std::filesystem::path model_path =
            std::filesystem::path(cache_dir) / entry->filename;
    std::free(cache_dir);
    return model_path.string();
}

aicore_loma_model_variant LomaDescriptorVariant(const FeatureExtractorType type) {
    switch (type) {
        case FeatureExtractorType::LOMA_B128:
            return AICORE_LOMA_MODEL_VARIANT_DEDODE_B;
        case FeatureExtractorType::LOMA_B:
            return AICORE_LOMA_MODEL_VARIANT_DEDODE_G;
        default:
            LOG_FATAL_THROW(std::runtime_error)
                    << "Not a LoMa feature extractor type: " << type;
    }
    return AICORE_LOMA_MODEL_VARIANT_DEDODE_G;
}

aicore_loma_model_variant LomaMatcherVariant(const FeatureMatcherType type) {
    switch (type) {
        case FeatureMatcherType::LOMA_B:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_B;
        case FeatureMatcherType::LOMA_B128:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_B128;
        case FeatureMatcherType::LOMA_R:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_R;
        case FeatureMatcherType::LOMA_L:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_L;
        case FeatureMatcherType::LOMA_G:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_G;
        default:
            LOG_FATAL_THROW(std::runtime_error)
                    << "Not a dedicated LoMa matcher type: " << type;
    }
    return AICORE_LOMA_MODEL_VARIANT_MATCHER_B;
}

namespace {

std::string& DescriptorModelPath(const FeatureExtractorType type,
                                 LomaExtractionOptions* options) {
    return type == FeatureExtractorType::LOMA_B128
                   ? options->descriptor_b128_model_path
                   : options->descriptor_model_path;
}

std::string& MatcherModelPath(const FeatureMatcherType type,
                              LomaMatchingOptions* options) {
    switch (type) {
        case FeatureMatcherType::LOMA_B128:
            return options->b128.model_path;
        case FeatureMatcherType::LOMA_R:
            return options->r.model_path;
        case FeatureMatcherType::LOMA_L:
            return options->l.model_path;
        case FeatureMatcherType::LOMA_G:
            return options->g.model_path;
        default:
            return options->b.model_path;
    }
}

}  // namespace

void ResolveDefaultLomaModelPaths(const FeatureExtractorType type,
                                  LomaExtractionOptions* options) {
    CHECK_NOTNULL(options);
    if (options->detector_model_path.empty()) {
        options->detector_model_path =
                DefaultLomaModelPath(AICORE_LOMA_MODEL_ROLE_DETECTOR);
    }
    std::string& descriptor_path = DescriptorModelPath(type, options);
    if (descriptor_path.empty()) {
        descriptor_path = DefaultLomaModelPath(LomaDescriptorVariant(type));
    }
}

void ResolveDefaultLomaModelPaths(const FeatureMatcherType type,
                                  LomaMatchingOptions* options) {
    CHECK_NOTNULL(options);
    if (type == FeatureMatcherType::LOMA_BRUTEFORCE) {
        return;
    }
    std::string& matcher_path = MatcherModelPath(type, options);
    if (matcher_path.empty()) {
        matcher_path = DefaultLomaModelPath(LomaMatcherVariant(type));
    }
}

bool ProvisionDefaultLomaModels(const FeatureExtractorType type,
                                LomaExtractionOptions* options) {
    CHECK_NOTNULL(options);
    const aicore_loma_model_variant descriptor_variant =
            LomaDescriptorVariant(type);
    const std::string default_detector_path =
            DefaultLomaModelPath(AICORE_LOMA_MODEL_ROLE_DETECTOR);
    const std::string default_descriptor_path =
            DefaultLomaModelPath(descriptor_variant);
    ResolveDefaultLomaModelPaths(type, options);
    if (options->detector_model_path == default_detector_path &&
        !EnsureCachedLomaModel(
                aicore_loma_model_by_role(AICORE_LOMA_MODEL_ROLE_DETECTOR),
                &options->detector_model_path)) {
        return false;
    }
    std::string& descriptor_path = DescriptorModelPath(type, options);
    if (descriptor_path == default_descriptor_path &&
        !EnsureCachedLomaModel(aicore_loma_model_by_variant(descriptor_variant),
                               &descriptor_path)) {
        return false;
    }
    return true;
}

bool ProvisionDefaultLomaModels(const FeatureMatcherType type,
                                LomaMatchingOptions* options) {
    CHECK_NOTNULL(options);
    if (type == FeatureMatcherType::LOMA_BRUTEFORCE) {
        return true;
    }
    const aicore_loma_model_variant matcher_variant = LomaMatcherVariant(type);
    const std::string default_matcher_path =
            DefaultLomaModelPath(matcher_variant);
    ResolveDefaultLomaModelPaths(type, options);
    std::string& matcher_path = MatcherModelPath(type, options);
    return matcher_path != default_matcher_path ||
           EnsureCachedLomaModel(aicore_loma_model_by_variant(matcher_variant),
                                 &matcher_path);
}

bool LomaExtractionOptions::Check() const {
    CHECK_OPTION_GT(max_num_features, 0);
    CHECK_OPTION_GE(min_score, 0.0);
    CHECK_OPTION_LE(min_score, 1.0);
    return true;
}

bool LomaMatchingOptions::Check() const {
    CHECK_OPTION_GE(min_score, 0.0);
    CHECK_OPTION_LE(min_score, 1.0);
    return brute_force.Check();
}

namespace {

bool LoadResolvedLomaFeatureModels(const FeatureExtractorType type,
                                   const LomaExtractionOptions& options,
                                   aicore_loma_detector_ctx** detector,
                                   aicore_loma_descriptor_ctx** descriptor) {
    LomaExtractionOptions resolved = options;
    if (!ProvisionDefaultLomaModels(type, &resolved) || !resolved.Check()) {
        return false;
    }
    // The internal loader reads descriptor_model_path; point it at the
    // descriptor the extractor type consumes.
    resolved.descriptor_model_path = DescriptorModelPath(type, &resolved);
    if (resolved.detector_model_path.empty() ||
        resolved.descriptor_model_path.empty()) {
        return false;
    }
    return LoadLomaFeatureModels(resolved, detector, descriptor);
}

aicore_loma_matcher_ctx* LoadResolvedLomaMatcher(
        const FeatureMatcherType type, const LomaMatchingOptions& options) {
    LomaMatchingOptions resolved = options;
    if (!ProvisionDefaultLomaModels(type, &resolved) || !resolved.Check()) {
        return nullptr;
    }
    const std::string& model_path = MatcherModelPath(type, &resolved);
    if (model_path.empty()) {
        return nullptr;
    }
    auto matcher_options = aicore_loma_matcher_options_new();
    aicore_loma_matcher_options_set_device(matcher_options,
                                           resolved.device.c_str());
    aicore_loma_matcher_options_set_min_score(matcher_options,
                                              resolved.min_score);
    auto matcher = aicore_loma_matcher_load(model_path.c_str(), matcher_options);
    aicore_loma_matcher_options_free(matcher_options);
    if (matcher == nullptr || !aicore_loma_matcher_is_ready(matcher)) {
        aicore_loma_matcher_free(matcher);
        return nullptr;
    }
    return matcher;
}

bool RunLomaMatcher(aicore_loma_matcher_ctx* matcher,
                    const FeatureKeypoints& keypoints1,
                    const FeatureDescriptorsFloatData& descriptors1,
                    const FeatureKeypoints& keypoints2,
                    const FeatureDescriptorsFloatData& descriptors2,
                    const int image_width1,
                    const int image_height1,
                    const int image_width2,
                    const int image_height2,
                    FeatureMatches* matches) {
    matches->clear();
    if (descriptors1.rows() != static_cast<Eigen::Index>(keypoints1.size()) ||
        descriptors2.rows() != static_cast<Eigen::Index>(keypoints2.size()) ||
        descriptors1.cols() != descriptors2.cols()) {
        return false;
    }
    if (keypoints1.empty() || keypoints2.empty()) {
        return true;
    }
    std::vector<aicore_loma_keypoint> kp1(keypoints1.size()),
            kp2(keypoints2.size());
    for (size_t i = 0; i < kp1.size(); ++i)
        kp1[i] = {keypoints1[i].x, keypoints1[i].y};
    for (size_t i = 0; i < kp2.size(); ++i)
        kp2[i] = {keypoints2[i].x, keypoints2[i].y};
    aicore_loma_features f1{
            kp1.data(),          static_cast<int32_t>(kp1.size()),
            descriptors1.data(), static_cast<int32_t>(descriptors1.cols()),
            image_width1,        image_height1};
    aicore_loma_features f2{
            kp2.data(),          static_cast<int32_t>(kp2.size()),
            descriptors2.data(), static_cast<int32_t>(descriptors2.cols()),
            image_width2,        image_height2};
    aicore_loma_match* raw_matches = nullptr;
    int32_t count = 0;
    const bool ok = aicore_loma_matcher_run(matcher, &f1, &f2, &raw_matches,
                                            &count) == 0;
    if (ok) {
        matches->reserve(static_cast<size_t>(count));
        for (int32_t i = 0; i < count; ++i) {
            matches->emplace_back(static_cast<point2D_t>(raw_matches[i].idx0),
                                  static_cast<point2D_t>(raw_matches[i].idx1));
        }
    }
    aicore_loma_free_matches(raw_matches);
    return ok;
}

class LomaFeatureExtractorShell : public FeatureExtractor {
public:
    explicit LomaFeatureExtractorShell(const FeatureExtractionOptions& options)
        : options_(options) {
        THROW_CHECK_NOTNULL(options_.loma);
        if (!LoadResolvedLomaFeatureModels(options_.type, *options_.loma,
                                           &detector_, &descriptor_)) {
            throw std::runtime_error(
                    "LoMa extractor model provisioning or initialization "
                    "failed");
        }
    }

    ~LomaFeatureExtractorShell() override {
        aicore_loma_detector_free(detector_);
        aicore_loma_descriptor_free(descriptor_);
    }

    bool Extract(const Bitmap& bitmap,
                 FeatureKeypoints* keypoints,
                 FeatureDescriptors* descriptors) override {
        THROW_CHECK_NOTNULL(keypoints);
        THROW_CHECK_NOTNULL(descriptors);
        descriptors->type = options_.type;
        descriptors->data.resize(0, 0);
        FeatureDescriptorsFloat descriptors_float(
                options_.type, FeatureDescriptorsFloatData());
        if (bitmap.Width() <= 0 || bitmap.Height() <= 0 ||
            !ExtractLomaFeaturesWithModels(bitmap, detector_, descriptor_,
                                           keypoints, &descriptors_float.data)) {
            return false;
        }
        *descriptors = descriptors_float.ToBytes();
        return true;
    }

private:
    const FeatureExtractionOptions options_;
    aicore_loma_detector_ctx* detector_ = nullptr;
    aicore_loma_descriptor_ctx* descriptor_ = nullptr;
};

class LomaFeatureMatcherShell : public FeatureMatcher {
public:
    explicit LomaFeatureMatcherShell(const FeatureMatchingOptions& options)
        : options_(options) {
        THROW_CHECK_NOTNULL(options_.loma);
        THROW_CHECK(options_.loma->Check());
        matcher_ = LoadResolvedLomaMatcher(options_.type, *options_.loma);
        if (matcher_ == nullptr) {
            throw std::runtime_error(
                    "LoMa matcher model provisioning or initialization "
                    "failed");
        }
    }

    ~LomaFeatureMatcherShell() override { aicore_loma_matcher_free(matcher_); }

    void Match(const Image& image1,
               const Image& image2,
               FeatureMatches* matches) override {
        THROW_CHECK_NOTNULL(matches);
        THROW_CHECK_NOTNULL(image1.keypoints);
        THROW_CHECK_NOTNULL(image2.keypoints);
        THROW_CHECK_NOTNULL(image1.descriptors);
        THROW_CHECK_NOTNULL(image2.descriptors);
        const FeatureDescriptorsFloat descriptors1 =
                image1.descriptors->ToFloat();
        const FeatureDescriptorsFloat descriptors2 =
                image2.descriptors->ToFloat();
        THROW_CHECK_NOTNULL(image1.camera);
        THROW_CHECK_NOTNULL(image2.camera);
        THROW_CHECK(RunLomaMatcher(
                matcher_, *image1.keypoints, descriptors1.data,
                *image2.keypoints, descriptors2.data,
                static_cast<int>(image1.camera->Width()),
                static_cast<int>(image1.camera->Height()),
                static_cast<int>(image2.camera->Width()),
                static_cast<int>(image2.camera->Height()), matches));
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
                "Guided matching is not supported by the LoMa matcher");
    }

private:
    const FeatureMatchingOptions options_;
    aicore_loma_matcher_ctx* matcher_ = nullptr;
};

}  // namespace

std::unique_ptr<FeatureExtractor> CreateLomaFeatureExtractor(
        const FeatureExtractionOptions& options) {
    return std::make_unique<LomaFeatureExtractorShell>(options);
}

std::unique_ptr<FeatureMatcher> CreateLomaFeatureMatcher(
        const FeatureMatchingOptions& options) {
    THROW_CHECK_NOTNULL(options.loma);
    if (options.type == FeatureMatcherType::LOMA_BRUTEFORCE) {
        // DeDoDe descriptors are not unit-norm; normalize before cosine.
        return CreateBruteForceGGMLFeatureMatcher(
                options, options.loma->brute_force,
                /*normalize_descriptors=*/true);
    }
    return std::make_unique<LomaFeatureMatcherShell>(options);
}

}  // namespace colmap
