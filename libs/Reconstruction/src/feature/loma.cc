#include "feature/loma.h"

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
#include "base/database.h"
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
                                   FeatureDescriptorsFloat* descriptors) {
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
        Eigen::Map<const FeatureDescriptorsFloat> mapped(
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

aicore_loma_model_variant DescriptorVariant(const FeatureDescriptorType type) {
    return type == FeatureDescriptorType::kLomaB128
                   ? AICORE_LOMA_MODEL_VARIANT_DEDODE_B
                   : AICORE_LOMA_MODEL_VARIANT_DEDODE_G;
}

aicore_loma_model_variant MatcherVariant(const LomaMatcherVariant variant) {
    switch (variant) {
        case LomaMatcherVariant::kB128:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_B128;
        case LomaMatcherVariant::kR:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_R;
        case LomaMatcherVariant::kL:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_L;
        case LomaMatcherVariant::kG:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_G;
        case LomaMatcherVariant::kB:
            return AICORE_LOMA_MODEL_VARIANT_MATCHER_B;
    }
    return AICORE_LOMA_MODEL_VARIANT_MATCHER_B;
}

void ResolveDefaultLomaModelPaths(LomaExtractionOptions* options) {
    CHECK_NOTNULL(options);
    if (options->detector_model_path.empty()) {
        options->detector_model_path =
                DefaultLomaModelPath(AICORE_LOMA_MODEL_ROLE_DETECTOR);
    }
    if (options->descriptor_model_path.empty()) {
        options->descriptor_model_path = DefaultLomaModelPath(
                DescriptorVariant(options->descriptor_type));
    }
}

void ResolveDefaultLomaModelPaths(LomaMatchingOptions* options) {
    CHECK_NOTNULL(options);
    if (options->matcher_model_path.empty()) {
        options->matcher_model_path =
                DefaultLomaModelPath(MatcherVariant(options->matcher_variant));
    }
}

bool ProvisionDefaultLomaModels(LomaExtractionOptions* options) {
    CHECK_NOTNULL(options);
    const std::string default_detector_path =
            DefaultLomaModelPath(AICORE_LOMA_MODEL_ROLE_DETECTOR);
    const std::string default_descriptor_path = DefaultLomaModelPath(
            DescriptorVariant(options->descriptor_type));
    ResolveDefaultLomaModelPaths(options);
    if (options->detector_model_path == default_detector_path &&
        !EnsureCachedLomaModel(
                aicore_loma_model_by_role(AICORE_LOMA_MODEL_ROLE_DETECTOR),
                &options->detector_model_path)) {
        return false;
    }
    if (options->descriptor_model_path == default_descriptor_path &&
        !EnsureCachedLomaModel(
                aicore_loma_model_by_variant(
                        DescriptorVariant(options->descriptor_type)),
                &options->descriptor_model_path)) {
        return false;
    }
    return true;
}

bool ProvisionDefaultLomaModels(LomaMatchingOptions* options) {
    CHECK_NOTNULL(options);
    const std::string default_matcher_path =
            DefaultLomaModelPath(MatcherVariant(options->matcher_variant));
    ResolveDefaultLomaModelPaths(options);
    return options->matcher_model_path != default_matcher_path ||
           EnsureCachedLomaModel(
                   aicore_loma_model_by_variant(MatcherVariant(options->matcher_variant)),
                   &options->matcher_model_path);
}

bool ParseLomaMatcherVariant(const std::string& value,
                             LomaMatcherVariant* variant) {
    CHECK_NOTNULL(variant);
    if (value == "b") {
        *variant = LomaMatcherVariant::kB;
    } else if (value == "b128") {
        *variant = LomaMatcherVariant::kB128;
    } else if (value == "r") {
        *variant = LomaMatcherVariant::kR;
    } else if (value == "l") {
        *variant = LomaMatcherVariant::kL;
    } else if (value == "g") {
        *variant = LomaMatcherVariant::kG;
    } else {
        return false;
    }
    return true;
}

bool LomaExtractionOptions::Check() const {
    CHECK_OPTION_GT(max_num_features, 0);
    CHECK_OPTION_GE(min_score, 0.0f);
    CHECK_OPTION_LE(min_score, 1.0f);
    CHECK_OPTION(!detector_model_path.empty());
    CHECK_OPTION(!descriptor_model_path.empty());
    return true;
}

bool LomaMatchingOptions::Check() const {
    CHECK_OPTION_GE(min_score, 0.0f);
    CHECK_OPTION_LE(min_score, 1.0f);
    CHECK_OPTION(!matcher_model_path.empty());
    return true;
}

bool ExtractLomaFeatures(const LomaExtractionOptions& options,
                         const Bitmap& bitmap,
                         FeatureKeypoints* keypoints,
                         FeatureDescriptorsFloat* descriptors) {
    LomaExtractionOptions resolved_options = options;
    if (!ProvisionDefaultLomaModels(&resolved_options) ||
        !resolved_options.Check() || bitmap.Width() <= 0 ||
        bitmap.Height() <= 0) {
        return false;
    }
    aicore_loma_detector_ctx* detector = nullptr;
    aicore_loma_descriptor_ctx* descriptor = nullptr;
    if (!LoadLomaFeatureModels(resolved_options, &detector, &descriptor)) {
        return false;
    }
    const bool describe_ok = ExtractLomaFeaturesWithModels(
            bitmap, detector, descriptor, keypoints, descriptors);
    aicore_loma_detector_free(detector);
    aicore_loma_descriptor_free(descriptor);
    return describe_ok;
}

bool MatchLomaFeatures(const LomaMatchingOptions& options,
                       const FeatureKeypoints& keypoints1,
                       const FeatureDescriptorsFloat& descriptors1,
                       const FeatureKeypoints& keypoints2,
                       const FeatureDescriptorsFloat& descriptors2,
                       int image_width1,
                       int image_height1,
                       int image_width2,
                       int image_height2,
                       FeatureMatches* matches) {
    matches->clear();
    LomaMatchingOptions resolved_options = options;
    if (!ProvisionDefaultLomaModels(&resolved_options) ||
        !resolved_options.Check() ||
        descriptors1.rows() != static_cast<Eigen::Index>(keypoints1.size()) ||
        descriptors2.rows() != static_cast<Eigen::Index>(keypoints2.size()) ||
        descriptors1.cols() != descriptors2.cols()) {
        return false;
    }
    auto matcher_options = aicore_loma_matcher_options_new();
    aicore_loma_matcher_options_set_device(matcher_options,
                                           resolved_options.device.c_str());
    aicore_loma_matcher_options_set_min_score(matcher_options,
                                              resolved_options.min_score);
    auto matcher = aicore_loma_matcher_load(
            resolved_options.matcher_model_path.c_str(), matcher_options);
    aicore_loma_matcher_options_free(matcher_options);
    if (matcher == nullptr || !aicore_loma_matcher_is_ready(matcher)) {
        aicore_loma_matcher_free(matcher);
        return false;
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
    aicore_loma_matcher_free(matcher);
    return ok;
}

LomaFeatureExtractor::LomaFeatureExtractor(
        const ImageReaderOptions& reader_options,
        const LomaExtractionOptions& loma_options)
    : reader_options_(reader_options), loma_options_(loma_options) {
    CHECK(reader_options_.Check());
    CHECK(ProvisionDefaultLomaModels(&loma_options_));
    CHECK(loma_options_.Check());
}

void LomaFeatureExtractor::Run() {
    succeeded_ = false;
    Database database(reader_options_.database_path);
    ImageReader reader(reader_options_, &database);
    PrintHeading1("LoMa feature extraction");
    aicore_loma_detector_ctx* detector = nullptr;
    aicore_loma_descriptor_ctx* descriptor = nullptr;
    if (!LoadLomaFeatureModels(loma_options_, &detector, &descriptor)) {
        std::cerr << "ERROR: LoMa model load failed (detector="
                  << loma_options_.detector_model_path
                  << ", descriptor=" << loma_options_.descriptor_model_path
                  << ")" << std::endl;
        return;
    }
    succeeded_ = true;
    while (reader.NextIndex() < reader.NumImages() && !IsStopped()) {
        Camera camera;
        Image image;
        Bitmap bitmap;
        Bitmap mask;
        const auto status = reader.Next(&camera, &image, &bitmap, &mask);
        if (status == ImageReader::Status::IMAGE_EXISTS) continue;
        if (status != ImageReader::Status::SUCCESS) {
            std::cerr << "ERROR: LoMa could not read " << image.Name()
                      << std::endl;
            succeeded_ = false;
            continue;
        }
        FeatureKeypoints keypoints;
        FeatureDescriptorsFloat descriptors;
        if (!ExtractLomaFeaturesWithModels(bitmap, detector, descriptor,
                                           &keypoints, &descriptors)) {
            std::cerr << "ERROR: LoMa extraction failed for " << image.Name()
                      << std::endl;
            succeeded_ = false;
            continue;
        }
        DatabaseTransaction transaction(&database);
        if (image.ImageId() == kInvalidImageId)
            image.SetImageId(database.WriteImage(image));
        database.WriteKeypoints(image.ImageId(), keypoints);
        database.WriteFloatDescriptors(image.ImageId(), descriptors,
                                       loma_options_.descriptor_type);
    }
    aicore_loma_detector_free(detector);
    aicore_loma_descriptor_free(descriptor);
}

}  // namespace colmap
