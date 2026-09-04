// Real-image DaD gate. export_loma_detector_reference.py obtains the fixture
// from the pinned upstream ONNX graph; this executable never imports ONNX.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "aicore/loma_capi.h"

namespace {

constexpr char kMagic[] = "LOMDAD1";

template <typename T>
bool Read(std::ifstream& stream, T* value) {
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(value), sizeof(*value)));
}

struct Reference {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t count = 0;
    std::vector<uint8_t> rgb;
    std::vector<aicore_loma_keypoint> keypoints;
    std::vector<float> scores;
};

bool LoadReference(const char* path, Reference* reference) {
    std::ifstream stream(path, std::ios::binary);
    char magic[sizeof(kMagic)]{};
    uint32_t version = 0;
    uint32_t rgb_bytes = 0;
    if (!stream.read(magic, sizeof(magic)) ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
        !Read(stream, &version) || !Read(stream, &reference->width) ||
        !Read(stream, &reference->height) || !Read(stream, &reference->count) ||
        !Read(stream, &rgb_bytes) || version != 1 || reference->width < 8 ||
        reference->height < 8 || reference->count == 0 ||
        rgb_bytes != reference->width * reference->height * 3) {
        return false;
    }
    reference->rgb.resize(rgb_bytes);
    reference->keypoints.resize(reference->count);
    reference->scores.resize(reference->count);
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(reference->rgb.data()),
                                         reference->rgb.size())) &&
           static_cast<bool>(stream.read(reinterpret_cast<char*>(reference->keypoints.data()),
                                         reference->keypoints.size() * sizeof(aicore_loma_keypoint))) &&
           static_cast<bool>(stream.read(reinterpret_cast<char*>(reference->scores.data()),
                                         reference->scores.size() * sizeof(float))) &&
           stream.peek() == std::ifstream::traits_type::eof();
}

}  // namespace

int main() {
    const char* model = std::getenv("AICORE_TEST_LOMA_DETECTOR_GGUF");
    const char* fixture = std::getenv("AICORE_TEST_LOMA_DETECTOR_REFERENCE");
    if (model == nullptr || fixture == nullptr || model[0] == '\0' || fixture[0] == '\0') {
        std::fprintf(stderr, "SKIP: set AICORE_TEST_LOMA_DETECTOR_GGUF and "
                             "AICORE_TEST_LOMA_DETECTOR_REFERENCE\n");
        return 77;
    }
    Reference reference;
    if (!LoadReference(fixture, &reference)) {
        std::fprintf(stderr, "invalid DaD ONNX-reference fixture\n");
        return 1;
    }
    aicore_loma_detector_options* options = aicore_loma_detector_options_new();
    aicore_loma_detector_options_set_device(options, "cpu");
    aicore_loma_detector_options_set_max_keypoints(options,
                                                    static_cast<int32_t>(reference.count));
    aicore_loma_detector_ctx* detector = aicore_loma_detector_load(model, options);
    aicore_loma_detector_options_free(options);
    if (!aicore_loma_detector_is_ready(detector)) {
        std::fprintf(stderr, "DaD load failed: %s\n",
                     aicore_loma_detector_last_error(detector));
        aicore_loma_detector_free(detector);
        return 1;
    }
    const aicore_loma_rgb_image image = {reference.rgb.data(),
                                         static_cast<int32_t>(reference.width),
                                         static_cast<int32_t>(reference.height),
                                         static_cast<int32_t>(reference.width * 3)};
    aicore_loma_detected_features output{};
    const int rc = aicore_loma_detector_run(detector, &image, &output);
    if (rc != 0 || output.count != static_cast<int32_t>(reference.count)) {
        std::fprintf(stderr, "DaD run failed: count=%d error=%s\n", output.count,
                     aicore_loma_detector_last_error(detector));
        aicore_loma_detected_features_free(&output);
        aicore_loma_detector_free(detector);
        return 1;
    }
    size_t true_positive = 0;
    float max_coordinate_error = 0.0f;
    float max_score_error = 0.0f;
    for (int32_t index = 0; index < output.count; ++index) {
        const float dx = output.keypoints[index].x - reference.keypoints[index].x;
        const float dy = output.keypoints[index].y - reference.keypoints[index].y;
        const float coordinate_error = std::sqrt(dx * dx + dy * dy);
        max_coordinate_error = std::max(max_coordinate_error, coordinate_error);
        max_score_error = std::max(max_score_error,
                std::abs(output.scores[index] - reference.scores[index]));
        if (coordinate_error <= 0.02f) ++true_positive;
    }
    const double precision = static_cast<double>(true_positive) / output.count;
    const double recall = static_cast<double>(true_positive) / reference.count;
    std::printf("{\"suite\":\"loma-dad-reference\",\"expected\":%u,"
                "\"predicted\":%d,\"tp\":%zu,\"precision\":%.8f,"
                "\"recall\":%.8f,\"max_coordinate_error\":%.8g,"
                "\"max_score_error\":%.8g}\n", reference.count, output.count,
                true_positive, precision, recall, max_coordinate_error, max_score_error);
    aicore_loma_detected_features_free(&output);
    aicore_loma_detector_free(detector);
    if (precision < 0.995 || recall < 0.995 || max_coordinate_error > 0.02f ||
        max_score_error > 1e-5f) {
        std::fprintf(stderr, "DaD real-image precision/recall gate failed\n");
        return 1;
    }
    return 0;
}
