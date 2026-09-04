// LoMa matcher smoke test.  The model is intentionally supplied by the test
// environment: published weights remain outside the source tree.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "aicore/loma_capi.h"

namespace {

struct OwnedFeatures {
    std::vector<aicore_loma_keypoint> keypoints;
    std::vector<float> descriptors;
    aicore_loma_features view{};
};

OwnedFeatures MakeFeatures(int count, int dimension) {
    OwnedFeatures result;
    result.keypoints.resize(count);
    result.descriptors.resize(static_cast<size_t>(count) * dimension);
    for (int index = 0; index < count; ++index) {
        result.keypoints[index] = {
                20.0f + static_cast<float>((index * 47) % 600),
                15.0f + static_cast<float>((index * 31) % 450)};
        double squared_norm = 0.0;
        float* descriptor = result.descriptors.data() +
                            static_cast<size_t>(index) * dimension;
        for (int dim = 0; dim < dimension; ++dim) {
            const float value = std::sin(0.013f * (index + 1) * (dim + 3)) +
                                std::cos(0.017f * (index + 7) * (dim + 1));
            descriptor[dim] = value;
            squared_norm += static_cast<double>(value) * value;
        }
        const float inverse_norm =
                1.0f / static_cast<float>(std::sqrt(squared_norm));
        for (int dim = 0; dim < dimension; ++dim) {
            descriptor[dim] *= inverse_norm;
        }
    }
    result.view = {result.keypoints.data(), count, result.descriptors.data(),
                   dimension, 640, 480};
    return result;
}

bool ValidAndEqual(const aicore_loma_match* left,
                   int32_t left_count,
                   const aicore_loma_match* right,
                   int32_t right_count) {
    if (left_count <= 0 || left_count != right_count) return false;
    for (int32_t index = 0; index < left_count; ++index) {
        if (left[index].idx0 < 0 || left[index].idx0 >= 64 ||
            left[index].idx1 < 0 || left[index].idx1 >= 64 ||
            !std::isfinite(left[index].score) ||
            left[index].idx0 != right[index].idx0 ||
            left[index].idx1 != right[index].idx1 ||
            left[index].score != right[index].score) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const char* model = std::getenv("AICORE_TEST_LOMA_GGUF");
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    const char* dimension_env = std::getenv("AICORE_TEST_LOMA_EXPECTED_DESCRIPTOR_DIM");
    if (model == nullptr || model[0] == '\0') {
        std::fprintf(stderr, "SKIP: set AICORE_TEST_LOMA_GGUF\n");
        return 77;
    }
    if (device == nullptr || device[0] == '\0') device = "cpu";
    int descriptor_dim = 256;
    if (dimension_env != nullptr && dimension_env[0] != '\0') {
        descriptor_dim = std::atoi(dimension_env);
    }
    if (descriptor_dim != 128 && descriptor_dim != 256) {
        std::fprintf(stderr, "unsupported LoMa descriptor dimension: %d\n", descriptor_dim);
        return 2;
    }

    aicore_loma_matcher_options* options = aicore_loma_matcher_options_new();
    aicore_loma_matcher_options_set_device(options, device);
    aicore_loma_matcher_options_set_min_score(options, 0.0);
    aicore_loma_matcher_ctx* context =
            aicore_loma_matcher_load(model, options);
    aicore_loma_matcher_options_free(options);
    if (!aicore_loma_matcher_is_ready(context)) {
        std::fprintf(stderr, "LoMa load failed: %s\n",
                     aicore_loma_matcher_last_error(context));
        aicore_loma_matcher_free(context);
        return 1;
    }

    OwnedFeatures features = MakeFeatures(64, descriptor_dim);
    aicore_loma_match* first = nullptr;
    aicore_loma_match* second = nullptr;
    int32_t first_count = 0;
    int32_t second_count = 0;
    const int first_rc = aicore_loma_matcher_run(
            context, &features.view, &features.view, &first, &first_count);
    const int second_rc = aicore_loma_matcher_run(
            context, &features.view, &features.view, &second, &second_count);
    const bool valid = first_rc == 0 && second_rc == 0 &&
                       ValidAndEqual(first, first_count, second, second_count);
    if (!valid) {
        std::fprintf(stderr, "LoMa inference failed: count=%d/%d error=%s\n",
                     first_count, second_count,
                     aicore_loma_matcher_last_error(context));
    }
    aicore_loma_free_matches(first);
    aicore_loma_free_matches(second);
    aicore_loma_matcher_free(context);
    return valid ? 0 : 1;
}
