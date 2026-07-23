// C API implementation for LightGlue feature matching.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/lightglue_capi.h"
#include "path_util.hpp"
#include "types.hpp"

namespace {

template <typename T>
bool read_bin(std::istream& stream, T* value) {
    return static_cast<bool>(
            stream.read(reinterpret_cast<char*>(value), sizeof(T)));
}

bool read_floats(std::istream& stream, std::vector<float>* values, size_t count) {
    values->resize(count);
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(values->data()),
                                         count * sizeof(float)));
}

aicore::lightglue::Features to_native(const aicore_lightglue_features* in) {
    aicore::lightglue::Features out;
    if (!in) return out;
    out.descriptor_dim = in->descriptor_dim;
    out.image_width = in->image_width;
    out.image_height = in->image_height;
    if (in->keypoints && in->n_keypoints > 0) {
        out.keypoints.resize(static_cast<size_t>(in->n_keypoints));
        for (int32_t i = 0; i < in->n_keypoints; ++i) {
            out.keypoints[static_cast<size_t>(i)] = {
                    in->keypoints[i].x, in->keypoints[i].y,
                    in->keypoints[i].scale, in->keypoints[i].orientation};
        }
    }
    if (in->descriptors && in->n_keypoints > 0 && in->descriptor_dim > 0) {
        const size_t n = static_cast<size_t>(in->n_keypoints) *
                         static_cast<size_t>(in->descriptor_dim);
        out.descriptors.assign(in->descriptors, in->descriptors + n);
    }
    return out;
}

char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out) std::strcpy(out, s.c_str());
    return out;
}

}  // namespace

struct aicore_lightglue_options {
    aicore::lightglue::MatchingOptions o;
};

struct aicore_lightglue_ctx {
    std::unique_ptr<aicore::lightglue::FeatureMatcher> matcher;
    aicore::lightglue::MatchingOptions opts;
    std::string error;
};

extern "C" {

AICORE_CAPI int aicore_lightglue_abi_version(void) { return 1; }

AICORE_CAPI aicore_lightglue_options* aicore_lightglue_options_new(void) {
    return new aicore_lightglue_options();
}

AICORE_CAPI void aicore_lightglue_options_free(
        aicore_lightglue_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_lightglue_options_set_device(
        aicore_lightglue_options* opts, const char* device) {
    if (opts) opts->o.device = device ? device : "";
}

AICORE_CAPI void aicore_lightglue_options_set_threads(
        aicore_lightglue_options* opts, int n_threads) {
    if (opts) opts->o.num_threads = n_threads;
}

AICORE_CAPI void aicore_lightglue_options_set_min_score(
        aicore_lightglue_options* opts, double min_score) {
    if (opts) opts->o.min_score = min_score;
}

AICORE_CAPI void aicore_lightglue_options_set_matcher_type(
        aicore_lightglue_options* opts, int matcher_type) {
    if (!opts) return;
    switch (matcher_type) {
        case 1:
            opts->o.type = aicore::lightglue::FeatureMatcherType::kSiftLightGlue;
            break;
        case 2:
            opts->o.type =
                    aicore::lightglue::FeatureMatcherType::kAlikedLightGlue;
            break;
        default:
            opts->o.type = aicore::lightglue::FeatureMatcherType::kAuto;
            break;
    }
}

static aicore_lightglue_ctx* load_internal(
        const char* gguf_path, const aicore_lightglue_options* opts) {
    if (!gguf_path) return nullptr;
    auto* ctx = new (std::nothrow) aicore_lightglue_ctx();
    if (!ctx) return nullptr;
    if (opts) ctx->opts = opts->o;
    if (ctx->opts.device.empty()) ctx->opts.device = "auto";
    ctx->opts.model_path = gguf_path;
    std::string err;
    ctx->matcher = aicore::lightglue::create_feature_matcher(ctx->opts, &err);
    if (!ctx->matcher) {
        ctx->error = err.empty() ? "failed to load LightGlue model" : err;
    }
    return ctx;
}

AICORE_CAPI aicore_lightglue_ctx* aicore_lightglue_load(const char* gguf_path,
                                                        int n_threads) {
    aicore_lightglue_options opts{};
    opts.o.num_threads = n_threads;
    return load_internal(gguf_path, &opts);
}

AICORE_CAPI aicore_lightglue_ctx* aicore_lightglue_load_opts(
        const char* gguf_path, const aicore_lightglue_options* opts) {
    return load_internal(gguf_path, opts);
}

AICORE_CAPI void aicore_lightglue_free(aicore_lightglue_ctx* ctx) { delete ctx; }

AICORE_CAPI const char*
aicore_lightglue_last_error(const aicore_lightglue_ctx* ctx) {
    if (!ctx) return "NULL context";
    return ctx->error.empty() ? nullptr : ctx->error.c_str();
}

AICORE_CAPI int aicore_lightglue_geometry_of(
        const aicore_lightglue_ctx* ctx, aicore_lightglue_geometry* out) {
    if (!ctx || !out || !ctx->matcher) return -1;
    aicore::lightglue::ModelGeometry geo{};
    if (!ctx->matcher->geometry(&geo)) return -1;
    out->input_dim = geo.input_dim;
    out->descriptor_dim = geo.descriptor_dim;
    out->num_heads = geo.num_heads;
    out->num_layers = geo.num_layers;
    out->feature_type = geo.feature_type;
    out->add_scale_orientation = geo.add_scale_orientation;
    return 0;
}

AICORE_CAPI char* aicore_lightglue_info_json(aicore_lightglue_ctx* ctx) {
    if (!ctx || !ctx->matcher) return nullptr;
    aicore::lightglue::ModelGeometry geo{};
    if (!ctx->matcher->geometry(&geo)) return nullptr;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\n"
                  "  \"architecture\": \"lightglue\",\n"
                  "  \"feature_type\": \"%s\",\n"
                  "  \"input_dim\": %d,\n"
                  "  \"descriptor_dim\": %d,\n"
                  "  \"num_heads\": %d,\n"
                  "  \"num_layers\": %d,\n"
                  "  \"add_scale_orientation\": %s,\n"
                  "  \"device\": \"%s\"\n"
                  "}",
                  aicore::lightglue::feature_type_name(
                          static_cast<aicore::lightglue::FeatureType>(
                                  geo.feature_type)),
                  geo.input_dim, geo.descriptor_dim, geo.num_heads,
                  geo.num_layers, geo.add_scale_orientation ? "true" : "false",
                  ctx->matcher->device().c_str());
    return dup_cstr(buf);
}

AICORE_CAPI void aicore_lightglue_free_string(char* s) { std::free(s); }

AICORE_CAPI int aicore_lightglue_run_match(
        aicore_lightglue_ctx* ctx,
        const aicore_lightglue_features* image1,
        const aicore_lightglue_features* image2,
        aicore_lightglue_match** out_matches,
        int32_t* n_matches) {
    if (!ctx || !ctx->matcher || !image1 || !image2 || !out_matches ||
        !n_matches) {
        return -1;
    }
    *out_matches = nullptr;
    *n_matches = 0;

    const auto f1 = to_native(image1);
    const auto f2 = to_native(image2);
    std::vector<aicore::lightglue::Match> matches;
    if (!ctx->matcher->match(f1, f2, &matches)) {
        ctx->error = ctx->matcher->error();
        return -1;
    }
    if (matches.empty()) return 0;

    auto* out = static_cast<aicore_lightglue_match*>(
            std::malloc(matches.size() * sizeof(aicore_lightglue_match)));
    if (!out) return -1;
    for (size_t i = 0; i < matches.size(); ++i) {
        out[i].idx1 = matches[i].point2D_idx1;
        out[i].idx2 = matches[i].point2D_idx2;
        out[i].score = matches[i].score;
    }
    *out_matches = out;
    *n_matches = static_cast<int32_t>(matches.size());
    return 0;
}

AICORE_CAPI void aicore_lightglue_free_matches(
        aicore_lightglue_match* matches) {
    std::free(matches);
}

AICORE_CAPI int aicore_lightglue_load_fixture(
        const char* path, aicore_lightglue_features* image0,
        aicore_lightglue_features* image1) {
    if (!path || !image0 || !image1) return -1;
    std::ifstream stream(path, std::ios::binary);
    char magic[8];
    uint32_t version, m, n, dim, width0, height0, width1, height1, flags;
    if (!stream.read(magic, sizeof(magic)) ||
        std::memcmp(magic, "LGINP01\0", sizeof(magic)) != 0 ||
        !read_bin(stream, &version) || !read_bin(stream, &m) ||
        !read_bin(stream, &n) || !read_bin(stream, &dim) ||
        !read_bin(stream, &width0) || !read_bin(stream, &height0) ||
        !read_bin(stream, &width1) || !read_bin(stream, &height1) ||
        !read_bin(stream, &flags) || version != 1) {
        return -1;
    }
    (void)flags;

    std::vector<float> keypoints0, keypoints1, desc0, desc1;
    std::vector<float> scales0, scales1, oris0, oris1;
    if (!read_floats(stream, &keypoints0, static_cast<size_t>(m) * 2) ||
        !read_floats(stream, &keypoints1, static_cast<size_t>(n) * 2) ||
        !read_floats(stream, &desc0, static_cast<size_t>(m) * dim) ||
        !read_floats(stream, &desc1, static_cast<size_t>(n) * dim) ||
        !read_floats(stream, &scales0, m) || !read_floats(stream, &scales1, n) ||
        !read_floats(stream, &oris0, m) || !read_floats(stream, &oris1, n)) {
        return -1;
    }

    image0->n_keypoints = static_cast<int32_t>(m);
    image1->n_keypoints = static_cast<int32_t>(n);
    image0->descriptor_dim = image1->descriptor_dim = static_cast<int32_t>(dim);
    image0->image_width = static_cast<int32_t>(width0);
    image0->image_height = static_cast<int32_t>(height0);
    image1->image_width = static_cast<int32_t>(width1);
    image1->image_height = static_cast<int32_t>(height1);

    image0->keypoints = static_cast<aicore_lightglue_keypoint*>(
            std::calloc(m, sizeof(aicore_lightglue_keypoint)));
    image1->keypoints = static_cast<aicore_lightglue_keypoint*>(
            std::calloc(n, sizeof(aicore_lightglue_keypoint)));
    image0->descriptors =
            static_cast<float*>(std::malloc(desc0.size() * sizeof(float)));
    image1->descriptors =
            static_cast<float*>(std::malloc(desc1.size() * sizeof(float)));
    if (!image0->keypoints || !image1->keypoints || !image0->descriptors ||
        !image1->descriptors) {
        aicore_lightglue_free_features(image0);
        aicore_lightglue_free_features(image1);
        return -1;
    }
    for (uint32_t i = 0; i < m; ++i) {
        image0->keypoints[i].x = keypoints0[2 * i];
        image0->keypoints[i].y = keypoints0[2 * i + 1];
        image0->keypoints[i].scale = scales0[i];
        image0->keypoints[i].orientation = oris0[i];
    }
    for (uint32_t i = 0; i < n; ++i) {
        image1->keypoints[i].x = keypoints1[2 * i];
        image1->keypoints[i].y = keypoints1[2 * i + 1];
        image1->keypoints[i].scale = scales1[i];
        image1->keypoints[i].orientation = oris1[i];
    }
    std::memcpy(image0->descriptors, desc0.data(), desc0.size() * sizeof(float));
    std::memcpy(image1->descriptors, desc1.data(), desc1.size() * sizeof(float));
    return 0;
}

AICORE_CAPI void aicore_lightglue_free_features(
        aicore_lightglue_features* features) {
    if (!features) return;
    std::free(features->keypoints);
    std::free(features->descriptors);
    features->keypoints = nullptr;
    features->descriptors = nullptr;
    features->n_keypoints = 0;
}

AICORE_CAPI int aicore_lightglue_quantize(const char* input_gguf,
                                          const char* output_gguf,
                                          const char* type) {
    if (!input_gguf || !output_gguf || !type) return -1;
    std::string err;
    if (!aicore::lightglue::quantize_model(input_gguf, output_gguf, type,
                                           &err)) {
        return -1;
    }
    return 0;
}

AICORE_CAPI int aicore_lightglue_warmup_backend(const char* device) {
    return aicore_warmup_backend(device);
}

AICORE_CAPI char* aicore_lightglue_model_cache_dir(void) {
    return dup_cstr(aicore::lightglue::default_model_cache_dir());
}

}  // extern "C"
