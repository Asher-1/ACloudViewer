#include "types.hpp"

#include "backend.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

namespace aicore {
namespace lightglue {
namespace {

constexpr const char *kArchitecture = "lightglue";
constexpr size_t kMaxGraphNodes = 16384;

struct Linear {
  ggml_tensor *weight = nullptr;
  ggml_tensor *bias = nullptr;
};

struct FeedForward {
  Linear input;
  ggml_tensor *norm_weight = nullptr;
  ggml_tensor *norm_bias = nullptr;
  Linear output;
};

struct LayerWeights {
  Linear self_qkv;
  Linear self_out;
  FeedForward self_ffn;
  Linear cross_qk;
  Linear cross_v;
  Linear cross_out;
  FeedForward cross_ffn;
};

struct HyperParameters {
  FeatureType feature_type = FeatureType::kAliked;
  int32_t input_dim = 0;
  int32_t descriptor_dim = 0;
  int32_t num_heads = 0;
  int32_t num_layers = 0;
  bool add_scale_orientation = false;
  float filter_threshold = 0.1f;
  float layer_norm_epsilon = 1e-5f;
};

bool ParseFeatureType(const std::string &name, FeatureType *type) {
  if (name == "superpoint") {
    *type = FeatureType::kSuperPoint;
  } else if (name == "disk") {
    *type = FeatureType::kDisk;
  } else if (name == "aliked" || name == "raco-aliked") {
    *type = FeatureType::kAliked;
  } else if (name == "sift") {
    *type = FeatureType::kSift;
  } else if (name == "doghardnet") {
    *type = FeatureType::kDogHardNet;
  } else {
    return false;
  }
  return true;
}

class ModelFile {
public:
  ~ModelFile() { Close(); }

  bool Open(const std::string &path) {
    Close();
    error.clear();
    gguf_init_params params{/*no_alloc=*/false, /*ctx=*/&context_};
    gguf_ = gguf_init_from_file(path.c_str(), params);
    if (gguf_ == nullptr || context_ == nullptr) {
      error = "failed to read GGUF model: " + path;
      Close();
      return false;
    }

    const std::string architecture = String("general.architecture");
    if (error.empty() && architecture != kArchitecture) {
      error = "unsupported GGUF architecture '" + architecture +
              "' (expected lightglue)";
    }
    const std::string feature = String("lightglue.feature_type");
    if (error.empty() && !ParseFeatureType(feature, &hp.feature_type)) {
      error = "unsupported LightGlue feature type '" + feature + "'";
    }
    Uint32("lightglue.input_dimension", &hp.input_dim);
    Uint32("lightglue.descriptor_dimension", &hp.descriptor_dim);
    Uint32("lightglue.attention.head_count", &hp.num_heads);
    Uint32("lightglue.block_count", &hp.num_layers);
    Boolean("lightglue.add_scale_orientation", &hp.add_scale_orientation);
    Float("lightglue.filter_threshold", &hp.filter_threshold);
    Float("lightglue.layer_norm_epsilon", &hp.layer_norm_epsilon);

    if (error.empty() &&
        (hp.input_dim <= 0 || hp.descriptor_dim <= 0 || hp.num_heads <= 0 ||
         hp.num_layers <= 0 || hp.descriptor_dim % hp.num_heads != 0)) {
      error = "invalid LightGlue dimensions in GGUF metadata";
    }
    if (!error.empty()) {
      Close();
      return false;
    }
    return true;
  }

  void Close() {
    if (context_ != nullptr) {
      ggml_free(context_);
      context_ = nullptr;
    }
    if (gguf_ != nullptr) {
      gguf_free(gguf_);
      gguf_ = nullptr;
    }
    hp = HyperParameters{};
  }

  ggml_tensor *Require(const std::string &name) {
    ggml_tensor *tensor =
        context_ == nullptr ? nullptr : ggml_get_tensor(context_, name.c_str());
    if (tensor == nullptr && error.empty()) {
      error = "missing tensor: " + name;
    }
    return tensor;
  }

  int64_t TensorCount() const { return gguf_get_n_tensors(gguf_); }
  const char *TensorName(int64_t index) const {
    return gguf_get_tensor_name(gguf_, index);
  }
  ggml_context *Context() const { return context_; }

  HyperParameters hp;
  std::string error;

private:
  int64_t Find(const char *key) {
    const int64_t index = gguf_find_key(gguf_, key);
    if (index < 0 && error.empty()) {
      error = std::string("missing GGUF key: ") + key;
    }
    return index;
  }

  std::string String(const char *key) {
    const int64_t index = Find(key);
    return index < 0 ? std::string() : gguf_get_val_str(gguf_, index);
  }
  void Uint32(const char *key, int32_t *value) {
    const int64_t index = Find(key);
    if (index >= 0) {
      *value = static_cast<int32_t>(gguf_get_val_u32(gguf_, index));
    }
  }
  void Float(const char *key, float *value) {
    const int64_t index = Find(key);
    if (index >= 0) {
      *value = gguf_get_val_f32(gguf_, index);
    }
  }
  void Boolean(const char *key, bool *value) {
    const int64_t index = Find(key);
    if (index >= 0) {
      *value = gguf_get_val_bool(gguf_, index);
    }
  }

  gguf_context *gguf_ = nullptr;
  ggml_context *context_ = nullptr;
};

ggml_tensor *LinearForward(ggml_context *context, const Linear &linear,
                           ggml_tensor *input) {
  ggml_tensor *output = ggml_mul_mat(context, linear.weight, input);
  if (linear.bias != nullptr) {
    output = ggml_add(context, output, linear.bias);
  }
  return output;
}

ggml_tensor *FeedForwardBlock(ggml_context *context, const FeedForward &weights,
                              ggml_tensor *residual, ggml_tensor *message,
                              float epsilon) {
  ggml_tensor *joined = ggml_concat(context, residual, message, 0);
  ggml_tensor *output = LinearForward(context, weights.input, joined);
  output = ggml_norm(context, output, epsilon);
  output = ggml_mul(context, output, weights.norm_weight);
  output = ggml_add(context, output, weights.norm_bias);
  output = ggml_gelu_erf(context, output);
  output = LinearForward(context, weights.output, output);
  return ggml_add(context, residual, output);
}

ggml_tensor *RepeatPairs(ggml_context *context, ggml_tensor *input,
                         int64_t half_dimension, int64_t tokens) {
  ggml_tensor *shaped =
      ggml_reshape_3d(context, input, 1, half_dimension, tokens);
  ggml_tensor *repeated =
      ggml_repeat_4d(context, shaped, 2, half_dimension, tokens, 1);
  return ggml_reshape_2d(context, repeated, 2 * half_dimension, tokens);
}

ggml_tensor *RotateHalf(ggml_context *context, ggml_tensor *input,
                        int64_t head_dim, int64_t tokens, int64_t heads) {
  ggml_tensor *pairs =
      ggml_reshape_4d(context, input, 2, head_dim / 2, tokens, heads);
  ggml_tensor *first =
      ggml_view_4d(context, pairs, 1, head_dim / 2, tokens, heads, pairs->nb[1],
                   pairs->nb[2], pairs->nb[3], 0);
  ggml_tensor *second =
      ggml_view_4d(context, pairs, 1, head_dim / 2, tokens, heads, pairs->nb[1],
                   pairs->nb[2], pairs->nb[3], sizeof(float));
  first = ggml_cont(context, first);
  second = ggml_cont(context, second);
  ggml_tensor *rotated =
      ggml_concat(context, ggml_neg(context, second), first, 0);
  return ggml_reshape_3d(context, rotated, head_dim, tokens, heads);
}

ggml_tensor *ApplyRotary(ggml_context *context, ggml_tensor *input,
                         ggml_tensor *cosine, ggml_tensor *sine,
                         int64_t head_dim, int64_t tokens, int64_t heads) {
  ggml_tensor *cos_all = ggml_repeat(context, cosine, input);
  ggml_tensor *sin_all = ggml_repeat(context, sine, input);
  ggml_tensor *rotated = RotateHalf(context, input, head_dim, tokens, heads);
  return ggml_add(context, ggml_mul(context, input, cos_all),
                  ggml_mul(context, rotated, sin_all));
}

struct Qkv {
  ggml_tensor *q = nullptr;
  ggml_tensor *k = nullptr;
  ggml_tensor *v = nullptr;
};

Qkv SplitInterleavedQkv(ggml_context *context, ggml_tensor *input,
                        int64_t head_dim, int64_t tokens, int64_t heads) {
  ggml_tensor *shaped =
      ggml_reshape_4d(context, input, 3, head_dim, heads, tokens);
  // ggml_permute maps each old axis to a new position. This is the inverse of
  // PyTorch's permute argument convention: [3, D, H, T] -> [D, T, H, 3].
  ggml_tensor *ordered =
      ggml_cont(context, ggml_permute(context, shaped, 3, 0, 2, 1));
  auto component = [&](int index) {
    ggml_tensor *view = ggml_view_3d(
        context, ordered, head_dim, tokens, heads, ordered->nb[1],
        ordered->nb[2], static_cast<size_t>(index) * ordered->nb[3]);
    return ggml_cont(context, view);
  };
  return {component(0), component(1), component(2)};
}

ggml_tensor *Heads(ggml_context *context, ggml_tensor *input, int64_t head_dim,
                   int64_t tokens, int64_t heads) {
  ggml_tensor *shaped =
      ggml_reshape_3d(context, input, head_dim, heads, tokens);
  return ggml_cont(context, ggml_permute(context, shaped, 0, 2, 1, 3));
}

ggml_tensor *Attention(ggml_context *context, ggml_tensor *query,
                       ggml_tensor *key, ggml_tensor *value, float scale,
                       bool fused) {
  if (fused) {
    ggml_tensor *output = ggml_flash_attn_ext(context, query, key, value,
                                              nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    return output; // [head_dim, heads, query_tokens]
  }
  ggml_tensor *scores = ggml_mul_mat(context, key, query);
  ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
  scores = ggml_soft_max_ext(context, scores, nullptr, scale, 0.0f);
  ggml_tensor *value_transposed =
      ggml_cont(context, ggml_permute(context, value, 1, 0, 2, 3));
  ggml_tensor *output = ggml_mul_mat(context, value_transposed, scores);
  return ggml_cont(context, ggml_permute(context, output, 0, 2, 1, 3));
}

struct Encoding {
  ggml_tensor *cosine = nullptr;
  ggml_tensor *sine = nullptr;
};

using TapFunction =
    std::function<ggml_tensor *(ggml_tensor *, const std::string &)>;

Encoding PositionalEncoding(ggml_context *context, ggml_tensor *positions,
                            ggml_tensor *weight, int64_t head_dim,
                            int64_t tokens) {
  ggml_tensor *projected = ggml_mul_mat(context, weight, positions);
  return {
      RepeatPairs(context, ggml_cos(context, projected), head_dim / 2, tokens),
      RepeatPairs(context, ggml_sin(context, projected), head_dim / 2, tokens)};
}

ggml_tensor *SelfBlock(ggml_context *context, const LayerWeights &weights,
                       ggml_tensor *descriptors, const Encoding &encoding,
                       int64_t descriptor_dim, int64_t head_dim, int64_t tokens,
                       int64_t heads, float epsilon, bool fused_attention,
                       const TapFunction &tap, const std::string &prefix) {
  ggml_tensor *qkv_projection = tap(
      LinearForward(context, weights.self_qkv, descriptors), prefix + ".qkv");
  Qkv qkv =
      SplitInterleavedQkv(context, qkv_projection, head_dim, tokens, heads);
  qkv.q = ApplyRotary(context, qkv.q, encoding.cosine, encoding.sine, head_dim,
                      tokens, heads);
  qkv.k = ApplyRotary(context, qkv.k, encoding.cosine, encoding.sine, head_dim,
                      tokens, heads);
  tap(qkv.q, prefix + ".q");
  tap(qkv.k, prefix + ".k");
  tap(qkv.v, prefix + ".v");
  ggml_tensor *message = Attention(context, qkv.q, qkv.k, qkv.v,
                                   1.0f / std::sqrt(head_dim), fused_attention);
  tap(message, prefix + ".attention");
  message = ggml_reshape_2d(context, message, descriptor_dim, tokens);
  message = LinearForward(context, weights.self_out, message);
  tap(message, prefix + ".output");
  return tap(FeedForwardBlock(context, weights.self_ffn, descriptors, message,
                              epsilon),
             prefix + ".ffn");
}

std::pair<ggml_tensor *, ggml_tensor *>
CrossBlock(ggml_context *context, const LayerWeights &weights,
           ggml_tensor *descriptors0, ggml_tensor *descriptors1,
           int64_t descriptor_dim, int64_t head_dim, int64_t tokens0,
           int64_t tokens1, int64_t heads, float epsilon,
           bool fused_attention) {
  ggml_tensor *qk0 =
      Heads(context, LinearForward(context, weights.cross_qk, descriptors0),
            head_dim, tokens0, heads);
  ggml_tensor *qk1 =
      Heads(context, LinearForward(context, weights.cross_qk, descriptors1),
            head_dim, tokens1, heads);
  ggml_tensor *value0 =
      Heads(context, LinearForward(context, weights.cross_v, descriptors0),
            head_dim, tokens0, heads);
  ggml_tensor *value1 =
      Heads(context, LinearForward(context, weights.cross_v, descriptors1),
            head_dim, tokens1, heads);
  const float scale = 1.0f / std::sqrt(head_dim);
  ggml_tensor *message0 =
      Attention(context, qk0, qk1, value1, scale, fused_attention);
  ggml_tensor *message1 =
      Attention(context, qk1, qk0, value0, scale, fused_attention);
  message0 = ggml_reshape_2d(context, message0, descriptor_dim, tokens0);
  message1 = ggml_reshape_2d(context, message1, descriptor_dim, tokens1);
  message0 = LinearForward(context, weights.cross_out, message0);
  message1 = LinearForward(context, weights.cross_out, message1);
  return {FeedForwardBlock(context, weights.cross_ffn, descriptors0, message0,
                           epsilon),
          FeedForwardBlock(context, weights.cross_ffn, descriptors1, message1,
                           epsilon)};
}

ggml_tensor *LogSigmoid(ggml_context *context, ggml_tensor *input) {
  return ggml_neg(context, ggml_softplus(context, ggml_neg(context, input)));
}

} // namespace

class FeatureMatcherImpl final : public FeatureMatcher {
public:
  explicit FeatureMatcherImpl(MatchingOptions options)
      : options_(std::move(options)) {}

  ~FeatureMatcherImpl() override { Release(); }

  bool Load() {
    if (!file_.Open(options_.model_path)) {
      error_ = file_.error;
      return false;
    }
    const bool wrong_sift_model =
        options_.type == FeatureMatcherType::kSiftLightGlue &&
        file_.hp.feature_type != FeatureType::kSift;
    const bool wrong_aliked_model =
        options_.type == FeatureMatcherType::kAlikedLightGlue &&
        file_.hp.feature_type != FeatureType::kAliked;
    if (wrong_sift_model || wrong_aliked_model) {
      error_ = std::string("model feature type '") +
               feature_type_name(file_.hp.feature_type) +
               "' does not match matcher type '" +
               feature_matcher_type_name(options_.type) + "'";
      return false;
    }
    if (!backend_.init(options_.device, options_.num_threads)) {
      error_ = backend_.error;
      return false;
    }
    if (!MapWeights() || !RealizeWeights()) {
      return false;
    }
    return true;
  }

  bool match(const Features &image1, const Features &image2,
             std::vector<Match> *matches) override {
    if (matches == nullptr) {
      error_ = "matches output is null";
      return false;
    }
    matches->clear();
    RawResult raw;
    if (!run_raw(image1, image2, &raw)) {
      return false;
    }
    for (size_t i = 0; i < raw.matches0.size(); ++i) {
      if (raw.matches0[i] >= 0) {
        matches->push_back(
            {static_cast<int32_t>(i), raw.matches0[i], raw.mscores0[i]});
      }
    }
    return true;
  }

  bool run_raw(const Features &image1, const Features &image2,
              RawResult *result) override {
    error_.clear();
    if (result == nullptr) {
      error_ = "raw result output is null";
      return false;
    }
    result->matches0.assign(image1.keypoints.size(), -1);
    result->mscores0.assign(image1.keypoints.size(), 0.0f);
    if (!ValidateFeatures(image1, "image1") ||
        !ValidateFeatures(image2, "image2")) {
      return false;
    }
    if (image1.keypoints.empty() || image2.keypoints.empty()) {
      return true;
    }

    const int64_t tokens0 = static_cast<int64_t>(image1.keypoints.size());
    const int64_t tokens1 = static_cast<int64_t>(image2.keypoints.size());
    const auto profile_start = std::chrono::steady_clock::now();
    const int64_t input_dim = file_.hp.input_dim;
    const int64_t dim = file_.hp.descriptor_dim;
    const int64_t heads = file_.hp.num_heads;
    const int64_t head_dim = dim / heads;
    const int64_t position_dim = file_.hp.add_scale_orientation ? 4 : 2;
    bool fused_attention =
        !backend_.is_cpu() || std::max(tokens0, tokens1) > 256;
    if (const char *mode = std::getenv("LIGHTGLUE_ATTENTION")) {
      fused_attention = std::string(mode) != "manual";
    }
    const char *dump_directory = std::getenv("LIGHTGLUE_DUMP_DIR");
    std::vector<std::pair<std::string, ggml_tensor *>> taps;
    auto tap = [&](ggml_tensor *tensor, const std::string &name) {
      if (dump_directory != nullptr) {
        for (ggml_tensor *storage = tensor; storage != nullptr;
             storage = storage->view_src) {
          ggml_set_output(storage);
        }
        taps.emplace_back(name, tensor);
      }
      return tensor;
    };

    std::vector<float> positions0 = NormalizeKeypoints(image1);
    std::vector<float> positions1 = NormalizeKeypoints(image2);

    ggml_init_params params{
        ggml_tensor_overhead() * kMaxGraphNodes +
            ggml_graph_overhead_custom(kMaxGraphNodes, false),
        nullptr,
        /*no_alloc=*/true};
    ggml_context *context = ggml_init(params);
    if (context == nullptr) {
      error_ = "failed to initialize the ggml graph context";
      return false;
    }

    ggml_tensor *position0 =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, position_dim, tokens0);
    ggml_tensor *position1 =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, position_dim, tokens1);
    ggml_tensor *descriptor0 =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, input_dim, tokens0);
    ggml_tensor *descriptor1 =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, input_dim, tokens1);
    ggml_set_name(position0, "positions0");
    ggml_set_name(position1, "positions1");
    ggml_set_name(descriptor0, "descriptors0");
    ggml_set_name(descriptor1, "descriptors1");

    if (input_projection_.weight != nullptr) {
      descriptor0 = LinearForward(context, input_projection_, descriptor0);
      descriptor1 = LinearForward(context, input_projection_, descriptor1);
    }
    tap(descriptor0, "input_projection0");
    tap(descriptor1, "input_projection1");
    const Encoding encoding0 = PositionalEncoding(
        context, position0, positional_weight_, head_dim, tokens0);
    const Encoding encoding1 = PositionalEncoding(
        context, position1, positional_weight_, head_dim, tokens1);
    tap(encoding0.cosine, "encoding_cos0");
    tap(encoding0.sine, "encoding_sin0");
    tap(encoding1.cosine, "encoding_cos1");
    tap(encoding1.sine, "encoding_sin1");

    for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
      const LayerWeights &layer = layers_[layer_index];
      descriptor0 = SelfBlock(context, layer, descriptor0, encoding0, dim,
                              head_dim, tokens0, heads,
                              file_.hp.layer_norm_epsilon, fused_attention, tap,
                              "layer" + std::to_string(layer_index) + ".self0");
      tap(descriptor0, "layer" + std::to_string(layer_index) + ".self0");
      descriptor1 = SelfBlock(context, layer, descriptor1, encoding1, dim,
                              head_dim, tokens1, heads,
                              file_.hp.layer_norm_epsilon, fused_attention, tap,
                              "layer" + std::to_string(layer_index) + ".self1");
      tap(descriptor1, "layer" + std::to_string(layer_index) + ".self1");
      std::tie(descriptor0, descriptor1) = CrossBlock(
          context, layer, descriptor0, descriptor1, dim, head_dim, tokens0,
          tokens1, heads, file_.hp.layer_norm_epsilon, fused_attention);
      tap(descriptor0, "layer" + std::to_string(layer_index) + ".cross0");
      tap(descriptor1, "layer" + std::to_string(layer_index) + ".cross1");
    }

    ggml_tensor *projected0 =
        LinearForward(context, assignment_projection_, descriptor0);
    ggml_tensor *projected1 =
        LinearForward(context, assignment_projection_, descriptor1);
    const float projection_scale = 1.0f / std::sqrt(std::sqrt(dim));
    projected0 = ggml_scale(context, projected0, projection_scale);
    projected1 = ggml_scale(context, projected1, projection_scale);
    ggml_tensor *similarity =
        ggml_mul_mat(context, projected1, projected0); // [tokens1, tokens0]
    tap(similarity, "similarity");

    ggml_tensor *scores0 =
        ggml_log(context, ggml_soft_max(context, similarity));
    ggml_tensor *similarity_t =
        ggml_cont(context, ggml_transpose(context, similarity));
    ggml_tensor *scores1_t =
        ggml_log(context, ggml_soft_max(context, similarity_t));
    ggml_tensor *scores1 =
        ggml_cont(context, ggml_transpose(context, scores1_t));

    ggml_tensor *certainty0 =
        LogSigmoid(context, LinearForward(context, matchability_, descriptor0));
    ggml_tensor *certainty1 =
        LogSigmoid(context, LinearForward(context, matchability_, descriptor1));
    certainty0 = ggml_repeat(context, certainty0, similarity);
    certainty1 =
        ggml_repeat(context, ggml_transpose(context, certainty1), similarity);
    ggml_tensor *scores = ggml_add(context, ggml_add(context, scores0, scores1),
                                   ggml_add(context, certainty0, certainty1));
    ggml_set_name(scores, "log_assignment");
    tap(scores, "scores");

    ggml_cgraph *graph = ggml_new_graph_custom(context, kMaxGraphNodes, false);
    ggml_tensor *device_best0 = nullptr;
    ggml_tensor *device_mutual1 = nullptr;
    ggml_tensor *device_best_score0 = nullptr;
    if (backend_.is_cpu()) {
      ggml_set_output(scores);
      ggml_build_forward_expand(graph, scores);
    } else {
      // Reduce matching on the accelerator. Copying the full MxN assignment
      // matrix to the host dominates CUDA latency at large keypoint counts.
      ggml_tensor *scores_transposed =
          ggml_cont(context, ggml_transpose(context, scores));
      device_best0 = ggml_argmax(context, scores);
      ggml_tensor *device_best1 = ggml_argmax(context, scores_transposed);
      ggml_tensor *best1_rows = ggml_reshape_2d(context, device_best1, 1, tokens1);
      device_mutual1 = ggml_get_rows(context, best1_rows, device_best0);

      // Gather scores[i, best0[i]] from the flattened assignment matrix.
      // Linear indices avoid materializing an additional tokens0^2 tensor.
      ggml_tensor *row_offsets = ggml_arange(
          context, 0.0f, static_cast<float>(tokens0 * tokens1),
          static_cast<float>(tokens1));
      ggml_tensor *linear_indices = ggml_cast(
          context,
          ggml_add(context, ggml_cast(context, device_best0, GGML_TYPE_F32),
                   row_offsets),
          GGML_TYPE_I32);
      ggml_tensor *scores_flat =
          ggml_reshape_2d(context, scores, 1, tokens0 * tokens1);
      device_best_score0 =
          ggml_get_rows(context, scores_flat, linear_indices);
      ggml_set_output(device_best0);
      ggml_set_output(device_mutual1);
      ggml_set_output(device_best_score0);
      ggml_build_forward_expand(graph, device_best0);
      ggml_build_forward_expand(graph, device_mutual1);
      ggml_build_forward_expand(graph, device_best_score0);
    }
    const auto profile_graph = std::chrono::steady_clock::now();
    if (!ggml_gallocr_alloc_graph(backend_.galloc, graph)) {
      error_ = "failed to allocate the LightGlue compute graph";
      ggml_free(context);
      return false;
    }
    const auto profile_allocate = std::chrono::steady_clock::now();

    ggml_backend_tensor_set(position0, positions0.data(), 0,
                            positions0.size() * sizeof(float));
    ggml_backend_tensor_set(position1, positions1.data(), 0,
                            positions1.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_get_tensor(context, "descriptors0"),
                            image1.descriptors.data(), 0,
                            image1.descriptors.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_get_tensor(context, "descriptors1"),
                            image2.descriptors.data(), 0,
                            image2.descriptors.size() * sizeof(float));

    const ggml_status status =
        ggml_backend_graph_compute(backend_.be, graph);
    if (status != GGML_STATUS_SUCCESS) {
      error_ = "ggml backend failed while computing LightGlue";
      ggml_free(context);
      return false;
    }
    const auto profile_compute = std::chrono::steady_clock::now();

    if (backend_.is_cpu()) {
      std::vector<float> host_scores(static_cast<size_t>(tokens0 * tokens1));
      ggml_backend_tensor_get(scores, host_scores.data(), 0,
                              host_scores.size() * sizeof(float));
      FilterMatches(host_scores, tokens0, tokens1, result);
    } else {
      std::vector<int32_t> best0(static_cast<size_t>(tokens0));
      std::vector<int32_t> mutual1(static_cast<size_t>(tokens0));
      std::vector<float> best_score0(static_cast<size_t>(tokens0));
      ggml_backend_tensor_get(device_best0, best0.data(), 0,
                              best0.size() * sizeof(int32_t));
      ggml_backend_tensor_get(device_mutual1, mutual1.data(), 0,
                              mutual1.size() * sizeof(int32_t));
      ggml_backend_tensor_get(device_best_score0, best_score0.data(), 0,
                              best_score0.size() * sizeof(float));
      FilterDeviceMatches(best0, mutual1, best_score0, result);
    }
    if (dump_directory != nullptr) {
      std::filesystem::create_directories(dump_directory);
      for (const auto &entry : taps) {
        const ggml_tensor *tensor = entry.second;
        std::ofstream stream(std::filesystem::path(dump_directory) /
                                 (entry.first + ".bin"),
                             std::ios::binary);
        const uint32_t dimensions = GGML_MAX_DIMS;
        stream.write("LGTAP01\0", 8);
        stream.write(reinterpret_cast<const char *>(&dimensions),
                     sizeof(dimensions));
        stream.write(reinterpret_cast<const char *>(tensor->ne),
                     sizeof(tensor->ne));
        std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, values.data(), 0,
                                values.size() * sizeof(float));
        stream.write(reinterpret_cast<const char *>(values.data()),
                     values.size() * sizeof(float));
      }
    }
    if (std::getenv("LIGHTGLUE_PROFILE") != nullptr) {
      const auto profile_end = std::chrono::steady_clock::now();
      auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
      };
      std::fprintf(stderr,
                   "profile: graph=%.3fms allocate=%.3fms compute=%.3fms "
                   "read_filter=%.3fms\n",
                   milliseconds(profile_start, profile_graph),
                   milliseconds(profile_graph, profile_allocate),
                   milliseconds(profile_allocate, profile_compute),
                   milliseconds(profile_compute, profile_end));
    }
    ggml_free(context);
    return true;
  }

  const std::string &error() const override { return error_; }
  FeatureType model_feature_type() const override {
    return file_.hp.feature_type;
  }
  const std::string &device() const override { return backend_.device; }

  bool geometry(ModelGeometry *out) const override {
    if (out == nullptr) return false;
    out->input_dim = file_.hp.input_dim;
    out->descriptor_dim = file_.hp.descriptor_dim;
    out->num_heads = file_.hp.num_heads;
    out->num_layers = file_.hp.num_layers;
    out->feature_type = static_cast<int32_t>(file_.hp.feature_type);
    out->add_scale_orientation = file_.hp.add_scale_orientation ? 1 : 0;
    return true;
  }

private:
  bool ValidateFeatures(const Features &features, const char *label) {
    if (features.image_width <= 0 || features.image_height <= 0) {
      error_ = std::string(label) + " has invalid image dimensions";
      return false;
    }
    if (features.descriptor_dim != file_.hp.input_dim) {
      error_ = std::string(label) + " descriptor dimension is " +
               std::to_string(features.descriptor_dim) + ", model expects " +
               std::to_string(file_.hp.input_dim);
      return false;
    }
    if (features.descriptors.size() !=
        features.keypoints.size() *
            static_cast<size_t>(features.descriptor_dim)) {
      error_ = std::string(label) + " descriptor buffer has the wrong size";
      return false;
    }
    return true;
  }

  std::vector<float> NormalizeKeypoints(const Features &features) const {
    const int position_dim = file_.hp.add_scale_orientation ? 4 : 2;
    std::vector<float> output(features.keypoints.size() * position_dim);
    const float width = static_cast<float>(features.image_width);
    const float height = static_cast<float>(features.image_height);
    const float scale = std::max(width, height) / 2.0f;
    for (size_t i = 0; i < features.keypoints.size(); ++i) {
      const Keypoint &keypoint = features.keypoints[i];
      output[i * position_dim + 0] = (keypoint.x - width / 2.0f) / scale;
      output[i * position_dim + 1] = (keypoint.y - height / 2.0f) / scale;
      if (position_dim == 4) {
        output[i * position_dim + 2] = keypoint.scale;
        output[i * position_dim + 3] = keypoint.orientation;
      }
    }
    return output;
  }

  bool MapWeights() {
    auto linear = [&](const std::string &prefix) {
      return Linear{file_.Require(prefix + ".weight"),
                    file_.Require(prefix + ".bias")};
    };
    auto feed_forward = [&](const std::string &prefix) {
      FeedForward result;
      result.input = linear(prefix + ".input");
      result.norm_weight = file_.Require(prefix + ".norm.weight");
      result.norm_bias = file_.Require(prefix + ".norm.bias");
      result.output = linear(prefix + ".output");
      return result;
    };

    positional_weight_ = file_.Require("positional_encoding.weight");
    if (file_.hp.input_dim != file_.hp.descriptor_dim) {
      input_projection_ = linear("input_projection");
    }
    layers_.resize(file_.hp.num_layers);
    for (int i = 0; i < file_.hp.num_layers; ++i) {
      const std::string prefix = "block." + std::to_string(i);
      LayerWeights &layer = layers_[i];
      layer.self_qkv = linear(prefix + ".self.qkv");
      layer.self_out = linear(prefix + ".self.output");
      layer.self_ffn = feed_forward(prefix + ".self.ffn");
      layer.cross_qk = linear(prefix + ".cross.qk");
      layer.cross_v = linear(prefix + ".cross.value");
      layer.cross_out = linear(prefix + ".cross.output");
      layer.cross_ffn = feed_forward(prefix + ".cross.ffn");
    }
    assignment_projection_ = linear("assignment.projection");
    matchability_ = linear("assignment.matchability");
    if (!file_.error.empty()) {
      error_ = file_.error;
      return false;
    }
    return true;
  }

  void ForEachWeight(const std::function<void(ggml_tensor *&)> &callback) {
    auto visit_linear = [&](Linear &linear) {
      callback(linear.weight);
      callback(linear.bias);
    };
    auto visit_ffn = [&](FeedForward &ffn) {
      visit_linear(ffn.input);
      callback(ffn.norm_weight);
      callback(ffn.norm_bias);
      visit_linear(ffn.output);
    };
    callback(positional_weight_);
    if (input_projection_.weight != nullptr) {
      visit_linear(input_projection_);
    }
    for (LayerWeights &layer : layers_) {
      visit_linear(layer.self_qkv);
      visit_linear(layer.self_out);
      visit_ffn(layer.self_ffn);
      visit_linear(layer.cross_qk);
      visit_linear(layer.cross_v);
      visit_linear(layer.cross_out);
      visit_ffn(layer.cross_ffn);
    }
    visit_linear(assignment_projection_);
    visit_linear(matchability_);
  }

  bool RealizeWeights() {
    bool expand_f16_for_cpu = false;
    if (backend_.is_cpu()) {
      for (int64_t i = 0; i < file_.TensorCount(); ++i) {
        const ggml_tensor *tensor =
            ggml_get_tensor(file_.Context(), file_.TensorName(i));
        expand_f16_for_cpu =
            expand_f16_for_cpu || tensor->type == GGML_TYPE_F16;
      }
    }
    if (backend_.is_cpu() && !expand_f16_for_cpu) {
      void *base = ggml_get_mem_buffer(file_.Context());
      const size_t size = ggml_get_mem_size(file_.Context());
      weight_buffer_ = ggml_backend_cpu_buffer_from_ptr(base, size);
      if (weight_buffer_ == nullptr) {
        error_ = "failed to wrap CPU model weights";
        return false;
      }
      for (int64_t i = 0; i < file_.TensorCount(); ++i) {
        ggml_tensor *tensor =
            ggml_get_tensor(file_.Context(), file_.TensorName(i));
        tensor->buffer = weight_buffer_;
      }
      return true;
    }

    ggml_init_params params{ggml_tensor_overhead() *
                                static_cast<size_t>(file_.TensorCount() + 8),
                            nullptr,
                            /*no_alloc=*/true};
    device_context_ = ggml_init(params);
    if (device_context_ == nullptr) {
      error_ = "failed to create the device weight context";
      return false;
    }
    for (int64_t i = 0; i < file_.TensorCount(); ++i) {
      ggml_tensor *source =
          ggml_get_tensor(file_.Context(), file_.TensorName(i));
      const ggml_type target_type =
          expand_f16_for_cpu && source->type == GGML_TYPE_F16
              ? GGML_TYPE_F32
              : source->type;
      ggml_tensor *target = ggml_new_tensor(device_context_, target_type,
                                            GGML_MAX_DIMS, source->ne);
      ggml_set_name(target, file_.TensorName(i));
    }
    weight_buffer_ =
        ggml_backend_alloc_ctx_tensors(device_context_, backend_.be);
    if (weight_buffer_ == nullptr) {
      error_ = "failed to allocate device memory for model weights";
      return false;
    }
    for (int64_t i = 0; i < file_.TensorCount(); ++i) {
      const char *name = file_.TensorName(i);
      ggml_tensor *source = ggml_get_tensor(file_.Context(), name);
      ggml_tensor *target = ggml_get_tensor(device_context_, name);
      if (source->type == target->type) {
        ggml_backend_tensor_set(target, source->data, 0, ggml_nbytes(source));
      } else if (source->type == GGML_TYPE_F16 &&
                 target->type == GGML_TYPE_F32) {
        std::vector<float> values(static_cast<size_t>(ggml_nelements(source)));
        ggml_fp16_to_fp32_row(
            static_cast<const ggml_fp16_t *>(source->data), values.data(),
            static_cast<int64_t>(values.size()));
        ggml_backend_tensor_set(target, values.data(), 0,
                                values.size() * sizeof(float));
      } else {
        error_ = std::string("unsupported CPU weight conversion for ") + name;
        return false;
      }
    }
    ForEachWeight([&](ggml_tensor *&tensor) {
      if (tensor != nullptr) {
        tensor = ggml_get_tensor(device_context_, ggml_get_name(tensor));
      }
    });
    return true;
  }

  void FilterMatches(const std::vector<float> &scores, int64_t tokens0,
                     int64_t tokens1, RawResult *result) const {
    std::vector<int32_t> best0(static_cast<size_t>(tokens0));
    std::vector<int32_t> best1(static_cast<size_t>(tokens1));
    std::vector<float> best_score0(static_cast<size_t>(tokens0));

    for (int64_t i = 0; i < tokens0; ++i) {
      int32_t best = 0;
      float value = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < tokens1; ++j) {
        const float candidate = scores[static_cast<size_t>(j + tokens1 * i)];
        if (candidate > value) {
          value = candidate;
          best = static_cast<int32_t>(j);
        }
      }
      best0[static_cast<size_t>(i)] = best;
      best_score0[static_cast<size_t>(i)] = value;
    }
    for (int64_t j = 0; j < tokens1; ++j) {
      int32_t best = 0;
      float value = -std::numeric_limits<float>::infinity();
      for (int64_t i = 0; i < tokens0; ++i) {
        const float candidate = scores[static_cast<size_t>(j + tokens1 * i)];
        if (candidate > value) {
          value = candidate;
          best = static_cast<int32_t>(i);
        }
      }
      best1[static_cast<size_t>(j)] = best;
    }

    const float threshold = static_cast<float>(options_.min_score);
    for (int64_t i = 0; i < tokens0; ++i) {
      const int32_t j = best0[static_cast<size_t>(i)];
      const float score = std::exp(best_score0[static_cast<size_t>(i)]);
      if (best1[static_cast<size_t>(j)] == i && score > threshold) {
        result->matches0[static_cast<size_t>(i)] = j;
        result->mscores0[static_cast<size_t>(i)] = score;
      }
    }
  }

  void FilterDeviceMatches(const std::vector<int32_t> &best0,
                           const std::vector<int32_t> &mutual1,
                           const std::vector<float> &best_score0,
                           RawResult *result) const {
    const float threshold = static_cast<float>(options_.min_score);
    for (size_t i = 0; i < best0.size(); ++i) {
      const float score = std::exp(best_score0[i]);
      if (mutual1[i] == static_cast<int32_t>(i) && score > threshold) {
        result->matches0[i] = best0[i];
        result->mscores0[i] = score;
      }
    }
  }

  void Release() {
    if (weight_buffer_ != nullptr) {
      ggml_backend_buffer_free(weight_buffer_);
      weight_buffer_ = nullptr;
    }
    if (device_context_ != nullptr) {
      ggml_free(device_context_);
      device_context_ = nullptr;
    }
    backend_.release();
    layers_.clear();
    file_.Close();
  }

  MatchingOptions options_;
  ModelFile file_;
  engine_backend backend_;
  ggml_context *device_context_ = nullptr;
  ggml_backend_buffer_t weight_buffer_ = nullptr;
  ggml_tensor *positional_weight_ = nullptr;
  Linear input_projection_;
  std::vector<LayerWeights> layers_;
  Linear assignment_projection_;
  Linear matchability_;
  std::string error_;
};

bool MatchingOptions::check(std::string *error) const {
  std::string message;
  if (model_path.empty()) {
    message = "model_path must not be empty";
  } else if (num_threads < 0) {
    message = "num_threads must be non-negative";
  } else if (!(min_score >= 0.0 && min_score <= 1.0)) {
    message = "min_score must be in [0, 1]";
  }
  if (error != nullptr) {
    *error = message;
  }
  return message.empty();
}

std::unique_ptr<FeatureMatcher>
create_feature_matcher(const MatchingOptions &options,
                              std::string *error) {
  if (!options.check(error)) {
    return nullptr;
  }
  auto matcher = std::make_unique<FeatureMatcherImpl>(options);
  if (!matcher->Load()) {
    if (error != nullptr) {
      *error = matcher->error();
    }
    return nullptr;
  }
  if (error != nullptr)
    error->clear();
  return matcher;
}

const char *feature_type_name(FeatureType type) {
  switch (type) {
  case FeatureType::kSuperPoint:
    return "superpoint";
  case FeatureType::kDisk:
    return "disk";
  case FeatureType::kAliked:
    return "aliked";
  case FeatureType::kSift:
    return "sift";
  case FeatureType::kDogHardNet:
    return "doghardnet";
  }
  return "unknown";
}

const char *feature_matcher_type_name(FeatureMatcherType type) {
  switch (type) {
  case FeatureMatcherType::kAuto:
    return "auto";
  case FeatureMatcherType::kSiftLightGlue:
    return "sift_lightglue";
  case FeatureMatcherType::kAlikedLightGlue:
    return "aliked_lightglue";
  }
  return "unknown";
}

}  // namespace lightglue
}  // namespace aicore
