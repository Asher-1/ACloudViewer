// DaD detector graph for COLMAP LoMa.  This is a direct lowering of
// loma/detector/dad.py: VGG11-BN (already fused in the ONNX export), the four
// ConvRefiners, bicubic/bilinear half-pixel resampling, and the source's
// keypoint selection formula.  Only the host-side selection is outside ggml;
// it is data-independent model postprocessing and contains no ONNX runtime.

#include "tasks/loma/detector.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tasks/lightglue/backend.hpp"

namespace aicore {
namespace loma {
namespace {

constexpr const char* kArchitecture = "loma";
constexpr const char* kDetectorVariant = "detector_B";
constexpr size_t kMaxGraphNodes = 2048;

class ModelFile {
public:
    ~ModelFile() { Close(); }

    bool Open(const std::string& path) {
        Close();
        error_.clear();
        gguf_init_params params{/*no_alloc=*/false, /*ctx=*/&context_};
        gguf_ = gguf_init_from_file(path.c_str(), params);
        if (gguf_ == nullptr || context_ == nullptr) {
            error_ = "failed to read DaD GGUF model: " + path;
            Close();
            return false;
        }
        if (String("general.architecture") != kArchitecture ||
            String("loma.variant") != kDetectorVariant) {
            if (error_.empty()) {
                error_ = "GGUF is not the pinned LoMa DaD detector";
            }
            Close();
            return false;
        }
        return error_.empty();
    }

    void Close() {
        device_tensors_.clear();
        if (context_ != nullptr) {
            ggml_free(context_);
            context_ = nullptr;
        }
        if (gguf_ != nullptr) {
            gguf_free(gguf_);
            gguf_ = nullptr;
        }
    }

    ggml_tensor* Require(const std::string& name) {
        const auto device_tensor = device_tensors_.find(name);
        if (device_tensor != device_tensors_.end()) return device_tensor->second;
        ggml_tensor* tensor = context_ == nullptr
                ? nullptr : ggml_get_tensor(context_, name.c_str());
        if (tensor == nullptr && error_.empty()) {
            error_ = "DaD GGUF is missing tensor: " + name;
        }
        return tensor;
    }

    int64_t TensorCount() const { return gguf_get_n_tensors(gguf_); }
    const char* TensorName(int64_t index) const {
        return gguf_get_tensor_name(gguf_, index);
    }
    ggml_context* Context() const { return context_; }
    ggml_tensor* Tensor(int64_t index) const {
        return context_ == nullptr ? nullptr
                : ggml_get_tensor(context_, TensorName(index));
    }
    void SetDeviceTensors(
            std::unordered_map<std::string, ggml_tensor*> device_tensors) {
        device_tensors_ = std::move(device_tensors);
    }
    const std::string& error() const { return error_; }

private:
    std::string String(const char* key) {
        const int64_t index = gguf_find_key(gguf_, key);
        if (index < 0 || gguf_get_kv_type(gguf_, index) != GGUF_TYPE_STRING) {
            if (error_.empty()) error_ = std::string("DaD GGUF is missing key: ") + key;
            return {};
        }
        return gguf_get_val_str(gguf_, index);
    }

    gguf_context* gguf_ = nullptr;
    ggml_context* context_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> device_tensors_;
    std::string error_;
};

ggml_tensor* AddBias(ggml_context* context,
                     ggml_tensor* value,
                     ggml_tensor* bias) {
    ggml_tensor* shaped = ggml_reshape_4d(context, bias, 1, 1, bias->ne[0], 1);
    return ggml_add(context, value, shaped);
}

ggml_tensor* SliceChannels(ggml_context* context,
                           ggml_tensor* value,
                           int64_t offset,
                           int64_t channels) {
    return ggml_cont(context, ggml_view_4d(context, value, value->ne[0],
                                            value->ne[1], channels, value->ne[3],
                                            value->nb[1], value->nb[2], value->nb[3],
                                            static_cast<size_t>(offset) * value->nb[2]));
}

struct GraphBuilder {
    GraphBuilder(ggml_context* context, ModelFile* model,
                 bool use_accelerator_convolution)
        : context(context), model(model),
          use_accelerator_convolution(use_accelerator_convolution) {}

    ggml_tensor* Conv(ggml_tensor* input,
                      const std::string& prefix,
                      int padding,
                      bool depthwise = false) {
        ggml_tensor* weight = model->Require("loma." + prefix + ".weight");
        ggml_tensor* bias = model->Require("loma." + prefix + ".bias");
        if (weight == nullptr || bias == nullptr) return nullptr;
        // CUDA's F32 direct-convolution IGEMM path is not numerically stable
        // for LoMa's learned filters. The portable im2col + matmul lowering is
        // already used by the device DPT path and agrees with CPU here. Keep
        // CPU direct convolution and device depthwise convolution unchanged.
        ggml_tensor* output = depthwise
                ? ggml_conv_2d_dw_direct(context, weight, input, 1, 1, padding,
                                          padding, 1, 1)
                : use_accelerator_convolution
                ? ggml_conv_2d(context, weight, input, 1, 1, padding, padding,
                               1, 1)
                : ggml_conv_2d_direct(context, weight, input, 1, 1, padding,
                                      padding, 1, 1);
        return AddBias(context, output, bias);
    }

    ggml_tensor* Refiner(ggml_tensor* feature,
                         ggml_tensor* context_feature,
                         const char* scale) {
        ggml_tensor* input = feature;
        if (context_feature != nullptr) {
            input = ggml_concat(context, input, context_feature, 2);
        }
        const std::string root = std::string("det.decoder.layers.") + scale;
        ggml_tensor* x = Conv(input, root + ".block1.0", 0);
        if (x == nullptr) return nullptr;
        x = ggml_relu(context, x);
        x = Conv(x, root + ".block1.3", 0);
        if (x == nullptr) return nullptr;
        ggml_tensor* residual = x;
        for (int block = 0; block < 3; ++block) {
            const std::string prefix = root + ".hidden_blocks." +
                                       std::to_string(block);
            x = Conv(x, prefix + ".0", 2, true);
            if (x == nullptr) return nullptr;
            x = ggml_relu(context, x);
            x = Conv(x, prefix + ".3", 0);
            if (x == nullptr) return nullptr;
        }
        x = ggml_scale(context, ggml_add(context, x, residual), 1.0f / 1.4f);
        return Conv(x, root + ".out_conv", 0);
    }

    ggml_context* context;
    ModelFile* model;
    bool use_accelerator_convolution;
};

bool ValidateImage(const aicore_loma_rgb_image& image, std::string* error) {
    if (image.rgb == nullptr || image.width < 8 || image.height < 8 ||
        image.row_stride_bytes < image.width * 3) {
        *error = "DaD expects non-null RGB data, dimensions >= 8, and a valid row stride";
        return false;
    }
    const int64_t pixels = static_cast<int64_t>(image.width) * image.height;
    if (pixels > 64LL * 1024LL * 1024LL) {
        *error = "DaD image dimensions exceed the supported allocation limit";
        return false;
    }
    return true;
}

std::vector<float> NormalizeRgbToWhcn(const aicore_loma_rgb_image& image) {
    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    const size_t plane = static_cast<size_t>(image.width) * image.height;
    std::vector<float> result(plane * 3);
    for (int32_t y = 0; y < image.height; ++y) {
        const uint8_t* row = image.rgb + static_cast<size_t>(y) * image.row_stride_bytes;
        for (int32_t x = 0; x < image.width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const float normalized = static_cast<float>(row[3 * x + channel]) /
                                         255.0f;
                result[static_cast<size_t>(x) + static_cast<size_t>(y) * image.width +
                       static_cast<size_t>(channel) * plane] =
                        (normalized - kMean[channel]) / kStd[channel];
            }
        }
    }
    return result;
}

void SelectKeypoints(const std::vector<float>& logits,
                     int32_t width,
                     int32_t height,
                     int32_t count,
                     DetectorResult* result) {
    const size_t pixels = static_cast<size_t>(width) * height;
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(pixels);
    double denominator = 0.0;
    for (size_t index = 0; index < pixels; ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        denominator += probabilities[index];
    }
    for (float& value : probabilities) value /= static_cast<float>(denominator);

    // PyTorch evaluates max_pool2d over the immutable dense map, then applies
    // its equality mask. Do not clear candidates in-place while scanning: that
    // would feed an earlier zero into a later 3x3 window.
    const std::vector<float> dense_probabilities = probabilities;
    std::vector<float> nms_probabilities(pixels, 0.0f);
    std::vector<size_t> candidates(pixels);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * width + x;
            float local_maximum = -std::numeric_limits<float>::infinity();
            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    const int32_t yy = y + dy;
                    const int32_t xx = x + dx;
                    if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                        local_maximum = std::max(local_maximum,
                                dense_probabilities[static_cast<size_t>(yy) * width + xx]);
                    }
                }
            }
            candidates[index] = index;
            if (dense_probabilities[index] == local_maximum) {
                nms_probabilities[index] = dense_probabilities[index];
            }
        }
    }
    std::partial_sort(candidates.begin(), candidates.begin() + count,
                      candidates.end(), [&](size_t left, size_t right) {
                          return nms_probabilities[left] > nms_probabilities[right];
                      });
    result->keypoints.resize(static_cast<size_t>(count));
    result->scores.resize(static_cast<size_t>(count));
    for (int32_t index = 0; index < count; ++index) {
        const size_t flat = candidates[static_cast<size_t>(index)];
        const int32_t y = static_cast<int32_t>(flat / width);
        const int32_t x = static_cast<int32_t>(flat % width);
        float patch[9];
        float patch_maximum = -std::numeric_limits<float>::infinity();
        int patch_index = 0;
        for (int32_t dy = -1; dy <= 1; ++dy) {
            for (int32_t dx = -1; dx <= 1; ++dx) {
                const int32_t yy = y + dy;
                const int32_t xx = x + dx;
                const float value = yy >= 0 && yy < height && xx >= 0 && xx < width
                        ? logits[static_cast<size_t>(yy) * width + xx] : 0.0f;
                patch[patch_index++] = value;
                patch_maximum = std::max(patch_maximum, value / 0.5f);
            }
        }
        float weights[9];
        float weight_sum = 0.0f;
        for (int i = 0; i < 9; ++i) {
            weights[i] = std::exp(patch[i] / 0.5f - patch_maximum);
            weight_sum += weights[i];
        }
        float offset_x = 0.0f;
        float offset_y = 0.0f;
        patch_index = 0;
        for (int32_t dy = -1; dy <= 1; ++dy) {
            for (int32_t dx = -1; dx <= 1; ++dx) {
                const float weight = weights[patch_index++] / weight_sum;
                offset_x += weight * (2.0f * dx / width);
                offset_y += weight * (2.0f * dy / height);
            }
        }
        const float normalized_x = 2.0f * (static_cast<float>(x) + 0.5f) / width -
                                   1.0f + offset_x;
        const float normalized_y = 2.0f * (static_cast<float>(y) + 0.5f) / height -
                                   1.0f + offset_y;
        result->keypoints[static_cast<size_t>(index)] = {
                0.5f * (normalized_x + 1.0f) * width,
                0.5f * (normalized_y + 1.0f) * height};
        result->scores[static_cast<size_t>(index)] = nms_probabilities[flat];
    }
}

}  // namespace

class Detector::Impl {
public:
    ~Impl() { Release(); }

    bool Load(const std::string& path, const DetectorOptions& requested) {
        Release();
        options = requested;
        if (options.max_keypoints <= 0) {
            error = "DaD max_keypoints must be positive";
            return false;
        }
        if (!model.Open(path)) {
            error = model.error();
            return false;
        }
        if (!backend.init(options.device, options.num_threads)) {
            error = backend.error;
            return false;
        }
        if (backend.is_cpu()) {
            void* base = ggml_get_mem_buffer(model.Context());
            const size_t size = ggml_get_mem_size(model.Context());
            weights = ggml_backend_cpu_buffer_from_ptr(base, size);
            if (weights == nullptr) {
                error = "failed to bind DaD GGUF weights to the CPU backend";
                return false;
            }
            for (int64_t index = 0; index < model.TensorCount(); ++index) {
                model.Tensor(index)->buffer = weights;
            }
            return true;
        }
        return OffloadWeights();
    }

    bool OffloadWeights() {
        const auto lock = backend.lock();
        const int64_t count = model.TensorCount();
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * static_cast<size_t>(count + 8);
        params.no_alloc = true;
        device_context = ggml_init(params);
        if (device_context == nullptr) {
            error = "failed to create DaD device-weight context";
            return false;
        }

        std::unordered_map<std::string, ggml_tensor*> device_tensors;
        device_tensors.reserve(static_cast<size_t>(count));
        std::vector<std::pair<ggml_tensor*, const void*>> uploads;
        uploads.reserve(static_cast<size_t>(count));
        std::vector<std::vector<float>> decoded_weights;
        decoded_weights.reserve(static_cast<size_t>(count));
        for (int64_t index = 0; index < count; ++index) {
            ggml_tensor* host = model.Tensor(index);
            if (host == nullptr) {
                error = "DaD GGUF tensor table is inconsistent";
                return false;
            }
            // GGUF also carries integer ONNX constants (slice axes/indices).
            // They are not arithmetic weights and must preserve their type.
            // Only F16 and quantized floating-point weights need promotion so
            // CUDA depthwise convolutions see the same F32 type as activations.
            const bool decode_to_f32 = host->type == GGML_TYPE_F16 ||
                                       ggml_is_quantized(host->type);
            ggml_tensor* device = ggml_new_tensor(
                    device_context, decode_to_f32 ? GGML_TYPE_F32 : host->type,
                    GGML_MAX_DIMS, host->ne);
            if (device == nullptr) {
                error = std::string("failed to create DaD device tensor: ") + host->name;
                return false;
            }
            ggml_set_name(device, host->name);
            device_tensors.emplace(model.TensorName(index), device);
            if (decode_to_f32) {
                const ggml_type_traits* traits = ggml_get_type_traits(host->type);
                const int64_t elements = ggml_nelements(host);
                if (traits == nullptr || traits->to_float == nullptr || elements <= 0) {
                    error = std::string("cannot decode DaD weight for GPU: ") + host->name;
                    return false;
                }
                std::vector<float>& decoded = decoded_weights.emplace_back();
                decoded.resize(static_cast<size_t>(elements));
                traits->to_float(host->data, decoded.data(), elements);
                uploads.emplace_back(device, decoded.data());
            } else {
                uploads.emplace_back(device, host->data);
            }
        }
        device_weights = ggml_backend_alloc_ctx_tensors(device_context, backend.be);
        if (device_weights == nullptr) {
            error = "failed to allocate DaD weights on requested ggml backend";
            return false;
        }
        for (const auto& upload : uploads) {
            ggml_backend_tensor_set(upload.first, upload.second, 0,
                                    ggml_nbytes(upload.first));
        }
        ggml_backend_synchronize(backend.be);
        model.SetDeviceTensors(std::move(device_tensors));
        return true;
    }

    bool Detect(const aicore_loma_rgb_image& image, DetectorResult* result) {
        error.clear();
        if (result == nullptr) {
            error = "DaD result is null";
            return false;
        }
        result->keypoints.clear();
        result->scores.clear();
        if (!ValidateImage(image, &error)) return false;
        const auto lock = backend.lock();
        ggml_init_params params{
                ggml_tensor_overhead() * kMaxGraphNodes +
                        ggml_graph_overhead_custom(kMaxGraphNodes, false),
                nullptr, /*no_alloc=*/true};
        ggml_context* context = ggml_init(params);
        if (context == nullptr) {
            error = "failed to allocate DaD graph context";
            return false;
        }
        const auto free_context = [&] { ggml_free(context); };
        ggml_tensor* input = ggml_new_tensor_4d(context, GGML_TYPE_F32,
                                                 image.width, image.height, 3, 1);
        GraphBuilder graph_builder(context, &model, !backend.is_cpu());
        ggml_tensor* x = graph_builder.Conv(input, "det.encoder.layers.0", 1);
        x = x != nullptr ? ggml_relu(context, x) : nullptr;
        ggml_tensor* f1 = x;
        x = x != nullptr ? ggml_pool_2d(context, x, GGML_OP_POOL_MAX, 2, 2, 2, 2,
                                         0, 0) : nullptr;
        x = x != nullptr ? graph_builder.Conv(x, "det.encoder.layers.4", 1) : nullptr;
        x = x != nullptr ? ggml_relu(context, x) : nullptr;
        ggml_tensor* f2 = x;
        x = x != nullptr ? ggml_pool_2d(context, x, GGML_OP_POOL_MAX, 2, 2, 2, 2,
                                         0, 0) : nullptr;
        x = x != nullptr ? graph_builder.Conv(x, "det.encoder.layers.8", 1) : nullptr;
        x = x != nullptr ? ggml_relu(context, x) : nullptr;
        x = x != nullptr ? graph_builder.Conv(x, "det.encoder.layers.11", 1) : nullptr;
        x = x != nullptr ? ggml_relu(context, x) : nullptr;
        ggml_tensor* f4 = x;
        x = x != nullptr ? ggml_pool_2d(context, x, GGML_OP_POOL_MAX, 2, 2, 2, 2,
                                         0, 0) : nullptr;
        x = x != nullptr ? graph_builder.Conv(x, "det.encoder.layers.15", 1) : nullptr;
        x = x != nullptr ? ggml_relu(context, x) : nullptr;
        x = x != nullptr ? graph_builder.Conv(x, "det.encoder.layers.18", 1) : nullptr;
        x = x != nullptr ? ggml_relu(context, x) : nullptr;
        if (x == nullptr || !model.error().empty()) {
            error = model.error().empty() ? "failed to build DaD encoder graph" : model.error();
            free_context();
            return false;
        }
        ggml_tensor* d8 = graph_builder.Refiner(x, nullptr, "8");
        ggml_tensor* logits = d8 != nullptr ? SliceChannels(context, d8, 0, 1) : nullptr;
        ggml_tensor* context8 = d8 != nullptr ? SliceChannels(context, d8, 1, 256) : nullptr;
        if (logits == nullptr || context8 == nullptr) {
            error = model.error().empty() ? "failed to build DaD scale-8 graph" : model.error();
            free_context();
            return false;
        }
        logits = ggml_interpolate(context, logits, f4->ne[0], f4->ne[1], 1, 1,
                                  GGML_SCALE_MODE_BICUBIC);
        context8 = ggml_interpolate(context, context8, f4->ne[0], f4->ne[1],
                                    context8->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
        ggml_tensor* d4 = graph_builder.Refiner(f4, context8, "4");
        logits = ggml_add(context, logits, SliceChannels(context, d4, 0, 1));
        ggml_tensor* context4 = SliceChannels(context, d4, 1, 128);
        logits = ggml_interpolate(context, logits, f2->ne[0], f2->ne[1], 1, 1,
                                  GGML_SCALE_MODE_BICUBIC);
        context4 = ggml_interpolate(context, context4, f2->ne[0], f2->ne[1],
                                    context4->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
        ggml_tensor* d2 = graph_builder.Refiner(f2, context4, "2");
        logits = ggml_add(context, logits, SliceChannels(context, d2, 0, 1));
        ggml_tensor* context2 = SliceChannels(context, d2, 1, 32);
        logits = ggml_interpolate(context, logits, f1->ne[0], f1->ne[1], 1, 1,
                                  GGML_SCALE_MODE_BICUBIC);
        context2 = ggml_interpolate(context, context2, f1->ne[0], f1->ne[1],
                                    context2->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
        ggml_tensor* d1 = graph_builder.Refiner(f1, context2, "1");
        logits = ggml_add(context, logits, SliceChannels(context, d1, 0, 1));
        if (d4 == nullptr || d2 == nullptr || d1 == nullptr || !model.error().empty()) {
            error = model.error().empty() ? "failed to build DaD decoder graph" : model.error();
            free_context();
            return false;
        }
        ggml_set_output(logits);
        ggml_cgraph* graph = ggml_new_graph_custom(context, kMaxGraphNodes, false);
        ggml_build_forward_expand(graph, logits);
        if (!ggml_gallocr_alloc_graph(backend.galloc, graph)) {
            error = "failed to allocate DaD ggml graph";
            free_context();
            return false;
        }
        const std::vector<float> pixels = NormalizeRgbToWhcn(image);
        ggml_backend_tensor_set(input, pixels.data(), 0, pixels.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
            error = "DaD ggml graph compute failed";
            free_context();
            return false;
        }
        std::vector<float> output(static_cast<size_t>(image.width) * image.height);
        ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
        free_context();
        SelectKeypoints(output, image.width, image.height, options.max_keypoints, result);
        return true;
    }

    void Release() {
        if (device_weights != nullptr) {
            ggml_backend_buffer_free(device_weights);
            device_weights = nullptr;
        }
        if (device_context != nullptr) {
            ggml_free(device_context);
            device_context = nullptr;
        }
        if (weights != nullptr) {
            ggml_backend_buffer_free(weights);
            weights = nullptr;
        }
        backend.release();
        model.Close();
    }

    DetectorOptions options;
    ModelFile model;
    lightglue::engine_backend backend;
    ggml_backend_buffer_t weights = nullptr;
    ggml_context* device_context = nullptr;
    ggml_backend_buffer_t device_weights = nullptr;
    std::string error;
};

Detector::Detector() : impl_(std::make_unique<Impl>()) {}
Detector::~Detector() = default;

bool Detector::Load(const std::string& path, const DetectorOptions& options) {
    return impl_->Load(path, options);
}

bool Detector::Detect(const aicore_loma_rgb_image& image, DetectorResult* result) {
    return impl_->Detect(image, result);
}

const std::string& Detector::error() const { return impl_->error; }

}  // namespace loma
}  // namespace aicore
