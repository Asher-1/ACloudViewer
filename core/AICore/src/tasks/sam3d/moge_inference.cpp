// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "moge_inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <thread>
#include <utility>

#include "backend.hpp"
#include "gguf_loader.hpp"
#include "graph_builder.hpp"
#include "moge_graph.hpp"

namespace sam3d {
namespace {

// The full-resolution pointmap postprocess walks ~30 M pixels several times
// (resample, exp, sigmoid, remap). These passes are per-pixel independent,
// so run them on a thread pool; the per-pixel math is unchanged and the
// result is bit-identical to the scalar version.
unsigned postprocess_thread_count() {
    const unsigned hardware = std::thread::hardware_concurrency();
    return hardware == 0 ? 1u : std::min(hardware, 16u);
}

template <typename Body>
void parallel_for(size_t begin, size_t end, const Body& body) {
    const size_t count = end - begin;
    const unsigned threads = postprocess_thread_count();
    if (count == 0 || threads <= 1) {
        body(begin, end);
        return;
    }
    const size_t chunk = (count + threads - 1) / threads;
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (size_t start = begin; start < end; start += chunk) {
        const size_t stop = std::min(end, start + chunk);
        workers.emplace_back([&body, start, stop] { body(start, stop); });
    }
    for (auto& worker : workers) worker.join();
}

struct HostImage {
    int channels = 0;
    int width = 0;
    int height = 0;
    std::vector<float> values;  // HWC

    float& at(int channel, int x, int y) {
        return values[(static_cast<size_t>(y) * width + x) * channels +
                      channel];
    }
    float at(int channel, int x, int y) const {
        return values[(static_cast<size_t>(y) * width + x) * channels +
                      channel];
    }
};

struct ResizeTap {
    int index = 0;
    float weight = 0.0f;
};

HostImage rgba_to_rgb(const RgbaImage& rgba) {
    HostImage rgb;
    rgb.channels = 3;
    rgb.width = rgba.width;
    rgb.height = rgba.height;
    rgb.values.resize(static_cast<size_t>(rgb.width) * rgb.height *
                      rgb.channels);
    for (size_t pixel = 0; pixel < static_cast<size_t>(rgb.width) * rgb.height;
         ++pixel) {
        for (int channel = 0; channel < rgb.channels; ++channel) {
            // Match InferencePipeline.image_to_float: NumPy promotes the
            // uint8 division then casts the result back to F32.
            rgb.values[pixel * 3 + channel] = static_cast<float>(
                    static_cast<double>(rgba.rgba[pixel * 4 + channel]) /
                    255.0);
        }
    }
    return rgb;
}

float cubic_filter(float distance) {
    // PyTorch's bicubic antialias kernel uses Keys cubic a=-0.5. The
    // a=-0.75 coefficient is reserved for its non-antialiased OpenCV path.
    constexpr float kA = -0.5f;
    const float x = std::fabs(distance);
    if (x <= 1.0f) return ((kA + 2.0f) * x - (kA + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return ((kA * x - 5.0f * kA) * x + 8.0f * kA) * x - 4.0f * kA;
    return 0.0f;
}

std::vector<std::vector<ResizeTap>> cubic_antialias_taps(int input_size,
                                                         int output_size) {
    const float scale =
            static_cast<float>(input_size) / static_cast<float>(output_size);
    const float support = scale >= 1.0f ? 2.0f * scale : 2.0f;
    const float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
    const int max_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    std::vector<std::vector<ResizeTap>> taps(static_cast<size_t>(output_size));
    for (int destination = 0; destination < output_size; ++destination) {
        const float center = scale * (static_cast<float>(destination) + 0.5f);
        const int first =
                std::max(static_cast<int>(center - support + 0.5f), 0);
        int count = std::min(static_cast<int>(center + support + 0.5f),
                             input_size) -
                    first;
        count = std::max(0, std::min(count, max_size));
        float sum = 0.0f;
        auto& axis_taps = taps[static_cast<size_t>(destination)];
        axis_taps.reserve(static_cast<size_t>(count));
        for (int offset = 0; offset < count; ++offset) {
            const int source = first + offset;
            const float distance =
                    (static_cast<float>(source) - center + 0.5f) * invscale;
            const float weight = cubic_filter(distance);
            axis_taps.push_back({source, weight});
            sum += weight;
        }
        if (sum == 0.0f) {
            axis_taps.assign(
                    1, {std::min(input_size - 1, std::max(0, first)), 1.0f});
        } else {
            for (ResizeTap& tap : axis_taps) tap.weight /= sum;
        }
    }
    return taps;
}

HostImage resize_bicubic_antialias(const HostImage& input,
                                   int width,
                                   int height) {
    if (input.width == width && input.height == height) return input;
    const auto x_taps = cubic_antialias_taps(input.width, width);
    const auto y_taps = cubic_antialias_taps(input.height, height);
    HostImage horizontal{
            input.channels, width, input.height,
            std::vector<float>(static_cast<size_t>(input.channels) * width *
                               input.height)};
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto& taps = x_taps[static_cast<size_t>(x)];
            for (int channel = 0; channel < input.channels; ++channel) {
                float value = taps.front().weight *
                              input.at(channel, taps.front().index, y);
                for (size_t tap = 1; tap < taps.size(); ++tap) {
                    value += taps[tap].weight *
                             input.at(channel, taps[tap].index, y);
                }
                horizontal.at(channel, x, y) = value;
            }
        }
    }
    HostImage output{input.channels, width, height,
                     std::vector<float>(static_cast<size_t>(input.channels) *
                                        width * height)};
    for (int y = 0; y < height; ++y) {
        const auto& taps = y_taps[static_cast<size_t>(y)];
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < input.channels; ++channel) {
                float value = taps.front().weight *
                              horizontal.at(channel, x, taps.front().index);
                for (size_t tap = 1; tap < taps.size(); ++tap) {
                    value += taps[tap].weight *
                             horizontal.at(channel, x, taps[tap].index);
                }
                output.at(channel, x, y) = value;
            }
        }
    }
    return output;
}

HostImage resize_bilinear(const HostImage& input, int width, int height) {
    if (input.width == width && input.height == height) return input;
    HostImage output{input.channels, width, height,
                     std::vector<float>(static_cast<size_t>(input.channels) *
                                        width * height)};
    const float scale_x =
            static_cast<float>(input.width) / static_cast<float>(width);
    const float scale_y =
            static_cast<float>(input.height) / static_cast<float>(height);
    parallel_for(0, static_cast<size_t>(height), [&](size_t begin, size_t end) {
        for (size_t y = begin; y < end; ++y) {
            const float source_y = std::max(
                    0.0f, scale_y * (static_cast<float>(y) + 0.5f) - 0.5f);
            const int y0 =
                    std::min(static_cast<int>(source_y), input.height - 1);
            const int y1 = std::min(y0 + 1, input.height - 1);
            const float wy = source_y - static_cast<float>(y0);
            for (int x = 0; x < width; ++x) {
                const float source_x = std::max(
                        0.0f, scale_x * (static_cast<float>(x) + 0.5f) - 0.5f);
                const int x0 =
                        std::min(static_cast<int>(source_x), input.width - 1);
                const int x1 = std::min(x0 + 1, input.width - 1);
                const float wx = source_x - static_cast<float>(x0);
                for (int channel = 0; channel < input.channels; ++channel) {
                    const float top = input.at(channel, x0, y0) +
                                      (input.at(channel, x1, y0) -
                                       input.at(channel, x0, y0)) *
                                              wx;
                    const float bottom = input.at(channel, x0, y1) +
                                         (input.at(channel, x1, y1) -
                                          input.at(channel, x0, y1)) *
                                                 wx;
                    output.at(channel, x, y) = top + (bottom - top) * wy;
                }
            }
        }
    });
    return output;
}

HostImage chw_to_hwc(std::vector<float> values,
                     int width,
                     int height,
                     int channels) {
    HostImage output{
            channels, width, height,
            std::vector<float>(static_cast<size_t>(channels) * width * height)};
    for (int channel = 0; channel < channels; ++channel) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                output.at(channel, x, y) =
                        values[(static_cast<size_t>(channel) * height + y) *
                                       width +
                               x];
            }
        }
    }
    return output;
}

bool upload_moge_constants(Backend& backend,
                           const MogeGraph& builder,
                           std::string& error) {
    size_t f32_index = 0;
    size_t i32_index = 0;
    for (ggml_tensor* tensor : builder.inputs) {
        bool uploaded = false;
        if (tensor->type == GGML_TYPE_F32 &&
            f32_index < builder.f32_data.size()) {
            const auto& values = builder.f32_data[f32_index++];
            uploaded = backend.set_input_f32(tensor, values->data(),
                                             values->size());
        } else if (tensor->type == GGML_TYPE_I32 &&
                   i32_index < builder.i32_data.size()) {
            const auto& values = builder.i32_data[i32_index++];
            uploaded = backend.set_input_i32(tensor, values->data(),
                                             values->size());
        }
        if (!uploaded) {
            error = std::string("failed to upload MoGe graph input ") +
                    tensor->name;
            return false;
        }
    }
    return true;
}

void make_normalized_uv(int width, int height, std::vector<float>& uv) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float span_x = aspect / std::sqrt(1.0f + aspect * aspect);
    const float span_y = 1.0f / std::sqrt(1.0f + aspect * aspect);
    uv.resize(static_cast<size_t>(width) * height * 2);
    for (int y = 0; y < height; ++y) {
        const float v = -span_y * static_cast<float>(height - 1) / height +
                        2.0f * span_y * static_cast<float>(y) / height;
        for (int x = 0; x < width; ++x) {
            const float u = -span_x * static_cast<float>(width - 1) / width +
                            2.0f * span_x * static_cast<float>(x) / width;
            const size_t offset = (static_cast<size_t>(y) * width + x) * 2;
            uv[offset] = u;
            uv[offset + 1] = v;
        }
    }
}

struct FocalShift {
    float focal = 1.0f;
    float shift = 0.0f;
};

FocalShift recover_focal_shift(const HostImage& points,
                               const std::vector<uint8_t>& mask) {
    constexpr int kLowResolution = 64;
    std::vector<float> uv;
    make_normalized_uv(points.width, points.height, uv);
    struct Sample {
        double ux, uy, x, y, z;
    };
    std::vector<Sample> samples;
    samples.reserve(kLowResolution * kLowResolution);
    for (int y = 0; y < kLowResolution; ++y) {
        const int source_y =
                std::min(points.height - 1, y * points.height / kLowResolution);
        for (int x = 0; x < kLowResolution; ++x) {
            const int source_x = std::min(points.width - 1,
                                          x * points.width / kLowResolution);
            const size_t pixel =
                    static_cast<size_t>(source_y) * points.width + source_x;
            if (mask[pixel] == 0) continue;
            const float px = points.at(0, source_x, source_y);
            const float py = points.at(1, source_x, source_y);
            const float pz = points.at(2, source_x, source_y);
            if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz))
                continue;
            const size_t uv_offset = pixel * 2;
            samples.push_back({uv[uv_offset], uv[uv_offset + 1], px, py, pz});
        }
    }
    if (samples.size() < 2) return {};

    // SciPy's reference uses Levenberg-Marquardt over the one-dimensional
    // shift, eliminating focal analytically for every residual evaluation.
    // Use the same objective and initial value. The derivative includes the
    // focal derivative, keeping this native solve stable without SciPy.
    double shift = 0.0;
    double damping = 1.0e-3;
    double previous_cost = std::numeric_limits<double>::infinity();
    FocalShift result;
    for (int iteration = 0; iteration < 100; ++iteration) {
        double numerator = 0.0;
        double denominator = 0.0;
        double numerator_prime = 0.0;
        double denominator_prime = 0.0;
        for (const Sample& sample : samples) {
            const double depth = sample.z + shift;
            if (std::fabs(depth) < 1.0e-12) continue;
            const double qx = sample.x / depth;
            const double qy = sample.y / depth;
            const double qx_prime = -sample.x / (depth * depth);
            const double qy_prime = -sample.y / (depth * depth);
            numerator += qx * sample.ux + qy * sample.uy;
            denominator += qx * qx + qy * qy;
            numerator_prime += qx_prime * sample.ux + qy_prime * sample.uy;
            denominator_prime += 2.0 * (qx * qx_prime + qy * qy_prime);
        }
        if (!(denominator > 1.0e-20)) break;
        const double focal = numerator / denominator;
        const double focal_prime = (numerator_prime * denominator -
                                    numerator * denominator_prime) /
                                   (denominator * denominator);
        double jtj = 0.0;
        double jtr = 0.0;
        double cost = 0.0;
        for (const Sample& sample : samples) {
            const double depth = sample.z + shift;
            if (std::fabs(depth) < 1.0e-12) continue;
            const double qx = sample.x / depth;
            const double qy = sample.y / depth;
            const double qx_prime = -sample.x / (depth * depth);
            const double qy_prime = -sample.y / (depth * depth);
            const double rx = focal * qx - sample.ux;
            const double ry = focal * qy - sample.uy;
            const double jx = focal_prime * qx + focal * qx_prime;
            const double jy = focal_prime * qy + focal * qy_prime;
            cost += rx * rx + ry * ry;
            jtj += jx * jx + jy * jy;
            jtr += jx * rx + jy * ry;
        }
        result.focal = static_cast<float>(focal);
        result.shift = static_cast<float>(shift);
        if (!(jtj > 0.0) || !std::isfinite(cost)) break;
        const double step = -jtr / (jtj + damping);
        if (std::fabs(step) <= 1.0e-7 * (1.0 + std::fabs(shift))) break;
        const double candidate = shift + step;
        if (!std::isfinite(candidate)) break;
        if (cost < previous_cost) {
            previous_cost = cost;
            damping = std::max(damping * 0.1, 1.0e-12);
            shift = candidate;
        } else {
            damping = std::min(damping * 10.0, 1.0e12);
        }
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (const Sample& sample : samples) {
        const double depth = sample.z + shift;
        if (std::fabs(depth) < 1.0e-12) continue;
        const double qx = sample.x / depth;
        const double qy = sample.y / depth;
        numerator += qx * sample.ux + qy * sample.uy;
        denominator += qx * qx + qy * qy;
    }
    if (denominator > 1.0e-20)
        result.focal = static_cast<float>(numerator / denominator);
    result.shift = static_cast<float>(shift);
    return result;
}

void fill_intrinsics(int width,
                     int height,
                     float focal,
                     std::array<float, 9>& intrinsics) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float diagonal_factor = std::sqrt(1.0f + aspect * aspect);
    const float fx = focal * 0.5f * diagonal_factor / aspect;
    const float fy = focal * 0.5f * diagonal_factor;
    intrinsics = {fx, 0.0f, 0.5f, 0.0f, fy, 0.5f, 0.0f, 0.0f, 1.0f};
}

}  // namespace

bool moge_resized_dimensions(int original_width,
                             int original_height,
                             int num_tokens,
                             int& resized_width,
                             int& resized_height,
                             std::string& error) {
    if (original_width <= 0 || original_height <= 0 || num_tokens <= 0) {
        error = "MoGe image dimensions and num_tokens must be positive";
        return false;
    }
    const double factor =
            std::sqrt((static_cast<double>(num_tokens) * 14.0 * 14.0) /
                      (static_cast<double>(original_width) * original_height));
    resized_width =
            static_cast<int>(static_cast<double>(original_width) * factor);
    resized_height =
            static_cast<int>(static_cast<double>(original_height) * factor);
    if (resized_width < 14 || resized_height < 14) {
        error = "MoGe target resolution is below the 14-pixel image patch size";
        return false;
    }
    return true;
}

bool run_moge_inference(const RgbaImage& image,
                        const GGUFModel& model,
                        Backend& backend,
                        const MogeInferenceOptions& options,
                        MogeInferenceResult& output,
                        std::string& error) {
    if (image.width <= 0 || image.height <= 0 ||
        image.rgba.size() !=
                static_cast<size_t>(image.width) * image.height * 4) {
        error = "MoGe inference requires a non-empty RGBA input";
        return false;
    }
    if (model.str("sam3d.model") != "moge_vitl") {
        error = "MoGe inference requires a moge_vitl GGUF";
        return false;
    }
    int resized_width = 0;
    int resized_height = 0;
    if (!moge_resized_dimensions(image.width, image.height, options.num_tokens,
                                 resized_width, resized_height, error)) {
        return false;
    }
    HostImage rgb = resize_bicubic_antialias(rgba_to_rgb(image), resized_width,
                                             resized_height);

    GraphContext context;
    MogeGraph builder;
    builder.g = &context;
    builder.m = &model;
    ggml_tensor* input =
            context.input_f32("moge_rgb", {3, resized_width, resized_height});
    MogeOutputs graph_output = builder.build(input);
    ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 32768, false);
    ggml_set_output(graph_output.raw_points);
    ggml_set_output(graph_output.mask_logits);
    if (options.capture_block_outputs) {
        ggml_set_output(graph_output.backbone_image);
        ggml_set_output(graph_output.backbone_patch_tokens);
        ggml_set_output(graph_output.backbone_position_tokens);
        ggml_set_output(graph_output.backbone_input);
        ggml_set_output(graph_output.debug_attention_context);
        ggml_set_output(graph_output.debug_q);
        ggml_set_output(graph_output.debug_k);
        ggml_set_output(graph_output.debug_v);
        for (ggml_tensor* tensor :
             graph_output.backbone_attention_projection_outputs) {
            ggml_set_output(tensor);
        }
        for (ggml_tensor* tensor : graph_output.backbone_attention_outputs)
            ggml_set_output(tensor);
        for (ggml_tensor* tensor : graph_output.backbone_mlp_fc1_outputs)
            ggml_set_output(tensor);
        for (ggml_tensor* tensor : graph_output.backbone_mlp_gelu_outputs)
            ggml_set_output(tensor);
        for (ggml_tensor* tensor : graph_output.backbone_mlp_outputs)
            ggml_set_output(tensor);
        for (ggml_tensor* tensor : graph_output.backbone_block_outputs)
            ggml_set_output(tensor);
    }
    ggml_build_forward_expand(graph, graph_output.raw_points);
    ggml_build_forward_expand(graph, graph_output.mask_logits);
    if (options.capture_block_outputs) {
        ggml_build_forward_expand(graph, graph_output.backbone_image);
        ggml_build_forward_expand(graph, graph_output.backbone_patch_tokens);
        ggml_build_forward_expand(graph, graph_output.backbone_position_tokens);
        ggml_build_forward_expand(graph, graph_output.backbone_input);
        ggml_build_forward_expand(graph, graph_output.debug_attention_context);
        ggml_build_forward_expand(graph, graph_output.debug_q);
        ggml_build_forward_expand(graph, graph_output.debug_k);
        ggml_build_forward_expand(graph, graph_output.debug_v);
        for (ggml_tensor* tensor :
             graph_output.backbone_attention_projection_outputs) {
            ggml_build_forward_expand(graph, tensor);
        }
        for (ggml_tensor* tensor : graph_output.backbone_attention_outputs) {
            ggml_build_forward_expand(graph, tensor);
        }
        for (ggml_tensor* tensor : graph_output.backbone_mlp_fc1_outputs) {
            ggml_build_forward_expand(graph, tensor);
        }
        for (ggml_tensor* tensor : graph_output.backbone_mlp_gelu_outputs) {
            ggml_build_forward_expand(graph, tensor);
        }
        for (ggml_tensor* tensor : graph_output.backbone_mlp_outputs) {
            ggml_build_forward_expand(graph, tensor);
        }
        for (ggml_tensor* tensor : graph_output.backbone_block_outputs) {
            ggml_build_forward_expand(graph, tensor);
        }
    }
    if (!backend.alloc(graph)) {
        error = "failed to allocate the MoGe inference graph";
        return false;
    }
    if (!backend.set_input_f32(input, rgb.values.data(), rgb.values.size()) ||
        !upload_moge_constants(backend, builder, error) ||
        !backend.run(graph)) {
        if (error.empty()) error = "failed to execute the MoGe inference graph";
        return false;
    }
    std::vector<float> raw_points;
    std::vector<float> raw_mask;
    if (!backend.get_tensor_f32(graph_output.raw_points, raw_points) ||
        !backend.get_tensor_f32(graph_output.mask_logits, raw_mask)) {
        error = "failed to retrieve the MoGe inference outputs";
        return false;
    }
    const size_t resized_pixels =
            static_cast<size_t>(resized_width) * resized_height;
    if (raw_points.size() != resized_pixels * 3 ||
        raw_mask.size() != resized_pixels) {
        error = "MoGe graph returned an unexpected output shape";
        return false;
    }
    output = {};
    output.width = image.width;
    output.height = image.height;
    output.resized_width = resized_width;
    output.resized_height = resized_height;
    if (options.capture_block_outputs) {
        const auto capture = [&](const std::vector<ggml_tensor*>& tensors,
                                 std::vector<std::vector<float>>& destination) {
            destination.resize(tensors.size());
            for (size_t index = 0; index < tensors.size(); ++index) {
                if (!backend.get_tensor_f32(tensors[index], destination[index]))
                    return false;
            }
            return true;
        };
        if (!backend.get_tensor_f32(graph_output.backbone_image,
                                    output.backbone_image) ||
            !backend.get_tensor_f32(graph_output.backbone_patch_tokens,
                                    output.backbone_patch_tokens) ||
            !backend.get_tensor_f32(graph_output.backbone_position_tokens,
                                    output.backbone_position_tokens) ||
            !backend.get_tensor_f32(graph_output.backbone_input,
                                    output.backbone_input) ||
            !backend.get_tensor_f32(graph_output.debug_attention_context,
                                    output.backbone_attention_context) ||
            !backend.get_tensor_f32(graph_output.debug_q, output.backbone_q) ||
            !backend.get_tensor_f32(graph_output.debug_k, output.backbone_k) ||
            !backend.get_tensor_f32(graph_output.debug_v, output.backbone_v) ||
            !capture(graph_output.backbone_attention_projection_outputs,
                     output.backbone_attention_projection) ||
            !capture(graph_output.backbone_attention_outputs,
                     output.backbone_attention) ||
            !capture(graph_output.backbone_mlp_fc1_outputs,
                     output.backbone_mlp_fc1) ||
            !capture(graph_output.backbone_mlp_gelu_outputs,
                     output.backbone_mlp_gelu) ||
            !capture(graph_output.backbone_mlp_outputs, output.backbone_mlp) ||
            !capture(graph_output.backbone_block_outputs,
                     output.backbone_blocks)) {
            error = "failed to retrieve MoGe DINO block regression tensors";
            return false;
        }
        if (graph_output.backbone_block_outputs.empty() ||
            graph_output.backbone_mlp_fc1_outputs.empty()) {
            error = "MoGe graph did not expose DINO block regression tensors";
            return false;
        }
        output.backbone_tokens =
                static_cast<int>(graph_output.backbone_input->ne[1]);
        output.backbone_hidden =
                static_cast<int>(graph_output.backbone_input->ne[0]);
        output.backbone_image_width =
                static_cast<int>(graph_output.backbone_image->ne[1]);
        output.backbone_image_height =
                static_cast<int>(graph_output.backbone_image->ne[2]);
        output.backbone_mlp_hidden = static_cast<int>(
                graph_output.backbone_mlp_fc1_outputs.front()->ne[0]);
    }
    // ggml Conv2d tensors are [W,H,C,N] and therefore download as CHW. The
    // host resamplers use HWC, so make the storage transition explicit.
    HostImage raw =
            chw_to_hwc(std::move(raw_points), resized_width, resized_height, 3);
    HostImage raw_mask_image{1, resized_width, resized_height,
                             std::move(raw_mask)};
    HostImage points = resize_bilinear(raw, image.width, image.height);
    HostImage mask_logits =
            resize_bilinear(raw_mask_image, image.width, image.height);

    if (options.capture_intermediates) output.resized_rgb = rgb.values;
    const size_t pixels = static_cast<size_t>(image.width) * image.height;
    output.points_moge.resize(pixels * 3);
    output.pointmap_pytorch3d.resize(pixels * 3);
    output.mask_logits = std::move(mask_logits.values);
    output.mask_probability.resize(pixels);
    output.depth.resize(pixels);
    output.mask.resize(pixels);
    const float threshold = model.f32("moge.mask_threshold", 0.5f);
    parallel_for(0, pixels, [&](size_t begin, size_t end) {
        for (size_t pixel = begin; pixel < end; ++pixel) {
            const float mask_logit = output.mask_logits[pixel];
            output.mask[pixel] = mask_logit > threshold ? 1 : 0;
            output.mask_probability[pixel] =
                    1.0f / (1.0f + std::exp(-mask_logit));
        }
    });
    parallel_for(0, static_cast<size_t>(image.height),
                 [&](size_t begin, size_t end) {
                     for (size_t y = begin; y < end; ++y) {
                         for (int x = 0; x < image.width; ++x) {
                             const size_t pixel =
                                     static_cast<size_t>(y) * image.width + x;
                             const float z = std::exp(points.at(2, x, y));
                             output.points_moge[pixel * 3] =
                                     points.at(0, x, y) * z;
                             output.points_moge[pixel * 3 + 1] =
                                     points.at(1, x, y) * z;
                             output.points_moge[pixel * 3 + 2] = z;
                         }
                     }
                 });
    if (options.capture_intermediates)
        output.forward_points = output.points_moge;
    HostImage remapped{3, image.width, image.height, output.points_moge};
    const FocalShift focal_shift = recover_focal_shift(remapped, output.mask);
    output.focal = focal_shift.focal;
    output.shift = focal_shift.shift;
    fill_intrinsics(image.width, image.height, output.focal, output.intrinsics);

    parallel_for(
            0, static_cast<size_t>(image.height),
            [&](size_t begin, size_t end) {
                for (size_t y = begin; y < end; ++y) {
                    for (int x = 0; x < image.width; ++x) {
                        const size_t pixel =
                                static_cast<size_t>(y) * image.width + x;
                        float px = output.points_moge[pixel * 3];
                        float py = output.points_moge[pixel * 3 + 1];
                        float pz = output.points_moge[pixel * 3 + 2] +
                                   output.shift;
                        if (options.force_projection) {
                            const float u = (static_cast<float>(x) + 0.5f) /
                                            image.width;
                            const float v = (static_cast<float>(y) + 0.5f) /
                                            image.height;
                            px = (u - output.intrinsics[2]) * pz /
                                 output.intrinsics[0];
                            py = (v - output.intrinsics[5]) * pz /
                                 output.intrinsics[4];
                        }
                        if (options.apply_mask && output.mask[pixel] == 0) {
                            px = py = pz =
                                    std::numeric_limits<float>::infinity();
                        }
                        output.points_moge[pixel * 3] = px;
                        output.points_moge[pixel * 3 + 1] = py;
                        output.points_moge[pixel * 3 + 2] = pz;
                        output.depth[pixel] = pz;
                        // camera_to_pytorch3d_camera() is a pure diagonal
                        // rotation for the official eye/at/up tuple: (-x, -y,
                        // z).
                        output.pointmap_pytorch3d[(0 * image.height + y) *
                                                          image.width +
                                                  x] = -px;
                        output.pointmap_pytorch3d[(1 * image.height + y) *
                                                          image.width +
                                                  x] = -py;
                        output.pointmap_pytorch3d[(2 * image.height + y) *
                                                          image.width +
                                                  x] = pz;
                    }
                }
            });
    return true;
}

}  // namespace sam3d
