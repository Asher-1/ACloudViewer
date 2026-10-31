// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "image_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <numeric>

#include "common.hpp"

namespace sam3d {
namespace {

struct ImageTensor {
    int channels = 0;
    int width = 0;
    int height = 0;
    std::vector<float> values;  // CHW

    float& at(int channel, int x, int y) {
        return values[(static_cast<size_t>(channel) * height + y) * width + x];
    }
    float at(int channel, int x, int y) const {
        return values[(static_cast<size_t>(channel) * height + y) * width + x];
    }
};

size_t element_count(const ImageTensor& image) {
    return static_cast<size_t>(image.channels) * image.width * image.height;
}

bool valid_image(const ImageTensor& image) {
    return image.channels > 0 && image.width > 0 && image.height > 0 &&
           image.values.size() == element_count(image);
}

ImageTensor from_rgba(const RgbaImage& rgba) {
    ImageTensor image;
    image.channels = 4;
    image.width = rgba.width;
    image.height = rgba.height;
    image.values.resize(static_cast<size_t>(image.channels) * image.width *
                        image.height);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const size_t pixel = (static_cast<size_t>(y) * image.width + x) * 4;
            for (int channel = 0; channel < 4; ++channel) {
                // InferencePipelinePointMap::image_to_float performs NumPy's
                // uint8 / 255 promotion in float64, then casts to float32.
                // Cast after the division so the native tensor retains the
                // same nearest float32 value rather than the float32-divide
                // result used by the C++ default expression.
                image.at(channel, x, y) = static_cast<float>(
                        static_cast<double>(rgba.rgba[pixel + channel]) /
                        255.0);
            }
        }
    }
    return image;
}

ImageTensor select_channels(const ImageTensor& input, int first, int count) {
    ImageTensor output;
    output.channels = count;
    output.width = input.width;
    output.height = input.height;
    output.values.resize(element_count(output));
    for (int channel = 0; channel < count; ++channel) {
        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                output.at(channel, x, y) = input.at(first + channel, x, y);
            }
        }
    }
    return output;
}

ImageTensor make_image(int channels, int width, int height, float fill) {
    ImageTensor output;
    output.channels = channels;
    output.width = width;
    output.height = height;
    output.values.assign(static_cast<size_t>(channels) * width * height, fill);
    return output;
}

ImageTensor crop_with_padding(
        const ImageTensor& input, int x1, int y1, int x2, int y2, float fill) {
    const int width = x2 - x1;
    const int height = y2 - y1;
    ImageTensor output = make_image(input.channels, width, height, fill);
    for (int y = 0; y < height; ++y) {
        const int source_y = y1 + y;
        if (source_y < 0 || source_y >= input.height) continue;
        for (int x = 0; x < width; ++x) {
            const int source_x = x1 + x;
            if (source_x < 0 || source_x >= input.width) continue;
            for (int channel = 0; channel < input.channels; ++channel) {
                output.at(channel, x, y) =
                        input.at(channel, source_x, source_y);
            }
        }
    }
    return output;
}

ImageTensor pad_to_square_centered(const ImageTensor& input, float fill) {
    const int size = std::max(input.width, input.height);
    ImageTensor output = make_image(input.channels, size, size, fill);
    const int offset_x = (size - input.width) / 2;
    const int offset_y = (size - input.height) / 2;
    for (int channel = 0; channel < input.channels; ++channel) {
        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                output.at(channel, x + offset_x, y + offset_y) =
                        input.at(channel, x, y);
            }
        }
    }
    return output;
}

int nearest_source_index(int destination, int input_size, int output_size) {
    return std::min(input_size - 1, destination * input_size / output_size);
}

ImageTensor resize_nearest(const ImageTensor& input, int width, int height) {
    ImageTensor output = make_image(input.channels, width, height, 0.0f);
    for (int y = 0; y < height; ++y) {
        const int source_y = nearest_source_index(y, input.height, height);
        for (int x = 0; x < width; ++x) {
            const int source_x = nearest_source_index(x, input.width, width);
            for (int channel = 0; channel < input.channels; ++channel) {
                output.at(channel, x, y) =
                        input.at(channel, source_x, source_y);
            }
        }
    }
    return output;
}

// The released pipeline config uses torchvision Resize's default bilinear
// interpolation. Keep PyTorch's source index span semantics: crop the span
// before normalizing rather than folding out-of-bounds weights into edge
// pixels.
struct ResizeTap {
    int index = 0;
    float weight = 0.0f;
};

std::vector<std::vector<ResizeTap>> make_bilinear_taps(int input_size,
                                                       int output_size) {
    // This is HelperInterpBase::_compute_indices_min_size_weights_aa from
    // PyTorch 2.5.1's aten/src/ATen/native/cpu/UpSampleKernel.cpp, specialized
    // for the linear filter and scalar_t=float. `scale` is input/output.
    const float scale =
            static_cast<float>(input_size) / static_cast<float>(output_size);
    const float support = scale >= 1.0f ? scale : 1.0f;
    const int max_interp_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    std::vector<std::vector<ResizeTap>> taps(static_cast<size_t>(output_size));
    for (int destination = 0; destination < output_size; ++destination) {
        const float center = scale * (static_cast<float>(destination) + 0.5f);
        const float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
        const int first =
                std::max(static_cast<int>(center - support + 0.5f), 0);
        int count = std::min(static_cast<int>(center + support + 0.5f),
                             input_size) -
                    first;
        count = std::max(0, std::min(count, max_interp_size));
        float sum = 0.0f;
        std::vector<ResizeTap>& axis_taps =
                taps[static_cast<size_t>(destination)];
        axis_taps.reserve(static_cast<size_t>(count));
        for (int offset = 0; offset < count; ++offset) {
            const int source = first + offset;
            const float distance =
                    (static_cast<float>(offset + first) - center + 0.5f) *
                    invscale;
            const float weight = std::max(0.0f, 1.0f - std::fabs(distance));
            axis_taps.push_back({source, weight});
            sum += weight;
        }
        if (sum == 0.0f) {
            const int closest = std::max(0, std::min(input_size - 1, first));
            axis_taps.clear();
            axis_taps.push_back({closest, 1.0f});
            continue;
        }
        for (ResizeTap& tap : axis_taps) {
            tap.weight /= sum;
        }
    }
    return taps;
}

ImageTensor resize_bilinear_antialias(const ImageTensor& input,
                                      int width,
                                      int height) {
    if (input.width == width && input.height == height) return input;
    const auto x_taps = make_bilinear_taps(input.width, width);
    const auto y_taps = make_bilinear_taps(input.height, height);
    ImageTensor horizontal =
            make_image(input.channels, width, input.height, 0.0f);
    for (int channel = 0; channel < input.channels; ++channel) {
        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::vector<ResizeTap>& taps =
                        x_taps[static_cast<size_t>(x)];
                float value = taps.front().weight *
                              input.at(channel, taps.front().index, y);
                for (size_t tap_index = 1; tap_index < taps.size();
                     ++tap_index) {
                    const ResizeTap& tap = taps[tap_index];
                    value += tap.weight * input.at(channel, tap.index, y);
                }
                horizontal.at(channel, x, y) = value;
            }
        }
    }
    ImageTensor output = make_image(input.channels, width, height, 0.0f);
    for (int channel = 0; channel < input.channels; ++channel) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::vector<ResizeTap>& taps =
                        y_taps[static_cast<size_t>(y)];
                float value = taps.front().weight *
                              horizontal.at(channel, x, taps.front().index);
                for (size_t tap_index = 1; tap_index < taps.size();
                     ++tap_index) {
                    const ResizeTap& tap = taps[tap_index];
                    value += tap.weight * horizontal.at(channel, x, tap.index);
                }
                output.at(channel, x, y) = value;
            }
        }
    }
    return output;
}

bool make_crop_bounds(const ImageTensor& mask,
                      float factor,
                      int& x1,
                      int& y1,
                      int& x2,
                      int& y2,
                      std::string& error) {
    int min_x = mask.width;
    int min_y = mask.height;
    int max_x = -1;
    int max_y = -1;
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.at(0, x, y) == 0.0f) continue;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }
    if (max_x < 0 || max_y < 0) {
        error = "input alpha mask is empty";
        return false;
    }
    const int bbox_width = max_x - min_x;
    const int bbox_height = max_y - min_y;
    if (bbox_width < 2 || bbox_height < 2) {
        error = "input alpha mask bounding box must be at least 2x2";
        return false;
    }
    const int size =
            static_cast<int>(std::max({bbox_width, bbox_height, 2}) * factor);
    if (size <= 0) {
        error = "invalid crop size";
        return false;
    }
    const float center_x = (static_cast<float>(min_x) + max_x) / 2.0f;
    const float center_y = (static_cast<float>(min_y) + max_y) / 2.0f;
    x1 = static_cast<int>(center_x - size / 2);
    y1 = static_cast<int>(center_y - size / 2);
    x2 = static_cast<int>(center_x + size / 2);
    y2 = static_cast<int>(center_y + size / 2);
    if (x2 <= x1 || y2 <= y1) {
        error = "official crop calculation produced an empty region";
        return false;
    }
    return true;
}

bool lower_nan_median(std::vector<float> values, float& result) {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](float value) { return std::isnan(value); }),
                 values.end());
    if (values.empty()) return false;
    const size_t index = (values.size() - 1) / 2;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    result = values[index];
    return true;
}

bool normalize_object_centric(const ImageTensor& raw_pointmap,
                              const ImageTensor& raw_mask,
                              ImageTensor& normalized,
                              std::array<float, 3>& scale,
                              std::array<float, 3>& shift,
                              std::string& error) {
    std::array<std::vector<float>, 3> masked;
    for (int channel = 0; channel < 3; ++channel)
        masked[channel].reserve(raw_mask.width * raw_mask.height);
    for (int y = 0; y < raw_mask.height; ++y) {
        for (int x = 0; x < raw_mask.width; ++x) {
            if (raw_mask.at(0, x, y) <= 0.5f) continue;
            for (int channel = 0; channel < 3; ++channel) {
                masked[channel].push_back(raw_pointmap.at(channel, x, y));
            }
        }
    }
    bool has_finite = false;
    for (const auto& values : masked) {
        has_finite = has_finite ||
                     std::any_of(values.begin(), values.end(), [](float value) {
                         return std::isfinite(value);
                     });
    }
    if (!has_finite) {
        error = "pointmap has no finite values within the input alpha mask";
        return false;
    }
    for (int channel = 0; channel < 3; ++channel) {
        if (!lower_nan_median(masked[channel], shift[channel])) {
            error = "cannot compute ObjectCentricSSI shift";
            return false;
        }
    }
    std::vector<float> max_dimensions;
    max_dimensions.reserve(static_cast<size_t>(raw_pointmap.width) *
                           raw_pointmap.height);
    for (int y = 0; y < raw_pointmap.height; ++y) {
        for (int x = 0; x < raw_pointmap.width; ++x) {
            float maximum = 0.0f;
            bool any_nan = false;
            for (int channel = 0; channel < 3; ++channel) {
                const float centered =
                        raw_pointmap.at(channel, x, y) - shift[channel];
                any_nan = any_nan || std::isnan(centered);
                maximum = std::max(maximum, std::fabs(centered));
            }
            max_dimensions.push_back(
                    any_nan ? std::numeric_limits<float>::quiet_NaN()
                            : maximum);
        }
    }
    float scalar_scale = 0.0f;
    if (!lower_nan_median(std::move(max_dimensions), scalar_scale) ||
        !std::isfinite(scalar_scale) || scalar_scale <= 0.0f) {
        error = "cannot compute a positive finite ObjectCentricSSI scale";
        return false;
    }
    scale = {scalar_scale, scalar_scale, scalar_scale};
    normalized = make_image(3, raw_pointmap.width, raw_pointmap.height, 0.0f);

    // `_apply_metric_to_ssi` does not evaluate `(point - shift) / scale`
    // directly. It builds `Transform3d().scale(scale).translate(shift)`,
    // inverts the homogeneous matrix, then calls `transform_points`. For an
    // affine scale/translation this yields these two matrix terms. Keep the
    // same F32 operation order: it is mathematically equivalent to direct
    // subtraction/division, but differs by one ULP for real MoGe pointmaps.
    std::array<float, 3> inverse_scale{};
    std::array<float, 3> inverse_translation{};
    for (int channel = 0; channel < 3; ++channel) {
        inverse_scale[channel] = 1.0f / scale[channel];
        inverse_translation[channel] = -shift[channel] * inverse_scale[channel];
    }
    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < raw_pointmap.height; ++y) {
            for (int x = 0; x < raw_pointmap.width; ++x) {
                normalized.at(channel, x, y) = raw_pointmap.at(channel, x, y) *
                                                       inverse_scale[channel] +
                                               inverse_translation[channel];
            }
        }
    }
    return true;
}

ImageTensor resize_pointmap_to_image(const ImageTensor& pointmap,
                                     int width,
                                     int height) {
    if (pointmap.width == width && pointmap.height == height) {
        // The official chain feeds the preprocessor a fully finite point map
        // (its raw receipt has zero NaN/Inf), so a same-size pass-through is
        // also the official behavior - no mask-out, no re-fill.
        return pointmap;
    }
    ImageTensor clean = pointmap;
    for (int y = 0; y < pointmap.height; ++y) {
        for (int x = 0; x < pointmap.width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const float value = clean.at(channel, x, y);
                // Inf participates in the bilinear weights as Inf*0 = NaN, so
                // both non-finite kinds must be neutralized before the resize.
                if (!std::isfinite(value)) clean.at(channel, x, y) = 0.0f;
            }
        }
    }
    ImageTensor output = resize_bilinear_antialias(clean, width, height);
    // No NaN re-fill: the official semantics keep every pixel finite (the
    // raw predicted values outside the predicted mask), and re-spreading a
    // NaN invalid mask here is exactly what poisoned the PointPatch tokens
    // of the diverged scene objects.
    return output;
}

bool save_condition(const std::filesystem::path& directory,
                    const char* name,
                    const std::vector<int64_t>& shape,
                    const std::vector<float>& values) {
    return save_raw_tensor_f32((directory / name).string(), shape,
                               values.data());
}

}  // namespace

bool preprocess_ss_conditions(const RgbaImage& rgba,
                              const std::vector<float>& pointmap,
                              int pointmap_width,
                              int pointmap_height,
                              const NativePreprocessConfig& config,
                              NativeConditionInputs& output,
                              std::string& error) {
    if (rgba.width <= 0 || rgba.height <= 0 ||
        rgba.rgba.size() != static_cast<size_t>(rgba.width) * rgba.height * 4) {
        error = "input must be a non-empty decoded RGBA image";
        return false;
    }
    if (config.target_size <= 0 || config.crop_box_size_factor <= 0.0f) {
        error = "target size and crop factor must be positive";
        return false;
    }
    if (pointmap_width <= 0 || pointmap_height <= 0 ||
        pointmap.size() !=
                static_cast<size_t>(pointmap_width) * pointmap_height * 3) {
        error = "pointmap must be contiguous F32 CHW with exactly 3 channels";
        return false;
    }

    ImageTensor rgba_tensor = from_rgba(rgba);
    ImageTensor raw_image = select_channels(rgba_tensor, 0, 3);
    ImageTensor raw_mask = select_channels(rgba_tensor, 3, 1);
    ImageTensor raw_pointmap;
    raw_pointmap.channels = 3;
    raw_pointmap.width = pointmap_width;
    raw_pointmap.height = pointmap_height;
    raw_pointmap.values = pointmap;
    raw_pointmap = resize_pointmap_to_image(raw_pointmap, raw_image.width,
                                            raw_image.height);

    ImageTensor normalized_pointmap;
    if (!normalize_object_centric(raw_pointmap, raw_mask, normalized_pointmap,
                                  output.pointmap_scale, output.pointmap_shift,
                                  error)) {
        return false;
    }

    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    if (!make_crop_bounds(raw_mask, config.crop_box_size_factor, x1, y1, x2, y2,
                          error)) {
        return false;
    }

    ImageTensor cropped_image =
            crop_with_padding(raw_image, x1, y1, x2, y2, 0.0f);
    ImageTensor cropped_mask =
            crop_with_padding(raw_mask, x1, y1, x2, y2, 0.0f);
    // The official receipt keeps the cropped point map FULLY FINITE: its
    // joint crop runs with padding_factor 0 and the downstream square pad
    // fills zeros (the dumped official ss_input_pointmap has zero NaN).
    // The former NaN pad here poisoned the PointPatch token segment and
    // diverged the seven small-object trajectories.
    ImageTensor cropped_pointmap =
            crop_with_padding(normalized_pointmap, x1, y1, x2, y2, 0.0f);

    cropped_image = resize_bilinear_antialias(
            pad_to_square_centered(cropped_image, 0.0f), config.target_size,
            config.target_size);
    cropped_mask = resize_nearest(pad_to_square_centered(cropped_mask, 0.0f),
                                  config.target_size, config.target_size);
    cropped_pointmap =
            resize_nearest(pad_to_square_centered(cropped_pointmap, 0.0f),
                           config.target_size, config.target_size);

    ImageTensor full_image =
            resize_bilinear_antialias(pad_to_square_centered(raw_image, 0.0f),
                                      config.target_size, config.target_size);
    ImageTensor full_mask =
            resize_nearest(pad_to_square_centered(raw_mask, 0.0f),
                           config.target_size, config.target_size);
    // The configured pointmap_transform calls pad_to_square_centered directly,
    // whose default fill is zero. NaN fill applies only inside the preceding
    // joint crop transform where the pointmap is passed explicitly.
    ImageTensor full_pointmap =
            resize_nearest(pad_to_square_centered(normalized_pointmap, 0.0f),
                           config.target_size, config.target_size);

    if (!valid_image(cropped_image) || !valid_image(cropped_mask) ||
        !valid_image(cropped_pointmap) || !valid_image(full_image) ||
        !valid_image(full_mask) || !valid_image(full_pointmap)) {
        error = "native preprocessing produced an invalid tensor";
        return false;
    }
    output.width = config.target_size;
    output.height = config.target_size;
    output.image = std::move(cropped_image.values);
    output.mask = std::move(cropped_mask.values);
    output.pointmap = std::move(cropped_pointmap.values);
    output.rgb_image = std::move(full_image.values);
    output.rgb_image_mask = std::move(full_mask.values);
    output.rgb_pointmap = std::move(full_pointmap.values);
    return true;
}

bool write_ss_conditions(const std::string& directory,
                         const NativeConditionInputs& input,
                         std::string& error) {
    if (input.width <= 0 || input.height <= 0) {
        error = "condition tensors have invalid dimensions";
        return false;
    }
    const size_t image_elements =
            static_cast<size_t>(input.width) * input.height * 3;
    const size_t mask_elements =
            static_cast<size_t>(input.width) * input.height;
    if (input.image.size() != image_elements ||
        input.rgb_image.size() != image_elements ||
        input.pointmap.size() != image_elements ||
        input.rgb_pointmap.size() != image_elements ||
        input.mask.size() != mask_elements ||
        input.rgb_image_mask.size() != mask_elements) {
        error = "condition tensor payload sizes do not match their dimensions";
        return false;
    }
    std::error_code filesystem_error;
    const std::filesystem::path root(directory);
    std::filesystem::create_directories(root, filesystem_error);
    if (filesystem_error) {
        error = "cannot create condition directory '" + directory +
                "': " + filesystem_error.message();
        return false;
    }
    const std::vector<int64_t> image_shape = {input.width, input.height, 3, 1};
    const std::vector<int64_t> mask_shape = {input.width, input.height, 1, 1};
    const std::vector<int64_t> vector_shape = {3, 1};
    if (!save_condition(root, "ss_input_image.samt", image_shape,
                        input.image) ||
        !save_condition(root, "ss_input_mask.samt", mask_shape, input.mask) ||
        !save_condition(root, "ss_input_pointmap.samt", image_shape,
                        input.pointmap) ||
        !save_condition(root, "ss_input_rgb_image.samt", image_shape,
                        input.rgb_image) ||
        !save_condition(root, "ss_input_rgb_image_mask.samt", mask_shape,
                        input.rgb_image_mask) ||
        !save_condition(root, "ss_input_rgb_pointmap.samt", image_shape,
                        input.rgb_pointmap) ||
        !save_raw_tensor_f32((root / "ss_input_pointmap_scale.samt").string(),
                             vector_shape, input.pointmap_scale.data()) ||
        !save_raw_tensor_f32((root / "ss_input_pointmap_shift.samt").string(),
                             vector_shape, input.pointmap_shift.data()) ||
        !save_raw_tensor_f32(
                (root / "ss_input_rgb_pointmap_scale.samt").string(),
                vector_shape, input.pointmap_scale.data()) ||
        !save_raw_tensor_f32(
                (root / "ss_input_rgb_pointmap_shift.samt").string(),
                vector_shape, input.pointmap_shift.data())) {
        error = "failed to write one or more native condition SAMT tensors";
        return false;
    }
    return true;
}

}  // namespace sam3d
