// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "lightglue/lightglue.h"

namespace lightglue {

struct AlikedExtractionOptions {
    std::string model_path;
    // cpu, gpu, cuda, vulkan, optionally followed by :<device index>.
    std::string device = "cpu";
    int32_t num_threads = 0;
    int32_t resize_long_edge = 1024;
    int32_t max_keypoints = 1024;
    float detection_threshold = 0.2f;
    int32_t nms_radius = 2;
    // When true, standard conv blocks use GGML graph ops (hybrid with CPU DCN).
    bool use_ggml_cnn = false;
};

// Feature extractor for ALIKED-n16rot. Output matches cvg/LightGlue ALIKED:
// pixel-coordinate keypoints and row-major L2-normalized 128-D descriptors.
class AlikedFeatureExtractor {
public:
    virtual ~AlikedFeatureExtractor() = default;

    virtual bool ExtractFromRgb(const uint8_t *rgb,
                                int32_t width,
                                int32_t height,
                                int32_t row_stride,
                                Features *features) = 0;
    virtual const std::string &Error() const = 0;
    virtual const std::string &Device() const = 0;

    // Notify the extractor that a device-lost event occurred (e.g. Vulkan
    // vk::DeviceLostError).  The implementation should flag the backend for
    // re-initialization on the next ExtractFromRgb call.
    virtual void MarkDeviceLost() {}
};

std::unique_ptr<AlikedFeatureExtractor> CreateAlikedFeatureExtractor(
        const AlikedExtractionOptions &options, std::string *error = nullptr);

bool QuantizeAlikedModel(const std::string &input_gguf,
                         const std::string &output_gguf,
                         const std::string &type,
                         std::string *error = nullptr);

// Runs dense-map extraction with DCN parity capture (Vulkan vs CPU bridge).
bool DumpAlikedDcnParity(const AlikedExtractionOptions &options,
                         const uint8_t *rgb,
                         int32_t width,
                         int32_t height,
                         int32_t row_stride,
                         const std::string &output_dump,
                         std::string *error = nullptr);

}  // namespace lightglue
